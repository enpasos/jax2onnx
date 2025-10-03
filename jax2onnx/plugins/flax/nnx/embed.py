from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar

import jax.numpy as jnp
from flax import nnx
from jax import core
from jax.extend.core import Primitive
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins._ir_shapes import (
    _dim_label_from_value_or_aval,
    _ensure_value_info as _add_value_info,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.plugins.plugin_system import _IRBuildContext as IRBuildContext  # type: ignore


EMBED_PRIM = Primitive("nnx.embed")
EMBED_PRIM.multiple_results = False


EXPECT_EMBED_GATHER = EG(
    [
        (
            "Gather",
            {
                "counts": {
                    "Gather": 1,
                }
            },
        )
    ]
)


@register_primitive(
    jaxpr_primitive=EMBED_PRIM.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Embed",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        }
    ],
    since="v0.7.0",
    context="primitives.nnx",
    component="embed",
    testcases=[
        {
            "testcase": "token_embedding",
            "callable": construct_and_call(
                nnx.Embed,
                num_embeddings=3144,
                features=48,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 64)],
            "input_dtypes": [jnp.int32],
            "post_check_onnx_graph": EXPECT_EMBED_GATHER,
        },
        {
            "testcase": "positional_embedding",
            "callable": construct_and_call(
                nnx.Embed,
                num_embeddings=64,
                features=48,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 64)],
            "input_dtypes": [jnp.int32],
            "post_check_onnx_graph": EXPECT_EMBED_GATHER,
        },
    ],
)
class EmbedPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = EMBED_PRIM
    _ORIG_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(indices, embedding):
        features = embedding.shape[-1]
        return core.ShapedArray(indices.shape + (features,), embedding.dtype)

    def lower(self, ctx: "IRBuildContext", eqn):  # type: ignore[override]
        indices_var, embedding_var = eqn.invars[:2]
        (out_var,) = eqn.outvars

        indices_val = ctx.get_value_for_var(
            indices_var, name_hint=ctx.fresh_name("embed_idx")
        )
        embedding_val = ctx.get_value_for_var(
            embedding_var, name_hint=ctx.fresh_name("embed_table")
        )

        idx_dtype = getattr(getattr(indices_val, "type", None), "dtype", None)
        if idx_dtype not in (ir.DataType.INT64, None):
            casted = ir.Value(
                name=ctx.fresh_name("embed_idx_i64"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=indices_val.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Cast",
                    domain="",
                    inputs=[indices_val],
                    outputs=[casted],
                    name=ctx.fresh_name("Cast"),
                    attributes=[
                        IRAttr("to", IRAttrType.INT, int(ir.DataType.INT64.value))
                    ],
                )
            )
            indices_val = casted

        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("embed_out"))

        gather = ir.Node(
            op_type="Gather",
            domain="",
            inputs=[embedding_val, indices_val],
            outputs=[out_val],
            name=ctx.fresh_name("Gather"),
            attributes=[IRAttr("axis", IRAttrType.INT, 0)],
        )
        ctx.add_node(gather)

        indices_shape = tuple(getattr(getattr(indices_var, "aval", None), "shape", ()))
        embedding_shape = tuple(
            getattr(getattr(embedding_var, "aval", None), "shape", ())
        )

        out_dims = [
            _dim_label_from_value_or_aval(indices_val, indices_shape, i)
            for i in range(len(indices_shape))
        ]
        feat_dim = _dim_label_from_value_or_aval(
            embedding_val, embedding_shape, len(embedding_shape) - 1
        )
        if feat_dim is None:
            feat_dim = embedding_shape[-1] if embedding_shape else None
        out_dims.append(feat_dim)

        _stamp_type_and_shape(out_val, tuple(out_dims))
        _add_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        def _make_patch(orig):
            cls._ORIG_CALL = orig

            def _patched(self: nnx.Embed, indices):
                table = self.embedding.value
                return cls._PRIM.bind(indices, table)

            return _patched

        return [
            AssignSpec("flax.nnx", "embed_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.nnx.Embed",
                attr="__call__",
                make_value=_make_patch,
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


@EmbedPlugin._PRIM.def_impl
def _embed_impl(indices, embedding):
    return jnp.take(embedding, indices, axis=0)


EmbedPlugin.ensure_abstract_eval_bound()
