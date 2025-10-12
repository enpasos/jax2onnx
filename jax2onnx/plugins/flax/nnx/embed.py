# jax2onnx/plugins/flax/nnx/embed.py

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar, Final

import jax.numpy as jnp
from flax import nnx
from jax import core
from jax.extend.core import Primitive
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import (
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
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


EMBED_PRIM: Final[Primitive] = Primitive("nnx.embed")
EMBED_PRIM.multiple_results = False


EXPECT_EMBED_GATHER: Final = EG(
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

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for Embed lowering")

        idx_dtype = getattr(getattr(indices_val, "type", None), "dtype", None)
        if idx_dtype not in (ir.DataType.INT64, None):
            casted = builder.Cast(
                indices_val,
                _outputs=[ctx.fresh_name("embed_idx_i64")],
                to=int(ir.DataType.INT64.value),
            )
            casted.type = ir.TensorType(ir.DataType.INT64)
            casted.shape = indices_val.shape
            _stamp_type_and_shape(
                casted, tuple(getattr(getattr(indices_var, "aval", None), "shape", ()))
            )
            _ensure_value_metadata(ctx, casted)
            indices_val = casted

        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("embed_out"))
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("embed_out")

        result = builder.Gather(
            embedding_val,
            indices_val,
            axis=0,
            _outputs=[desired_name],
        )
        embed_dtype = getattr(getattr(embedding_val, "type", None), "dtype", None)
        if embed_dtype is not None:
            result.type = ir.TensorType(embed_dtype)

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

        _stamp_type_and_shape(result, tuple(out_dims))
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

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
