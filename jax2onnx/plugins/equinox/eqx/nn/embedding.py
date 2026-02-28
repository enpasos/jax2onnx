# jax2onnx/plugins/equinox/eqx/nn/embedding.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import (
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

_EQX_EMBED_TABLE: Final = jnp.asarray(
    np.arange(20, dtype=np.float32).reshape(5, 4), dtype=jnp.float32
)
_EQX_EMBED: Final[eqx.nn.Embedding] = eqx.nn.Embedding(weight=_EQX_EMBED_TABLE)


@register_primitive(
    jaxpr_primitive="eqx.nn.embedding",
    jax_doc="https://docs.kidger.site/equinox/api/nn/embedding/#equinox.nn.Embedding",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        }
    ],
    since="0.12.3",
    context="primitives.eqx",
    component="embedding",
    testcases=[
        {
            "testcase": "eqx_embedding_scalar_index",
            "callable": _EQX_EMBED,
            "input_shapes": [()],
            "input_dtypes": [jnp.int32],
            "expected_output_shapes": [(4,)],
            "post_check_onnx_graph": expect_graph(
                ["Cast -> Gather:4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_embedding_batched_vmap",
            "callable": lambda idx, _mod=_EQX_EMBED: jax.vmap(_mod)(idx),
            "input_shapes": [("B",)],
            "input_dtypes": [jnp.int32],
            "expected_output_shapes": [("B", 4)],
            "post_check_onnx_graph": expect_graph(
                ["Cast -> Gather:Bx4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class EmbeddingPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.embedding")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        indices: jax_core.AbstractValue,
        embedding: jax_core.AbstractValue,
    ) -> jax_core.ShapedArray:
        features = embedding.shape[-1]
        return jax_core.ShapedArray(indices.shape + (features,), embedding.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax_core.JaxprEqn) -> None:
        (indices_var, embedding_var) = eqn.invars[:2]
        (out_var,) = eqn.outvars

        indices_val = ctx.get_value_for_var(
            indices_var, name_hint=ctx.fresh_name("eqx_embed_idx")
        )
        embedding_val = ctx.get_value_for_var(
            embedding_var, name_hint=ctx.fresh_name("eqx_embed_table")
        )

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for Embedding")

        idx_dtype = getattr(getattr(indices_val, "type", None), "dtype", None)
        if idx_dtype not in (ir.DataType.INT64, None):
            casted = builder.Cast(
                indices_val,
                _outputs=[ctx.fresh_name("eqx_embed_idx_i64")],
                to=int(ir.DataType.INT64.value),
            )
            casted.type = ir.TensorType(ir.DataType.INT64)
            casted.shape = indices_val.shape
            _stamp_type_and_shape(
                casted, tuple(getattr(getattr(indices_var, "aval", None), "shape", ()))
            )
            _ensure_value_metadata(ctx, casted)
            indices_val = casted

        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("eqx_embed_out")
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("eqx_embed")

        result = builder.Gather(
            embedding_val,
            indices_val,
            axis=0,
            _outputs=[desired_name],
        )

        out_type = getattr(out_spec, "type", None)
        if out_type is not None:
            result.type = out_type
        else:
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
        if feat_dim is None and embedding_shape:
            feat_dim = embedding_shape[-1]
        out_dims.append(feat_dim)

        _stamp_type_and_shape(result, tuple(out_dims))
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("equinox.nn", "embedding_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="equinox.nn.Embedding",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _patch_call(
        orig: Callable[..., jax.Array] | None,
    ) -> Callable[..., jax.Array]:
        del orig

        def wrapped(
            self: eqx.nn.Embedding,
            x: Any,
            *,
            key: jax.Array | None = None,
        ) -> jax.Array:
            del key
            return EmbeddingPlugin._PRIM.bind(x, jnp.asarray(self.weight))

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


@EmbeddingPlugin._PRIM.def_impl
def _embedding_impl(indices: jax.Array, embedding: jax.Array) -> jax.Array:
    return jnp.take(embedding, indices, axis=0)


def _embedding_batch_rule(
    batched_args: tuple[jax.Array, jax.Array],
    batch_dims: tuple[int | None, int | None],
) -> tuple[jax.Array, int | None]:
    indices, embedding = batched_args
    idx_bdim, emb_bdim = batch_dims

    if emb_bdim is not None:
        raise NotImplementedError("Batching over embedding table is not supported.")

    out = EmbeddingPlugin._PRIM.bind(indices, embedding)
    return out, idx_bdim


batching.primitive_batchers[EmbeddingPlugin._PRIM] = _embedding_batch_rule
