# jax2onnx/plugins/jax/numpy/outer.py

from __future__ import annotations

from typing import ClassVar, Final

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
from jax.interpreters import batching
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_OUTER_PRIM: Final = make_jnp_primitive("jax.numpy.outer")


@register_primitive(
    jaxpr_primitive=_OUTER_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.outer.html",
    onnx=[{"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"}],
    since="v0.10.0",
    context="primitives.jnp",
    component="outer",
    testcases=[
        {
            "testcase": "outer_vector",
            "callable": lambda a, b: jnp.outer(a, b),
            "input_shapes": [(3,), (4,)],
        },
        {
            "testcase": "outer",
            "callable": lambda a, b: jnp.outer(a, b),
            "input_shapes": [(3,), (5,)],
        },
        {
            "testcase": "outer_vmap_batching",
            "callable": lambda a, b: jax.vmap(jnp.outer)(a, b),
            "input_shapes": [(3, 3), (3, 4)],
        },
    ],
)
class JnpOuterPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _OUTER_PRIM
    _FUNC_NAME: ClassVar[str] = "outer"

    @staticmethod
    def abstract_eval(a: core.AbstractValue, b: core.AbstractValue) -> core.ShapedArray:
        result_shape = tuple(a.shape) + tuple(b.shape)
        result_dtype = np.result_type(a.dtype, b.dtype)
        return jax.core.ShapedArray(result_shape, result_dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (a_var, b_var) = eqn.invars
        (out_var,) = eqn.outvars

        a_val = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("outer_a"))
        b_val = ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("outer_b"))
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for outer lowering")

        a_shape = tuple(getattr(a_var.aval, "shape", ()))
        b_shape = tuple(getattr(b_var.aval, "shape", ()))
        target_shape = tuple(getattr(out_var.aval, "shape", ()))

        a_reshape = builder.Reshape(
            a_val,
            _const_i64(
                ctx,
                np.asarray(a_shape + (1,) * len(b_shape), dtype=np.int64),
                "outer_a_shape",
            ),
            _outputs=[ctx.fresh_name("outer_a_broadcast")],
        )
        _stamp_type_and_shape(a_reshape, a_shape + (1,) * len(b_shape))
        _ensure_value_metadata(ctx, a_reshape)

        b_reshape = builder.Reshape(
            b_val,
            _const_i64(
                ctx,
                np.asarray((1,) * len(a_shape) + b_shape, dtype=np.int64),
                "outer_b_shape",
            ),
            _outputs=[ctx.fresh_name("outer_b_broadcast")],
        )
        _stamp_type_and_shape(b_reshape, (1,) * len(a_shape) + b_shape)
        _ensure_value_metadata(ctx, b_reshape)

        result = builder.Mul(
            a_reshape,
            b_reshape,
            _outputs=[ctx.fresh_name("Outer")],
        )
        _stamp_type_and_shape(result, target_shape)
        _ensure_value_metadata(ctx, result)

        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return jnp_binding_specs(cls._PRIM, cls._FUNC_NAME)


@JnpOuterPlugin._PRIM.def_impl
def _outer_impl(a: ArrayLike, b: ArrayLike) -> jax.Array:
    orig = get_orig_impl(JnpOuterPlugin._PRIM, JnpOuterPlugin._FUNC_NAME)
    return orig(a, b)


JnpOuterPlugin._PRIM.def_abstract_eval(JnpOuterPlugin.abstract_eval)


BatchDim = int | type(batching.not_mapped)


def _outer_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
) -> tuple[jax.Array, BatchDim]:
    a, b = batched_args
    a_bdim, b_bdim = batch_dims
    mapped = [
        (arg, bd)
        for arg, bd in zip(batched_args, batch_dims)
        if bd is not batching.not_mapped
    ]
    if not mapped:
        out = JnpOuterPlugin._PRIM.bind(a, b)
        return out, batching.not_mapped

    sample_arg, sample_bd = mapped[0]
    batch_size = sample_arg.shape[sample_bd]

    if a_bdim is not batching.not_mapped:
        a = batching.bdim_at_front(a, a_bdim, batch_size)
    if b_bdim is not batching.not_mapped:
        b = batching.bdim_at_front(b, b_bdim, batch_size)

    in_axes = (
        0 if a_bdim is not batching.not_mapped else None,
        0 if b_bdim is not batching.not_mapped else None,
    )
    orig = get_orig_impl(JnpOuterPlugin._PRIM, JnpOuterPlugin._FUNC_NAME)

    def _call_single(a_slice: jax.Array, b_slice: jax.Array) -> jax.Array:
        return orig(a_slice, b_slice)

    result = jax.vmap(_call_single, in_axes=in_axes)(a, b)
    return result, 0


batching.primitive_batchers[JnpOuterPlugin._PRIM] = _outer_batch_rule
