# jax2onnx/plugins/jax/numpy/outer.py

from __future__ import annotations

from typing import ClassVar, Final, TypeAlias

import jax
from jax2onnx._compat.jax import (
    AbstractValue,
    JaxprEqn,
    ShapedArray,
    batching,
)
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._precision_utils import emit_float16_arithmetic_with_rounding
from jax2onnx.plugins._utils import cast_value_to_dtype
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import (
    abstract_eval_via_jax,
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


_OUTER_PRIM: Final = make_jnp_primitive("jax.numpy.outer")


@register_native_primitive(
    jaxpr_primitive=_OUTER_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.outer.html",
    onnx=[{"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"}],
    since="0.10.0",
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
        {
            "testcase": "outer_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.outer(y, y)))(x),
            "input_shapes": [(3,)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpOuterPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _OUTER_PRIM
    _FUNC_NAME: ClassVar[str] = "outer"

    @staticmethod
    def abstract_eval(a: AbstractValue, b: AbstractValue) -> ShapedArray:
        original = get_orig_impl(JnpOuterPlugin._PRIM, JnpOuterPlugin._FUNC_NAME)
        return abstract_eval_via_jax(original, a, b)

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        (a_var, b_var) = eqn.invars
        (out_var,) = eqn.outvars

        a_val = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("outer_a"))
        b_val = ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("outer_b"))
        builder = ctx.builder
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("Outer")
        )
        dtype = numpy_dtype_to_ir(out_var.aval.dtype)
        a_val = cast_value_to_dtype(ctx, a_val, dtype, name_hint="outer_a_cast")
        b_val = cast_value_to_dtype(ctx, b_val, dtype, name_hint="outer_b_cast")
        target_shape = tuple(out_var.aval.shape)

        a_reshape = builder.Reshape(
            a_val,
            _const_i64(
                ctx,
                np.asarray([-1, 1], dtype=np.int64),
                "outer_a_shape",
            ),
            _outputs=[ctx.fresh_name("outer_a_broadcast")],
        )
        a_reshape.type = ir.TensorType(dtype)
        _stamp_type_and_shape(a_reshape, (target_shape[0], 1))
        _ensure_value_metadata(ctx, a_reshape)

        b_reshape = builder.Reshape(
            b_val,
            _const_i64(
                ctx,
                np.asarray([1, -1], dtype=np.int64),
                "outer_b_shape",
            ),
            _outputs=[ctx.fresh_name("outer_b_broadcast")],
        )
        b_reshape.type = ir.TensorType(dtype)
        _stamp_type_and_shape(b_reshape, (1, target_shape[1]))
        _ensure_value_metadata(ctx, b_reshape)

        if dtype == ir.DataType.FLOAT16:
            result = emit_float16_arithmetic_with_rounding(
                ctx, "Mul", a_reshape, b_reshape, output=out_spec
            )
        else:
            result = builder.op("Mul", [a_reshape, b_reshape], output=out_spec)

        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpOuterPlugin._PRIM.def_impl
def _outer_impl(a: ArrayLike, b: ArrayLike) -> jax.Array:
    orig = get_orig_impl(JnpOuterPlugin._PRIM, JnpOuterPlugin._FUNC_NAME)
    return orig(a, b)


register_jvp_via_jax_jvp(JnpOuterPlugin._PRIM, _outer_impl)


JnpOuterPlugin._PRIM.def_abstract_eval(JnpOuterPlugin.abstract_eval)


BatchDim: TypeAlias = int | None


def _outer_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
) -> tuple[jax.Array, BatchDim]:
    a, b = batched_args
    a_bdim, b_bdim = batch_dims
    mapped: list[tuple[jax.Array, int]] = [
        (arg, bd) for arg, bd in zip(batched_args, batch_dims) if bd is not None
    ]
    if not mapped:
        out = JnpOuterPlugin._PRIM.bind(a, b)
        return out, None

    sample_arg, sample_bd = mapped[0]
    batch_size = sample_arg.shape[sample_bd]

    if a_bdim is not None:
        a = batching.bdim_at_front(a, a_bdim, batch_size)
    if b_bdim is not None:
        b = batching.bdim_at_front(b, b_bdim, batch_size)

    in_axes = (
        0 if a_bdim is not None else None,
        0 if b_bdim is not None else None,
    )
    orig = get_orig_impl(JnpOuterPlugin._PRIM, JnpOuterPlugin._FUNC_NAME)

    def _call_single(a_slice: jax.Array, b_slice: jax.Array) -> jax.Array:
        return orig(a_slice, b_slice)

    result = jax.vmap(_call_single, in_axes=in_axes)(a, b)
    return result, 0


batching.primitive_batchers[JnpOuterPlugin._PRIM] = _outer_batch_rule
