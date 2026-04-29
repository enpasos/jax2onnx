# jax2onnx/plugins/jax/numpy/right_shift.py

from __future__ import annotations

from typing import Any, ClassVar, Final, cast

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_RIGHT_SHIFT_PRIM: Final = make_jnp_primitive("jax.numpy.right_shift")

_SIGNED_TO_UNSIGNED: dict[ir.DataType, tuple[ir.DataType, np.dtype[Any], int]] = {
    ir.DataType.INT8: (ir.DataType.UINT8, np.dtype(np.uint8), 8),
    ir.DataType.INT16: (ir.DataType.UINT16, np.dtype(np.uint16), 16),
    ir.DataType.INT32: (ir.DataType.UINT32, np.dtype(np.uint32), 32),
    ir.DataType.INT64: (ir.DataType.UINT64, np.dtype(np.uint64), 64),
}

_SIGNED_INTEGER_DTYPES: frozenset[np.dtype[Any]] = frozenset(
    {
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.int32),
        np.dtype(np.int64),
    }
)


def abstract_eval_via_orig_binary(
    prim: Any,
    func_name: str,
    x: core.AbstractValue,
    y: core.AbstractValue,
) -> core.ShapedArray:
    x_shape = tuple(getattr(x, "shape", ()))
    y_shape = tuple(getattr(y, "shape", ()))
    x_dtype: np.dtype[Any] = np.dtype(getattr(x, "dtype", np.int32))
    y_dtype: np.dtype[Any] = np.dtype(getattr(y, "dtype", np.int32))
    x_spec = jax.ShapeDtypeStruct(x_shape, x_dtype)
    y_spec = jax.ShapeDtypeStruct(y_shape, y_dtype)
    orig = get_orig_impl(prim, func_name)
    out = jax.eval_shape(lambda a, b: orig(a, b), x_spec, y_spec)
    out_shape = tuple(getattr(out, "shape", ()))
    out_dtype = np.dtype(getattr(out, "dtype", x_dtype))
    return core.ShapedArray(out_shape, out_dtype)


def cast_to_dtype(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    np_dtype: np.dtype[Any],
    name_hint: str,
) -> ir.Value:
    dtype_enum = numpy_dtype_to_ir(np_dtype)
    if getattr(getattr(val, "type", None), "dtype", None) == dtype_enum:
        return val
    cast_val = cast(
        ir.Value,
        ctx.builder.Cast(
            val,
            to=int(dtype_enum.value),
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    cast_val.type = ir.TensorType(dtype_enum)
    cast_val.shape = getattr(val, "shape", None)
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


def _stamp_like(
    value: ir.Value, ref: ir.Value, *, dtype: ir.DataType | None = None
) -> None:
    if dtype is not None:
        value.type = ir.TensorType(dtype)
    elif getattr(ref, "type", None) is not None:
        value.type = ref.type


def _lower_signed_arithmetic_right_shift(
    ctx: LoweringContextProtocol,
    x_ready: ir.Value,
    s_ready: ir.Value,
    *,
    signed_enum: ir.DataType,
    signed_np_dtype: np.dtype[Any],
    desired_name: str,
    out_spec: ir.Value,
    output_hint: str,
) -> ir.Value:
    unsigned_enum, unsigned_np_dtype, bit_width = _SIGNED_TO_UNSIGNED[signed_enum]

    zero_signed = ctx.bind_const_for_var(object(), np.asarray(0, dtype=signed_np_dtype))
    zero_unsigned = ctx.bind_const_for_var(
        object(), np.asarray(0, dtype=unsigned_np_dtype)
    )
    ones_unsigned = ctx.bind_const_for_var(
        object(),
        np.asarray(np.iinfo(unsigned_np_dtype).max, dtype=unsigned_np_dtype),
    )
    bit_width_unsigned = ctx.bind_const_for_var(
        object(), np.asarray(bit_width, dtype=unsigned_np_dtype)
    )

    s_nonneg = ctx.builder.Max(
        s_ready,
        zero_signed,
        _outputs=[ctx.fresh_name(f"{output_hint}_nonneg_shift")],
    )
    _stamp_like(s_nonneg, s_ready, dtype=signed_enum)

    s_unsigned = ctx.builder.Cast(
        s_nonneg,
        to=int(unsigned_enum.value),
        _outputs=[ctx.fresh_name(f"{output_hint}_shift_u")],
    )
    _stamp_like(s_unsigned, s_ready, dtype=unsigned_enum)

    x_unsigned = ctx.builder.Cast(
        x_ready,
        to=int(unsigned_enum.value),
        _outputs=[ctx.fresh_name(f"{output_hint}_input_u")],
    )
    _stamp_like(x_unsigned, x_ready, dtype=unsigned_enum)

    s_clamped = ctx.builder.Min(
        s_unsigned,
        bit_width_unsigned,
        _outputs=[ctx.fresh_name(f"{output_hint}_shift_clamped")],
    )
    _stamp_like(s_clamped, s_unsigned, dtype=unsigned_enum)

    shifted = ctx.builder.BitShift(
        x_unsigned,
        s_clamped,
        direction="RIGHT",
        _outputs=[ctx.fresh_name(f"{output_hint}_shifted")],
    )
    _stamp_like(shifted, x_unsigned, dtype=unsigned_enum)

    n_minus_s = ctx.builder.Sub(
        bit_width_unsigned,
        s_clamped,
        _outputs=[ctx.fresh_name(f"{output_hint}_n_minus_s")],
    )
    _stamp_like(n_minus_s, s_unsigned, dtype=unsigned_enum)

    mask_raw = ctx.builder.BitShift(
        ones_unsigned,
        n_minus_s,
        direction="LEFT",
        _outputs=[ctx.fresh_name(f"{output_hint}_sign_mask_raw")],
    )
    _stamp_like(mask_raw, shifted, dtype=unsigned_enum)

    is_zero_shift = ctx.builder.Equal(
        s_clamped,
        zero_unsigned,
        _outputs=[ctx.fresh_name(f"{output_hint}_is_zero_shift")],
    )
    _stamp_like(is_zero_shift, shifted, dtype=ir.DataType.BOOL)

    nonzero_shift = ctx.builder.Not(
        is_zero_shift,
        _outputs=[ctx.fresh_name(f"{output_hint}_nonzero_shift")],
    )
    _stamp_like(nonzero_shift, shifted, dtype=ir.DataType.BOOL)

    nonzero_shift_u = ctx.builder.Cast(
        nonzero_shift,
        to=int(unsigned_enum.value),
        _outputs=[ctx.fresh_name(f"{output_hint}_nonzero_shift_u")],
    )
    _stamp_like(nonzero_shift_u, shifted, dtype=unsigned_enum)

    sign_mask = ctx.builder.Mul(
        mask_raw,
        nonzero_shift_u,
        _outputs=[ctx.fresh_name(f"{output_hint}_sign_mask")],
    )
    _stamp_like(sign_mask, shifted, dtype=unsigned_enum)

    neg_shifted = ctx.builder.BitwiseOr(
        shifted,
        sign_mask,
        _outputs=[ctx.fresh_name(f"{output_hint}_neg_shifted")],
    )
    _stamp_like(neg_shifted, shifted, dtype=unsigned_enum)

    is_negative = ctx.builder.Less(
        x_ready,
        zero_signed,
        _outputs=[ctx.fresh_name(f"{output_hint}_is_negative")],
    )
    _stamp_like(is_negative, x_ready, dtype=ir.DataType.BOOL)

    is_negative_u = ctx.builder.Cast(
        is_negative,
        to=int(unsigned_enum.value),
        _outputs=[ctx.fresh_name(f"{output_hint}_is_negative_u")],
    )
    _stamp_like(is_negative_u, shifted, dtype=unsigned_enum)

    delta = ctx.builder.Sub(
        neg_shifted,
        shifted,
        _outputs=[ctx.fresh_name(f"{output_hint}_delta")],
    )
    _stamp_like(delta, shifted, dtype=unsigned_enum)

    correction = ctx.builder.Mul(
        delta,
        is_negative_u,
        _outputs=[ctx.fresh_name(f"{output_hint}_correction")],
    )
    _stamp_like(correction, shifted, dtype=unsigned_enum)

    selected = ctx.builder.Add(
        shifted,
        correction,
        _outputs=[ctx.fresh_name(f"{output_hint}_selected_u")],
    )
    _stamp_like(selected, shifted, dtype=unsigned_enum)

    result = cast(
        ir.Value,
        ctx.builder.Cast(
            selected,
            to=int(signed_enum.value),
            _outputs=[desired_name],
        ),
    )
    if getattr(out_spec, "type", None) is not None:
        result.type = out_spec.type
    if getattr(out_spec, "shape", None) is not None:
        result.shape = out_spec.shape
    return result


def lower_right_shift_core(
    ctx: LoweringContextProtocol,
    eqn: core.JaxprEqn,
    *,
    input_x_hint: str,
    input_y_hint: str,
    output_hint: str,
    cast_x_hint: str,
    cast_y_hint: str,
) -> None:
    x_var, y_var = eqn.invars
    (out_var,) = eqn.outvars

    x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name(input_x_hint))
    prefer_dt: np.dtype[Any] = np.dtype(
        getattr(getattr(x_var, "aval", None), "dtype", np.int32)
    )
    y_val = ctx.get_value_for_var(
        y_var,
        name_hint=ctx.fresh_name(input_y_hint),
        prefer_np_dtype=prefer_dt,
    )
    out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name(output_hint))

    out_dtype: np.dtype[Any] = np.dtype(
        getattr(getattr(out_var, "aval", None), "dtype", np.int32)
    )
    x_ready = cast_to_dtype(ctx, x_val, np_dtype=out_dtype, name_hint=cast_x_hint)
    y_ready = cast_to_dtype(ctx, y_val, np_dtype=out_dtype, name_hint=cast_y_hint)

    desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(output_hint)
    producer = getattr(out_spec, "producer", None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name(output_hint)

    if np.issubdtype(out_dtype, np.unsignedinteger):
        result = ctx.builder.BitShift(
            x_ready,
            y_ready,
            direction="RIGHT",
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
        return

    if out_dtype not in _SIGNED_INTEGER_DTYPES:
        raise NotImplementedError(f"right_shift unsupported dtype '{out_dtype}'")
    signed_enum = numpy_dtype_to_ir(out_dtype)

    result = _lower_signed_arithmetic_right_shift(
        ctx,
        x_ready,
        y_ready,
        signed_enum=signed_enum,
        signed_np_dtype=out_dtype,
        desired_name=desired_name,
        out_spec=out_spec,
        output_hint=output_hint,
    )
    ctx.bind_value_for_var(out_var, result)


@register_primitive(
    jaxpr_primitive=_RIGHT_SHIFT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.right_shift.html",
    onnx=[
        {
            "component": "BitShift",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitShift.html",
        },
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="right_shift",
    testcases=[
        {
            "testcase": "jnp_right_shift_signed_arithmetic",
            "callable": lambda x, s: jnp.right_shift(x, s),
            "input_values": [
                np.array([-16, -1, 16], dtype=np.int32),
                np.array([1, 2, 1], dtype=np.int32),
            ],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["BitShift"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_right_shift_unsigned_logical",
            "callable": lambda x, s: jnp.right_shift(x, s),
            "input_values": [
                np.array([16, 32, 64], dtype=np.uint32),
                np.array([1, 2, 1], dtype=np.uint32),
            ],
            "expected_output_dtypes": [np.uint32],
            "post_check_onnx_graph": EG(
                ["BitShift:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_right_shift_mixed_bool_rhs",
            "callable": lambda x, s: jnp.right_shift(x, s),
            "input_values": [
                np.array([8, 4, 2], dtype=np.int32),
                np.array([True, False, True], dtype=np.bool_),
            ],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> BitShift"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_right_shift_broadcast",
            "callable": lambda x, s: jnp.right_shift(x, s),
            "input_shapes": [(2, 3), (1, 3)],
            "input_dtypes": [np.int32, np.int32],
            "post_check_onnx_graph": EG(
                ["BitShift"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "right_shift_vmap_batching",
            "callable": lambda x, s: jax.vmap(jnp.right_shift)(x, s),
            "input_shapes": [(3, 4), (3, 4)],
            "input_dtypes": [np.int32, np.int32],
        },
    ],
)
class JnpRightShiftPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _RIGHT_SHIFT_PRIM
    _FUNC_NAME: ClassVar[str] = "right_shift"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        y: core.AbstractValue,
    ) -> core.ShapedArray:
        return abstract_eval_via_orig_binary(
            JnpRightShiftPlugin._PRIM,
            JnpRightShiftPlugin._FUNC_NAME,
            x,
            y,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_right_shift_core(
            ctx,
            eqn,
            input_x_hint="jnp_right_shift_x",
            input_y_hint="jnp_right_shift_s",
            output_hint="jnp_right_shift_out",
            cast_x_hint="jnp_right_shift_x_cast",
            cast_y_hint="jnp_right_shift_s_cast",
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpRightShiftPlugin._PRIM.def_impl
def _right_shift_impl(*args: object, **kwargs: object) -> object:
    orig = get_orig_impl(JnpRightShiftPlugin._PRIM, JnpRightShiftPlugin._FUNC_NAME)
    return orig(*args, **kwargs)


def _right_shift_batch_rule(
    args: tuple[Any, ...], dims: tuple[Any, ...], **params: Any
) -> tuple[Any, Any]:
    return cast(
        tuple[Any, Any],
        broadcast_batcher_compat(JnpRightShiftPlugin._PRIM, args, dims, **params),
    )


batching.primitive_batchers[JnpRightShiftPlugin._PRIM] = _right_shift_batch_rule
