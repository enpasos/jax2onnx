# jax2onnx/plugins/jax/numpy/left_shift.py

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


_LEFT_SHIFT_PRIM: Final = make_jnp_primitive("jax.numpy.left_shift")

_NP_SIGNED_TO_UNSIGNED: dict[np.dtype[Any], np.dtype[Any]] = {
    np.dtype(np.int8): np.dtype(np.uint8),
    np.dtype(np.int16): np.dtype(np.uint16),
    np.dtype(np.int32): np.dtype(np.uint32),
    np.dtype(np.int64): np.dtype(np.uint64),
}


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


def lower_left_shift_core(
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

    desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(output_hint)
    producer = getattr(out_spec, "producer", None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name(output_hint)

    if np.issubdtype(out_dtype, np.unsignedinteger):
        x_ready = cast_to_dtype(ctx, x_val, np_dtype=out_dtype, name_hint=cast_x_hint)
        y_ready = cast_to_dtype(ctx, y_val, np_dtype=out_dtype, name_hint=cast_y_hint)
        result = ctx.builder.BitShift(
            x_ready,
            y_ready,
            direction="LEFT",
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
        return

    unsigned_dtype = _NP_SIGNED_TO_UNSIGNED.get(out_dtype)
    if unsigned_dtype is None:
        raise NotImplementedError(f"left_shift unsupported dtype '{out_dtype}'")
    x_unsigned = cast_to_dtype(
        ctx, x_val, np_dtype=unsigned_dtype, name_hint=f"{cast_x_hint}_u"
    )
    y_unsigned = cast_to_dtype(
        ctx, y_val, np_dtype=unsigned_dtype, name_hint=f"{cast_y_hint}_u"
    )
    shifted = ctx.builder.BitShift(
        x_unsigned,
        y_unsigned,
        direction="LEFT",
        _outputs=[ctx.fresh_name(f"{output_hint}_shifted_u")],
    )
    shifted.type = x_unsigned.type
    shifted.shape = out_spec.shape
    signed_ir = numpy_dtype_to_ir(out_dtype)
    result = ctx.builder.Cast(
        shifted,
        to=int(signed_ir.value),
        _outputs=[desired_name],
    )
    if getattr(out_spec, "type", None) is not None:
        result.type = out_spec.type
    if getattr(out_spec, "shape", None) is not None:
        result.shape = out_spec.shape
    ctx.bind_value_for_var(out_var, result)


@register_primitive(
    jaxpr_primitive=_LEFT_SHIFT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.left_shift.html",
    onnx=[
        {
            "component": "BitShift",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitShift.html",
        },
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="left_shift",
    testcases=[
        {
            "testcase": "jnp_left_shift_int",
            "callable": lambda x, s: jnp.left_shift(x, s),
            "input_values": [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([1, 2, 1], dtype=np.int32),
            ],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["BitShift:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_left_shift_unsigned",
            "callable": lambda x, s: jnp.left_shift(x, s),
            "input_values": [
                np.array([1, 2, 3], dtype=np.uint32),
                np.array([1, 2, 1], dtype=np.uint32),
            ],
            "expected_output_dtypes": [np.uint32],
            "post_check_onnx_graph": EG(
                ["BitShift:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_left_shift_mixed_bool_rhs",
            "callable": lambda x, s: jnp.left_shift(x, s),
            "input_values": [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([True, False, True], dtype=np.bool_),
            ],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> BitShift:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_left_shift_broadcast",
            "callable": lambda x, s: jnp.left_shift(x, s),
            "input_shapes": [(2, 3), (1, 3)],
            "input_dtypes": [np.int32, np.int32],
            "post_check_onnx_graph": EG(
                ["BitShift:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "left_shift_vmap_batching",
            "callable": lambda x, s: jax.vmap(jnp.left_shift)(x, s),
            "input_shapes": [(3, 4), (3, 4)],
            "input_dtypes": [np.int32, np.int32],
        },
    ],
)
class JnpLeftShiftPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _LEFT_SHIFT_PRIM
    _FUNC_NAME: ClassVar[str] = "left_shift"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        y: core.AbstractValue,
    ) -> core.ShapedArray:
        return abstract_eval_via_orig_binary(
            JnpLeftShiftPlugin._PRIM,
            JnpLeftShiftPlugin._FUNC_NAME,
            x,
            y,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_left_shift_core(
            ctx,
            eqn,
            input_x_hint="jnp_left_shift_x",
            input_y_hint="jnp_left_shift_s",
            output_hint="jnp_left_shift_out",
            cast_x_hint="jnp_left_shift_x_cast",
            cast_y_hint="jnp_left_shift_s_cast",
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpLeftShiftPlugin._PRIM.def_impl
def _left_shift_impl(*args: object, **kwargs: object) -> object:
    orig = get_orig_impl(JnpLeftShiftPlugin._PRIM, JnpLeftShiftPlugin._FUNC_NAME)
    return orig(*args, **kwargs)


def _left_shift_batch_rule(
    args: tuple[Any, ...], dims: tuple[Any, ...], **params: Any
) -> tuple[Any, Any]:
    return cast(
        tuple[Any, Any],
        broadcast_batcher_compat(JnpLeftShiftPlugin._PRIM, args, dims, **params),
    )


batching.primitive_batchers[JnpLeftShiftPlugin._PRIM] = _left_shift_batch_rule
