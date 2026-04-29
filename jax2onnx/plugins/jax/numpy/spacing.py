# jax2onnx/plugins/jax/numpy/spacing.py

from __future__ import annotations

from typing import Any, ClassVar, Final, cast

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import ir_dtype_to_numpy
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.nextafter import _float_layout
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.jax.numpy._unary_utils import (
    register_unary_elementwise_batch_rule,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SPACING_PRIM: Final = make_jnp_primitive("jax.numpy.spacing")


def _abstract_eval_via_orig(
    prim: Any,
    func_name: str,
    x: core.AbstractValue,
) -> core.ShapedArray:
    x_shape = tuple(getattr(x, "shape", ()))
    x_dtype: np.dtype[Any] = np.dtype(getattr(x, "dtype", np.float32))
    x_spec = jax.ShapeDtypeStruct(x_shape, x_dtype)
    orig = get_orig_impl(prim, func_name)
    out = jax.eval_shape(lambda value: orig(value), x_spec)
    out_shape = tuple(getattr(out, "shape", ()))
    out_dtype = np.dtype(getattr(out, "dtype", x_dtype))
    return core.ShapedArray(out_shape, out_dtype)


def _const(
    ctx: LoweringContextProtocol,
    *,
    dtype_enum: ir.DataType,
    np_dtype: np.dtype[Any],
    value: float,
    name_hint: str,
) -> ir.Value:
    const = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(name_hint),
        array=np.asarray(value, dtype=np_dtype),
    )
    const.type = ir.TensorType(dtype_enum)
    _stamp_type_and_shape(const, ())
    _ensure_value_metadata(ctx, const)
    return const


def _cast_to_dtype(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    np_dtype: np.dtype[Any],
    name_hint: str,
) -> ir.Value:
    dtype_enum = _dtype_to_ir(np_dtype, ctx.builder.enable_double_precision)
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


def _stamp_like_x(value: ir.Value, x: ir.Value) -> None:
    if getattr(x, "type", None) is not None:
        value.type = x.type
    if getattr(x, "shape", None) is not None:
        value.shape = x.shape


@register_primitive(
    jaxpr_primitive=_SPACING_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.spacing.html",
    onnx=[
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "IsInf",
            "doc": "https://onnx.ai/onnx/operators/onnx__IsInf.html",
        },
        {
            "component": "IsNaN",
            "doc": "https://onnx.ai/onnx/operators/onnx__IsNaN.html",
        },
    ],
    since="0.13.0",
    context="primitives.jnp",
    component="spacing",
    testcases=[
        {
            "testcase": "jnp_spacing_basic",
            "callable": lambda x: jnp.spacing(x),
            "input_values": [
                np.asarray([-2.0, -1.0, -0.0, 0.0, 1.0, 2.0], dtype=np.float32)
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Abs", "Log", "Floor", "Pow", "Where", "Mul"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_spacing_subnormal_and_special",
            "callable": lambda x: jnp.spacing(x),
            "input_values": [
                np.asarray(
                    [
                        np.nextafter(np.float32(0.0), np.float32(1.0)),
                        np.finfo(np.float32).tiny,
                        np.inf,
                        -np.inf,
                        np.nan,
                    ],
                    dtype=np.float32,
                )
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["IsInf", "IsNaN", "Where"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "spacing_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.spacing)(x),
            "input_shapes": [(3, 4)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpSpacingPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SPACING_PRIM
    _FUNC_NAME: ClassVar[str] = "spacing"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return _abstract_eval_via_orig(
            JnpSpacingPlugin._PRIM,
            JnpSpacingPlugin._FUNC_NAME,
            x,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        out_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(out_var, "aval", None), "dtype", np.float32)
        )
        dtype_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
        np_dtype = ir_dtype_to_numpy(dtype_enum, default=out_dtype)
        if np_dtype is None:
            np_dtype = out_dtype
        layout = _float_layout(np_dtype)
        if layout is None:
            raise TypeError(f"spacing only supports floating dtypes, got '{np_dtype}'")
        mantissa_bits, _min_exp, _max_exp = layout

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("spacing_x"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("spacing_out")
        )
        x_ready = _cast_to_dtype(
            ctx,
            x_val,
            np_dtype=out_dtype,
            name_hint="spacing_x_cast",
        )

        zero = _const(
            ctx, dtype_enum=dtype_enum, np_dtype=np_dtype, value=0.0, name_hint="zero"
        )
        one = _const(
            ctx, dtype_enum=dtype_enum, np_dtype=np_dtype, value=1.0, name_hint="one"
        )
        neg_one = _const(
            ctx,
            dtype_enum=dtype_enum,
            np_dtype=np_dtype,
            value=-1.0,
            name_hint="neg_one",
        )
        two = _const(
            ctx, dtype_enum=dtype_enum, np_dtype=np_dtype, value=2.0, name_hint="two"
        )
        log2 = _const(
            ctx,
            dtype_enum=dtype_enum,
            np_dtype=np_dtype,
            value=float(np.log(2.0)),
            name_hint="log2",
        )
        tiny_subnormal = _const(
            ctx,
            dtype_enum=dtype_enum,
            np_dtype=np_dtype,
            value=float(
                np.nextafter(
                    np.asarray(0.0, dtype=np_dtype),
                    np.asarray(1.0, dtype=np_dtype),
                )
            ),
            name_hint="tiny_subnormal",
        )
        min_normal = _const(
            ctx,
            dtype_enum=dtype_enum,
            np_dtype=np_dtype,
            value=float(np.finfo(np_dtype).tiny),
            name_hint="min_normal",
        )
        mantissa_bits_f = _const(
            ctx,
            dtype_enum=dtype_enum,
            np_dtype=np_dtype,
            value=float(mantissa_bits),
            name_hint="mantissa_bits",
        )
        nan_const = _const(
            ctx, dtype_enum=dtype_enum, np_dtype=np_dtype, value=np.nan, name_hint="nan"
        )

        abs_x = ctx.builder.Abs(x_ready, _outputs=[ctx.fresh_name("spacing_abs")])
        _stamp_like_x(abs_x, x_ready)

        x_is_zero = ctx.builder.Equal(
            abs_x,
            zero,
            _outputs=[ctx.fresh_name("spacing_is_zero")],
        )
        safe_abs = ctx.builder.Where(
            x_is_zero,
            one,
            abs_x,
            _outputs=[ctx.fresh_name("spacing_safe_abs")],
        )
        _stamp_like_x(safe_abs, x_ready)

        log_abs = ctx.builder.Log(safe_abs, _outputs=[ctx.fresh_name("spacing_log")])
        _stamp_like_x(log_abs, x_ready)
        log2_abs = ctx.builder.Div(
            log_abs,
            log2,
            _outputs=[ctx.fresh_name("spacing_log2")],
        )
        _stamp_like_x(log2_abs, x_ready)
        exponent = ctx.builder.Floor(
            log2_abs,
            _outputs=[ctx.fresh_name("spacing_exp")],
        )
        _stamp_like_x(exponent, x_ready)

        ulp_exp = ctx.builder.Sub(
            exponent,
            mantissa_bits_f,
            _outputs=[ctx.fresh_name("spacing_ulp_exp")],
        )
        _stamp_like_x(ulp_exp, x_ready)
        ulp = ctx.builder.Pow(two, ulp_exp, _outputs=[ctx.fresh_name("spacing_ulp")])
        _stamp_like_x(ulp, x_ready)

        is_subnormal = ctx.builder.Less(
            abs_x,
            min_normal,
            _outputs=[ctx.fresh_name("spacing_is_subnormal")],
        )
        magnitude = ctx.builder.Where(
            is_subnormal,
            tiny_subnormal,
            ulp,
            _outputs=[ctx.fresh_name("spacing_magnitude")],
        )
        _stamp_like_x(magnitude, x_ready)

        is_negative_nonzero = ctx.builder.Less(
            x_ready,
            zero,
            _outputs=[ctx.fresh_name("spacing_is_negative_nonzero")],
        )
        reciprocal = ctx.builder.Div(
            one,
            x_ready,
            _outputs=[ctx.fresh_name("spacing_reciprocal")],
        )
        _stamp_like_x(reciprocal, x_ready)
        is_negative_zero = ctx.builder.Less(
            reciprocal,
            zero,
            _outputs=[ctx.fresh_name("spacing_is_negative_zero")],
        )
        is_signed_zero = ctx.builder.And(
            x_is_zero,
            is_negative_zero,
            _outputs=[ctx.fresh_name("spacing_is_signed_zero")],
        )
        is_negative = ctx.builder.Or(
            is_negative_nonzero,
            is_signed_zero,
            _outputs=[ctx.fresh_name("spacing_is_negative")],
        )
        sign = ctx.builder.Where(
            is_negative,
            neg_one,
            one,
            _outputs=[ctx.fresh_name("spacing_sign")],
        )
        _stamp_like_x(sign, x_ready)
        signed_magnitude = ctx.builder.Mul(
            sign,
            magnitude,
            _outputs=[ctx.fresh_name("spacing_signed")],
        )
        _stamp_like_x(signed_magnitude, x_ready)

        x_is_inf = ctx.builder.IsInf(
            x_ready, _outputs=[ctx.fresh_name("spacing_is_inf")]
        )
        x_is_nan = ctx.builder.IsNaN(
            x_ready, _outputs=[ctx.fresh_name("spacing_is_nan")]
        )
        special = ctx.builder.Or(
            x_is_inf,
            x_is_nan,
            _outputs=[ctx.fresh_name("spacing_special")],
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("spacing_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("spacing_out")

        result = ctx.builder.Where(
            special,
            nan_const,
            signed_magnitude,
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = ir.TensorType(dtype_enum)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpSpacingPlugin._PRIM.def_impl
def _spacing_impl(*args: object, **kwargs: object) -> object:
    orig = get_orig_impl(JnpSpacingPlugin._PRIM, JnpSpacingPlugin._FUNC_NAME)
    return orig(*args, **kwargs)


register_unary_elementwise_batch_rule(JnpSpacingPlugin._PRIM)
