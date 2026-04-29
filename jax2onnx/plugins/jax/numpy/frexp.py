# jax2onnx/plugins/jax/numpy/frexp.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, TypeAlias, cast

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import ir_dtype_to_numpy, numpy_dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.nextafter import _float_layout
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_FREXP_PRIM: Final = make_jnp_primitive("jax.numpy.frexp")
_FREXP_PRIM.multiple_results = True

BatchDim: TypeAlias = int | None


def _abstract_eval_via_orig(
    prim: Any,
    func_name: str,
    x: core.AbstractValue,
) -> tuple[core.ShapedArray, core.ShapedArray]:
    x_shape = tuple(getattr(x, "shape", ()))
    x_dtype: np.dtype[Any] = np.dtype(getattr(x, "dtype", np.float32))
    x_spec = jax.ShapeDtypeStruct(x_shape, x_dtype)
    orig = get_orig_impl(prim, func_name)
    mantissa, exponent = jax.eval_shape(lambda value: orig(value), x_spec)
    mantissa_shape = tuple(getattr(mantissa, "shape", ()))
    exponent_shape = tuple(getattr(exponent, "shape", ()))
    mantissa_dtype = np.dtype(getattr(mantissa, "dtype", np.float32))
    exponent_dtype = np.dtype(getattr(exponent, "dtype", np.int32))
    return (
        core.ShapedArray(mantissa_shape, mantissa_dtype),
        core.ShapedArray(exponent_shape, exponent_dtype),
    )


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
    jaxpr_primitive=_FREXP_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.frexp.html",
    onnx=[
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"},
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
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
    component="frexp",
    testcases=[
        {
            "testcase": "jnp_frexp_basic",
            "callable": lambda x: jnp.frexp(x),
            "input_values": [
                np.asarray(
                    [-8.0, -1.5, -1.0, -0.0, 0.0, 0.75, 1.0, 2.0, 3.5],
                    dtype=np.float32,
                )
            ],
            "expected_output_dtypes": [np.float32, np.int32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Abs", "Log", "Floor", "Pow", "Div", "Mul"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_frexp_special",
            "callable": lambda x: jnp.frexp(x),
            "input_values": [
                np.asarray([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
            ],
            "expected_output_dtypes": [np.float32, np.int32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["IsInf", "IsNaN", "Where", "Cast"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "frexp_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.frexp)(x),
            "input_shapes": [(3, 4)],
            "expected_output_dtypes": [np.float32, np.int32],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpFrexpPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _FREXP_PRIM
    _FUNC_NAME: ClassVar[str] = "frexp"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> tuple[core.ShapedArray, ...]:
        return _abstract_eval_via_orig(
            JnpFrexpPlugin._PRIM,
            JnpFrexpPlugin._FUNC_NAME,
            x,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        mantissa_var, exponent_var = eqn.outvars

        mantissa_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(mantissa_var, "aval", None), "dtype", np.float32)
        )
        exponent_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(exponent_var, "aval", None), "dtype", np.int32)
        )
        mantissa_enum = _dtype_to_ir(
            mantissa_dtype, ctx.builder.enable_double_precision
        )
        exponent_enum = numpy_dtype_to_ir(exponent_dtype)
        np_dtype = ir_dtype_to_numpy(mantissa_enum, default=mantissa_dtype)
        if np_dtype is None:
            np_dtype = mantissa_dtype
        layout = _float_layout(np_dtype)
        if layout is None:
            raise TypeError(f"frexp only supports floating mantissas, got '{np_dtype}'")

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("frexp_x"))
        mantissa_spec = ctx.get_value_for_var(
            mantissa_var, name_hint=ctx.fresh_name("frexp_mantissa")
        )
        exponent_spec = ctx.get_value_for_var(
            exponent_var, name_hint=ctx.fresh_name("frexp_exponent")
        )

        x_ready = _cast_to_dtype(
            ctx,
            x_val,
            np_dtype=mantissa_dtype,
            name_hint="frexp_x_cast",
        )

        zero = _const(
            ctx,
            dtype_enum=mantissa_enum,
            np_dtype=np_dtype,
            value=0.0,
            name_hint="zero",
        )
        one = _const(
            ctx, dtype_enum=mantissa_enum, np_dtype=np_dtype, value=1.0, name_hint="one"
        )
        half = _const(
            ctx,
            dtype_enum=mantissa_enum,
            np_dtype=np_dtype,
            value=0.5,
            name_hint="half",
        )
        two = _const(
            ctx, dtype_enum=mantissa_enum, np_dtype=np_dtype, value=2.0, name_hint="two"
        )
        log2 = _const(
            ctx,
            dtype_enum=mantissa_enum,
            np_dtype=np_dtype,
            value=float(np.log(2.0)),
            name_hint="log2",
        )

        abs_x = ctx.builder.Abs(x_ready, _outputs=[ctx.fresh_name("frexp_abs")])
        _stamp_like_x(abs_x, x_ready)

        x_is_zero = ctx.builder.Equal(
            abs_x,
            zero,
            _outputs=[ctx.fresh_name("frexp_is_zero")],
        )
        x_is_inf = ctx.builder.IsInf(x_ready, _outputs=[ctx.fresh_name("frexp_is_inf")])
        x_is_nan = ctx.builder.IsNaN(x_ready, _outputs=[ctx.fresh_name("frexp_is_nan")])
        special = ctx.builder.Or(
            x_is_inf,
            x_is_nan,
            _outputs=[ctx.fresh_name("frexp_special")],
        )
        special_or_zero = ctx.builder.Or(
            special,
            x_is_zero,
            _outputs=[ctx.fresh_name("frexp_special_or_zero")],
        )

        safe_abs = ctx.builder.Where(
            special_or_zero,
            one,
            abs_x,
            _outputs=[ctx.fresh_name("frexp_safe_abs")],
        )
        _stamp_like_x(safe_abs, x_ready)

        log_abs = ctx.builder.Log(safe_abs, _outputs=[ctx.fresh_name("frexp_log")])
        _stamp_like_x(log_abs, x_ready)
        log2_abs = ctx.builder.Div(
            log_abs,
            log2,
            _outputs=[ctx.fresh_name("frexp_log2")],
        )
        _stamp_like_x(log2_abs, x_ready)
        floor_log2 = ctx.builder.Floor(
            log2_abs,
            _outputs=[ctx.fresh_name("frexp_floor_log2")],
        )
        _stamp_like_x(floor_log2, x_ready)
        exponent_float = ctx.builder.Add(
            floor_log2,
            one,
            _outputs=[ctx.fresh_name("frexp_exponent_float")],
        )
        _stamp_like_x(exponent_float, x_ready)

        scale = ctx.builder.Pow(
            two,
            floor_log2,
            _outputs=[ctx.fresh_name("frexp_scale")],
        )
        _stamp_like_x(scale, x_ready)
        scaled = ctx.builder.Div(
            x_ready,
            scale,
            _outputs=[ctx.fresh_name("frexp_scaled")],
        )
        _stamp_like_x(scaled, x_ready)
        mantissa_calc = ctx.builder.Mul(
            scaled,
            half,
            _outputs=[ctx.fresh_name("frexp_mantissa_calc")],
        )
        _stamp_like_x(mantissa_calc, x_ready)

        mantissa_name = getattr(mantissa_spec, "name", None) or ctx.fresh_name(
            "frexp_mantissa"
        )
        mantissa_producer = getattr(mantissa_spec, "producer", None)
        if callable(mantissa_producer) and mantissa_producer() is not None:
            mantissa_name = ctx.fresh_name("frexp_mantissa")

        mantissa_result = ctx.builder.Where(
            special_or_zero,
            x_ready,
            mantissa_calc,
            _outputs=[mantissa_name],
        )
        if getattr(mantissa_spec, "type", None) is not None:
            mantissa_result.type = mantissa_spec.type
        else:
            mantissa_result.type = ir.TensorType(mantissa_enum)
        if getattr(mantissa_spec, "shape", None) is not None:
            mantissa_result.shape = mantissa_spec.shape
        else:
            _stamp_type_and_shape(
                mantissa_result, tuple(getattr(mantissa_var.aval, "shape", ()))
            )
        _ensure_value_metadata(ctx, mantissa_result)

        exponent_zero = zero
        exponent_float_safe = ctx.builder.Where(
            special_or_zero,
            exponent_zero,
            exponent_float,
            _outputs=[ctx.fresh_name("frexp_exponent_float_safe")],
        )
        _stamp_like_x(exponent_float_safe, x_ready)

        exponent_name = getattr(exponent_spec, "name", None) or ctx.fresh_name(
            "frexp_exponent"
        )
        exponent_producer = getattr(exponent_spec, "producer", None)
        if callable(exponent_producer) and exponent_producer() is not None:
            exponent_name = ctx.fresh_name("frexp_exponent")

        exponent_result = ctx.builder.Cast(
            exponent_float_safe,
            to=int(exponent_enum.value),
            _outputs=[exponent_name],
        )
        exponent_result.type = ir.TensorType(exponent_enum)
        if getattr(exponent_spec, "shape", None) is not None:
            exponent_result.shape = exponent_spec.shape
        else:
            _stamp_type_and_shape(
                exponent_result, tuple(getattr(exponent_var.aval, "shape", ()))
            )
        _ensure_value_metadata(ctx, exponent_result)

        ctx.bind_value_for_var(mantissa_var, mantissa_result)
        ctx.bind_value_for_var(exponent_var, exponent_result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., tuple[jax.Array, jax.Array]] | None,
        ) -> Callable[..., tuple[jax.Array, jax.Array]]:
            if orig is None:
                raise RuntimeError("Original jnp.frexp not found for monkey patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(x: object) -> tuple[jax.Array, jax.Array]:
                return cast(tuple[jax.Array, jax.Array], cls._PRIM.bind(x))

            return _patched

        return [
            AssignSpec(
                "jax.numpy", f"{cls._FUNC_NAME}_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@JnpFrexpPlugin._PRIM.def_impl
def _frexp_impl(x: object) -> tuple[jax.Array, jax.Array]:
    orig = get_orig_impl(JnpFrexpPlugin._PRIM, JnpFrexpPlugin._FUNC_NAME)
    return cast(tuple[jax.Array, jax.Array], orig(x))


JnpFrexpPlugin._PRIM.def_abstract_eval(JnpFrexpPlugin.abstract_eval)


def _frexp_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
) -> tuple[tuple[jax.Array, jax.Array], tuple[BatchDim, BatchDim]]:
    (x,) = batched_args
    (bdim,) = batch_dims
    mantissa, exponent = JnpFrexpPlugin._PRIM.bind(x)
    return (mantissa, exponent), (bdim, bdim)


batching.primitive_batchers[JnpFrexpPlugin._PRIM] = _frexp_batch_rule
