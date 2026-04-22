# jax2onnx/plugins/jax/numpy/polyfit.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.jax.numpy.linalg_inv import (
    _all_static_ints,
    _as_output_name,
    _binary_op,
    _concat,
    _const_scalar,
    _unsqueeze,
)
from jax2onnx.plugins.jax.numpy.linalg_solve import _cast_to_output_dtype
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_POLYFIT_PRIM: Final = make_jnp_primitive("jax.numpy.polyfit")
_JAX_POLYFIT_ORIG: Final = jnp.polyfit


def _normalise_linear_shapes(
    x_shape_raw: tuple[object, ...],
    y_shape_raw: tuple[object, ...],
    out_shape_raw: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    if (
        len(x_shape_raw) != 1
        or len(y_shape_raw) != 1
        or not _all_static_ints(x_shape_raw)
        or not _all_static_ints(y_shape_raw)
        or not _all_static_ints(out_shape_raw)
    ):
        raise TypeError("jnp.polyfit lowering requires static 1D x/y shapes")
    x_shape = tuple(int(dim) for dim in x_shape_raw)
    y_shape = tuple(int(dim) for dim in y_shape_raw)
    out_shape = tuple(int(dim) for dim in out_shape_raw)
    sample_count = x_shape[0]
    if y_shape[0] != sample_count:
        raise ValueError("jnp.polyfit lowering requires matching x/y lengths")
    if sample_count < 2:
        raise ValueError("jnp.polyfit linear lowering requires at least two samples")
    if out_shape != (2,):
        raise ValueError("jnp.polyfit linear lowering output shape mismatch")
    return x_shape, y_shape, out_shape, sample_count


def _abstract_eval_via_orig(
    x: core.AbstractValue,
    y: core.AbstractValue,
    *,
    deg: int,
    rcond: float | None,
    full: bool,
    cov: bool | str,
) -> core.ShapedArray:
    x_shape = tuple(getattr(x, "shape", ()))
    y_shape = tuple(getattr(y, "shape", ()))
    x_dtype = np.dtype(getattr(x, "dtype", np.float32))
    y_dtype = np.dtype(getattr(y, "dtype", np.float32))
    if np.issubdtype(x_dtype, np.complexfloating) or np.issubdtype(
        y_dtype, np.complexfloating
    ):
        raise TypeError("jnp.polyfit lowering does not support complex inputs")
    orig = get_orig_impl(_POLYFIT_PRIM, "polyfit")
    out = jax.eval_shape(
        lambda x_value, y_value: orig(
            x_value,
            y_value,
            deg,
            rcond=rcond,
            full=full,
            cov=cov,
        ),
        jax.ShapeDtypeStruct(x_shape, x_dtype),
        jax.ShapeDtypeStruct(y_shape, y_dtype),
    )
    return core.ShapedArray(
        tuple(getattr(out, "shape", ())), getattr(out, "dtype", np.float32)
    )


def _reduce_sum_1d(
    ctx: LoweringContextProtocol,
    value: ir.Value,
    *,
    dtype_enum: ir.DataType,
    name_hint: str,
) -> ir.Value:
    axes = _const_i64(ctx, np.asarray([0], dtype=np.int64), f"{name_hint}_axes")
    result = ctx.builder.ReduceSum(
        value,
        axes,
        keepdims=0,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    result.type = ir.TensorType(dtype_enum)
    _stamp_type_and_shape(result, ())
    _ensure_value_metadata(ctx, result)
    return result


@register_primitive(
    jaxpr_primitive=_POLYFIT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polyfit.html",
    onnx=[
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {
            "component": "Unsqueeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Unsqueeze.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
    ],
    since="0.13.0",
    context="primitives.jnp",
    component="polyfit",
    testcases=[
        {
            "testcase": "jnp_polyfit_linear_two_points",
            "callable": lambda x, y: jnp.polyfit(x, y, 1),
            "input_values": [
                np.asarray([0.0, 1.0], dtype=np.float32),
                np.asarray([1.0, 3.0], dtype=np.float32),
            ],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["ReduceSum", "Div", "Unsqueeze", "Concat:2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_polyfit_linear_three_points",
            "callable": lambda x, y: jnp.polyfit(x, y, 1),
            "input_values": [
                np.asarray([1.0, 2.0, 4.0], dtype=np.float32),
                np.asarray([3.0, 5.0, 9.0], dtype=np.float32),
            ],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Mul:3", "ReduceSum", "Sub", "Div", "Concat:2"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpPolyfitPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _POLYFIT_PRIM
    _FUNC_NAME: ClassVar[str] = "polyfit"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        y: core.AbstractValue,
        *,
        deg: int,
        rcond: float | None = None,
        full: bool = False,
        cov: bool | str = False,
    ) -> core.AbstractValue:
        return _abstract_eval_via_orig(
            x,
            y,
            deg=int(deg),
            rcond=rcond,
            full=bool(full),
            cov=cov,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        x_var, y_var = eqn.invars
        (out_var,) = eqn.outvars
        params = dict(getattr(eqn, "params", {}) or {})
        deg = int(params.get("deg", -1))
        rcond = params.get("rcond", None)
        full = bool(params.get("full", False))
        cov = params.get("cov", False)
        if deg != 1 or rcond is not None or full or cov:
            raise NotImplementedError(
                "jnp.polyfit lowering currently supports only unweighted "
                "linear fits with rcond=None, full=False, cov=False"
            )

        x_shape, y_shape, out_shape, sample_count = _normalise_linear_shapes(
            tuple(getattr(x_var.aval, "shape", ())),
            tuple(getattr(y_var.aval, "shape", ())),
            tuple(getattr(out_var.aval, "shape", ())),
        )

        x_dtype = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        y_dtype = np.dtype(getattr(y_var.aval, "dtype", x_dtype))
        out_dtype = np.dtype(getattr(out_var.aval, "dtype", y_dtype))
        if (
            np.issubdtype(x_dtype, np.complexfloating)
            or np.issubdtype(y_dtype, np.complexfloating)
            or np.issubdtype(out_dtype, np.complexfloating)
        ):
            raise TypeError("jnp.polyfit lowering does not support complex dtypes")

        dtype_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("polyfit_x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("polyfit_y"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("polyfit_out")
        )
        desired_name = _as_output_name(ctx, out_spec, "polyfit_out")

        x_val = _cast_to_output_dtype(
            ctx,
            x_val,
            dtype_enum=dtype_enum,
            shape=x_shape,
            name_hint="polyfit_x_cast",
        )
        y_val = _cast_to_output_dtype(
            ctx,
            y_val,
            dtype_enum=dtype_enum,
            shape=y_shape,
            name_hint="polyfit_y_cast",
        )

        x_sq = _binary_op(
            ctx,
            "Mul",
            x_val,
            x_val,
            dtype_enum=dtype_enum,
            shape=x_shape,
            name_hint="polyfit_x_sq",
        )
        xy = _binary_op(
            ctx,
            "Mul",
            x_val,
            y_val,
            dtype_enum=dtype_enum,
            shape=x_shape,
            name_hint="polyfit_xy",
        )

        sum_x = _reduce_sum_1d(
            ctx, x_val, dtype_enum=dtype_enum, name_hint="polyfit_sum_x"
        )
        sum_y = _reduce_sum_1d(
            ctx, y_val, dtype_enum=dtype_enum, name_hint="polyfit_sum_y"
        )
        sum_xx = _reduce_sum_1d(
            ctx, x_sq, dtype_enum=dtype_enum, name_hint="polyfit_sum_xx"
        )
        sum_xy = _reduce_sum_1d(
            ctx, xy, dtype_enum=dtype_enum, name_hint="polyfit_sum_xy"
        )

        n_const = _const_scalar(
            ctx,
            dtype=out_dtype,
            value=float(sample_count),
            name_hint="polyfit_sample_count",
        )
        n_sum_xy = _binary_op(
            ctx,
            "Mul",
            n_const,
            sum_xy,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="polyfit_n_sum_xy",
        )
        sum_x_sum_y = _binary_op(
            ctx,
            "Mul",
            sum_x,
            sum_y,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="polyfit_sum_x_sum_y",
        )
        slope_num = _binary_op(
            ctx,
            "Sub",
            n_sum_xy,
            sum_x_sum_y,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="polyfit_slope_num",
        )
        n_sum_xx = _binary_op(
            ctx,
            "Mul",
            n_const,
            sum_xx,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="polyfit_n_sum_xx",
        )
        sum_x_sq = _binary_op(
            ctx,
            "Mul",
            sum_x,
            sum_x,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="polyfit_sum_x_sq",
        )
        denom = _binary_op(
            ctx,
            "Sub",
            n_sum_xx,
            sum_x_sq,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="polyfit_denom",
        )
        slope = _binary_op(
            ctx,
            "Div",
            slope_num,
            denom,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="polyfit_slope",
        )

        slope_sum_x = _binary_op(
            ctx,
            "Mul",
            slope,
            sum_x,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="polyfit_slope_sum_x",
        )
        intercept_num = _binary_op(
            ctx,
            "Sub",
            sum_y,
            slope_sum_x,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="polyfit_intercept_num",
        )
        intercept = _binary_op(
            ctx,
            "Div",
            intercept_num,
            n_const,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="polyfit_intercept",
        )

        slope_1d = _unsqueeze(
            ctx,
            slope,
            axis=0,
            shape=(1,),
            name_hint="polyfit_slope_1d",
        )
        intercept_1d = _unsqueeze(
            ctx,
            intercept,
            axis=0,
            shape=(1,),
            name_hint="polyfit_intercept_1d",
        )
        result = _concat(
            ctx,
            (slope_1d, intercept_1d),
            axis=0,
            dtype_enum=dtype_enum,
            shape=out_shape,
            name_hint="polyfit_result",
        )
        if getattr(result, "name", None) != desired_name:
            result = ctx.builder.Identity(result, _outputs=[desired_name])
            result.type = ir.TensorType(dtype_enum)
            _stamp_type_and_shape(result, out_shape)
            _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.polyfit not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                x: ArrayLike,
                y: ArrayLike,
                deg: int,
                rcond: float | None = None,
                full: bool = False,
                w: ArrayLike | None = None,
                cov: bool | str = False,
            ) -> Any:
                x_shape = getattr(x, "shape", None)
                y_shape = getattr(y, "shape", None)
                if (
                    int(deg) == 1
                    and rcond is None
                    and not full
                    and w is None
                    and not cov
                    and x_shape is not None
                    and y_shape is not None
                    and len(tuple(x_shape)) == 1
                    and len(tuple(y_shape)) == 1
                    and tuple(x_shape) == tuple(y_shape)
                    and int(tuple(x_shape)[0]) >= 2
                ):
                    return cls._PRIM.bind(
                        jnp.asarray(x),
                        jnp.asarray(y),
                        deg=1,
                        rcond=None,
                        full=False,
                        cov=False,
                    )
                return orig(x, y, deg, rcond=rcond, full=full, w=w, cov=cov)

            return _patched

        return [
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            )
        ]


@JnpPolyfitPlugin._PRIM.def_impl
def _polyfit_impl(
    x: ArrayLike,
    y: ArrayLike,
    *,
    deg: int,
    rcond: float | None = None,
    full: bool = False,
    cov: bool | str = False,
) -> Any:
    try:
        orig = get_orig_impl(JnpPolyfitPlugin._PRIM, JnpPolyfitPlugin._FUNC_NAME)
    except RuntimeError:
        orig = _JAX_POLYFIT_ORIG
    return orig(x, y, deg, rcond=rcond, full=full, cov=cov)


JnpPolyfitPlugin._PRIM.def_abstract_eval(JnpPolyfitPlugin.abstract_eval)
