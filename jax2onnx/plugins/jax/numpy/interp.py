# jax2onnx/plugins/jax/numpy/interp.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, TypeAlias, cast

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._ir_shapes import (
    DimInput,
    _ensure_value_metadata,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_INTERP_PRIM: Final = make_jnp_primitive("jax.numpy.interp")

BatchDim: TypeAlias = int | None


def _all_static_ints(shape: tuple[object, ...]) -> bool:
    return all(isinstance(dim, (int, np.integer)) for dim in shape)


def _cast_to_dtype(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    from_dtype: np.dtype[Any],
    to_dtype: np.dtype[Any],
    name_hint: str,
    shape: tuple[DimInput, ...],
) -> ir.Value:
    target_enum = _dtype_to_ir(to_dtype, ctx.builder.enable_double_precision)
    if (
        from_dtype == to_dtype
        and getattr(getattr(val, "type", None), "dtype", None) == target_enum
    ):
        return val
    result = cast(
        ir.Value,
        ctx.builder.Cast(
            val,
            to=int(target_enum.value),
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    result.type = ir.TensorType(target_enum)
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _unsqueeze(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    axis: int,
    shape: tuple[DimInput, ...],
    name_hint: str,
) -> ir.Value:
    axes = _const_i64(ctx, np.asarray([axis], dtype=np.int64), f"{name_hint}_axes")
    result = cast(
        ir.Value,
        ctx.builder.Unsqueeze(
            val,
            axes,
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    if getattr(val, "type", None) is not None:
        result.type = val.type
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _const_i64_scalar(
    ctx: LoweringContextProtocol,
    value: int,
    *,
    name_hint: str,
) -> ir.Value:
    result = _const_i64(ctx, np.asarray(value, dtype=np.int64), name_hint)
    _stamp_type_and_shape(result, ())
    _ensure_value_metadata(ctx, result)
    return result


def _gather(
    ctx: LoweringContextProtocol,
    data: ir.Value,
    indices: ir.Value,
    *,
    shape: tuple[DimInput, ...],
    name_hint: str,
) -> ir.Value:
    result = cast(
        ir.Value,
        ctx.builder.Gather(
            data,
            indices,
            axis=0,
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    if getattr(data, "type", None) is not None:
        result.type = data.type
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _binary_float_op(
    ctx: LoweringContextProtocol,
    op_type: str,
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    dtype_enum: ir.DataType,
    shape: tuple[DimInput, ...],
    name_hint: str,
) -> ir.Value:
    if op_type == "Add":
        result = cast(
            ir.Value,
            ctx.builder.Add(lhs, rhs, _outputs=[ctx.fresh_name(name_hint)]),
        )
    elif op_type == "Sub":
        result = cast(
            ir.Value,
            ctx.builder.Sub(lhs, rhs, _outputs=[ctx.fresh_name(name_hint)]),
        )
    elif op_type == "Mul":
        result = cast(
            ir.Value,
            ctx.builder.Mul(lhs, rhs, _outputs=[ctx.fresh_name(name_hint)]),
        )
    elif op_type == "Div":
        result = cast(
            ir.Value,
            ctx.builder.Div(lhs, rhs, _outputs=[ctx.fresh_name(name_hint)]),
        )
    else:  # pragma: no cover - defensive guard for internal callers
        raise ValueError(f"Unsupported interp op: {op_type}")
    result.type = ir.TensorType(dtype_enum)
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _abstract_eval_via_orig(
    prim: Any,
    func_name: str,
    x: core.AbstractValue,
    xp: core.AbstractValue,
    fp: core.AbstractValue,
) -> core.ShapedArray:
    x_shape = tuple(getattr(x, "shape", ()))
    xp_shape = tuple(getattr(xp, "shape", ()))
    fp_shape = tuple(getattr(fp, "shape", ()))
    if len(xp_shape) != 1 or len(fp_shape) != 1:
        raise TypeError("jnp.interp lowering requires 1-D xp and fp")
    if not _all_static_ints(xp_shape) or not _all_static_ints(fp_shape):
        raise TypeError("jnp.interp lowering requires static xp/fp lengths")
    if int(xp_shape[0]) != int(fp_shape[0]):
        raise TypeError("jnp.interp lowering requires xp and fp with equal length")
    if int(xp_shape[0]) < 2:
        raise ValueError("jnp.interp lowering requires at least two samples")

    orig = get_orig_impl(prim, func_name)
    out = jax.eval_shape(
        lambda x_val, xp_val, fp_val: orig(x_val, xp_val, fp_val),
        jax.ShapeDtypeStruct(x_shape, np.dtype(getattr(x, "dtype", np.float32))),
        jax.ShapeDtypeStruct(xp_shape, np.dtype(getattr(xp, "dtype", np.float32))),
        jax.ShapeDtypeStruct(fp_shape, np.dtype(getattr(fp, "dtype", np.float32))),
    )
    out_dtype = np.dtype(getattr(out, "dtype", np.float32))
    if np.issubdtype(out_dtype, np.complexfloating):
        raise TypeError("jnp.interp lowering does not support complex fp values")
    return core.ShapedArray(tuple(getattr(out, "shape", ())), out_dtype)


@register_primitive(
    jaxpr_primitive=_INTERP_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.interp.html",
    onnx=[
        {
            "component": "LessOrEqual",
            "doc": "https://onnx.ai/onnx/operators/onnx__LessOrEqual.html",
        },
        {"component": "Less", "doc": "https://onnx.ai/onnx/operators/onnx__Less.html"},
        {
            "component": "Greater",
            "doc": "https://onnx.ai/onnx/operators/onnx__Greater.html",
        },
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
        {"component": "Max", "doc": "https://onnx.ai/onnx/operators/onnx__Max.html"},
        {"component": "Min", "doc": "https://onnx.ai/onnx/operators/onnx__Min.html"},
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.13.0",
    context="primitives.jnp",
    component="interp",
    testcases=[
        {
            "testcase": "jnp_interp_vector_default_bounds",
            "callable": lambda x, xp, fp: jnp.interp(x, xp, fp),
            "input_values": [
                np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0, 4.0], dtype=np.float32),
                np.asarray([0.0, 1.0, 3.0], dtype=np.float32),
                np.asarray([0.0, 10.0, 30.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    "LessOrEqual:6x3 -> Cast:6x3 -> ReduceSum:6",
                    "Gather:6",
                    "Div:6 -> Mul:6 -> Add:6",
                    "Where:6",
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_interp_matrix_values",
            "callable": lambda x, xp, fp: jnp.interp(x, xp, fp),
            "input_values": [
                np.asarray([[-1.0, 0.0, 0.5], [1.0, 2.0, 4.0]], dtype=np.float32),
                np.asarray([0.0, 1.0, 3.0], dtype=np.float32),
                np.asarray([0.0, 10.0, 30.0], dtype=np.float32),
            ],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float32],
        },
        {
            "testcase": "jnp_interp_integer_inputs",
            "callable": lambda x, xp, fp: jnp.interp(x, xp, fp),
            "input_values": [
                np.asarray([-1, 0, 1, 2, 4], dtype=np.int32),
                np.asarray([0, 1, 3], dtype=np.int32),
                np.asarray([0, 10, 30], dtype=np.int32),
            ],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpInterpPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _INTERP_PRIM
    _FUNC_NAME: ClassVar[str] = "interp"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        xp: core.AbstractValue,
        fp: core.AbstractValue,
    ) -> core.ShapedArray:
        return _abstract_eval_via_orig(
            JnpInterpPlugin._PRIM,
            JnpInterpPlugin._FUNC_NAME,
            x,
            xp,
            fp,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        x_var, xp_var, fp_var = eqn.invars
        (out_var,) = eqn.outvars

        x_shape = cast(tuple[DimInput, ...], tuple(getattr(x_var.aval, "shape", ())))
        xp_shape = cast(tuple[DimInput, ...], tuple(getattr(xp_var.aval, "shape", ())))
        fp_shape = cast(tuple[DimInput, ...], tuple(getattr(fp_var.aval, "shape", ())))
        if len(xp_shape) != 1 or len(fp_shape) != 1:
            raise TypeError("jnp.interp lowering requires 1-D xp and fp")
        if not _all_static_ints(xp_shape) or not _all_static_ints(fp_shape):
            raise TypeError("jnp.interp lowering requires static xp/fp lengths")
        xp_len_raw = xp_shape[0]
        fp_len_raw = fp_shape[0]
        if not isinstance(xp_len_raw, (int, np.integer)) or not isinstance(
            fp_len_raw, (int, np.integer)
        ):
            raise TypeError("jnp.interp lowering requires static xp/fp lengths")
        xp_len = int(xp_len_raw)
        if xp_len != int(fp_len_raw):
            raise TypeError("jnp.interp lowering requires xp and fp with equal length")
        if xp_len < 2:
            raise ValueError("jnp.interp lowering requires at least two samples")

        out_dtype: np.dtype[Any] = np.dtype(getattr(out_var.aval, "dtype", np.float32))
        if np.issubdtype(out_dtype, np.complexfloating):
            raise TypeError("jnp.interp lowering does not support complex fp values")
        out_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
        x_dtype: np.dtype[Any] = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        xp_dtype: np.dtype[Any] = np.dtype(getattr(xp_var.aval, "dtype", np.float32))
        fp_dtype: np.dtype[Any] = np.dtype(getattr(fp_var.aval, "dtype", out_dtype))
        compare_dtype: np.dtype[Any] = np.promote_types(x_dtype, xp_dtype)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("interp_x"))
        xp_val = ctx.get_value_for_var(xp_var, name_hint=ctx.fresh_name("interp_xp"))
        fp_val = ctx.get_value_for_var(fp_var, name_hint=ctx.fresh_name("interp_fp"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("interp_out")
        )

        x_compare = _cast_to_dtype(
            ctx,
            x_val,
            from_dtype=x_dtype,
            to_dtype=compare_dtype,
            name_hint="interp_x_compare_cast",
            shape=x_shape,
        )
        xp_compare = _cast_to_dtype(
            ctx,
            xp_val,
            from_dtype=xp_dtype,
            to_dtype=compare_dtype,
            name_hint="interp_xp_compare_cast",
            shape=xp_shape,
        )
        x_arith = _cast_to_dtype(
            ctx,
            x_val,
            from_dtype=x_dtype,
            to_dtype=out_dtype,
            name_hint="interp_x_arith_cast",
            shape=x_shape,
        )
        xp_arith = _cast_to_dtype(
            ctx,
            xp_val,
            from_dtype=xp_dtype,
            to_dtype=out_dtype,
            name_hint="interp_xp_arith_cast",
            shape=xp_shape,
        )
        fp_arith = _cast_to_dtype(
            ctx,
            fp_val,
            from_dtype=fp_dtype,
            to_dtype=out_dtype,
            name_hint="interp_fp_arith_cast",
            shape=fp_shape,
        )

        xp_broadcast = xp_compare
        xp_broadcast_shape: tuple[DimInput, ...] = (xp_len,)
        for _ in range(len(x_shape)):
            xp_broadcast_shape = (1, *xp_broadcast_shape)
            xp_broadcast = _unsqueeze(
                ctx,
                xp_broadcast,
                axis=0,
                shape=xp_broadcast_shape,
                name_hint="interp_xp_unsqueeze",
            )
        x_broadcast = _unsqueeze(
            ctx,
            x_compare,
            axis=len(x_shape),
            shape=(*x_shape, 1),
            name_hint="interp_x_unsqueeze",
        )
        compare_shape: tuple[DimInput, ...] = (*x_shape, xp_len)
        segment_lte = ctx.builder.LessOrEqual(
            xp_broadcast,
            x_broadcast,
            _outputs=[ctx.fresh_name("interp_segment_lte")],
        )
        segment_lte.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(segment_lte, compare_shape)
        _ensure_value_metadata(ctx, segment_lte)

        i64_enum = numpy_dtype_to_ir(np.dtype(np.int64))
        segment_hits = ctx.builder.Cast(
            segment_lte,
            to=int(i64_enum.value),
            _outputs=[ctx.fresh_name("interp_segment_hits")],
        )
        segment_hits.type = ir.TensorType(i64_enum)
        _stamp_type_and_shape(segment_hits, compare_shape)
        _ensure_value_metadata(ctx, segment_hits)

        reduce_axes = _const_i64(
            ctx,
            np.asarray([len(x_shape)], dtype=np.int64),
            "interp_reduce_axes",
        )
        hi_unclipped = ctx.builder.ReduceSum(
            segment_hits,
            reduce_axes,
            keepdims=0,
            _outputs=[ctx.fresh_name("interp_hi_unclipped")],
        )
        hi_unclipped.type = ir.TensorType(i64_enum)
        _stamp_type_and_shape(hi_unclipped, x_shape)
        _ensure_value_metadata(ctx, hi_unclipped)

        one = _const_i64_scalar(ctx, 1, name_hint="interp_one")
        last_idx = _const_i64_scalar(ctx, xp_len - 1, name_hint="interp_last_idx")
        hi_at_least_one = ctx.builder.Max(
            hi_unclipped,
            one,
            _outputs=[ctx.fresh_name("interp_hi_at_least_one")],
        )
        hi_at_least_one.type = ir.TensorType(i64_enum)
        _stamp_type_and_shape(hi_at_least_one, x_shape)
        _ensure_value_metadata(ctx, hi_at_least_one)
        hi = ctx.builder.Min(
            hi_at_least_one,
            last_idx,
            _outputs=[ctx.fresh_name("interp_hi")],
        )
        hi.type = ir.TensorType(i64_enum)
        _stamp_type_and_shape(hi, x_shape)
        _ensure_value_metadata(ctx, hi)
        lo = ctx.builder.Sub(hi, one, _outputs=[ctx.fresh_name("interp_lo")])
        lo.type = ir.TensorType(i64_enum)
        _stamp_type_and_shape(lo, x_shape)
        _ensure_value_metadata(ctx, lo)

        xp_lo = _gather(ctx, xp_arith, lo, shape=x_shape, name_hint="interp_xp_lo")
        xp_hi = _gather(ctx, xp_arith, hi, shape=x_shape, name_hint="interp_xp_hi")
        fp_lo = _gather(ctx, fp_arith, lo, shape=x_shape, name_hint="interp_fp_lo")
        fp_hi = _gather(ctx, fp_arith, hi, shape=x_shape, name_hint="interp_fp_hi")

        x_delta = _binary_float_op(
            ctx,
            "Sub",
            x_arith,
            xp_lo,
            dtype_enum=out_enum,
            shape=x_shape,
            name_hint="interp_x_delta",
        )
        xp_delta = _binary_float_op(
            ctx,
            "Sub",
            xp_hi,
            xp_lo,
            dtype_enum=out_enum,
            shape=x_shape,
            name_hint="interp_xp_delta",
        )
        fp_delta = _binary_float_op(
            ctx,
            "Sub",
            fp_hi,
            fp_lo,
            dtype_enum=out_enum,
            shape=x_shape,
            name_hint="interp_fp_delta",
        )
        ratio = _binary_float_op(
            ctx,
            "Div",
            x_delta,
            xp_delta,
            dtype_enum=out_enum,
            shape=x_shape,
            name_hint="interp_ratio",
        )
        scaled = _binary_float_op(
            ctx,
            "Mul",
            ratio,
            fp_delta,
            dtype_enum=out_enum,
            shape=x_shape,
            name_hint="interp_scaled",
        )
        interpolated = _binary_float_op(
            ctx,
            "Add",
            fp_lo,
            scaled,
            dtype_enum=out_enum,
            shape=x_shape,
            name_hint="interp_interpolated",
        )

        first_idx = _const_i64_scalar(ctx, 0, name_hint="interp_first_idx")
        first_xp = _gather(
            ctx, xp_compare, first_idx, shape=(), name_hint="interp_first_xp"
        )
        last_xp = _gather(
            ctx, xp_compare, last_idx, shape=(), name_hint="interp_last_xp"
        )
        first_fp = _gather(
            ctx, fp_arith, first_idx, shape=(), name_hint="interp_first_fp"
        )
        last_fp = _gather(ctx, fp_arith, last_idx, shape=(), name_hint="interp_last_fp")

        below = ctx.builder.Less(
            x_compare,
            first_xp,
            _outputs=[ctx.fresh_name("interp_below")],
        )
        below.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(below, x_shape)
        _ensure_value_metadata(ctx, below)
        above = ctx.builder.Greater(
            x_compare,
            last_xp,
            _outputs=[ctx.fresh_name("interp_above")],
        )
        above.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(above, x_shape)
        _ensure_value_metadata(ctx, above)

        left_applied = ctx.builder.Where(
            below,
            first_fp,
            interpolated,
            _outputs=[ctx.fresh_name("interp_left_applied")],
        )
        left_applied.type = ir.TensorType(out_enum)
        _stamp_type_and_shape(left_applied, x_shape)
        _ensure_value_metadata(ctx, left_applied)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("interp_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("interp_out")
        result = ctx.builder.Where(
            above,
            last_fp,
            left_applied,
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = ir.TensorType(out_enum)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            out_shape = cast(
                tuple[DimInput, ...], tuple(getattr(out_var.aval, "shape", ()))
            )
            _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.interp not found for monkey patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                x: ArrayLike,
                xp: ArrayLike,
                fp: ArrayLike,
                left: ArrayLike | str | None = None,
                right: ArrayLike | str | None = None,
                period: ArrayLike | None = None,
            ) -> jax.Array:
                if left is not None or right is not None:
                    raise NotImplementedError(
                        "jnp.interp with explicit left/right is not supported for ONNX export"
                    )
                if period is not None:
                    raise NotImplementedError(
                        "jnp.interp with period is not supported for ONNX export"
                    )
                return cls._PRIM.bind(jnp.asarray(x), jnp.asarray(xp), jnp.asarray(fp))

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


@JnpInterpPlugin._PRIM.def_impl
def _interp_impl(x: object, xp: object, fp: object) -> jax.Array:
    orig = get_orig_impl(JnpInterpPlugin._PRIM, JnpInterpPlugin._FUNC_NAME)
    return cast(jax.Array, orig(x, xp, fp))


def _interp_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
) -> tuple[jax.Array, BatchDim]:
    x, xp, fp = batched_args
    x_bdim, xp_bdim, fp_bdim = batch_dims
    if xp_bdim is not None or fp_bdim is not None:
        raise NotImplementedError("vmap over jnp.interp xp/fp is not supported")
    result = JnpInterpPlugin._PRIM.bind(x, xp, fp)
    return result, x_bdim


batching.primitive_batchers[JnpInterpPlugin._PRIM] = _interp_batch_rule
