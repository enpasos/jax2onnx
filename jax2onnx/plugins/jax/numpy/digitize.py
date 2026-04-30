# jax2onnx/plugins/jax/numpy/digitize.py

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


_DIGITIZE_PRIM: Final = make_jnp_primitive("jax.numpy.digitize")

BatchDim: TypeAlias = int | None


def _result_dtype() -> np.dtype[Any]:
    return np.dtype(np.int32)


def _cast_to_dtype(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    from_dtype: np.dtype[Any],
    to_dtype: np.dtype[Any],
    name_hint: str,
    shape: tuple[DimInput, ...],
) -> ir.Value:
    if from_dtype == to_dtype:
        return val
    target_enum = _dtype_to_ir(to_dtype, ctx.builder.enable_double_precision)
    cast_val = cast(
        ir.Value,
        ctx.builder.Cast(
            val,
            to=int(target_enum.value),
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    cast_val.type = ir.TensorType(target_enum)
    _stamp_type_and_shape(cast_val, shape)
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


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


def _gather_bin_scalar(
    ctx: LoweringContextProtocol,
    bins: ir.Value,
    *,
    index: int,
    name_hint: str,
) -> ir.Value:
    idx = _const_i64(ctx, np.asarray(index, dtype=np.int64), f"{name_hint}_index")
    result = cast(
        ir.Value,
        ctx.builder.Gather(
            bins,
            idx,
            axis=0,
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    if getattr(bins, "type", None) is not None:
        result.type = bins.type
    _stamp_type_and_shape(result, ())
    _ensure_value_metadata(ctx, result)
    return result


def _compare(
    ctx: LoweringContextProtocol,
    bins: ir.Value,
    x: ir.Value,
    *,
    op_type: str,
    name_hint: str,
) -> ir.Value:
    if op_type == "Less":
        result = cast(
            ir.Value,
            ctx.builder.Less(bins, x, _outputs=[ctx.fresh_name(name_hint)]),
        )
    elif op_type == "LessOrEqual":
        result = cast(
            ir.Value,
            ctx.builder.LessOrEqual(
                bins,
                x,
                _outputs=[ctx.fresh_name(name_hint)],
            ),
        )
    elif op_type == "Greater":
        result = cast(
            ir.Value,
            ctx.builder.Greater(bins, x, _outputs=[ctx.fresh_name(name_hint)]),
        )
    elif op_type == "GreaterOrEqual":
        result = cast(
            ir.Value,
            ctx.builder.GreaterOrEqual(
                bins,
                x,
                _outputs=[ctx.fresh_name(name_hint)],
            ),
        )
    else:  # pragma: no cover - defensive guard for internal callers
        raise ValueError(f"Unsupported digitize comparison op: {op_type}")
    result.type = ir.TensorType(ir.DataType.BOOL)
    return result


def _count_matches(
    ctx: LoweringContextProtocol,
    bins: ir.Value,
    x: ir.Value,
    *,
    op_type: str,
    compare_shape: tuple[DimInput, ...],
    output_shape: tuple[DimInput, ...],
    reduce_axis: int,
    name_hint: str,
) -> ir.Value:
    comparison = _compare(
        ctx,
        bins,
        x,
        op_type=op_type,
        name_hint=f"{name_hint}_compare",
    )
    _stamp_type_and_shape(comparison, compare_shape)
    _ensure_value_metadata(ctx, comparison)

    out_enum = numpy_dtype_to_ir(_result_dtype())
    counts_input = cast(
        ir.Value,
        ctx.builder.Cast(
            comparison,
            to=int(out_enum.value),
            _outputs=[ctx.fresh_name(f"{name_hint}_counts_input")],
        ),
    )
    counts_input.type = ir.TensorType(out_enum)
    _stamp_type_and_shape(counts_input, compare_shape)
    _ensure_value_metadata(ctx, counts_input)

    axes = _const_i64(
        ctx,
        np.asarray([reduce_axis], dtype=np.int64),
        f"{name_hint}_reduce_axes",
    )
    result = cast(
        ir.Value,
        ctx.builder.ReduceSum(
            counts_input,
            axes,
            keepdims=0,
            _outputs=[ctx.fresh_name(f"{name_hint}_count")],
        ),
    )
    result.type = ir.TensorType(out_enum)
    _stamp_type_and_shape(result, output_shape)
    _ensure_value_metadata(ctx, result)
    return result


@register_primitive(
    jaxpr_primitive=_DIGITIZE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.digitize.html",
    onnx=[
        {"component": "Less", "doc": "https://onnx.ai/onnx/operators/onnx__Less.html"},
        {
            "component": "LessOrEqual",
            "doc": "https://onnx.ai/onnx/operators/onnx__LessOrEqual.html",
        },
        {
            "component": "Greater",
            "doc": "https://onnx.ai/onnx/operators/onnx__Greater.html",
        },
        {
            "component": "GreaterOrEqual",
            "doc": "https://onnx.ai/onnx/operators/onnx__GreaterOrEqual.html",
        },
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.13.0",
    context="primitives.jnp",
    component="digitize",
    testcases=[
        {
            "testcase": "jnp_digitize_increasing_left",
            "callable": lambda x, bins: jnp.digitize(x, bins, right=False),
            "input_values": [
                np.asarray([0.0, 1.0, 2.0, 7.0, 8.0], dtype=np.float32),
                np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["LessOrEqual:5x4 -> Cast:5x4 -> ReduceSum:5", "Where:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_digitize_increasing_right",
            "callable": lambda x, bins: jnp.digitize(x, bins, right=True),
            "input_values": [
                np.asarray([0.0, 1.0, 2.0, 7.0, 8.0], dtype=np.float32),
                np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Less:5x4 -> Cast:5x4 -> ReduceSum:5", "Where:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_digitize_decreasing_left",
            "callable": lambda x, bins: jnp.digitize(x, bins, right=False),
            "input_values": [
                np.asarray([0.0, 1.0, 2.0, 7.0, 8.0], dtype=np.float32),
                np.asarray([7.0, 5.0, 3.0, 1.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Greater:5x4 -> Cast:5x4 -> ReduceSum:5", "Where:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_digitize_scalar_value",
            "callable": lambda x, bins: jnp.digitize(x, bins, right=False),
            "input_values": [
                np.asarray(4, dtype=np.int32),
                np.asarray([1, 3, 5, 7], dtype=np.int32),
            ],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["LessOrEqual:4 -> Cast:4 -> ReduceSum", "Where"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "digitize_vmap_values",
            "callable": lambda x, bins: jax.vmap(
                lambda scalar: jnp.digitize(scalar, bins)
            )(x),
            "input_values": [
                np.asarray([0.0, 2.0, 8.0], dtype=np.float32),
                np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.int32],
        },
    ],
)
class JnpDigitizePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _DIGITIZE_PRIM
    _FUNC_NAME: ClassVar[str] = "digitize"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        bins: core.AbstractValue,
        *,
        right: bool = False,
        method: str = "scan",
    ) -> core.ShapedArray:
        del right, method
        if len(tuple(getattr(bins, "shape", ()))) != 1:
            raise TypeError("jnp.digitize lowering requires 1-D bins")
        return core.ShapedArray(tuple(getattr(x, "shape", ())), _result_dtype())

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        x_var, bins_var = eqn.invars
        (out_var,) = eqn.outvars
        params = getattr(eqn, "params", {})
        right = bool(params.get("right", False))

        x_shape: tuple[DimInput, ...] = tuple(getattr(x_var.aval, "shape", ()))
        bins_shape: tuple[DimInput, ...] = tuple(getattr(bins_var.aval, "shape", ()))
        if len(bins_shape) != 1:
            raise TypeError("jnp.digitize lowering requires 1-D bins")
        bins_len = bins_shape[0]
        if not isinstance(bins_len, (int, np.integer)):
            raise TypeError("jnp.digitize lowering requires static bins length")
        if int(bins_len) < 1:
            raise ValueError("jnp.digitize lowering requires at least one bin")

        x_dtype: np.dtype[Any] = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        bins_dtype: np.dtype[Any] = np.dtype(getattr(bins_var.aval, "dtype", x_dtype))
        compare_dtype: np.dtype[Any] = np.promote_types(x_dtype, bins_dtype)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("digitize_x"))
        bins_val = ctx.get_value_for_var(
            bins_var,
            name_hint=ctx.fresh_name("digitize_bins"),
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("digitize_out")
        )

        x_ready = _cast_to_dtype(
            ctx,
            x_val,
            from_dtype=x_dtype,
            to_dtype=compare_dtype,
            name_hint="digitize_x_cast",
            shape=x_shape,
        )
        bins_ready = _cast_to_dtype(
            ctx,
            bins_val,
            from_dtype=bins_dtype,
            to_dtype=compare_dtype,
            name_hint="digitize_bins_cast",
            shape=bins_shape,
        )

        bins_broadcast = bins_ready
        bins_broadcast_shape: tuple[DimInput, ...] = (int(bins_len),)
        for _ in range(len(x_shape)):
            bins_broadcast_shape = (1, *bins_broadcast_shape)
            bins_broadcast = _unsqueeze(
                ctx,
                bins_broadcast,
                axis=0,
                shape=bins_broadcast_shape,
                name_hint="digitize_bins_unsqueeze",
            )

        x_broadcast = _unsqueeze(
            ctx,
            x_ready,
            axis=len(x_shape),
            shape=(*x_shape, 1),
            name_hint="digitize_x_unsqueeze",
        )

        compare_shape: tuple[DimInput, ...] = (*x_shape, int(bins_len))
        reduce_axis = len(x_shape)
        inc_op = "Less" if right else "LessOrEqual"
        dec_op = "GreaterOrEqual" if right else "Greater"

        increasing_count = _count_matches(
            ctx,
            bins_broadcast,
            x_broadcast,
            op_type=inc_op,
            compare_shape=compare_shape,
            output_shape=x_shape,
            reduce_axis=reduce_axis,
            name_hint="digitize_increasing",
        )
        decreasing_count = _count_matches(
            ctx,
            bins_broadcast,
            x_broadcast,
            op_type=dec_op,
            compare_shape=compare_shape,
            output_shape=x_shape,
            reduce_axis=reduce_axis,
            name_hint="digitize_decreasing",
        )

        first_bin = _gather_bin_scalar(
            ctx,
            bins_ready,
            index=0,
            name_hint="digitize_first_bin",
        )
        last_bin = _gather_bin_scalar(
            ctx,
            bins_ready,
            index=int(bins_len) - 1,
            name_hint="digitize_last_bin",
        )
        increasing = ctx.builder.GreaterOrEqual(
            last_bin,
            first_bin,
            _outputs=[ctx.fresh_name("digitize_increasing_bins")],
        )
        increasing.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(increasing, ())
        _ensure_value_metadata(ctx, increasing)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("digitize_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("digitize_out")

        result = ctx.builder.Where(
            increasing,
            increasing_count,
            decreasing_count,
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = ir.TensorType(numpy_dtype_to_ir(_result_dtype()))
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError(
                    "Original jnp.digitize not found for monkey patching"
                )
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                x: ArrayLike,
                bins: ArrayLike,
                right: bool = False,
                *,
                method: str | None = None,
            ) -> jax.Array:
                method_param = "scan" if method is None else str(method)
                return cls._PRIM.bind(
                    jnp.asarray(x),
                    jnp.asarray(bins),
                    right=bool(right),
                    method=method_param,
                )

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


@JnpDigitizePlugin._PRIM.def_impl
def _digitize_impl(
    x: object,
    bins: object,
    *,
    right: bool = False,
    method: str = "scan",
) -> jax.Array:
    orig = get_orig_impl(JnpDigitizePlugin._PRIM, JnpDigitizePlugin._FUNC_NAME)
    return cast(jax.Array, orig(x, bins, right=right, method=method))


def _digitize_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    right: bool = False,
    method: str = "scan",
) -> tuple[jax.Array, BatchDim]:
    x, bins = batched_args
    x_bdim, bins_bdim = batch_dims
    if bins_bdim is not None:
        raise NotImplementedError("vmap over jnp.digitize bins is not supported")
    result = JnpDigitizePlugin._PRIM.bind(
        x,
        bins,
        right=bool(right),
        method=method,
    )
    return result, x_bdim


batching.primitive_batchers[JnpDigitizePlugin._PRIM] = _digitize_batch_rule
