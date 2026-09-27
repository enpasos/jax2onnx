# jax2onnx/plugins/_normalization_utils.py

"""Explicit LayerNorm lowering shared by the Equinox and Flax plugins.

``normalization_mode="auto"`` selects this explicit graph because it has the
best reproducible accuracy (see ``tests/extra_tests/test_layer_norm_precision.py``
for the locked bounds). It follows the framework's statistics: two-pass variance (with exact zeros for constant
rows) or Flax's clamped fast variance. ONNX ``LayerNormalization`` is emitted
only for ``"prefer_native"`` at opset 17 or newer; ONNX Runtime's CPU kernel for
it accumulates statistics in one sequential float32 pass, which loses precision
on rows with large activations. Squares are written as ``Mul`` so ONNX Runtime's
``LayerNormFusion`` does not rebuild the native op.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, cast

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import ir_dtype_to_numpy
from jax2onnx.plugins._ir_shapes import (
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins.jax.lax._opset_utils import builder_reduce_with_axes

LAYER_NORM_MIN_NATIVE_OPSET: Final[int] = 17

LAYER_NORM_ONNX_COMPONENTS: Final[list[dict[str, str]]] = [
    {"component": op, "doc": f"https://onnx.ai/onnx/operators/onnx__{op}.html"}
    for op in (
        "LayerNormalization",
        "ReduceMean",
        "ReduceMin",
        "ReduceMax",
        "Sub",
        "Mul",
        "Add",
        "Max",
        "Sqrt",
        "Div",
        "Equal",
        "And",
        "Where",
        "Cast",
    )
]

_LOW_PRECISION: Final[frozenset[ir.DataType]] = frozenset(
    {ir.DataType.FLOAT16, ir.DataType.BFLOAT16}
)


def use_native_layer_norm(ctx: LoweringContextProtocol) -> bool:
    """Whether to emit ONNX ``LayerNormalization`` rather than the explicit graph."""
    return (
        ctx.normalization_mode == "prefer_native"
        and int(ctx.opset) >= LAYER_NORM_MIN_NATIVE_OPSET
    )


def lower_explicit_layer_norm(
    ctx: LoweringContextProtocol,
    x_val: ir.Value,
    scale_val: ir.Value,
    bias_val: ir.Value,
    *,
    x_shape: Sequence[Any],
    axis: int,
    epsilon: float,
    use_fast_variance: bool,
    clamp_negative_variance: bool,
) -> ir.Value:
    """Normalize ``x`` over axes ``axis..rank-1``, then apply ``scale`` and ``bias``.

    ``scale``/``bias`` must broadcast against the trailing normalized axes.
    Float16/bfloat16 inputs are normalized in float32 and cast back, like the
    Equinox and Flax implementations.
    """
    builder = ctx.builder
    rank = len(x_shape)
    axis = axis % rank if rank else 0
    reduce_axes = tuple(range(axis, rank))
    dims = tuple(
        _dim_label_from_value_or_aval(x_val, tuple(x_shape), idx) or x_shape[idx]
        for idx in range(rank)
    )
    reduced_dims = tuple(1 if idx >= axis else dim for idx, dim in enumerate(dims))

    x_dtype = x_val.dtype
    if x_dtype is None:
        raise TypeError("explicit LayerNorm lowering requires a typed input")
    stats_dtype = ir.DataType.FLOAT if x_dtype in _LOW_PRECISION else x_dtype
    stats_np_dtype = ir_dtype_to_numpy(stats_dtype, default=None)
    if stats_np_dtype is None:
        raise TypeError(f"unsupported LayerNorm dtype {x_dtype}")

    def stamp(
        value: ir.Value, value_dims: Sequence[Any], dtype: ir.DataType
    ) -> ir.Value:
        value.type = ir.TensorType(dtype)
        _stamp_type_and_shape(value, tuple(value_dims))
        _ensure_value_metadata(ctx, value)
        return value

    def stats(value: Any, value_dims: Sequence[Any]) -> ir.Value:
        return stamp(cast(ir.Value, value), value_dims, stats_dtype)

    def boolean(value: Any) -> ir.Value:
        return stamp(cast(ir.Value, value), reduced_dims, ir.DataType.BOOL)

    def reduce(value: ir.Value, op_type: str, name_hint: str) -> ir.Value:
        out = builder_reduce_with_axes(
            ctx,
            value,
            op_type=op_type,
            axes=reduce_axes,
            keepdims=1,
            name_hint=name_hint,
        )
        return stats(out, reduced_dims)

    x = x_val
    if stats_dtype != x_dtype:
        x = stats(
            builder.Cast(
                x_val,
                to=int(stats_dtype.value),
                _outputs=[ctx.fresh_name("ln_stats_input")],
            ),
            dims,
        )

    def scalar(value: float, name_hint: str) -> ir.Value:
        const = ctx.bind_const_for_var(
            object(), np.asarray(value, dtype=stats_np_dtype)
        )
        return cast_param_like(ctx, const, x, name_hint=name_hint)

    zero = scalar(0.0, "ln_zero_cast")
    mean = reduce(x, "ReduceMean", "ln_mean")
    centered = stats(
        builder.Sub(x, mean, _outputs=[ctx.fresh_name("ln_centered")]), dims
    )

    if use_fast_variance:
        squared = stats(
            builder.Mul(x, x, _outputs=[ctx.fresh_name("ln_squared")]), dims
        )
        second_moment = reduce(squared, "ReduceMean", "ln_second_moment")
        mean_squared = stats(
            builder.Mul(mean, mean, _outputs=[ctx.fresh_name("ln_mean_squared")]),
            reduced_dims,
        )
        variance = stats(
            builder.Sub(
                second_moment, mean_squared, _outputs=[ctx.fresh_name("ln_variance")]
            ),
            reduced_dims,
        )
    else:
        # Runtime reductions can move the mean of a constant row by a few ulps;
        # force exact zeros for finite constant rows (as for GroupNorm).
        row_min = reduce(x, "ReduceMin", "ln_row_min")
        row_max = reduce(x, "ReduceMax", "ln_row_max")
        equal_extrema = boolean(
            builder.Equal(
                row_min, row_max, _outputs=[ctx.fresh_name("ln_equal_extrema")]
            )
        )
        # The mean's self-subtraction is zero only if every element is finite:
        # ReduceMin/ReduceMax may skip NaN, but ReduceMean propagates it.
        finite_delta = stats(
            builder.Sub(mean, mean, _outputs=[ctx.fresh_name("ln_finite_delta")]),
            reduced_dims,
        )
        finite_row = boolean(
            builder.Equal(
                finite_delta, zero, _outputs=[ctx.fresh_name("ln_finite_row")]
            )
        )
        constant_row = boolean(
            builder.And(
                equal_extrema, finite_row, _outputs=[ctx.fresh_name("ln_constant_row")]
            )
        )
        centered = stats(
            builder.Where(
                constant_row,
                zero,
                centered,
                _outputs=[ctx.fresh_name("ln_stable_centered")],
            ),
            dims,
        )
        squared = stats(
            builder.Mul(centered, centered, _outputs=[ctx.fresh_name("ln_squared")]),
            dims,
        )
        variance = reduce(squared, "ReduceMean", "ln_variance")

    if clamp_negative_variance:
        variance = stats(
            builder.Max(
                variance, zero, _outputs=[ctx.fresh_name("ln_nonnegative_variance")]
            ),
            reduced_dims,
        )
    variance_eps = stats(
        builder.Add(
            variance,
            scalar(epsilon, "ln_epsilon_cast"),
            _outputs=[ctx.fresh_name("ln_variance_eps")],
        ),
        reduced_dims,
    )
    stddev = stats(
        builder.Sqrt(variance_eps, _outputs=[ctx.fresh_name("ln_stddev")]),
        reduced_dims,
    )
    normalized = stats(
        builder.Div(centered, stddev, _outputs=[ctx.fresh_name("ln_normalized")]),
        dims,
    )
    scale = cast_param_like(ctx, scale_val, x, name_hint="ln_scale_cast")
    bias = cast_param_like(ctx, bias_val, x, name_hint="ln_bias_cast")
    scaled = stats(
        builder.Mul(normalized, scale, _outputs=[ctx.fresh_name("ln_scaled")]), dims
    )
    result = stats(
        builder.Add(scaled, bias, _outputs=[ctx.fresh_name("LayerNorm")]), dims
    )
    if stats_dtype != x_dtype:
        result = stamp(
            cast(
                ir.Value,
                builder.Cast(
                    result,
                    to=int(x_dtype.value),
                    _outputs=[ctx.fresh_name("LayerNorm")],
                ),
            ),
            dims,
            x_dtype,
        )
    return result
