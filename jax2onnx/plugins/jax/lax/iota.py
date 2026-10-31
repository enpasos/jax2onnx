# jax2onnx/plugins/jax/lax/iota.py

from __future__ import annotations

from typing import Any, Final, cast

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._ir_shapes import (
    DimInput,
    _ensure_value_metadata,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import (
    _const_i64,
    _lower_i64_vector,
    _scalar_i64,
)
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


_SUPPORTED_IOTA_DTYPES: Final[frozenset[np.dtype[Any]]] = frozenset(
    {
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.int32),
        np.dtype(np.int64),
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
        np.dtype(np.uint64),
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.float16),
        np.dtype(jnp.bfloat16),
        np.dtype(np.bool_),
    }
)
_OPSET27_NATIVE_RANGE_DTYPES: Final[frozenset[np.dtype[Any]]] = frozenset(
    {
        np.dtype(np.float16),
        np.dtype(jnp.bfloat16),
    }
)


def _const_scalar_of_dtype(
    ctx: LoweringContextProtocol,
    value: int | float,
    dtype: np.dtype[Any],
) -> ir.Value:
    return ctx.bind_const_for_var(object(), np.asarray(value, dtype=dtype))


def _source_int32_range_extent(ctx: LoweringContextProtocol, eqn: Any) -> int | None:
    """Use Range11 INT32 only while its FP32 length calculation is exact."""
    if ctx.precision_policy is not PrecisionPolicy.SOURCE:
        return None
    if np.dtype(eqn.params["dtype"]) != np.dtype(np.int32):
        return None
    shape = eqn.params.get("shape")
    dimension = eqn.params.get("dimension")
    if (
        type(shape) is not tuple
        or type(dimension) is not int
        or not 0 <= dimension < len(shape)
    ):
        return None
    extent = shape[dimension]
    if (
        isinstance(extent, (int, np.integer))
        and not isinstance(extent, (bool, np.bool_))
        and 0 <= extent <= (1 << 24)
    ):
        return int(extent)
    return None


@register_native_primitive(
    jaxpr_primitive=jax.lax.iota_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.iota.html",
    onnx=[
        {
            "component": "Range",
            "doc": "https://onnx.ai/onnx/operators/onnx__Range.html",
        }
    ],
    since="0.5.0",
    context="primitives.lax",
    component="iota",
    testcases=[
        {
            "testcase": "iota_int32",
            "callable": lambda: jax.lax.iota(np.int32, 5),
            "input_shapes": [],
            "post_check_onnx_graph": EG(
                ["Range:5 -> Cast:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "iota_float32",
            "callable": lambda: jax.lax.iota(np.float32, 10),
            "input_shapes": [],
            "post_check_onnx_graph": EG(
                ["Range:10 -> Cast:10"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "iota_bfloat16_opset23_cast_fallback",
            "callable": lambda: jax.lax.iota(jnp.bfloat16, 5),
            "input_shapes": [],
            "opset_version": 23,
            "expected_output_dtypes": [jnp.bfloat16],
            "post_check_onnx_graph": EG(
                ["Range:5 -> Cast:5"],
                no_unused_inputs=True,
            ),
            "skip_numeric_validation": True,
            "run_only_f32_variant": True,
        },
        {
            "testcase": "iota_bfloat16_opset27_native_range",
            "callable": lambda: jax.lax.iota(jnp.bfloat16, 5),
            "input_shapes": [],
            "opset_version": 27,
            "expected_output_dtypes": [jnp.bfloat16],
            "post_check_onnx_graph": EG(
                [
                    (
                        "Range:5",
                        {
                            "counts": {"Range": 1},
                            "must_absent": ["Cast"],
                        },
                    )
                ],
                no_unused_inputs=True,
            ),
            "skip_numeric_validation": True,
            "run_only_f32_variant": True,
        },
        {
            "testcase": "iota_float16_opset27_native_range",
            "callable": lambda: jax.lax.iota(np.float16, 5),
            "input_shapes": [],
            "opset_version": 27,
            "expected_output_dtypes": [np.float16],
            "post_check_onnx_graph": EG(
                [
                    (
                        "Range:5",
                        {
                            "counts": {"Range": 1},
                            "must_absent": ["Cast"],
                        },
                    )
                ],
                no_unused_inputs=True,
            ),
            "skip_numeric_validation": True,
            "run_only_f32_variant": True,
        },
        {
            "testcase": "iota_uint8",
            "callable": lambda: jax.lax.iota(np.uint8, 8),
            "input_shapes": [],
            "expected_output_dtypes": [np.uint8],
            "post_check_onnx_graph": EG(
                ["Range:8 -> Cast:8"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "broadcasted_iota",
            "callable": lambda: jax.lax.broadcasted_iota(np.int32, (3, 4), 1),
            "input_shapes": [],
            "post_check_onnx_graph": EG(
                ["Range:4 -> Unsqueeze:1x4 -> Expand:3x4 -> Cast:3x4"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class IotaPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.iota`` (and broadcasted variants) with pure IR ops."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        if _source_int32_range_extent(ctx, eqn) is not None:
            self._lower_source_int32_range(ctx, eqn)
            return
        builder = ctx.builder

        params = eqn.params
        dtype = np.dtype(params["dtype"])
        shape_param = params.get("shape", ())
        dimension = int(params.get("dimension", 0))

        out_var = eqn.outvars[0]

        if not shape_param:
            # scalar iota is treated as vector of length size param (when provided as int)
            shape_param = (params.get("size", 0),)

        shape: tuple[DimInput, ...] = tuple(shape_param)

        rank = len(shape)
        if dimension < 0 or dimension >= rank:
            raise ValueError(
                f"iota dimension {dimension} out of range for shape {shape}"
            )

        if dtype not in _SUPPORTED_IOTA_DTYPES:
            raise TypeError(f"Unsupported dtype for lax.iota: {dtype}")
        target_dtype = numpy_dtype_to_ir(dtype)
        dim_extent = shape[dimension]
        use_native_range_dtype = (
            int(getattr(builder, "opset", 0) or 0) >= 27
            and dtype in _OPSET27_NATIVE_RANGE_DTYPES
            and isinstance(dim_extent, (int, np.integer))
            and int(dim_extent) <= (256 if dtype.name == "bfloat16" else 2048)
        )
        range_dtype = target_dtype if use_native_range_dtype else ir.DataType.INT64

        if use_native_range_dtype and isinstance(dim_extent, (int, np.integer)):
            limit = _const_scalar_of_dtype(ctx, int(dim_extent), dtype)
        else:
            limit_vec = _lower_i64_vector(ctx, [dim_extent], "iota_limit_vec")
            squeeze_axes = _const_i64(
                ctx, np.asarray([0], dtype=np.int64), "iota_limit_squeeze_axes"
            )
            limit = cast(
                ir.Value,
                builder.Squeeze(
                    limit_vec,
                    squeeze_axes,
                    _outputs=[ctx.fresh_name("iota_limit")],
                ),
            )
            limit.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(limit, ())
            _ensure_value_metadata(ctx, limit)
            if use_native_range_dtype:
                limit = cast(
                    ir.Value,
                    builder.Cast(
                        limit,
                        _outputs=[ctx.fresh_name("iota_limit_cast")],
                        to=int(target_dtype.value),
                    ),
                )
                limit.type = ir.TensorType(target_dtype)
                _stamp_type_and_shape(limit, ())
                _ensure_value_metadata(ctx, limit)

        # Build the 1-D range along the chosen axis.
        if use_native_range_dtype:
            start = _const_scalar_of_dtype(ctx, 0, dtype)
            delta = _const_scalar_of_dtype(ctx, 1, dtype)
        else:
            start = _scalar_i64(ctx, 0, "iota_start")
            delta = _scalar_i64(ctx, 1, "iota_delta")

        range_out = cast(
            ir.Value,
            builder.Range(
                start,
                limit,
                delta,
                _outputs=[ctx.fresh_name("iota_range")],
            ),
        )
        range_out.type = ir.TensorType(range_dtype)
        _stamp_type_and_shape(range_out, (dim_extent,))
        _ensure_value_metadata(ctx, range_out)

        current = range_out
        if rank > 1:
            axes = [ax for ax in range(rank) if ax != dimension]
            axes_tensor = (
                _const_i64(ctx, np.asarray(axes, dtype=np.int64), "iota_unsq_axes")
                if axes
                else None
            )
            if axes_tensor is not None:
                unsq_shape: list[DimInput] = [1] * rank
                unsq_shape[dimension] = dim_extent
                current_unsq = cast(
                    ir.Value,
                    builder.Unsqueeze(
                        current,
                        axes_tensor,
                        _outputs=[ctx.fresh_name("iota_unsq")],
                    ),
                )
                current_unsq.type = ir.TensorType(range_dtype)
                _stamp_type_and_shape(current_unsq, tuple(unsq_shape))
                _ensure_value_metadata(ctx, current_unsq)
                current = current_unsq

            expand_shape = _lower_i64_vector(ctx, shape, "iota_expand_shape")
            expanded = cast(
                ir.Value,
                builder.Expand(
                    current,
                    expand_shape,
                    _outputs=[ctx.fresh_name("iota_expanded")],
                ),
            )
            expanded.type = ir.TensorType(range_dtype)
            _stamp_type_and_shape(expanded, tuple(shape))
            _ensure_value_metadata(ctx, expanded)
            current = expanded

        if target_dtype != range_dtype:
            cast_out = cast(
                ir.Value,
                builder.Cast(
                    current,
                    _outputs=[ctx.fresh_name("iota_cast")],
                    to=int(target_dtype.value),
                ),
            )
            cast_out.type = ir.TensorType(target_dtype)
            _stamp_type_and_shape(cast_out, tuple(shape))
            _ensure_value_metadata(ctx, cast_out)
            ctx.bind_value_for_var(out_var, cast_out)
        else:
            identity_out = cast(
                ir.Value,
                builder.Identity(
                    current,
                    _outputs=[ctx.fresh_name("iota_out")],
                ),
            )
            identity_out.type = ir.TensorType(target_dtype)
            _stamp_type_and_shape(identity_out, tuple(shape))
            _ensure_value_metadata(ctx, identity_out)
            ctx.bind_value_for_var(out_var, identity_out)

    def _lower_source_int32_range(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        from jax2onnx.converter.source_profile import _Site, _tensor, _validate_iota

        site = _Site("native INT32 iota emission", ctx.opset, eqn.primitive.name)
        site.require(
            eqn.primitive is jax.lax.iota_p and not eqn.effects,
            "INT32 iota emission needs the pure native primitive",
            "the existing canonical native iota contract",
            category="lowering-defect",
        )
        _validate_iota(eqn, [_tensor(v, site) for v in eqn.outvars], site)
        shape = tuple(eqn.params["shape"])
        dimension = eqn.params["dimension"]
        extent = _source_int32_range_extent(ctx, eqn)
        if extent is None:
            raise ValueError(
                "SOURCE INT32 Range requires its exact native extent proof"
            )
        start = _const_scalar_of_dtype(ctx, 0, np.dtype(np.int32))
        limit = _const_scalar_of_dtype(ctx, extent, np.dtype(np.int32))
        delta = _const_scalar_of_dtype(ctx, 1, np.dtype(np.int32))
        value = ctx.builder.op("Range", [start, limit, delta])
        value.type = ir.TensorType(ir.DataType.INT32)
        _stamp_type_and_shape(value, (extent,))
        _ensure_value_metadata(ctx, value)
        if len(shape) > 1:
            axes = _const_i64(
                ctx, [i for i in range(len(shape)) if i != dimension], "iota_unsq_axes"
            )
            value = ctx.builder.op("Unsqueeze", [value, axes])
            value.type = ir.TensorType(ir.DataType.INT32)
            _stamp_type_and_shape(
                value, tuple(extent if i == dimension else 1 for i in range(len(shape)))
            )
            _ensure_value_metadata(ctx, value)
            target_shape = _lower_i64_vector(ctx, shape, "iota_expand_shape")
            value = ctx.builder.op("Expand", [value, target_shape])
            value.type = ir.TensorType(ir.DataType.INT32)
            _stamp_type_and_shape(value, shape)
            _ensure_value_metadata(ctx, value)
        ctx.bind_value_for_var(eqn.outvars[0], value)
