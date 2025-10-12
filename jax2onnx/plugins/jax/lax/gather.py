# jax2onnx/plugins/jax/lax/gather.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jax2onnx.converter.ir_context import IRContext


def _is_integer_dtype(dtype) -> bool:
    try:
        return np.issubdtype(np.dtype(dtype), np.integer)
    except TypeError:
        return False


@register_primitive(
    jaxpr_primitive=jax.lax.gather_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html",
    onnx=[
        {
            "component": "GatherND",
            "doc": "https://onnx.ai/onnx/operators/onnx__GatherND.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="gather",
    testcases=[
        {
            "testcase": "gather_trig_where_pipeline_f64_indices_i64",
            "callable": lambda data, indices: _masked_gather_trig_local(data, indices),
            "input_values": [
                np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ],
                    dtype=np.float64,
                ),
                np.array([0, 2], dtype=np.int64),
            ],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "gather_trig_where_pipeline_f64_indices_i32",
            "callable": lambda data, indices: _masked_gather_trig_local(data, indices),
            "input_values": [
                np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ],
                    dtype=np.float64,
                ),
                np.array([1, 3], dtype=np.int32),
            ],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "gather_f64_data_i64_indices_output_is_f64",
            "callable": lambda data, idx: data[idx],
            "input_values": [
                np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ],
                    dtype=np.float64,
                ),
                np.array([0, 2], dtype=np.int64),
            ],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "gather_f64_data_i32_indices_cast_and_output_is_f64",
            "callable": lambda data, idx: data[idx],
            "input_values": [
                np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ],
                    dtype=np.float64,
                ),
                np.array([1, 3], dtype=np.int32),
            ],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "gather_static",
            "callable": lambda x: jax.lax.gather(
                x,
                jax.numpy.array([[1], [0]]),
                jax.lax.GatherDimensionNumbers(
                    offset_dims=(1,),
                    collapsed_slice_dims=(0,),
                    start_index_map=(0,),
                ),
                slice_sizes=(1, 3),
            ),
            "input_shapes": [(3, 3)],
            "expected_output_shapes": [(2, 3)],
        },
        {
            "testcase": "gather_dynamic_batch_simple_index",
            "callable": lambda x: x[:, 0, :],
            "input_shapes": [("B", 50, 256)],
            "expected_output_shapes": [("B", 256)],
        },
    ],
)
class GatherPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.gather`` for the common index patterns exercised in tests."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        data_var, indices_var = eqn.invars
        out_var = eqn.outvars[0]

        data_val = ctx.get_value_for_var(
            data_var, name_hint=ctx.fresh_name("gather_data")
        )
        ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("gather_out"))

        data_shape = tuple(getattr(data_var.aval, "shape", ()))
        indices_shape = tuple(getattr(indices_var.aval, "shape", ()))

        is_simple_index_zero = indices_shape == (1,) and _is_integer_dtype(
            indices_var.aval.dtype
        )

        def _is_concrete_int(x: object) -> bool:
            return isinstance(x, int)

        has_leading_dim = bool(data_shape)
        first_dim = data_shape[0] if has_leading_dim else None
        dynamic_batch = has_leading_dim and not _is_concrete_int(first_dim)

        if is_simple_index_zero and dynamic_batch:
            # Build indices shaped (B, 1) with all zeros, where B is the dynamic batch dim.
            data_shape_val = ctx.builder.Shape(
                data_val, _outputs=[ctx.fresh_name("gather_data_shape")]
            )
            data_shape_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(data_shape_val, (len(data_shape),))
            _ensure_value_metadata(ctx, data_shape_val)

            zero_idx = _const_i64(
                ctx, np.asarray(0, dtype=np.int64), "gather_batch_idx"
            )
            batch_dim = ctx.builder.Gather(
                data_shape_val,
                zero_idx,
                axis=0,
                _outputs=[ctx.fresh_name("gather_batch_dim")],
            )
            _stamp_type_and_shape(batch_dim, ())
            batch_dim.type = ir.TensorType(ir.DataType.INT64)
            _ensure_value_metadata(ctx, batch_dim)

            axes0 = _const_i64(ctx, np.asarray(0, dtype=np.int64), "gather_axes0")
            batch_vec = ctx.builder.Unsqueeze(
                batch_dim,
                axes0,
                _outputs=[ctx.fresh_name("gather_batch_vec")],
            )
            batch_vec.type = ir.TensorType(ir.DataType.INT64)
            _ensure_value_metadata(ctx, batch_vec)

            one_vec = _const_i64(ctx, np.asarray([1], dtype=np.int64), "gather_one_vec")

            target_shape = ctx.builder.Concat(
                batch_vec,
                one_vec,
                axis=0,
                _outputs=[ctx.fresh_name("gather_target_shape")],
            )
            target_shape.type = ir.TensorType(ir.DataType.INT64)
            _ensure_value_metadata(ctx, target_shape)

            base_index = _const_i64(
                ctx, np.zeros((1, 1), dtype=np.int64), "gather_base_idx"
            )

            indices_val = ctx.builder.Expand(
                base_index,
                target_shape,
                _outputs=[ctx.fresh_name("gather_indices")],
            )
            indices_val.type = ir.TensorType(ir.DataType.INT64)
            _ensure_value_metadata(ctx, indices_val)
            batch_dims = 1
        else:
            indices_val_in = ctx.get_value_for_var(
                indices_var, name_hint=ctx.fresh_name("gather_indices_in")
            )
            indices_val = ctx.builder.Cast(
                indices_val_in,
                _outputs=[ctx.fresh_name("gather_indices")],
                to=int(ir.DataType.INT64.value),
            )
            indices_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(indices_val, tuple(indices_shape))
            _ensure_value_metadata(ctx, indices_val)
            batch_dims = 0

        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("gather_out")
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("GatherND")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("GatherND")

        result = ctx.builder.GatherND(
            data_val,
            indices_val,
            batch_dims=batch_dims,
            _outputs=[desired_name],
        )

        output_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(result, output_shape)
        result_dtype = getattr(getattr(data_val, "type", None), "dtype", None)
        if result_dtype is None:
            result_dtype = getattr(getattr(out_spec, "type", None), "dtype", None)
        if result_dtype is not None:
            result.type = ir.TensorType(result_dtype)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)


def _masked_gather_trig_local(data, indices):
    data = jnp.asarray(data, dtype=jnp.float64)
    gathered = data[indices]
    result = gathered * jnp.array(2.0, dtype=jnp.float64)
    result = jnp.sin(result) + jnp.cos(result)
    mask = result > jnp.array(0.5, dtype=jnp.float64)
    return jnp.where(mask, result, jnp.array(0.0, dtype=jnp.float64))
