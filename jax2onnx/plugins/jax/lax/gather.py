# jax2onnx/plugins/jax/lax/gather.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

from .gather_helpers import get_gir_output_shape
from .gather_compile import compile_to_gir

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jax2onnx.converter.ir_context import IRContext


def _is_integer_dtype(dtype) -> bool:
    try:
        return np.issubdtype(np.dtype(dtype), np.integer)
    except TypeError:
        return False


def _dtype_enum_from_value(val: ir.Value) -> ir.DataType:
    dtype = getattr(getattr(val, "type", None), "dtype", None)
    if dtype is None:
        raise TypeError("Missing dtype on value; ensure inputs are typed.")
    return dtype


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
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Gather:2x3 -> Mul:2x3 -> Sin:2x3 -> Add:2x3 -> Greater:2x3 -> Where:2x3",
                        "inputs": {2: {"const": 0.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
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
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Gather:2x3 -> Mul:2x3 -> Sin:2x3 -> Add:2x3 -> Greater:2x3 -> Where:2x3",
                        "inputs": {2: {"const": 0.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
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
            "post_check_onnx_graph": EG(
                ["Gather:2x3"],
                no_unused_inputs=True,
            ),
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
            "post_check_onnx_graph": EG(
                ["Gather:2x3"],
                no_unused_inputs=True,
            ),
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
            "post_check_onnx_graph": EG(
                ["Gather:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "gather_dynamic_batch_simple_index",
            "callable": lambda x: x[:, 0, :],
            "input_shapes": [("B", 50, 256)],
            "expected_output_shapes": [("B", 256)],
            "post_check_onnx_graph": EG(
                [
                    "Slice -> Squeeze",
                    {
                        "path": "Transpose:50xBx256 -> Gather:Bx256",
                        "inputs": {1: {"const": 0.0}},
                    },
                ],
                mode="any",
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class GatherPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.gather`` for the common index patterns exercised in tests."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        data_var, indices_var = eqn.invars
        out_var = eqn.outvars[0]
        constant_indices_value = ctx.try_evaluate_const(indices_var, _eval_primitive)

        ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("gather_out"))

        gir = compile_to_gir(eqn, constant_indices_value)

        current_indices_var = ctx.get_value_for_var(
            indices_var, name_hint=ctx.fresh_name("gather_indices")
        )
        current_data_var = ctx.get_value_for_var(
            data_var, name_hint=ctx.fresh_name("gather_data")
        )

        for gir_instr in gir:
            if gir_instr["op"] == "index_tensor":
                current_indices_var = self._emit_constant_index(ctx, gir_instr)
            elif gir_instr["op"] == "index_transpose":
                current_indices_var = self._emit_index_transpose_from_gir(
                    ctx, gir_instr, current_indices_var
                )
            elif gir_instr["op"] == "index_reshape":
                current_indices_var = self._emit_index_reshape_from_gir(
                    ctx, gir_instr, current_indices_var
                )
            elif gir_instr["op"] == "index_lastdim_gather":
                current_indices_var = self._emit_index_lastdim_gather_from_gir(
                    ctx, gir_instr, current_indices_var
                )
            elif gir_instr["op"] == "index_expand":
                current_indices_var = self._emit_index_expand_range_gir_from_gir(
                    ctx, gir_instr, current_indices_var
                )
            elif gir_instr["op"] == "ONNX_Gather":
                current_data_var = self._emit_gather_from_gir(
                    ctx, gir_instr, current_data_var, current_indices_var
                )
            elif gir_instr["op"] == "ONNX_GatherND":
                current_data_var = self._emit_gather_nd_from_gir(
                    ctx, gir_instr, current_data_var, current_indices_var
                )
            elif gir_instr["op"] == "transpose":
                current_data_var = self._emit_transpose_from_gir(
                    ctx, gir_instr, current_data_var
                )
            elif gir_instr["op"] == "ONNX_Slice":
                current_data_var = self._emit_slice_from_gir(
                    ctx, gir_instr, current_data_var
                )
            else:
                raise RuntimeError(f"Unhandled internal op in Gather: {gir_instr}")

        ctx.bind_value_for_var(out_var, current_data_var)

    def _emit_transpose_from_gir(
        self, ctx: "IRContext", gir_instr: dict, input_tensor: ir.Value
    ) -> ir.Value:
        result_val = ctx.builder.Transpose(
            input_tensor,
            _outputs=[ctx.fresh_name("transpose_gather_data")],
            perm=gir_instr["numpy_transpose"],
        )
        _stamp_type_and_shape(result_val, get_gir_output_shape(gir_instr))
        result_val.type = ir.TensorType(_dtype_enum_from_value(input_tensor))
        _ensure_value_metadata(ctx, result_val)
        return result_val

    def _emit_index_transpose_from_gir(
        self, ctx: "IRContext", gir_instr: dict, index_tensor: ir.Value
    ) -> ir.Value:
        result_val = ctx.builder.Transpose(
            index_tensor,
            _outputs=[ctx.fresh_name("transpose_gather_index")],
            perm=gir_instr["numpy_transpose"],
        )
        _stamp_type_and_shape(result_val, get_gir_output_shape(gir_instr))
        result_val.type = ir.TensorType(_dtype_enum_from_value(index_tensor))
        _ensure_value_metadata(ctx, result_val)
        return result_val

    def _emit_index_reshape_from_gir(
        self, ctx: "IRContext", gir_instr: dict, index_tensor: ir.Value
    ) -> ir.Value:
        new_shape = get_gir_output_shape(gir_instr)
        new_shape_val = _const_i64(
            ctx, np.asarray(new_shape, dtype=np.int64), "new_shape_for_gather_index"
        )
        result_val = ctx.builder.Reshape(
            index_tensor,
            new_shape_val,
            _outputs=[ctx.fresh_name("reshape_gather_index")],
        )
        _stamp_type_and_shape(result_val, new_shape)
        result_val.type = ir.TensorType(_dtype_enum_from_value(index_tensor))
        _ensure_value_metadata(ctx, result_val)
        return result_val

    def _emit_index_lastdim_gather_from_gir(
        self, ctx: "IRContext", gir_instr: dict, index_tensor: ir.Value
    ) -> ir.Value:
        gather_indices_val = _const_i64(
            ctx,
            np.asarray(gir_instr["gather_indices"], dtype=np.int64),
            "gather_index_for_lastdim_gather_index",
        )
        result_val = ctx.builder.Gather(
            index_tensor,
            gather_indices_val,
            axis=-1,
            _outputs=[ctx.fresh_name("lastdim_reorder_gather_on_gather_index")],
        )
        _stamp_type_and_shape(result_val, get_gir_output_shape(gir_instr))
        result_val.type = ir.TensorType(_dtype_enum_from_value(index_tensor))
        _ensure_value_metadata(ctx, result_val)
        return result_val

    def _emit_index_expand_range_gir_from_gir(
        self, ctx: "IRContext", gir_instr: dict, index_tensor: ir.Value
    ) -> ir.Value:
        # TODO: relatively complex, one possible implementation in numpy:

        # new_dims_shape = [dim["slice_size"] for dim in instr["new_dims"]]
        # indices_var_index = [dim["indices_var_index"] for dim in instr["new_dims"]]
        # index_tensor = np.reshape(index_tensor, tuple(instr["input_shape"][:-1] + [1]*len(new_dims_shape) + instr["input_shape"][-1:]))
        # new_parts = np.zeros(tuple([1]*(len(instr["input_shape"])-1) + new_dims_shape + instr["input_shape"][-1:]))
        # for i,size in enumerate(new_dims_shape):
        #     A = np.reshape(np.arange(size), tuple([1]*(len(instr["input_shape"])-1) + [1]*i + [size] + [1]*(len(new_dims_shape)-1-i)))
        #     new_parts[...,indices_var_index[i]] = A
        # index_tensor = index_tensor + new_parts

        raise RuntimeError(
            "index expand for breaking complex gather+slice is not yet supported"
        )

    def _emit_gather_from_gir(
        self,
        ctx: "IRContext",
        gir_instr: dict,
        input_tensor: ir.Value,
        index_tensor: ir.Value,
    ) -> ir.Value:
        # emit a cast if the indices are not int32 or int64, cast to int64 just to be sure
        if index_tensor.type != ir.TensorType(
            ir.DataType.INT64
        ) and index_tensor.type != ir.TensorType(ir.DataType.INT32):
            index_tensor_final = ctx.builder.Cast(
                index_tensor,
                _outputs=[ctx.fresh_name("gather_nd_indices")],
                to=int(ir.DataType.INT64.value),
            )
            index_tensor_final.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(index_tensor_final, tuple(index_tensor.shape))
            _ensure_value_metadata(ctx, index_tensor_final)
        else:
            index_tensor_final = index_tensor

        gather_axis = None

        # find the gather axis, this will be relevant if we later add some optimisations
        for dim in gir_instr["dims"]:
            if dim["mode"] == "gather":
                assert gather_axis is None
                gather_axis = dim["dim"]
            else:
                assert dim["mode"] == "passthrough"

        assert gather_axis is not None

        # emit gather
        result_val = ctx.builder.Gather(
            input_tensor,
            index_tensor_final,
            axis=gather_axis,
            _outputs=[ctx.fresh_name("simple_gather")],
        )
        _stamp_type_and_shape(result_val, get_gir_output_shape(gir_instr))
        result_val.type = ir.TensorType(_dtype_enum_from_value(input_tensor))
        _ensure_value_metadata(ctx, result_val)
        return result_val

    def _convert_symbolic_1d_int_vec(
        self, ctx: "IRContext", values: list[Any], name: str
    ) -> ir.Value:
        if all(isinstance(x, int) for x in values):
            return _const_i64(ctx, np.asarray(values, dtype=np.int64), name)
        else:
            return ctx.dim_expr_lowerer(values)

    def _emit_slice_from_gir(
        self, ctx: "IRContext", gir_instr: dict, input_tensor: ir.Value
    ) -> ir.Value:
        axes = [dim["dim"] for dim in gir_instr["dims"] if dim["mode"] == "range_slice"]
        starts = [
            dim["start"] for dim in gir_instr["dims"] if dim["mode"] == "range_slice"
        ]
        ends = [dim["end"] for dim in gir_instr["dims"] if dim["mode"] == "range_slice"]

        starts_val = self._convert_symbolic_1d_int_vec(
            ctx, starts, "gather_slice_starts"
        )
        ends_val = self._convert_symbolic_1d_int_vec(ctx, ends, "gather_slice_ends")
        axes_val = self._convert_symbolic_1d_int_vec(ctx, axes, "gather_slice_axes")

        result_val = ctx.builder.Slice(
            input_tensor,
            starts_val,
            ends_val,
            axes_val,
            _outputs=[ctx.fresh_name("gather_slice")],
        )
        _stamp_type_and_shape(result_val, get_gir_output_shape(gir_instr))
        result_val.type = ir.TensorType(_dtype_enum_from_value(input_tensor))
        _ensure_value_metadata(ctx, result_val)
        return result_val

    def _emit_gather_nd_from_gir(
        self,
        ctx: "IRContext",
        gir_instr: dict,
        input_tensor: ir.Value,
        index_tensor: ir.Value,
    ) -> ir.Value:
        batch_dims = 0
        # emit cast on index if it is not the correct type
        if index_tensor.type != ir.TensorType(ir.DataType.INT64):
            index_tensor_final = ctx.builder.Cast(
                index_tensor,
                _outputs=[ctx.fresh_name("gather_nd_indices")],
                to=int(ir.DataType.INT64.value),
            )
            index_tensor_final.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(index_tensor_final, tuple(index_tensor.shape))
            _ensure_value_metadata(ctx, index_tensor_final)
        else:
            index_tensor_final = index_tensor

        # count batch dimensions
        for dim in gir_instr["dims"]:
            if dim["mode"] == "batched":
                batch_dims += 1
            else:
                assert dim["mode"] in ["passthrough", "gather"]

        # emit GatherND
        result_val = ctx.builder.GatherND(
            input_tensor,
            index_tensor_final,
            batch_dims=batch_dims,
            _outputs=[ctx.fresh_name("gather_nd")],
        )
        _stamp_type_and_shape(result_val, get_gir_output_shape(gir_instr))
        result_val.type = ir.TensorType(_dtype_enum_from_value(input_tensor))
        _ensure_value_metadata(ctx, result_val)
        return result_val

    def _emit_constant_index(self, ctx: "IRContext", gir_instr: dict) -> ir.Value:
        index_val = _const_i64(
            ctx,
            np.asarray(gir_instr["value"], dtype=np.int64),
            "gather_constant_index_base",
        )
        return index_val


def _masked_gather_trig_local(data, indices):
    data = jnp.asarray(data, dtype=jnp.float64)
    gathered = data[indices]
    result = gathered * jnp.array(2.0, dtype=jnp.float64)
    result = jnp.sin(result) + jnp.cos(result)
    mask = result > jnp.array(0.5, dtype=jnp.float64)
    return jnp.where(mask, result, jnp.array(0.0, dtype=jnp.float64))


def _eval_primitive(primitive, *args, **kwargs):
    return primitive.bind(*args, **kwargs)
