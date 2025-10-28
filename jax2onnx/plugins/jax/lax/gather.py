# jax2onnx/plugins/jax/lax/gather.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax import lax

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64, _scalar_i64, _shape_of
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

from .gather_helpers import get_gir_output_shape
from .gather_compile import compile_to_gir


_CONST_HANDLERS_REGISTERED: bool = False


def _ensure_constant_folders_registered(ctx: "IRContext") -> None:
    global _CONST_HANDLERS_REGISTERED
    if _CONST_HANDLERS_REGISTERED:
        return

    register = getattr(ctx, "register_constant_evaluator", None)
    if not callable(register):
        _CONST_HANDLERS_REGISTERED = True
        return

    from jax import lax

    def _bind(primitive: Any) -> Callable[..., Any]:
        return lambda *args, **kwargs: primitive.bind(*args, **kwargs)

    primitive_names = [
        "broadcast_in_dim_p",
        "reshape_p",
        "squeeze_p",
        "dynamic_slice_p",
        "slice_p",
        "concatenate_p",
        "add_p",
        "sub_p",
        "mul_p",
        "min_p",
        "max_p",
        "transpose_p",
        "rev_p",
        "broadcasted_iota_p",
        "convert_element_type_p",
    ]

    for name in primitive_names:
        primitive = getattr(lax, name, None)
        if primitive is None:
            continue
        try:
            register(primitive, _bind(primitive))
        except Exception:
            continue

    _CONST_HANDLERS_REGISTERED = True


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
        mode = eqn.params.get("mode", lax.GatherScatterMode.PROMISE_IN_BOUNDS)
        allowed_modes = {
            lax.GatherScatterMode.PROMISE_IN_BOUNDS,
            lax.GatherScatterMode.FILL_OR_DROP,
            lax.GatherScatterMode.CLIP,
        }
        if mode is not None and mode not in allowed_modes:
            raise NotImplementedError(
                "gather lowering currently supports modes "
                f"{sorted(m.name for m in allowed_modes)}; got {mode!r}"
            )
        _ensure_constant_folders_registered(ctx)
        constant_indices_value = ctx.try_evaluate_const(indices_var)

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
        new_dims = gir_instr.get("new_dims", [])
        if not new_dims:
            return index_tensor

        input_shape_descr = list(gir_instr.get("input_shape", []))
        if not input_shape_descr:
            raise RuntimeError("Missing input_shape metadata for index_expand")

        base_shape_descr = input_shape_descr[:-1]
        index_dim_descr = input_shape_descr[-1]
        if not isinstance(index_dim_descr, (int, np.integer)):
            raise NotImplementedError(
                "Dynamic gather index vector length is not supported in index_expand"
            )
        index_dim_value = int(index_dim_descr)

        new_dims_shape = [int(dim["slice_size"]) for dim in new_dims]
        indices_positions = [int(dim["indices_var_index"]) for dim in new_dims]

        base_rank = len(base_shape_descr)
        num_new_dims = len(new_dims_shape)

        dtype_enum = _dtype_enum_from_value(index_tensor)

        # Insert unit dimensions before the index component.
        current_shape_descr = list(input_shape_descr)
        current_val = index_tensor
        for offset in range(num_new_dims):
            axis = base_rank + offset
            axes_const = _const_i64(
                ctx,
                np.asarray([axis], dtype=np.int64),
                f"index_expand_unsq_axes_{offset}",
            )
            current_val = ctx.builder.Unsqueeze(
                current_val,
                axes_const,
                _outputs=[ctx.fresh_name("index_expand_unsqueeze")],
            )
            current_shape_descr.insert(axis, 1)
            _stamp_type_and_shape(current_val, tuple(current_shape_descr))
            current_val.type = ir.TensorType(dtype_enum)
            _ensure_value_metadata(ctx, current_val)

        # Prepare shape helpers.
        input_shape_val = _shape_of(ctx, index_tensor, "index_expand_input_shape")
        _stamp_type_and_shape(input_shape_val, (len(input_shape_descr),))
        input_shape_val.type = ir.TensorType(ir.DataType.INT64)
        _ensure_value_metadata(ctx, input_shape_val)

        if base_rank > 0:
            base_slice_starts = _const_i64(
                ctx, np.asarray([0], dtype=np.int64), "index_expand_base_start"
            )
            base_slice_ends = _const_i64(
                ctx,
                np.asarray([base_rank], dtype=np.int64),
                "index_expand_base_end",
            )
            slice_axes = _const_i64(
                ctx, np.asarray([0], dtype=np.int64), "index_expand_base_axes"
            )
            slice_steps = _const_i64(
                ctx, np.asarray([1], dtype=np.int64), "index_expand_base_steps"
            )
            base_shape_val = ctx.builder.Slice(
                input_shape_val,
                base_slice_starts,
                base_slice_ends,
                slice_axes,
                slice_steps,
                _outputs=[ctx.fresh_name("index_expand_base_shape")],
            )
            _stamp_type_and_shape(base_shape_val, (base_rank,))
            base_shape_val.type = ir.TensorType(ir.DataType.INT64)
            _ensure_value_metadata(ctx, base_shape_val)
        else:
            base_shape_val = None

        new_dims_const_val = _const_i64(
            ctx,
            np.asarray(new_dims_shape, dtype=np.int64),
            "index_expand_new_dims",
        )

        if base_shape_val is not None:
            target_no_index_shape_val = ctx.builder.Concat(
                base_shape_val,
                new_dims_const_val,
                axis=0,
                _outputs=[ctx.fresh_name("index_expand_target_shape")],
            )
        else:
            target_no_index_shape_val = new_dims_const_val
        target_no_index_descr = list(base_shape_descr) + new_dims_shape
        _stamp_type_and_shape(target_no_index_shape_val, (len(target_no_index_descr),))
        target_no_index_shape_val.type = ir.TensorType(ir.DataType.INT64)
        _ensure_value_metadata(ctx, target_no_index_shape_val)

        index_dim_const = _const_i64(
            ctx,
            np.asarray([index_dim_value], dtype=np.int64),
            "index_expand_index_dim",
        )
        target_full_shape_val = ctx.builder.Concat(
            target_no_index_shape_val,
            index_dim_const,
            axis=0,
            _outputs=[ctx.fresh_name("index_expand_full_shape")],
        )
        target_full_descr = target_no_index_descr + [index_dim_descr]
        _stamp_type_and_shape(target_full_shape_val, (len(target_full_descr),))
        target_full_shape_val.type = ir.TensorType(ir.DataType.INT64)
        _ensure_value_metadata(ctx, target_full_shape_val)

        total_add: ir.Value | None = None

        for dim_idx, (slice_size, coord_position) in enumerate(
            zip(new_dims_shape, indices_positions)
        ):
            start_val = _scalar_i64(ctx, 0, f"index_expand_range_start_{dim_idx}")
            limit_val = _scalar_i64(
                ctx, slice_size, f"index_expand_range_limit_{dim_idx}"
            )
            delta_val = _scalar_i64(ctx, 1, f"index_expand_range_delta_{dim_idx}")
            range_val = ctx.builder.Range(
                start_val,
                limit_val,
                delta_val,
                _outputs=[ctx.fresh_name("index_expand_range")],
            )
            _stamp_type_and_shape(range_val, (slice_size,))
            range_val.type = ir.TensorType(ir.DataType.INT64)
            _ensure_value_metadata(ctx, range_val)

            reshape_shape = (
                [1] * (base_rank + dim_idx)
                + [slice_size]
                + [1] * (num_new_dims - dim_idx - 1)
            )
            reshape_shape_const = _const_i64(
                ctx,
                np.asarray(reshape_shape, dtype=np.int64),
                f"index_expand_range_shape_{dim_idx}",
            )
            range_reshaped = ctx.builder.Reshape(
                range_val,
                reshape_shape_const,
                _outputs=[ctx.fresh_name("index_expand_range_reshape")],
            )
            _stamp_type_and_shape(range_reshaped, tuple(reshape_shape))
            range_reshaped.type = ir.TensorType(ir.DataType.INT64)
            _ensure_value_metadata(ctx, range_reshaped)

            range_expanded = ctx.builder.Expand(
                range_reshaped,
                target_no_index_shape_val,
                _outputs=[ctx.fresh_name("index_expand_range_expand")],
            )
            _stamp_type_and_shape(range_expanded, tuple(target_no_index_descr))
            range_expanded.type = ir.TensorType(ir.DataType.INT64)
            _ensure_value_metadata(ctx, range_expanded)

            unsqueeze_axis = len(target_no_index_descr)
            axes_last_const = _const_i64(
                ctx,
                np.asarray([unsqueeze_axis], dtype=np.int64),
                f"index_expand_range_unsq_axes_{dim_idx}",
            )
            range_unsqueezed = ctx.builder.Unsqueeze(
                range_expanded,
                axes_last_const,
                _outputs=[ctx.fresh_name("index_expand_range_unsq")],
            )
            _stamp_type_and_shape(range_unsqueezed, tuple(target_no_index_descr + [1]))
            range_unsqueezed.type = ir.TensorType(ir.DataType.INT64)
            _ensure_value_metadata(ctx, range_unsqueezed)

            one_hot = np.zeros(index_dim_value, dtype=np.int64)
            one_hot[coord_position] = 1
            one_hot_const = _const_i64(ctx, one_hot, f"index_expand_one_hot_{dim_idx}")
            one_hot_shape = [1] * len(target_no_index_descr) + [index_dim_value]
            one_hot_shape_const = _const_i64(
                ctx,
                np.asarray(one_hot_shape, dtype=np.int64),
                f"index_expand_one_hot_shape_{dim_idx}",
            )
            one_hot_reshaped = ctx.builder.Reshape(
                one_hot_const,
                one_hot_shape_const,
                _outputs=[ctx.fresh_name("index_expand_one_hot_reshape")],
            )
            _stamp_type_and_shape(one_hot_reshaped, tuple(one_hot_shape))
            one_hot_reshaped.type = ir.TensorType(ir.DataType.INT64)
            _ensure_value_metadata(ctx, one_hot_reshaped)

            range_contribution = ctx.builder.Mul(
                range_unsqueezed,
                one_hot_reshaped,
                _outputs=[ctx.fresh_name("index_expand_contribution")],
            )
            _stamp_type_and_shape(range_contribution, tuple(target_full_descr))
            range_contribution.type = ir.TensorType(ir.DataType.INT64)
            _ensure_value_metadata(ctx, range_contribution)

            if total_add is None:
                total_add = range_contribution
            else:
                total_add = ctx.builder.Add(
                    total_add,
                    range_contribution,
                    _outputs=[ctx.fresh_name("index_expand_total_add")],
                )
                _stamp_type_and_shape(total_add, tuple(target_full_descr))
                total_add.type = ir.TensorType(ir.DataType.INT64)
                _ensure_value_metadata(ctx, total_add)

        if dtype_enum != ir.DataType.INT64:
            total_add = ctx.builder.Cast(
                total_add,
                _outputs=[ctx.fresh_name("index_expand_cast")],
                to=int(dtype_enum.value),
            )
            _stamp_type_and_shape(total_add, tuple(target_full_descr))
            total_add.type = ir.TensorType(dtype_enum)
            _ensure_value_metadata(ctx, total_add)

        result_val = ctx.builder.Add(
            current_val,
            total_add,
            _outputs=[ctx.fresh_name("index_expand_apply")],
        )
        _stamp_type_and_shape(result_val, tuple(target_full_descr))
        result_val.type = ir.TensorType(dtype_enum)
        _ensure_value_metadata(ctx, result_val)
        return result_val

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
