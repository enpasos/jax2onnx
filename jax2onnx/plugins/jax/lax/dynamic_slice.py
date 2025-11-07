# jax2onnx/plugins/jax/lax/dynamic_slice.py

from typing import Any, Dict

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import (
    LoweringContextProtocol,
    SymbolicDimOrigin,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._axis0_utils import _axis0_debug
from jax2onnx.plugins._loop_extent_meta import (
    get_axis0_override,
    propagate_axis0_override,
    set_axis0_override,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.lax._index_utils import (
    _const_i64,
    _cast_to_i64,
    _infer_rank,
)


@register_primitive(
    jaxpr_primitive=jax.lax.dynamic_slice_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html",
    onnx=[
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        }
    ],
    since="v0.1.0",
    context="primitives.lax",
    component="dynamic_slice",
    testcases=[
        {
            "testcase": "dynamic_slice_test1",
            "callable": lambda x: jax.lax.dynamic_slice(x, [1], [2]),
            "input_shapes": [(5,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Slice:2",
                        "inputs": {3: {"const": 0.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dynamic_slice_2d",
            "callable": lambda x: jax.lax.dynamic_slice(x, (1, 2), (2, 3)),
            "input_shapes": [(4, 6)],
            "post_check_onnx_graph": EG(
                ["Slice:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dynamic_slice_3d",
            "callable": lambda x: jax.lax.dynamic_slice(x, (1, 0, 2), (2, 3, 1)),
            "input_shapes": [(3, 4, 5)],
            "post_check_onnx_graph": EG(
                ["Slice:2x3x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dynamic_slice_vit_like",
            "context": "jax.lax.dynamic_slice",
            "callable": lambda x: jax.lax.dynamic_slice(
                x, (0, 0, 0), (x.shape[0], 1, 256)
            ),
            "input_shapes": [("B", 50, 256)],
            "expected_output_shapes": [("B", 1, 256)],
            "post_check_onnx_graph": EG(
                ["Slice:Bx1x256"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class DynamicSlicePlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn):
        operand_var = eqn.invars[0]
        start_vars = eqn.invars[1:]
        out_var = eqn.outvars[0]

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("dyn_slice_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("dyn_slice_out")
        )

        origin_getter = getattr(ctx, "get_symbolic_dim_origin", None)
        shape_cache: Dict[ir.Value, ir.Value] = {}

        def _shape_vec_for(src: ir.Value, axis: int) -> ir.Value:
            shape_val = shape_cache.get(src)
            if shape_val is not None:
                return shape_val
            rank = _infer_rank(src, axis)
            shape_val = ctx.builder.Shape(
                src, _outputs=[ctx.fresh_name("dyn_slice_shape")]
            )
            shape_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(shape_val, (rank,))
            _ensure_value_metadata(ctx, shape_val)
            shape_cache[src] = shape_val
            return shape_val

        def _symbolic_dim_vector(dim_expr: Any, idx: int) -> ir.Value:
            if origin_getter is None:
                raise ValueError(
                    f"Symbolic dimension '{dim_expr}' encountered without origin resolver"
                )
            origin = SymbolicDimOrigin.resolve(origin_getter, dim_expr)
            if origin is None:
                raise ValueError(
                    f"Symbolic dimension '{dim_expr}' has no registered origin"
                )
            axis = int(origin.axis)
            shape_vec = _shape_vec_for(origin.value, axis)
            gather_idx = _const_i64(ctx, [axis], f"dyn_slice_size_axis_{idx}")
            gathered = ctx.builder.Gather(
                shape_vec,
                gather_idx,
                axis=0,
                _outputs=[ctx.fresh_name("dyn_slice_size")],
            )
            gathered.type = ir.TensorType(ir.DataType.INT64)
            return gathered

        rank = len(getattr(operand_var.aval, "shape", ()))
        if rank != len(start_vars):
            raise ValueError(
                f"dynamic_slice expected {rank} start indices but got {len(start_vars)}"
            )

        starts_vec_parts = []
        axes_const = _const_i64(ctx, [0], "dyn_slice_unsq_axis")

        for start_var in start_vars:
            start_val = ctx.get_value_for_var(
                start_var, name_hint=ctx.fresh_name("dyn_slice_start")
            )
            cast_scalar = _cast_to_i64(ctx, start_val, "dyn_slice_start_i64")

            unsqueezed = ctx.builder.Unsqueeze(
                cast_scalar,
                axes_const,
                _outputs=[ctx.fresh_name("dyn_slice_unsq")],
            )
            unsqueezed.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(unsqueezed, (1,))
            _ensure_value_metadata(ctx, unsqueezed)
            starts_vec_parts.append(unsqueezed)

        starts_concat = ctx.builder.Concat(
            *starts_vec_parts,
            axis=0,
            _outputs=[ctx.fresh_name("dyn_slice_starts")],
        )
        starts_concat.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(starts_concat, (rank,))
        _ensure_value_metadata(ctx, starts_concat)

        slice_sizes = eqn.params.get("slice_sizes", ())
        try:
            const_sizes = [int(v) for v in slice_sizes]
            slice_sizes_val = _const_i64(ctx, const_sizes, "dyn_slice_sizes")
        except Exception:
            size_vectors = []
            all_const = True
            for idx, val in enumerate(slice_sizes):
                try:
                    const_val = int(val)
                except Exception:
                    const_val = None
                if const_val is not None:
                    size_vectors.append(
                        _const_i64(ctx, [const_val], f"dyn_slice_size_const_{idx}")
                    )
                else:
                    all_const = False
                    size_vectors.append(_symbolic_dim_vector(val, idx))
            if all_const:
                slice_sizes_val = _const_i64(
                    ctx,
                    [int(v) for v in slice_sizes],
                    "dyn_slice_sizes",
                )
            else:
                if len(size_vectors) == 1 and len(slice_sizes) == 1:
                    slice_sizes_val = size_vectors[0]
                else:
                    slice_sizes_val = ctx.builder.Concat(
                        *size_vectors,
                        axis=0,
                        _outputs=[ctx.fresh_name("dyn_slice_sizes")],
                    )
                    slice_sizes_val.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(slice_sizes_val, (len(slice_sizes),))
                    _ensure_value_metadata(ctx, slice_sizes_val)

        ends_val = ctx.builder.Add(
            starts_concat,
            slice_sizes_val,
            _outputs=[ctx.fresh_name("dyn_slice_ends")],
        )
        ends_val.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(ends_val, (rank,))
        _ensure_value_metadata(ctx, ends_val)

        axes_val = _const_i64(ctx, list(range(rank)), "dyn_slice_axes")

        inputs = [operand_val, starts_concat, ends_val, axes_val]
        strides = eqn.params.get("strides")
        if strides:
            inputs.append(_const_i64(ctx, strides, "dyn_slice_strides"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Slice")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Slice")

        result = ctx.builder.Slice(
            *inputs,
            _outputs=[desired_name],
        )

        output_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(result, output_shape)
        result_dtype = getattr(getattr(operand_val, "type", None), "dtype", None)
        if result_dtype is None:
            result_dtype = getattr(getattr(out_spec, "type", None), "dtype", None)
        if result_dtype is not None:
            result.type = ir.TensorType(result_dtype)
        _ensure_value_metadata(ctx, result)

        x_override = get_axis0_override(operand_val)
        spec_override = get_axis0_override(out_spec)
        ctx_override = getattr(ctx, "_static_loop_extent_axis0", None)
        override_sources = (x_override, spec_override, ctx_override)
        _axis0_debug(
            "dynamic_slice override sources "
            f"value={getattr(result, 'name', None)} "
            f"sources={override_sources} "
            f"x={getattr(operand_val, 'name', None)} "
            f"spec={getattr(out_spec, 'name', None)}"
        )
        override_candidates = [
            int(candidate)
            for candidate in override_sources
            if isinstance(candidate, (int, np.integer)) and int(candidate) > 1
        ]
        _axis0_debug(
            "dynamic_slice override candidates "
            f"value={getattr(result, 'name', None)} "
            f"candidates={override_candidates}"
        )
        axis0_override = max(override_candidates, default=None)
        propagate_axis0_override(operand_val, result)
        if axis0_override is not None:
            set_axis0_override(result, axis0_override)
        ctx.bind_value_for_var(out_var, result)
        ctx.bind_value_for_var(out_var, result)
