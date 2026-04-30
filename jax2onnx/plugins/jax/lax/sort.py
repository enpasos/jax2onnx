# jax2onnx/plugins/jax/lax/sort.py

from __future__ import annotations

from typing import Any, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive=jax.lax.sort_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html",
    onnx=[
        {
            "component": "GatherElements",
            "doc": "https://onnx.ai/onnx/operators/onnx__GatherElements.html",
        },
        {"component": "TopK", "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html"},
    ],
    since="0.2.0",
    context="primitives.lax",
    component="sort",
    testcases=[
        {
            "testcase": "sort_1d",
            "callable": lambda x: jax.lax.sort(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["TopK:3 -> Identity:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "sort_2d",
            "callable": lambda x: jax.lax.sort(x, dimension=0),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["TopK:3x4 -> Identity:3x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "sort_two_keys",
            "callable": lambda primary, secondary: jax.lax.sort(
                (primary, secondary), dimension=0, num_keys=2, is_stable=True
            ),
            "input_values": [
                np.asarray([1, 0, 1, 0], dtype=np.int32),
                np.asarray([3, 2, 1, 0], dtype=np.int32),
            ],
            "expected_output_dtypes": [np.int32, np.int32],
            "post_check_onnx_graph": EG(
                [("TopK", {"counts": {"TopK": 2, "GatherElements": 2}})],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SortPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        params = getattr(eqn, "params", {})
        axis = int(params.get("dimension", -1))
        num_keys = int(params.get("num_keys", 1))

        invars = list(eqn.invars)
        outvars = list(eqn.outvars)
        if not invars:
            raise ValueError("lax.sort expects at least one operand")
        if len(invars) != len(outvars):
            raise ValueError("lax.sort expects the same number of inputs and outputs")
        if num_keys < 1 or num_keys > len(invars):
            raise ValueError("lax.sort num_keys must fit the operand count")

        key_var = invars[0]
        key_shape = tuple(getattr(key_var.aval, "shape", ()))
        if not key_shape:
            axis = 0
        else:
            if axis < 0:
                axis += len(key_shape)
            if axis < 0 or axis >= len(key_shape):
                raise ValueError("sort axis out of range")

        key_val = ctx.get_value_for_var(key_var, name_hint=ctx.fresh_name("sort_key"))
        out_specs = [
            ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("sort_out"))
            for out_var in outvars
        ]

        axis_size = key_shape[axis] if key_shape else 1
        if not isinstance(axis_size, (int, np.integer)):
            raise TypeError("lax.sort currently requires static axis length")

        k_val = _const_i64(ctx, np.asarray([axis_size], dtype=np.int64), "sort_k")
        current_values = [
            ctx.get_value_for_var(in_var, name_hint=ctx.fresh_name("sort_in"))
            for in_var in invars
        ]
        current_values[0] = key_val

        for key_idx in reversed(range(num_keys)):
            key_var = invars[key_idx]
            key_value = current_values[key_idx]
            key_shape = tuple(getattr(key_var.aval, "shape", ()))
            key_dtype = getattr(getattr(key_value, "type", None), "dtype", None)
            key_is_bool = False
            key_aval_dtype = getattr(getattr(key_var, "aval", None), "dtype", None)
            if key_aval_dtype is not None:
                try:
                    key_is_bool = np.dtype(key_aval_dtype) == np.bool_
                except TypeError:
                    key_is_bool = False

            key_input = key_value
            if key_is_bool:
                key_input = cast(
                    ir.Value,
                    ctx.builder.Cast(
                        key_value,
                        to=ir.DataType.INT32,
                        _outputs=[ctx.fresh_name("sort_key_i32")],
                    ),
                )
                key_input.type = ir.TensorType(ir.DataType.INT32)
                key_input.dtype = ir.DataType.INT32
                _stamp_type_and_shape(key_input, key_shape)
                _ensure_value_metadata(ctx, key_input)

            values, indices = cast(
                tuple[ir.Value, ir.Value],
                ctx.builder.TopK(
                    key_input,
                    k_val,
                    _outputs=[
                        ctx.fresh_name("sort_values"),
                        ctx.fresh_name("sort_indices"),
                    ],
                    axis=int(axis),
                    largest=0,
                    sorted=1,
                ),
            )

            if key_is_bool:
                values = cast(
                    ir.Value,
                    ctx.builder.Cast(
                        values,
                        to=ir.DataType.BOOL,
                        _outputs=[ctx.fresh_name("sort_values_bool")],
                    ),
                )
                values.type = ir.TensorType(ir.DataType.BOOL)
                values.dtype = ir.DataType.BOOL
            elif key_dtype is not None:
                values.type = ir.TensorType(key_dtype)
                values.dtype = key_dtype

            _stamp_type_and_shape(values, key_shape)
            _ensure_value_metadata(ctx, values)
            current_values[key_idx] = values

            indices.type = ir.TensorType(ir.DataType.INT64)
            indices.dtype = ir.DataType.INT64
            _stamp_type_and_shape(indices, key_shape)
            _ensure_value_metadata(ctx, indices)

            for value_idx, value in enumerate(current_values):
                if value_idx == key_idx:
                    continue

                gathered = cast(
                    ir.Value,
                    ctx.builder.GatherElements(
                        value,
                        indices,
                        axis=int(axis),
                        _outputs=[ctx.fresh_name("sort_gathered")],
                    ),
                )
                value_dtype = getattr(getattr(value, "type", None), "dtype", None)
                if value_dtype is not None:
                    gathered.type = ir.TensorType(value_dtype)
                    gathered.dtype = value_dtype
                value_shape = tuple(getattr(invars[value_idx].aval, "shape", ()))
                _stamp_type_and_shape(gathered, value_shape)
                _ensure_value_metadata(ctx, gathered)
                current_values[value_idx] = gathered

        for out_var, out_spec, value in zip(
            outvars, out_specs, current_values, strict=True
        ):
            result_name = getattr(out_spec, "name", None) or ctx.fresh_name("sort_out")
            result = cast(ir.Value, ctx.builder.Identity(value, _outputs=[result_name]))
            value_dtype = getattr(getattr(value, "type", None), "dtype", None)
            if value_dtype is not None:
                result.type = ir.TensorType(value_dtype)
                result.dtype = value_dtype
            output_shape = tuple(getattr(out_var.aval, "shape", ()))
            _stamp_type_and_shape(result, output_shape)
            _ensure_value_metadata(ctx, result)
            ctx.bind_value_for_var(out_var, result)
