# jax2onnx/plugins/jax/lax/sort.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.sort_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html",
    onnx=[
        {"component": "TopK", "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html"}
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
    ],
)
class SortPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        params = getattr(eqn, "params", {})
        axis = int(params.get("dimension", -1))
        num_keys = int(params.get("num_keys", 1))

        invars = list(eqn.invars)
        outvars = list(eqn.outvars)
        if not invars:
            raise ValueError("lax.sort expects at least one operand")
        if len(invars) != len(outvars):
            raise ValueError("lax.sort expects the same number of inputs and outputs")
        if num_keys != 1:
            raise NotImplementedError("lax.sort with num_keys > 1 is not supported yet")

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
        values, indices = ctx.builder.TopK(
            key_val,
            k_val,
            _outputs=[
                ctx.fresh_name("sort_values"),
                ctx.fresh_name("sort_indices"),
            ],
            axis=int(axis),
            largest=0,
            sorted=1,
        )
        key_dtype = getattr(getattr(key_val, "type", None), "dtype", None)
        if key_dtype is not None:
            values.type = ir.TensorType(key_dtype)
            values.dtype = key_dtype

        out_shape = tuple(getattr(outvars[0].aval, "shape", ()))
        _stamp_type_and_shape(values, out_shape)
        _ensure_value_metadata(ctx, values)

        indices.type = ir.TensorType(ir.DataType.INT64)
        indices.dtype = ir.DataType.INT64
        _stamp_type_and_shape(indices, out_shape)
        _ensure_value_metadata(ctx, indices)

        for idx, (in_var, out_var, out_spec) in enumerate(
            zip(invars, outvars, out_specs, strict=True)
        ):
            result_name = getattr(out_spec, "name", None) or ctx.fresh_name("sort_out")
            if idx == 0:
                result = ctx.builder.Identity(values, _outputs=[result_name])
                if key_dtype is not None:
                    result.type = ir.TensorType(key_dtype)
                    result.dtype = key_dtype
                _stamp_type_and_shape(result, out_shape)
                _ensure_value_metadata(ctx, result)
                ctx.bind_value_for_var(out_var, result)
                continue

            in_val = ctx.get_value_for_var(in_var, name_hint=ctx.fresh_name("sort_in"))
            gathered = ctx.builder.GatherElements(
                in_val,
                indices,
                axis=int(axis),
                _outputs=[result_name],
            )
            in_dtype = getattr(getattr(in_val, "type", None), "dtype", None)
            if in_dtype is not None:
                gathered.type = ir.TensorType(in_dtype)
                gathered.dtype = in_dtype
            output_shape = tuple(getattr(out_var.aval, "shape", ()))
            _stamp_type_and_shape(gathered, output_shape)
            _ensure_value_metadata(ctx, gathered)
            ctx.bind_value_for_var(out_var, gathered)
