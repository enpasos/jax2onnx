# jax2onnx/plugins/jax/lax/squeeze.py

from __future__ import annotations
from typing import TYPE_CHECKING, List

import numpy as np
import jax.numpy as jnp
from jax import lax
from jax._src.export.shape_poly import _DimExpr as DimExpr

import onnx_ir as ir
from jax2onnx.plugins._ir_shapes import (
    _ensure_value_metadata,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _const_i64(ctx: "IRContext", values, name_hint: str) -> ir.Value:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    val = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((arr.size,)),
        const_value=ir.tensor(arr),
    )
    ctx._initializers.append(val)
    return val


def _dim_const_value(dim) -> int | None:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    if isinstance(dim, DimExpr):
        try:
            text = str(dim).strip()
            if text.lstrip("-").isdigit():
                return int(text)
        except Exception:
            return None
    return None


@register_primitive(
    jaxpr_primitive=lax.squeeze_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.squeeze.html",
    onnx=[
        {
            "component": "Squeeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="squeeze",
    testcases=[
        {
            "testcase": "squeeze_single_axis",
            "callable": lambda x: lax.squeeze(x, dimensions=(0,)),
            "input_shapes": [(1, 3, 4)],
            "expected_output_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Squeeze:3x4",
                        "inputs": {1: {"const": 0.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "squeeze_all_unit_dims_default",
            "callable": lambda x: jnp.squeeze(x),
            "input_shapes": [(1, 3, 1, 4, 1)],
            "expected_output_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["Squeeze:3x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "lax_squeeze_specific_axis_0",
            "callable": lambda x: lax.squeeze(x, dimensions=(0,)),
            "input_shapes": [(1, 3)],
            "expected_output_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                [{"path": "Squeeze:3", "inputs": {1: {"const": 0.0}}}],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "lax_squeeze_multiple_axes",
            "callable": lambda x: lax.squeeze(x, dimensions=(0, 2, 4)),
            "input_shapes": [(1, 3, 1, 4, 1)],
            "expected_output_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["Squeeze:3x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "lax_squeeze_no_op_empty_dims",
            "callable": lambda x: lax.squeeze(x, dimensions=()),
            "input_shapes": [(1, 3, 1)],
            "expected_output_shapes": [(1, 3, 1)],
            "post_check_onnx_graph": EG(
                [],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "lax_squeeze_problem_case_input_squeeze_only_axis_0",
            "callable": lambda x: lax.squeeze(x, dimensions=(0,)),
            "input_shapes": [(1, 201, 1, 1)],
            "expected_output_shapes": [(201, 1, 1)],
            "post_check_onnx_graph": EG(
                [{"path": "Squeeze:201x1x1", "inputs": {1: {"const": 0.0}}}],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "lax_squeeze_problem_case_input_squeeze_axes_0_2",
            "callable": lambda x: lax.squeeze(x, dimensions=(0, 2)),
            "input_shapes": [(1, 201, 1, 1)],
            "expected_output_shapes": [(201, 1)],
            "post_check_onnx_graph": EG(
                ["Squeeze:201x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "lax_squeeze_problem_case_input_squeeze_all_dims_explicitly",
            "callable": lambda x: lax.squeeze(x, dimensions=(0, 2, 3)),
            "input_shapes": [(1, 201, 1, 1)],
            "expected_output_shapes": [(201,)],
            "post_check_onnx_graph": EG(
                ["Squeeze:201"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SqueezePlugin(PrimitiveLeafPlugin):
    """plugins IR converter for jax.lax.squeeze → ONNX Squeeze."""

    def lower(self, ctx: "IRContext", eqn):
        x_var = eqn.invars[0]
        y_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("squeeze_in"))
        out_spec = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("squeeze_out"))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()) or ())
        rank = len(x_shape)

        dims_param = eqn.params.get("dimensions")
        axes: List[int]
        if dims_param is None:
            axes = [
                idx for idx, dim in enumerate(x_shape) if _dim_const_value(dim) == 1
            ]
        else:
            axes = []
            for dim in dims_param:
                axis = int(dim)
                if axis < 0:
                    axis += rank
                if axis < 0 or axis >= rank:
                    raise ValueError(
                        f"Squeeze axis {dim} out of bounds for rank {rank}"
                    )
                axes.append(axis)

        # Canonicalize and preserve deterministic ordering for ONNX.
        axes = sorted(set(axes))

        axes_val = _const_i64(ctx, np.asarray(axes, dtype=np.int64), "squeeze_axes")

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("squeeze_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("squeeze_out")

        result = ctx.builder.Squeeze(
            x_val,
            axes_val,
            _outputs=[desired_name],
        )

        if getattr(x_val, "type", None) and isinstance(x_val.type, ir.TensorType):
            result.type = ir.TensorType(x_val.type.dtype)

        if x_shape:
            out_dims = [d for i, d in enumerate(x_shape) if i not in axes]
            _stamp_type_and_shape(
                result, tuple(_to_ir_dim_for_shape(d) for d in out_dims)
            )
        else:
            _stamp_type_and_shape(
                result, tuple(getattr(getattr(y_var, "aval", None), "shape", ()))
            )

        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(y_var, result)
