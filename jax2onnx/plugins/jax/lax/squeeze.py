# jax2onnx/plugins/jax/lax/squeeze.py

from __future__ import annotations
from typing import TYPE_CHECKING, List

import numpy as np
import jax.numpy as jnp
from jax import lax

import onnx_ir as ir
from jax2onnx.plugins._axis0_utils import ensure_axis0_extent, _axis0_debug
from jax2onnx.plugins._ir_shapes import (
    _ensure_value_metadata,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)
from jax2onnx.plugins._loop_extent_meta import (
    get_axis0_override,
    propagate_axis0_override,
    set_axis0_override,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.utils.shape_poly import dim_expr_constant_value

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _const_i64(ctx: "IRContext", values, name_hint: str) -> ir.Value:
    """Emit an INT64 constant via the builder to centralize initializer policy."""
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    name = ctx.fresh_name(name_hint)
    # Route through builder so function-mode and duplicate policies apply.
    return ctx.builder.const_i64(name, arr.tolist())


def _dim_const_value(dim) -> int | None:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    return dim_expr_constant_value(dim)


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

        x_override = get_axis0_override(x_val)
        spec_override = get_axis0_override(out_spec)
        ctx_override = getattr(ctx, "_static_loop_extent_axis0", None)
        override_sources = (x_override, spec_override, ctx_override)
        _axis0_debug(
            "squeeze override sources "
            f"value={getattr(result, 'name', None)} "
            f"sources={override_sources} "
            f"x={getattr(x_val, 'name', None)} "
            f"spec={getattr(out_spec, 'name', None)}"
        )
        override_candidates = [
            int(candidate)
            for candidate in override_sources
            if isinstance(candidate, (int, np.integer)) and int(candidate) > 1
        ]
        _axis0_debug(
            "squeeze override candidates "
            f"value={getattr(result, 'name', None)} "
            f"candidates={override_candidates}"
        )
        axis0_override = max(override_candidates, default=None)
        axis0_removed = 0 in axes
        _axis0_debug(
            "squeeze axis0 state "
            f"value={getattr(result, 'name', None)} "
            f"override={axis0_override} removed={axis0_removed}"
        )
        if not axis0_removed:
            result = ensure_axis0_extent(ctx, result, axis0_override, reference=x_val)

        if x_shape:
            out_dims = [d for i, d in enumerate(x_shape) if i not in axes]
            if axis0_override is not None and out_dims and not axis0_removed:
                out_dims = list(out_dims)
                out_dims[0] = axis0_override
            _stamp_type_and_shape(
                result, tuple(_to_ir_dim_for_shape(d) for d in out_dims)
            )
        else:
            target_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()))
            if axis0_override is not None and target_shape and not axis0_removed:
                target_shape = (axis0_override,) + target_shape[1:]
            _stamp_type_and_shape(result, target_shape)

        _ensure_value_metadata(ctx, result)
        if axis0_removed and axis0_override is not None:
            set_axis0_override(out_spec, axis0_override)
        elif not axis0_removed:
            propagate_axis0_override(x_val, result)
        ctx.bind_value_for_var(y_var, result)
