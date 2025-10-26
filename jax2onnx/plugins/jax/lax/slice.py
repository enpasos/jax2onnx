# jax2onnx/plugins/jax/lax/slice.py

from typing import TYPE_CHECKING

import numpy as np
import jax
import onnx_ir as ir

from jax2onnx.plugins._axis0_utils import ensure_axis0_extent, _axis0_debug
from jax2onnx.plugins._loop_extent_meta import (
    get_axis0_override,
    propagate_axis0_override,
    set_axis0_override,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.lax._index_utils import _const_i64

if TYPE_CHECKING:
    pass


@register_primitive(
    jaxpr_primitive=jax.lax.slice_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.slice.html",
    onnx=[
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        }
    ],
    since="v0.1.0",
    context="primitives.lax",
    component="slice",
    testcases=[
        {
            "testcase": "slice_test1",
            "callable": lambda x: x[1:3],
            "input_shapes": [(5,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            1: {"const": 1.0},
                            2: {"const": 3.0},
                            3: {"const": 0.0},
                        },
                        "path": "Slice:2",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "slice_3d_none_strides",
            "callable": lambda a: a[0:2, 0:1, 0:256],
            "input_shapes": [(2, 50, 256)],
            "post_check_onnx_graph": EG(
                ["Slice:2x1x256"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "slice_scan_axis_drop",
            "callable": lambda x: (
                jax.lax.scan(
                    lambda c, xt: (
                        c,
                        jax.numpy.squeeze(xt[None, ...][0:1, :, :, :], axis=0),
                    ),
                    jax.numpy.zeros(x.shape[1:], dtype=x.dtype),
                    x,
                )[1]
            ),
            "input_shapes": [(2, 3, 4, 5)],
            "post_check_onnx_graph": EG(
                ["Loop"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SlicePlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("slice_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("slice_out"))

        params = eqn.params
        starts = params.get("start_indices", ())
        limits = params.get("limit_indices", ())
        strides = params.get("strides", ()) or None

        axes = tuple(range(len(starts)))

        starts_val = _const_i64(ctx, starts, "slice_starts")

        def _coerce_limit(val):
            if isinstance(val, (int, np.integer)):
                return int(val)
            return np.iinfo(np.int64).max

        normalized_limits = tuple(_coerce_limit(v) for v in limits)
        limits_val = _const_i64(ctx, normalized_limits, "slice_limits")
        axes_val = _const_i64(ctx, axes, "slice_axes")

        inputs = [x_val, starts_val, limits_val, axes_val]
        if strides:
            steps_val = _const_i64(ctx, strides, "slice_steps")
            inputs.append(steps_val)
        dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        out_name = getattr(out_val, "name", None) or ctx.fresh_name("slice_out")
        out_tensor = ctx.builder.Slice(*inputs, _outputs=[out_name])
        if dtype is not None:
            out_tensor.type = ir.TensorType(dtype)
        _ensure_value_metadata(ctx, out_tensor)

        target_shape = tuple(getattr(out_var.aval, "shape", ()))
        axis0_extent = None
        if target_shape:
            dim0 = target_shape[0]
            if isinstance(dim0, (int, np.integer)):
                axis0_extent = int(dim0)
        x_override = get_axis0_override(x_val)
        spec_override = get_axis0_override(out_val)
        ctx_override = getattr(ctx, "_static_loop_extent_axis0", None)
        override_sources = (x_override, spec_override, ctx_override)
        _axis0_debug(
            "slice override sources "
            f"value={getattr(out_tensor, 'name', None)} "
            f"sources={override_sources} "
            f"x={getattr(x_val, 'name', None)} "
            f"spec={getattr(out_val, 'name', None)}"
        )
        override_candidates = [
            int(candidate)
            for candidate in override_sources
            if isinstance(candidate, (int, np.integer)) and int(candidate) > 1
        ]
        _axis0_debug(
            "slice override candidates "
            f"value={getattr(out_tensor, 'name', None)} "
            f"candidates={override_candidates}"
        )
        axis0_override = max(override_candidates, default=None)
        if (
            axis0_override is not None
            and axis0_extent is not None
            and isinstance(axis0_extent, (int, np.integer))
            and axis0_extent > 1
            and axis0_override < axis0_extent
        ):
            axis0_override = int(axis0_extent)
        need_expand = axis0_override is not None and (
            axis0_extent is None or axis0_extent > 1
        )
        if target_shape:
            _stamp_type_and_shape(out_tensor, target_shape)
        if axis0_override is not None and target_shape and need_expand:
            expanded_target = (axis0_override,) + target_shape[1:]
        else:
            expanded_target = target_shape
        if need_expand:
            out_tensor = ensure_axis0_extent(
                ctx, out_tensor, axis0_override, reference=x_val
            )
        if expanded_target:
            _stamp_type_and_shape(out_tensor, expanded_target)
        propagate_axis0_override(x_val, out_tensor)
        if axis0_override is not None:
            set_axis0_override(out_tensor, axis0_override)
        ctx.bind_value_for_var(out_var, out_tensor)
