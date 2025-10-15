# jax2onnx/plugins/jax/lax/slice.py

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape
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
        _stamp_type_and_shape(out_tensor, getattr(out_var.aval, "shape", ()))
        ctx.bind_value_for_var(out_var, out_tensor)
