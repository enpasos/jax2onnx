from typing import TYPE_CHECKING

import jax
import onnx_ir as ir

from jax2onnx.plugins2._ir_shapes import _stamp_type_and_shape
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2.jax.lax._index_utils import _const_i64

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
    context="primitives2.lax",
    component="slice",
    testcases=[
        {
            "testcase": "slice_test1",
            "callable": lambda x: x[1:3],
            "input_shapes": [(5,)],
        },
        {
            "testcase": "slice_3d_none_strides",
            "callable": lambda a: a[0:2, 0:1, 0:256],
            "input_shapes": [(2, 50, 256)],
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
        limits_val = _const_i64(ctx, limits, "slice_limits")
        axes_val = _const_i64(ctx, axes, "slice_axes")

        inputs = [x_val, starts_val, limits_val, axes_val]
        if strides:
            steps_val = _const_i64(ctx, strides, "slice_steps")
            inputs.append(steps_val)

        node = ir.Node(
            op_type="Slice",
            domain="",
            inputs=inputs,
            outputs=[out_val],
            name=ctx.fresh_name("Slice"),
        )
        ctx.add_node(node)

        _stamp_type_and_shape(out_val, getattr(out_var.aval, "shape", ()))
