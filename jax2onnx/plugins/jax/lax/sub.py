# jax2onnx/plugins/jax/lax/sub.py

from typing import TYPE_CHECKING, Optional
import jax
import numpy as np
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass  # hints


@register_primitive(
    jaxpr_primitive=jax.lax.sub_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sub.html",
    onnx=[{"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"}],
    since="v0.1.0",
    context="primitives.lax",
    component="sub",
    testcases=[
        {
            "testcase": "sub_test1",
            "callable": lambda x1, x2: x1 - x2,
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Sub:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "sub_test2",
            "callable": lambda x1, x2: jax.lax.sub(x1, x2),
            "input_shapes": [(2, 2), (2, 2)],
            "post_check_onnx_graph": EG(
                ["Sub:2x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "sub_const",
            "callable": lambda x: x - 1.0,
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Sub:3",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SubPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var, y_var = eqn.invars
        out_var = eqn.outvars[0]

        prefer_dt: Optional[np.dtype] = np.dtype(
            getattr(x_var.aval, "dtype", np.float32)
        )
        a_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("sub_lhs"))
        b_val = ctx.get_value_for_var(
            y_var, name_hint=ctx.fresh_name("sub_rhs"), prefer_np_dtype=prefer_dt
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("sub_out"))
        result = ctx.builder.Sub(a_val, b_val, _outputs=[out_spec.name])
        result.type = out_spec.type
        result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
