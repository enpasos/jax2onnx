# jax2onnx/plugins/jax/lax/add_any.py

from __future__ import annotations

import jax

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.add import AddPlugin, lower_add
from jax2onnx.plugins.plugin_system import register_primitive


@register_primitive(
    jaxpr_primitive="add_any",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.add.html",
    onnx=[{"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"}],
    since="v0.8.0",
    context="primitives.lax",
    component="add_any",
    testcases=[
        {
            "testcase": "add_any_via_jvp_on_mul",
            "callable": lambda x1, x2: jax.jvp(lambda a, b: a * b, (x1, x2), (x1, x2))[
                1
            ],
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Mul:3 -> Add:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class AddAnyPlugin(AddPlugin):
    """Alias for JAX's internal ``add_any`` primitive."""

    def lower(self, ctx, eqn):
        lower_add(ctx, eqn)
