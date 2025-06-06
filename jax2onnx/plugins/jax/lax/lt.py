from typing import TYPE_CHECKING

import jax

from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.lt_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.lt.html",
    onnx=[
        {
            "component": "Less",
            "doc": "https://onnx.ai/onnx/operators/onnx__Less.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="lt",
    testcases=[
        {
            "testcase": "lt",
            "callable": lambda x1, x2: x1 < x2,
            "input_shapes": [(3,), (3,)],
            # --- FIX: Use shape tuple only ---
            # "expected_output_shapes": [(3,)],
        },
        # {
        #     "testcase": "lt_int32_int64",
        #     "callable": lambda x1, x2: x1 < x2,
        #     "input_shapes": [((), np.int32), ((), np.int64)],
        #     # --- FIX: Use shape tuple only ---
        #     "expected_output_shapes": [()],
        # },
    ],
)
class LtPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.lt to ONNX Less."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX lt primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Less",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("less"),
        )
        s.add_node(node)
