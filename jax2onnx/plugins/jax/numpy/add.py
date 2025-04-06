from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
from typing import TYPE_CHECKING
from jax2onnx.plugin_system import register_primitive, PrimitiveLeafPlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the Add primitive
jnp.add_p = Primitive("jnp.add")
jnp.add_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=jnp.add_p.name,
    since="v0.1.0",
    context="primitives.jnp",
    component="add",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.add.html",
    onnx=[
        {
            "component": "Add",
            "doc": "https://onnx.ai/onnx/operators/onnx__Add.html",
        }
    ],
    testcases=[
        {
            "testcase": "add",
            "callable": lambda x, y: jnp.add(x, y),
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class AddPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.numpy.add to ONNX.
    """

    @staticmethod
    def abstract_eval(x, y):
        """Abstract evaluation function for Add."""
        return core.ShapedArray(x.shape, x.dtype)  # Should handle broadcasting

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of Add to ONNX format."""
        # Expect node_inputs: [x, y]
        x_var = node_inputs[0]
        y_var = node_inputs[1]
        output_var = node_outputs[0]

        x_name = s.get_name(x_var)
        y_name = s.get_name(y_var)
        output_name = s.get_name(output_var)

        add_node = helper.make_node(
            "Add",
            inputs=[x_name, y_name],
            outputs=[output_name],
            name=s.get_unique_name("add"),
        )
        s.add_node(add_node)

    @staticmethod
    def _add(x, y):
        """Defines the primitive binding for Add."""
        return jnp.add_p.bind(x, y)

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for Add."""

        def patched_add(x, y):
            return AddPlugin._add(x, y)

        return patched_add

    @staticmethod
    def patch_info():
        """Provides patching information for Add."""
        return {
            "patch_targets": [jnp],
            "patch_function": lambda _: AddPlugin.get_monkey_patch(),
            "target_attribute": "add",
        }


# Register abstract evaluation function
jnp.add_p.def_abstract_eval(AddPlugin.abstract_eval)
