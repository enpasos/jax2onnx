from typing import TYPE_CHECKING

from jax import core
from jax import numpy as jnp
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

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
        x_shape = x.shape
        y_shape = y.shape
        if len(x_shape) != len(y_shape):
            raise ValueError(
                f"Shapes of x {x_shape} and y {y_shape} must have the same number of dimensions."
            )
        z_shape = []
        for i in range(len(x_shape)):
            # Check if dimensions are compatible
            # In case of abstract dimensions, do the best you can do: be optimistic
            # if one of the dimensions is -1, it is compatible, and the other dimension is taken
            if x_shape[i] == -1 or y_shape[i] == -1:
                z_shape.append(max(x_shape[i], y_shape[i]))
            elif x_shape[i] != y_shape[i] and x_shape[i] != 1 and y_shape[i] != 1:
                raise ValueError(
                    f"Shapes of x {x_shape} and y {y_shape} are not broadcastable."
                )
            else:
                z_shape.append(max(x_shape[i], y_shape[i]))
        # Return the shape of the result
        return core.ShapedArray(tuple(z_shape), x.dtype)

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
