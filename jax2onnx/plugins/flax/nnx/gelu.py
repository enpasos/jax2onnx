from typing import TYPE_CHECKING

from flax import nnx
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the GELU primitive
nnx.gelu_p = Primitive("nnx.gelu")
nnx.gelu_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=nnx.gelu_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.gelu",
    onnx=[
        {
            "component": "Gelu",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gelu.html",
        }
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="gelu",
    testcases=[
        {
            "testcase": "gelu",
            "callable": lambda x: nnx.gelu(x, approximate=False),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "gelu_1",
            "callable": lambda x: nnx.gelu(x, approximate=False),
            "input_shapes": [(1, 10)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "gelu_2",
            "callable": lambda x: nnx.gelu(x, approximate=True),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "gelu_3",
            "callable": lambda x: nnx.gelu(x, approximate=True),
            "input_shapes": [("B", 10)],
        },
    ],
)
class GeluPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.gelu to ONNX.
    """

    @staticmethod
    def abstract_eval(x, approximate=True):
        """Abstract evaluation function for GELU."""
        # Use update instead of creating a new ShapedArray to avoid issues with unhashable tracers
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of GELU to ONNX format."""
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        # Retrieve the approximate parameter (defaulting to True if not provided)
        approximate = params.get("approximate", True)
        approximation = "tanh" if approximate else "none"

        gelu_node = helper.make_node(
            "Gelu",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("gelu"),
            approximate=approximation,  # Correctly pass 'approximation'
        )
        s.add_node(gelu_node)

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for GELU."""

        def patched_gelu(x, approximate=True):
            return nnx.gelu_p.bind(x, approximate=approximate)

        return patched_gelu

    @staticmethod
    def patch_info():
        """Provides patching information for GELU."""
        return {
            "patch_targets": [nnx],
            "patch_function": lambda _: GeluPlugin.get_monkey_patch(),
            "target_attribute": "gelu",
        }


# Register abstract evaluation function
nnx.gelu_p.def_abstract_eval(GeluPlugin.abstract_eval)
