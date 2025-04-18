from typing import TYPE_CHECKING
import numpy as np
from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define a primitive for nnx.Einsum (the module) with the name 'einsum'
nnx.einsum_p = Primitive("nnx.einsum")
nnx.einsum_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=nnx.einsum_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/array.html#flax.nnx.Einsum",
    onnx=[
        {
            "component": "Einsum",
            "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
        },
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    ],
    since="v0.4.2",
    context="primitives.nnx",
    component="einsum",
    testcases=[
        {
            "testcase": "einsum_module_with_bias",
            "callable": nnx.Einsum(
                "nta,hab->nthb", (8, 2, 4), (8, 4), rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(16, 11, 2)],
        },
        {
            "testcase": "einsum_module_no_bias",
            "callable": nnx.Einsum("nta,hab->nthb", (8, 2, 4), None, rngs=nnx.Rngs(0)),
            "input_shapes": [(16, 11, 2)],
        },
    ],
)
class EinsumPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.Einsum (module) to ONNX.
    Handles both with and without bias.
    """

    @staticmethod
    def abstract_eval(x, kernel, *args, einsum_str):
        bias = args[0] if args else None
        input_shape = x.shape
        kernel_shape = kernel.shape
        bias_shape = bias.shape if bias is not None else None
        dummy_x = np.zeros(input_shape)
        dummy_kernel = np.zeros(kernel_shape)
        einsum_out = np.einsum(einsum_str, dummy_x, dummy_kernel)
        if bias_shape is not None:
            dummy_bias = np.zeros(bias_shape)
            einsum_out = einsum_out + dummy_bias
        return core.ShapedArray(einsum_out.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        einsum_str = params.get("einsum_str")
        input_name = s.get_name(node_inputs[0])
        kernel_name = s.get_name(node_inputs[1])
        output_name = s.get_name(node_outputs[0])
        has_bias = len(node_inputs) > 2
        if has_bias:
            bias_name = s.get_name(node_inputs[2])
            einsum_node_out = s.get_unique_name("einsum_out")
        else:
            bias_name = None
            einsum_node_out = output_name
        einsum_node = helper.make_node(
            "Einsum",
            inputs=[input_name, kernel_name],
            outputs=[einsum_node_out],
            name=s.get_unique_name("einsum"),
            equation=einsum_str,
        )
        s.add_node(einsum_node)
        # Register shape info for einsum output
        input_shape = node_inputs[0].aval.shape
        kernel_shape = node_inputs[1].aval.shape
        dummy_x = np.zeros(input_shape)
        dummy_kernel = np.zeros(kernel_shape)
        einsum_out = np.einsum(einsum_str, dummy_x, dummy_kernel)
        s.add_shape_info(einsum_node_out, einsum_out.shape)
        # Add bias if present
        if has_bias:
            add_node = helper.make_node(
                "Add",
                inputs=[einsum_node_out, bias_name],
                outputs=[output_name],
                name=s.get_unique_name("einsum_add_bias"),
            )
            s.add_node(add_node)
            # Register shape info for final output
            bias_shape = node_inputs[2].aval.shape
            dummy_bias = np.zeros(bias_shape)
            einsum_out = einsum_out + dummy_bias
        s.add_shape_info(output_name, einsum_out.shape)

    @staticmethod
    def _einsum(x, kernel, *args, einsum_str):
        if args:
            bias = args[0]
            return nnx.einsum_p.bind(x, kernel, bias, einsum_str=einsum_str)
        else:
            return nnx.einsum_p.bind(x, kernel, einsum_str=einsum_str)

    @staticmethod
    def get_monkey_patch():
        def patched_einsum_module_call(self, x):
            if self.bias is not None:
                return EinsumPlugin._einsum(
                    x, self.kernel.value, self.bias.value, einsum_str=self.einsum_str
                )
            else:
                return EinsumPlugin._einsum(
                    x, self.kernel.value, einsum_str=self.einsum_str
                )

        return patched_einsum_module_call

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx.Einsum],
            "patch_function": lambda _: EinsumPlugin.get_monkey_patch(),
            "target_attribute": "__call__",
        }


# Register abstract evaluation function
nnx.einsum_p.def_abstract_eval(EinsumPlugin.abstract_eval)
