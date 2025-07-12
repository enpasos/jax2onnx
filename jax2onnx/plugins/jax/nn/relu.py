# file: jax2onnx/plugins/jax/nn/relu.py

from typing import TYPE_CHECKING

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define our own primitive
jax.nn.relu_p = Primitive("jax.nn.relu")
jax.nn.relu_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jax.nn.relu_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.relu.html",
    onnx=[
        {
            "component": "Relu",
            "doc": "https://onnx.ai/onnx/operators/onnx__Relu.html",
        }
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="relu",
    testcases=[
        {
            "testcase": "jaxnn_relu",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_relu_1",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [(2, 5)],
        },
    ],
)
class JaxReluPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.relu calls to the ONNX Relu operator.
    """

    @staticmethod
    def abstract_eval(x):
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        relu_node = helper.make_node(
            "Relu",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("relu"),
        )
        s.add_node(relu_node)

    @staticmethod
    def get_monkey_patch():
        def patched_relu(x):
            return jax.nn.relu_p.bind(x)

        return patched_relu

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jax.nn],
            "patch_function": lambda _: JaxReluPlugin.get_monkey_patch(),
            "target_attribute": "relu",
        }


def relu_batching_rule(batched_args, batch_dims):
    """
    Batching rule for jax.nn.relu.
    Since Relu is elementwise, we simply apply the primitive to the batched input.
    """
    (x,) = batched_args
    (bdim,) = batch_dims

    y = jax.nn.relu_p.bind(x)
    return y, bdim


# === Registration ===

# Register the abstract evaluation function
jax.nn.relu_p.def_abstract_eval(JaxReluPlugin.abstract_eval)

# Register the batching rule
batching.primitive_batchers[jax.nn.relu_p] = relu_batching_rule
