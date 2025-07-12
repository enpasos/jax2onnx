# file: jax2onnx/plugins/jax/nn/sigmoid.py

from typing import TYPE_CHECKING

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define our own primitive
jax.nn.sigmoid_p = Primitive("jax.nn.sigmoid")
jax.nn.sigmoid_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jax.nn.sigmoid_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.sigmoid.html",
    onnx=[
        {
            "component": "Sigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
        }
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="sigmoid",
    testcases=[
        {
            "testcase": "jaxnn_sigmoid",
            "callable": lambda x: jax.nn.sigmoid(x),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_sigmoid_1",
            "callable": lambda x: jax.nn.sigmoid(x),
            "input_shapes": [(2, 5)],
        },
    ],
)
class JaxSigmoidPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.sigmoid calls to the ONNX Sigmoid operator.
    """

    @staticmethod
    def abstract_eval(x):
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        sigmoid_node = helper.make_node(
            "Sigmoid",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("sigmoid"),
        )
        s.add_node(sigmoid_node)

    @staticmethod
    def get_monkey_patch():
        def patched_sigmoid(x):
            return jax.nn.sigmoid_p.bind(x)

        return patched_sigmoid

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jax.nn],
            "patch_function": lambda _: JaxSigmoidPlugin.get_monkey_patch(),
            "target_attribute": "sigmoid",
        }


def sigmoid_batching_rule(batched_args, batch_dims):
    """
    Batching rule for jax.nn.sigmoid.
    Since sigmoid is elementwise, we simply apply the primitive to the batched input.
    """
    (x,) = batched_args
    (bdim,) = batch_dims

    y = jax.nn.sigmoid_p.bind(x)
    return y, bdim


# === Registration ===

# Register the abstract evaluation function
jax.nn.sigmoid_p.def_abstract_eval(JaxSigmoidPlugin.abstract_eval)

# Register the batching rule
batching.primitive_batchers[jax.nn.sigmoid_p] = sigmoid_batching_rule
