# file: jax2onnx/plugins/jax/nn/selu.py

from typing import TYPE_CHECKING

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define our own primitive
jax.nn.selu_p = Primitive("jax.nn.selu")
jax.nn.selu_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jax.nn.selu_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.selu.html",
    onnx=[
        {
            "component": "Selu",
            "doc": "https://onnx.ai/onnx/operators/onnx__Selu.html",
        }
    ],
    since="v0.7.0",
    context="primitives.nn",
    component="selu",
    testcases=[
        {
            "testcase": "jaxnn_selu",
            "callable": lambda x: jax.nn.selu(x),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_selu_1",
            "callable": lambda x: jax.nn.selu(x),
            "input_shapes": [(2, 5)],
        },
    ],
)
class JaxSeluPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.selu calls to the ONNX Selu operator.
    """

    @staticmethod
    def abstract_eval(x):
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        # Forcing default selu values
        # Theoretically, they are vars according to the specs, but jax consider them as constants
        alpha = 1.67326319217681884765625
        gamma = 1.05070102214813232421875

        selu_node = helper.make_node(
            "Selu",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("selu"),
            alpha=alpha,
            gamma=gamma,
        )
        s.add_node(selu_node)

    @staticmethod
    def get_monkey_patch():
        def patched_selu(x):
            return jax.nn.selu_p.bind(x)

        return patched_selu

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jax.nn],
            "patch_function": lambda _: JaxSeluPlugin.get_monkey_patch(),
            "target_attribute": "selu",
        }


def selu_batching_rule(batched_args, batch_dims):
    """
    Batching rule for jax.nn.selu.
    Since selu is elementwise, we simply apply the primitive to the batched input.
    """
    (x,) = batched_args
    (bdim,) = batch_dims

    y = jax.nn.selu_p.bind(
        x,
    )
    return y, bdim


# === Registration ===

# Register the abstract evaluation function
jax.nn.selu_p.def_abstract_eval(JaxSeluPlugin.abstract_eval)

# Register the batching rule
batching.primitive_batchers[jax.nn.selu_p] = selu_batching_rule
