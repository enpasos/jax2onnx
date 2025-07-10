# file: jax2onnx/plugins/jax/nn/mish.py

from typing import TYPE_CHECKING

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define our own primitive
jax.nn.mish_p = Primitive("jax.nn.mish")
jax.nn.mish_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jax.nn.mish_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.mish.html",
    onnx=[
        {
            "component": "Mish",
            "doc": "https://onnx.ai/onnx/operators/onnx__Mish.html",
        }
    ],
    since="v0.7.0",
    context="primitives.nn",
    component="mish",
    testcases=[
        {
            "testcase": "jaxnn_mish",
            "callable": lambda x: jax.nn.mish(x),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_mish_1",
            "callable": lambda x: jax.nn.mish(x),
            "input_shapes": [(2, 5)],
        },
    ],
)
class JaxMishPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.mish calls to the ONNX Mish operator.
    """

    @staticmethod
    def abstract_eval(x):
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        mish_node = helper.make_node(
            "Mish",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("mish"),
        )
        s.add_node(mish_node)

    @staticmethod
    def get_monkey_patch():
        def patched_mish(x):
            return jax.nn.mish_p.bind(x)

        return patched_mish

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jax.nn],
            "patch_function": lambda _: JaxMishPlugin.get_monkey_patch(),
            "target_attribute": "mish",
        }


def mish_batching_rule(batched_args, batch_dims):
    """
    Batching rule for jax.nn.mish.
    Since mish is elementwise, we simply apply the primitive to the batched input.
    """
    (x,) = batched_args
    (bdim,) = batch_dims

    y = jax.nn.mish_p.bind(
        x,
    )
    return y, bdim


# === Registration ===

# Register the abstract evaluation function
jax.nn.mish_p.def_abstract_eval(JaxMishPlugin.abstract_eval)

# Register the batching rule
batching.primitive_batchers[jax.nn.mish_p] = mish_batching_rule
