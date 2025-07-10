# file: jax2onnx/plugins/jax/nn/softplus.py

from typing import TYPE_CHECKING

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define our own primitive
jax.nn.softplus_p = Primitive("jax.nn.softplus")
jax.nn.softplus_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jax.nn.softplus_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html",
    onnx=[
        {
            "component": "Softplus",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softplus.html",
        }
    ],
    since="v0.7.0",
    context="primitives.nn",
    component="softplus",
    testcases=[
        {
            "testcase": "jaxnn_softplus",
            "callable": lambda x: jax.nn.softplus(x),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_softplus_1",
            "callable": lambda x: jax.nn.softplus(x),
            "input_shapes": [(2, 5)],
        },
    ],
)
class JaxSoftplusPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.softplus calls to the ONNX Softplus operator.
    """

    @staticmethod
    def abstract_eval(x):
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        softplus_node = helper.make_node(
            "Softplus",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("softplus"),
        )
        s.add_node(softplus_node)

    @staticmethod
    def get_monkey_patch():
        def patched_softplus(x):
            return jax.nn.softplus_p.bind(x)

        return patched_softplus

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jax.nn],
            "patch_function": lambda _: JaxSoftplusPlugin.get_monkey_patch(),
            "target_attribute": "softplus",
        }


def softplus_batching_rule(batched_args, batch_dims):
    """
    Batching rule for jax.nn.softplus.
    Since Softplus is elementwise, we simply apply the primitive to the batched input.
    """
    (x,) = batched_args
    (bdim,) = batch_dims

    y = jax.nn.softplus_p.bind(x)
    return y, bdim


# === Registration ===

# Register the abstract evaluation function
jax.nn.softplus_p.def_abstract_eval(JaxSoftplusPlugin.abstract_eval)

# Register the batching rule
batching.primitive_batchers[jax.nn.softplus_p] = softplus_batching_rule
