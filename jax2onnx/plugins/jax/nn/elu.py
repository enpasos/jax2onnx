# file: jax2onnx/plugins/jax/nn/elu.py

from typing import TYPE_CHECKING

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define our own primitive
jax.nn.elu_p = Primitive("jax.nn.elu")
jax.nn.elu_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jax.nn.elu_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.elu.html",
    onnx=[
        {
            "component": "Elu",
            "doc": "https://onnx.ai/onnx/operators/onnx__Elu.html",
        }
    ],
    since="v0.7.0",
    context="primitives.nn",
    component="elu",
    testcases=[
        {
            "testcase": "jaxnn_elu",
            "callable": lambda x: jax.nn.elu(x, alpha=0.1),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_elu_1",
            "callable": lambda x: jax.nn.elu(x, alpha=0.2),
            "input_shapes": [(2, 5)],
        },
    ],
)
class JaxEluPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.elu calls to the ONNX Elu operator.
    """

    @staticmethod
    def abstract_eval(x, alpha=1.0):
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        alpha = params.get("alpha", 1.0)

        elu_node = helper.make_node(
            "Elu",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("elu"),
            alpha=alpha,
        )
        s.add_node(elu_node)

    @staticmethod
    def get_monkey_patch():
        def patched_elu(x, alpha=1.0):
            return jax.nn.elu_p.bind(x, alpha=alpha)

        return patched_elu

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jax.nn],
            "patch_function": lambda _: JaxEluPlugin.get_monkey_patch(),
            "target_attribute": "elu",
        }


def elu_batching_rule(batched_args, batch_dims, *, alpha):
    """
    Batching rule for jax.nn.elu.
    Since elu is elementwise, we simply apply the primitive to the batched input.
    """
    (x,) = batched_args
    (bdim,) = batch_dims

    y = jax.nn.elu_p.bind(x, alpha=alpha)
    return y, bdim


# === Registration ===

# Register the abstract evaluation function
jax.nn.elu_p.def_abstract_eval(JaxEluPlugin.abstract_eval)

# Register the batching rule
batching.primitive_batchers[jax.nn.elu_p] = elu_batching_rule
