# file: jax2onnx/plugins/jax/nn/celu.py

from typing import TYPE_CHECKING

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define our own primitive
jax.nn.celu_p = Primitive("jax.nn.celu")
jax.nn.celu_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jax.nn.celu_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.celu.html",
    onnx=[
        {
            "component": "Celu",
            "doc": "https://onnx.ai/onnx/operators/onnx__Celu.html",
        }
    ],
    since="v0.7.0",
    context="primitives.nn",
    component="celu",
    testcases=[
        {
            "testcase": "jaxnn_celu",
            "callable": lambda x: jax.nn.celu(x, alpha=0.1),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_celu_1",
            "callable": lambda x: jax.nn.celu(x, alpha=0.2),
            "input_shapes": [(2, 5)],
        },
    ],
)
class JaxCeluPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.celu calls to the ONNX Celu operator.
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

        celu_node = helper.make_node(
            "Celu",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("celu"),
            alpha=alpha,
        )
        s.add_node(celu_node)

    @staticmethod
    def get_monkey_patch():
        def patched_celu(x, alpha=1.0):
            return jax.nn.celu_p.bind(x, alpha=alpha)

        return patched_celu

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jax.nn],
            "patch_function": lambda _: JaxCeluPlugin.get_monkey_patch(),
            "target_attribute": "celu",
        }


def celu_batching_rule(batched_args, batch_dims, *, alpha):
    """
    Batching rule for jax.nn.celu.
    Since celu is elementwise, we simply apply the primitive to the batched input.
    """
    (x,) = batched_args
    (bdim,) = batch_dims

    y = jax.nn.celu_p.bind(x, alpha=alpha)
    return y, bdim


# === Registration ===

# Register the abstract evaluation function
jax.nn.celu_p.def_abstract_eval(JaxCeluPlugin.abstract_eval)

# Register the batching rule
batching.primitive_batchers[jax.nn.celu_p] = celu_batching_rule
