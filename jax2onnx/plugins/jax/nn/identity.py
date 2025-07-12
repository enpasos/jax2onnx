# file: jax2onnx/plugins/jax/nn/identity.py

from typing import TYPE_CHECKING

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define our own primitive
jax.nn.identity_p = Primitive("jax.nn.identity")
jax.nn.identity_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jax.nn.identity_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.identity.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="identity",
    testcases=[
        {
            "testcase": "jaxnn_identity",
            "callable": lambda x: jax.nn.identity(x),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_identity_1",
            "callable": lambda x: jax.nn.identity(x),
            "input_shapes": [(2, 5)],
        },
    ],
)
class JaxIdentityPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.identity calls to the ONNX Identity operator.
    """

    @staticmethod
    def abstract_eval(x):
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        identity_node = helper.make_node(
            "Identity",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("identity"),
        )
        s.add_node(identity_node)

    @staticmethod
    def get_monkey_patch():
        def patched_identity(x):
            return jax.nn.identity_p.bind(x)

        return patched_identity

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jax.nn],
            "patch_function": lambda _: JaxIdentityPlugin.get_monkey_patch(),
            "target_attribute": "identity",
        }


def identity_batching_rule(batched_args, batch_dims):
    """
    Batching rule for jax.nn.identity.
    Since identity is elementwise, we simply apply the primitive to the batched input.
    """
    (x,) = batched_args
    (bdim,) = batch_dims

    y = jax.nn.identity_p.bind(
        x,
    )
    return y, bdim


# === Registration ===

# Register the abstract evaluation function
jax.nn.identity_p.def_abstract_eval(JaxIdentityPlugin.abstract_eval)

# Register the batching rule
batching.primitive_batchers[jax.nn.identity_p] = identity_batching_rule
