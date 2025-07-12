# file: jax2onnx/plugins/jax/nn/softsign.py

from typing import TYPE_CHECKING

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper
import numpy as np

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define our own primitive
jax.nn.soft_sign_p = Primitive("jax.nn.soft_sign")
jax.nn.soft_sign_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jax.nn.soft_sign_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.soft_sign.html",
    onnx=[
        {
            "component": "Softsign",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softsign.html",
        }
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="soft_sign",
    testcases=[
        {
            "testcase": "jaxnn_soft_sign",
            "callable": lambda x: jax.nn.soft_sign(x),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_soft_sign_1",
            "callable": lambda x: jax.nn.soft_sign(x),
            "input_shapes": [(2, 5)],
        },
    ],
)
class JaxSoftsignPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.soft_sign calls to the ONNX Softsign operator.
    """

    @staticmethod
    def abstract_eval(x):
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)
        dtype = input_var.aval.dtype

        if dtype == np.float32:
            # use the native ONNX Softsign op for float32
            node = helper.make_node(
                "Softsign",
                inputs=[input_name],
                outputs=[output_name],
                name=s.get_unique_name("softsign"),
            )
            s.add_node(node)
        else:
            # Softsign(x) = x / (1 + |x|)
            # 1) abs_x = Abs(x)
            abs_x = s.get_unique_name("abs_x")
            s.add_node(
                helper.make_node(
                    "Abs",
                    inputs=[input_name],
                    outputs=[abs_x],
                    name=s.get_unique_name("abs"),
                )
            )
            s.add_shape_info(abs_x, input_var.aval.shape, dtype)

            # 2) denom = Add(abs_x, 1)
            one = np.array(1, dtype=dtype)
            one_const = s.get_constant_name(one)
            denom = s.get_unique_name("denom")
            s.add_node(
                helper.make_node(
                    "Add",
                    inputs=[abs_x, one_const],
                    outputs=[denom],
                    name=s.get_unique_name("add"),
                )
            )
            s.add_shape_info(denom, input_var.aval.shape, dtype)

            # 3) out = Div(x, denom)
            s.add_node(
                helper.make_node(
                    "Div",
                    inputs=[input_name, denom],
                    outputs=[output_name],
                    name=s.get_unique_name("div"),
                )
            )
            # shape_info for output is inferred by the converter

    @staticmethod
    def get_monkey_patch():
        def patched_softsign(x):
            return jax.nn.soft_sign_p.bind(x)

        return patched_softsign

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jax.nn],
            "patch_function": lambda _: JaxSoftsignPlugin.get_monkey_patch(),
            "target_attribute": "soft_sign",
        }


def softsign_batching_rule(batched_args, batch_dims):
    """
    Batching rule for jax.nn.soft_sign.
    Since Softsign is elementwise, we simply apply the primitive to the batched input.
    """
    (x,) = batched_args
    (bdim,) = batch_dims

    y = jax.nn.soft_sign_p.bind(x)
    return y, bdim


# === Registration ===

# Register the abstract evaluation function
jax.nn.soft_sign_p.def_abstract_eval(JaxSoftsignPlugin.abstract_eval)

# Register the batching rule
batching.primitive_batchers[jax.nn.soft_sign_p] = softsign_batching_rule
