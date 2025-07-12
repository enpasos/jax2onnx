# file: jax2onnx/plugins/jax/nn/selu.py

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
    since="v0.7.1",
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
        dtype = input_var.aval.dtype

        alpha = 1.6732631921768188
        gamma = 1.0507010221481323

        if dtype == np.float32:
            # native ONNX Selu for float32
            selu_node = helper.make_node(
                "Selu",
                inputs=[input_name],
                outputs=[output_name],
                name=s.get_unique_name("selu"),
                alpha=alpha,
                gamma=gamma,
            )
            s.add_node(selu_node)
        else:
            # 1) exp_x = Exp(x)
            exp_x = s.get_unique_name("exp_x")
            s.add_node(
                helper.make_node(
                    "Exp", [input_name], [exp_x], name=s.get_unique_name("exp")
                )
            )
            s.add_shape_info(exp_x, input_var.aval.shape, dtype)

            # 2) expm1 = Sub(exp_x, 1)
            one = np.array(1, dtype=dtype)
            one_const = s.get_constant_name(one)
            expm1 = s.get_unique_name("expm1")
            s.add_node(
                helper.make_node(
                    "Sub", [exp_x, one_const], [expm1], name=s.get_unique_name("sub")
                )
            )
            s.add_shape_info(expm1, input_var.aval.shape, dtype)

            # 3) neg_part = Mul(alpha, expm1)
            alpha_const = s.get_constant_name(np.array(alpha, dtype=dtype))
            neg_part = s.get_unique_name("neg_part")
            s.add_node(
                helper.make_node(
                    "Mul",
                    [alpha_const, expm1],
                    [neg_part],
                    name=s.get_unique_name("mul"),
                )
            )
            s.add_shape_info(neg_part, input_var.aval.shape, dtype)

            # 4) mask = Greater(x, 0)
            zero_const = s.get_constant_name(np.array(0, dtype=dtype))
            mask = s.get_unique_name("selu_mask")
            s.add_node(
                helper.make_node(
                    "Greater",
                    [input_name, zero_const],
                    [mask],
                    name=s.get_unique_name("gt"),
                )
            )
            s.add_shape_info(mask, input_var.aval.shape, np.bool_)

            # 5) inner = Where(mask, x, neg_part)
            inner = s.get_unique_name("selu_inner")
            s.add_node(
                helper.make_node(
                    "Where",
                    [mask, input_name, neg_part],
                    [inner],
                    name=s.get_unique_name("where"),
                )
            )
            s.add_shape_info(inner, input_var.aval.shape, dtype)

            # 6) out = Mul(gamma, inner)
            gamma_const = s.get_constant_name(np.array(gamma, dtype=dtype))
            s.add_node(
                helper.make_node(
                    "Mul",
                    [gamma_const, inner],
                    [output_name],
                    name=s.get_unique_name("mul"),
                )
            )
            s.add_shape_info(output_name, input_var.aval.shape, dtype)

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
