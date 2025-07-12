# file: jax2onnx/plugins/jax/nn/leakyrelu.py

from typing import TYPE_CHECKING

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define our own primitive
jax.nn.leaky_relu_p = Primitive("jax.nn.leaky_relu")
jax.nn.leaky_relu_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jax.nn.leaky_relu_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.leaky_relu.html",
    onnx=[
        {
            "component": "LeakyRelu",
            "doc": "https://onnx.ai/onnx/operators/onnx__LeakyRelu.html",
        }
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="leaky_relu",
    testcases=[
        {
            "testcase": "jaxnn_leaky_relu",
            "callable": lambda x: jax.nn.leaky_relu(x, negative_slope=0.1),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_leaky_relu_1",
            "callable": lambda x: jax.nn.leaky_relu(x, negative_slope=0.2),
            "input_shapes": [(2, 5)],
        },
    ],
)
class JaxLeakyReluPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.leaky_relu calls to the ONNX LeakyRelu operator.
    """

    @staticmethod
    def abstract_eval(x, negative_slope=0.01):
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        import numpy as np

        (input_var,) = node_inputs
        (output_var,) = node_outputs
        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        # JAX default negative_slope is 0.01
        alpha = params.get("negative_slope", 0.01)
        dtype = input_var.aval.dtype
        shape = input_var.aval.shape

        if dtype == np.float32:
            # native LeakyRelu for float32
            leaky = helper.make_node(
                "LeakyRelu",
                inputs=[input_name],
                outputs=[output_name],
                name=s.get_unique_name("leakyrelu"),
                alpha=alpha,
            )
            s.add_node(leaky)
            s.add_shape_info(output_name, shape, dtype)
        else:
            # fallback subgraph for float64:
            # out = Relu(x) - alpha * Relu(-x)

            # 1) pos = Relu(x)
            pos = s.get_unique_name("leakyrelu_pos")
            s.add_node(
                helper.make_node(
                    "Relu", [input_name], [pos], name=s.get_unique_name("relu")
                )
            )
            s.add_shape_info(pos, shape, dtype)

            # 2) neg_x = Neg(x)
            neg_x = s.get_unique_name("leakyrelu_neg_x")
            s.add_node(
                helper.make_node(
                    "Neg", [input_name], [neg_x], name=s.get_unique_name("neg")
                )
            )
            s.add_shape_info(neg_x, shape, dtype)

            # 3) neg_relu = Relu(neg_x)
            neg_relu = s.get_unique_name("leakyrelu_neg_relu")
            s.add_node(
                helper.make_node(
                    "Relu", [neg_x], [neg_relu], name=s.get_unique_name("relu")
                )
            )
            s.add_shape_info(neg_relu, shape, dtype)

            # 4) alpha_const
            alpha_const = s.get_constant_name(np.array(alpha, dtype=dtype))

            # 5) scaled = Mul(neg_relu, alpha_const)
            scaled = s.get_unique_name("leakyrelu_scaled")
            s.add_node(
                helper.make_node(
                    "Mul",
                    [neg_relu, alpha_const],
                    [scaled],
                    name=s.get_unique_name("mul"),
                )
            )
            s.add_shape_info(scaled, shape, dtype)

            # 6) out = Sub(pos, scaled)
            s.add_node(
                helper.make_node(
                    "Sub", [pos, scaled], [output_name], name=s.get_unique_name("sub")
                )
            )
            s.add_shape_info(output_name, shape, dtype)

    @staticmethod
    def get_monkey_patch():
        def patched_leaky_relu(x, negative_slope=0.01):
            return jax.nn.leaky_relu_p.bind(x, negative_slope=negative_slope)

        return patched_leaky_relu

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jax.nn],
            "patch_function": lambda _: JaxLeakyReluPlugin.get_monkey_patch(),
            "target_attribute": "leaky_relu",
        }


def leakyrelu_batching_rule(batched_args, batch_dims, *, negative_slope):
    """
    Batching rule for jax.nn.leaky_relu.
    Since leaky relu is elementwise, we simply apply the primitive to the batched input.
    """
    (x,) = batched_args
    (bdim,) = batch_dims

    y = jax.nn.leaky_relu_p.bind(x, negative_slope=negative_slope)
    return y, bdim


# === Registration ===

# Register the abstract evaluation function
jax.nn.leaky_relu_p.def_abstract_eval(JaxLeakyReluPlugin.abstract_eval)

# Register the batching rule
batching.primitive_batchers[jax.nn.leaky_relu_p] = leakyrelu_batching_rule
