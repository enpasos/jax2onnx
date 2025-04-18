"""
RMS Norm Plugin for JAX to ONNX conversion.

This plugin enables conversion of flax.nnx.RMSNorm layers to ONNX format.
It transforms JAX's rms_norm operations into an ONNX RMSNormalization operator
and falls back to a manual graph construction if needed.
"""

from typing import TYPE_CHECKING

import numpy as np
from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define a new primitive for RMS norm.
nnx.rms_norm_p = Primitive("nnx.rms_norm")
nnx.rms_norm_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=nnx.rms_norm_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.RMSNorm",
    onnx=[
        {
            "component": "RMSNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__RMSNormalization.html",
        },
    ],
    since="v0.3.0",
    context="primitives.nnx",
    component="rms_norm",
    testcases=[
        {
            "testcase": "rms_norm",
            "callable": nnx.RMSNorm(6, rngs=nnx.Rngs(0)),
            "input_shapes": [(11, 2, 2, 6)],
        },
        {
            "testcase": "rms_norm_2",
            "callable": nnx.RMSNorm(num_features=20, rngs=nnx.Rngs(0)),
            "input_shapes": [(2, 20)],
        },
    ],
)
class RMSNormPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.RMSNorm to ONNX.

    Attempts to use native RMSNormalization ONNX op, otherwise falls back to manual construction.
    """

    @staticmethod
    def abstract_eval(x, scale, *args, **kwargs):
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_name = s.get_name(node_inputs[0])
        scale_name = s.get_name(node_inputs[1])
        final_output_name = s.get_name(node_outputs[0])
        epsilon = params.get("epsilon", 1e-5)

        input_shape = s.shape_env[input_name]
        input_dtype = s.builder.dtype_env[
            input_name
        ]  # assuming dtype_env is attached to builder

        def register(name):
            s.builder.add_value_info(name, input_shape, input_dtype)

        # === ReduceMean(input) ===
        mean_square_name = s.get_unique_name("mean_square")
        s.add_node(
            helper.make_node(
                "ReduceMean",
                [input_name],
                [mean_square_name],
                name=s.get_unique_name("reduce_mean_square"),
                keepdims=1,
            )
        )
        register(mean_square_name)

        # === Sub(input, mean_square) ===
        sub_name = s.get_unique_name("sub")
        s.add_node(
            helper.make_node(
                "Sub",
                [input_name, mean_square_name],
                [sub_name],
                name=s.get_unique_name("sub_mean_square"),
            )
        )
        register(sub_name)

        # === Pow(sub, 2.0) ===
        square_name = s.get_unique_name("square")
        pow_const = s.get_constant_name(np.array(2.0, dtype=np.float32))
        s.add_node(
            helper.make_node(
                "Pow",
                [sub_name, pow_const],
                [square_name],
                name=s.get_unique_name("square"),
            )
        )
        register(square_name)

        # === ReduceMean(square) ===
        mean_square_2_name = s.get_unique_name("mean_square_2")
        s.add_node(
            helper.make_node(
                "ReduceMean",
                [square_name],
                [mean_square_2_name],
                name=s.get_unique_name("reduce_mean_square_2"),
                keepdims=1,
            )
        )
        register(mean_square_2_name)

        # === Add(epsilon) ===
        add_epsilon_name = s.get_unique_name("add_epsilon")
        eps_const = s.get_constant_name(np.array(epsilon, dtype=np.float32))
        s.add_node(
            helper.make_node(
                "Add",
                [mean_square_2_name, eps_const],
                [add_epsilon_name],
                name=s.get_unique_name("add_epsilon"),
            )
        )
        register(add_epsilon_name)

        # === Sqrt ===
        sqrt_name = s.get_unique_name("sqrt")
        s.add_node(
            helper.make_node(
                "Sqrt",
                [add_epsilon_name],
                [sqrt_name],
                name=s.get_unique_name("sqrt"),
            )
        )
        register(sqrt_name)

        # === Div(input, sqrt) ===
        div_name = s.get_unique_name("div")
        s.add_node(
            helper.make_node(
                "Div",
                [input_name, sqrt_name],
                [div_name],
                name=s.get_unique_name("div"),
            )
        )
        register(div_name)

        # === Mul(div, scale) === final output
        s.add_node(
            helper.make_node(
                "Mul",
                [div_name, scale_name],
                [final_output_name],
                name=s.get_unique_name("mul"),
            )
        )

    @staticmethod
    def _rms_norm(x, scale, epsilon):
        return nnx.rms_norm_p.bind(x, scale, epsilon=epsilon)

    @staticmethod
    def rms_norm(x, scale, epsilon):
        return RMSNormPlugin._rms_norm(x, scale, epsilon)

    @staticmethod
    def get_monkey_patch():
        def patched_rms_norm_call(self, x):
            return RMSNormPlugin._rms_norm(x, self.scale.value, self.epsilon)

        return patched_rms_norm_call

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx.RMSNorm],
            "patch_function": lambda _: RMSNormPlugin.get_monkey_patch(),
            "target_attribute": "__call__",
        }


# Register abstract evaluation function
nnx.rms_norm_p.def_abstract_eval(RMSNormPlugin.abstract_eval)
