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

        # Get input shape and dtype
        input_shape = s.shape_env[input_name]
        input_dtype = s.builder.dtype_env[input_name]

        # Determine the axis to normalize over (last dimension)
        axis = [len(input_shape) - 1]

        # Always use manual RMSNorm construction for ONNX compatibility
        # 1. ReduceMean(x ** 2, axis=-1, keepdims=True)
        pow2_name = s.get_unique_name("pow2")
        pow2_const = s.get_constant_name(np.array(2.0, dtype=np.float32))
        s.add_node(
            helper.make_node(
                "Pow",
                [input_name, pow2_const],
                [pow2_name],
                name=s.get_unique_name("pow2"),
            )
        )
        s.builder.add_value_info(pow2_name, input_shape, input_dtype)

        # ONNX ≥ 13 expects axes as a tensor input, **not** as an attribute
        axes_tensor_name = s.get_constant_name(np.array(axis, dtype=np.int64))
        mean_name = s.get_unique_name("mean")
        s.add_node(
            helper.make_node(
                "ReduceMean",
                [pow2_name, axes_tensor_name],
                [mean_name],
                keepdims=1,
                name=s.get_unique_name("reduce_mean"),
            )
        )
        mean_shape = list(input_shape)
        mean_shape[-1] = 1
        s.builder.add_value_info(mean_name, tuple(mean_shape), input_dtype)

        # 2. Add epsilon
        add_eps_name = s.get_unique_name("add_eps")
        eps_const = s.get_constant_name(np.array(epsilon, dtype=np.float32))
        s.add_node(
            helper.make_node(
                "Add",
                [mean_name, eps_const],
                [add_eps_name],
                name=s.get_unique_name("add_eps"),
            )
        )
        s.builder.add_value_info(add_eps_name, tuple(mean_shape), input_dtype)

        # 3. Sqrt
        sqrt_name = s.get_unique_name("sqrt")
        s.add_node(
            helper.make_node(
                "Sqrt",
                [add_eps_name],
                [sqrt_name],
                name=s.get_unique_name("sqrt"),
            )
        )
        s.builder.add_value_info(sqrt_name, tuple(mean_shape), input_dtype)

        # 4. Divide input by sqrt
        div_name = s.get_unique_name("div")
        s.add_node(
            helper.make_node(
                "Div",
                [input_name, sqrt_name],
                [div_name],
                name=s.get_unique_name("div"),
            )
        )
        s.builder.add_value_info(div_name, tuple(input_shape), input_dtype)

        # 5. Multiply by scale
        s.add_node(
            helper.make_node(
                "Mul",
                [div_name, scale_name],
                [final_output_name],
                name=s.get_unique_name("mul"),
            )
        )
        s.builder.add_value_info(final_output_name, tuple(input_shape), input_dtype)

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
