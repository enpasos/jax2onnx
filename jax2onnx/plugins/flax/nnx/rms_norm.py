"""
RMS Norm Plugin for JAX to ONNX conversion.

This plugin enables conversion of flax.nnx.RMSNorm layers to ONNX format.
It transforms JAX's rms_norm operations into an ONNX RMSNormalization operator
and falls back to a manual graph construction if needed.
"""

from typing import TYPE_CHECKING
from flax import nnx
from jax import core
from onnx import helper
from jax.extend.core import Primitive
from jax2onnx.plugin_system import register_primitive, PrimitiveLeafPlugin
import numpy as np

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

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
        # is supported in ONNX implementation, yet ... coming soon

        # try:
        #     rms_node = helper.make_node(
        #         "RMSNormalization",
        #         inputs=[input_name, scale_name],
        #         outputs=[final_output_name],
        #         name=s.get_unique_name("RMSNormalization"),
        #         epsilon=epsilon,
        #         axes=[-1],
        #     )
        #     s.add_node(rms_node)
        #     return
        # except Exception:
        #     # fallback implementation
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

        sub_name = s.get_unique_name("sub")
        s.add_node(
            helper.make_node(
                "Sub",
                [input_name, mean_square_name],
                [sub_name],
                name=s.get_unique_name("sub_mean_square"),
            )
        )

        square_name = s.get_unique_name("square")
        s.add_node(
            helper.make_node(
                "Pow",
                [sub_name, s.get_constant_name(np.array(2.0, dtype=np.float32))],
                [square_name],
                name=s.get_unique_name("square"),
            )
        )

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

        add_epsilon_name = s.get_unique_name("add_epsilon")
        s.add_node(
            helper.make_node(
                "Add",
                [
                    mean_square_2_name,
                    s.get_constant_name(np.array(epsilon, dtype=np.float32)),
                ],
                [add_epsilon_name],
                name=s.get_unique_name("add_epsilon"),
            )
        )

        sqrt_name = s.get_unique_name("sqrt")
        s.add_node(
            helper.make_node(
                "Sqrt", [add_epsilon_name], [sqrt_name], name=s.get_unique_name("sqrt")
            )
        )

        div_name = s.get_unique_name("div")
        s.add_node(
            helper.make_node(
                "Div",
                [input_name, sqrt_name],
                [div_name],
                name=s.get_unique_name("div"),
            )
        )

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
