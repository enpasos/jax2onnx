from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the LayerNorm primitive
nnx.layer_norm_p = Primitive("nnx.layer_norm")
nnx.layer_norm_p.multiple_results = False  # Correctly set at initialization


@register_primitive(
    jaxpr_primitive=nnx.layer_norm_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm",
    onnx=[
        {
            "component": "LayerNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__LayerNormalization.html",
        }
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="layer_norm",
    testcases=[
        {
            "testcase": "layer_norm",
            "callable": nnx.LayerNorm(num_features=32, epsilon=1e-5, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 20, 32)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "layer_norm_no_bias_no_scale",
            "callable": nnx.LayerNorm(
                32, use_bias=False, use_scale=False, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 20, 32)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "layer_norm_bias_no_scale",
            "callable": nnx.LayerNorm(
                32, use_bias=True, use_scale=False, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 20, 32)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "layer_norm_no_bias_scale",
            "callable": nnx.LayerNorm(
                32, use_bias=False, use_scale=True, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 20, 32)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "layer_norm_bias_scale",
            "callable": nnx.LayerNorm(
                32, use_bias=True, use_scale=True, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 20, 32)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "layer_norm_multiaxis",
            "callable": nnx.LayerNorm(
                3 * 3 * 64,
                reduction_axes=(1, 2, 3),
                feature_axes=(1, 2, 3),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 3, 3, 64)],
            "run_only_f32_variant": True,
        },
        # TODO: enable testcases
        # {
        #     "testcase": "layer_norm_symbolic_batch",
        #     # one LayerNorm, no other ops
        #     "callable": lambda x: nnx.LayerNorm(num_features=16, rngs=nnx.Rngs(0))(x),
        #     #     └─ dynamic batch ─┬─ sequence length ─┬─ feature dim
        #     "input_shapes": [("B", 8, 16)],
        #     "run_only_f32_variant": True,
        #     # ── sanity-check the generated graph ───────────────────────────
        #     "post_check_onnx_graph": lambda m: all(
        #         n.op_type != "Unsqueeze" and n.op_type != "Reshape"
        #         for n in m.graph.node
        #     ),
        # }
    ],
)
class LayerNormPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.LayerNorm to ONNX.
    """

    @staticmethod
    def abstract_eval(x, scale, bias, epsilon, axis):
        """Abstract evaluation function for LayerNorm."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of LayerNorm to ONNX format."""
        input_name = s.get_name(node_inputs[0])
        scale_name = s.get_name(node_inputs[1])
        bias_name = s.get_name(node_inputs[2])
        output_name = s.get_name(node_outputs[0])

        epsilon = params.get("epsilon")
        axis = params.get("axis", -1)  # Default normalization axis: last dimension

        ln_node = helper.make_node(
            "LayerNormalization",
            inputs=[input_name, scale_name, bias_name],
            outputs=[output_name],
            name=s.get_unique_name("layer_norm"),
            axis=axis,
            epsilon=epsilon,
        )
        s.add_node(ln_node)

    @staticmethod
    def _layer_norm(x, scale, bias, epsilon, axis):
        """Defines the primitive binding for LayerNorm."""
        return nnx.layer_norm_p.bind(
            x,
            scale,
            bias,
            epsilon=epsilon,
            axis=axis,
        )

    @staticmethod
    def get_monkey_patch():
        """Returns a patched version of LayerNorm's call method."""

        def patched_layer_norm_call(self, x):
            norm_axis = -1
            if hasattr(self, "reduction_axes"):
                if isinstance(self.reduction_axes, (list, tuple)):
                    norm_axis = min(self.reduction_axes)
                else:
                    norm_axis = self.reduction_axes

            param_dtype = self.param_dtype if self.param_dtype is not None else x.dtype

            feature_axes = self.feature_axes
            if isinstance(feature_axes, int):
                feature_axes = (feature_axes,)
            feature_shape = tuple(x.shape[i] for i in feature_axes)

            if self.use_scale and self.scale is not None:
                scale_value = self.scale.value
            else:
                scale_value = jnp.ones(feature_shape, dtype=param_dtype)

            if self.use_bias and self.bias is not None:
                bias_value = self.bias.value
            else:
                bias_value = jnp.zeros(feature_shape, dtype=param_dtype)

            return LayerNormPlugin._layer_norm(
                x,
                scale_value,
                bias_value,
                epsilon=self.epsilon,
                axis=norm_axis,
            )

        return patched_layer_norm_call

    @staticmethod
    def patch_info():
        """Provides patching information for LayerNorm."""
        return {
            "patch_targets": [nnx.LayerNorm],
            "patch_function": lambda _: LayerNormPlugin.get_monkey_patch(),
            "target_attribute": "__call__",
        }


# Register abstract evaluation function
nnx.layer_norm_p.def_abstract_eval(LayerNormPlugin.abstract_eval)
