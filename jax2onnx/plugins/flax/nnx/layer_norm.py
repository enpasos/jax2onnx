# file: jax2onnx/plugins/flax/nnx/layer_norm.py

from typing import TYPE_CHECKING

import numpy as np
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
            "input_shapes": [(10, 20, 32)],
        },
        {
            "testcase": "layer_norm_multiaxis",
            "callable": nnx.LayerNorm(
                3 * 3 * 64,
                reduction_axes=(1, 2, 3),
                feature_axes=(1, 2, 3),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(1, 3, 3, 64)],
        },
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
        # Expect node_inputs: [x, scale, bias]
        input_name = s.get_name(node_inputs[0])
        scale_name = s.get_name(node_inputs[1]) if node_inputs[1] is not None else None
        bias_name = s.get_name(node_inputs[2]) if node_inputs[2] is not None else None
        output_name = s.get_name(node_outputs[0])

        epsilon = params.get("epsilon")
        axis = params.get("axis", -1)  # Default normalization axis: last dimension

        # ONNX LayerNormalization expects three inputs: input, scale, bias.
        # Handle optional scale and bias.
        inputs = [input_name]
        if scale_name is not None:
            inputs.append(scale_name)
        else:  # If scale is None, a 1s tensor with proper shape must be created.
            input_shape = node_inputs[0].aval.shape
            scale_shape = [1] * len(input_shape)
            scale_shape[axis] = input_shape[axis]
            scale_name = s.builder.get_constant_name(
                np.ones(scale_shape, dtype=np.float32)
            )
            inputs.append(scale_name)

        if bias_name is not None:
            inputs.append(bias_name)
        else:  # If bias is None, a 0s tensor with proper shape must be created.
            input_shape = node_inputs[0].aval.shape
            bias_shape = [1] * len(input_shape)
            bias_shape[axis] = input_shape[axis]
            bias_name = s.builder.get_constant_name(
                np.zeros(bias_shape, dtype=np.float32)
            )
            inputs.append(bias_name)

        ln_node = helper.make_node(
            "LayerNormalization",
            inputs=inputs,
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
            # Default to axis=-1 if no reduction_axes are provided.
            norm_axis = -1
            if hasattr(self, "reduction_axes"):
                # If reduction_axes is iterable (list/tuple), take the minimum; otherwise, use it directly.
                if isinstance(self.reduction_axes, (list, tuple)):
                    norm_axis = min(self.reduction_axes)
                else:
                    norm_axis = self.reduction_axes
            return LayerNormPlugin._layer_norm(
                x,
                self.scale.value if self.scale is not None else None,
                self.bias.value if self.bias is not None else None,
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
