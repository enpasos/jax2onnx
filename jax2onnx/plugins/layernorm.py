# file: jax2onnx/plugins/layernorm.py
# JAX API reference: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__LayerNormalization.html

import onnx.helper as oh
import numpy as np
import onnx
from flax import nnx
import jax
import jax.numpy as jnp

from transpose_utils import jax_shape_to_onnx_shape


def build_onnx_node(self, jax_inputs, input_names, onnx_graph, parameters=None):
    """
    Build the ONNX node for a LayerNorm operation.

    Args:
        self: The nnx.LayerNorm instance.
        jax_inputs: List of input tensors in JAX format.
        input_names: List of corresponding input names in ONNX format.
        onnx_graph: The ONNX graph being constructed.
        parameters: Additional parameters (not used here).

    Returns:
        jax_outputs: The output tensors in JAX format.
        onnx_output_names: The corresponding output names in ONNX format.
    """
    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    # Extract parameters
    num_features = self.num_features
    feature_axes = self.feature_axes
    epsilon = self.epsilon
    use_bias = self.bias is not None
    use_scale = self.scale is not None

    # Define ONNX input names
    inputs = [input_names[0]]
    if use_scale:
        scale_name = f"{node_name}_scale"
        inputs.append(scale_name)
        onnx_graph.add_initializer(
            oh.make_tensor(scale_name, onnx.TensorProto.FLOAT, self.scale.value.shape, self.scale.value.flatten())
        )
    if use_bias:
        bias_name = f"{node_name}_bias"
        inputs.append(bias_name)
        onnx_graph.add_initializer(
            oh.make_tensor(bias_name, onnx.TensorProto.FLOAT, self.bias.value.shape, self.bias.value.flatten())
        )

    # Define ONNX output names
    onnx_output_names = [f"{node_name}_output"]

    onnx_input_shape = jax_shape_to_onnx_shape(jax_inputs[0].shape)

    # TODO: Room for better mapping JAX to ONNX functionality
    # axis as left value of feature_axes
    # axis equals feature_axes if feature_axes is of type int otherwise axis is feature_axes[0]
    axis = feature_axes if isinstance(feature_axes, int) else feature_axes[0]


    # Add the LayerNormalization node
    onnx_graph.add_node(
        oh.make_node(
            "LayerNormalization",
            inputs=inputs,
            outputs=onnx_output_names,
            name=node_name,
            epsilon=epsilon,
            axis = axis,
        )
    )

    # Compute the JAX outputs
    jax_outputs = [self(jax_inputs[0])]
    onnx_graph.add_local_outputs(jax_outputs, onnx_output_names)

    return jax_outputs, onnx_output_names

# Attach the build_onnx_node method to nnx.LayerNorm
nnx.LayerNorm.build_onnx_node = build_onnx_node

def get_test_params():
    """
    Define test parameters for LayerNorm.
    """
    return [
        {
            "model_name": "layernorm_default",
            "model": lambda: nnx.LayerNorm(64, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 10, 64)],
            "build_onnx_node": nnx.LayerNorm.build_onnx_node,
        },

        {
            "model_name": "layernorm_multiaxis",
            "model": lambda: nnx.LayerNorm(3*3*64, reduction_axes = (1, 2, 3), feature_axes=(1, 2, 3), rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 3, 3, 64)],
            "build_onnx_node": nnx.LayerNorm.build_onnx_node,
        },
    ]
