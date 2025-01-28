# file: jax2onnx/plugins/layernorm.py
# JAX API reference: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__LayerNormalization.html

import onnx.helper as oh
import numpy as np
import onnx
from flax import nnx

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

    # Extract the epsilon parameter or use a default value
    epsilon = getattr(self, "epsilon", 1e-5)

    # Extract scale and bias parameters
    scale_tensor = np.asarray(self.scale.value, dtype=np.float32) if hasattr(self, "scale") else np.ones(jax_inputs[0].shape[-1], dtype=np.float32)
    bias_tensor = np.asarray(self.bias.value, dtype=np.float32) if hasattr(self, "bias") else np.zeros(jax_inputs[0].shape[-1], dtype=np.float32)

    # Define scale and bias tensor names
    scale_name = f"{node_name}_scale"
    bias_name = f"{node_name}_bias"

    # Add scale and bias to the initializers
    onnx_graph.add_initializer(
        oh.make_tensor(scale_name, onnx.TensorProto.FLOAT, scale_tensor.shape, scale_tensor.flatten())
    )
    onnx_graph.add_initializer(
        oh.make_tensor(bias_name, onnx.TensorProto.FLOAT, bias_tensor.shape, bias_tensor.flatten())
    )

    # Define ONNX output names
    onnx_output_names = [f"{node_name}_output"]

    # Add the LayerNormalization node
    onnx_graph.add_node(
        oh.make_node(
            "LayerNormalization",
            inputs=[input_names[0], scale_name, bias_name],
            outputs=onnx_output_names,
            name=node_name,
            epsilon=epsilon,
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
            "model_name": "layernorm",
            "model": lambda: nnx.LayerNorm(num_features=64, epsilon=1e-5, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 10, 64)],  # JAX shape: (N, *, num_features)
            "build_onnx_node": lambda jax_inputs, input_names, onnx_graph, parameters=None: (
                nnx.LayerNorm.build_onnx_node(
                    nnx.LayerNorm(num_features=64, epsilon=1e-5, rngs=nnx.Rngs(0)),
                    jax_inputs,
                    input_names,
                    onnx_graph,
                    parameters
                )
            ),
        }
    ]
