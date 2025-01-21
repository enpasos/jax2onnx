# file: jax2onnx/plugins/layernorm.py
# JAX API reference: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
import onnx.helper as oh
import numpy as np
import onnx
from flax import nnx

def build_onnx_node(self, example_input, input_name, nodes, initializers, counter):
    node_name = f"node{counter[0]}"
    counter[0] += 1

    # Extract the epsilon parameter from the LayerNorm instance or use a default
    epsilon = 1e-5  # Default value for epsilon

    # Create a scale and bias tensor for LayerNorm
    input_shape = example_input.shape
    feature_dim = input_shape[-1]  # Assuming normalization is applied to the last dimension
    scale_name = f"{node_name}_scale"
    bias_name = f"{node_name}_bias"

    scale_tensor = np.ones((feature_dim,), dtype=np.float32)
    bias_tensor = np.zeros((feature_dim,), dtype=np.float32)

    initializers.append(
        oh.make_tensor(scale_name, onnx.TensorProto.FLOAT, scale_tensor.shape, scale_tensor.flatten())
    )
    initializers.append(
        oh.make_tensor(bias_name, onnx.TensorProto.FLOAT, bias_tensor.shape, bias_tensor.flatten())
    )

    # Add LayerNormalization node
    nodes.append(
        oh.make_node(
            "LayerNormalization",
            inputs=[input_name, scale_name, bias_name],
            outputs=[f"{node_name}_output"],
            name=node_name,
            epsilon=epsilon,
        )
    )

    return f"{node_name}_output"


nnx.LayerNorm.build_onnx_node = build_onnx_node

def get_test_params():
    return [
        {
            "model_name": "layernorm",
            "model": lambda: nnx.LayerNorm(num_features=64, epsilon=1e-5, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 10, 64)],
            "build_onnx_node": nnx.LayerNorm.build_onnx_node,
        }
    ]
