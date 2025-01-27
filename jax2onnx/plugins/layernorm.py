# file: jax2onnx/plugins/layernorm.py
# JAX API reference: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
import onnx.helper as oh
import numpy as np
import onnx
from flax import nnx
from jax2onnx.onnx_export import OnnxGraph

def build_onnx_node(self, jax_inputs, input_names, onnx_graph, parameters=None):
    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    # Extract the epsilon parameter from the LayerNorm instance or use a default
    epsilon = getattr(self, "epsilon", 1e-5)  # Default value for epsilon

    # Use scale and bias from the LayerNorm instance if available
    scale_tensor = np.asarray(self.scale.value, dtype=np.float32) if hasattr(self, "scale") else np.ones(jax_inputs[0].shape[-1], dtype=np.float32)
    bias_tensor = np.asarray(self.bias.value, dtype=np.float32) if hasattr(self, "bias") else np.zeros(jax_inputs[0].shape[-1], dtype=np.float32)

    scale_name = f"{node_name}_scale"
    bias_name = f"{node_name}_bias"

    onnx_graph.add_initializer(
        oh.make_tensor(scale_name, onnx.TensorProto.FLOAT, scale_tensor.shape, scale_tensor.flatten())
    )
    onnx_graph.add_initializer(
        oh.make_tensor(bias_name, onnx.TensorProto.FLOAT, bias_tensor.shape, bias_tensor.flatten())
    )

    output_names = [f"{node_name}_output"]

    # Add LayerNormalization node
    onnx_graph.add_node(
        oh.make_node(
            "LayerNormalization",
            inputs=[input_names[0], scale_name, bias_name],
            outputs=output_names,
            name=node_name,
            epsilon=epsilon,
        )
    )

    return output_names

nnx.LayerNorm.build_onnx_node = build_onnx_node

def get_test_params():
    return [
        {
            "model_name": "layernorm",
            "model": lambda: nnx.LayerNorm(num_features=64, epsilon=1e-5, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 10, 64)],
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
