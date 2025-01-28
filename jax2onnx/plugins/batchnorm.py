# file: jax2onnx/plugins/batchnorm.py
# JAX API reference: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
import onnx.helper as oh
import numpy as np
import onnx
from flax import nnx

def build_onnx_node(self, jax_inputs, input_names, onnx_graph, parameters=None):
    """
    Build the ONNX node for a BatchNorm operation.

    Args:
        self: The nnx.BatchNorm instance.
        jax_inputs: List of input tensors in JAX format.
        input_names: List of corresponding input names in ONNX format.
        onnx_graph: The ONNX graph being constructed.
        parameters: Additional parameters (not used here).
    """
    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    # Extract epsilon and momentum or use defaults
    epsilon = getattr(self, "epsilon", 1e-5)
    momentum = 1 - getattr(self, "momentum", 0.9)

    # Extract learned parameters from the BatchNorm instance
    scale = self.scale
    bias = self.bias
    mean = self.mean
    var = self.var

    # Ensure parameters are numpy arrays
    scale_tensor = np.asarray(scale, dtype=np.float32)
    bias_tensor = np.asarray(bias, dtype=np.float32)
    mean_tensor = np.asarray(mean, dtype=np.float32)
    var_tensor = np.asarray(var, dtype=np.float32)

    # Create tensor names
    scale_name = f"{node_name}_scale"
    bias_name = f"{node_name}_bias"
    mean_name = f"{node_name}_mean"
    var_name = f"{node_name}_variance"

    # Add parameters to the initializers
    onnx_graph.add_initializer(
        oh.make_tensor(scale_name, onnx.TensorProto.FLOAT, scale_tensor.shape, scale_tensor.flatten())
    )
    onnx_graph.add_initializer(
        oh.make_tensor(bias_name, onnx.TensorProto.FLOAT, bias_tensor.shape, bias_tensor.flatten())
    )
    onnx_graph.add_initializer(
        oh.make_tensor(mean_name, onnx.TensorProto.FLOAT, mean_tensor.shape, mean_tensor.flatten())
    )
    onnx_graph.add_initializer(
        oh.make_tensor(var_name, onnx.TensorProto.FLOAT, var_tensor.shape, var_tensor.flatten())
    )

    onnx_output_names = [f"{node_name}_output"]

    # Add BatchNormalization node
    onnx_graph.add_node(
        oh.make_node(
            "BatchNormalization",
            inputs=[input_names[0], scale_name, bias_name, mean_name, var_name],
            outputs=onnx_output_names,
            name=node_name,
            epsilon=epsilon,
            momentum=momentum,
        )
    )

    # Compute the JAX output
    jax_outputs = [self(jax_inputs[0])]

    onnx_graph.add_local_outputs(jax_outputs, onnx_output_names)

    return jax_outputs, onnx_output_names

# Attach the method to nnx.BatchNorm
nnx.BatchNorm.build_onnx_node = build_onnx_node


def get_test_params():
    """
    Define test parameters for BatchNorm.
    """
    return [
        {
            "model_name": "batchnorm",
            "model": lambda: nnx.BatchNorm(num_features=64, epsilon=1e-5, momentum=0.9, rngs=nnx.Rngs(0)),
            "input_shapes": [(11, 2, 2, 64)],  # JAX shape: (N, H, W, C)
            "build_onnx_node": nnx.BatchNorm.build_onnx_node,
        }
    ]
