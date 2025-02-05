# file: jax2onnx/plugins/batchnorm.py

# JAX API reference: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__BatchNormalization.html


import numpy as np
import onnx
import onnx.helper as oh
from flax import nnx
from jax2onnx.to_onnx import Z


def to_onnx(self, z, parameters=None):
    """
    Converts an `nnx.BatchNorm` layer into an ONNX `BatchNormalization` node.

    Args:
        self: The `nnx.BatchNorm` instance.
        z (Z): Contains input shapes, names, and the ONNX graph.
        parameters (dict, optional): Additional conversion parameters.

    Returns:
        Z: Updated instance with new shapes and names.
    """

    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]

    # Generate unique node name
    node_name = f"node{onnx_graph.next_id()}"

    # Extract epsilon and momentum or use defaults
    epsilon = getattr(self, "epsilon", 1e-5)
    momentum = 1 - getattr(self, "momentum", 0.9)

    # Extract learned parameters
    scale = self.scale.value
    bias = self.bias.value
    mean = self.mean.value
    var = self.var.value

    # Create tensor names
    scale_name = f"{node_name}_scale"
    bias_name = f"{node_name}_bias"
    mean_name = f"{node_name}_mean"
    var_name = f"{node_name}_variance"

    # Add parameters to the initializers
    onnx_graph.add_initializer(oh.make_tensor(scale_name, onnx.TensorProto.FLOAT, scale.shape, scale.flatten().astype(np.float32)))
    onnx_graph.add_initializer(oh.make_tensor(bias_name, onnx.TensorProto.FLOAT, bias.shape, bias.flatten().astype(np.float32)))
    onnx_graph.add_initializer(oh.make_tensor(mean_name, onnx.TensorProto.FLOAT, mean.shape, mean.flatten().astype(np.float32)))
    onnx_graph.add_initializer(oh.make_tensor(var_name, onnx.TensorProto.FLOAT, var.shape, var.flatten().astype(np.float32)))

    # Define ONNX output names
    onnx_output_names = [f"{node_name}_output"]

    # Add BatchNormalization node
    onnx_graph.add_node(
        oh.make_node(
            "BatchNormalization",
            inputs=[input_name, scale_name, bias_name, mean_name, var_name],
            outputs=onnx_output_names,
            name=node_name,
            epsilon=epsilon,
            momentum=momentum,
        )
    )

    # Compute output shape (BatchNorm does not change shape)
    output_shapes = [input_shape]

    # Register the output tensor in the ONNX graph
    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)

    return Z(output_shapes, onnx_output_names, onnx_graph)


# Attach the `to_onnx` method to `nnx.BatchNorm`
nnx.BatchNorm.to_onnx = to_onnx


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of `nnx.BatchNorm`.

    The test parameters define:
    - A simple `nnx.BatchNorm` model with input and output dimensions.
    - The corresponding input tensor shape.
    - The ONNX conversion function to be used in unit tests.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "batchnorm",
            "model":  nnx.BatchNorm(num_features=64, epsilon=1e-5, momentum=0.9, rngs=nnx.Rngs(0)),
            "input_shapes": [(11, 2, 2, 64)],  # JAX shape: (N, H, W, C)
            "to_onnx": nnx.BatchNorm.to_onnx,
            "export": {
                "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX (N, H, W, C) to ONNX (N, C, H, W)
                "post_transpose": [(0, 2, 3, 1)],  # Convert ONNX output back to JAX format
            }
        }
    ]
