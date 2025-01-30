# file: jax2onnx/plugins/batchnorm.py

# JAX API reference: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__BatchNormalization.html

import onnx.helper as oh
import numpy as np
import onnx
from flax import nnx
import jax.numpy as jnp
from transpose_utils import jax_shape_to_onnx_shape


def build_onnx_node(self, input_shapes, input_names, onnx_graph, parameters=None):
    """
    Constructs an ONNX node for a BatchNorm operation.

    This function converts an `nnx.BatchNorm` layer into an ONNX `BatchNormalization` node,
    adding the scale, bias, mean, and variance initializers to the ONNX graph.

    Args:
        self: The `nnx.BatchNorm` instance.
        input_shapes (list of tuples): List containing input tensor shapes.
        input_names (list of str): Names of input tensors.
        onnx_graph: The ONNX graph object where the node will be added.
        parameters (optional): Additional parameters, currently unused.

    Returns:
        tuple:
            - output_shapes (list of tuples): Shape of the output tensor.
            - onnx_output_names (list of str): Names of the generated ONNX output tensors.
    """

    # Extract input shape
    input_shape = input_shapes[0]

    # Generate a unique node name
    node_name = f"node{onnx_graph.counter_plusplus()}"


    # Extract epsilon and momentum or use defaults
    epsilon = getattr(self, "epsilon", 1e-5)
    momentum = 1 - getattr(self, "momentum", 0.9)

    # Extract learned parameters from the BatchNorm instance
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
    onnx_graph.add_initializer(
        oh.make_tensor(scale_name, onnx.TensorProto.FLOAT, scale.shape, scale.flatten().astype(np.float32))
    )
    onnx_graph.add_initializer(
        oh.make_tensor(bias_name, onnx.TensorProto.FLOAT, bias.shape, bias.flatten().astype(np.float32))
    )
    onnx_graph.add_initializer(
        oh.make_tensor(mean_name, onnx.TensorProto.FLOAT, mean.shape, mean.flatten().astype(np.float32))
    )
    onnx_graph.add_initializer(
        oh.make_tensor(var_name, onnx.TensorProto.FLOAT, var.shape, var.flatten().astype(np.float32))
    )

    # Define ONNX output names
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

    # Compute output shape (BatchNorm does not change shape)
    output_shapes = [input_shape]

    # Register the output tensor in the ONNX graph
    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)

    return output_shapes, onnx_output_names


# Attach the `build_onnx_node` method to nnx.BatchNorm
nnx.BatchNorm.build_onnx_node = build_onnx_node


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
            "model": lambda: nnx.BatchNorm(num_features=64, epsilon=1e-5, momentum=0.9, rngs=nnx.Rngs(0)),
            "input_shapes": [(11, 2, 2, 64)],  # JAX shape: (N, H, W, C)
            "build_onnx_node": nnx.BatchNorm.build_onnx_node,
            "export": {
                "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX (N, H, W, C) to ONNX (N, C, H, W)
                "post_transpose": [(0, 2, 3, 1)],  # Convert ONNX output back to JAX format
            }
        }
    ]
