# file: jax2onnx/plugins/linear.py

import numpy as np
import onnx
import onnx.helper as oh
from flax import nnx
from jax2onnx.to_onnx import Z


def to_onnx(self, z, parameters=None):
    """
    Converts an `nnx.Linear` layer into an ONNX `Gemm` (General Matrix Multiplication) node.

    This function adds the corresponding weight and bias initializers to the ONNX graph.

    Args:
        self: The `nnx.Linear` instance.
        z (Z): Contains input shapes, names, and the ONNX graph.
        parameters (dict, optional): Additional conversion parameters.

    Returns:
        Z: Updated instance with new shapes and names.
    """

    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]

    # Compute output shape from the input shape and kernel dimensions
    output_shape = (input_shape[0], self.kernel.shape[1])

    # Generate a unique node name
    node_name = f"node{onnx_graph.next_id()}"

    # Define ONNX node using the Gemm operator
    onnx_graph.add_node(
        oh.make_node(
            "Gemm",
            inputs=[input_name, f"{node_name}_weight", f"{node_name}_bias"],
            outputs=[f"{node_name}_output"],
            name=node_name,
        )
    )

    # Add weight matrix as an ONNX initializer
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node_name}_weight",
            onnx.TensorProto.FLOAT,
            self.kernel.shape,
            self.kernel.value.reshape(-1).astype(np.float32),
        )
    )

    # Add bias vector as an ONNX initializer
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node_name}_bias",
            onnx.TensorProto.FLOAT,
            [output_shape[-1]],
            self.bias.value.astype(np.float32),
        )
    )

    # Register the output tensor in the ONNX graph
    output_names = [f"{node_name}_output"]
    onnx_graph.add_local_outputs([output_shape], output_names)

    return Z([output_shape], output_names, onnx_graph)


# Attach the `to_onnx` method to `nnx.Linear`
nnx.Linear.to_onnx = to_onnx


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of `nnx.Linear`.

    The test parameters define:
    - A simple `nnx.Linear` model with input and output dimensions.
    - The corresponding input tensor shape.
    - The ONNX conversion function to be used in unit tests.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "linear",
            "model":  nnx.Linear(5, 3, rngs=nnx.Rngs(0)),  # Linear layer with input dim 5, output dim 3
            "input_shapes": [(1, 5)],  # Example input shape (batch_size=1, input_dim=5)
            "to_onnx": nnx.Linear.to_onnx,
        }
    ]
