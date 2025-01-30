# file: jax2onnx/plugins/linear.py

import onnx.helper as oh
import onnx
import numpy as np
from flax import nnx

def build_onnx_node(self, input_shapes, input_names, onnx_graph, parameters=None):
    """
    Constructs an ONNX node for a linear (fully connected) layer.

    This function converts an `nnx.Linear` layer into an ONNX `Gemm` (General Matrix Multiplication) node
    and adds the corresponding weight and bias initializers to the ONNX graph.

    Args:
        self: The `nnx.Linear` instance.
        input_shapes (list of tuples): List containing input tensor shapes, e.g., [(batch_size, input_dim)].
        input_names (list of str): Names of input tensors.
        onnx_graph: The ONNX graph object where the node will be added.
        parameters (optional): Additional parameters, currently unused.

    Returns:
        tuple:
            - output_shapes (list of tuples): Shape of the output tensor.
            - onnx_output_names (list of str): Names of the generated ONNX output tensors.
    """

    # Compute the output shape from the input shape and kernel dimensions
    output_shapes = [(input_shapes[0][0], self.kernel.shape[1])]

    # Generate a unique node name
    node1_name = f"node{onnx_graph.counter_plusplus()}"


    # Define the ONNX node for the linear layer using the Gemm operator
    node = oh.make_node(
        'Gemm',
        inputs=[input_names[0], f'{node1_name}_weight', f'{node1_name}_bias'],
        outputs=[f'{node1_name}_output'],
        name=node1_name,
    )
    onnx_graph.add_node(node)

    # Add the weight matrix as an ONNX initializer
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node1_name}_weight",
            onnx.TensorProto.FLOAT,
            self.kernel.shape,
            self.kernel.value.reshape(-1).astype(np.float32),
        )
    )

    # Add the bias vector as an ONNX initializer
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node1_name}_bias",
            onnx.TensorProto.FLOAT,
            [output_shapes[0][-1]],
            self.bias.value.astype(np.float32),
        )
    )

    # Register the output tensor in the ONNX graph
    onnx_output_names = [f'{node1_name}_output']
    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)

    return output_shapes, onnx_output_names


# Attach the ONNX conversion function to the nnx.Linear class
nnx.Linear.build_onnx_node = build_onnx_node


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
            "model": lambda: nnx.Linear(5, 3, rngs=nnx.Rngs(0)),  # Creates a Linear layer with input dim 5, output dim 3
            "input_shapes": [(1, 5)],  # Example input shape (batch_size=1, input_dim=5)
            "build_onnx_node": nnx.Linear.build_onnx_node
        }
    ]
