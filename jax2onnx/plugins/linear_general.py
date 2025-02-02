# file: jax2onnx/plugins/linear_general.py

import onnx
import onnx.helper as oh
import numpy as np
from flax import nnx

def build_onnx_node(self, input_shapes, input_names, onnx_graph, parameters=None):
    """
    Constructs an ONNX node for `LinearGeneral`, which generalizes the linear layer
    to transformations over multiple axes.

    Args:
        self: The `nnx.LinearGeneral` instance.
        input_shapes (list of tuples): List containing input tensor shapes, e.g., [(batch_size, input_dim)].
        input_names (list of str): Names of input tensors.
        onnx_graph: The ONNX graph object where the node will be added.
        parameters (optional): Additional parameters.

    Returns:
        tuple:
            - output_shapes (list of tuples): Shape of the output tensor.
            - onnx_output_names (list of str): Names of the generated ONNX output tensors.
    """

    if parameters is None:
        parameters = {}

    # Ensure a valid input shape
    if len(input_shapes) < 1:
        raise ValueError("Expected at least one input for LinearGeneral.")

    input_shape = input_shapes[0]
    input_name = input_names[0]

    # Extract relevant attributes from `LinearGeneral`
    in_features = self.in_features  # Tuple or int
    out_features = self.out_features  # Tuple or int
    axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)  # Ensure tuple
    use_bias = self.use_bias

    # Compute output shape
    batch_dims = input_shape[:-len(axis)]
    # Convert batch_dims to tuple before concatenation
    output_shape = tuple(batch_dims) + tuple(out_features)

    node_prefix = f"node{onnx_graph.counter_plusplus()}"

    # Register kernel as an ONNX initializer
    kernel_name = f"{node_prefix}_kernel"
    onnx_graph.add_initializer(
        oh.make_tensor(
            kernel_name,
            onnx.TensorProto.FLOAT,
            self.kernel.shape,
            self.kernel.value.reshape(-1).astype(np.float32),
        )
    )

    # Register bias if used
    bias_name = None
    if use_bias:
        bias_name = f"{node_prefix}_bias"
        onnx_graph.add_initializer(
            oh.make_tensor(
                bias_name,
                onnx.TensorProto.FLOAT,
                self.bias.shape,
                self.bias.value.reshape(-1).astype(np.float32),
            )
        )

    # ONNX `Gemm` operation (general matrix multiplication)
    onnx_output_name = f"{node_prefix}_output"
    onnx_inputs = [input_name, kernel_name] + ([bias_name] if use_bias else [])

    onnx_graph.add_node(
        oh.make_node(
            "Gemm",
            inputs=onnx_inputs,
            outputs=[onnx_output_name],
            name=node_prefix,
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=0
        )
    )

    # Register the output tensor in the ONNX graph
    onnx_graph.add_local_outputs([output_shape], [onnx_output_name])

    return [output_shape], [onnx_output_name]


# Attach the ONNX conversion function to the nnx.LinearGeneral class
nnx.LinearGeneral.build_onnx_node = build_onnx_node


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of `nnx.LinearGeneral`.

    The test parameters define:
    - A simple `nnx.LinearGeneral` model with input and output dimensions.
    - The corresponding input tensor shape.
    - The ONNX conversion function to be used in unit tests.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "linear_general",
            "model": lambda: nnx.LinearGeneral(
                in_features=(64,),
                out_features=(128,),
                axis=(-1,),
                use_bias=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(4, 64)],  # Example input shape (batch_size=4, input_dim=64)
            "build_onnx_node": nnx.LinearGeneral.build_onnx_node
        }
    ]
