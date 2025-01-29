# file: jax2onnx/plugins/dropout.py
import onnx.helper as oh
import onnx
import numpy as np
from flax import nnx
from jax2onnx.onnx_export import OnnxGraph

def build_dropout_onnx_node(self, jax_inputs, input_names, onnx_graph, parameters=None):
    """
    Build the ONNX node for a Dropout operation.

    Args:
        self: The nnx.Dropout instance.
        jax_inputs: List of input tensors in JAX format.
        input_names: List of corresponding input names in ONNX format.
        onnx_graph: The ONNX graph being constructed.
        parameters: Additional parameters (not used here).

    Returns:
        jax_outputs: The output tensors in JAX format.
        output_names: The corresponding output names in ONNX format.
    """
    # Compute the JAX output for reference (assuming deterministic=True for ONNX inference)
    jax_output = self(jax_inputs[0], deterministic=True)

    # Generate a unique node name for the ONNX graph
    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()

    # Ensure dropout ratio is a proper scalar (empty shape [])
    dropout_ratio_name = f"{node_name}_ratio"

    # Add dropout ratio as an ONNX initializer with empty `dims` for scalar
    onnx_graph.add_initializer(
        oh.make_tensor(
            dropout_ratio_name,
            onnx.TensorProto.FLOAT,
            [],  # Empty list for scalar (not [1]!)
            [self.rate],  # Correctly formatted as a scalar value
        )
    )

    # Create the ONNX Dropout node
    dropout_node = oh.make_node(
        'Dropout',
        inputs=[input_names[0], dropout_ratio_name],  # Pass ratio as second input
        outputs=[f'{node_name}_output'],
        name=node_name,
    )
    onnx_graph.add_node(dropout_node)

    # Define ONNX output names and return both JAX outputs and ONNX output names
    output_names = [f"{node_name}_output"]
    jax_outputs = [jax_output]
    onnx_graph.add_local_outputs(jax_outputs, output_names)

    return jax_outputs, output_names

# Attach the `build_dropout_onnx_node` method to nnx.Dropout
nnx.Dropout.build_onnx_node = build_dropout_onnx_node


def get_test_params():
    """
    Define test parameters for Dropout.
    """
    return [
        {
            "model_name": "dropout",
            "model": lambda: nnx.Dropout(rate=0.5, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 64, 64, 3)],  # JAX shape: (B, H, W, C)
            "build_onnx_node": nnx.Dropout.build_onnx_node
        }
    ]
