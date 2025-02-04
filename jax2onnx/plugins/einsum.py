# file: jax2onnx/plugins/einsum.py
import onnx.helper as oh
import jax.numpy as jnp
import flax.nnx as nnx

def build_einsum_onnx_node(function, input_shapes, input_names, onnx_graph, parameters):
    """
    Constructs an ONNX node for an Einsum operation.

    Args:
        input_shapes (list of tuples): List of input tensor shapes.
        input_names (list of str): Names of input tensors.
        onnx_graph: The ONNX graph object where the node will be added.
        parameters (dict): Dictionary containing 'equation' information.

    Returns:
        tuple:
            - output_shapes (list of tuples): Shape of the output tensor.
            - onnx_output_names (list of str): Names of the generated ONNX output tensors.
    """
    equation = parameters.get("equation", "BNHE,BMHE->BNHM")

    # Generate a unique node name
    node_name = f"node{onnx_graph.counter_plusplus()}"

    # ONNX Einsum output shape is derived from the equation
    jnp_inputs = [jnp.zeros(shape) for shape in input_shapes]
    jax_outputs = [function(*jnp_inputs)]
    output_shapes = [jax_outputs[0].shape]
    output_names = [f"{node_name}_output"]

    # Add the Einsum node to the ONNX graph
    onnx_graph.add_node(
        oh.make_node(
            "Einsum",
            inputs=input_names,
            outputs=output_names,
            name=node_name,
            equation=equation
        )
    )

    onnx_graph.add_local_outputs(output_shapes, output_names)
    return output_shapes, output_names

# Register the ONNX node builder for einsum
jnp.einsum.build_onnx_node = build_einsum_onnx_node

# Example test parameters
def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of einsum.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """

    equation = "BNHE,BMHE->BNHM"
    return [
        {
            "model_name": "einsum",
            "model": lambda: lambda a, b: jnp.einsum(equation, a, b),
            "input_shapes": [(1, 64, 8, 32), (1, 128, 8, 32)],
            "build_onnx_node": jnp.einsum.build_onnx_node,
            "export": {"equation": equation},
        },
    ]
