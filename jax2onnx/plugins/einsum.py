# file: jax2onnx/plugins/einsum.py
import onnx.helper as oh
import jax.numpy as jnp
import flax


# Einsum
def build_einsum_onnx_node(jax_inputs, input_names, onnx_graph, parameters):
    equation = parameters.get("equation", "BNHE,BMHE->BNHM")

    # Perform einsum operation in JAX
    jax_outputs = [jnp.einsum(equation, *jax_inputs)]

    node_name = f"node{onnx_graph.counter_plusplus()}"
    output_names = [f"{node_name}_output"]
    onnx_graph.increment_counter()

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

    onnx_graph.add_local_outputs(jax_outputs, output_names)
    return jax_outputs, output_names


# Assign ONNX node builder to einsum function
flax.nnx.einsum = lambda *args, **kwargs: None  # Placeholder for JAX implementation
flax.nnx.einsum.build_onnx_node = build_einsum_onnx_node


# Example test parameters
def get_test_params():
    return [
        {
            "model_name": "einsum_attention",
            "model": lambda: lambda q, k: flax.nnx.einsum(q, k),
            "input_shapes": [(1, 64, 8, 32), (1, 128, 8, 32)],
            "build_onnx_node": flax.nnx.einsum.build_onnx_node,
            "export": {"equation": "BNHE,BMHE->BNHM"},
        },
    ]
