# file: jax2onnx/plugins/numpy_functions.py
import onnx.helper as oh
import jax
import jax.numpy as jnp
import flax
import onnx



# Add (element-wise addition)
def build_add_onnx_node(jax_inputs, input_names, onnx_graph, parameters=None):
    # Perform element-wise addition in JAX
    jax_outputs = [jax_inputs[0] + jax_inputs[1]]

    # jax_outputs = [jax_inputs[0]]  # Use the first input as an example for simplicity

    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()
    output_names = [f'{node_name}_output']

    onnx_graph.add_node(
        oh.make_node(
            'Add',
            inputs=input_names,
            outputs=output_names,
            name=node_name,
        )
    )

    onnx_graph.add_local_outputs(jax_outputs, output_names)
    return jax_outputs, output_names

# Assign ONNX node builder to jax.numpy.add
jax.numpy.add.build_onnx_node = build_add_onnx_node

# Concatenate (along a specified axis)
def build_concat_onnx_node(jax_inputs, input_names, onnx_graph, parameters):
    if not isinstance(parameters, dict):
        raise TypeError("Expected parameters to be a dictionary.")
    axis = parameters.get("axis", 0)  # Default axis is 0

    # Perform concatenation in JAX
    jax_outputs = [jnp.concatenate(jax_inputs, axis=axis)]

    # Create a unique node name and output
    node_name = f"node{onnx_graph.get_counter()}"
    output_names = [f"{node_name}_output"]
    onnx_graph.increment_counter()

    # Add the Concat node with the axis attribute
    onnx_graph.add_node(
        oh.make_node(
            "Concat",
            inputs=input_names,
            outputs=output_names,
            name=node_name,
            axis=axis,  # Set the axis correctly
        )
    )

    onnx_graph.add_local_outputs(jax_outputs, output_names)
    return jax_outputs, output_names

# Assign ONNX node builder to jax.numpy.concatenate
jax.numpy.concatenate.build_onnx_node = build_concat_onnx_node



# Example test parameters
def get_test_params():
    return [
        {
            "model_name": "add",
            "model": lambda: lambda x, y: jnp.add(x, y),
            "input_shapes": [(1, 10), (1, 10)],  # Two input shapes for add
            "build_onnx_node": jax.numpy.add.build_onnx_node
        },
        {
            "model_name": "concat",
            "model": lambda: lambda x, y: jnp.concatenate([x, y], axis=1),
            "input_shapes": [(1, 10), (1, 10)],  # Compatible shapes for axis=1
            "build_onnx_node": jnp.concatenate.build_onnx_node,
            "parameters": {"axis": 1},  # Correct axis for concatenation
        },



    ]
