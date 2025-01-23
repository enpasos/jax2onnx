# file: jax2onnx/plugins/numpy_functions.py
import onnx.helper as oh
import jax
import jax.numpy as jnp

# Generic function to create ONNX nodes for numpy functions
def build_generic_onnx_node(op_type, input_names, nodes, parameters, counter):
    node_name = f"node{counter[0]}"
    output_names=[f'{node_name}_output']
    counter[0] += 1
    nodes.append(
        oh.make_node(
            op_type,
            inputs=input_names,  # Now accepts a list of input names
            outputs=output_names,
            name=node_name,
        )
    )
    return output_names

# Add (element-wise addition)
def build_add_onnx_node(jax_inputs, input_names, nodes, parameters, counter):
    return build_generic_onnx_node('Add', input_names, nodes, parameters, counter)

# Assign ONNX node builder to jax.numpy.add and jax.numpy.concatenate
jax.numpy.add.build_onnx_node = build_add_onnx_node

def build_concat_onnx_node(jax_inputs, input_names, nodes, parameters, counter):
    if not isinstance(parameters, dict):
        raise TypeError("Expected parameters to be a dictionary.")
    axis = parameters.get("axis", 0)  # Default axis is 0

    # Create a unique node name and output
    node_name = f"node{counter[0]}"
    output_names = [f"{node_name}_output"]
    counter[0] += 1

    # Add the Concat node with the axis attribute
    nodes.append(
        oh.make_node(
            "Concat",
            inputs=input_names,
            outputs=output_names,
            name=node_name,
            axis=axis,  # Set the axis correctly
        )
    )
    return output_names



jax.numpy.concatenate.build_onnx_node = build_concat_onnx_node

# Define test parameters for each numpy function
def get_test_params():
    return [
        {
        "model_name": "add",
        "model": lambda: lambda x, y: jnp.add(x, y),
        "input_shapes": [(1, 10), (1, 10)],  # Two input shapes for add
        "build_onnx_node": jax.numpy.add.build_onnx_node,
        },
        {
            "model_name": "concat",
            "model": lambda: lambda x, y: jnp.concatenate([x, y], axis=1),
            "input_shapes": [(1, 10), (1, 10)],  # Compatible shapes for axis=1
            "build_onnx_node": build_concat_onnx_node,
            "parameters": {"axis": 1},  # Correct axis for concatenation
        }
    ]
