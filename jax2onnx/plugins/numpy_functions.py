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

# Assign ONNX node builder to jax.numpy.add
jax.numpy.add.build_onnx_node = build_add_onnx_node

# Define test parameters for each numpy function
def get_test_params():
    return [
        {
        "model_name": "add",
        "model": lambda: lambda x, y: jnp.add(x, y),
        "input_shapes": [(1, 10), (1, 10)],  # Two input shapes for add
        "build_onnx_node": jax.numpy.add.build_onnx_node,
        },
    ]
