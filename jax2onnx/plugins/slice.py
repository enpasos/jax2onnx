# file: jax2onnx/plugins/slice.py

# JAX API reference: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.slice.html
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__Slice.html

import jax
import onnx
import onnx.helper as oh
import numpy as np
from jax2onnx.to_onnx import Z

from functools import partial

def build_slice_onnx_node(z, parameters=None):
    """
    Converts JAX lax.slice operation to ONNX Slice operation.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        parameters (dict, optional): Dictionary containing 'start', 'end', and optionally 'strides'.

    Returns:
        Z: Updated instance with new shapes and names.
    """
    if parameters is None or "start" not in parameters or "end" not in parameters:
        raise ValueError("Slice operation requires 'start' and 'end' parameters.")

    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]

    start = parameters["start"]
    end = parameters["end"]
    strides = parameters.get("strides", [1] * len(start))  # Default to step size 1

    node_name = f"node{onnx_graph.next_id()}"
    output_name = f"{node_name}_output"

    # Add Slice node to the ONNX graph
    onnx_graph.add_node(
        oh.make_node(
            "Slice",
            inputs=[input_name, f"{node_name}_start", f"{node_name}_end", f"{node_name}_axes", f"{node_name}_steps"],
            outputs=[output_name],
            name=node_name,
        )
    )

    # Add initializers for slice parameters
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node_name}_start",
            onnx.TensorProto.INT64,
            [len(start)],
            np.array(start, dtype=np.int64),
        )
    )
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node_name}_end",
            onnx.TensorProto.INT64,
            [len(end)],
            np.array(end, dtype=np.int64),
        )
    )
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node_name}_axes",
            onnx.TensorProto.INT64,
            [len(start)],
            np.arange(len(start), dtype=np.int64),
        )
    )
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node_name}_steps",
            onnx.TensorProto.INT64,
            [len(strides)],
            np.array(strides, dtype=np.int64),
        )
    )

    # Update output shapes
    output_shape = [(e - s) // step for s, e, step in zip(start, end, strides)]
    onnx_graph.add_local_outputs([output_shape], [output_name])

    # Corrected jax_function
    jax_function = partial(jax.lax.slice, start_indices=start, limit_indices=end, strides=strides)

    return Z([output_shape], [output_name], onnx_graph, jax_function=jax_function)


# Attach ONNX conversion method to JAX slice function
jax.lax.slice.to_onnx = build_slice_onnx_node

def get_test_params():
    """
    Defines test parameters for verifying the ONNX conversion of the Slice operation.

    Returns:
        list: A list of test cases with expected slice parameters.
    """
    return [
        {
            "model_name": "slice_basic",
            "input_shapes": [(1, 5, 5, 3)],  # Example input shape (batch, height, width, channels)
            "to_onnx": jax.lax.slice.to_onnx,
            "export": {"start": [0, 1, 1, 0], "end": [1, 4, 4, 3]},  # Extracting a 3x3 region
        },
        {
            "model_name": "slice_with_stride",
            "input_shapes": [(1, 6, 6, 3)],
            "to_onnx": jax.lax.slice.to_onnx,
            "export": {"start": [0, 0, 0, 0], "end": [1, 6, 6, 3], "strides": [1, 2, 2, 1]},  # Strided slice
        },
        {
            "model_name": "slice_single_element",
            "input_shapes": [(3, 3)],  # Example 2D matrix
            "to_onnx": jax.lax.slice.to_onnx,
            "export": {"start": [1, 1], "end": [2, 2]},  # Extracting a single element
        },
        {
            "model_name": "slice_last_column",
            "input_shapes": [(4, 5)],  # Example 2D matrix
            "to_onnx": jax.lax.slice.to_onnx,
            "export": {"start": [0, 4], "end": [4, 5]},  # Extracting the last column
        }
    ]
