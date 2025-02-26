# file: jax2onnx/plugins/jax/lax/slice.py


import jax
import numpy as np
import onnx
import onnx.helper as oh
import jax.numpy as jnp
from jax2onnx.convert import Z, OnnxGraph


def build_slice_onnx_node(z: Z, **params) -> Z:
    """Convert JAX lax.slice operation to ONNX Slice operation."""
    if "start" not in params or "end" not in params:
        raise ValueError("Slice operation requires 'start' and 'end' parameters.")

    onnx_graph: OnnxGraph = z.onnx_graph
    input_name = z.names[0]
    input_shape = z.shapes[0]

    # assign the highest int value to indicate dynamic batch dimension
    # using a literal value here is not ideal, but it is the best we can do
    b = 9223372036854775807
    batch_dim = input_shape[0]

    start = list(map(int, params["start"]))  # Convert to Python int list
    end = list(map(int, params["end"]))
    strides = list(map(int, params.get("strides", [1] * len(start))))  # Default step=1
    axes = list(range(len(start)))  # Default axes

    # Define JAX slice function
    def jax_slice_fn(x):
        start_copy = start.copy()
        end_copy = end.copy()
        strides_copy = strides.copy()

        limit_indices = []
        for i in range(len(start_copy)):
            dim_size = x.shape[i]
            if end_copy[i] < 0:
                limit_indices.append(dim_size + end_copy[i])
            else:
                limit_indices.append(min(end_copy[i], dim_size))

        return jax.lax.slice(
            x,
            start_indices=start_copy,
            limit_indices=limit_indices,
            strides=strides_copy,
        )

    if onnx_graph.dynamic_batch_dim:
        input_shape = [b] + list(input_shape[1:])  # Remove batch dimension

    # Handle negative indices for `start` and `end`
    start_ = [(s + input_shape[i]) if s < 0 else s for i, s in enumerate(start)]
    end_ = [(e + input_shape[i] + 1) if e < 0 else e for i, e in enumerate(end)]

    output_shape = [
        max(0, (e - s + (step - 1)) // step)  # Ensures no negative dims
        for s, e, step in zip(start_, end_, strides)
    ]

    output_shape = [batch_dim if s == b else s for s in output_shape]

    # Create unique node name and output name
    node_name = f"node{onnx_graph.next_id()}"
    output_name = f"{node_name}_output"

    # Create ONNX slice parameters as initializers
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
            [len(axes)],
            np.array(axes, dtype=np.int64),
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

    # Add Slice node to ONNX graph
    onnx_graph.add_node(
        oh.make_node(
            "Slice",
            inputs=[
                input_name,
                f"{node_name}_start",
                f"{node_name}_end",
                f"{node_name}_axes",
                f"{node_name}_steps",
            ],
            outputs=[output_name],
            name=node_name,
        )
    )

    onnx_graph.add_local_outputs([tuple(output_shape)], [output_name])

    return Z(
        [tuple(output_shape)], [output_name], onnx_graph, jax_function=jax_slice_fn
    )


# Attach ONNX conversion method to JAX `lax.slice`
jax.lax.slice.to_onnx = build_slice_onnx_node


def get_test_params() -> list:
    """Define test parameters for verifying the ONNX conversion of the Slice operation."""
    return [
        {
            "jax_component": "jax.lax.slice",
            "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.slice.html",
            "onnx": [
                {
                    "component": "Slice",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "slice_basic",
                    "input_shapes": [
                        (1, 5, 5, 3)
                    ],  # Example input shape (batch, height, width, channels)
                    "component": jax.lax.slice,
                    "params": {
                        "start": [0, 1, 1, 0],
                        "end": [-1, 4, 4, 3],
                    },  # Extracting a 3x3 region
                },
                {
                    "testcase": "slice_with_stride",
                    "input_shapes": [(1, 6, 6, 3)],
                    "component": jax.lax.slice,
                    "params": {
                        "start": [0, 0, 0, 0],
                        "end": [-1, 6, 6, 3],
                        "strides": [1, 2, 2, 1],
                    },  # Strided slice
                },
                {
                    "testcase": "slice_single_element",
                    "input_shapes": [(1, 3, 3)],  # Example 2D matrix
                    "component": jax.lax.slice,
                    "params": {
                        "start": [0, 1, 1],
                        "end": [-1, 2, 2],
                    },  # Extracting a single element
                },
                {
                    "testcase": "slice_last_column",
                    "input_shapes": [(1, 4, 5)],  # Example 2D matrix
                    "component": jax.lax.slice,
                    "params": {
                        "start": [0, 0, 4],
                        "end": [-1, 4, 5],
                    },  # Extracting the last column
                },
                {
                    "testcase": "slice_out_of_bounds",
                    "input_shapes": [(7, 4, 5)],
                    "component": jax.lax.slice,
                    "params": {"start": [0, 0, 3], "end": [-1, 4, 7]},  # Exceeds bounds
                },
            ],
        }
    ]
