# file: jax2onnx/plugins/jax/lax/slice.py


import jax
import numpy as np
import onnx
import onnx.helper as oh

from jax2onnx.convert import Z


def build_slice_onnx_node(z: Z, **params) -> Z:
    """Convert JAX lax.slice operation to ONNX Slice operation."""
    if "start" not in params or "end" not in params:
        raise ValueError("Slice operation requires 'start' and 'end' parameters.")

    onnx_graph = z.onnx_graph
    input_name = z.names[0]
    input_shape = list(map(int, z.shapes[0]))  # Convert to Python int list

    start = list(map(int, params["start"]))  # Convert to Python int list
    end = list(map(int, params["end"]))
    strides = list(map(int, params.get("strides", [1] * len(start))))  # Default step=1

    # ✅ Ensure `end` is within input shape (avoid out-of-bounds slicing)
    end = [min(e, input_shape[i]) for i, e in enumerate(end)]

    # ✅ Ensure `start` is within input shape
    start = [max(0, s) for s in start]

    node_name = f"node{onnx_graph.next_id()}"
    output_name = f"{node_name}_output"

    # ✅ Create ONNX slice parameters as initializers
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
            np.arange(len(start), dtype=np.int64),  # Axes are sequential
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

    # ✅ Add Slice node to ONNX graph
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

    # ✅ Compute output shape dynamically
    output_shape = [
        max(0, (e - s + (step - 1)) // step)  # Ensures no negative dims
        for s, e, step in zip(start, end, strides)
    ]
    onnx_graph.add_local_outputs([tuple(output_shape)], [output_name])

    # ✅ Define JAX slice function
    def jax_slice_fn(x):
        return jax.lax.slice(x, start_indices=start, limit_indices=end, strides=strides)

    return Z(
        [tuple(output_shape)], [output_name], onnx_graph, jax_function=jax_slice_fn
    )


# ✅ Attach ONNX conversion method to JAX `lax.slice`
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
                        "end": [1, 4, 4, 3],
                    },  # Extracting a 3x3 region
                },
                {
                    "testcase": "slice_with_stride",
                    "input_shapes": [(1, 6, 6, 3)],
                    "component": jax.lax.slice,
                    "params": {
                        "start": [0, 0, 0, 0],
                        "end": [1, 6, 6, 3],
                        "strides": [1, 2, 2, 1],
                    },  # Strided slice
                },
                {
                    "testcase": "slice_single_element",
                    "input_shapes": [(3, 3)],  # Example 2D matrix
                    "component": jax.lax.slice,
                    "params": {
                        "start": [1, 1],
                        "end": [2, 2],
                    },  # Extracting a single element
                },
                {
                    "testcase": "slice_last_column",
                    "input_shapes": [(4, 5)],  # Example 2D matrix
                    "component": jax.lax.slice,
                    "params": {
                        "start": [0, 4],
                        "end": [4, 5],
                    },  # Extracting the last column
                },
                {
                    "testcase": "slice_out_of_bounds",
                    "input_shapes": [(4, 5)],
                    "component": jax.lax.slice,
                    "params": {"start": [0, 3], "end": [4, 7]},  # Exceeds bounds
                },
            ],
        }
    ]
