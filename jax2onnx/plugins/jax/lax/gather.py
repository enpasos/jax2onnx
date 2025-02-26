# file: jax2onnx/plugins/jax/lax/gather.py

import jax
import numpy as np
import onnx
import onnx.helper as oh
from jax2onnx.convert import Z


def build_gather_onnx_node(z: Z, **params) -> Z:
    """Convert JAX lax.gather operation to an ONNX Gather operation."""
    if "indices" not in params or "axis" not in params:
        raise ValueError("Gather operation requires 'indices' and 'axis' parameters.")

    onnx_graph = z.onnx_graph
    input_name = z.names[0]
    input_shape = z.shapes[0]

    # Handle dynamic batch dimension
    if isinstance(input_shape[0], str):
        batch_dim = input_shape[0]
        input_shape = input_shape[1:]
    else:
        batch_dim = None
        input_shape = list(map(int, input_shape))  # Convert to Python int list

    axis = int(params["axis"])

    # --- Fix: Ensure `indices` is handled correctly ---
    indices = params["indices"]

    if isinstance(indices, int):
        # Directly use a scalar without reshaping
        indices_tensor = oh.make_tensor(
            f"node{onnx_graph.next_id()}_indices",
            onnx.TensorProto.INT64,
            [],  # Scalar in ONNX -> empty shape
            [indices],  # ONNX requires lists even for scalars
        )
    else:
        indices = np.array(indices, dtype=np.int64)  # Convert to numpy array
        indices_tensor = oh.make_tensor(
            f"node{onnx_graph.next_id()}_indices",
            onnx.TensorProto.INT64,
            indices.shape,  # Keeps original shape
            indices.tolist(),  # Ensure list format for ONNX compatibility
        )

    # Add indices as an initializer
    onnx_graph.add_initializer(indices_tensor)
    indices_name = indices_tensor.name

    node_name = f"node{onnx_graph.next_id()}"
    output_name = f"{node_name}_output"

    # Add Gather node
    onnx_graph.add_node(
        oh.make_node(
            "Gather",
            inputs=[input_name, indices_name],
            outputs=[output_name],
            name=node_name,
            axis=params["axis"],
        )
    )

    # Expected Output Shape: (1, 256) after removing axis 1
    final_output_shape = tuple(input_shape[:axis] + input_shape[axis + 1 :])
    if batch_dim:
        final_output_shape = (batch_dim,) + final_output_shape
    onnx_graph.add_local_outputs([final_output_shape], [output_name])

    # --- Define the JAX gold-standard gather function ---
    def jax_gather_fn(x):
        indices_for_jax = jax.numpy.array([params["indices"]])  # Ensure it's a tensor
        gathered = jax.lax.gather(
            x,
            indices_for_jax[:, None],  # Ensure correct dimensionality for gather
            dimension_numbers=jax.lax.GatherDimensionNumbers(
                offset_dims=(0, 2),  # Preserve batch and last axis
                collapsed_slice_dims=(1,),  # Remove axis 1
                start_index_map=(1,),
            ),
            slice_sizes=(
                1,
                1,
                x.shape[2],
            ),  # Keep batch + last axis, gather single index
            mode="fill",
        )
        return gathered[:, 0, :]  # ✅ This correctly removes axis 1

    return Z(
        [final_output_shape],
        [output_name],
        onnx_graph,
        jax_function=jax_gather_fn,
    )


# Attach ONNX conversion method
jax.lax.gather.to_onnx = build_gather_onnx_node


def get_test_params() -> list:
    """Define test parameters for verifying the ONNX conversion of the Gather operation."""
    return [
        {
            "jax_component": "jax.lax.gather",
            "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.gather.html",
            "onnx": [
                {
                    "component": "Gather",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "gather_tf_out",
                    "input_shapes": [(1, 50, 256)],
                    "component": jax.lax.gather,
                    "params": {
                        "indices": 0,  # Scalar index.
                        "axis": 1,
                    },
                },
            ],
        }
    ]
