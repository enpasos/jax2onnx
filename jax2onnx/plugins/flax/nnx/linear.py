# file: jax2onnx/plugins/linear.py


import numpy as np
import onnx
import onnx.helper as oh
from flax import nnx

from jax2onnx.to_onnx import Z
from jax2onnx.typing_helpers import Supports2Onnx


def to_onnx(self: Supports2Onnx, z: Z, **params) -> Z:
    """
    Converts an `nnx.Linear` layer into an ONNX `Gemm` (General Matrix Multiplication) node.

    This function adds the corresponding weight and bias initializers to the ONNX graph.

    Args:
        self: The `nnx.Linear` instance.
        z (Z): Contains input shapes, names, and the ONNX graph.
        **params: Additional conversion parameters.

    Returns:
        Z: Updated instance with new shapes and names.
    """
    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]
    out_features = self.kernel.shape[1]

    # Determine if reshaping is necessary
    if len(input_shape) > 2:
        new_first_dim = int(np.prod(input_shape[:-1]))
        flattened_shape = (new_first_dim, input_shape[-1])
        reshape_input_name = f"{input_name}_reshaped"

        # Add reshape node
        onnx_graph.add_node(
            oh.make_node(
                "Reshape",
                inputs=[input_name, f"{reshape_input_name}_shape"],
                outputs=[reshape_input_name],
                name=f"reshape_before_{input_name}",
            )
        )

        # Store reshape shape as an initializer
        onnx_graph.add_initializer(
            oh.make_tensor(
                f"{reshape_input_name}_shape",
                onnx.TensorProto.INT64,
                [2],
                np.array(flattened_shape, dtype=np.int64),
            )
        )
        onnx_graph.add_local_outputs([list(flattened_shape)], [reshape_input_name])
    else:
        reshape_input_name = input_name
        flattened_shape = input_shape  # Ensure it's defined for later use

    # Output shape derivation
    output_shape = list(input_shape[:-1]) + [out_features]
    node_name = f"node{onnx_graph.next_id()}"

    # Define ONNX node using the Gemm operator
    gemm_output_name = f"{node_name}_output"
    onnx_graph.add_node(
        oh.make_node(
            "Gemm",
            inputs=[reshape_input_name, f"{node_name}_weight", f"{node_name}_bias"],
            outputs=[gemm_output_name],
            name=node_name,
        )
    )
    onnx_graph.add_local_outputs(
        [[flattened_shape[0], out_features]], [gemm_output_name]
    )

    # Add weight matrix as an ONNX initializer
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node_name}_weight",
            onnx.TensorProto.FLOAT,
            self.kernel.shape,
            self.kernel.value.reshape(-1).astype(np.float32),
        )
    )

    # Add bias vector as an ONNX initializer
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node_name}_bias",
            onnx.TensorProto.FLOAT,
            [out_features],
            self.bias.value.astype(np.float32),
        )
    )

    # Reshape back to the original input shape with last dimension changed to out_features
    if len(input_shape) > 2:
        final_output_name = f"{gemm_output_name}_reshaped"
        onnx_graph.add_node(
            oh.make_node(
                "Reshape",
                inputs=[gemm_output_name, f"{final_output_name}_shape"],
                outputs=[final_output_name],
                name=f"reshape_after_{gemm_output_name}",
            )
        )
        onnx_graph.add_initializer(
            oh.make_tensor(
                f"{final_output_name}_shape",
                onnx.TensorProto.INT64,
                [len(output_shape)],
                np.array(output_shape, dtype=np.int64),
            )
        )
        onnx_graph.add_local_outputs([output_shape], [final_output_name])
    else:
        final_output_name = gemm_output_name

    # Register the output tensor in the ONNX graph
    onnx_graph.add_local_outputs([output_shape], [final_output_name])

    return Z([output_shape], [final_output_name], onnx_graph)


# Attach the `to_onnx` method to `nnx.Linear`
nnx.Linear.to_onnx = to_onnx


def get_test_params():
    return [
        {
            "jax_component": "flax.nnx.Linear",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Linear",
            "onnx": [
                {
                    "component": "Gemm",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "linear",
                    "component": nnx.Linear(
                        5, 3, rngs=nnx.Rngs(0)
                    ),  # Linear layer with input dim 5, output dim 3
                    "input_shapes": [
                        (1, 5)
                    ],  # Example input shape (batch_size=1, input_dim=5)
                },
                {
                    "testcase": "linear_2",
                    "component": nnx.Linear(
                        256, 512, rngs=nnx.Rngs(0)
                    ),  # Linear layer with input dim 256, output dim 512
                    "input_shapes": [
                        (1, 10, 256)
                    ],  # Batched input for testing reshape handling
                },
            ],
        }
    ]
