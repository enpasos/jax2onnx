# file: jax2onnx/plugins/dropout.py

import onnx
import onnx.helper as oh
from flax import nnx

from jax2onnx.to_onnx import Z
from jax2onnx.typing_helpers import Supports2Onnx


def build_dropout_onnx_node(self: Supports2Onnx, z: Z, **params) -> Z:
    """
    Converts an `nnx.Dropout` layer into an ONNX `Dropout` node.

    Ensures that Dropout is correctly disabled in inference mode.

    Args:
        self: The `nnx.Dropout` instance.
        z (Z): Contains input shapes, names, and the ONNX graph.
        **params: Additional conversion parameters.

    Returns:
        Z: Updated instance with new shapes and names.
    """

    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]

    node_name = f"node{onnx_graph.next_id()}"
    dropout_ratio_name = f"{node_name}_ratio"

    # Create an initializer for dropout ratio
    onnx_graph.add_initializer(
        oh.make_tensor(
            dropout_ratio_name,
            onnx.TensorProto.FLOAT,
            [],  # Scalar value
            [self.rate],
        )
    )

    output_names = [f"{node_name}_output"]

    # Dropout takes (data, ratio) as inputs in inference mode
    onnx_graph.add_node(
        oh.make_node(
            "Dropout",
            inputs=[input_name, dropout_ratio_name],
            outputs=output_names,
            name=node_name,
        )
    )

    output_shapes = [input_shape]
    onnx_graph.add_local_outputs(output_shapes, output_names)

    # Update and return Z
    z.shapes = output_shapes
    z.names = output_names
    z.jax_function = self
    return z


# Attach the ONNX conversion method to `nnx.Dropout`
nnx.Dropout.to_onnx = build_dropout_onnx_node


def get_test_params():
    """
    Define test parameters for Dropout.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "jax_component": "flax.nnx.Dropout",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/stochastic.html#flax.nnx.Dropout",
            "onnx": [
                {
                    "component": "Dropout",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Dropout.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "dropout",
                    "component": nnx.Dropout(rate=0.5, rngs=nnx.Rngs(0)),
                    "input_shapes": [(1, 64, 64, 3)],  # JAX shape: (B, H, W, C)
                    "params": {
                        "pre_transpose": [(0, 3, 1, 2)],
                        "post_transpose": [(0, 2, 3, 1)],
                    },
                },
                {
                    "testcase": "dropout_low",
                    "component": nnx.Dropout(rate=0.1, rngs=nnx.Rngs(0)),
                    "input_shapes": [(1, 32, 32, 3)],
                },
                {
                    "testcase": "dropout_high",
                    "component": nnx.Dropout(rate=0.9, rngs=nnx.Rngs(0)),
                    "input_shapes": [(10, 32, 32, 3)],
                },
                {
                    "testcase": "dropout_1d",
                    "component": nnx.Dropout(rate=0.5, rngs=nnx.Rngs(0)),
                    "input_shapes": [(10,)],
                },
                {
                    "testcase": "dropout_2d",
                    "component": nnx.Dropout(rate=0.5, rngs=nnx.Rngs(0)),
                    "input_shapes": [(10, 20)],
                },
                {
                    "testcase": "dropout_3d",
                    "component": nnx.Dropout(rate=0.1, rngs=nnx.Rngs(17)),
                    "input_shapes": [(1, 10, 512)],
                },
                {
                    "testcase": "dropout_4d",
                    "component": nnx.Dropout(rate=0.5, rngs=nnx.Rngs(0)),
                    "input_shapes": [(10, 20, 30, 40)],
                },
            ],
        }
    ]
