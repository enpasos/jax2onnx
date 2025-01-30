# file: jax2onnx/plugins/dropout.py

import onnx.helper as oh
import onnx
import numpy as np
from flax import nnx
from jax2onnx.onnx_export import OnnxGraph


def build_dropout_onnx_node(self, input_shapes, input_names, onnx_graph, parameters=None):
    """
    Constructs an ONNX node for a Dropout operation.

    Ensures that Dropout is correctly disabled in inference mode.
    """
    print("\n[DEBUG] --- Dropout ONNX Export ---")
    print(f"Input Shapes: {input_shapes}")
    print(f"Input Names: {input_names}")
    print(f"Dropout Rate: {self.rate}")

    node_name = f"node{onnx_graph.counter_plusplus()}"
    dropout_ratio_name = f"{node_name}_ratio"
    training_mode_name = f"{node_name}_training_mode"

    # Ensure dropout ratio is properly initialized
    onnx_graph.add_initializer(
        oh.make_tensor(
            dropout_ratio_name,
            onnx.TensorProto.FLOAT,
            [],  # Scalar value
            [self.rate],
        )
    )

    # Ensure training_mode is explicitly set to False (0) for inference
    # onnx_graph.add_initializer(
    #     oh.make_tensor(
    #         training_mode_name,
    #         onnx.TensorProto.BOOL,
    #         [],  # Scalar value
    #         [False],  # Explicitly disable Dropout
    #     )
    # )


    onnx_output_names = [f"{node_name}_output"]

    # Dropout takes three inputs: (data, ratio, training_mode)
    onnx_graph.add_node(
        oh.make_node(
            "Dropout",
            inputs=[input_names[0], dropout_ratio_name], #, training_mode_name],  # Explicit training_mode input
            outputs=onnx_output_names,
            name=node_name,
        )
    )

    print(f"[DEBUG] Added Dropout Node: {node_name} (Inference Mode)")
    print(f"[DEBUG] ONNX Graph Nodes: {len(onnx_graph.nodes)}")

    output_shapes = [input_shapes[0]]
    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)



    return output_shapes, onnx_output_names


# Attach the `build_dropout_onnx_node` method to nnx.Dropout
nnx.Dropout.build_onnx_node = build_dropout_onnx_node


def get_test_params():
    """
    Define test parameters for Dropout.
    """
    return [
        # Standard case (last working test)
        {
            "model_name": "dropout",
            "model": lambda: nnx.Dropout(rate=0.5, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 64, 64, 3)],  # JAX shape: (B, H, W, C)
            "build_onnx_node": nnx.Dropout.build_onnx_node,
            "export": {
                "pre_transpose": [(0, 3, 1, 2)],
                "post_transpose": [(0, 2, 3, 1)]
            }
        },
        # Dropout with a lower rate (0.1)
        {
            "model_name": "dropout_low",
            "model": lambda: nnx.Dropout(rate=0.1, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 32, 32, 3)],
            "build_onnx_node": nnx.Dropout.build_onnx_node,
        },
        # Dropout with a higher rate (0.9)
        {
            "model_name": "dropout_high",
            "model": lambda: nnx.Dropout(rate=0.9, rngs=nnx.Rngs(0)),
            "input_shapes": [(10, 32, 32, 3)],
            "build_onnx_node": nnx.Dropout.build_onnx_node,
        },
        # Dropout on a 1D input tensor
        {
            "model_name": "dropout_1d",
            "model": lambda: nnx.Dropout(rate=0.5, rngs=nnx.Rngs(0)),
            "input_shapes": [(10,)],
            "build_onnx_node": nnx.Dropout.build_onnx_node,
        },
        # Dropout on a 2D input tensor
        {
            "model_name": "dropout_2d",
            "model": lambda: nnx.Dropout(rate=0.5, rngs=nnx.Rngs(0)),
            "input_shapes": [(10, 20)],
            "build_onnx_node": nnx.Dropout.build_onnx_node,
        },
        # Dropout on a 4D input tensor
        {
            "model_name": "dropout_4d",
            "model": lambda: nnx.Dropout(rate=0.5, rngs=nnx.Rngs(0)),
            "input_shapes": [(10, 20, 30, 40)],
            "build_onnx_node": nnx.Dropout.build_onnx_node,
        },
    ]

