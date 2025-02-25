import numpy as np
import onnx
import onnx.helper as oh
from flax import nnx
from typing import Any, cast

from jax2onnx.convert import Z, OnnxGraph
from jax2onnx.typing_helpers import Supports2Onnx


def to_onnx(self: Supports2Onnx, z: Z, **params: Any) -> Z:
    """Converts `nnx.BatchNorm` into an ONNX `BatchNormalization` node."""
    onnx_graph : OnnxGraph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]

    node_name = f"node{onnx_graph.next_id()}"
    epsilon = getattr(self, "epsilon", 1e-5)
    momentum = 1 - getattr(self, "momentum", 0.9)

    self = cast(nnx.BatchNorm, self)
    scale, bias, mean, var = (
        self.scale.value,
        self.bias.value,
        self.mean.value,
        self.var.value,
    )

    scale_name, bias_name = f"{node_name}_scale", f"{node_name}_bias"
    mean_name, var_name = f"{node_name}_mean", f"{node_name}_variance"

    for name, tensor in zip(
        [scale_name, bias_name, mean_name, var_name], [scale, bias, mean, var]
    ):
        onnx_graph.add_initializer(
            oh.make_tensor(
                name,
                onnx.TensorProto.FLOAT,
                tensor.shape,
                tensor.flatten().astype(np.float32),
            )
        )

    onnx_output_names = [f"{node_name}_output"]

    onnx_graph.add_node(
        oh.make_node(
            "BatchNormalization",
            inputs=[input_name, scale_name, bias_name, mean_name, var_name],
            outputs=onnx_output_names,
            name=node_name,
            epsilon=epsilon,
            momentum=momentum,
        )
    )

    output_shapes = [input_shape]
    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)

    return Z(output_shapes, onnx_output_names, onnx_graph)


# Attach ONNX conversion method to `nnx.BatchNorm`
nnx.BatchNorm.to_onnx = to_onnx


def get_test_params():
    return [
        {
            "jax_component": "flax.nnx.BatchNorm",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm",
            "onnx": [
                {
                    "component": "BatchNormalization",
                    "doc": "https://onnx.ai/onnx/operators/onnx__BatchNormalization.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "batchnorm",
                    "component": nnx.BatchNorm(
                        num_features=64, epsilon=1e-5, momentum=0.9, rngs=nnx.Rngs(0)
                    ),
                    "input_shapes": [(11, 2, 2, 64)],  # (B, H, W, C) format
                    "params": {
                        "pre_transpose": [
                            (0, 3, 1, 2)
                        ],  # Convert JAX (B, H, W, C) → ONNX (B, C, H, W)
                        "post_transpose": [
                            (0, 2, 3, 1)
                        ],  # Convert ONNX back to JAX format
                    },
                },
            ],
        }
    ]
