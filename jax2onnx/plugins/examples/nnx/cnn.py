# jax2onnx/plugins/examples/nnx/cnn.py

import jax
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)


class CNN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


register_example(
    component="CNN",
    description="A simple convolutional neural network (CNN).",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.2.0",
    context="examples.nnx",
    children=[
        "nnx.Conv",
        "nnx.Linear",
        "nnx.avg_pool",
        "nnx.relu",
        "lax.reshape",
    ],
    testcases=[
        {
            "testcase": "simple_cnn_static",
            "callable": construct_and_call(CNN, rngs=with_rng_seed(0)),
            "input_shapes": [(3, 28, 28, 1)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(3, 10)],
            "post_check_onnx_graph": EG(
                [
                    (
                        "Transpose:3x1x28x28 -> Conv:3x32x28x28 -> Relu:3x32x28x28 -> "
                        "AveragePool:3x32x14x14 -> Conv:3x64x14x14 -> Relu:3x64x14x14 -> "
                        "AveragePool:3x64x7x7 -> Transpose:3x7x7x64 -> Reshape:3x3136 -> "
                        "Gemm:3x256 -> Relu:3x256 -> Gemm:3x10",
                        {
                            "counts": {
                                "Transpose": 2,
                                "Conv": 2,
                                "Relu": 3,
                                "AveragePool": 2,
                                "Reshape": 1,
                                "Gemm": 2,
                            }
                        },
                    ),
                ],
                no_unused_inputs=True,
                mode="all",
            ),
        },
        {
            "testcase": "simple_cnn",
            "callable": construct_and_call(CNN, rngs=with_rng_seed(0)),
            "input_shapes": [("B", 28, 28, 1)],
            "run_only_f32_variant": True,
            "run_only_dynamic": True,
            "expected_output_shapes": [("B", 10)],
            "post_check_onnx_graph": EG(
                [
                    (
                        "Transpose:Bx1x28x28 -> Conv:Bx32x28x28 -> Relu:Bx32x28x28 -> "
                        "AveragePool:Bx32x14x14 -> Conv:Bx64x14x14 -> Relu:Bx64x14x14 -> "
                        "AveragePool:Bx64x7x7 -> Transpose:Bx7x7x64 -> Reshape:Bx3136 -> "
                        "Gemm:Bx256 -> Relu:Bx256 -> Gemm:Bx10",
                        {
                            "counts": {
                                "Transpose": 2,
                                "Conv": 2,
                                "Relu": 3,
                                "AveragePool": 2,
                                "Reshape": 1,
                                "Gemm": 2,
                                "Shape": 1,
                                "Gather": 1,
                                "Unsqueeze": 1,
                                "Concat": 1,
                            }
                        },
                    ),
                ],
                no_unused_inputs=True,
                mode="all",
            ),
        },
    ],
)
