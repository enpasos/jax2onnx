# jax2onnx/plugins/examples/linen/cnn.py

from __future__ import annotations

from flax import linen as nn

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)


class LinenCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


register_example(
    component="LinenCNN",
    description="A simple convolutional neural network (CNN).",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.11.0",
    context="examples.linen",
    children=[
        "flax.linen.Conv",
        "flax.linen.Dense",
        "flax.linen.avg_pool",
        "flax.linen.relu",
        "jax.numpy.reshape",
    ],
    testcases=[
        {
            "testcase": "simple_cnn_static",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=LinenCNN,
                input_shape=(1, 28, 28, 1),
                rngs=with_rng_seed(0),
            ),
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
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=LinenCNN,
                input_shape=(1, 28, 28, 1),
                rngs=with_rng_seed(0),
            ),
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
