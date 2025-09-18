import jax
from flax import nnx

from jax2onnx.plugins2._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins2.plugin_system import register_example


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
    context="examples2.nnx",
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
            "callable": CNN(rngs=nnx.Rngs(0)),
            "input_shapes": [(3, 28, 28, 1)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(3, 10)],
            "post_check_onnx_graph": EG(
                [
                    (
                        "Transpose -> Conv -> Relu -> AveragePool -> Conv -> Relu -> AveragePool -> Transpose -> Reshape -> Gemm -> Relu -> Gemm",
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
                ]
            ),
            "use_onnx_ir": True,
        },
        {
            "testcase": "simple_cnn",
            "callable": CNN(rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 28, 28, 1)],
            "run_only_f32_variant": True,
            "run_only_dynamic": True,
            "expected_output_shapes": [("B", 10)],
            "use_onnx_ir": True,
        },
    ],
)
