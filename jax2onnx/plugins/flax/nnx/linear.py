# file: jax2onnx/plugins/linear.py

from flax import nnx
from jax2onnx.convert import Z
from jax2onnx.typing_helpers import Supports2Onnx
from jax2onnx.plugins.flax.nnx.linear_general import build_linear_general_onnx_node


def to_onnx(self: Supports2Onnx, z: Z, **params) -> Z:
    """Convert an `nnx.Linear` layer into an ONNX `Gemm` node."""
    return build_linear_general_onnx_node(self, z, **params)


# Attach `to_onnx` method to `nnx.Linear`
nnx.Linear.to_onnx = to_onnx


def get_test_params() -> list:
    """Return test parameters for verifying the ONNX conversion of `nnx.Linear`."""
    return [
        {
            "jax_component": "flax.nnx.Linear",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Linear",
            "onnx": [
                {
                    "component": "Gemm",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html",
                },
                {
                    "component": "MatMul",
                    "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "linear",
                    "component": nnx.Linear(
                        in_features=128,
                        out_features=64,
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(32, 128)],
                },
                {
                    "testcase": "linear_2d",
                    "component": nnx.Linear(
                        in_features=128,
                        out_features=64,
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(32, 10, 128)],
                },
            ],
        }
    ]
