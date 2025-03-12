# file: jax2onnx/examples/multi_head_attention.py

from flax import nnx
import jax
from flax.nnx import Linear, Dropout, BatchNorm


def get_metadata() -> dict:
    return {
        "component": "MultiHeadAttention",
        "description": "...",
        "since": "v0.2.0",
        "context": "examples.nnx",
        "testcases": [
            {
                "testcase": "multihead_attention",
                "callable": nnx.MultiHeadAttention(
                    num_heads=8,
                    in_features=256,
                    qkv_features=256,
                    out_features=256,
                    rngs=nnx.Rngs(0),
                    decode=False,
                ),
                "input_shapes": [(2, 4, 256)],
            }
        ],
    }
