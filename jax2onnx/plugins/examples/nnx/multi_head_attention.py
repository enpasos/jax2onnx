# file: jax2onnx/examples/multi_head_attention.py

from flax import nnx

from jax2onnx.plugin_system import register_example

register_example(
    component="MultiHeadAttention",
    description="This is a multi-head attention module implemented by Flax/nnx that has no ONNX correspondent on the same granularity.",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.2.0",
    context="examples.nnx",
    children=["nnx.GeneralLinear", "nnx.dot_product_attention"],
    testcases=[
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
)
