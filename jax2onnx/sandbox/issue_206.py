# jax2onnx/sandbox/issue_206.py

from jax2onnx import to_onnx
from flax import nnx

reflect_conv = nnx.Conv(
    in_features=1,
    out_features=1,
    kernel_size=3,
    padding="REFLECT",
    rngs=nnx.Rngs(0),
)

to_onnx(
    fn=lambda x: reflect_conv(x),
    inputs=[("B", 128, 1)],
)
