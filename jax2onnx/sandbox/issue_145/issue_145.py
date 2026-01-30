# jax2onnx/sandbox/issue_145/issue_145.py

import dm_pix as pix
from flax import nnx
from jax2onnx import to_onnx


class DepthToSpace(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.block_size = 2

    def __call__(self, input):
        x = pix.depth_to_space(input, self.block_size)
        return x


class ResBlock(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(32, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv1 = nnx.Conv(32, 32, kernel_size=(3, 3), rngs=rngs)

    def __call__(self, input):
        x = nnx.silu(self.conv0(input))
        x = self.conv1(x)
        return x + input


class Model(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.res_blocks = nnx.List(ResBlock(rngs=rngs) for _ in range(4))
        self.conv1 = nnx.Conv(32, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 4, kernel_size=(3, 3), rngs=rngs)
        self.depth_to_space = DepthToSpace(rngs=rngs)

    def __call__(self, input):
        conv0 = self.conv0(input)
        x = conv0
        for block in self.res_blocks:
            x = block(x)
        conv1 = self.conv1(x)
        x = conv1 + conv0
        x = self.conv2(x)
        x = self.depth_to_space(x)
        return x


model = Model(rngs=nnx.Rngs(0))
to_onnx(
    model,
    [("N", "H", "W", 1)],
    return_mode="file",
    output_path="model.onnx",
    inputs_as_nchw=[0],
    outputs_as_nchw=[0],
)
