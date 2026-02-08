# jax2onnx/sandbox/issue_173/export_onnx.py

import dm_pix as pix
import jax.numpy as jnp
from flax import nnx
from jax2onnx import to_onnx

# Settings
filters = 64
blocks = 4
groups = 4
kernel_size = (3, 3)


class DepthToSpace(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.block_size = 2

    def __call__(self, input):
        x = pix.depth_to_space(input, self.block_size)
        return x


class ResGroup(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.res_blocks = nnx.List(ResBlock(rngs=rngs) for _ in range(blocks))
        self.conv0 = nnx.Conv(filters, filters, kernel_size=kernel_size, rngs=rngs)

    def __call__(self, input):
        x = input
        for block in self.res_blocks:
            x = block(x)
        x = self.conv0(x)
        x = x + input
        return x


class ResBlock(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(filters, filters, kernel_size=kernel_size, rngs=rngs)
        self.conv1 = nnx.Conv(filters, filters, kernel_size=kernel_size, rngs=rngs)

    def __call__(self, input):
        x = nnx.silu(self.conv0(input))
        x = self.conv1(x)
        x = x + input
        return x


class ResGroupModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(1, filters, kernel_size=kernel_size, rngs=rngs)
        self.res_groups = nnx.List(ResGroup(rngs=rngs) for _ in range(groups))
        self.conv1 = nnx.Conv(filters, filters, kernel_size=kernel_size, rngs=rngs)
        self.feats_conv = nnx.Conv(filters, 4, kernel_size=kernel_size, rngs=rngs)
        self.depth_to_space = DepthToSpace(rngs=rngs)

    def __call__(self, input):
        conv0 = self.conv0(input)
        x = conv0
        for group in self.res_groups:
            x = group(x)
        conv1 = self.conv1(x)
        x = conv1 + conv0
        x = self.feats_conv(x)
        x = self.depth_to_space(x)
        x = jnp.clip(x, 0.0, 1.0)
        return x


model = ResGroupModel(rngs=nnx.Rngs(0))
print(nnx.tabulate(model, jnp.ones((1, 256, 256, 1))))
to_onnx(
    model,
    [(1, 256, 256, 1)],
    return_mode="file",
    output_path="model.onnx",
    inputs_as_nchw=[0],
    outputs_as_nchw=[0],
)
