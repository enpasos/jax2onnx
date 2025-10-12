# jax2onnx/sandbox/example_2.py

from flax import nnx
from jax2onnx import onnx_function, to_onnx


# just an @onnx_function decorator to make your callable an ONNX function
@onnx_function
class MLPBlock(nnx.Module):
    def __init__(self, dim, *, rngs):
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs)
        self.batchnorm = nnx.BatchNorm(dim, rngs=rngs)

    def __call__(self, x):
        return nnx.gelu(self.linear2(self.batchnorm(nnx.gelu(self.linear1(x)))))


# Use it inside another module
class MyModel(nnx.Module):
    def __init__(self, dim, *, rngs):
        self.block1 = MLPBlock(dim, rngs=rngs)
        self.block2 = MLPBlock(dim, rngs=rngs)

    def __call__(self, x):
        return self.block2(self.block1(x))


callable = MyModel(256, rngs=nnx.Rngs(0))
to_onnx(
    callable,
    [(100, 256)],
    return_mode="file",
    output_path="model_with_function.onnx",
)
