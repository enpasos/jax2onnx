from jax2onnx import save_onnx
from flax import nnx


# Example: A minimal MLP (from Flax documentation)
class MLP(nnx.Module):
    def __init__(self, din, dmid, dout, *, rngs):
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x):
        x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
        return self.linear2(x)


# Instantiate model
my_callable = MLP(din=30, dmid=20, dout=10, rngs=nnx.Rngs(0))

# Convert and save to ONNX
save_onnx(
    my_callable,
    [("B", 30)],  # Input shapes, batch size 'B' is symbolic
    "my_callable.onnx",  # Output path
)
