import os

os.environ["XLA_CLIENT_PREALLOCATE"] = "false"
import jax
import equinox as eqx
from jax2onnx import to_onnx

use_bias = False


class EquinoxMLP(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        *,
        key,
        use_bias: bool = False,
    ):
        in_dims = [in_dim] + hidden_dims
        out_dims = hidden_dims + [out_dim]
        keys = jax.random.split(key, len(in_dims))
        self.layers = [
            eqx.nn.Linear(a, b, use_bias=use_bias, key=k)
            for (a, b, k) in zip(in_dims, out_dims, keys)
        ]

    def __call__(self, feat: jax.Array, key: jax.Array | None = None):
        x = feat
        for layer in self.layers:
            x = layer(x)
        return x


mlp = EquinoxMLP(
    128, [256, 256, 512, 512, 256, 256], 128, key=jax.random.key(1), use_bias=use_bias
)
to_onnx(mlp, [(100, 128)])
