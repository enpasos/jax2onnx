# file: jax2onnx/plugins/examples/onnx_functions/onnx_functions_007.py


from flax import nnx
import jax.numpy as jnp

from jax2onnx.plugin_system import onnx_function, register_example


@onnx_function
class MLPBlock007(nnx.Module):
    """MLP block for Transformer layers."""

    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
            lambda x: nnx.gelu(x, approximate=False),
            nnx.Dropout(rate=0.1, rngs=rngs),
            nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
            nnx.Dropout(rate=0.1, rngs=rngs),
        ]

    #    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        deterministic = True
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x


@onnx_function
class TransformerBlock007(nnx.Module):
    """Transformer block with multi-head attention and MLP."""

    def __init__(
        self,
        num_hiddens: int,
        num_heads: int,
        mlp_dim: int,
        attention_dropout_rate: float = 0.1,
        mlp_dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.rng_collection = rngs
        self.layer_norm1 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            qkv_features=num_hiddens,
            out_features=num_hiddens,
            in_features=num_hiddens,
            rngs=rngs,
            decode=False,
        )
        self.layer_norm2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.mlp_block = MLPBlock007(num_hiddens, mlp_dim, mlp_dropout_rate, rngs=rngs)
        self.dropout = nnx.Dropout(rate=attention_dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        y = self.attention(self.layer_norm1(x))
        y = self.dropout(y, deterministic=deterministic)
        x = x + y
        # return x + self.mlp_block(self.layer_norm2(x), deterministic)
        return x + self.mlp_block(self.layer_norm2(x))


register_example(
    component="onnx_functions_007",
    description="transformer block with nested mlp block",
    # source="https:/",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=["MLPBlock007"],
    testcases=[
        {
            "testcase": "007_transformer_block",
            "callable": TransformerBlock007(
                num_hiddens=256,
                num_heads=8,
                mlp_dim=512,
                attention_dropout_rate=0.1,
                mlp_dropout_rate=0.1,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 10, 256)],
        },
    ],
)
