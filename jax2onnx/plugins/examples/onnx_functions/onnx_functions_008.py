# file: jax2onnx/plugins/examples/onnx_functions/onnx_functions_008.py


from flax import nnx
import jax.numpy as jnp

from jax2onnx.plugin_system import onnx_function, register_example


# === Renamed ===
@onnx_function
class FeedForwardA(nnx.Module):
    """MLP block for Transformer layers."""

    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
            lambda x: nnx.gelu(x, approximate=False),
            nnx.Dropout(rate=0.1, rngs=rngs),
            nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
            nnx.Dropout(rate=0.1, rngs=rngs),
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        deterministic = True
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x


# === Renamed ===
@onnx_function
class AttentionA:  # Renamed from Attention008
    # Assuming this is intended to wrap the primitive/function
    def __call__(self, *args, **kwargs):
        return nnx.dot_product_attention(*args, **kwargs)


# === Renamed ===
@onnx_function
class MultiHeadAttentionA(nnx.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_heads: int,
        attention_dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            qkv_features=num_hiddens,
            out_features=num_hiddens,
            in_features=num_hiddens,
            # === Updated internal reference ===
            attention_fn=AttentionA(),  # Use renamed class
            rngs=rngs,
            decode=False,
        )
        self.dropout = nnx.Dropout(rate=attention_dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.attention(x)
        x = self.dropout(x, deterministic=True)
        return x


# === Renamed ===
@onnx_function
class TransformerBlockA(nnx.Module):
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
        self.layer_norm1 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        # === Updated internal reference ===
        self.attention = MultiHeadAttentionA(
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            attention_dropout_rate=attention_dropout_rate,
            rngs=rngs,
        )
        self.layer_norm2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        # === Updated internal reference ===
        self.mlp_block = FeedForwardA(num_hiddens, mlp_dim, mlp_dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        r = self.layer_norm1(x)
        r = self.attention(r)
        x = x + r
        r = self.layer_norm2(x)
        return x + self.mlp_block(r)


register_example(
    component="onnx_functions_008",  # Keep component name matching file
    description="transformer block with nested mlp block no call parameter",
    since="v0.4.0",
    context="examples.onnx_functions",
    # === Updated children names ===
    children=[
        "FeedForwardA",
        "MultiHeadAttentionA",
    ],  # Note: FeedForwardBlock008 was likely a typo
    testcases=[
        {
            "testcase": "008_transformer_block",
            # === Updated callable name ===
            "callable": TransformerBlockA(
                num_hiddens=256,
                num_heads=8,
                mlp_dim=512,
                attention_dropout_rate=0.1,
                mlp_dropout_rate=0.1,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 10, 256)],
            "expected_number_of_function_instances": 4,
        },
    ],
)
