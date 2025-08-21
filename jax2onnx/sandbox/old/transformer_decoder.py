import os
import numpy as np
import jax
import jax.numpy as jnp
import onnx
import onnxruntime as ort
from flax import nnx
from jax2onnx import to_onnx

# FIX: Import and call `import_all_plugins` to load the necessary
# conversion logic for complex nnx modules like MultiHeadAttention.
from jax2onnx.plugin_system import import_all_plugins

import_all_plugins()


# The model definitions are correct and do not need to change.
class TransformerDecoderLayer(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        rngs: nnx.Rngs,
        rate: float = 0.1,
        attention_dropout: float = 0.0,
        encoder_attention_dropout: float = 0.0,
        allow_residue: bool = True,
    ):
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            dropout_rate=attention_dropout,
            decode=False,
            rngs=rngs,
        )
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            dropout_rate=encoder_attention_dropout,
            decode=False,
            rngs=rngs,
        )
        self.ffn = nnx.Sequential(
            nnx.Linear(in_features=embed_dim, out_features=ff_dim, rngs=rngs),
            lambda x: nnx.relu(x),
            nnx.Linear(in_features=ff_dim, out_features=embed_dim, rngs=rngs),
        )
        self.layernorm1 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.layernorm2 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.layernorm3 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=rate, rngs=rngs)
        self.allow_residue = allow_residue

    def __call__(
        self,
        x: jax.Array,
        encoder_output: jax.Array,
        mask: jax.Array | None = None,
        cross_attn_mask: jax.Array | None = None,
        *,
        deterministic: bool = True,
        decode=None,
    ) -> jax.Array:
        attn_output = self.self_attn(
            inputs_q=x, mask=mask, deterministic=deterministic, decode=decode
        )
        attn_output = self.dropout1(attn_output, deterministic=deterministic)
        x_resid = (x + attn_output) if self.allow_residue else attn_output
        x = self.layernorm1(x_resid)

        cross_attn_output = self.cross_attn(
            inputs_q=x,
            inputs_k=encoder_output,
            mask=cross_attn_mask,
            deterministic=deterministic,
        )
        x = self.layernorm2(
            x + self.dropout2(cross_attn_output, deterministic=deterministic)
        )

        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout3(ffn_output, deterministic=deterministic))
        return x


class TransformerDecoder(nnx.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        rngs: nnx.Rngs,
        rate: float = 0.1,
        attention_dropout: float = 0.0,
        encoder_attention_dropout: float = 0.0,
        allow_residue: bool = True,
        deterministic: bool = True,
    ):
        self.deterministic = deterministic
        self.layers = [
            TransformerDecoderLayer(
                embed_dim,
                num_heads,
                ff_dim,
                rngs=rngs,
                rate=rate,
                attention_dropout=attention_dropout,
                encoder_attention_dropout=encoder_attention_dropout,
                allow_residue=allow_residue,
            )
            for _ in range(num_layers)
        ]

    def __call__(self, x: jax.Array, encoder_output: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x, encoder_output, deterministic=self.deterministic)
        return x


# --- Initialize the model with dummy input ---
rngs = nnx.Rngs(0, params=42, dropout=1)

model = TransformerDecoder(
    num_layers=1,
    embed_dim=16,
    num_heads=4,
    ff_dim=32,
    rngs=rngs,
    attention_dropout=0.5,
    encoder_attention_dropout=0.5,
)

# --- Convert to ONNX ---
onnx_model = to_onnx(model, [("B", 8, 16), ("B", 4, 16)])


# --- Save the ONNX model ---
out_dir = "docs/onnx"
os.makedirs(out_dir, exist_ok=True)
onnx_path = os.path.join(out_dir, "transformer_decoder.onnx")
onnx.save_model(onnx_model, onnx_path)
print(f"✅ Saved ONNX model to {onnx_path}")

# --- Verify outputs match ---

# ⚠️ both inputs must be rank-3 (B, seq_len, emb_dim) to match the ONNX signature:
decoder_input = jnp.ones((1, 8, 16), dtype=jnp.float32)  # (B=1, seq=8, emb=16)
encoder_output = jnp.ones((1, 4, 16), dtype=jnp.float32)  # (B=1, seq=4, emb=16)

jax_out = model(decoder_input, encoder_output)
sess = ort.InferenceSession(onnx_path)
inp_names = [i.name for i in sess.get_inputs()]
onnx_out = sess.run(
    None,
    {
        inp_names[0]: np.array(decoder_input),
        inp_names[1]: np.array(encoder_output),
    },
)[0]

np.testing.assert_allclose(jax_out, onnx_out, rtol=1e-5, atol=1e-5)
print("✅ JAX vs ONNX output match confirmed.")
