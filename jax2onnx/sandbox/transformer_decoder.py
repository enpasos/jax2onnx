import os
import numpy as np
import jax
import jax.numpy as jnp
import onnxruntime as ort
from flax import nnx
from jax2onnx import to_onnx 

class TransformerDecoderLayer(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs, rate: float = 0.1, attention_dropout: float = 0.0, encoder_attention_dropout: float = 0.0, allow_residue: bool = True):
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            dropout_rate=attention_dropout,
            decode=False,
            rngs=rngs
        )
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            dropout_rate=encoder_attention_dropout,
            decode=False,
            rngs=rngs
        )
        # --- FIX: Replace nnx.Sequential with explicit layers ---
        self.ffn1 = nnx.Linear(in_features=embed_dim, out_features=ff_dim, rngs=rngs)
        self.ffn2 = nnx.Linear(in_features=ff_dim, out_features=embed_dim, rngs=rngs)
        # --------------------------------------------------------
        self.layernorm1 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.layernorm2 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.layernorm3 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=rate, rngs=rngs)
        self.allow_residue = allow_residue

    def __call__(self,
                 x: jax.Array,
                 encoder_output: jax.Array,
                 mask: jax.Array | None = None,
                 cross_attn_mask: jax.Array | None = None,
                 *, deterministic: bool = False, decode=None) -> jax.Array:
        # Self-Attention Block
        attn_output = self.self_attn(inputs_q=x, mask=mask, deterministic=deterministic, decode=decode)
        attn_output = self.dropout1(attn_output, deterministic=deterministic)
        x_resid = (x + attn_output) if self.allow_residue else attn_output
        x = self.layernorm1(x_resid)

        # Cross-Attention Block
        cross_attn_output = self.cross_attn(inputs_q=x, inputs_k=encoder_output, mask=cross_attn_mask, deterministic=deterministic)
        cross_resid = x + self.dropout2(cross_attn_output, deterministic=deterministic)
        x = self.layernorm2(cross_resid)

        # --- FIX: Use explicit FFN layers in the forward pass ---
        ffn_out = self.ffn1(x)
        ffn_out = nnx.relu(ffn_out)
        ffn_out = self.ffn2(ffn_out)
        # ------------------------------------------------------

        ffn_resid = x + self.dropout3(ffn_out, deterministic=deterministic)
        x = self.layernorm3(ffn_resid)
        return x


class TransformerDecoder(nnx.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs, rate: float = 0.1, attention_dropout: float = 0.0, encoder_attention_dropout: float = 0.0, allow_residue: bool = True):
        self.layers = [
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, rngs=rngs, rate=rate, attention_dropout=attention_dropout, encoder_attention_dropout=encoder_attention_dropout, allow_residue=allow_residue)
            for _ in range(num_layers)
        ]

    def __call__(self,
                 x: jax.Array,
                 encoder_output: jax.Array,
                 mask: jax.Array | None = None,
                 cross_attn_mask: jax.Array | None = None,
                 *, deterministic: bool = False, decode=None) -> jax.Array:
        for layer in self.layers:
            x = layer(x, encoder_output, mask, cross_attn_mask, deterministic=deterministic, decode=decode)
        return x


# --- Initialize the model with dummy input ---
rngs = nnx.Rngs(0, params=42, dropout=1)

model = TransformerDecoder(
    num_layers=1,
    embed_dim=16,
    num_heads=4,
    ff_dim=32,
    rngs=rngs,
    attention_dropout=0.0,
    encoder_attention_dropout=0.0
)

# Dummy input (decoder sequence length is 8)
decoder_input = jnp.zeros((1, 8, 16), dtype=jnp.float32)
encoder_output = jnp.zeros((1, 4, 16), dtype=jnp.float32)

# Apply function (stateless)
def model_apply(x, encoder_output):
    return model(x, encoder_output, deterministic=True)

# --- Convert to ONNX ---
onnx_model = to_onnx(
    model_apply,
    inputs=[decoder_input, encoder_output],
    model_name="TransformerDecoder",
)

# --- Save the ONNX model ---
out_dir = "docs/onnx"
os.makedirs(out_dir, exist_ok=True)
onnx_path = os.path.join(out_dir, "transformer_decoder.onnx")
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"✅ Saved ONNX model to {onnx_path}")

# --- Verify outputs match ---
jax_out = model_apply(decoder_input, encoder_output)
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