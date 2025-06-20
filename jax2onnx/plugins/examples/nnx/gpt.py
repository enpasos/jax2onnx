# file: jax2onnx/plugins/examples/nnx/gpt.py
import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugin_system import onnx_function, register_example


@onnx_function
class CausalSelfAttention(nnx.Module):
    def __init__(
        self,
        n_head: int,
        n_embd: int,
        block_size: int,
        dropout: float,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.attn = nnx.MultiHeadAttention(
            num_heads=n_head,
            in_features=n_embd,
            qkv_features=n_embd,
            out_features=n_embd,
            dropout_rate=dropout,
            rngs=rngs,
        )
        self.causal_mask = nnx.Param(
            jnp.tril(jnp.ones((block_size, block_size))).reshape(
                1, 1, block_size, block_size
            )
        )

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        B, T, C = x.shape
        mask = self.causal_mask[:, :, :T, :T]
        y = self.attn(
            inputs_q=x,
            mask=mask,
            deterministic=deterministic,
            decode=False,
        )
        return y


register_example(
    component="CausalSelfAttention",
    description="A causal self-attention module.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.6.6",
    context="examples.gpt",
    children=["MultiHeadAttention"],
    testcases=[
        {
            "testcase": "causal_self_attention",
            "callable": CausalSelfAttention(
                n_head=12,
                n_embd=768,
                block_size=1024,
                dropout=0.0,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 1024, 768)],
            "input_params": {"deterministic": True},
        }
    ],
)


@onnx_function
class MLP(nnx.Module):
    def __init__(self, n_embd: int, dropout: float, *, rngs: nnx.Rngs):
        super().__init__()
        self.c_fc = nnx.Linear(n_embd, 4 * n_embd, rngs=rngs)
        self.c_proj = nnx.Linear(4 * n_embd, n_embd, rngs=rngs)
        self.dropout = nnx.Dropout(dropout)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        x = self.c_fc(x)
        x = nnx.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


register_example(
    component="GPT_MLP",
    description="An MLP block with GELU activation from nanoGPT.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.6.6",
    context="examples.gpt",
    children=["nnx.Linear", "nnx.gelu", "nnx.Dropout"],
    testcases=[
        {
            "testcase": "gpt_mlp",
            "callable": MLP(
                n_embd=768,
                dropout=0.0,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 1024, 768)],
            "input_params": {"deterministic": True},
        }
    ],
)


@onnx_function
class Block(nnx.Module):
    def __init__(
        self,
        n_head: int,
        n_embd: int,
        block_size: int,
        dropout: float,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.ln_1 = nnx.LayerNorm(n_embd, rngs=rngs)
        self.attn = CausalSelfAttention(
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
            rngs=rngs,
        )
        self.ln_2 = nnx.LayerNorm(n_embd, rngs=rngs)
        self.mlp = MLP(n_embd, dropout, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        x = x + self.attn(self.ln_1(x), deterministic=deterministic)
        x = x + self.mlp(self.ln_2(x), deterministic=deterministic)
        return x


register_example(
    component="Block",
    description="A transformer block combining attention and MLP.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.6.6",
    context="examples.gpt",
    children=["CausalSelfAttention", "GPT_MLP", "nnx.LayerNorm"],
    testcases=[
        {
            "testcase": "gpt_block",
            "callable": Block(
                n_head=12,
                n_embd=768,
                block_size=1024,
                dropout=0.0,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 1024, 768)],
            "input_params": {"deterministic": True},
        }
    ],
)


@onnx_function
class GPT(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        block_size: int,
        dropout: float,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.wte = nnx.Embed(vocab_size, n_embd, rngs=rngs)
        self.wpe = nnx.Embed(block_size, n_embd, rngs=rngs)
        self.drop = nnx.Dropout(dropout)
        self.h = [
            Block(
                n_head=n_head,
                n_embd=n_embd,
                block_size=block_size,
                dropout=dropout,
                rngs=rngs,
            )
            for _ in range(n_layer)
        ]
        self.ln_f = nnx.LayerNorm(n_embd, rngs=rngs)
        self.lm_head = nnx.Linear(n_embd, vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, idx: jax.Array, deterministic: bool = True) -> jax.Array:
        B, T = idx.shape
        pos = jnp.arange(0, T, dtype=jnp.int32).reshape(1, T)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb, deterministic=deterministic)
        for block in self.h:
            x = block(x, deterministic=deterministic)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


register_example(
    component="GPT",
    description="A simple GPT model that reuses nnx.MultiHeadAttention.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.6.6",
    context="examples.gpt",
    children=["Block"],
    testcases=[
        {
            "testcase": "gpt",
            "callable": GPT(
                vocab_size=3144,  # downsampled from 50304 to 50304/16=3144 for testing
                n_layer=2,  # downsampled from 12 to 2 layers for testing
                n_head=12,
                n_embd=768,
                block_size=1024,
                dropout=0.0,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 1024)],
            "input_dtypes": [jnp.int32],
            "input_params": {"deterministic": True},
            "run_only_f32_variant": True,
        }
    ],
)
