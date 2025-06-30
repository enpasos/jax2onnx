# file: jax2onnx/plugins/examples/nnx/gpt.py
import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugin_system import onnx_function, register_example


# @onnx_function   todo
def attention(*args, **kwargs):
    return nnx.dot_product_attention(*args, **kwargs)


@onnx_function
class CausalSelfAttention(nnx.Module):
    def __init__(
        self,
        n_head: int,
        n_embd: int,
        block_size: int,
        dropout: float,  # ← new
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.attn = nnx.MultiHeadAttention(
            num_heads=n_head,
            in_features=n_embd,
            qkv_features=n_embd,
            out_features=n_embd,
            # attention_fn=lambda *args, **kwargs: attention(*args),
            attention_fn=attention,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout)  # ← new
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
            decode=False,
        )
        # apply dropout to the projected attention output
        y = self.dropout(y, deterministic=deterministic)
        return y


register_example(
    component="GPT_CausalSelfAttention",
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
                dropout=0.0,  # ← new
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 1024, 768)],
            "run_only_f32_variant": True,
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
            "run_only_f32_variant": True,
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
            dropout=dropout,  # ← pass it through
            rngs=rngs,
        )
        self.ln_2 = nnx.LayerNorm(n_embd, rngs=rngs)
        self.mlp = MLP(n_embd, dropout, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x), deterministic=deterministic)
        return x


register_example(
    component="GPT_TransformerBlock",
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
            "run_only_f32_variant": True,
        }
    ],
)


@onnx_function
class TokenEmbedding(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.wte = nnx.Embed(vocab_size, n_embd, rngs=rngs)

    def __call__(self, idx: jax.Array) -> jax.Array:
        return self.wte(idx)


register_example(
    component="GPT_TokenEmbedding",
    description="A token embedding layer using nnx.Embed.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.6.6",
    context="examples.gpt",
    children=["nnx.Embed"],
    testcases=[
        {
            "testcase": "token_embedding",
            "callable": TokenEmbedding(
                vocab_size=3144,
                n_embd=768,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 1024)],
            "input_dtypes": [jnp.int32],
            "run_only_f32_variant": True,
        }
    ],
)


@onnx_function
class PositionEmbedding(nnx.Module):
    def __init__(
        self,
        block_size: int,
        n_embd: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.wpe = nnx.Embed(block_size, n_embd, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T = x.shape
        pos = jnp.arange(0, T, dtype=jnp.int32).reshape(1, T)
        return self.wpe(pos)


register_example(
    component="GPT_PositionEmbedding",
    description="A positional embedding layer using nnx.Embed.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.6.6",
    context="examples.gpt",
    children=["nnx.Embed"],
    testcases=[
        {
            "testcase": "position_embedding",
            "callable": PositionEmbedding(
                block_size=1024,
                n_embd=768,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 1024)],
            "input_dtypes": [jnp.int32],
            "run_only_f32_variant": True,
        }
    ],
)


@onnx_function
class GPTTransformerStack(nnx.Module):
    def __init__(
        self,
        n_layer: int,
        n_head: int,
        n_embd: int,
        block_size: int,
        dropout: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.blocks = [
            Block(
                n_head=n_head,
                n_embd=n_embd,
                block_size=block_size,
                dropout=dropout,
                rngs=rngs,
            )
            for _ in range(n_layer)
        ]

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        return x


register_example(
    component="GPT_TransformerStack",
    description="A stack of transformer blocks.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.6.6",
    context="examples.gpt",
    children=["Block"],
    testcases=[
        {
            "testcase": "transformer_stack",
            "callable": GPTTransformerStack(
                n_layer=2,
                n_head=12,
                n_embd=768,
                block_size=1024,
                dropout=0.0,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 1024, 768)],
            "input_params": {"deterministic": True},
            "run_only_f32_variant": True,
        }
    ],
)


@onnx_function
class GPTEmbeddings(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        dropout: float,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.wte = TokenEmbedding(vocab_size, n_embd, rngs=rngs)
        self.wpe = PositionEmbedding(block_size, n_embd, rngs=rngs)
        self.drop = nnx.Dropout(dropout)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        pos_emb = self.wpe(x)
        x = self.wte(x)
        x = self.drop(x + pos_emb, deterministic=deterministic)
        return x


register_example(
    component="GPT_Embeddings",
    description="Combines token and position embeddings with dropout.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.6.6",
    context="examples.gpt",
    children=[
        "TokenEmbedding",
        "PositionEmbedding",
        "nnx.Dropout",
    ],
    testcases=[
        {
            "testcase": "gpt_embeddings",
            "callable": GPTEmbeddings(
                vocab_size=3144,
                n_embd=768,
                block_size=1024,
                dropout=0.0,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 1024)],
            "input_params": {"deterministic": True},
            "expected_output_shape": ("B", 1024, 768),
            "run_only_f32_variant": True,
        }
    ],
)


@onnx_function
class GPTHead(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.ln_f = nnx.LayerNorm(n_embd, rngs=rngs)
        self.lm_head = nnx.Linear(n_embd, vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


register_example(
    component="GPT_Head",
    description="The head of the GPT model.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.6.6",
    context="examples.gpt",
    children=[
        "nnx.LayerNorm",
        "nnx.Linear",
    ],
    testcases=[
        {
            "testcase": "gpt_head",
            "callable": GPTHead(
                vocab_size=3144,
                n_embd=768,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 1024, 768)],
            "expected_output_shape": ("B", 1024, 3144),
            "run_only_f32_variant": True,
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
        self.embeddings = GPTEmbeddings(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
            rngs=rngs,
        )
        self.stack = GPTTransformerStack(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
            rngs=rngs,
        )
        self.head = GPTHead(
            vocab_size=vocab_size,
            n_embd=n_embd,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        x = self.embeddings(x, deterministic=deterministic)
        x = self.stack(x, deterministic=deterministic)
        x = self.head(x)
        return x


register_example(
    component="GPT",
    description="A simple GPT model that reuses nnx.MultiHeadAttention.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.6.6",
    context="examples.gpt",
    children=[
        "TokenEmbedding",
        "PositionEmbedding",
        "TransformerStack",
        "nnx.LayerNorm",
        "nnx.Linear",
        "nnx.Dropout",
    ],
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
            "input_params": {"deterministic": True},
            "expected_output_shape": ("B", 1024, 3144),  # logits for each token
            "run_only_f32_variant": True,
        }
    ],
)
