# jax2onnx/plugins/examples/nnx/gpt.py

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from functools import wraps

from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    register_example,
    with_rng_seed,
)


@onnx_function(unique=True)
def attention(q, k, v, mask=None, **kwargs):
    """A thin wrapper around nnx.dot_product_attention exposing q, k, v, mask."""
    return nnx.dot_product_attention(q, k, v, mask=mask, **kwargs)


@wraps(attention)
def _call_attention(*args, **kwargs):
    return attention(*args, **kwargs)


def _no_cast_where(model) -> bool:
    """Fail if a Cast feeds directly into a Where node anywhere in the graph."""

    has_cast_where = expect_graph(
        ["Cast -> Where"],
        search_functions=True,
        explain_on_fail=False,
        mode="any",
    )(model)
    if has_cast_where:
        # Re-run with diagnostics enabled to surface the offending path.
        expect_graph(
            ["Cast -> Where"],
            search_functions=True,
            mode="any",
        )(model)
        return False
    return True


register_example(
    component="GPT_Attention",
    description="A multi-head attention layer.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.7.1",
    context="examples.gpt",
    children=["nnx.dot_product_attention"],
    testcases=[
        {
            "testcase": "gpt_attention",
            "callable": _call_attention,
            "input_values": [
                np.random.randn(1, 1024, 12, 64).astype(np.float32),
                np.random.randn(1, 1024, 12, 64).astype(np.float32),
                np.random.randn(1, 1024, 12, 64).astype(np.float32),
                np.tril(np.ones((1, 12, 1024, 1024), dtype=bool)),
            ],
            "expected_number_of_function_instances": 1,
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _no_cast_where,
        }
    ],
)


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
            broadcast_dropout=True,
            dropout_rate=dropout,
            attention_fn=_call_attention,
            rngs=rngs,
        )
        self.resid_dropout = nnx.Dropout(dropout, rngs=rngs)
        self.causal_mask = nnx.Param(
            jnp.tril(jnp.ones((block_size, block_size), dtype=jnp.bool_)).reshape(
                1, 1, block_size, block_size
            )
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        _b, t, _c = x.shape
        mask = self.causal_mask[:, :, :t, :t]
        y = self.attn(inputs_q=x, mask=mask, deterministic=deterministic, decode=False)
        return self.resid_dropout(y, deterministic=deterministic)


register_example(
    component="GPT_CausalSelfAttention",
    description="A causal self-attention module.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.7.0",
    context="examples.gpt",
    children=["MultiHeadAttention"],
    testcases=[
        {
            "testcase": "gpt_causal_self_attention",
            "callable": construct_and_call(
                CausalSelfAttention,
                n_head=12,
                n_embd=768,
                block_size=1024,
                dropout=0.0,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 1024, 768)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _no_cast_where,
        }
    ],
)


@onnx_function
class MLP(nnx.Module):
    def __init__(self, n_embd: int, dropout: float, *, rngs: nnx.Rngs):
        super().__init__()
        self.c_fc = nnx.Linear(n_embd, 4 * n_embd, rngs=rngs)
        self.c_proj = nnx.Linear(4 * n_embd, n_embd, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        batch_dims = x.shape[:-1]
        x = self.c_fc(x)
        x = jnp.reshape(x, batch_dims + (x.shape[-1],))
        x = nnx.gelu(x)
        x = self.c_proj(x)
        x = jnp.reshape(x, batch_dims + (x.shape[-1],))
        return self.dropout(x, deterministic=deterministic)


register_example(
    component="GPT_MLP",
    description="An MLP block with GELU activation from nanoGPT.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.7.0",
    context="examples.gpt",
    children=["nnx.Linear", "nnx.gelu", "nnx.Dropout"],
    testcases=[
        {
            "testcase": "gpt_mlp",
            "callable": construct_and_call(
                MLP,
                n_embd=768,
                dropout=0.0,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 1024, 768)],
            "input_params": {"deterministic": True},
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["MLP_1:Bx1024x768"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
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

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = x + self.attn(self.ln_1(x), deterministic=deterministic)
        x = x + self.mlp(self.ln_2(x), deterministic=deterministic)
        return x


register_example(
    component="GPT_TransformerBlock",
    description="A transformer block combining attention and MLP.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.7.0",
    context="examples.gpt",
    children=["CausalSelfAttention", "GPT_MLP", "nnx.LayerNorm"],
    testcases=[
        {
            "testcase": "gpt_block",
            "callable": construct_and_call(
                Block,
                n_head=12,
                n_embd=768,
                block_size=1024,
                dropout=0.0,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 1024, 768)],
            "input_params": {"deterministic": True},
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["Block_1:Bx1024x768"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
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

    def __call__(self, idx: jnp.ndarray) -> jnp.ndarray:
        return self.wte(idx)


register_example(
    component="GPT_TokenEmbedding",
    description="A token embedding layer using nnx.Embed.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.7.0",
    context="examples.gpt",
    children=["nnx.Embed"],
    testcases=[
        {
            "testcase": "gpt_token_embedding",
            "callable": construct_and_call(
                TokenEmbedding,
                vocab_size=3144,
                n_embd=768,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 1024)],
            "input_dtypes": [jnp.int32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["TokenEmbedding_1:Bx1024x768"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        }
    ],
)


@onnx_function
class PositionEmbedding(nnx.Module):
    def __init__(self, block_size: int, n_embd: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.block_size = block_size
        self.wpe = nnx.Embed(block_size, n_embd, rngs=rngs)

    def __call__(self) -> jnp.ndarray:
        pos = jax.lax.broadcasted_iota(jnp.int32, (1, self.block_size), dimension=1)
        return self.wpe(pos)


register_example(
    component="GPT_PositionEmbedding",
    description="A positional embedding layer using nnx.Embed.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.7.0",
    context="examples.gpt",
    children=["nnx.Embed"],
    testcases=[
        {
            "testcase": "gpt_position_embedding",
            "callable": construct_and_call(
                PositionEmbedding,
                block_size=1024,
                n_embd=768,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [],
            "expected_output_shapes": [(1, 1024, 768)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["PositionEmbedding_1:1x1024x768"],
                no_unused_inputs=True,
            ),
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
        self.blocks = nnx.List(
            [
                Block(
                    n_head=n_head,
                    n_embd=n_embd,
                    block_size=block_size,
                    dropout=dropout,
                    rngs=rngs,
                )
                for _ in range(n_layer)
            ]
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        return x


register_example(
    component="GPT_TransformerStack",
    description="A stack of transformer blocks.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.7.0",
    context="examples.gpt",
    children=["Block"],
    testcases=[
        {
            "testcase": "gpt_transformer_stack",
            "callable": construct_and_call(
                GPTTransformerStack,
                n_layer=2,
                n_head=12,
                n_embd=768,
                block_size=1024,
                dropout=0.0,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 1024, 768)],
            "input_params": {"deterministic": True},
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["GPTTransformerStack_1:Bx1024x768"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        }
    ],
)


@onnx_function
def broadcast_add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return x + y


register_example(
    component="GPT_broadcast_add",
    description="Simple dynamic broadcast + add",
    source="(your patch)",
    since="v0.7.0",
    context="examples.gpt",
    testcases=[
        {
            "testcase": "gpt_broadcast_add_dynamic",
            "callable": broadcast_add,
            "input_shapes": [("B", 4, 5), (1, 4, 5)],
            "expected_output_shape": ("B", 4, 5),
            "post_check_onnx_graph": expect_graph(
                ["Add:Bx4x5"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
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
        self.drop = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        pos_emb = self.wpe()
        x = self.wte(x)
        x = broadcast_add(x, pos_emb)
        return self.drop(x, deterministic=deterministic)


register_example(
    component="GPT_Embeddings",
    description="Combines token and position embeddings with dropout.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.7.0",
    context="examples.gpt",
    children=["TokenEmbedding", "PositionEmbedding", "nnx.Dropout"],
    testcases=[
        {
            "testcase": "gpt_embeddings",
            "callable": construct_and_call(
                GPTEmbeddings,
                vocab_size=3144,
                n_embd=768,
                block_size=1024,
                dropout=0.0,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 1024)],
            "input_dtypes": [jnp.int32],
            "input_params": {"deterministic": True},
            "expected_output_shape": ("B", 1024, 768),
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["GPTEmbeddings_1:Bx1024x768"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
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

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.ln_f(x)
        return self.lm_head(x)


register_example(
    component="GPT_Head",
    description="The head of the GPT model.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.7.0",
    context="examples.gpt",
    children=["nnx.LayerNorm", "nnx.Linear"],
    testcases=[
        {
            "testcase": "gpt_head",
            "callable": construct_and_call(
                GPTHead,
                vocab_size=3144,
                n_embd=768,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 1024, 768)],
            "expected_output_shape": ("B", 1024, 3144),
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["GPTHead_1:Bx1024x3144"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
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

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.embeddings(x, deterministic=deterministic)
        x = self.stack(x, deterministic=deterministic)
        return self.head(x)


register_example(
    component="GPT",
    description="A simple GPT model that reuses nnx.MultiHeadAttention.",
    source="https://github.com/karpathy/nanoGPT",
    since="v0.7.0",
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
            "callable": construct_and_call(
                GPT,
                vocab_size=3144,
                n_layer=2,
                n_head=12,
                n_embd=768,
                block_size=1024,
                dropout=0.0,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 1024)],
            "input_dtypes": [jnp.int32],
            "input_params": {"deterministic": True},
            "expected_output_shape": ("B", 1024, 3144),
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                [
                    {
                        "graph": "custom.PositionEmbedding.1:PositionEmbedding",
                        "path": "Range -> Unsqueeze -> Expand -> Gather",
                        "must_absent": ["Cast"],
                    }
                ],
                no_unused_inputs=True,
                no_unused_function_inputs=True,
                search_functions=True,
            ),
        }
    ],
)
