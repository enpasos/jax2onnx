# jax2onnx/plugins/examples/eqx/dino.py

"""Example of converting a DINOv3 Vision Transformer model from Equinox."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    register_example,
    with_prng_key,
)


# --- Model code from https://github.com/clementpoiret/Equimo ---


@onnx_function
class RoPE(eqx.Module):
    """Rotary Positional Embedding."""

    dim: int

    def __call__(self, x: Float[Array, "b n d"], seq_len: int) -> Float[Array, "b n d"]:
        theta = 10000.0
        freqs = 1.0 / (
            theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
        )
        t = jnp.arange(seq_len)
        freqs_cis = jnp.outer(t, freqs)
        sin_freqs, cos_freqs = jnp.sin(freqs_cis), jnp.cos(freqs_cis)
        sin_freqs, cos_freqs = (
            jnp.broadcast_to(sin_freqs, x.shape),
            jnp.broadcast_to(cos_freqs, x.shape),
        )

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_rope = jnp.stack(
            [
                x_even * cos_freqs - x_odd * sin_freqs,
                x_even * sin_freqs + x_odd * cos_freqs,
            ],
            axis=-1,
        )
        return x_rope.reshape(x.shape)


register_example(
    component="RoPE",
    description="Rotary Positional Embedding.",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.9.1",
    context="examples.eqx",
    children=[],
    testcases=[
        {
            "testcase": "rope_embedding",
            "callable": construct_and_call(RoPE, dim=64),
            "input_shapes": [("B", 50, 64), 50],
            "post_check_onnx_graph": EG(["RoPE_1:Bx50x64"], symbols={"B": None}),
            "run_only_f32_variant": True,
        }
    ],
)


@onnx_function
class PatchEmbed(eqx.Module):
    """Image to Patch Embedding."""

    proj: eqx.nn.Conv2d
    num_patches: int

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        *,
        key: jax.Array,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = eqx.nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, key=key
        )

    def __call__(self, x: Array) -> Array:
        x = self.proj(x)
        x = jnp.transpose(x.reshape(x.shape[0], x.shape[1], -1), (0, 2, 1))
        return x


register_example(
    component="PatchEmbed",
    description="Image to Patch Embedding.",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.9.0",
    context="examples.eqx",
    children=["equinox.nn.Conv2d"],
    testcases=[
        {
            "testcase": "patch_embed",
            "callable": construct_and_call(
                PatchEmbed,
                img_size=224,
                patch_size=14,
                embed_dim=384,
                key=with_prng_key(0),
            ),
            "input_shapes": [(1, 3, 224, 224)],
            "post_check_onnx_graph": EG(["PatchEmbed_1:1x256x384"]),
            "run_only_f32_variant": True,
        }
    ],
)


@onnx_function
class Attention(eqx.Module):
    """Multi-Head Self-Attention."""

    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    num_heads: int
    scale: float
    rope: RoPE

    def __init__(self, dim: int, num_heads: int, *, key: jax.Array):
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        keys = jax.random.split(key, 2)
        self.qkv = eqx.nn.Linear(dim, dim * 3, use_bias=True, key=keys[0])
        self.proj = eqx.nn.Linear(dim, dim, use_bias=True, key=keys[1])
        self.rope = RoPE(dim=head_dim)

    def __call__(self, x: Array) -> Array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rope(q, N)
        k = self.rope(k, N)

        attn = (q @ jnp.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = jax.nn.softmax(attn, axis=-1)

        x = jnp.transpose((attn @ v), (0, 2, 1, 3)).reshape(B, N, C)
        x = self.proj(x)
        return x


register_example(
    component="Attention",
    description="Multi-Head Self-Attention with RoPE.",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.9.1",
    context="examples.eqx",
    children=["equinox.nn.Linear", "RoPE"],
    testcases=[
        {
            "testcase": "attention",
            "callable": construct_and_call(
                Attention, dim=384, num_heads=6, key=with_prng_key(0)
            ),
            "input_shapes": [("B", 257, 384)],
            "post_check_onnx_graph": EG(["Attention_1:Bx257x384"], symbols={"B": None}),
            "run_only_f32_variant": True,
        }
    ],
)


@onnx_function
class Block(eqx.Module):
    """Transformer Block."""

    norm1: eqx.nn.LayerNorm
    attn: Attention
    norm2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0, *, key: jax.Array
    ):
        keys = jax.random.split(key, 2)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, key=keys[0])
        self.norm2 = eqx.nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = eqx.nn.MLP(
            in_size=dim,
            out_size=dim,
            width_size=mlp_hidden_dim,
            depth=1,
            activation=jax.nn.gelu,
            key=keys[1],
        )

    def __call__(self, x: Array) -> Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


register_example(
    component="Block",
    description="Transformer Block.",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.9.1",
    context="examples.eqx",
    children=["equinox.nn.LayerNorm", "Attention", "equinox.nn.MLP"],
    testcases=[
        {
            "testcase": "transformer_block",
            "callable": construct_and_call(
                Block, dim=384, num_heads=6, key=with_prng_key(0)
            ),
            "input_shapes": [("B", 257, 384)],
            "post_check_onnx_graph": EG(["Block_1:Bx257x384"], symbols={"B": None}),
            "run_only_f32_variant": True,
        }
    ],
)


@onnx_function
class VisionTransformer(eqx.Module):
    """Vision Transformer."""

    patch_embed: PatchEmbed
    cls_token: jax.Array
    blocks: list[Block]
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, depth + 2)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            key=keys[0],
        )
        self.cls_token = jax.random.normal(keys[1], (1, 1, embed_dim))
        self.blocks = [
            Block(dim=embed_dim, num_heads=num_heads, key=k) for k in keys[2:]
        ]
        self.norm = eqx.nn.LayerNorm(embed_dim)

    def __call__(self, x: Array) -> Array:
        x = self.patch_embed(x)
        cls_tokens = jnp.broadcast_to(self.cls_token, (x.shape[0], 1, x.shape[-1]))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def _get_test_cases():
    """Returns a list of test cases for DINOv3."""
    test_cases = []
    img_size = 224

    vit_configs = {
        "Ti14": {"patch": 14, "dim": 192, "heads": 3, "depth": 12},
        "S14": {"patch": 14, "dim": 384, "heads": 6, "depth": 12},
        "B14": {"patch": 14, "dim": 768, "heads": 12, "depth": 12},
        "S16": {"patch": 16, "dim": 384, "heads": 6, "depth": 12},
    }

    for idx, (name, config) in enumerate(vit_configs.items()):
        num_patches = (img_size // config["patch"]) ** 2
        output_shape = f"B,{num_patches + 1},{config['dim']}"

        test_cases.append(
            {
                "testcase": f"eqx_dinov3_vit_{name}",
                "callable": construct_and_call(
                    VisionTransformer,
                    img_size=img_size,
                    patch_size=config["patch"],
                    embed_dim=config["dim"],
                    depth=config["depth"],
                    num_heads=config["heads"],
                    key=with_prng_key(idx),
                ),
                "input_shapes": [("B", 3, img_size, img_size)],
                "post_check_onnx_graph": EG(
                    [f"VisionTransformer_1:{output_shape}"], symbols={"B": None}
                ),
                "run_only_f32_variant": True,
            }
        )

    return test_cases


register_example(
    component="VisionTransformer",
    description="DINOv3 Vision Transformer.",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.9.1",
    context="examples.eqx",
    children=["PatchEmbed", "Block"],
    testcases=_get_test_cases(), 
)
