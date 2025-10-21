# jax2onnx/plugins/examples/eqx/dino.py

"""Example of converting a DINOv3 Vision Transformer model from Equinox."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.equinox.eqx.nn.rotary_positional_embedding import (
    RotaryProcessHeads,
)
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    register_example,
    with_prng_key,
)


# --- Model code from https://github.com/clementpoiret/Equimo ---


def _apply_pointwise(module, x: Array) -> Array:
    """Apply an Equinox module independently across batch and sequence axes."""
    apply_tokens = eqx.filter_vmap(module, in_axes=0, out_axes=0)
    apply_batch = eqx.filter_vmap(apply_tokens, in_axes=0, out_axes=0)
    return apply_batch(x)


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
        apply_conv = eqx.filter_vmap(self.proj, in_axes=0, out_axes=0)
        x = apply_conv(x)
        x = jnp.transpose(x.reshape(x.shape[0], x.shape[1], -1), (0, 2, 1))
        return x


register_example(
    component="PatchEmbed",
    description="Image to Patch Embedding.",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.10.0",
    context="examples.eqx_dino",
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
            "post_check_onnx_graph": EG(
                ["PatchEmbed_1:1x256x384"],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        }
    ],
)


class AttentionCore(eqx.Module):
    """Multi-Head Self-Attention"""

    attn: eqx.nn.MultiheadAttention
    num_heads: int

    def __init__(self, dim: int, num_heads: int, *, key: jax.Array):
        self.num_heads = num_heads
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=dim,
            key_size=dim,
            value_size=dim,
            output_size=dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        def _attend(tokens: Array) -> Array:
            return self.attn(tokens, tokens, tokens, inference=True)

        apply_batch = eqx.filter_vmap(_attend, in_axes=0, out_axes=0)
        return apply_batch(x)


register_example(
    component="AttentionCore",
    description="Multi-Head Self-Attention without rotary processing.",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.10.0",
    context="examples.eqx_dino",
    children=[
        "equinox.nn.MultiheadAttention",
    ],
    testcases=[
        {
            "testcase": "attention_core",
            "callable": construct_and_call(
                AttentionCore, dim=384, num_heads=6, key=with_prng_key(0)
            ),
            "input_shapes": [("B", 257, 384)],
            "post_check_onnx_graph": EG(
                [
                    {"path": "Gemm", "counts": {"Gemm": 4}},
                    {"path": "MatMul", "counts": {"MatMul": 2}},
                    {"path": "Softmax", "counts": {"Softmax": 1}},
                ],
                symbols={"B": None},
                search_functions=True,
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        }
    ],
)


@onnx_function
class Attention(eqx.Module):
    """Multi-Head Self-Attention driven by Equinox primitives."""

    core: AttentionCore
    rope: eqx.nn.RotaryPositionalEmbedding
    process_heads: RotaryProcessHeads
    num_heads: int

    def __init__(self, dim: int, num_heads: int, *, key: jax.Array):
        self.num_heads = num_heads
        self.core = AttentionCore(dim=dim, num_heads=num_heads, key=key)
        head_dim = dim // num_heads
        self.rope = eqx.nn.RotaryPositionalEmbedding(embedding_size=head_dim)
        self.process_heads = RotaryProcessHeads(self.rope)

    def __call__(self, x: Array) -> Array:
        def _attend(tokens: Array) -> Array:
            return self.core.attn(
                tokens,
                tokens,
                tokens,
                inference=True,
                process_heads=self.process_heads,
            )

        return eqx.filter_vmap(_attend, in_axes=0, out_axes=0)(x)


register_example(
    component="Attention",
    description="Multi-Head Self-Attention using Equinox modules.",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.10.0",
    context="examples.eqx_dino",
    children=[
        "AttentionCore",
        "RotaryHeads",
    ],
    testcases=[
        {
            "testcase": "attention",
            "callable": construct_and_call(
                Attention, dim=384, num_heads=6, key=with_prng_key(0)
            ),
            "input_shapes": [("B", 257, 384)],
            "post_check_onnx_graph": EG(
                [
                    {"path": "MatMul", "counts": {"MatMul": 2}},
                    "Softmax",
                ],
                symbols={"B": None},
                search_functions=True,
            ),
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
        norm1_out = _apply_pointwise(self.norm1, x)
        x = x + self.attn(norm1_out)

        norm2_out = _apply_pointwise(self.norm2, x)
        mlp_out = _apply_pointwise(self.mlp, norm2_out)
        x = x + mlp_out
        return x


register_example(
    component="Block",
    description="Transformer Block.",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.10.0",
    context="examples.eqx_dino",
    children=["equinox.nn.LayerNorm", "Attention", "equinox.nn.MLP"],
    testcases=[
        {
            "testcase": "transformer_block",
            "callable": construct_and_call(
                Block, dim=384, num_heads=6, key=with_prng_key(0)
            ),
            "input_shapes": [("B", 257, 384)],
            "post_check_onnx_graph": EG(
                ["Block_1:Bx257x384"],
                symbols={"B": None},
            ),
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
        return _apply_pointwise(self.norm, x)


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
        output_shape = f"Bx{num_patches + 1}x{config['dim']}"

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
                    [f"VisionTransformer_1:{output_shape}"],
                    symbols={"B": None},
                ),
                "run_only_f32_variant": True,
            }
        )

    return test_cases


_DEFAULT_DINOV3_VARIANT: str = "dinov3_vits16_pretrain_lvd1689m"


def load_pretrained_dinov3(
    variant: str = _DEFAULT_DINOV3_VARIANT,
    *,
    weights_path: Optional[str | Path] = None,
    inference_mode: bool = True,
):
    """Load a converted DINOv3 Equinox checkpoint produced via Equimo.

    Parameters
    ----------
    variant:
        Identifier used by Equimo’s converter (e.g. ``dinov3_vits16_pretrain_lvd1689m``).
    weights_path:
        Optional override pointing to the ``.tar.lz4`` archive (or directory) generated
        by :mod:`scripts.convert_dinov3_from_equimo`. Defaults to
        ``~/.cache/equimo/dinov3/{variant}.tar.lz4``.
    inference_mode:
        Forwarded to :func:`equimo.io.load_model`; disable dropout when ``True``.

    Returns
    -------
    eqx.Module
        A :class:`VisionTransformer` instance initialised with pretrained weights.
    """

    try:
        from equimo.io import load_model as _equimo_load_model
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Loading pretrained DINOv3 weights requires `equimo`. "
            "Install it with: pip install 'equimo[conversion]'"
        ) from exc

    candidates: list[Path] = []
    if weights_path is not None:
        candidates.append(Path(weights_path).expanduser())
    else:
        default_archive = Path(f"~/.cache/equimo/dinov3/{variant}.tar.lz4").expanduser()
        default_dir = Path(f"~/.cache/equimo/dinov3/{variant}").expanduser()
        candidates.extend([default_archive, default_dir])

    chosen: Optional[Path] = None
    for candidate in candidates:
        if candidate.is_file():
            chosen = candidate
            break
        if candidate.is_dir():
            chosen = candidate
            break
        if candidate.suffix != ".tar.lz4":
            archive = candidate.with_suffix(".tar.lz4")
            if archive.exists():
                chosen = archive
                break

    if chosen is None:
        locations = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            f"Could not locate pretrained DINOv3 weights for '{variant}'. Checked:\n{locations}\n"
            "Download the checkpoint using scripts/convert_dinov3_from_equimo.py "
            "or provide --weights to the archive."
        )

    return _equimo_load_model(
        cls="vit",
        path=chosen,
        inference_mode=inference_mode,
    )


register_example(
    component="DINOv3VisionTransformer",
    description="DINOv3 Vision Transformer",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.10.0",
    context="examples.eqx_dino",
    children=["PatchEmbed", "Block"],
    testcases=_get_test_cases(),
)
