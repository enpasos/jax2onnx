#!/usr/bin/env python3
# scripts/map_equimo_dino_weights.py

"""Transfer Equimo DINOv3 weights into the simplified `examples.eqx_dino` model.

This script is a first step towards parity between the upstream Equimo checkpoints
and the @onnx_function-backed examples that live in ``jax2onnx/plugins/examples``.
At the moment the plain example architecture does **not** model the extra register
tokens used by Equimo’s pretrained variants. Because of that structural mismatch
we abort when the source checkpoint exposes ``reg_tokens > 0``—please extend the
example modules (or teach the mapper how to fold register tokens away) before
lifting weights for those variants.

Usage
-----
    poetry run python scripts/map_equimo_dino_weights.py \\
        --variant dinov3_vits16_pretrain_lvd1689m \\
        --weights ~/.cache/equimo/dinov3/dinov3_vits16_pretrain_lvd1689m.tar.lz4 \\
        --output docs/onnx/examples/eqx_dino/dinov3_vits16_pretrain_lvd1689m.eqx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from jax2onnx.plugins.examples.eqx.dino import (
    VisionTransformer,
    load_pretrained_dinov3,
)
from scripts.export_dinov3_pretrained import (
    _disable_random_split,
    _force_gelu_activation,
    _freeze_rope_grids,
    _patch_rope_split,
    _patch_vit_features,
)


def _copy_linear(
    module: eqx.nn.Linear,
    weight,
    bias,
) -> eqx.nn.Linear:
    """Return ``module`` with weight/bias replaced."""

    return eqx.tree_at(
        lambda m: (m.weight, m.bias),
        module,
        (jnp.asarray(weight), jnp.asarray(bias)),
    )


def _copy_layernorm(dst: eqx.nn.LayerNorm, src: eqx.nn.LayerNorm) -> eqx.nn.LayerNorm:
    return eqx.tree_at(
        lambda ln: (ln.weight, ln.bias),
        dst,
        (jnp.asarray(src.weight), jnp.asarray(src.bias)),
    )


def _flatten_blocks(equimo_blocks: Sequence) -> list:
    return [blk for chunk in equimo_blocks for blk in chunk.blocks]


def _build_example_from_equimo(equimo_model, *, strip_register_tokens: bool = False):
    """Create a VisionTransformer with parameters mapped from the Equimo checkpoint."""

    num_reg_tokens = int(getattr(equimo_model, "num_reg_tokens", 0))
    if strip_register_tokens:
        num_storage_tokens = 0
    else:
        num_storage_tokens = num_reg_tokens

    config = {
        "img_size": int(equimo_model.patch_embed.img_size[0]),
        "patch_size": int(equimo_model.patch_embed.patch_size[0]),
        "embed_dim": int(equimo_model.dim),
        "depth": sum(len(chunk.blocks) for chunk in equimo_model.blocks),
        "num_heads": int(equimo_model.blocks[0].blocks[0].attn.num_heads),
        "num_storage_tokens": num_storage_tokens,
    }

    example = VisionTransformer(**config, key=jax.random.PRNGKey(0))

    # Patch embedding / CLS token / output norm
    example = eqx.tree_at(
        lambda m: (m.patch_embed.proj.weight, m.patch_embed.proj.bias),
        example,
        (
            jnp.asarray(equimo_model.patch_embed.proj.weight),
            jnp.asarray(equimo_model.patch_embed.proj.bias),
        ),
    )
    example = eqx.tree_at(
        lambda m: m.cls_token,
        example,
        jnp.asarray(equimo_model.cls_token).reshape(1, 1, -1),
    )
    if num_storage_tokens:
        reg_tokens = jnp.asarray(equimo_model.reg_tokens).reshape(
            1, num_storage_tokens, -1
        )
        example = eqx.tree_at(lambda m: m.storage_tokens, example, reg_tokens)
    example = eqx.tree_at(
        lambda m: (m.norm.weight, m.norm.bias),
        example,
        (
            jnp.asarray(equimo_model.norm.weight),
            jnp.asarray(equimo_model.norm.bias),
        ),
    )

    source_blocks = _flatten_blocks(equimo_model.blocks)
    new_blocks = []
    for dst_block, src_block in zip(example.blocks, source_blocks, strict=True):
        dst_block = eqx.tree_at(
            lambda b: (b.norm1.weight, b.norm1.bias),
            dst_block,
            (
                jnp.asarray(src_block.prenorm.weight),
                jnp.asarray(src_block.prenorm.bias),
            ),
        )
        dst_block = eqx.tree_at(
            lambda b: (b.norm2.weight, b.norm2.bias),
            dst_block,
            (
                jnp.asarray(src_block.norm.weight),
                jnp.asarray(src_block.norm.bias),
            ),
        )

        # Multi-head attention
        attn = dst_block.attn.core.attn
        src_attn = src_block.attn
        Wqkv = jnp.asarray(src_attn.qkv.weight)
        bqkv = jnp.asarray(src_attn.qkv.bias)
        Wq, Wk, Wv = jnp.split(Wqkv, 3, axis=0)
        bq, bk, bv = jnp.split(bqkv, 3, axis=0)

        attn = eqx.tree_at(
            lambda a: a.query_proj, attn, _copy_linear(attn.query_proj, Wq, bq)
        )
        attn = eqx.tree_at(
            lambda a: a.key_proj, attn, _copy_linear(attn.key_proj, Wk, bk)
        )
        attn = eqx.tree_at(
            lambda a: a.value_proj, attn, _copy_linear(attn.value_proj, Wv, bv)
        )

        gamma1 = jnp.asarray(src_block.ls1.gamma)
        jnp.asarray(src_attn.proj.weight) * gamma1[:, None]
        jnp.asarray(src_attn.proj.bias) * gamma1
        attn = eqx.tree_at(
            lambda a: a.output_proj,
            attn,
            _copy_linear(attn.output_proj, src_attn.proj.weight, src_attn.proj.bias),
        )
        dst_block = eqx.tree_at(lambda b: b.attn.core.attn, dst_block, attn)
        dst_block = eqx.tree_at(
            lambda b: b.ls1.gamma, dst_block, jnp.asarray(src_block.ls1.gamma)
        )

        # MLP (LayerScale copied separately)
        dst_block = eqx.tree_at(
            lambda b: b.mlp.fc1.weight,
            dst_block,
            jnp.asarray(src_block.mlp.fc1.weight),
        )
        dst_block = eqx.tree_at(
            lambda b: b.mlp.fc1.bias,
            dst_block,
            jnp.asarray(src_block.mlp.fc1.bias),
        )
        dst_block = eqx.tree_at(
            lambda b: b.mlp.fc2.weight,
            dst_block,
            jnp.asarray(src_block.mlp.fc2.weight),
        )
        dst_block = eqx.tree_at(
            lambda b: b.mlp.fc2.bias,
            dst_block,
            jnp.asarray(src_block.mlp.fc2.bias),
        )
        dst_block = eqx.tree_at(
            lambda b: b.ls2.gamma, dst_block, jnp.asarray(src_block.ls2.gamma)
        )

        new_blocks.append(dst_block)

    example = eqx.tree_at(lambda m: m.blocks, example, list(new_blocks))
    return example


def map_weights(
    variant: str,
    weights_path: Path,
    output: Path,
    *,
    strip_register_tokens: bool = False,
) -> None:
    equimo = load_pretrained_dinov3(
        variant=variant,
        weights_path=weights_path,
        inference_mode=True,
    )
    if getattr(equimo, "dynamic_img_size", False):
        object.__setattr__(equimo, "dynamic_img_size", False)
    _freeze_rope_grids(equimo)
    _patch_rope_split()
    _patch_vit_features()
    _disable_random_split()
    _force_gelu_activation(equimo)

    mapped = _build_example_from_equimo(
        equimo, strip_register_tokens=strip_register_tokens
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(output, mapped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        default="dinov3_vits16_pretrain_lvd1689m",
        help="Equimo identifier for the pretrained checkpoint.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the Equinox checkpoint (.tar.lz4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .eqx file where the remapped model will be stored.",
    )
    parser.add_argument(
        "--strip-register-tokens",
        action="store_true",
        help=(
            "Ignore register/storage tokens present in Equimo checkpoints while mapping. "
            "This preserves the simplified example structure but may change semantics."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    map_weights(
        args.variant,
        args.weights.expanduser(),
        args.output.expanduser(),
        strip_register_tokens=args.strip_register_tokens,
    )
    print(f"Wrote remapped weights to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
