#!/usr/bin/env python3
# scripts/dino_block_token_deltas.py

"""
Dump per-token-category deltas (CLS vs register vs patches) for each block/stage
when comparing the simplified examples.eqx_dino VisionTransformer against the
converted Equimo Equinox model.

Example:
  poetry run python scripts/dino_block_token_deltas.py \
    --image /tmp/coco_39769.jpg \
    --variant dinov3_vits16_pretrain_lvd1689m \
    --eqx ~/.cache/equimo/dinov3/eqx_dinov3_vits16_mapped.eqx \
    --start-block 6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


def _preprocess(image_path: Path, size: int = 224) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))
    return chw


def _load_example(eqx_path: Path):
    from jax2onnx.plugins.examples.eqx.dino import VisionTransformer

    like = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_storage_tokens=4,
        key=jax.random.PRNGKey(0),
    )
    return eqx.tree_deserialise_leaves(eqx_path, like)


def _example_block_debug(model, x_bchw: np.ndarray) -> List[Dict[str, jax.Array]]:
    from jax2onnx.plugins.examples.eqx.dino import VisionTransformer as _VT

    assert isinstance(model, _VT)
    x = jnp.asarray(x_bchw)
    debugs = model.block_debug_outputs(x)
    out: List[Dict[str, jax.Array]] = []
    for d in debugs:
        out.append({k: v[0] if v.ndim == 3 else v for k, v in d.items()})
    return out


def _equimo_block_debug(
    x_chw: np.ndarray, *, variant: str
) -> Tuple[List[Dict[str, jax.Array]], Dict[str, int]]:
    from jax2onnx.plugins.examples.eqx.dino import (
        load_pretrained_dinov3,
        DinoRoPE,
        _dino_rope_inference_sincos,
    )

    model = load_pretrained_dinov3(variant=variant, inference_mode=True)
    key = jax.random.PRNGKey(0)

    pe = model.patch_embed(jnp.asarray(x_chw))
    patch_tokens = jnp.transpose(pe.reshape(pe.shape[0], -1), (1, 0))
    cls = model.cls_token
    regs = model.reg_tokens
    prefix = jnp.concatenate([cls, regs], axis=0)
    tokens = jnp.concatenate([prefix, patch_tokens], axis=0)

    first_block = model.blocks[0].blocks[0]
    num_heads = int(first_block.attn.num_heads)
    rope = DinoRoPE(dim=int(model.dim), num_heads=num_heads)
    grid = int(np.sqrt(patch_tokens.shape[0]))
    sin, cos = _dino_rope_inference_sincos(rope, H=grid, W=grid)

    def _pw(mod, arr):
        return eqx.filter_vmap(mod, in_axes=0, out_axes=0)(arr)

    debugs: List[Dict[str, jax.Array]] = []
    current = tokens
    for idx, blk in enumerate(model.blocks[0].blocks):
        key_block = jax.random.fold_in(key, idx)
        key_attn, key_mlp = jax.random.split(key_block)

        attn_in = _pw(blk.prenorm, current)
        attn_raw = blk.attn(
            attn_in,
            key=key_attn,
            inference=True,
            rope_sincos=(sin, cos),
        )
        attn_norm = _pw(blk.postnorm, attn_raw)
        attn_scaled = _pw(blk.ls1, attn_norm)
        post_attn = current + attn_scaled

        mlp_in = _pw(blk.norm, post_attn)
        mlp_raw = blk.mlp(mlp_in, key=key_mlp, inference=True)
        mlp_scaled = _pw(blk.ls2, mlp_raw)
        out = post_attn + mlp_scaled

        debugs.append(
            {
                "attn_in": attn_in,
                "attn_raw": attn_raw,
                "attn_norm": attn_norm,
                "attn_scaled": attn_scaled,
                "post_attn": post_attn,
                "mlp_in": mlp_in,
                "mlp_raw": mlp_raw,
                "mlp_scaled": mlp_scaled,
                "output": out,
            }
        )
        current = out

    meta = {
        "tokens": int(tokens.shape[0]),
        "dim": int(tokens.shape[1]),
        "heads": num_heads,
        "grid": grid,
        "prefix": int(prefix.shape[0]),
        "regs": int(regs.shape[0]),
    }
    return debugs, meta


def _max_abs(a: jax.Array) -> float:
    return float(jnp.max(jnp.abs(a)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument(
        "--eqx",
        type=Path,
        default=Path(
            "~/.cache/equimo/dinov3/eqx_dinov3_vits16_mapped.eqx"
        ).expanduser(),
    )
    p.add_argument(
        "--variant",
        type=str,
        default="dinov3_vits16_pretrain_lvd1689m",
    )
    p.add_argument("--start-block", type=int, default=6)
    args = p.parse_args(argv)

    ex_model = _load_example(args.eqx)
    chw = _preprocess(args.image, 224)
    ex_debug = _example_block_debug(ex_model, chw[None, ...])
    eq_debug, meta = _equimo_block_debug(chw, variant=args.variant)

    prefix = meta["prefix"]  # 1 + regs
    regs = meta["regs"]

    stages = [
        "attn_in",
        "attn_raw",
        "attn_norm",
        "attn_scaled",
        "post_attn",
        "mlp_raw",
        "mlp_scaled",
        "output",
    ]

    print(
        f"Variant: {args.variant} | Tokens={meta['tokens']} Dim={meta['dim']} Prefix={prefix} (regs={regs})"
    )
    print("Category deltas are max|Δ| over features for the selected tokens.")
    print("\nCLS vs REG vs PATCH (blocks >= %d)" % args.start_block)
    header = "| Block | Stage |   CLS   |   REG   |  PATCH  |"
    print(header)
    print("|------:|:------|:-------:|:-------:|:-------:|")
    for i, (a, b) in enumerate(zip(ex_debug, eq_debug)):
        if i < args.start_block:
            continue
        for s in stages:
            if s not in a or s not in b:
                continue
            da = a[s]
            db = b[s]
            # Align shapes if needed
            t = min(da.shape[0], db.shape[0])
            d = min(da.shape[1], db.shape[1])
            diff = da[:t, :d] - db[:t, :d]
            cls_delta = _max_abs(diff[0:1, :])
            reg_delta = _max_abs(diff[1:prefix, :]) if prefix > 1 else float("nan")
            patch_delta = _max_abs(diff[prefix:, :])
            print(
                f"| {i:5d} | {s:7s} | {cls_delta:7.2e} | {reg_delta:7.2e} | {patch_delta:7.2e} |"
            )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
