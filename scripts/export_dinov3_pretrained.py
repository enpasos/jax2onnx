#!/usr/bin/env python3
# scripts/export_dinov3_pretrained.py

"""Export a pretrained DINOv3 Equinox checkpoint to ONNX.

By default this script loads the converted Equimo checkpoint from
``~/.cache/equimo/dinov3/dinov3_vits16_pretrain_lvd1689m.tar.lz4`` and emits an
ONNX graph under ``docs/onnx/examples/eqx_dino/``. Override the paths or
variant with CLI flags if you want to export a different model.

Usage
-----
    poetry run python scripts/export_dinov3_pretrained.py \
        --weights ~/.cache/equimo/dinov3/dinov3_vits16_pretrain_lvd1689m.tar.lz4 \
        --variant dinov3_vits16_pretrain_lvd1689m \
        --output docs/onnx/examples/eqx_dino/dinov3_vits16_pretrain_lvd1689m.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import types
from jax import core as jax_core
from jax import random

from jax2onnx import to_onnx
from jax2onnx.plugins.examples.eqx.dino import load_pretrained_dinov3


def _freeze_rope_grids(model) -> None:
    """Precompute RoPE sin/cos grids so tracing does not rebuild them."""

    pos_embed = getattr(model, "pos_embed", None)
    patch_embed = getattr(model, "patch_embed", None)
    if pos_embed is None or patch_embed is None or not hasattr(pos_embed, "get_sincos"):
        return

    grid_size = getattr(patch_embed, "grid_size", None)
    if grid_size is None or len(grid_size) != 2:
        return
    height = int(grid_size[0])
    width = int(grid_size[1])

    original = pos_embed.get_sincos
    sin_cache, cos_cache = original(
        H=height,
        W=width,
        key=random.PRNGKey(0),
        inference=True,
    )
    sin_cache = jax.lax.stop_gradient(sin_cache)
    cos_cache = jax.lax.stop_gradient(cos_cache)

    def _patched(self, *, H, W, key, inference=None):
        if inference is False:
            return original(H=H, W=W, key=key, inference=inference)
        height_req = jax_core.concrete_or_error(
            int, H, "DINO export requires static H."
        )
        width_req = jax_core.concrete_or_error(int, W, "DINO export requires static W.")
        if height_req != height or width_req != width:
            raise ValueError(
                f"Precomputed RoPE grid supports {(height, width)}; "
                f"requested {(height_req, width_req)}."
            )
        return sin_cache, cos_cache

    object.__setattr__(
        pos_embed,
        "get_sincos",
        types.MethodType(_patched, pos_embed),
    )


def _patch_rope_split() -> None:
    """Replace split-based RoPE helper with slice operations for tracing stability."""

    try:
        from equimo.layers import attention as eq_attn  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return

    rope_apply = getattr(eq_attn, "rope_apply", None)
    original = getattr(eq_attn, "rope_apply_qk_last_hw", None)
    rotate_half = getattr(eq_attn, "rope_rotate_half", None)
    if rope_apply is None or original is None:
        return

    jr = getattr(eq_attn, "jr", random)
    rearrange = getattr(eq_attn, "rearrange")

    def _rotate_half_slices(x):
        width = x.shape[-1]
        if not isinstance(width, (int, np.integer)):
            width = jax_core.concrete_or_error(
                int,
                width,
                "RoPE requires static head width.",
            )
        half = width // 2
        first = x[..., :half]
        second = x[..., half:]
        return jnp.concatenate([-second, first], axis=-1)

    if rotate_half is not None:
        eq_attn.rope_rotate_half = _rotate_half_slices

    def _rope_apply_qk_last_hw(q, k, sin, cos):
        head_dim = q.shape[-1]
        if sin.shape[-1] != head_dim or cos.shape[-1] != head_dim:
            raise ValueError(
                f"RoPE sin/cos last dim ({sin.shape[-1]}) must equal head_dim ({head_dim})."
            )

        tokens = q.shape[-2]
        if not isinstance(tokens, (int, np.integer)):
            tokens = jax_core.concrete_or_error(
                int,
                tokens,
                "RoPE requires the sequence length to be static.",
            )
        spatial = sin.shape[-2]
        if not isinstance(spatial, (int, np.integer)):
            spatial = jax_core.concrete_or_error(
                int,
                spatial,
                "RoPE requires the spatial length to be static.",
            )
        prefix = tokens - int(spatial)
        if prefix < 0:
            raise ValueError(
                f"Sequence length {tokens} smaller than rotary span {spatial}."
            )

        q_dtype = q.dtype
        k_dtype = k.dtype
        rope_dtype = sin.dtype
        q_cast = q.astype(rope_dtype)
        k_cast = k.astype(rope_dtype)
        sin_b = sin[None, :, :]
        cos_b = cos[None, :, :]

        if prefix > 0:
            q_prefix = q_cast[..., :prefix, :]
            k_prefix = k_cast[..., :prefix, :]
            q_tail = q_cast[..., prefix:, :]
            k_tail = k_cast[..., prefix:, :]
            q_tail = rope_apply(q_tail, sin_b, cos_b).astype(q_dtype)
            k_tail = rope_apply(k_tail, sin_b, cos_b).astype(k_dtype)
            q_out = jnp.concatenate(
                [q_prefix.astype(q_dtype), q_tail],
                axis=-2,
            )
            k_out = jnp.concatenate(
                [k_prefix.astype(k_dtype), k_tail],
                axis=-2,
            )
        else:
            q_out = rope_apply(q_cast, sin_b, cos_b).astype(q_dtype)
            k_out = rope_apply(k_cast, sin_b, cos_b).astype(k_dtype)
        return q_out, k_out

    eq_attn.rope_apply_qk_last_hw = _rope_apply_qk_last_hw

    def _attention_call(
        self,
        x,
        key,
        inference: bool | None = None,
        mask=None,
        rope_sincos=None,
        **kwargs,
    ):
        if inference:
            key1 = key
            key2 = key
        else:
            key1, key2 = jr.split(key, 2)

        qkv = jax.vmap(self.qkv)(x)
        qkv = rearrange(
            qkv,
            "s (n h d) -> n h s d",
            n=3,
            h=self.num_heads,
            d=self.dim // self.num_heads,
        )
        q, k, v = qkv
        q = jax.vmap(jax.vmap(self.q_norm))(q)
        k = jax.vmap(jax.vmap(self.k_norm))(k)

        if rope_sincos is not None:
            sin, cos = rope_sincos
            if sin.shape[-1] != self.head_dim or cos.shape[-1] != self.head_dim:
                raise ValueError(
                    f"RoPE sin/cos last dim ({sin.shape[-1]}) must equal head_dim ({self.head_dim})."
                )
            q, k = eq_attn.rope_apply_qk_last_hw(q, k, sin, cos)

        attn = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(self.head_dim)

        if mask is not None:
            attn = jnp.where(mask == 0, jnp.finfo(attn.dtype).min, attn)

        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, inference=inference, key=key1)

        x = jnp.einsum("hqk,hkd->hqd", attn, v)
        x = rearrange(x, "h s d -> s (h d)")
        x = jax.vmap(self.proj)(x)
        x = self.proj_drop(x, inference=inference, key=key2)

        return x

    eq_attn.Attention.__call__ = _attention_call

    def _attention_block_call(
        self,
        x,
        key,
        rope_sincos=None,
        inference: bool | None = None,
        mask=None,
    ):
        if inference:
            key_attn = key
            key_mlp = key
            key_dr1 = key
            key_dr2 = key
        else:
            key_attn, key_mlp, key_dr1, key_dr2 = jr.split(key, 4)

        extra_kwargs = {"mask": mask} if mask is not None else {}
        attn_kwargs = (
            extra_kwargs | {"rope_sincos": rope_sincos}
            if rope_sincos is not None
            else extra_kwargs
        )

        x = self.drop_path1(
            x,
            self.ls1(
                jax.vmap(self.postnorm)(
                    self.attn(
                        jax.vmap(self.prenorm)(x),
                        inference=inference,
                        key=key_attn,
                        **attn_kwargs,
                    )
                )
            ),
            inference=inference,
            key=key_dr1,
        )
        x = self.drop_path2(
            x,
            self.ls2(
                self.mlp(
                    jax.vmap(self.norm)(x),
                    inference=inference,
                    key=key_mlp,
                    **extra_kwargs,
                )
            ),
            inference=inference,
            key=key_dr2,
        )
        return x

    eq_attn.AttentionBlock.__call__ = _attention_block_call


def _patch_vit_features() -> None:
    """Ensure VisionTransformer.features avoids random split during inference."""

    try:
        from equimo.models import vit as eq_vit  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return

    jr = getattr(eq_vit, "jr", random)
    rearrange = getattr(eq_vit, "rearrange")

    def _blockchunk_call(
        self,
        x,
        *,
        key,
        inference: bool | None = None,
        **kwargs,
    ):
        if inference:
            keys = [key] * len(self.blocks)
        else:
            keys = jr.split(key, len(self.blocks))

        x = self.posemb(x)

        for blk, key_block in zip(self.blocks, keys):
            x = blk(x, inference=inference, key=key_block, **kwargs)

        if self.downsampler_contains_dropout:
            x = self.downsample(x, inference=inference, key=key)
        else:
            x = self.downsample(x)
        return x

    def _features(
        self,
        x,
        key,
        mask=None,
        inference: bool | None = None,
        **kwargs,
    ):
        if inference:
            key_pos = key
            block_subkeys = [key] * len(self.blocks)
        else:
            split_keys = list(jr.split(key, len(self.blocks) + 1))
            key_pos, *block_subkeys = split_keys

        x = self.patch_embed(x)

        if mask is not None:
            assert (
                self.mask_token is not None
            ), "To use masked forward, init the model with `use_mask_token=True`."
            if self.dynamic_img_size:
                mask_arr = rearrange(mask, "h w -> 1 h w")
                value = rearrange(self.mask_token, "1 c -> c 1 1")
            else:
                mask_arr = rearrange(mask, "h w -> (h w) 1")
                value = self.mask_token
            x = jnp.where(mask_arr, x, value.astype(x.dtype))

        if self.use_rope_pos_embed:
            _, H, W = x.shape
            if inference:
                rope_sincos = self.pos_embed.get_sincos(
                    H=H,
                    W=W,
                    inference=inference,
                    key=key_pos,
                )
            x = jnp.concatenate(
                [
                    self.cls_token,
                    self.reg_tokens,
                    rearrange(x, "c h w -> (h w) c"),
                ],
                axis=0,
            )
        else:
            rope_sincos = None
            x = self.pos_embed(
                x,
                cls_token=self.cls_token,
                reg_tokens=self.reg_tokens,
                dynamic_img_size=self.dynamic_img_size,
            )

        for blk, key_block in zip(self.blocks, block_subkeys):
            if self.use_rope_pos_embed and not inference:
                key_pos, key_rope = jr.split(key_pos, 2)
                rope_sincos = self.pos_embed.get_sincos(
                    H=H,
                    W=W,
                    inference=inference,
                    key=key_rope,
                )
            x = blk(
                x,
                rope_sincos=rope_sincos,
                inference=inference,
                key=key_block,
                **kwargs,
            )

        return x

    eq_vit.VisionTransformer.features = _features
    eq_vit.BlockChunk.__call__ = _blockchunk_call


def _disable_random_split() -> None:
    """Override jax.random.split with a deterministic, broadcast-based variant."""

    def _deterministic_split(key, num=2):
        key_arr = jnp.asarray(key)
        reps = int(num)
        expanded = key_arr.reshape((1,) + key_arr.shape)
        target_shape = (reps,) + key_arr.shape
        return jnp.broadcast_to(expanded, target_shape)

    jax.random.split = _deterministic_split  # type: ignore[assignment]


def _force_gelu_activation(module) -> None:
    """Recursively replace MLP activations with jax.nn.gelu to avoid erfc."""

    try:
        from equimo.layers.ffn import Mlp  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return

    if isinstance(module, Mlp):
        object.__setattr__(module, "act_layer", jax.nn.gelu)

    if isinstance(module, eqx.Module):
        from dataclasses import fields

        for field in fields(module):
            value = getattr(module, field.name)
            _force_gelu_activation(value)
    elif isinstance(module, (list, tuple)):
        for item in module:
            _force_gelu_activation(item)
    elif isinstance(module, dict):
        for item in module.values():
            _force_gelu_activation(item)


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
        help="Path to the Equinox checkpoint (.tar.lz4). Defaults to ~/.cache/equimo/dinov3/{variant}.tar.lz4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination ONNX file. Defaults to docs/onnx/examples/eqx_dino/{variant}.onnx.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed used during the forward pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    weights_path = (
        args.weights
        if args.weights is not None
        else Path(f"~/.cache/equimo/dinov3/{args.variant}.tar.lz4").expanduser()
    )
    output_path = (
        args.output
        if args.output is not None
        else Path(f"docs/onnx/examples/eqx_dino/{args.variant}.onnx")
    )

    model = load_pretrained_dinov3(
        variant=args.variant,
        weights_path=weights_path,
        inference_mode=True,
    )
    if getattr(model, "dynamic_img_size", False):
        object.__setattr__(model, "dynamic_img_size", False)
    _freeze_rope_grids(model)
    _patch_rope_split()
    _patch_vit_features()
    _disable_random_split()
    _force_gelu_activation(model)

    base_key = random.PRNGKey(args.seed)

    def _features_single(img, key):
        return model.features(img, inference=True, key=key)

    def features(batch):
        keys = jnp.tile(base_key[None, :], (batch.shape[0], 1))
        return jax.vmap(_features_single)(batch, keys)

    dummy = jnp.zeros((1, 3, 224, 224), dtype=jnp.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_onnx(
        features,
        inputs=[dummy],
        model_name=args.variant,
        output_path=output_path,
        return_mode="file",
    )
    print(f"Exported ONNX to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
