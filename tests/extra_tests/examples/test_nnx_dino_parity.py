# tests/extra_tests/examples/test_nnx_dino_parity.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax2onnx.plugins.examples.eqx.dino import VisionTransformer as EqxVisionTransformer
from jax2onnx.plugins.examples.nnx.dinov3 import (
    VisionTransformer as NnxVisionTransformer,
)


def _copy_conv2d(eqx_conv, nnx_conv) -> None:
    """Map Equinox Conv2d weights/biases onto an nnx.Conv (layout conversion)."""
    # eqx.Conv2d uses (out, in, kh, kw); nnx.Conv expects (kh, kw, in, out)
    kernel = jnp.asarray(eqx_conv.weight)
    kernel_nnx = jnp.transpose(kernel, (2, 3, 1, 0))
    nnx_conv.kernel = nnx.Param(kernel_nnx)
    if eqx_conv.bias is not None:
        bias = jnp.asarray(eqx_conv.bias).reshape(-1)
        nnx_conv.bias = nnx.Param(bias)


def _copy_linear(eqx_linear, nnx_linear) -> None:
    """Map Equinox Linear (out x in) params to nnx.Linear (in x out)."""
    nnx_linear.kernel = nnx.Param(jnp.asarray(eqx_linear.weight).T)
    if eqx_linear.bias is not None:
        nnx_linear.bias = nnx.Param(jnp.asarray(eqx_linear.bias))


def _copy_layer_norm(eqx_ln, nnx_ln) -> None:
    nnx_ln.scale = nnx.Param(jnp.asarray(eqx_ln.weight))
    nnx_ln.bias = nnx.Param(jnp.asarray(eqx_ln.bias))


def _copy_layerscale(eqx_ls, nnx_ls) -> None:
    nnx_ls.gamma = nnx.Param(jnp.asarray(eqx_ls.gamma))


def _copy_block(eqx_block, nnx_block) -> None:
    _copy_layer_norm(eqx_block.norm1, nnx_block.norm1)

    eqx_attn = eqx_block.attn.core.attn
    _copy_linear(eqx_attn.query_proj, nnx_block.attn.core.q_proj)
    _copy_linear(eqx_attn.key_proj, nnx_block.attn.core.k_proj)
    _copy_linear(eqx_attn.value_proj, nnx_block.attn.core.v_proj)
    _copy_linear(eqx_attn.output_proj, nnx_block.attn.core.out_proj)

    _copy_layer_norm(eqx_block.norm2, nnx_block.norm2)

    _copy_linear(eqx_block.mlp.fc1, nnx_block.mlp.fc1)
    _copy_linear(eqx_block.mlp.fc2, nnx_block.mlp.fc2)

    _copy_layerscale(eqx_block.ls1, nnx_block.ls1)
    _copy_layerscale(eqx_block.ls2, nnx_block.ls2)


def _copy_eqx_to_nnx(
    eqx_model: EqxVisionTransformer, nnx_model: NnxVisionTransformer
) -> None:
    _copy_conv2d(eqx_model.patch_embed.proj, nnx_model.patch_embed.proj)
    nnx_model.cls_token = nnx.Param(jnp.asarray(eqx_model.cls_token))
    if eqx_model.storage_tokens is not None and nnx_model.storage_tokens is not None:
        nnx_model.storage_tokens = nnx.Param(jnp.asarray(eqx_model.storage_tokens))

    for eqx_blk, nnx_blk in zip(eqx_model.blocks, nnx_model.blocks):
        _copy_block(eqx_blk, nnx_blk)

    _copy_layer_norm(eqx_model.norm, nnx_model.norm)

    # RoPE periods are constructed deterministically; mirror them explicitly for parity.
    nnx_model.dino_rope.periods = nnx.data(np.asarray(eqx_model.dino_rope.periods))


def test_nnx_dino_matches_eqx_forward():
    img_size = 56
    cfg = dict(
        img_size=img_size,
        patch_size=14,
        embed_dim=64,
        depth=2,
        num_heads=4,
        num_storage_tokens=0,
    )

    eqx_model = EqxVisionTransformer(key=jax.random.PRNGKey(0), **cfg)
    nnx_model = NnxVisionTransformer(rngs=nnx.Rngs(0), **cfg)
    _copy_eqx_to_nnx(eqx_model, nnx_model)

    x = jax.random.normal(jax.random.PRNGKey(1), (1, 3, img_size, img_size))
    eqx_out = eqx_model(x)
    nnx_out = nnx_model(x)

    np.testing.assert_allclose(
        np.asarray(nnx_out),
        np.asarray(eqx_out),
        rtol=1e-4,
        atol=1e-4,
    )
