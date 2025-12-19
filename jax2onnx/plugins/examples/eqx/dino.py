# jax2onnx/plugins/examples/eqx/dino.py

"""Example of converting a DINOv3 Vision Transformer model from Equinox."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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


def _rotate_half_last_dim(x: jax.Array) -> jax.Array:
    """Rotate pairs in the last dimension by 90 degrees."""
    half = x.shape[-1] // 2
    first, second = jnp.split(x, [half], axis=-1)
    return jnp.concatenate([-second, first], axis=-1)


@onnx_function
class DinoRoPE(eqx.Module):
    """2D rotary embedding helper mirroring Equimo's DinoRoPE behaviour."""

    D_head: int = eqx.field(static=True)
    normalize_coords: Literal["min", "max", "separate"] = eqx.field(static=True)
    shift_coords: Optional[float] = eqx.field(static=True)
    jitter_coords: Optional[float] = eqx.field(static=True)
    rescale_coords: Optional[float] = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    periods: jax.Array

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        base: Optional[float] = 100.0,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
        periods_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.float32,
    ):
        if dim % (4 * num_heads) != 0:
            raise ValueError("dim must be divisible by 4 * num_heads.")
        both_periods = (min_period is not None) and (max_period is not None)
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided."
            )
        if normalize_coords not in ("min", "max", "separate"):
            raise ValueError(f"Unknown normalize_coords: {normalize_coords}")

        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        self.D_head = dim // num_heads
        denom = self.D_head // 2
        D_quarter = self.D_head // 4

        if base is not None:
            k = jnp.arange(D_quarter, dtype=periods_dtype)
            periods = base ** (2.0 * k / float(denom))
        else:
            assert min_period is not None and max_period is not None
            base_ratio = max_period / min_period
            exponents = jnp.linspace(0.0, 1.0, D_quarter, dtype=periods_dtype)
            periods = base_ratio**exponents
            periods = periods / base_ratio
            periods = periods * max_period
            periods = periods.astype(periods_dtype)

        self.periods = periods.astype(dtype)

    def _make_coords(self, H: int, W: int) -> jax.Array:
        dtype = self.dtype
        if self.normalize_coords == "max":
            denom = float(max(H, W))
            coords_h = jnp.arange(0.5, H, step=1.0) / denom
            coords_w = jnp.arange(0.5, W, step=1.0) / denom
        elif self.normalize_coords == "min":
            denom = float(min(H, W))
            coords_h = jnp.arange(0.5, H, step=1.0) / denom
            coords_w = jnp.arange(0.5, W, step=1.0) / denom
        else:
            coords_h = jnp.arange(0.5, H, step=1.0) / float(H)
            coords_w = jnp.arange(0.5, W, step=1.0) / float(W)

        hh, ww = jnp.meshgrid(coords_h, coords_w, indexing="ij")
        coords = jnp.stack([hh, ww], axis=-1).reshape(H * W, 2)
        coords = 2.0 * coords - 1.0
        return coords.astype(dtype)

    def get_sincos(
        self,
        *,
        H: int,
        W: int,
        key: Optional[jax.Array] = None,
        inference: Optional[bool] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        if key is None:
            key = jax.random.PRNGKey(0)
        k_shift, k_jitter, k_rescale = jax.random.split(key, 3)

        dtype = self.dtype
        D_head = self.D_head
        D_quarter = D_head // 4

        coords = self._make_coords(H, W)

        if not inference and (self.shift_coords is not None):
            shift_hw = jax.random.uniform(
                k_shift, shape=(2,), minval=-self.shift_coords, maxval=self.shift_coords
            ).astype(dtype)
            coords = coords + shift_hw[None, :]

        if not inference and (self.jitter_coords is not None):
            if self.jitter_coords <= 0:
                raise ValueError("jitter_coords must be > 0.")
            jitter_max = jnp.log(jnp.asarray(self.jitter_coords, dtype=dtype))
            jitter_min = -jitter_max
            jitter_hw = jax.random.uniform(
                k_jitter, shape=(2,), minval=jitter_min, maxval=jitter_max
            )
            jitter_hw = jnp.exp(jitter_hw).astype(dtype)
            coords = coords * jitter_hw[None, :]

        if not inference and (self.rescale_coords is not None):
            if self.rescale_coords <= 0:
                raise ValueError("rescale_coords must be > 0.")
            rescale_max = jnp.log(jnp.asarray(self.rescale_coords, dtype=dtype))
            rescale_min = -rescale_max
            rescale = jax.random.uniform(
                k_rescale, shape=(1,), minval=rescale_min, maxval=rescale_max
            )
            rescale = jnp.exp(rescale).astype(dtype)
            coords = coords * rescale

        periods = jax.lax.stop_gradient(self.periods).astype(dtype)
        angles = (2.0 * jnp.pi * coords[:, :, None]) / periods[None, None, :]
        angles = angles.reshape(angles.shape[0], 2 * D_quarter)
        angles = jnp.tile(angles, reps=(1, 2))

        cos = jnp.cos(angles).astype(dtype)
        sin = jnp.sin(angles).astype(dtype)
        return sin, cos


def _dino_rope_inference_sincos(
    rope: DinoRoPE,
    *,
    H: int,
    W: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute DinoRoPE sin/cos caches for inference using NumPy host ops."""

    dtype = np.dtype(rope.dtype)

    if rope.normalize_coords == "max":
        denom = float(max(H, W))
        coords_h = np.arange(0.5, H, step=1.0) / denom
        coords_w = np.arange(0.5, W, step=1.0) / denom
    elif rope.normalize_coords == "min":
        denom = float(min(H, W))
        coords_h = np.arange(0.5, H, step=1.0) / denom
        coords_w = np.arange(0.5, W, step=1.0) / denom
    else:
        coords_h = np.arange(0.5, H, step=1.0) / float(H)
        coords_w = np.arange(0.5, W, step=1.0) / float(W)

    hh, ww = np.meshgrid(coords_h, coords_w, indexing="ij")
    coords = np.stack([hh, ww], axis=-1).reshape(H * W, 2)
    coords = (2.0 * coords - 1.0).astype(dtype)

    periods = np.asarray(rope.periods, dtype=dtype)
    periods = np.asarray(periods, dtype=dtype)
    D_head = rope.D_head
    D_quarter = D_head // 4

    angles = (2.0 * np.pi * coords[:, :, None]) / periods[None, None, :]
    angles = angles.reshape(angles.shape[0], 2 * D_quarter)
    angles = np.tile(angles, reps=(1, 2))

    cos = np.cos(angles).astype(dtype, copy=False)
    sin = np.sin(angles).astype(dtype, copy=False)

    return sin, cos


class DinoRotaryProcessHeads(eqx.Module):
    """process_heads adapter that rotates only the patch grid tokens."""

    _sin_data: bytes = eqx.field(static=True)
    _cos_data: bytes = eqx.field(static=True)
    _shape: tuple[int, int] = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    prefix_tokens: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        sin: np.ndarray,
        cos: np.ndarray,
        prefix_tokens: int,
    ):
        sin_arr = np.asarray(sin)
        cos_arr = np.asarray(cos)
        self._sin_data = sin_arr.tobytes()
        self._cos_data = cos_arr.tobytes()
        self._shape = tuple(int(dim) for dim in sin_arr.shape)
        self._dtype = str(sin_arr.dtype)
        self.prefix_tokens = int(prefix_tokens)

    def __call__(
        self,
        query_heads: jax.Array,
        key_heads: jax.Array,
        value_heads: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        prefix = self.prefix_tokens
        sin = jnp.asarray(self.sin, dtype=query_heads.dtype)
        cos = jnp.asarray(self.cos, dtype=query_heads.dtype)
        expected_tokens = prefix + sin.shape[0]
        if query_heads.shape[0] != expected_tokens:
            raise ValueError(
                f"Expected sequence length {expected_tokens}, "
                f"got {query_heads.shape[0]} when applying DinoRoPE."
            )

        def _apply_rope(x: jax.Array) -> jax.Array:
            if prefix == 0:
                x_tail = x
                x_prefix = jnp.zeros((0,) + x.shape[1:], dtype=x.dtype)
            else:
                x_prefix, x_tail = jnp.split(x, [prefix], axis=0)
            sin_b = sin[:, None, :]
            cos_b = cos[:, None, :]
            rotated_tail = _rotate_half_last_dim(x_tail)
            x_tail = (x_tail * cos_b) + (rotated_tail * sin_b)
            return jnp.concatenate([x_prefix, x_tail], axis=0)

        rotated_q = _apply_rope(query_heads)
        rotated_k = _apply_rope(key_heads)
        return rotated_q, rotated_k, value_heads

    def __hash__(self) -> int:
        return hash(
            (
                self._shape,
                self._dtype,
                self._sin_data,
                self._cos_data,
                self.prefix_tokens,
            )
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, DinoRotaryProcessHeads):
            return False
        return (
            self.prefix_tokens == other.prefix_tokens
            and self._shape == other._shape
            and self._dtype == other._dtype
            and self._sin_data == other._sin_data
            and self._cos_data == other._cos_data
        )

    @property
    def sin(self) -> np.ndarray:
        dtype = np.dtype(self._dtype)
        return np.frombuffer(self._sin_data, dtype=dtype).reshape(self._shape)

    @property
    def cos(self) -> np.ndarray:
        dtype = np.dtype(self._dtype)
        return np.frombuffer(self._cos_data, dtype=dtype).reshape(self._shape)


# --- Model code derived from https://github.com/clementpoiret/Equimo ---


def _apply_pointwise(module, x: Array) -> Array:
    """Apply an Equinox module independently across batch and sequence axes."""
    if module is None or isinstance(module, eqx.nn.Identity):
        return x
    apply_tokens = eqx.filter_vmap(module, in_axes=0, out_axes=0)
    apply_batch = eqx.filter_vmap(apply_tokens, in_axes=0, out_axes=0)
    return apply_batch(x)


def _exact_gelu(x: Array) -> Array:
    """Exact GELU to mirror Equimo's activation."""
    return jax.nn.gelu(x, approximate=False)


@onnx_function
class LayerScale(eqx.Module):
    """Element-wise scaling with learned gamma."""

    gamma: jax.Array

    def __init__(self, dim: int, init_value: float = 1e-5):
        self.gamma = jnp.ones((dim,), dtype=jnp.float32) * init_value

    def __call__(self, x: Array) -> Array:
        gamma = jnp.reshape(self.gamma, (1, 1, -1))
        return x * gamma


@onnx_function
class LinearLastDim(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_dim: int, out_dim: int, *, key: jax.Array):
        k_w, k_b = jax.random.split(key, 2)
        self.weight = jax.random.normal(k_w, (out_dim, in_dim))
        self.bias = jax.random.normal(k_b, (out_dim,))

    def __call__(self, x: Array) -> Array:
        out = jnp.matmul(x, jnp.transpose(self.weight))
        out = out + self.bias
        return out


@onnx_function
class DinoMlp(eqx.Module):
    fc1: LinearLastDim
    fc2: LinearLastDim

    def __init__(self, dim: int, hidden_dim: int, *, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        self.fc1 = LinearLastDim(dim, hidden_dim, key=k1)
        self.fc2 = LinearLastDim(hidden_dim, dim, key=k2)

    def __call__(self, x: Array) -> Array:
        hidden = self.fc1(x)
        hidden = _exact_gelu(hidden)
        return self.fc2(hidden)


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
class MultiHeadAttention(eqx.Module):
    """Multi-Head Self-Attention driven by Equinox primitives."""

    core: AttentionCore
    rope: eqx.nn.RotaryPositionalEmbedding
    process_heads: eqx.Module
    num_heads: int

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: jax.Array,
        process_heads: Optional[eqx.Module] = None,
    ):
        self.num_heads = num_heads
        self.core = AttentionCore(dim=dim, num_heads=num_heads, key=key)
        head_dim = dim // num_heads
        self.rope = eqx.nn.RotaryPositionalEmbedding(embedding_size=head_dim)
        self.process_heads = process_heads or RotaryProcessHeads(self.rope)

    def __call__(
        self,
        x: Array,
        *,
        process_heads: Optional[eqx.Module] = None,
    ) -> Array:
        proc = process_heads or self.process_heads

        def _attend(tokens: Array) -> Array:
            return self.core.attn(
                tokens,
                tokens,
                tokens,
                inference=True,
                process_heads=proc,
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
                MultiHeadAttention, dim=384, num_heads=6, key=with_prng_key(0)
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
    attn: MultiHeadAttention
    post_attn_norm: Optional[eqx.Module]
    norm2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    ls1: eqx.Module
    ls2: eqx.Module

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        key: jax.Array,
        process_heads: Optional[eqx.Module] = None,
    ):
        keys = jax.random.split(key, 2)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim,
            num_heads=num_heads,
            key=keys[0],
            process_heads=process_heads,
        )
        self.post_attn_norm = None
        self.norm2 = eqx.nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DinoMlp(dim=dim, hidden_dim=mlp_hidden_dim, key=keys[1])
        self.ls1 = LayerScale(dim)
        self.ls2 = LayerScale(dim)

    def __call__(
        self,
        x: Array,
        *,
        process_heads: Optional[eqx.Module] = None,
    ) -> Array:
        out, _ = self._forward_internal(x, process_heads=process_heads)
        return out

    def forward_debug(
        self,
        x: Array,
        *,
        process_heads: Optional[eqx.Module] = None,
    ) -> tuple[Array, dict[str, Array]]:
        return self._forward_internal(x, process_heads=process_heads)

    def _forward_internal(
        self,
        x: Array,
        *,
        process_heads: Optional[eqx.Module] = None,
    ) -> tuple[Array, dict[str, Array]]:
        attn_in = _apply_pointwise(self.norm1, x)
        attn_raw = self.attn(attn_in, process_heads=process_heads)
        attn_norm = _apply_pointwise(self.post_attn_norm, attn_raw)
        attn_scaled = self.ls1(attn_norm)
        post_attn = x + attn_scaled

        mlp_in = _apply_pointwise(self.norm2, post_attn)
        mlp_raw = self.mlp(mlp_in)
        mlp_scaled = self.ls2(mlp_raw)
        out = post_attn + mlp_scaled

        debug = {
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
        return out, debug


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
                must_absent=["Identity"],
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
    storage_tokens: jax.Array | None
    blocks: list[Block]
    norm: eqx.nn.LayerNorm
    num_storage_tokens: int
    dino_rope: DinoRoPE

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        num_storage_tokens: int = 0,
        *,
        key: jax.Array,
    ):
        num_storage_tokens = int(num_storage_tokens)
        self.num_storage_tokens = num_storage_tokens
        extra_keys = 1 if num_storage_tokens > 0 else 0
        total_splits = depth + 2 + extra_keys
        keys = jax.random.split(key, total_splits)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            key=keys[0],
        )
        self.cls_token = jax.random.normal(keys[1], (1, 1, embed_dim))
        if num_storage_tokens > 0:
            storage_key = keys[2]
            self.storage_tokens = jax.random.normal(
                storage_key, (1, num_storage_tokens, embed_dim)
            )
            block_keys = keys[3:]
        else:
            self.storage_tokens = None
            block_keys = keys[2:]
        self.dino_rope = DinoRoPE(dim=embed_dim, num_heads=num_heads)
        self.blocks = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                key=k,
                process_heads=None,
            )
            for k in block_keys
        ]
        self.norm = eqx.nn.LayerNorm(embed_dim)

    def __call__(self, x: Array) -> Array:
        tokens = self._encode(x)
        tokens = _apply_pointwise(self.norm, tokens)
        if self.num_storage_tokens:
            cls = tokens[:, :1, :]
            patches = tokens[:, 1 + self.num_storage_tokens :, :]
            return jnp.concatenate([cls, patches], axis=1)
        return tokens

    def _encode(self, x: Array, *, capture: bool = False):
        x = self.patch_embed(x)
        cls_tokens = jnp.broadcast_to(self.cls_token, (x.shape[0], 1, x.shape[-1]))
        if self.num_storage_tokens and self.storage_tokens is not None:
            storage_tokens = jnp.broadcast_to(
                self.storage_tokens, (x.shape[0], self.num_storage_tokens, x.shape[-1])
            )
            x = jnp.concatenate([cls_tokens, storage_tokens, x], axis=1)
        else:
            x = jnp.concatenate([cls_tokens, x], axis=1)
        grid_size = int(math.isqrt(self.patch_embed.num_patches))
        sin, cos = _dino_rope_inference_sincos(
            self.dino_rope,
            H=grid_size,
            W=grid_size,
        )
        prefix_tokens = 1 + self.num_storage_tokens
        process_heads = DinoRotaryProcessHeads(
            sin=sin,
            cos=cos,
            prefix_tokens=prefix_tokens,
        )
        history: list[jax.Array] = []
        for blk in self.blocks:
            x = blk(x, process_heads=process_heads)
            if capture:
                history.append(x)
        if capture:
            return x, history
        return x

    def forward_features(self, x: Array) -> dict[str, Array]:
        """Return Equimo-style feature dictionary for analysis."""
        tokens = self._encode(x)
        tokens_norm = _apply_pointwise(self.norm, tokens)
        cls_norm = tokens_norm[:, 0, :]
        if self.num_storage_tokens:
            reg_norm = tokens_norm[:, 1 : 1 + self.num_storage_tokens, :]
            patch_start = 1 + self.num_storage_tokens
        else:
            reg_norm = jnp.empty(
                (tokens.shape[0], 0, tokens.shape[-1]),
                dtype=tokens_norm.dtype,
            )
            patch_start = 1
        patch_norm = tokens_norm[:, patch_start:, :]
        return {
            "x_norm_cls_token": cls_norm,
            "x_norm_reg_tokens": reg_norm,
            "x_norm_patchtokens": patch_norm,
            "x_prenorm": tokens,
        }

    def block_outputs(self, x: Array) -> list[jax.Array]:
        """Return tokens after each block for debugging."""
        _, history = self._encode(x, capture=True)
        return history

    def block_debug_outputs(self, x: Array) -> list[dict[str, Array]]:
        """Return detailed per-block activations (pre/post attention & MLP)."""
        tokens = self.patch_embed(x)
        cls_tokens = jnp.broadcast_to(
            self.cls_token, (tokens.shape[0], 1, tokens.shape[-1])
        )
        if self.num_storage_tokens and self.storage_tokens is not None:
            storage_tokens = jnp.broadcast_to(
                self.storage_tokens,
                (tokens.shape[0], self.num_storage_tokens, tokens.shape[-1]),
            )
            tokens = jnp.concatenate([cls_tokens, storage_tokens, tokens], axis=1)
        else:
            tokens = jnp.concatenate([cls_tokens, tokens], axis=1)

        grid_size = int(math.isqrt(self.patch_embed.num_patches))
        sin, cos = _dino_rope_inference_sincos(
            self.dino_rope,
            H=grid_size,
            W=grid_size,
        )
        prefix_tokens = 1 + self.num_storage_tokens
        process_heads = DinoRotaryProcessHeads(
            sin=sin,
            cos=cos,
            prefix_tokens=prefix_tokens,
        )

        debug_infos: list[dict[str, Array]] = []
        current = tokens
        for blk in self.blocks:
            entry: dict[str, Array] = {"input": current}
            out, dbg = blk.forward_debug(current, process_heads=process_heads)
            entry.update(dbg)
            debug_infos.append(entry)
            current = out
        return debug_infos


def _get_test_cases():
    """Returns a list of test cases for DINOv3."""
    test_cases = []
    img_size = 224

    vit_configs = {
        "Ti14": {"patch": 14, "dim": 192, "heads": 3, "depth": 12, "storage": 0},
        # S14 occasionally exhibits larger numeric drift; loosen tolerance a bit.
        "S14": {
            "patch": 14,
            "dim": 384,
            "heads": 6,
            "depth": 12,
            "storage": 0,
            "rtol": 1.0,
            "atol": 1.0,
        },
        # B14 runs at larger hidden size; allow slightly looser numeric tolerance.
        "B14": {
            "patch": 14,
            "dim": 768,
            "heads": 12,
            "depth": 12,
            "storage": 0,
            "rtol": 2.0,
            "atol": 2.0,
        },
        # S16 can drift on CI hardware; loosen tolerance slightly.
        "S16": {
            "patch": 16,
            "dim": 384,
            "heads": 6,
            "depth": 12,
            "storage": 4,
            "rtol": 1.0,
            "atol": 1.0,
        },
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
                    num_storage_tokens=config.get("storage", 0),
                    key=with_prng_key(idx),
                ),
                "input_shapes": [("B", 3, img_size, img_size)],
                "rtol": config.get("rtol", 5e-1),
                "atol": config.get("atol", 5e-1),
                "post_check_onnx_graph": EG(
                    [f"VisionTransformer:{output_shape}"],
                    symbols={"B": None},
                    no_unused_inputs=True,
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
