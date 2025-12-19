# jax2onnx/plugins/examples/nnx/dinov3.py

"""Flax/NNX implementation of the DINOv3 Vision Transformer for ONNX export."""

from __future__ import annotations

import math
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    register_example,
    with_rng_seed,
)


class _KeySeq:
    """Lightweight PRNG splitter for deterministic nnx module init."""

    def __init__(self, key: jax.Array | int | None):
        if key is None:
            key = 0
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)
        self._key = key

    def next(self) -> jax.Array:
        self._key, sub = jax.random.split(self._key)
        return sub


def _rotate_half_last_dim(x: jax.Array) -> jax.Array:
    """Rotate pairs in the last dimension by 90 degrees."""
    half = x.shape[-1] // 2
    first, second = jnp.split(x, [half], axis=-1)
    return jnp.concatenate([-second, first], axis=-1)


@onnx_function
class DinoRoPE(nnx.Module):
    """2D rotary embedding helper mirroring Equimo's DinoRoPE behaviour."""

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

        self.periods = nnx.data(periods.astype(dtype))

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
    ) -> tuple[jax.Array, jax.Array]:
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

        periods = jax.lax.stop_gradient(jnp.asarray(self.periods)).astype(dtype)
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


class DinoRotaryProcessHeads(nnx.Module):
    """process_heads adapter that rotates only the patch grid tokens."""

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
        if query_heads.shape[2] != expected_tokens:
            raise ValueError(
                f"Expected sequence length {expected_tokens}, "
                f"got {query_heads.shape[2]} when applying DinoRoPE."
            )

        def _apply_rope(x: jax.Array) -> jax.Array:
            if prefix == 0:
                x_tail = x
                x_prefix = jnp.zeros(
                    (x.shape[0], x.shape[1], 0, x.shape[3]), dtype=x.dtype
                )
            else:
                x_prefix, x_tail = jnp.split(x, [prefix], axis=2)
            sin_b = sin[None, None, :, :]
            cos_b = cos[None, None, :, :]
            rotated_tail = _rotate_half_last_dim(x_tail)
            x_tail = (x_tail * cos_b) + (rotated_tail * sin_b)
            return jnp.concatenate([x_prefix, x_tail], axis=2)

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


def _exact_gelu(x: jax.Array) -> jax.Array:
    """Exact GELU to mirror Equimo's activation."""
    return jax.nn.gelu(x, approximate=False)


@onnx_function
class LayerScale(nnx.Module):
    """Element-wise scaling with learned gamma."""

    def __init__(self, dim: int, init_value: float = 1e-5):
        self.gamma = nnx.Param(jnp.ones((dim,), dtype=jnp.float32) * init_value)

    def __call__(self, x: jax.Array) -> jax.Array:
        gamma = jnp.reshape(self.gamma.value, (1, 1, -1))
        return x * gamma


class LinearLastDim(nnx.Module):
    """Simple linear layer over the last dimension."""

    def __init__(self, in_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        k_w, k_b = rngs.params(), rngs.params()
        self.weight = nnx.Param(jax.random.normal(k_w, (out_dim, in_dim)))
        self.bias = nnx.Param(jax.random.normal(k_b, (out_dim,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        out = jnp.matmul(x, jnp.transpose(self.weight.value))
        out = out + self.bias.value
        return out


@onnx_function
class DinoMlp(nnx.Module):
    def __init__(self, dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        keys = _KeySeq(rngs.params())
        self.fc1 = LinearLastDim(dim, hidden_dim, rngs=nnx.Rngs(keys.next()))
        self.fc2 = LinearLastDim(hidden_dim, dim, rngs=nnx.Rngs(keys.next()))

    def __call__(self, x: jax.Array) -> jax.Array:
        hidden = self.fc1(x)
        hidden = _exact_gelu(hidden)
        return self.fc2(hidden)


@onnx_function
class PatchEmbed(nnx.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nnx.Conv(
            in_features=in_chans,
            out_features=embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        x = self.proj(x_nhwc)
        x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))
        return x


register_example(
    component="NnxDinoPatchEmbed",
    description="Image to Patch Embedding.",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.10.3",
    context="examples.nnx_dino",
    children=["flax.nnx.Conv"],
    testcases=[
        {
            "testcase": "nnx_patch_embed",
            "callable": construct_and_call(
                PatchEmbed,
                img_size=224,
                patch_size=14,
                embed_dim=384,
                rngs=with_rng_seed(0),
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


class AttentionCore(nnx.Module):
    """Multi-Head Self-Attention built from nnx.Linear blocks."""

    def __init__(self, dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        process_heads: Optional[nnx.Module] = None,
    ) -> jax.Array:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        B, T, D = q.shape
        H = self.num_heads
        head_dim = self.head_dim
        q_heads = jnp.transpose(q.reshape(B, T, H, head_dim), (0, 2, 1, 3))
        k_heads = jnp.transpose(k.reshape(B, T, H, head_dim), (0, 2, 1, 3))
        v_heads = jnp.transpose(v.reshape(B, T, H, head_dim), (0, 2, 1, 3))

        if process_heads is not None:
            q_heads, k_heads, v_heads = process_heads(q_heads, k_heads, v_heads)

        attn_logits = jnp.matmul(q_heads, jnp.swapaxes(k_heads, -1, -2))
        scale = 1.0 / math.sqrt(float(head_dim))
        attn_logits = attn_logits * jnp.asarray(scale, dtype=attn_logits.dtype)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_out = jnp.matmul(attn_weights, v_heads)
        attn_out = jnp.swapaxes(attn_out, 1, 2).reshape(B, T, D)
        return self.out_proj(attn_out)


register_example(
    component="NnxDinoAttentionCore",
    description="Multi-Head Self-Attention without rotary processing.",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.10.3",
    context="examples.nnx_dino",
    children=[
        "flax.nnx.Linear",
    ],
    testcases=[
        {
            "testcase": "nnx_attention_core",
            "callable": construct_and_call(
                AttentionCore, dim=384, num_heads=6, rngs=with_rng_seed(0)
            ),
            "input_shapes": [("B", 257, 384)],
            "post_check_onnx_graph": EG(
                [
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
class MultiHeadAttention(nnx.Module):
    """Multi-Head Self-Attention driven by nnx primitives."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
        process_heads: Optional[nnx.Module] = None,
    ):
        self.num_heads = num_heads
        self.core = AttentionCore(dim=dim, num_heads=num_heads, rngs=rngs)
        self.process_heads = process_heads

    def __call__(
        self,
        x: jax.Array,
        *,
        process_heads: Optional[nnx.Module] = None,
    ) -> jax.Array:
        proc = process_heads or self.process_heads
        return self.core(x, process_heads=proc)


register_example(
    component="NnxDinoAttention",
    description="Multi-Head Self-Attention using Flax/NNX modules.",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.10.3",
    context="examples.nnx_dino",
    children=[
        "AttentionCore",
    ],
    testcases=[
        {
            "testcase": "nnx_attention",
            "callable": construct_and_call(
                MultiHeadAttention, dim=384, num_heads=6, rngs=with_rng_seed(0)
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
class Block(nnx.Module):
    """Transformer Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        rngs: nnx.Rngs,
        process_heads: Optional[nnx.Module] = None,
    ):
        keys = _KeySeq(rngs.params())
        self.norm1 = nnx.LayerNorm(dim, rngs=nnx.Rngs(keys.next()))
        self.attn = MultiHeadAttention(
            dim,
            num_heads=num_heads,
            rngs=nnx.Rngs(keys.next()),
            process_heads=process_heads,
        )
        self.post_attn_norm = None
        self.norm2 = nnx.LayerNorm(dim, rngs=nnx.Rngs(keys.next()))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DinoMlp(
            dim=dim, hidden_dim=mlp_hidden_dim, rngs=nnx.Rngs(keys.next())
        )
        self.ls1 = LayerScale(dim)
        self.ls2 = LayerScale(dim)

    def __call__(
        self,
        x: jax.Array,
        *,
        process_heads: Optional[nnx.Module] = None,
    ) -> jax.Array:
        out, _ = self._forward_internal(x, process_heads=process_heads)
        return out

    def forward_debug(
        self,
        x: jax.Array,
        *,
        process_heads: Optional[nnx.Module] = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        return self._forward_internal(x, process_heads=process_heads)

    def _forward_internal(
        self,
        x: jax.Array,
        *,
        process_heads: Optional[nnx.Module] = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        attn_in = self.norm1(x)
        attn_raw = self.attn(attn_in, process_heads=process_heads)
        attn_norm = (
            attn_raw if self.post_attn_norm is None else self.post_attn_norm(attn_raw)
        )
        attn_scaled = self.ls1(attn_norm)
        post_attn = x + attn_scaled

        mlp_in = self.norm2(post_attn)
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
    component="NnxDinoBlock",
    description="Transformer Block.",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.10.3",
    context="examples.nnx_dino",
    children=["flax.nnx.LayerNorm", "Attention", "DinoMlp"],
    testcases=[
        {
            "testcase": "nnx_transformer_block",
            "callable": construct_and_call(
                Block, dim=384, num_heads=6, rngs=with_rng_seed(0)
            ),
            "input_shapes": [("B", 257, 384)],
            # CI hardware shows slightly larger numeric drift on this block.
            "rtol": 5e-2,
            "atol": 5e-2,
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
class VisionTransformer(nnx.Module):
    """Vision Transformer."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        num_storage_tokens: int = 0,
        *,
        rngs: nnx.Rngs,
    ):
        num_storage_tokens = int(num_storage_tokens)
        self.num_storage_tokens = num_storage_tokens
        key_seq = _KeySeq(rngs.params())

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            rngs=nnx.Rngs(key_seq.next()),
        )
        self.cls_token = nnx.Param(jax.random.normal(key_seq.next(), (1, 1, embed_dim)))
        if num_storage_tokens > 0:
            self.storage_tokens = nnx.Param(
                jax.random.normal(key_seq.next(), (1, num_storage_tokens, embed_dim))
            )
        else:
            self.storage_tokens = None
        self.dino_rope = DinoRoPE(dim=embed_dim, num_heads=num_heads)
        self.blocks = nnx.List(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    rngs=nnx.Rngs(key_seq.next()),
                    process_heads=None,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nnx.LayerNorm(embed_dim, rngs=nnx.Rngs(key_seq.next()))

    def __call__(self, x: jax.Array) -> jax.Array:
        tokens = self._encode(x)
        tokens = self.norm(tokens)
        if self.num_storage_tokens:
            cls = tokens[:, :1, :]
            patches = tokens[:, 1 + self.num_storage_tokens :, :]
            return jnp.concatenate([cls, patches], axis=1)
        return tokens

    def _encode(self, x: jax.Array, *, capture: bool = False):
        x = self.patch_embed(x)
        cls_tokens = jnp.broadcast_to(
            self.cls_token.value, (x.shape[0], 1, x.shape[-1])
        )
        if self.num_storage_tokens and self.storage_tokens is not None:
            storage_tokens = jnp.broadcast_to(
                self.storage_tokens.value,
                (x.shape[0], self.num_storage_tokens, x.shape[-1]),
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

    def forward_features(self, x: jax.Array) -> dict[str, jax.Array]:
        """Return Equimo-style feature dictionary for analysis."""
        tokens = self._encode(x)
        tokens_norm = self.norm(tokens)
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

    def block_outputs(self, x: jax.Array) -> list[jax.Array]:
        """Return tokens after each block for debugging."""
        _, history = self._encode(x, capture=True)
        return history

    def block_debug_outputs(self, x: jax.Array) -> list[dict[str, jax.Array]]:
        """Return detailed per-block activations (pre/post attention & MLP)."""
        tokens = self.patch_embed(x)
        cls_tokens = jnp.broadcast_to(
            self.cls_token.value, (tokens.shape[0], 1, tokens.shape[-1])
        )
        if self.num_storage_tokens and self.storage_tokens is not None:
            storage_tokens = jnp.broadcast_to(
                self.storage_tokens.value,
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

        debug_infos: list[dict[str, jax.Array]] = []
        current = tokens
        for blk in self.blocks:
            entry: dict[str, jax.Array] = {"input": current}
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
        "S14": {
            "patch": 14,
            "dim": 384,
            "heads": 6,
            "depth": 12,
            "storage": 0,
            "rtol": 1.0,
            "atol": 1.0,
        },
        "B14": {
            "patch": 14,
            "dim": 768,
            "heads": 12,
            "depth": 12,
            "storage": 0,
            "rtol": 1.0,
            "atol": 1.0,
        },
        "S16": {"patch": 16, "dim": 384, "heads": 6, "depth": 12, "storage": 4},
    }

    for idx, (name, config) in enumerate(vit_configs.items()):
        num_patches = (img_size // config["patch"]) ** 2
        output_shape = f"Bx{num_patches + 1}x{config['dim']}"

        test_cases.append(
            {
                "testcase": f"nnx_dinov3_vit_{name}",
                "callable": construct_and_call(
                    VisionTransformer,
                    img_size=img_size,
                    patch_size=config["patch"],
                    embed_dim=config["dim"],
                    depth=config["depth"],
                    num_heads=config["heads"],
                    num_storage_tokens=config.get("storage", 0),
                    rngs=with_rng_seed(idx),
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


register_example(
    component="FlaxDINOv3VisionTransformer",
    description="DINOv3 Vision Transformer",
    source="https://github.com/clementpoiret/Equimo",
    since="v0.10.3",
    context="examples.nnx_dino",
    children=["PatchEmbed", "Block"],
    testcases=_get_test_cases(),
)
