# jax2onnx/plugins/examples/eqx/gpt_oss.py

"""Equinox implementation of the GPT-OSS Transformer for ONNX conversion tests."""

from __future__ import annotations

import dataclasses
import gc
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import nn, lax
import jax.core as jax_core

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY,
    construct_and_call,
    onnx_function,
    register_example,
    with_prng_key,
)


@dataclasses.dataclass(frozen=True, slots=True)
class GPTOSSConfig:
    """Hyper-parameters describing a GPT-OSS Transformer stack."""

    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201_088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4_096
    rope_theta: float = 150_000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


def _swiglu(x: jnp.ndarray, *, limit: float, alpha: float = 1.702) -> jnp.ndarray:
    """Compute the SwiGLU activation on the final dimension."""

    x_glu = jnp.clip(x[..., ::2], a_min=None, a_max=limit)
    x_linear = jnp.clip(x[..., 1::2], a_min=-limit, a_max=limit)
    out_glu = x_glu * nn.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1.0)


def _build_causal_mask(
    seq_len: int, *, sliding_window: int, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """Return a causal + optional sliding-window mask suitable for attention logits."""
    seq_len_int = int(seq_len)
    mask = np.triu(np.full((seq_len_int, seq_len_int), -np.inf, dtype=np.float32), k=1)
    if sliding_window > 0:
        lower = np.tril(
            np.full((seq_len_int, seq_len_int), -np.inf, dtype=np.float32),
            k=-sliding_window,
        )
        mask = mask + lower
    mask = mask.reshape(1, 1, 1, seq_len_int, seq_len_int)
    return jnp.asarray(mask, dtype=dtype)


def _apply_pointwise(module, x: jnp.ndarray) -> jnp.ndarray:
    """Apply an Equinox module independently across batch and sequence axes."""

    if module is None or isinstance(module, eqx.nn.Identity):
        return x
    apply_tokens = eqx.filter_vmap(module, in_axes=0, out_axes=0)
    apply_batch = eqx.filter_vmap(apply_tokens, in_axes=0, out_axes=0)
    return apply_batch(x)


def _resolve_seq_length(length, query: jnp.ndarray) -> int:
    if isinstance(length, (int, np.integer)):
        return int(length)
    try:
        return int(np.asarray(length).item())
    except Exception:
        pass

    shape = getattr(query, "shape", None)
    if shape is not None and len(shape) > 1 and isinstance(shape[1], (int, np.integer)):
        return int(shape[1])

    return jax_core.concrete_or_error(
        int,
        length,
        "RotaryEmbedding requires a static sequence length.",
    )


def _apply_linear_nd(linear: eqx.nn.Linear, x: jnp.ndarray) -> jnp.ndarray:
    """Apply an Equinox linear module across the leading dimensions of ``x``."""

    apply_seq = eqx.filter_vmap(linear, in_axes=0, out_axes=0)
    apply_batch = eqx.filter_vmap(apply_seq, in_axes=0, out_axes=0)
    return apply_batch(x)


def _apply_linear_float32_accum(linear: eqx.nn.Linear, x: jnp.ndarray) -> jnp.ndarray:
    """Apply a linear layer while accumulating in float32 (Torch bf16 semantics)."""

    weight = jnp.asarray(linear.weight, dtype=jnp.float32)
    bias = (
        jnp.asarray(linear.bias, dtype=jnp.float32)
        if linear.bias is not None
        else None
    )

    orig_shape = x.shape[:-1]
    features_in = weight.shape[1]
    x_flat = jnp.reshape(x, (-1, features_in)).astype(jnp.float32)
    y_flat = jnp.matmul(x_flat, weight.T)
    if bias is not None:
        y_flat = y_flat + bias

    y = jnp.reshape(y_flat, orig_shape + (weight.shape[0],))
    target_dtype = getattr(linear.weight, "dtype", None)
    if target_dtype is not None:
        y = y.astype(target_dtype)
    return y


@onnx_function
class RMSNorm(eqx.Module):
    """RMS normalization that mirrors the GPT-OSS torch implementation."""

    weight: jnp.ndarray
    eps: float = eqx.field(static=True, default=1e-5)

    def __init__(self, hidden_size: int, *, eps: float = 1e-5):
        self.weight = jnp.ones((hidden_size,), dtype=jnp.float32)
        self.eps = float(eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_f32 = x.astype(jnp.float32)
        rms = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        normalized = x_f32 * jax.lax.rsqrt(rms + self.eps)
        scaled = normalized * self.weight
        return scaled.astype(x.dtype)


@onnx_function
class RotaryEmbedding(eqx.Module):
    """YaRN rotary embedding used by GPT-OSS attention blocks."""

    head_dim: int = eqx.field(static=True)
    base: float = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    initial_context_length: int = eqx.field(static=True)
    scaling_factor: float = eqx.field(static=True)
    ntk_alpha: float = eqx.field(static=True)
    ntk_beta: float = eqx.field(static=True)
    _concentration: np.ndarray = eqx.field(static=True)
    _inv_freq: np.ndarray = eqx.field(static=True)
    _cos_cache: np.ndarray = eqx.field(static=True)
    _sin_cache: np.ndarray = eqx.field(static=True)
    _concentration: jnp.ndarray
    _inv_freq: jnp.ndarray

    def __init__(self, config: GPTOSSConfig, dtype: jnp.dtype = jnp.float32):
        self.head_dim = int(config.head_dim)
        self.base = float(config.rope_theta)
        self.dtype = dtype
        self.initial_context_length = int(config.initial_context_length)
        self.scaling_factor = float(config.rope_scaling_factor)
        self.ntk_alpha = float(config.rope_ntk_alpha)
        self.ntk_beta = float(config.rope_ntk_beta)

        head_dim = self.head_dim
        base = np.float32(self.base)
        steps = np.arange(0, head_dim, 2, dtype=np.float32)
        freq = np.power(base, steps / np.float32(head_dim))
        if self.scaling_factor > 1.0:
            concentration = (
                np.float32(0.1) * np.log(np.float32(self.scaling_factor)) + 1.0
            )
            d_half = np.float32(head_dim / 2.0)
            log_base = np.log(base)
            low = (
                d_half
                * np.log(
                    np.float32(self.initial_context_length)
                    / (np.float32(self.ntk_beta) * np.float32(2.0 * np.pi))
                )
                / log_base
            )
            high = (
                d_half
                * np.log(
                    np.float32(self.initial_context_length)
                    / (np.float32(self.ntk_alpha) * np.float32(2.0 * np.pi))
                )
                / log_base
            )
            ramp = (np.arange(head_dim // 2, dtype=np.float32) - low) / (high - low)
            mask = 1.0 - np.clip(ramp, 0.0, 1.0)
            interpolation = 1.0 / (np.float32(self.scaling_factor) * freq)
            extrapolation = 1.0 / freq
            inv_freq = interpolation * (1.0 - mask) + extrapolation * mask
        else:
            concentration = np.float32(1.0)
            inv_freq = 1.0 / freq

        self._concentration = concentration.astype(np.float32)
        self._inv_freq = inv_freq.astype(np.float32)

        max_len = int(self.initial_context_length)
        positions = np.arange(max_len, dtype=np.float32)
        freqs = np.einsum("i,j->ij", positions, self._inv_freq)
        self._cos_cache = np.cos(freqs) * self._concentration
        self._sin_cache = np.sin(freqs) * self._concentration

    def compute_sin_cos(self, seq_len: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        try:
            seq_len_int = int(seq_len)
        except TypeError:
            seq_len_int = jax_core.concrete_or_error(
                int,
                seq_len,
                "RotaryEmbedding requires a static sequence length.",
            )

        cos = jnp.asarray(self._cos_cache[:seq_len_int], dtype=self.dtype)
        sin = jnp.asarray(self._sin_cache[:seq_len_int], dtype=self.dtype)
        return cos, sin

    @staticmethod
    def _broadcast_cache(cache: jnp.ndarray, target_ndim: int) -> jnp.ndarray:
        seq_len = int(cache.shape[0])
        half = int(cache.shape[1])
        target_shape = (1, seq_len) + (1,) * max(target_ndim - 3, 0) + (half,)
        broadcast_dims = (1, len(target_shape) - 1)
        return lax.broadcast_in_dim(cache, target_shape, broadcast_dims)

    @staticmethod
    def _apply_rotary(
        tensor: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
    ) -> jnp.ndarray:
        tensor_dtype = tensor.dtype
        if tensor.ndim < 2:
            raise ValueError("RotaryEmbedding expects tensors with rank ≥ 2.")

        seq_axis = 0 if tensor.ndim == 2 else 1
        seq_len = tensor.shape[seq_axis]
        head_dim = tensor.shape[-1]

        tensor_moved = jnp.moveaxis(tensor, seq_axis, 0)
        cos_moved = jnp.moveaxis(cos, seq_axis, 0)
        sin_moved = jnp.moveaxis(sin, seq_axis, 0)

        leading_shape = tensor_moved.shape[1:-1]

        flat = tensor_moved.reshape(seq_len, -1, head_dim)
        cos_flat = cos_moved.reshape(seq_len, -1, head_dim // 2)
        sin_flat = sin_moved.reshape(seq_len, -1, head_dim // 2)

        first, second = jnp.split(flat, 2, axis=-1)
        cos_cast = cos_flat.astype(tensor_dtype)
        sin_cast = sin_flat.astype(tensor_dtype)
        first_cast = first.astype(tensor_dtype)
        second_cast = second.astype(tensor_dtype)
        out_first = first_cast * cos_cast - second_cast * sin_cast
        out_second = second_cast * cos_cast + first_cast * sin_cast
        rotated_flat = jnp.concatenate([out_first, out_second], axis=-1)

        rotated = rotated_flat.reshape((seq_len,) + leading_shape + (head_dim,))
        rotated = jnp.moveaxis(rotated, 0, seq_axis)
        return rotated.astype(tensor_dtype)

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        *,
        seq_len: Optional[int] = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        seq_len_candidate = seq_len if seq_len is not None else query.shape[1]
        seq_len = _resolve_seq_length(seq_len_candidate, query)

        cos, sin = self.compute_sin_cos(seq_len)
        cos_q = self._broadcast_cache(cos, query.ndim).astype(query.dtype)
        sin_q = self._broadcast_cache(sin, query.ndim).astype(query.dtype)
        cos_k = self._broadcast_cache(cos, key.ndim).astype(key.dtype)
        sin_k = self._broadcast_cache(sin, key.ndim).astype(key.dtype)
        rotated_query = self._apply_rotary(query, cos_q, sin_q)
        rotated_key = self._apply_rotary(key, cos_k, sin_k)
        return rotated_query, rotated_key


PLUGIN_REGISTRY[
    "onnx_fn::jax2onnx.plugins.examples.eqx.gpt_oss.RotaryEmbedding"
].primitive.multiple_results = True


@onnx_function
class AttentionBlock(eqx.Module):
    """Self-attention block with sinks and optional sliding-window masking."""

    config: GPTOSSConfig = eqx.field(static=True)
    layer_idx: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    num_attention_heads: int = eqx.field(static=True)
    num_key_value_heads: int = eqx.field(static=True)
    query_multiplicity: int = eqx.field(static=True)
    sliding_window: int = eqx.field(static=True)
    sm_scale: float = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    norm: RMSNorm
    rope: RotaryEmbedding
    qkv: eqx.nn.Linear
    out: eqx.nn.Linear
    sinks: jnp.ndarray

    def __init__(
        self,
        config: GPTOSSConfig,
        layer_idx: int,
        *,
        key: jax.Array,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.layer_idx = int(layer_idx)
        self.head_dim = int(config.head_dim)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads."
            )
        self.query_multiplicity = self.num_attention_heads // self.num_key_value_heads
        self.sliding_window = int(config.sliding_window) if (layer_idx % 2 == 0) else 0
        self.sm_scale = float(1.0 / jnp.sqrt(float(self.head_dim)))
        self.param_dtype = param_dtype

        qkv_dim = self.head_dim * (
            self.num_attention_heads + 2 * self.num_key_value_heads
        )
        keys = jax.random.split(key, 3)
        self.norm = RMSNorm(config.hidden_size)
        self.rope = RotaryEmbedding(config, dtype=jnp.float32)
        self.qkv = eqx.nn.Linear(
            config.hidden_size,
            qkv_dim,
            use_bias=True,
            key=keys[0],
            dtype=param_dtype,
        )
        self.out = eqx.nn.Linear(
            self.head_dim * self.num_attention_heads,
            config.hidden_size,
            use_bias=True,
            key=keys[1],
            dtype=param_dtype,
        )
        self.sinks = jax.random.normal(
            keys[2], (self.num_attention_heads,), dtype=param_dtype
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(
                "AttentionBlock expects inputs shaped (batch, seq, hidden)."
            )
        batch, seq_len_dim, hidden = x.shape
        if hidden != self.config.hidden_size:
            raise ValueError(
                f"Hidden size mismatch: expected {self.config.hidden_size}, got {hidden}."
            )
        seq_len = _resolve_seq_length(seq_len_dim, x)

        normed = _apply_pointwise(self.norm, x)
        if self.param_dtype == jnp.bfloat16:
            qkv = _apply_linear_float32_accum(self.qkv, normed)
        else:
            qkv = _apply_linear_nd(self.qkv, normed)

        split_q = self.num_attention_heads * self.head_dim
        split_k = split_q + self.num_key_value_heads * self.head_dim
        q = qkv[..., :split_q]
        k = qkv[..., split_q:split_k]
        v = qkv[..., split_k:]

        q = q.reshape(
            batch,
            seq_len,
            self.num_key_value_heads,
            self.query_multiplicity,
            self.head_dim,
        )
        k = k.reshape(batch, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_key_value_heads, self.head_dim)

        q, k = self.rope(q, k, seq_len=seq_len)

        k_broadcast = jnp.broadcast_to(
            k[..., None, :],
            (
                batch,
                seq_len,
                self.num_key_value_heads,
                self.query_multiplicity,
                self.head_dim,
            ),
        )
        v_broadcast = jnp.broadcast_to(
            v[..., None, :],
            (
                batch,
                seq_len,
                self.num_key_value_heads,
                self.query_multiplicity,
                self.head_dim,
            ),
        )

        logits = jnp.einsum(
            "btmrd,bsmrd->bmrts",
            q,
            k_broadcast,
            optimize="optimal",
        )
        logits = logits * self.sm_scale
        mask = _build_causal_mask(
            seq_len,
            sliding_window=self.sliding_window,
            dtype=logits.dtype,
        )
        logits = logits + mask

        sinks = self.sinks.reshape(self.num_key_value_heads, self.query_multiplicity)
        sink_logits = sinks[:, :, None, None]
        sink_logits = jnp.broadcast_to(
            sink_logits,
            (self.num_key_value_heads, self.query_multiplicity, seq_len, 1),
        )
        sink_logits = sink_logits.astype(logits.dtype)
        sink_logits = jnp.broadcast_to(
            sink_logits[None, ...],
            (batch, self.num_key_value_heads, self.query_multiplicity, seq_len, 1),
        )

        extended_logits = jnp.concatenate([logits, sink_logits], axis=-1)
        weights = nn.softmax(extended_logits, axis=-1)[..., :-1]

        attn = jnp.einsum(
            "bmrqk,bkmrd->bqmrd",
            weights,
            v_broadcast,
            optimize="optimal",
        )
        attn = attn.reshape(
            batch,
            seq_len,
            self.num_attention_heads * self.head_dim,
        )
        if self.param_dtype == jnp.bfloat16:
            projected = _apply_linear_float32_accum(self.out, attn)
        else:
            projected = _apply_linear_nd(self.out, attn)
        projected = projected.astype(x.dtype)
        return x + projected


@onnx_function
class MLPBlock(eqx.Module):
    """Mixture-of-experts feed-forward block mirroring GPT-OSS."""

    config: GPTOSSConfig = eqx.field(static=True)
    num_experts: int = eqx.field(static=True)
    experts_per_token: int = eqx.field(static=True)
    swiglu_limit: float = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    intermediate_size: int = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    norm: RMSNorm
    gate: eqx.nn.Linear
    mlp1_weight: jnp.ndarray
    mlp1_bias: jnp.ndarray
    mlp2_weight: jnp.ndarray
    mlp2_bias: jnp.ndarray

    def __init__(
        self,
        config: GPTOSSConfig,
        *,
        key: jax.Array,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.num_experts = int(config.num_experts)
        self.experts_per_token = int(config.experts_per_token)
        self.swiglu_limit = float(config.swiglu_limit)
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(config.intermediate_size)
        self.param_dtype = param_dtype

        keys = jax.random.split(key, 4)
        self.norm = RMSNorm(self.hidden_size)
        self.gate = eqx.nn.Linear(
            self.hidden_size,
            self.num_experts,
            use_bias=True,
            key=keys[0],
            dtype=param_dtype,
        )

        std1 = jnp.float32(1.0 / jnp.sqrt(float(self.hidden_size)))
        std2 = jnp.float32(1.0 / jnp.sqrt(float(self.intermediate_size)))

        self.mlp1_weight = (
            jax.random.normal(
                keys[1],
                (
                    self.num_experts,
                    self.intermediate_size * 2,
                    self.hidden_size,
                ),
                dtype=param_dtype,
            )
            * std1
        )
        self.mlp1_bias = jnp.zeros(
            (self.num_experts, self.intermediate_size * 2),
            dtype=param_dtype,
        )
        self.mlp2_weight = (
            jax.random.normal(
                keys[2],
                (
                    self.num_experts,
                    self.hidden_size,
                    self.intermediate_size,
                ),
                dtype=param_dtype,
            )
            * std2
        )
        self.mlp2_bias = jnp.zeros(
            (self.num_experts, self.hidden_size),
            dtype=param_dtype,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError("MLPBlock expects inputs shaped (batch, seq, hidden).")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Hidden size mismatch: expected {self.hidden_size}, got {x.shape[-1]}."
            )

        normed = _apply_pointwise(self.norm, x)

        if self.param_dtype == jnp.bfloat16:
            gate_logits = _apply_linear_float32_accum(self.gate, normed).astype(
                self.param_dtype
            )
            expert_scores, expert_indices = jax.lax.top_k(
                gate_logits.astype(jnp.float32), self.experts_per_token
            )
            expert_scores = expert_scores.astype(self.param_dtype)
            expert_weights = nn.softmax(
                expert_scores.astype(jnp.float32), axis=-1
            ).astype(self.param_dtype)

            mlp1_weight = jnp.take(self.mlp1_weight, expert_indices, axis=0).astype(
                jnp.float32
            )
            mlp1_bias = jnp.take(self.mlp1_bias, expert_indices, axis=0).astype(
                jnp.float32
            )
            proj1 = jnp.einsum(
                "bskoh,bsh->bsko",
                mlp1_weight,
                normed.astype(jnp.float32),
                optimize="optimal",
            )
            proj1 = (proj1 + mlp1_bias).astype(self.param_dtype)
            act = _swiglu(proj1.astype(self.param_dtype), limit=self.swiglu_limit)

            mlp2_weight = jnp.take(self.mlp2_weight, expert_indices, axis=0).astype(
                self.param_dtype
            )
            mlp2_bias = jnp.take(self.mlp2_bias, expert_indices, axis=0).astype(
                self.param_dtype
            )
            proj2 = jnp.einsum(
                "bskhi,bski->bskh",
                mlp2_weight,
                act.astype(self.param_dtype),
                optimize="optimal",
            ).astype(self.param_dtype)
            proj2 = proj2 + mlp2_bias
            combined = jnp.einsum(
                "bskh,bsk->bsh",
                proj2,
                expert_weights,
                optimize="optimal",
            ).astype(self.param_dtype)
            return x + combined.astype(x.dtype)

        gate_logits = _apply_linear_nd(self.gate, normed).astype(jnp.float32)
        expert_scores, expert_indices = jax.lax.top_k(
            gate_logits, self.experts_per_token
        )
        expert_weights = nn.softmax(expert_scores, axis=-1)

        mlp1_weight = jnp.take(self.mlp1_weight, expert_indices, axis=0).astype(
            jnp.float32
        )
        mlp1_bias = jnp.take(self.mlp1_bias, expert_indices, axis=0).astype(jnp.float32)
        mlp2_weight = jnp.take(self.mlp2_weight, expert_indices, axis=0).astype(
            jnp.float32
        )
        mlp2_bias = jnp.take(self.mlp2_bias, expert_indices, axis=0).astype(jnp.float32)

        proj1 = jnp.einsum(
            "bskoh,bsh->bsko", mlp1_weight, normed.astype(jnp.float32), optimize="optimal"
        )
        proj1 = proj1 + mlp1_bias
        act = _swiglu(proj1, limit=self.swiglu_limit)

        proj2 = jnp.einsum(
            "bskhi,bski->bskh", mlp2_weight, act, optimize="optimal"
        )
        proj2 = proj2 + mlp2_bias
        combined = jnp.einsum(
            "bskh,bsk->bsh", proj2, expert_weights, optimize="optimal"
        )
        return x + combined.astype(x.dtype)


@onnx_function
class TransformerBlock(eqx.Module):
    """Attention + MLP block for GPT-OSS."""

    attn: AttentionBlock
    mlp: MLPBlock

    def __init__(
        self,
        config: GPTOSSConfig,
        layer_idx: int,
        *,
        key: jax.Array,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        attn_key, mlp_key = jax.random.split(key, 2)
        self.attn = AttentionBlock(
            config,
            layer_idx,
            key=attn_key,
            param_dtype=param_dtype,
        )
        self.mlp = MLPBlock(
            config,
            key=mlp_key,
            param_dtype=param_dtype,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.attn(x)
        x = self.mlp(x)
        return x


@onnx_function
class Transformer(eqx.Module):
    """GPT-OSS Transformer using Equinox modules."""

    config: GPTOSSConfig = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)
    embedding: eqx.nn.Embedding
    blocks: Sequence[TransformerBlock]
    norm: RMSNorm
    unembedding: eqx.nn.Linear

    def __init__(
        self,
        config: GPTOSSConfig = GPTOSSConfig(),
        *,
        key: jax.Array,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.param_dtype = param_dtype

        num_blocks = config.num_hidden_layers
        keys = jax.random.split(key, num_blocks + 2)
        self.embedding = eqx.nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            key=keys[0],
            dtype=param_dtype,
        )
        block_keys = keys[1:-1]
        self.blocks = tuple(
            TransformerBlock(
                config,
                layer_idx=i,
                key=block_keys[i],
                param_dtype=param_dtype,
            )
            for i in range(num_blocks)
        )
        self.norm = RMSNorm(config.hidden_size)
        self.unembedding = eqx.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            key=keys[-1],
            dtype=param_dtype,
        )

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        squeeze_batch = False
        if tokens.ndim == 1:
            tokens = tokens[None, :]
            squeeze_batch = True
        elif tokens.ndim != 2:
            raise ValueError("Transformer expects token tensors shaped (batch, seq).")

        x = jnp.take(self.embedding.weight, tokens, axis=0)
        for block in self.blocks:
            x = block(x)
        x = _apply_pointwise(self.norm, x)
        logits = _apply_linear_nd(self.unembedding, x)
        if self.param_dtype == jnp.bfloat16:
            logits = logits.astype(self.param_dtype)
        logits = logits.astype(jnp.float32)
        if squeeze_batch:
            return logits[0]
        return logits


def _config_from_torch_transformer(torch_model: Any) -> GPTOSSConfig:
    """Derive a GPTOSSConfig from a torch-based Transformer instance."""

    if not hasattr(torch_model, "block") or len(torch_model.block) == 0:
        raise ValueError("Torch Transformer has no blocks to inspect.")
    first_block = torch_model.block[0]
    if not hasattr(first_block, "attn") or not hasattr(first_block, "mlp"):
        raise ValueError("Torch Transformer blocks must expose attn and mlp modules.")

    rope = getattr(first_block.attn, "rope", None)
    default_config = GPTOSSConfig()
    vocab_size, hidden_size = torch_model.embedding.weight.shape
    intermediate_size = first_block.mlp.mlp1_weight.shape[1] // 2

    return GPTOSSConfig(
        num_hidden_layers=len(torch_model.block),
        num_experts=int(first_block.mlp.num_experts),
        experts_per_token=int(first_block.mlp.experts_per_token),
        vocab_size=int(vocab_size),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        swiglu_limit=float(first_block.mlp.swiglu_limit),
        head_dim=int(first_block.attn.head_dim),
        num_attention_heads=int(first_block.attn.num_attention_heads),
        num_key_value_heads=int(first_block.attn.num_key_value_heads),
        sliding_window=int(first_block.attn.sliding_window),
        initial_context_length=(
            int(
                getattr(
                    rope,
                    "initial_context_length",
                    default_config.initial_context_length,
                )
            )
            if rope is not None
            else default_config.initial_context_length
        ),
        rope_theta=(
            float(getattr(rope, "base", default_config.rope_theta))
            if rope is not None
            else default_config.rope_theta
        ),
        rope_scaling_factor=(
            float(getattr(rope, "scaling_factor", default_config.rope_scaling_factor))
            if rope is not None
            else default_config.rope_scaling_factor
        ),
        rope_ntk_alpha=(
            float(getattr(rope, "ntk_alpha", default_config.rope_ntk_alpha))
            if rope is not None
            else default_config.rope_ntk_alpha
        ),
        rope_ntk_beta=(
            float(getattr(rope, "ntk_beta", default_config.rope_ntk_beta))
            if rope is not None
            else default_config.rope_ntk_beta
        ),
    )


def _torch_tensor_to_jax(tensor: Any, *, dtype: jnp.dtype | None) -> jnp.ndarray:
    """Convert a torch tensor into a JAX array without mutating the source."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - defensive, should not happen
        raise ImportError("torch must be installed to map GPT-OSS weights.") from exc

    array = tensor.detach()
    if hasattr(array, "dtype") and getattr(array.dtype, "is_floating_point", False):
        array = array.to(torch.float32)
    array = array.cpu().numpy()
    result = jnp.asarray(array)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def _tensor_from_checkpoint(
    checkpoint: Any,
    name: str,
    *,
    dtype: jnp.dtype | None,
) -> jnp.ndarray:
    """Helper to fetch a tensor from a GPT-OSS checkpoint and convert to JAX."""

    tensor = checkpoint.get(name)
    try:
        return _torch_tensor_to_jax(tensor, dtype=dtype)
    finally:
        del tensor
        gc.collect()


def _populate_eqx_from_checkpoint(
    checkpoint: Any,
    eqx_model: Transformer,
    *,
    param_dtype: jnp.dtype,
) -> Transformer:
    """Copy parameters from a GPT-OSS checkpoint into the Equinox model."""

    eqx_model = eqx.tree_at(
        lambda m: m.embedding.weight,
        eqx_model,
        _tensor_from_checkpoint(
            checkpoint,
            "embedding.weight",
            dtype=param_dtype,
        ),
    )

    for idx, _ in enumerate(eqx_model.blocks):
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].attn.sinks,
            eqx_model,
            _tensor_from_checkpoint(
                checkpoint,
                f"block.{idx}.attn.sinks",
                dtype=param_dtype,
            ),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].attn.norm.weight,
            eqx_model,
            _tensor_from_checkpoint(
                checkpoint,
                f"block.{idx}.attn.norm.scale",
                dtype=jnp.float32,
            ),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].attn.qkv.weight,
            eqx_model,
            _tensor_from_checkpoint(
                checkpoint,
                f"block.{idx}.attn.qkv.weight",
                dtype=param_dtype,
            ),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].attn.qkv.bias,
            eqx_model,
            _tensor_from_checkpoint(
                checkpoint,
                f"block.{idx}.attn.qkv.bias",
                dtype=param_dtype,
            ),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].attn.out.weight,
            eqx_model,
            _tensor_from_checkpoint(
                checkpoint,
                f"block.{idx}.attn.out.weight",
                dtype=param_dtype,
            ),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].attn.out.bias,
            eqx_model,
            _tensor_from_checkpoint(
                checkpoint,
                f"block.{idx}.attn.out.bias",
                dtype=param_dtype,
            ),
        )

        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].mlp.norm.weight,
            eqx_model,
            _tensor_from_checkpoint(
                checkpoint,
                f"block.{idx}.mlp.norm.scale",
                dtype=jnp.float32,
            ),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].mlp.gate.weight,
            eqx_model,
            _tensor_from_checkpoint(
                checkpoint,
                f"block.{idx}.mlp.gate.weight",
                dtype=param_dtype,
            ),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].mlp.gate.bias,
            eqx_model,
            _tensor_from_checkpoint(
                checkpoint,
                f"block.{idx}.mlp.gate.bias",
                dtype=param_dtype,
            ),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].mlp.mlp1_weight,
            eqx_model,
            _tensor_from_checkpoint(
                checkpoint,
                f"block.{idx}.mlp.mlp1_weight",
                dtype=param_dtype,
            ),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].mlp.mlp1_bias,
            eqx_model,
            _tensor_from_checkpoint(
                checkpoint,
                f"block.{idx}.mlp.mlp1_bias",
                dtype=param_dtype,
            ),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].mlp.mlp2_weight,
            eqx_model,
            _tensor_from_checkpoint(
                checkpoint,
                f"block.{idx}.mlp.mlp2_weight",
                dtype=param_dtype,
            ),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].mlp.mlp2_bias,
            eqx_model,
            _tensor_from_checkpoint(
                checkpoint,
                f"block.{idx}.mlp.mlp2_bias",
                dtype=param_dtype,
            ),
        )

    eqx_model = eqx.tree_at(
        lambda m: m.norm.weight,
        eqx_model,
        _tensor_from_checkpoint(
            checkpoint,
            "norm.scale",
            dtype=jnp.float32,
        ),
    )
    eqx_model = eqx.tree_at(
        lambda m: m.unembedding.weight,
        eqx_model,
        _tensor_from_checkpoint(
            checkpoint,
            "unembedding.weight",
            dtype=param_dtype,
        ),
    )
    return eqx_model


def _populate_eqx_from_torch(
    torch_model: Any,
    eqx_model: Transformer,
    *,
    param_dtype: jnp.dtype,
) -> Transformer:
    """Copy parameters from the torch Transformer into the Equinox example."""

    eqx_model = eqx.tree_at(
        lambda m: m.embedding.weight,
        eqx_model,
        _torch_tensor_to_jax(torch_model.embedding.weight, dtype=param_dtype),
    )

    for idx, torch_block in enumerate(torch_model.block):
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].attn.sinks,
            eqx_model,
            _torch_tensor_to_jax(torch_block.attn.sinks, dtype=param_dtype),
        )
        eqx_model = eqx.tree_at(
        lambda m, idx=idx: m.blocks[idx].attn.norm.weight,
        eqx_model,
        _torch_tensor_to_jax(torch_block.attn.norm.scale, dtype=jnp.float32),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].attn.qkv.weight,
            eqx_model,
            _torch_tensor_to_jax(torch_block.attn.qkv.weight, dtype=param_dtype),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].attn.qkv.bias,
            eqx_model,
            _torch_tensor_to_jax(torch_block.attn.qkv.bias, dtype=param_dtype),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].attn.out.weight,
            eqx_model,
            _torch_tensor_to_jax(torch_block.attn.out.weight, dtype=param_dtype),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].attn.out.bias,
            eqx_model,
            _torch_tensor_to_jax(torch_block.attn.out.bias, dtype=param_dtype),
        )

        eqx_model = eqx.tree_at(
        lambda m, idx=idx: m.blocks[idx].mlp.norm.weight,
        eqx_model,
        _torch_tensor_to_jax(torch_block.mlp.norm.scale, dtype=jnp.float32),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].mlp.gate.weight,
            eqx_model,
            _torch_tensor_to_jax(torch_block.mlp.gate.weight, dtype=param_dtype),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].mlp.gate.bias,
            eqx_model,
            _torch_tensor_to_jax(torch_block.mlp.gate.bias, dtype=param_dtype),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].mlp.mlp1_weight,
            eqx_model,
            _torch_tensor_to_jax(torch_block.mlp.mlp1_weight, dtype=param_dtype),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].mlp.mlp1_bias,
            eqx_model,
            _torch_tensor_to_jax(torch_block.mlp.mlp1_bias, dtype=param_dtype),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].mlp.mlp2_weight,
            eqx_model,
            _torch_tensor_to_jax(torch_block.mlp.mlp2_weight, dtype=param_dtype),
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=idx: m.blocks[idx].mlp.mlp2_bias,
            eqx_model,
            _torch_tensor_to_jax(torch_block.mlp.mlp2_bias, dtype=param_dtype),
        )

    eqx_model = eqx.tree_at(
        lambda m: m.norm.weight,
        eqx_model,
        _torch_tensor_to_jax(torch_model.norm.scale, dtype=jnp.float32),
    )
    eqx_model = eqx.tree_at(
        lambda m: m.unembedding.weight,
        eqx_model,
        _torch_tensor_to_jax(torch_model.unembedding.weight, dtype=param_dtype),
    )
    return eqx_model


def load_pretrained_gpt_oss(
    checkpoint: str | Path,
    *,
    device: Any = "cpu",
    param_dtype: jnp.dtype = jnp.bfloat16,
    seed: int = 0,
) -> Transformer:
    """Load a GPT-OSS checkpoint and mirror it into the Equinox example."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Loading GPT-OSS checkpoints requires `torch`. "
            "Install it via `pip install torch --index-url https://download.pytorch.org/whl/cpu`."
        ) from exc

    try:
        from gpt_oss.torch.weights import Checkpoint
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Loading GPT-OSS checkpoints requires the `gpt-oss` package. "
            "Install it with `pip install gpt-oss`."
        ) from exc

    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Expected a directory containing .safetensors files: {checkpoint_path}"
        )

    config_json = json.loads((checkpoint_path / "config.json").read_text())
    config_kwargs = {
        field.name: config_json[field.name]
        for field in dataclasses.fields(GPTOSSConfig)
        if field.name in config_json
    }
    config = GPTOSSConfig(**config_kwargs)
    key = jax.random.PRNGKey(int(seed))
    eqx_model = Transformer(config=config, key=key, param_dtype=param_dtype)
    checkpoint_reader = Checkpoint(str(checkpoint_path), torch.device(device))
    eqx_model = _populate_eqx_from_checkpoint(
        checkpoint_reader,
        eqx_model,
        param_dtype=param_dtype,
    )
    gc.collect()
    return eqx_model


_TEST_CONFIG: GPTOSSConfig = GPTOSSConfig(
    num_hidden_layers=1,
    num_experts=4,
    experts_per_token=2,
    vocab_size=32,
    hidden_size=64,
    intermediate_size=64,
    swiglu_limit=7.0,
    head_dim=16,
    num_attention_heads=4,
    num_key_value_heads=2,
    sliding_window=16,
    initial_context_length=256,
    rope_theta=10_000.0,
    rope_scaling_factor=8.0,
    rope_ntk_alpha=1.0,
    rope_ntk_beta=8.0,
)
_TEST_SEQ_LEN: int = 8


register_example(
    component="RMSNorm",
    description="Root mean square normalisation used by GPT-OSS.",
    source="https://github.com/openai/gpt-oss",
    since="v0.11.0",
    context="examples.eqx_gpt_oss",
    children=[],
    testcases=[
        {
            "testcase": "gpt_oss_rmsnorm",
            "callable": construct_and_call(
                lambda: eqx.filter_vmap(
                    RMSNorm(_TEST_CONFIG.hidden_size),
                    in_axes=0,
                    out_axes=0,
                )
            ),
            "input_shapes": [("B", _TEST_CONFIG.hidden_size)],
            "post_check_onnx_graph": EG(
                [
                    (
                        "Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Add",
                        {"counts": {"ReduceMean": 1}},
                    )
                ],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        }
    ],
)


register_example(
    component="AttentionBlock",
    description="Self-attention block with rotary embeddings and sinks.",
    source="https://github.com/openai/gpt-oss",
    since="v0.11.0",
    context="examples.eqx_gpt_oss",
    children=["RMSNorm"],
    testcases=[
        {
            "testcase": "gpt_oss_attention_block",
            "callable": construct_and_call(
                AttentionBlock,
                config=_TEST_CONFIG,
                layer_idx=0,
                key=with_prng_key(0),
                param_dtype=jnp.float32,
            ),
            "input_shapes": [
                (3, _TEST_SEQ_LEN, _TEST_CONFIG.hidden_size),
            ],
            "post_check_onnx_graph": EG(
                [f"AttentionBlock:Bx{_TEST_SEQ_LEN}x{_TEST_CONFIG.hidden_size}"],
                symbols={"B": None},
                search_functions=True,
            ),
            "run_only_f32_variant": True,
        }
    ],
)


register_example(
    component="MLPBlock",
    description="Mixture-of-experts SwiGLU feed-forward block.",
    source="https://github.com/openai/gpt-oss",
    since="v0.11.0",
    context="examples.eqx_gpt_oss",
    children=["RMSNorm"],
    testcases=[
        {
            "testcase": "gpt_oss_mlp_block",
            "callable": construct_and_call(
                MLPBlock,
                config=_TEST_CONFIG,
                key=with_prng_key(1),
                param_dtype=jnp.float32,
            ),
            "input_shapes": [
                (3, _TEST_SEQ_LEN, _TEST_CONFIG.hidden_size),
            ],
            "post_check_onnx_graph": EG(
                [f"MLPBlock:Bx{_TEST_SEQ_LEN}x{_TEST_CONFIG.hidden_size}"],
                symbols={"B": None},
                search_functions=True,
            ),
            "run_only_f32_variant": True,
        }
    ],
)


register_example(
    component="TransformerBlock",
    description="GPT-OSS Transformer layer (attention + MoE).",
    source="https://github.com/openai/gpt-oss",
    since="v0.11.0",
    context="examples.eqx_gpt_oss",
    children=["AttentionBlock", "MLPBlock"],
    testcases=[
        {
            "testcase": "gpt_oss_transformer_block",
            "callable": construct_and_call(
                TransformerBlock,
                config=_TEST_CONFIG,
                layer_idx=0,
                key=with_prng_key(2),
                param_dtype=jnp.float32,
            ),
            "input_shapes": [
                ("B", _TEST_SEQ_LEN, _TEST_CONFIG.hidden_size),
            ],
            "post_check_onnx_graph": EG(
                [f"TransformerBlock:Bx{_TEST_SEQ_LEN}x{_TEST_CONFIG.hidden_size}"],
                symbols={"B": None},
                search_functions=True,
            ),
            "run_only_f32_variant": True,
        }
    ],
)


register_example(
    component="Transformer",
    description="Full GPT-OSS Transformer stack.",
    source="https://github.com/openai/gpt-oss",
    since="v0.11.0",
    context="examples.eqx_gpt_oss",
    children=["TransformerBlock"],
    testcases=[
        {
            "testcase": "gpt_oss_transformer",
            "callable": construct_and_call(
                Transformer,
                config=_TEST_CONFIG,
                key=with_prng_key(3),
                param_dtype=jnp.float32,
            ),
            "input_shapes": [(3, _TEST_SEQ_LEN)],
            "post_check_onnx_graph": EG(
                [f"Transformer:Bx{_TEST_SEQ_LEN}x{_TEST_CONFIG.vocab_size}"],
                symbols={"B": None},
                search_functions=True,
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        }
    ],
)
