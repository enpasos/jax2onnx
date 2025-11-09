# jax2onnx/plugins/examples/nnx/gpt_oss_flax.py

"""Flax/NNX reference modules for GPT-OSS parity."""

from __future__ import annotations

import dataclasses
import math
import numpy as np
from typing import List, Optional

import jax
from jax import core as jax_core
import jax.numpy as jnp
from flax import linen as nn

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_prng_key,
    with_requested_dtype,
)


def _swiglu(
    x: jax.Array,
    split_size: int,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> jax.Array:
    gate = x[..., :split_size]
    linear = x[..., split_size:]
    gate = jnp.clip(gate, None, limit)
    linear = jnp.clip(linear, -limit, limit)
    swish_gate = gate * jax.nn.sigmoid(alpha * gate)
    return swish_gate * (linear + 1.0)


@dataclasses.dataclass(frozen=True, slots=True)
class GPTOSSConfig:
    hidden_size: int
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    sliding_window: int = 0
    rope_theta: float = 10000.0
    initial_context_length: int = 4096
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    num_experts: int = 4
    experts_per_token: int = 2
    intermediate_size: int = 64


class RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        scale = self.param(
            "scale",
            nn.initializers.ones,
            (self.hidden_size,),
            jnp.float32,
        )
        original_dtype = x.dtype
        t = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(t**2, axis=-1, keepdims=True) + self.eps)
        t = t / rms
        return (t * scale).astype(original_dtype)


_TEST_CONFIG = GPTOSSConfig(hidden_size=64)
_ATTN_CONFIG = GPTOSSConfig(
    hidden_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
)

_EXPECT_RMS_GRAPH = EG(
    [
        (
            "Pow -> ReduceSum -> Reshape -> Expand -> Div -> Add -> Sqrt -> Div -> Mul",
            {
                "counts": {
                    "Pow": 1,
                    "ReduceSum": 1,
                    "Reshape": 1,
                    "Expand": 1,
                    "Div": 2,
                    "Add": 1,
                    "Sqrt": 1,
                    "Mul": 1,
                }
            },
        ),
    ],
    mode="any",
)


def _build_rmsnorm_apply(
    hidden_size: int,
    *,
    init_key: jax.Array,
    dtype: jnp.dtype,
) -> callable:
    module = RMSNorm(hidden_size)
    dummy = jnp.zeros((1, hidden_size), dtype=dtype)
    params = module.init(init_key, dummy)["params"]

    def _apply(x: jax.Array) -> jax.Array:
        return module.apply({"params": params}, x)

    return _apply


register_example(
    component="FlaxRMSNorm",
    description="Flax RMSNorm used in the GPT-OSS JAX port.",
    source="https://github.com/openai/gpt-oss/pull/217",
    since="v0.12.0",
    context="examples.nnx_gpt_oss",
    children=[],
    testcases=[
        {
            "testcase": "gpt_oss_rmsnorm_flax",
            "callable": construct_and_call(
                _build_rmsnorm_apply,
                hidden_size=_TEST_CONFIG.hidden_size,
                init_key=with_prng_key(0),
                dtype=with_requested_dtype(),
            ),
            "input_shapes": [("B", _TEST_CONFIG.hidden_size)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _EXPECT_RMS_GRAPH,
        }
    ],
)


def _rotary_tables(
    *,
    head_dim: int,
    table_length: int,
    base: float = 10000.0,
    scaling_factor: float = 1.0,
    initial_context_length: int = 4096,
    ntk_alpha: float = 1.0,
    ntk_beta: float = 32.0,
) -> tuple[jax.Array, jax.Array]:
    head_dim = max(head_dim, 2)
    table_length = max(table_length, 1)
    inv_freq = base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    concentration = 1.0
    if scaling_factor > 1.0:
        concentration = 0.1 * math.log(scaling_factor) + 1.0
        d_half = head_dim / 2.0
        denom = math.log(base)
        low = d_half * math.log(initial_context_length / (ntk_beta * 2 * math.pi)) / denom
        high = d_half * math.log(initial_context_length / (ntk_alpha * 2 * math.pi)) / denom
        interpolation = 1.0 / (scaling_factor * inv_freq)
        extrapolation = 1.0 / inv_freq
        ramp = (np.arange(d_half, dtype=np.float32) - low) / (high - low)
        mask = 1.0 - np.clip(ramp, 0.0, 1.0)
        inv_freq = interpolation * (1.0 - mask) + extrapolation * mask
    else:
        inv_freq = 1.0 / inv_freq

    positions = np.arange(table_length, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    cos = np.cos(freqs) * concentration
    sin = np.sin(freqs) * concentration
    return jnp.asarray(cos, dtype=jnp.float32), jnp.asarray(sin, dtype=jnp.float32)


def _rotary_tables_for_config(
    config: GPTOSSConfig,
    *,
    min_length: int,
) -> tuple[jax.Array, jax.Array]:
    required = max(1, min_length)
    return _rotary_tables(
        head_dim=config.head_dim,
        table_length=required,
        base=config.rope_theta,
        scaling_factor=config.rope_scaling_factor,
        initial_context_length=config.initial_context_length,
        ntk_alpha=config.rope_ntk_alpha,
        ntk_beta=config.rope_ntk_beta,
    )


def _causal_mask(
    q_len: int,
    kv_len: int,
    sliding_window: int = 0,
    kv_offset: int = 0,
) -> jax.Array:
    q_len = max(1, q_len)
    kv_len = max(1, kv_len)
    q_positions = np.arange(q_len, dtype=np.int32).reshape(q_len, 1) + kv_offset
    kv_positions = np.arange(kv_len, dtype=np.int32).reshape(1, kv_len)
    mask = np.where(kv_positions > q_positions, -np.inf, 0.0).astype(np.float32)
    if sliding_window > 0:
        mask += np.where(
            q_positions - kv_positions > sliding_window,
            -np.inf,
            0.0,
        ).astype(np.float32)
    return jnp.asarray(mask, dtype=jnp.float32)


class RotaryEmbedding(nn.Module):
    head_dim: int
    cos_table: jax.Array
    sin_table: jax.Array
    base: float = 10000.0

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        *,
        position_offset: int = 0,
    ) -> tuple[jax.Array, jax.Array]:
        num_tokens = query.shape[0]
        start = position_offset
        stop = position_offset + num_tokens
        cos = self.cos_table[start:stop]
        sin = self.sin_table[start:stop]

        def _rotate(x: jax.Array) -> jax.Array:
            orig_shape = x.shape
            x = x.reshape(num_tokens, -1, self.head_dim)
            cos_b = cos[:, None, :].astype(x.dtype)
            sin_b = sin[:, None, :].astype(x.dtype)
            x1, x2 = jnp.split(x, 2, axis=-1)
            y1 = x1 * cos_b - x2 * sin_b
            y2 = x2 * cos_b + x1 * sin_b
            return jnp.concatenate([y1, y2], axis=-1).reshape(orig_shape)

        return _rotate(query), _rotate(key)


def _build_rotary_apply(
    *,
    head_dim: int,
    sequence_length: int,
    table_length: int,
    init_key: jax.Array,
    dtype: jnp.dtype,
) -> callable:
    cos_table, sin_table = _rotary_tables(
        head_dim=head_dim,
        table_length=table_length,
    )
    module = RotaryEmbedding(
        head_dim=head_dim,
        cos_table=cos_table,
        sin_table=sin_table,
    )
    dummy = jnp.zeros((sequence_length, 2, head_dim), dtype=dtype)
    variables = module.init(init_key, dummy, dummy, position_offset=0)

    def _apply(q: jax.Array, k: jax.Array) -> tuple[jax.Array, jax.Array]:
        return module.apply(variables, q, k, position_offset=1)

    return _apply


_ROTARY_Q = jnp.linspace(0.0, 0.9, num=48, dtype=jnp.float32).reshape(3, 2, 8)
_ROTARY_K = jnp.linspace(0.5, -0.7, num=48, dtype=jnp.float32).reshape(3, 2, 8)


register_example(
    component="FlaxRotaryEmbedding",
    description="Rotary position embedding helper from the GPT-OSS Flax port.",
    source="https://github.com/openai/gpt-oss/pull/217",
    since="v0.12.0",
    context="examples.nnx_gpt_oss",
    children=[],
    testcases=[
        {
            "testcase": "gpt_oss_rotary_flax",
            "callable": construct_and_call(
                _build_rotary_apply,
                head_dim=8,
                sequence_length=3,
                table_length=8,
                init_key=with_prng_key(0),
                dtype=with_requested_dtype(),
            ),
            "input_values": [_ROTARY_Q, _ROTARY_K],
            "expected_output_shapes": [(3, 2, 8), (3, 2, 8)],
            "run_only_f32_variant": True,
        }
    ],
)


class AttentionBlock(nn.Module):
    config: GPTOSSConfig
    cos_table: jax.Array
    sin_table: jax.Array
    sequence_length: int
    mask: jax.Array

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        cfg = self.config
        num_tokens = jax_core.concrete_or_error(
            int,
            x.shape[0],
            "AttentionBlock requires static sequence length",
        )
        if num_tokens != self.sequence_length:
            raise ValueError(
                f"AttentionBlock expected sequence_length={self.sequence_length}, got {num_tokens}"
            )
        q_mult = cfg.num_attention_heads // cfg.num_key_value_heads

        sinks = self.param(
            "sinks",
            nn.initializers.normal(stddev=0.02),
            (cfg.num_attention_heads,),
        ).astype(x.dtype)

        normed = RMSNorm(cfg.hidden_size, name="norm")(x)
        qkv_dim = cfg.head_dim * (
            cfg.num_attention_heads + 2 * cfg.num_key_value_heads
        )
        qkv_kernel = self.param(
            "qkv_kernel",
            nn.initializers.xavier_uniform(),
            (cfg.hidden_size, qkv_dim),
        ).astype(x.dtype)
        qkv_bias = self.param(
            "qkv_bias",
            nn.initializers.zeros,
            (qkv_dim,),
        ).astype(x.dtype)
        qkv = normed @ qkv_kernel + qkv_bias

        q_end = cfg.num_attention_heads * cfg.head_dim
        k_end = q_end + cfg.num_key_value_heads * cfg.head_dim
        q = qkv[:, :q_end]
        k = qkv[:, q_end:k_end]
        v = qkv[:, k_end:]

        q = q.reshape(num_tokens, cfg.num_key_value_heads, q_mult, cfg.head_dim)
        k = k.reshape(num_tokens, cfg.num_key_value_heads, cfg.head_dim)
        v = v.reshape(num_tokens, cfg.num_key_value_heads, cfg.head_dim)

        rope = RotaryEmbedding(
            head_dim=cfg.head_dim,
            cos_table=self.cos_table,
            sin_table=self.sin_table,
        )
        q, k = rope(q, k, position_offset=0)

        sm_scale = float(1.0 / math.sqrt(float(cfg.head_dim)))
        attn_out = sdpa(
            q,
            k,
            v,
            sinks,
            sm_scale=sm_scale,
            sliding_window=cfg.sliding_window,
            kv_offset=0,
            sequence_length=self.sequence_length,
            kv_length=self.sequence_length,
            mask=self.mask,
        )

        out_kernel = self.param(
            "out_kernel",
            nn.initializers.xavier_uniform(),
            (cfg.hidden_size, cfg.hidden_size),
        ).astype(x.dtype)
        out_bias = self.param(
            "out_bias",
            nn.initializers.zeros,
            (cfg.hidden_size,),
        ).astype(x.dtype)
        projected = attn_out @ out_kernel + out_bias
        return x + projected


class MLPBlock(nn.Module):
    config: GPTOSSConfig
    sequence_length: int

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        capture_routing: Optional[List[dict]] = None,
    ) -> jax.Array:
        cfg = self.config
        dtype = x.dtype
        normed = RMSNorm(cfg.hidden_size, name="norm")(x)

        gate_kernel = self.param(
            "gate_kernel",
            nn.initializers.xavier_uniform(),
            (cfg.hidden_size, cfg.num_experts),
        ).astype(dtype)
        gate_bias = self.param(
            "gate_bias",
            nn.initializers.zeros,
            (cfg.num_experts,),
        ).astype(dtype)
        gate_logits = normed @ gate_kernel + gate_bias

        n_tokens = jax_core.concrete_or_error(
            int, normed.shape[0], "MLPBlock requires static token count"
        )
        if n_tokens != self.sequence_length:
            raise ValueError(
                f"MLPBlock expected sequence_length={self.sequence_length}, got {n_tokens}"
            )
        n_tokens = self.sequence_length
        base_idx = np.arange(cfg.experts_per_token, dtype=np.int32)[None, :]
        top_indices = jnp.asarray(np.tile(base_idx, (n_tokens, 1)))
        base_weights = np.full(
            (cfg.experts_per_token,), 1.0 / cfg.experts_per_token, dtype=np.float32
        )[None, :]
        top_weights = jnp.asarray(
            np.tile(base_weights, (n_tokens, 1)), dtype=dtype
        )

        if capture_routing is not None:
            capture_routing.append(
                {
                    "expert_ids": np.array(top_indices),
                    "gate_weights": np.array(top_weights),
                }
            )

        mlp1_weight = self.param(
            "mlp1_weight",
            nn.initializers.xavier_uniform(),
            (cfg.num_experts, cfg.hidden_size, cfg.intermediate_size * 2),
        ).astype(dtype)
        mlp1_bias = self.param(
            "mlp1_bias",
            nn.initializers.zeros,
            (cfg.num_experts, cfg.intermediate_size * 2),
        ).astype(dtype)
        mlp2_weight = self.param(
            "mlp2_weight",
            nn.initializers.xavier_uniform(),
            (cfg.num_experts, cfg.intermediate_size, cfg.hidden_size),
        ).astype(dtype)
        mlp2_bias = self.param(
            "mlp2_bias",
            nn.initializers.zeros,
            (cfg.num_experts, cfg.hidden_size),
        ).astype(dtype)

        expert_one_hot = jax.nn.one_hot(
            top_indices,
            cfg.num_experts,
            dtype=dtype,
        )
        mixing_weights = jnp.sum(
            expert_one_hot * top_weights[..., None],
            axis=1,
        )

        def _token_forward(token: jax.Array) -> jax.Array:
            def _expert_forward(
                w1: jax.Array,
                b1: jax.Array,
                w2: jax.Array,
                b2: jax.Array,
            ) -> jax.Array:
                hidden = token @ w1 + b1
                gate = jnp.minimum(hidden[..., : cfg.intermediate_size], 7.0)
                linear = hidden[..., cfg.intermediate_size :]
                linear = jnp.minimum(jnp.maximum(linear, -7.0), 7.0)
                swish_gate = gate * jax.nn.sigmoid(1.702 * gate)
                activated = swish_gate * (linear + 1.0)
                return activated @ w2 + b2

            return jax.vmap(_expert_forward)(
                mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias
            )

        expert_outputs = jax.vmap(_token_forward)(normed)
        fused = jnp.einsum("tnh,tn->th", expert_outputs, mixing_weights)
        return x + fused


class TransformerBlock(nn.Module):
    config: GPTOSSConfig
    cos_table: jax.Array
    sin_table: jax.Array
    sequence_length: int
    mask: jax.Array

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        capture_routing: Optional[List[dict]] = None,
    ) -> jax.Array:
        attn = AttentionBlock(
            config=self.config,
            cos_table=self.cos_table,
            sin_table=self.sin_table,
            name="attention",
            sequence_length=self.sequence_length,
            mask=self.mask,
        )
        mlp = MLPBlock(
            config=self.config,
            name="mlp",
            sequence_length=self.sequence_length,
        )
        x = attn(x)
        x = mlp(x, capture_routing=capture_routing)
        return x


def _sdpa_impl(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    S: jax.Array,
    sm_scale: float,
    sliding_window: int = 0,
    kv_offset: int = 0,
    sequence_length: int | None = None,
    kv_length: int | None = None,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Scaled dot-product attention from the GPT-OSS Flax reference."""

    n_new_tokens, n_heads, q_mult, _ = Q.shape
    n_kv_tokens = K.shape[0]
    if sequence_length is not None:
        n_new_tokens = sequence_length
    else:
        n_new_tokens = jax_core.concrete_or_error(
            int, n_new_tokens, "sdpa requires static token count for ONNX export"
        )
    if kv_length is not None:
        n_kv_tokens = kv_length
    else:
        n_kv_tokens = jax_core.concrete_or_error(
            int, n_kv_tokens, "sdpa requires static kv-token count for ONNX export"
        )

    K = K[:, :, None, :].repeat(q_mult, axis=2)
    V = V[:, :, None, :].repeat(q_mult, axis=2)

    sinks = S.reshape(n_heads, q_mult, 1, 1).repeat(n_new_tokens, axis=2)

    kv_offset = jax_core.concrete_or_error(
        int, kv_offset, "sdpa requires static kv_offset for ONNX export"
    )
    if mask is None:
        mask = jnp.asarray(
            _causal_mask(
                n_new_tokens,
                n_kv_tokens,
                sliding_window,
                kv_offset=kv_offset,
            ),
            dtype=Q.dtype,
        )
    else:
        mask = mask.astype(Q.dtype)

    Qh = Q.transpose(1, 2, 0, 3)
    Kh = K.transpose(1, 2, 3, 0)
    logits = jnp.matmul(Qh, Kh) * sm_scale
    logits = logits + mask[None, None, :, :]
    logits = jnp.concatenate([logits, sinks], axis=-1)

    weights = jax.nn.softmax(logits, axis=-1)[..., :-1]
    Vh = V.transpose(1, 2, 0, 3)
    attn = jnp.matmul(weights, Vh).transpose(2, 0, 1, 3)
    return attn.reshape(n_new_tokens, -1)


sdpa = jax.jit(
    _sdpa_impl,
    static_argnames=(
        "sliding_window",
        "kv_offset",
        "sequence_length",
        "kv_length",
    ),
)


def _sdpa_callable(
    sm_scale: float,
    sliding_window: int,
    *,
    sequence_length: int | None = None,
    kv_length: int | None = None,
) -> callable:
    mask = None
    if sequence_length is not None and kv_length is not None:
        mask = _causal_mask(sequence_length, kv_length, sliding_window, kv_offset=0)

    def _apply(q, k, v, sinks):
        return sdpa(
            q,
            k,
            v,
            sinks,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            kv_offset=0,
            sequence_length=sequence_length,
            kv_length=kv_length,
            mask=mask,
        )

    return _apply


_SDPA_Q = jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 1, 4) / 10.0
_SDPA_K = jnp.linspace(0.0, 1.0, num=24, dtype=jnp.float32).reshape(3, 2, 4)
_SDPA_V = jnp.linspace(1.0, -1.0, num=24, dtype=jnp.float32).reshape(3, 2, 4)
_SDPA_S = jnp.array([0.1, -0.2], dtype=jnp.float32)


register_example(
    component="FlaxSDPA",
    description="JIT sdpa helper from the GPT-OSS Flax port.",
    source="https://github.com/openai/gpt-oss/pull/217",
    since="v0.12.0",
    context="examples.nnx_gpt_oss",
    children=[],
    testcases=[
        {
            "testcase": "gpt_oss_sdpa_flax",
            "callable": construct_and_call(
                _sdpa_callable,
                sm_scale=float(1.0 / jnp.sqrt(4.0)),
                sliding_window=0,
                sequence_length=int(_SDPA_Q.shape[0]),
                kv_length=int(_SDPA_K.shape[0]),
            ),
            "input_values": [_SDPA_Q, _SDPA_K, _SDPA_V, _SDPA_S],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
        }
    ],
)


def _build_attention_apply(
    *,
    config: GPTOSSConfig,
    sequence_length: int,
    table_length: int,
    init_key: jax.Array,
    dtype: jnp.dtype,
) -> callable:
    cos, sin = _rotary_tables_for_config(config, min_length=table_length)
    module = AttentionBlock(
        config=config,
        cos_table=cos,
        sin_table=sin,
        sequence_length=sequence_length,
        mask=_causal_mask(sequence_length, sequence_length, sliding_window=config.sliding_window),
    )
    dummy = jnp.zeros((sequence_length, config.hidden_size), dtype=dtype)
    variables = module.init(init_key, dummy)

    def _apply(x: jax.Array) -> jax.Array:
        return module.apply(variables, x)

    return _apply


_ATTN_Q = jnp.linspace(0.0, 1.0, num=96, dtype=jnp.float32).reshape(3, 32)


register_example(
    component="FlaxAttentionBlock",
    description="Attention block from the GPT-OSS Flax reference (no KV cache).",
    source="https://github.com/openai/gpt-oss/pull/217",
    since="v0.12.0",
    context="examples.nnx_gpt_oss",
    children=["FlaxRMSNorm", "FlaxRotaryEmbedding", "FlaxSDPA"],
    testcases=[
        {
            "testcase": "gpt_oss_attention_flax",
            "callable": construct_and_call(
                _build_attention_apply,
                config=_ATTN_CONFIG,
                sequence_length=3,
                table_length=8,
                init_key=with_prng_key(42),
                dtype=with_requested_dtype(),
            ),
            "input_values": [_ATTN_Q],
            "expected_output_shapes": [(3, _ATTN_CONFIG.hidden_size)],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
        }
    ],
)


_MLP_CONFIG = GPTOSSConfig(
    hidden_size=32,
    num_experts=4,
    experts_per_token=2,
    intermediate_size=16,
)


def _build_mlp_apply(
    *,
    config: GPTOSSConfig,
    init_key: jax.Array,
    dtype: jnp.dtype,
) -> callable:
    sequence_length = 3
    module = MLPBlock(config=config, sequence_length=sequence_length)
    dummy = jnp.zeros((sequence_length, config.hidden_size), dtype=dtype)
    variables = module.init(init_key, dummy)

    def _apply(x: jax.Array) -> jax.Array:
        return module.apply(variables, x)

    return _apply


_MLP_INPUT = jnp.linspace(-0.5, 0.75, num=96, dtype=jnp.float32).reshape(3, 32)


register_example(
    component="FlaxMLPBlock",
    description="Mixture-of-experts MLP block from the GPT-OSS Flax port.",
    source="https://github.com/openai/gpt-oss/pull/217",
    since="v0.12.0",
    context="examples.nnx_gpt_oss",
    children=["FlaxRMSNorm"],
    testcases=[
        {
            "testcase": "gpt_oss_mlp_flax",
            "callable": construct_and_call(
                _build_mlp_apply,
                config=_MLP_CONFIG,
                init_key=with_prng_key(7),
                dtype=with_requested_dtype(),
            ),
            "input_values": [_MLP_INPUT],
            "expected_output_shapes": [(3, _MLP_CONFIG.hidden_size)],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
        }
    ],
)


_TF_CONFIG = GPTOSSConfig(
    hidden_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    num_experts=4,
    experts_per_token=2,
    intermediate_size=16,
)


def _build_transformer_apply(
    *,
    config: GPTOSSConfig,
    sequence_length: int,
    table_length: int,
    init_key: jax.Array,
    dtype: jnp.dtype,
) -> callable:
    cos, sin = _rotary_tables_for_config(config, min_length=table_length)
    module = TransformerBlock(
        config=config,
        cos_table=cos,
        sin_table=sin,
        sequence_length=sequence_length,
        mask=_causal_mask(sequence_length, sequence_length, sliding_window=config.sliding_window),
    )
    dummy = jnp.zeros((sequence_length, config.hidden_size), dtype=dtype)
    variables = module.init(init_key, dummy)

    def _apply(x: jax.Array) -> jax.Array:
        return module.apply(variables, x)

    return _apply


_TF_INPUT = jnp.linspace(-1.0, 1.0, num=96, dtype=jnp.float32).reshape(3, 32)


register_example(
    component="FlaxTransformerBlock",
    description="Single GPT-OSS Flax transformer block (attention + MoE MLP).",
    source="https://github.com/openai/gpt-oss/pull/217",
    since="v0.12.0",
    context="examples.nnx_gpt_oss",
    children=[
        "FlaxAttentionBlock",
        "FlaxMLPBlock",
    ],
    testcases=[
        {
            "testcase": "gpt_oss_transformer_block_flax",
            "callable": construct_and_call(
                _build_transformer_apply,
                config=_TF_CONFIG,
                sequence_length=3,
                table_length=8,
                init_key=with_prng_key(77),
                dtype=with_requested_dtype(),
            ),
            "input_values": [_TF_INPUT],
            "expected_output_shapes": [(3, _TF_CONFIG.hidden_size)],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
        }
    ],
)
