# jax2onnx/plugins/examples/nnx/gpt_oss_flax.py

"""Flax/NNX reference modules for GPT-OSS parity."""

from __future__ import annotations

import dataclasses
import math
import numpy as np
from typing import Final, List, Optional

import jax
from jax import core as jax_core
from jax import lax
import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_prng_key,
    with_requested_dtype,
)


class _KeySeq:
    """Lightweight PRNG splitter for deterministic nnx Module init."""

    def __init__(self, key: jax.Array | int | None):
        if key is None:
            key = 0
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)
        self._key = key

    def next(self) -> jax.Array:
        self._key, sub = jax.random.split(self._key)
        return sub


_XAVIER_UNIFORM = jax.nn.initializers.variance_scaling(
    1.0, "fan_avg", "uniform", dtype=jnp.float32
)
_NORMAL_002 = jax.nn.initializers.normal(stddev=0.02)


def _init_param(
    key: jax.Array,
    initializer,
    shape: tuple[int, ...],
    dtype: jnp.dtype = jnp.float32,
) -> nnx.Param:
    return nnx.Param(initializer(key, shape, dtype))


def _matmul_lastdim(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    return lax.dot_general(
        lhs,
        rhs,
        dimension_numbers=(((lhs.ndim - 1,), (0,)), ((), ())),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def _linear(lhs: jax.Array, kernel: jax.Array, bias: jax.Array | None = None) -> jax.Array:
    y = _matmul_lastdim(lhs, kernel)
    if bias is not None:
        y = y + bias
    return y.astype(lhs.dtype)


def _concat(values: list[jax.Array] | tuple[jax.Array, ...], axis: int) -> jax.Array:
    seq = list(values)
    if not seq:
        raise ValueError("concat requires at least one value")
    rank = seq[0].ndim
    if axis < 0:
        axis = axis + rank
    return lax.concatenate(seq, dimension=axis)


def _softmax(logits: jax.Array, axis: int = -1) -> jax.Array:
    max_logits = jnp.max(logits, axis=axis, keepdims=True)
    stabilized = logits - lax.stop_gradient(max_logits)
    exp = jnp.exp(stabilized)
    denom = jnp.sum(exp, axis=axis, keepdims=True)
    return exp / denom


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
    num_hidden_layers: int = 1
    vocab_size: int = 32
    sliding_window: int = 0
    rope_theta: float = 10000.0
    initial_context_length: int = 4096
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    num_experts: int = 4
    experts_per_token: int = 2
    intermediate_size: int = 64


class RMSNorm(nnx.Module):
    def __init__(self, hidden_size: int, *, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps
        self.scale = nnx.Param(jnp.ones((hidden_size,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        original_dtype = x.dtype
        t = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(t**2, axis=-1, keepdims=True) + self.eps)
        t = t / rms
        return (t * self.scale.value).astype(original_dtype)


_TEST_CONFIG: Final[GPTOSSConfig] = GPTOSSConfig(hidden_size=64)
_ATTN_CONFIG: Final[GPTOSSConfig] = GPTOSSConfig(
    hidden_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
)

_EXPECT_RMS_GRAPH: Final = EG(
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

    def _apply(x: jax.Array) -> jax.Array:
        return module(x)

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
        low = (
            d_half * math.log(initial_context_length / (ntk_beta * 2 * math.pi)) / denom
        )
        high = (
            d_half
            * math.log(initial_context_length / (ntk_alpha * 2 * math.pi))
            / denom
        )
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


@dataclasses.dataclass
class RotaryEmbedding:
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
            return _concat([y1, y2], axis=-1).reshape(orig_shape)

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

    def _apply(q: jax.Array, k: jax.Array) -> tuple[jax.Array, jax.Array]:
        return module(q, k, position_offset=1)

    return _apply


_ROTARY_Q: Final[jax.Array] = jnp.linspace(0.0, 0.9, num=48, dtype=jnp.float32).reshape(
    3, 2, 8
)
_ROTARY_K: Final[jax.Array] = jnp.linspace(
    0.5, -0.7, num=48, dtype=jnp.float32
).reshape(3, 2, 8)


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


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        *,
        config: GPTOSSConfig,
        cos_table: jax.Array,
        sin_table: jax.Array,
        sequence_length: int,
        mask: jax.Array,
        dtype: jnp.dtype = jnp.float32,
        rng: jax.Array | int | None = None,
    ):
        self.config = config
        self.cos_table = nnx.data(cos_table)
        self.sin_table = nnx.data(sin_table)
        self.sequence_length = sequence_length
        self.mask = nnx.data(mask)
        self.dtype = dtype

        self.norm = RMSNorm(config.hidden_size)
        rng_seq = _KeySeq(rng)

        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv_kernel = _init_param(
            rng_seq.next(), _XAVIER_UNIFORM, (config.hidden_size, qkv_dim), dtype
        )
        self.qkv_bias = nnx.Param(jnp.zeros((qkv_dim,), dtype=dtype))
        self.out_kernel = _init_param(
            rng_seq.next(),
            _XAVIER_UNIFORM,
            (config.hidden_size, config.hidden_size),
            dtype,
        )
        self.out_bias = nnx.Param(jnp.zeros((config.hidden_size,), dtype=dtype))
        self.sinks = _init_param(
            rng_seq.next(),
            _NORMAL_002,
            (config.num_attention_heads,),
            dtype,
        )

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

        normed = self.norm(x)
        qkv = _linear(normed, self.qkv_kernel.value, self.qkv_bias.value)

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
            self.sinks.value.astype(x.dtype),
            sm_scale=sm_scale,
            sliding_window=cfg.sliding_window,
            kv_offset=0,
            sequence_length=self.sequence_length,
            kv_length=self.sequence_length,
            mask=self.mask,
        )

        projected = _linear(attn_out, self.out_kernel.value, self.out_bias.value)
        return x + projected


class MLPBlock(nnx.Module):
    def __init__(
        self,
        *,
        config: GPTOSSConfig,
        sequence_length: int,
        dtype: jnp.dtype = jnp.float32,
        rng: jax.Array | int | None = None,
    ):
        self.config = config
        self.sequence_length = sequence_length
        self.norm = RMSNorm(config.hidden_size)
        rng_seq = _KeySeq(rng)

        self.gate_kernel = _init_param(
            rng_seq.next(),
            _XAVIER_UNIFORM,
            (config.hidden_size, config.num_experts),
            dtype,
        )
        self.gate_bias = nnx.Param(jnp.zeros((config.num_experts,), dtype=dtype))

        self.mlp1_weight = _init_param(
            rng_seq.next(),
            _XAVIER_UNIFORM,
            (config.num_experts, config.intermediate_size * 2, config.hidden_size),
            dtype,
        )
        self.mlp1_bias = nnx.Param(
            jnp.zeros((config.num_experts, config.intermediate_size * 2), dtype=dtype)
        )
        self.mlp2_weight = _init_param(
            rng_seq.next(),
            _XAVIER_UNIFORM,
            (config.num_experts, config.hidden_size, config.intermediate_size),
            dtype,
        )
        self.mlp2_bias = nnx.Param(
            jnp.zeros((config.num_experts, config.hidden_size), dtype=dtype)
        )

    def __call__(
        self,
        x: jax.Array,
        capture_routing: Optional[List[dict]] = None,
    ) -> jax.Array:
        cfg = self.config
        normed = self.norm(x)

        n_tokens = jax_core.concrete_or_error(
            int, normed.shape[0], "MLPBlock requires static token count"
        )
        if n_tokens != self.sequence_length:
            raise ValueError(
                f"MLPBlock expected sequence_length={self.sequence_length}, got {n_tokens}"
            )

        gate_logits = _linear(
            normed, self.gate_kernel.value, self.gate_bias.value
        )
        expert_logits, expert_indices = jax.lax.top_k(
            gate_logits, cfg.experts_per_token
        )
        expert_indices = expert_indices.astype(jnp.int32)
        expert_weights = jax.nn.softmax(expert_logits, axis=-1)

        if capture_routing is not None:
            capture_routing.append(
                {
                    "expert_ids": np.array(expert_indices),
                    "gate_weights": np.array(expert_weights),
                }
            )

        mlp1_kernel = jnp.transpose(self.mlp1_weight.value, (0, 2, 1))
        mlp2_kernel = jnp.transpose(self.mlp2_weight.value, (0, 2, 1))

        def _run_expert(
            mlp1_w: jax.Array,
            mlp1_b: jax.Array,
            mlp2_w: jax.Array,
            mlp2_b: jax.Array,
        ) -> jax.Array:
            mlp1 = _linear(
                normed,
                mlp1_w,
                mlp1_b,
            )
            gate_part = jnp.minimum(mlp1[:, : cfg.intermediate_size], 7.0)
            linear_part = jnp.clip(
                mlp1[:, cfg.intermediate_size :], -7.0, 7.0
            )
            swish_gate = gate_part * jax.nn.sigmoid(1.702 * gate_part)
            activated = swish_gate * (linear_part + 1.0)
            mlp2 = _linear(
                activated,
                mlp2_w,
                mlp2_b,
            )
            return mlp2

        expert_outputs = jax.vmap(
            _run_expert,
            in_axes=(0, 0, 0, 0),
            out_axes=0,
        )(
            mlp1_kernel,
            self.mlp1_bias.value,
            mlp2_kernel,
            self.mlp2_bias.value,
        )

        expert_outputs = expert_outputs.transpose(1, 0, 2)
        dense_gate_weights = jnp.sum(
            jax.nn.one_hot(
                expert_indices,
                cfg.num_experts,
                dtype=expert_outputs.dtype,
            )
            * expert_weights[..., None],
            axis=1,
        )
        fused = jnp.sum(
            expert_outputs * dense_gate_weights[..., None],
            axis=1,
        )
        return x + fused


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        *,
        config: GPTOSSConfig,
        cos_table: jax.Array,
        sin_table: jax.Array,
        sequence_length: int,
        mask: jax.Array,
        dtype: jnp.dtype = jnp.float32,
        rng: jax.Array | int | None = None,
    ):
        rng_seq = _KeySeq(rng)
        self.attention = AttentionBlock(
            config=config,
            cos_table=cos_table,
            sin_table=sin_table,
            sequence_length=sequence_length,
            mask=mask,
            dtype=dtype,
            rng=rng_seq.next(),
        )
        self.mlp = MLPBlock(
            config=config,
            sequence_length=sequence_length,
            dtype=dtype,
            rng=rng_seq.next(),
        )

    def __call__(
        self,
        x: jax.Array,
        capture_routing: Optional[List[dict]] = None,
    ) -> jax.Array:
        x = self.attention(x)
        x = self.mlp(x, capture_routing=capture_routing)
        return x


class Transformer(nnx.Module):
    def __init__(
        self,
        *,
        config: GPTOSSConfig,
        cos_table: jax.Array,
        sin_table: jax.Array,
        sequence_length: int,
        mask_sliding: jax.Array,
        mask_causal: jax.Array,
        dtype: jnp.dtype = jnp.float32,
        rng: jax.Array | int | None = None,
    ):
        self.config = config
        self.sequence_length = sequence_length
        self.cos_table = nnx.data(cos_table)
        self.sin_table = nnx.data(sin_table)
        self.mask_sliding = nnx.data(mask_sliding)
        self.mask_causal = nnx.data(mask_causal)

        rng_seq = _KeySeq(rng)
        self.embedding = _init_param(
            rng_seq.next(),
            _NORMAL_002,
            (config.vocab_size, config.hidden_size),
            dtype,
        )
        self.blocks = nnx.Dict()
        for layer_idx in range(config.num_hidden_layers):
            use_sliding = config.sliding_window if (layer_idx % 2 == 0) else 0
            mask = self.mask_sliding if use_sliding > 0 else self.mask_causal
            self.blocks[f"block_{layer_idx}"] = TransformerBlock(
                config=config,
                cos_table=self.cos_table,
                sin_table=self.sin_table,
                sequence_length=sequence_length,
                mask=mask,
                dtype=dtype,
                rng=rng_seq.next(),
            )
        self.norm = RMSNorm(config.hidden_size)
        self.unembedding_kernel = _init_param(
            rng_seq.next(),
            _XAVIER_UNIFORM,
            (config.hidden_size, config.vocab_size),
            dtype,
        )

    def __call__(
        self,
        tokens: jax.Array,
        capture_routing: Optional[List[List[dict]]] = None,
    ) -> jax.Array:
        cfg = self.config
        num_tokens = jax_core.concrete_or_error(
            int,
            tokens.shape[0],
            "Transformer requires static sequence length",
        )
        if num_tokens != self.sequence_length:
            raise ValueError(
                f"Transformer expected sequence_length={self.sequence_length}, got {num_tokens}"
            )
        embedding = self.embedding.value
        hidden = embedding[tokens.astype(jnp.int32)]

        num_blocks = self.config.num_hidden_layers
        if capture_routing is not None and len(capture_routing) != num_blocks:
            raise ValueError(
                f"capture_routing must contain {num_blocks} lists; got {len(capture_routing)}"
            )

        x = hidden
        for layer_idx in range(num_blocks):
            block = self.blocks[f"block_{layer_idx}"]
            layer_capture = (
                capture_routing[layer_idx] if capture_routing is not None else None
            )
            x = block(x, capture_routing=layer_capture)

        x = self.norm(x)
        logits = _linear(x, self.unembedding_kernel.value)
        return logits


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
    logits = lax.dot_general(
        Qh,
        Kh,
        dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    logits = logits * sm_scale
    logits = logits + mask[None, None, :, :]
    logits = _concat([logits, sinks], axis=-1)

    weights = _softmax(logits, axis=-1)[..., :-1]
    Vh = V.transpose(1, 2, 0, 3)
    attn = lax.dot_general(
        weights,
        Vh,
        dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).transpose(2, 0, 1, 3)
    return attn.reshape(n_new_tokens, -1)


sdpa: Final = jax.jit(
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


_SDPA_Q: Final[jax.Array] = jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 1, 4) / 10.0
_SDPA_K: Final[jax.Array] = jnp.linspace(0.0, 1.0, num=24, dtype=jnp.float32).reshape(
    3, 2, 4
)
_SDPA_V: Final[jax.Array] = jnp.linspace(1.0, -1.0, num=24, dtype=jnp.float32).reshape(
    3, 2, 4
)
_SDPA_S: Final[jax.Array] = jnp.array([0.1, -0.2], dtype=jnp.float32)


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
        mask=_causal_mask(
            sequence_length, sequence_length, sliding_window=config.sliding_window
        ),
        dtype=dtype,
        rng=init_key,
    )

    def _apply(x: jax.Array) -> jax.Array:
        return module(x)

    return _apply


_ATTN_Q: Final[jax.Array] = jnp.linspace(0.0, 1.0, num=96, dtype=jnp.float32).reshape(
    3, 32
)


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
        }
    ],
)


_MLP_CONFIG: Final[GPTOSSConfig] = GPTOSSConfig(
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
    module = MLPBlock(
        config=config,
        sequence_length=sequence_length,
        dtype=dtype,
        rng=init_key,
    )

    def _apply(x: jax.Array) -> jax.Array:
        return module(x)

    return _apply


_MLP_INPUT: Final[jax.Array] = jnp.linspace(
    -0.5, 0.75, num=96, dtype=jnp.float32
).reshape(3, 32)


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
        }
    ],
)


_TF_CONFIG: Final[GPTOSSConfig] = GPTOSSConfig(
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
        mask=_causal_mask(
            sequence_length, sequence_length, sliding_window=config.sliding_window
        ),
        dtype=dtype,
        rng=init_key,
    )

    def _apply(x: jax.Array) -> jax.Array:
        return module(x)

    return _apply


_TF_INPUT: Final[jax.Array] = jnp.linspace(
    -1.0, 1.0, num=96, dtype=jnp.float32
).reshape(3, 32)


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
        }
    ],
)
