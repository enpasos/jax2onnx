# tests/test_gpt_oss_components.py

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins.examples.eqx.gpt_oss import GPTOSSConfig, RotaryEmbedding


def _rotate_half_reference(
    x: np.ndarray, cos: np.ndarray, sin: np.ndarray
) -> np.ndarray:
    """RotateHalf RoPE reference using cached sin/cos tables."""

    x_np = np.asarray(x, dtype=np.float32)
    cos_np = np.asarray(cos, dtype=np.float32)
    sin_np = np.asarray(sin, dtype=np.float32)
    seq_len = x_np.shape[1]
    head_dim = x_np.shape[-1]
    head_dim // 2

    x_moved = np.moveaxis(x_np, 1, 0)
    flat = x_moved.reshape(seq_len, -1, head_dim)
    first, second = np.split(flat, 2, axis=-1)
    cos_flat = cos_np[:, None, :]
    sin_flat = sin_np[:, None, :]

    out_first = first * cos_flat - second * sin_flat
    out_second = second * cos_flat + first * sin_flat
    rotated_flat = np.concatenate([out_first, out_second], axis=-1)

    rotated = rotated_flat.reshape((seq_len,) + x_moved.shape[1:-1] + (head_dim,))
    return np.moveaxis(rotated, 0, 1)


def test_rotary_embedding_uses_float32_math_with_bfloat16_inputs() -> None:
    config = GPTOSSConfig(
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        initial_context_length=16,
        rope_theta=10_000.0,
        rope_scaling_factor=1.0,
        rope_ntk_alpha=1.0,
        rope_ntk_beta=1.0,
    )
    rope = RotaryEmbedding(config, dtype=np.float32)
    assert isinstance(rope.dtype, np.dtype)

    seq_len = 8
    batch = 1
    num_kv = config.num_key_value_heads
    query_mult = config.num_attention_heads // config.num_key_value_heads

    key = jax.random.PRNGKey(0)
    q_base = jax.random.normal(
        key, (batch, seq_len, num_kv, query_mult, config.head_dim)
    )
    key, subkey = jax.random.split(key)
    k_base = jax.random.normal(subkey, (batch, seq_len, num_kv, config.head_dim))

    q_bf16 = q_base.astype(jnp.bfloat16)
    k_bf16 = k_base.astype(jnp.bfloat16)
    q_rot, k_rot = rope(q_bf16, k_bf16, seq_len=seq_len)

    cos_ref = rope._cos_cache[:seq_len]
    sin_ref = rope._sin_cache[:seq_len]
    q_ref = _rotate_half_reference(q_bf16, cos_ref, sin_ref).astype(jnp.bfloat16)
    k_ref = _rotate_half_reference(k_bf16, cos_ref, sin_ref).astype(jnp.bfloat16)

    np.testing.assert_allclose(
        np.asarray(q_rot, dtype=np.float32),
        np.asarray(q_ref, dtype=np.float32),
        rtol=5e-4,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        np.asarray(k_rot, dtype=np.float32),
        np.asarray(k_ref, dtype=np.float32),
        rtol=5e-4,
        atol=5e-4,
    )
