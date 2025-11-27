"""Debug helper to compare Equinox and Flax/NNX RotaryEmbedding outputs.

Run directly:
    JAX_PLATFORM_NAME=cpu poetry run python scripts/debug_rope_parity.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins.examples.eqx.gpt_oss import GPTOSSConfig, RotaryEmbedding
from jax2onnx.plugins.examples.nnx.gpt_oss_flax import (
    RotaryEmbedding as FlaxRotaryEmbedding,
    _rotary_tables,
)


def main() -> None:
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
    query_mult = config.num_attention_heads // config.num_key_value_heads
    seq_len = 8
    batch = 1
    num_kv = config.num_key_value_heads

    # Deterministic inputs
    total = batch * seq_len * num_kv * query_mult * config.head_dim
    q_vals = jnp.arange(total, dtype=jnp.float32)
    k_vals = jnp.arange(total, dtype=jnp.float32) + 0.5
    q = q_vals.reshape(batch, seq_len, num_kv, query_mult, config.head_dim)
    k = k_vals.reshape(batch, seq_len, num_kv, query_mult, config.head_dim)

    # Equinox RoPE
    rope_eqx = RotaryEmbedding(config, dtype=np.float32)
    q_rot_eqx, k_rot_eqx = rope_eqx(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), seq_len=seq_len)

    # Flax/NNX RoPE (no batch dimension)
    cos_table, sin_table = _rotary_tables(
        head_dim=config.head_dim,
        table_length=config.initial_context_length,
    )
    rope_flax = FlaxRotaryEmbedding(
        head_dim=config.head_dim,
        cos_table=cos_table,
        sin_table=sin_table,
    )
    q_rot_flax, k_rot_flax = rope_flax(
        q[0].astype(jnp.bfloat16),
        k[0].astype(jnp.bfloat16),
        position_offset=0,
    )

    # Align shapes for comparison
    q_rot_flax = q_rot_flax[None, ...]
    k_rot_flax = k_rot_flax[None, ...]

    q_diff = np.max(np.abs(np.asarray(q_rot_eqx, dtype=np.float32) - np.asarray(q_rot_flax, dtype=np.float32)))
    k_diff = np.max(np.abs(np.asarray(k_rot_eqx, dtype=np.float32) - np.asarray(k_rot_flax, dtype=np.float32)))

    print("q_rot max |diff|:", float(q_diff))
    print("k_rot max |diff|:", float(k_diff))
    if q_diff == 0.0 and k_diff == 0.0:
        print("✅ RoPE parity achieved.")
    else:
        print("❌ RoPE parity mismatch detected.")


if __name__ == "__main__":
    main()
