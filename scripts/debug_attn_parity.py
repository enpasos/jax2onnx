"""Debug helper to stress-test SDPA with large sinks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax2onnx.plugins.examples.eqx.gpt_oss import _sdpa_torch_style


def main() -> None:
    B, H, M, T, D = 1, 4, 2, 8, 16
    dtype = jnp.bfloat16

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    q = jax.random.normal(k1, (T, H, M, D), dtype=dtype)
    k = jax.random.normal(k2, (T, H, D), dtype=dtype)
    v = jax.random.normal(k3, (T, H, D), dtype=dtype)

    # High-variance sinks to reproduce probe conditions
    sinks = jax.random.normal(k4, (H, M), dtype=dtype) * jnp.asarray(1.0, dtype=dtype)

    out = _sdpa_torch_style(
        q,
        k,
        v,
        sinks=sinks,
        sm_scale=jnp.asarray(1.0, dtype=jnp.float32),
        sliding_window=0,
    )

    print("SDPA Output Stats:")
    print(f"Mean: {jnp.mean(out)}")
    print(f"Max:  {jnp.max(out)}")
    print(f"Is Finite: {jnp.all(jnp.isfinite(out))}")

    if jnp.max(jnp.abs(out)) > 10.0:
        print("❌ FAILED: Output magnitude too high (likely sink weighting issue).")
    else:
        print("✅ PASSED: Output magnitude within expected range.")


if __name__ == "__main__":
    main()
