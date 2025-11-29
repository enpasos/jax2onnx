"""Debug helper to compare Equinox RMSNorm against a NumPy reference."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins.examples.eqx.gpt_oss import RMSNorm


def numpy_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x_f64 = x.astype(np.float64)
    ms = np.mean(x_f64**2, axis=-1, keepdims=True)
    rms = np.sqrt(ms + eps)
    normed = x_f64 / rms
    return (normed * weight.astype(np.float64)).astype(np.float32)


def main() -> None:
    hidden_size = 128
    eps = 1e-5

    np.random.seed(123)
    # Match GPT-OSS embedding scale (~0.02)
    x = (np.random.randn(1, 32, hidden_size) * 0.02).astype(np.float32)
    weight = np.random.randn(hidden_size).astype(np.float32)

    norm_eqx = RMSNorm(hidden_size, eps=eps)
    # Overwrite only the weight leaf; keep eps intact
    norm_eqx = eqx.tree_at(lambda m: m.weight, norm_eqx, jnp.asarray(weight))

    out_eqx = norm_eqx(jnp.asarray(x))
    out_np = numpy_rmsnorm(x, weight, eps)

    diff = np.max(np.abs(np.asarray(out_eqx) - np.asarray(out_np)))
    print(f"RMSNorm Max Diff: {diff:.8f}")
    if diff > 1e-5:
        print("❌ RMSNorm parity mismatch.")
    else:
        print("✅ RMSNorm parity achieved.")


if __name__ == "__main__":
    main()
