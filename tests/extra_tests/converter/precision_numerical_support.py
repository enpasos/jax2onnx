# tests/extra_tests/converter/precision_numerical_support.py

"""Model-chosen wider variance work and an independent staged reference.

Shared by durable source regressions and the internal numerical experiment.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np


def selected_variance_normalization(x: jax.Array) -> tuple[jax.Array, ...]:
    """Only the variance sum/division use FP64; centering/squares stay FP32."""
    mean = jax.lax.reduce_sum(x, axes=(1,)) / np.float32(16)
    centered = x - mean[:, None]
    squares = centered * centered
    variance64 = jax.lax.reduce_sum(squares.astype(jnp.float64), axes=(1,))
    variance = (variance64 / np.float64(16)).astype(jnp.float32)
    normalized = centered / jnp.sqrt(variance[:, None] + np.float32(1e-5))
    return centered, squares, variance[:, None], normalized


def normalization_input() -> np.ndarray:
    return np.asarray([[8192, -8192] + [1, -1] * 7, [1e8] * 16], dtype=np.float32)


def normalization_reference(x: np.ndarray) -> tuple[np.ndarray, ...]:
    """Independent staged reference using math.fsum over rounded squares.

    The chosen rows have an exactly representable mean, so reduction ordering
    of the FP32 mean is immaterial for this sample. This is not an arbitrary
    input oracle for JAX's reduction order.
    """
    assert x.shape == (2, 16) and x.dtype == np.float32
    mean = np.asarray([math.fsum(row) / 16 for row in x], dtype=np.float32)
    centered = np.subtract(x, mean[:, None], dtype=np.float32)
    squares = np.multiply(centered, centered, dtype=np.float32)
    variance = np.asarray(
        [[math.fsum(float(v) for v in row) / 16] for row in squares],
        dtype=np.float32,
    )
    denominator = np.sqrt(np.add(variance, np.float32(1e-5), dtype=np.float32))
    normalized = np.divide(centered, denominator, dtype=np.float32)
    return centered, squares, variance, normalized
