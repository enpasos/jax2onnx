# tests/extra_tests/test_round_precision_contract.py

"""Native rounding methods retain ties, adjacent values and zero signs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.extra_tests.test_precision_contract import check


def _values(dtype):
    cast = np.dtype(dtype).type
    half = cast(0.5)
    below = np.nextafter(half, cast(0))
    above = np.nextafter(half, cast(1))
    tiny = jnp.finfo(dtype).smallest_subnormal
    return np.asarray(
        [
            -np.inf,
            -65504,
            -4.5,
            -3.5,
            -2.5,
            -1.5,
            -above,
            -half,
            -below,
            -tiny,
            -0.0,
            0.0,
            tiny,
            below,
            half,
            above,
            1.5,
            2.5,
            3.5,
            4.5,
            65504,
            np.inf,
            np.nan,
        ],
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize(
    "method",
    [jax.lax.RoundingMethod.AWAY_FROM_ZERO, jax.lax.RoundingMethod.TO_NEAREST_EVEN],
)
def test_round_methods_keep_exact_source_results_and_shared_input(dtype, method):
    values = _values(dtype)

    def source(x):
        return jax.lax.round(x, rounding_method=method), x

    with jax.enable_x64():
        _, reference = check(source, [values])
    expected_half = 1 if method == jax.lax.RoundingMethod.AWAY_FROM_ZERO else 0
    assert reference[0][7] == -expected_half
    assert reference[0][14] == expected_half
    np.testing.assert_array_equal(reference[1], values)


@pytest.mark.parametrize(
    "method",
    [jax.lax.RoundingMethod.AWAY_FROM_ZERO, jax.lax.RoundingMethod.TO_NEAREST_EVEN],
)
def test_round_bfloat16_model_calculation_keeps_source_boundary(method):
    values = _values(jnp.bfloat16).astype(np.float32)

    def source(x):
        prepared = x.astype(jnp.bfloat16)
        return jax.lax.round(prepared, rounding_method=method).astype(jnp.float32), x

    with jax.enable_x64():
        check(source, [values])
