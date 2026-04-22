# tests/extra_tests/framework/test_jnp_composite_api_reuse.py

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx import allclose, to_onnx


@pytest.mark.parametrize(
    ("case_id", "fn", "inputs"),
    [
        (
            "argwhere",
            lambda x: jnp.argwhere(x, size=3),
            [np.array([0, 2, 0, 3], dtype=np.int32)],
        ),
        (
            "average",
            lambda x: jnp.average(x, axis=0),
            [np.arange(6, dtype=np.float32).reshape(2, 3)],
        ),
        (
            "bincount",
            lambda x: jnp.bincount(x, length=4),
            [np.array([0, 1, 1, 3], dtype=np.int32)],
        ),
        (
            "diff",
            lambda x: jnp.diff(x),
            [np.arange(5, dtype=np.float32)],
        ),
        (
            "divmod",
            lambda x, y: jnp.divmod(x, y),
            [
                np.array([5.0, -5.0], dtype=np.float32),
                np.array([2.0, 2.0], dtype=np.float32),
            ],
        ),
        (
            "fmax",
            lambda x, y: jnp.fmax(x, y),
            [
                np.array([1.0, np.nan], dtype=np.float32),
                np.array([2.0, 3.0], dtype=np.float32),
            ],
        ),
        (
            "fmin",
            lambda x, y: jnp.fmin(x, y),
            [
                np.array([1.0, np.nan], dtype=np.float32),
                np.array([2.0, 3.0], dtype=np.float32),
            ],
        ),
        (
            "heaviside",
            lambda x, y: jnp.heaviside(x, y),
            [
                np.array([-1.0, 0.0, 2.0], dtype=np.float32),
                np.array([0.5, 0.5, 0.5], dtype=np.float32),
            ],
        ),
        (
            "isin",
            lambda x, y: jnp.isin(x, y),
            [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([2, 4], dtype=np.int32),
            ],
        ),
        (
            "linalg_slogdet",
            lambda x: jnp.linalg.slogdet(x),
            [np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float32)],
        ),
        (
            "logaddexp",
            lambda x, y: jnp.logaddexp(x, y),
            [
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([3.0, 4.0], dtype=np.float32),
            ],
        ),
        (
            "logaddexp2",
            lambda x, y: jnp.logaddexp2(x, y),
            [
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([3.0, 4.0], dtype=np.float32),
            ],
        ),
        (
            "modf",
            lambda x: jnp.modf(x),
            [np.array([-1.5, 2.25], dtype=np.float32)],
        ),
        (
            "nan_to_num",
            lambda x: jnp.nan_to_num(x),
            [np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)],
        ),
        (
            "nonzero",
            lambda x: jnp.nonzero(x, size=3),
            [np.array([0, 2, 0, 3], dtype=np.int32)],
        ),
    ],
)
def test_jnp_composite_api_reuse_exports_and_runs(
    tmp_path: Path,
    case_id: str,
    fn: Callable[..., Any],
    inputs: list[np.ndarray],
) -> None:
    pytest.importorskip("onnxruntime")

    model_path = tmp_path / f"{case_id}.onnx"
    to_onnx(fn, inputs=inputs, return_mode="file", output_path=model_path)

    matches, message = allclose(fn, str(model_path), inputs, atol=1e-5, rtol=1e-4)
    assert matches, message
