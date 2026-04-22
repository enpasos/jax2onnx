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
            "argpartition",
            lambda x: jnp.argpartition(x, 2),
            [np.array([3.0, 1.0, 2.0, 4.0], dtype=np.float32)],
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
            "convolve",
            lambda x, y: jnp.convolve(x, y, mode="full"),
            [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([0.5, -1.0], dtype=np.float32),
            ],
        ),
        (
            "correlate",
            lambda x, y: jnp.correlate(x, y, mode="full"),
            [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([0.5, -1.0], dtype=np.float32),
            ],
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
            "extract",
            lambda x: jnp.extract(
                jnp.array([True, False, True]),
                x,
                size=3,
                fill_value=-1.0,
            ),
            [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
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
            "intersect1d",
            lambda x, y: jnp.intersect1d(x, y, size=3, fill_value=-1),
            [
                np.array([1, 2, 3, 4], dtype=np.int32),
                np.array([2, 4, 6], dtype=np.int32),
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
            "lexsort",
            lambda keys: jnp.lexsort(keys),
            [np.array([[2, 1, 2, 1], [0, 0, 1, 1]], dtype=np.int32)],
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
            "mask_indices",
            lambda: jnp.mask_indices(3, jnp.triu, size=6),
            [],
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
        (
            "packbits",
            lambda x: jnp.packbits(x, bitorder="big"),
            [np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)],
        ),
        (
            "poly",
            lambda x: jnp.poly(x),
            [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
        ),
        (
            "polymul",
            lambda x, y: jnp.polymul(x, y),
            [
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([3.0, 4.0, 5.0], dtype=np.float32),
            ],
        ),
        (
            "polyval",
            lambda coeffs, x: jnp.polyval(coeffs, x),
            [
                np.array([2.0, -3.0, 4.0], dtype=np.float32),
                np.array([1.5, -2.0], dtype=np.float32),
            ],
        ),
        (
            "place",
            lambda x, vals: jnp.place(
                x,
                jnp.array([True, False, True]),
                vals,
                inplace=False,
            ),
            [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([9, 8], dtype=np.int32),
            ],
        ),
        (
            "tril_indices",
            lambda: jnp.tril_indices(3),
            [],
        ),
        (
            "tril_indices_from",
            lambda x: jnp.tril_indices_from(x),
            [np.zeros((3, 3), dtype=np.float32)],
        ),
        (
            "triu_indices",
            lambda: jnp.triu_indices(3),
            [],
        ),
        (
            "triu_indices_from",
            lambda x: jnp.triu_indices_from(x),
            [np.zeros((3, 3), dtype=np.float32)],
        ),
        (
            "unique_all",
            lambda x: jnp.unique_all(x, size=4, fill_value=-1),
            [np.array([3, 1, 3, 2], dtype=np.int32)],
        ),
        (
            "unique_counts",
            lambda x: jnp.unique_counts(x, size=4, fill_value=-1),
            [np.array([3, 1, 3, 2], dtype=np.int32)],
        ),
        (
            "unique_inverse",
            lambda x: jnp.unique_inverse(x, size=4, fill_value=-1),
            [np.array([3, 1, 3, 2], dtype=np.int32)],
        ),
        (
            "unique_values",
            lambda x: jnp.unique_values(x, size=4, fill_value=-1),
            [np.array([3, 1, 3, 2], dtype=np.int32)],
        ),
        (
            "unpackbits",
            lambda x: jnp.unpackbits(x, bitorder="big"),
            [np.array([178], dtype=np.uint8)],
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


def test_jnp_empty_reuses_zero_broadcast_export(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")

    def fn() -> jnp.ndarray:
        return jnp.empty((2, 3), dtype=jnp.float32)

    model_path = tmp_path / "empty.onnx"
    to_onnx(fn, inputs=[], return_mode="file", output_path=model_path)

    matches, message = allclose(fn, str(model_path), [], atol=1e-5, rtol=1e-4)
    assert matches, message
