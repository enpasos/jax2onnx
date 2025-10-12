# jax2onnx/plugins/jax/lax/_reduction_registry.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ReductionTestSpec:
    testcase: str
    axis: tuple[int, ...] | str | None
    keepdims: bool = False
    dtype: np.dtype | None = None
    values: np.ndarray | None = None
    skip_numeric: bool = False


REDUCTION_TESTS: Dict[str, List[ReductionTestSpec]] = {
    "reduce_max": [
        ReductionTestSpec("reduce_max", axis=None),
        ReductionTestSpec("reduce_max_allaxes", axis=None),
        ReductionTestSpec("reduce_max_axes_input", axis=(1,)),
        ReductionTestSpec("reduce_max_keepdims", axis=(1,), keepdims=True),
    ],
    "reduce_min": [
        ReductionTestSpec("reduce_min", axis=None),
        ReductionTestSpec("reduce_min_allaxes", axis=None),
        ReductionTestSpec("reduce_min_keepdims", axis=(1,), keepdims=True),
    ],
    "reduce_or": [
        ReductionTestSpec(
            "reduce_or_all_false",
            axis=None,
            values=np.zeros((3, 4), dtype=bool),
            skip_numeric=True,
        ),
        ReductionTestSpec(
            "reduce_or_one_true",
            axis=None,
            values=np.array([[False, False], [True, False]], dtype=bool),
            skip_numeric=True,
        ),
        ReductionTestSpec("reduce_or_keepdims", axis=(1,), keepdims=True),
    ],
    "reduce_and": [
        ReductionTestSpec(
            "reduce_and_all_true",
            axis=None,
            values=np.ones((3, 4), dtype=bool),
            skip_numeric=True,
        ),
        ReductionTestSpec(
            "reduce_and_one_false",
            axis=None,
            values=np.array([[True, True], [True, False]], dtype=bool),
            skip_numeric=True,
        ),
        ReductionTestSpec("reduce_and_keepdims", axis=(1,), keepdims=True),
    ],
    "reduce_prod": [
        ReductionTestSpec("reduce_prod", axis=None),
        ReductionTestSpec("reduce_prod_allaxes", axis=None),
        ReductionTestSpec(
            "reduce_prod_dtype",
            axis=None,
            dtype=np.float32,
            values=np.ones((2, 3), dtype=np.float32),
        ),
        ReductionTestSpec(
            "reduce_prod_dtype_f64",
            axis=None,
            dtype=np.float64,
            values=np.ones((2, 3), dtype=np.float64),
        ),
        ReductionTestSpec("reduce_prod_keepdims", axis=(1,), keepdims=True),
    ],
    "reduce_sum": [
        ReductionTestSpec("reduce_sum", axis=None),
        ReductionTestSpec("reduce_sum_allaxes", axis=None),
        ReductionTestSpec(
            "reduce_sum_dtype",
            axis=None,
            dtype=np.float32,
            values=np.arange(6, dtype=np.float32).reshape(2, 3),
        ),
        ReductionTestSpec(
            "reduce_sum_dtype_f64",
            axis=None,
            dtype=np.float64,
            values=np.arange(6, dtype=np.float64).reshape(2, 3),
        ),
        ReductionTestSpec("reduce_sum_keepdims", axis=(1,), keepdims=True),
    ],
    "reduce_xor": [
        ReductionTestSpec(
            "reduce_xor_all_false",
            axis=None,
            values=np.zeros((3, 4), dtype=bool),
            skip_numeric=True,
        ),
        ReductionTestSpec(
            "reduce_xor_one_true",
            axis=None,
            values=np.array([[True, False], [False, False]], dtype=bool),
            skip_numeric=True,
        ),
        ReductionTestSpec(
            "reduce_xor_two_true",
            axis=None,
            values=np.array([[True, False], [False, True]], dtype=bool),
            skip_numeric=True,
        ),
        ReductionTestSpec("reduce_xor_keepdims", axis=(1,), keepdims=True),
    ],
}


__all__ = ["ReductionTestSpec", "REDUCTION_TESTS"]
