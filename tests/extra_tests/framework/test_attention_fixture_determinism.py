# tests/extra_tests/framework/test_attention_fixture_determinism.py

"""Attention sample recipes preserve domains and leave application RNG untouched."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from jax2onnx.plugins.examples.nnx.gpt import (
    _GPT_ATTENTION_FIXTURE_SEED,
    _gpt_attention_fixture_inputs,
)
from jax2onnx.plugins.flax.nnx.dot_product_attention import (
    _DPA_CAUSAL_FIXTURE_SEED,
    _dpa_causal_fixture_inputs,
)


@pytest.mark.parametrize(
    "recipe,seed,shape,mask_shape",
    [
        (
            _gpt_attention_fixture_inputs,
            _GPT_ATTENTION_FIXTURE_SEED,
            (1, 1024, 12, 64),
            (1, 12, 1024, 1024),
        ),
        (
            _dpa_causal_fixture_inputs,
            _DPA_CAUSAL_FIXTURE_SEED,
            (1, 8, 4, 16),
            (1, 4, 8, 8),
        ),
    ],
)
def test_attention_fixture_domains_draw_order_and_global_rng_are_preserved(
    recipe: Any, seed: int, shape: tuple[int, ...], mask_shape: tuple[int, ...]
) -> None:
    original = np.random.get_state()
    try:
        np.random.seed(713)
        caller_state = np.random.get_state()
        first = recipe()
        after_state = np.random.get_state()
        assert caller_state[0] == after_state[0]
        np.testing.assert_array_equal(caller_state[1], after_state[1])
        assert caller_state[2:] == after_state[2:]
        second = recipe()
        expected_rng = np.random.default_rng(seed)
        for actual, repeated in zip(first[:3], second[:3], strict=True):
            assert actual.shape == shape and actual.dtype == np.float32
            np.testing.assert_array_equal(actual, repeated, strict=True)
            np.testing.assert_array_equal(
                actual,
                expected_rng.standard_normal(shape).astype(np.float32),
                strict=True,
            )
        assert not np.array_equal(first[0], first[1])
        assert first[3].shape == mask_shape and first[3].dtype == np.bool_
        np.testing.assert_array_equal(
            first[3], np.tril(np.ones(mask_shape, dtype=bool)), strict=True
        )
    finally:
        np.random.set_state(original)


def test_fresh_imports_have_identical_payloads_and_preserve_global_numpy_rng() -> None:
    repository = Path(__file__).resolve().parents[3]
    probe = """
import hashlib
import json
import numpy as np
# Load dependencies before measuring fixture registration's RNG effects.
import jax2onnx
from flax import nnx
import onnx_ir
np.random.seed(123)
expected = np.random.get_state()
from jax2onnx.plugins.examples.nnx.gpt import _gpt_attention_fixture_inputs
from jax2onnx.plugins.flax.nnx.dot_product_attention import _dpa_causal_fixture_inputs
actual = np.random.get_state()
assert expected[0] == actual[0]
np.testing.assert_array_equal(expected[1], actual[1])
assert expected[2:] == actual[2:]
print(json.dumps([[hashlib.sha256(value.tobytes()).hexdigest() for value in recipe()] for recipe in (_gpt_attention_fixture_inputs, _dpa_causal_fixture_inputs)]))
"""
    results = [
        subprocess.run(
            [sys.executable, "-c", probe],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        for _ in range(2)
    ]
    assert json.loads(results[0].stdout) == json.loads(results[1].stdout)
