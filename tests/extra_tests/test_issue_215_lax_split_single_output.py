# tests/extra_tests/test_issue_215_lax_split_single_output.py

from __future__ import annotations

import numpy as np
from onnx.reference import ReferenceEvaluator

import jax
import jax.numpy as jnp

from jax2onnx.user_interface import to_onnx


def _lax_split_single_output(x: jax.Array) -> jax.Array:
    return jax.lax.split(x, (6,), axis=1)[0] * 2


def test_issue_215_lax_split_single_output_exports_and_runs() -> None:
    model = to_onnx(
        _lax_split_single_output,
        [(3, 6)],
        model_name="issue215_lax_split_single_output",
    )

    x = np.arange(18, dtype=np.float32).reshape(3, 6)
    expected = np.asarray(_lax_split_single_output(jnp.asarray(x)))

    evaluator = ReferenceEvaluator(model)
    input_name = model.graph.input[0].name
    (got,) = evaluator.run(None, {input_name: x})

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
