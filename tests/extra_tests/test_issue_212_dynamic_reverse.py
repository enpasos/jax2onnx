# tests/extra_tests/test_issue_212_dynamic_reverse.py

from __future__ import annotations

from typing import Callable

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest

import jax
import jax.numpy as jnp
from jax import lax

from jax2onnx.user_interface import to_onnx


def _dynamic_length_spec() -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        jax.export.symbolic_shape("N", constraints=("N >= 1024",)),
        dtype=np.float32,
    )


def _run_model_with_onnx_runtime(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    ort = pytest.importorskip(
        "onnxruntime", reason="onnxruntime is required for issue #212 regression test"
    )
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: x})[0]


def _run_model_with_reference_evaluator(
    model: onnx.ModelProto, x: np.ndarray
) -> np.ndarray:
    onnx.checker.check_model(model)
    ref_eval = ReferenceEvaluator(model)
    input_name = model.graph.input[0].name
    return ref_eval.run(None, {input_name: x})[0]


@pytest.mark.parametrize(
    ("approach_name", "reverse_fn"),
    [
        ("jnp.flip", lambda x: jnp.flip(x, axis=0)),
        ("lax.rev", lambda x: lax.rev(x, dimensions=(0,))),
        ("x[::-1]", lambda x: x[::-1]),
        (
            "advanced indexing",
            lambda x: x[jnp.arange(x.shape[0] - 1, -1, -1)],
        ),
    ],
    ids=["jnp_flip", "lax_rev", "slice_reverse", "advanced_indexing"],
)
@pytest.mark.parametrize("dynamic_shape", [False, True], ids=["static", "dynamic"])
def test_issue_212_reverse_exports_and_runs(
    approach_name: str, reverse_fn: Callable, dynamic_shape: bool
) -> None:
    model = to_onnx(
        reverse_fn,
        [_dynamic_length_spec() if dynamic_shape else (1024,)],
        model_name=f"issue212_{approach_name.replace(' ', '_')}",
        opset=22,
    )

    input_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    if dynamic_shape:
        assert input_dim.dim_param == "N"
    else:
        assert input_dim.dim_value == 1024

    x = np.random.default_rng(0).uniform(size=(1024,)).astype(np.float32)
    y_jax = np.asarray(reverse_fn(x))

    assert np.array_equal(y_jax, x[::-1])

    for run_fn, eval_name in [
        (_run_model_with_reference_evaluator, "ONNX Reference Evaluator"),
        (_run_model_with_onnx_runtime, "ONNX Runtime"),
    ]:
        y_model = run_fn(model, x)
        assert y_model.shape == y_jax.shape == (1024,), (
            f"Expected shape (1024,) but got {y_jax.shape} (JAX) "
            f"and {y_model.shape} ({eval_name})"
        )
        np.testing.assert_array_equal(y_model, y_jax)
