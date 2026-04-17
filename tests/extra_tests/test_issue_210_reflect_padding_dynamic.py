# tests/extra_tests/test_issue_210_reflect_padding_dynamic.py

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx.user_interface import to_onnx


def _symbolic_length_spec() -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        jax.export.symbolic_shape("N", constraints=("N >= 2",)),
        dtype=jnp.float32,
    )


def _reflect_pad_1d(x: jax.Array) -> jax.Array:
    return jnp.pad(x[None, :, None], ((0, 0), (1, 1), (0, 0)), mode="reflect")


def _build_ort_session(model):
    ort = pytest.importorskip("onnxruntime")
    return ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def test_reflect_pad_with_symbolic_length_matches_onnx_ir() -> None:
    model = to_onnx(
        _reflect_pad_1d,
        [_symbolic_length_spec()],
        model_name="issue210_reflect_pad_dynamic",
    )

    input_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    assert input_dim.dim_param == "N"

    session = _build_ort_session(model)
    input_name = session.get_inputs()[0].name
    rng = np.random.default_rng(0)

    for length in (2, 5, 17):
        x = rng.normal(size=(length,)).astype(np.float32)
        expected = np.asarray(_reflect_pad_1d(x))
        (got,) = session.run(None, {input_name: x})

        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_issue_210_nnx_conv_reflect_with_symbolic_length_matches_onnx_ir() -> None:
    conv = nnx.Conv(
        in_features=1,
        out_features=1,
        kernel_size=3,
        padding="REFLECT",
        rngs=nnx.Rngs(0),
    )

    def apply_conv(x: jax.Array) -> jax.Array:
        return conv(x[None, :, None])

    model = to_onnx(
        apply_conv,
        [_symbolic_length_spec()],
        model_name="issue210_nnx_conv_reflect_dynamic",
    )

    input_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    assert input_dim.dim_param == "N"

    session = _build_ort_session(model)
    input_name = session.get_inputs()[0].name
    rng = np.random.default_rng(1)

    for length in (2, 5, 17):
        x = rng.normal(size=(length,)).astype(np.float32)
        expected = np.asarray(apply_conv(x))
        (got,) = session.run(None, {input_name: x})

        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
