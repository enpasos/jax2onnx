# tests/extra_tests/test_opset_operator_availability.py

"""Lowerings choose operators and signatures that exist at the target opset."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest
from flax import nnx

from jax2onnx import to_onnx


def _run(model: onnx.ModelProto, *feeds: np.ndarray) -> list[np.ndarray]:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    names = [value.name for value in session.get_inputs()]
    return list(session.run(None, dict(zip(names, feeds, strict=True))))


def _node(model: onnx.ModelProto, op_type: str) -> onnx.NodeProto:
    (node,) = [node for node in model.graph.node if node.op_type == op_type]
    return node


def test_hard_swish_is_decomposed_before_opset_14() -> None:
    # The kinks at -3 and 3 and both saturated sides.
    x = np.asarray([-5.0, -3.0, -1.5, 0.0, 0.5, 3.0, 7.0], dtype=np.float32)
    expected = np.asarray(jax.nn.hard_swish(jnp.asarray(x)))

    fallback = to_onnx(lambda v: jax.nn.hard_swish(v), [jnp.asarray(x)], opset=13)
    assert [node.op_type for node in fallback.graph.node] == ["HardSigmoid", "Mul"]
    np.testing.assert_allclose(_run(fallback, x)[0], expected, rtol=1e-6, atol=1e-6)

    native = to_onnx(lambda v: jax.nn.hard_swish(v), [jnp.asarray(x)], opset=14)
    assert [node.op_type for node in native.graph.node] == ["HardSwish"]
    np.testing.assert_allclose(_run(native, x)[0], expected, rtol=1e-6, atol=1e-6)


def _nnx_rms_norm() -> Callable[[jax.Array], jax.Array]:
    module: Callable[[jax.Array], jax.Array] = nnx.RMSNorm(6, rngs=nnx.Rngs(0))
    return module


def _eqx_rms_norm() -> Callable[[jax.Array], jax.Array]:
    return jax.vmap(eqx.nn.RMSNorm(6))


@pytest.mark.parametrize("make", [_nnx_rms_norm, _eqx_rms_norm], ids=["nnx", "eqx"])
@pytest.mark.parametrize("opset", [13, 17, 18])
def test_rms_norm_mean_uses_the_axes_form_of_the_opset(
    make: Callable[[], Callable[[jax.Array], jax.Array]], opset: int
) -> None:
    module = make()
    x = np.random.default_rng(0).normal(size=(2, 6)).astype(np.float32) * 3
    model = to_onnx(module, [jnp.asarray(x)], opset=opset)

    mean = _node(model, "ReduceMean")
    attributes: dict[str, Any] = {attr.name: attr for attr in mean.attribute}
    if opset < 18:
        # ReduceMean takes its axes as an attribute before opset 18.
        assert len(mean.input) == 1
        assert list(attributes["axes"].ints) == [1]
    else:
        assert len(mean.input) == 2
        assert "axes" not in attributes
    np.testing.assert_allclose(
        _run(model, x)[0], np.asarray(module(jnp.asarray(x))), rtol=1e-6, atol=1e-6
    )


_KEY = jax.random.PRNGKey(0)

_RANDOM_CASES: dict[str, tuple[Callable[[], jax.Array], np.dtype[Any]]] = {
    "normal": (lambda: jax.random.normal(_KEY, (64,)), np.dtype(np.float32)),
    "bernoulli": (lambda: jax.random.bernoulli(_KEY, 0.25, (64,)), np.dtype(bool)),
    "bits": (
        lambda: jax.random.bits(_KEY, (64,), dtype=jnp.uint32),
        np.dtype(np.uint32),
    ),
    "rng_uniform": (lambda: jax.lax.rng_uniform(0.0, 1.0, (64,)), np.dtype(np.float32)),
}


@pytest.mark.parametrize("name", _RANDOM_CASES)
def test_random_exports_run_on_onnx_runtime_at_opset_21(name: str) -> None:
    # CPU ONNX Runtime 1.31 has no kernels for the opset-22 versions of the
    # random operators, so opset 21 is the newest opset whose random exports
    # it loads. See the kernel-gap table in docs/about/dependencies.md.
    fn, dtype = _RANDOM_CASES[name]
    (sample,) = _run(to_onnx(fn, [], opset=21))

    assert sample.shape == (64,) and sample.dtype == dtype
