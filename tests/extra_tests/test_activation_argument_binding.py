# tests/extra_tests/test_activation_argument_binding.py

from __future__ import annotations

from collections.abc import Callable
import importlib.util
import os
import subprocess
import sys
import textwrap
from typing import Any

from flax import linen as nn
from flax.linen import activation as linen_activation
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx


_APIS = {"jax_nn": jax.nn, "linen": nn, "linen_activation": linen_activation}
_PARAMETER_NAMES = {
    "gelu": "approximate",
    "elu": "alpha",
    "leaky_relu": "negative_slope",
    "celu": "alpha",
}


def _activation(
    x: jax.Array,
    *,
    name: str,
    parameter: bool | float,
    call_style: str = "positional",
    api: str = "jax_nn",
) -> jax.Array:
    # Resolve the function during tracing so the export-time binding applies.
    fn = getattr(_APIS[api], name)
    if call_style == "positional":
        return fn(x, parameter)
    if call_style == "default":
        return fn(x)
    kwargs = {_PARAMETER_NAMES[name]: parameter}
    if call_style == "keyword":
        return fn(x, **kwargs)
    if call_style == "input_keyword":
        return fn(x=x, **kwargs)
    raise AssertionError(f"Unknown call style: {call_style}")


def _input() -> jax.Array:
    # Negative values distinguish custom alpha/slope from the defaults.
    return jnp.asarray([[-3.0, -0.7, -0.1], [0.0, 0.4, 2.0]], dtype=jnp.float32)


def _assert_ort_parity(
    fn: Callable[[jax.Array], jax.Array], *, opset: int = 20
) -> onnx.ModelProto:
    x = _input()
    expected = np.asarray(fn(x))
    model = to_onnx(fn, [x], opset=opset)
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    (graph_input,) = session.get_inputs()
    (actual,) = session.run(None, {graph_input.name: np.asarray(x)})
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    return model


def _attributes(model: onnx.ModelProto, op_type: str) -> dict[str, Any]:
    (node,) = [node for node in model.graph.node if node.op_type == op_type]
    assert len(node.input) == 1
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


@pytest.mark.parametrize("opset", [18, 20, 23])
@pytest.mark.parametrize("approximate", [False, True], ids=["exact", "tanh"])
def test_gelu_positional_argument_preserves_opset(
    approximate: bool, opset: int
) -> None:
    model = _assert_ort_parity(lambda x: jax.nn.gelu(x, approximate), opset=opset)
    op_types = {node.op_type for node in model.graph.node}
    if opset >= 20:
        assert _attributes(model, "Gelu")["approximate"] == (
            b"tanh" if approximate else b"none"
        )
    else:
        assert "Gelu" not in op_types
        assert ("Tanh" if approximate else "Erf") in op_types


@pytest.mark.parametrize("call_style", ["keyword", "input_keyword"])
@pytest.mark.parametrize("approximate", [False, True], ids=["exact", "tanh"])
def test_gelu_keyword_arguments(approximate: bool, call_style: str) -> None:
    model = _assert_ort_parity(
        lambda x: _activation(
            x, name="gelu", parameter=approximate, call_style=call_style
        )
    )
    assert _attributes(model, "Gelu")["approximate"] == (
        b"tanh" if approximate else b"none"
    )


def test_gelu_default_argument() -> None:
    model = _assert_ort_parity(lambda x: jax.nn.gelu(x))
    assert _attributes(model, "Gelu")["approximate"] == b"tanh"


@pytest.mark.parametrize(
    "name, op_type, default",
    [("elu", "Elu", 1.0), ("leaky_relu", "LeakyRelu", 0.01), ("celu", "Celu", 1.0)],
)
@pytest.mark.parametrize(
    "call_style", ["positional", "keyword", "input_keyword", "default"]
)
def test_activation_static_parameter(
    name: str, op_type: str, default: float, call_style: str
) -> None:
    model = _assert_ort_parity(
        lambda x: _activation(x, name=name, parameter=0.2, call_style=call_style)
    )
    expected_alpha = default if call_style == "default" else 0.2
    assert _attributes(model, op_type)["alpha"] == pytest.approx(expected_alpha)


@pytest.mark.parametrize("api", ["linen", "linen_activation"])
@pytest.mark.parametrize(
    "name, parameter, op_type, expected",
    [
        ("gelu", False, "Gelu", b"none"),
        ("elu", 0.2, "Elu", 0.2),
        ("leaky_relu", 0.2, "LeakyRelu", 0.2),
        ("celu", 0.2, "Celu", 0.2),
    ],
)
def test_linen_positional_activation_aliases(
    api: str, name: str, parameter: bool | float, op_type: str, expected: bytes | float
) -> None:
    model = _assert_ort_parity(
        lambda x: _activation(x, name=name, parameter=parameter, api=api)
    )
    attrs = _attributes(model, op_type)
    if name == "gelu":
        assert attrs["approximate"] == expected
    else:
        assert attrs["alpha"] == pytest.approx(expected)


@pytest.mark.parametrize("approximate", [False, True], ids=["exact", "tanh"])
def test_vmap_gelu_positional_argument(approximate: bool) -> None:
    _assert_ort_parity(jax.vmap(lambda x: jax.nn.gelu(x, approximate)))


@pytest.mark.parametrize("approximate", [False, True], ids=["exact", "tanh"])
def test_grad_gelu_positional_argument(approximate: bool) -> None:
    _assert_ort_parity(jax.grad(lambda x: jnp.sum(jax.nn.gelu(x, approximate))))


def test_keras_gelu_with_jax_backend() -> None:
    if importlib.util.find_spec("keras") is None:
        pytest.skip("Keras is an optional integration")
    # Select the backend in a fresh interpreter: earlier tests may have imported
    # Keras with a different backend, whose global state cannot be switched here.
    script = textwrap.dedent(
        """
        import jax.numpy as jnp
        import keras
        import numpy as np
        import onnx
        import onnxruntime as ort
        from jax2onnx import to_onnx

        assert keras.backend.backend() == "jax"
        x = jnp.asarray([-3.0, -0.7, -0.1, 0.0, 0.4, 2.0], dtype=jnp.float32)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        for approximate in (False, True):
            def gelu(value):
                return keras.activations.gelu(value, approximate=approximate)
            expected = np.asarray(gelu(x))
            model = to_onnx(gelu, [x], opset=20)
            onnx.checker.check_model(model, full_check=True)
            session = ort.InferenceSession(
                model.SerializeToString(), sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            (graph_input,) = session.get_inputs()
            (actual,) = session.run(None, {graph_input.name: np.asarray(x)})
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "KERAS_BACKEND": "jax", "JAX_PLATFORM_NAME": "cpu"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
