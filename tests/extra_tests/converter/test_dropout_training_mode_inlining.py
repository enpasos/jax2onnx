# tests/extra_tests/converter/test_dropout_training_mode_inlining.py

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from flax import linen as nn
from flax import nnx

from jax2onnx import onnx_function, to_onnx

ExportCase = tuple[Callable[[jax.Array], jax.Array], tuple[int, ...]]


def _unresolved_inputs(model: onnx.ModelProto) -> list[tuple[str, str, str]]:
    """Node inputs that no graph input, initializer or earlier node provides."""
    missing: list[tuple[str, str, str]] = []
    known = {value.name for value in model.graph.input}
    known |= {tensor.name for tensor in model.graph.initializer}
    for node in model.graph.node:
        missing.extend(
            ("graph", node.op_type, name)
            for name in node.input
            if name and name not in known
        )
        known.update(node.output)
    for function in model.functions:
        known = set(function.input)
        for node in function.node:
            missing.extend(
                (function.name, node.op_type, name)
                for name in node.input
                if name and name not in known
            )
            known.update(node.output)
    return missing


def _unused_initializers(model: onnx.ModelProto) -> list[str]:
    used = {name for node in model.graph.node for name in node.input}
    used |= {value.name for value in model.graph.output}
    return [
        tensor.name for tensor in model.graph.initializer if tensor.name not in used
    ]


def _dropout_nodes(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    nodes = list(model.graph.node)
    for function in model.functions:
        nodes.extend(function.node)
    return [node for node in nodes if node.op_type == "Dropout"]


def _export_and_run(
    fn: Callable[[jax.Array], jax.Array], x: np.ndarray
) -> tuple[onnx.ModelProto, np.ndarray]:
    model = to_onnx(fn, [jax.ShapeDtypeStruct(x.shape, jnp.float32)])
    assert _unresolved_inputs(model) == []
    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    assert [value.name for value in session.get_inputs()] == [model.graph.input[0].name]
    (actual,) = session.run(None, {model.graph.input[0].name: x})
    return model, np.asarray(actual)


def _eqx_call_inference() -> ExportCase:
    dropout = eqx.nn.Dropout(0.5)
    return (lambda x: dropout(x, inference=True)), (2, 8)


def _eqx_call_inference_array() -> ExportCase:
    dropout = eqx.nn.Dropout(0.5)
    return (lambda x: dropout(x, inference=jnp.array(True))), (2, 8)


class _EqxLinearDropout(eqx.Module):
    linear: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, key: jax.Array) -> None:
        self.linear = eqx.nn.Linear(8, 6, key=key)
        self.dropout = eqx.nn.Dropout(0.5)

    def __call__(self, x: jax.Array, *, inference: bool = False) -> jax.Array:
        return self.dropout(self.linear(x), inference=inference)


def _eqx_linear_then_dropout() -> ExportCase:
    model = _EqxLinearDropout(jax.random.PRNGKey(0))
    return (lambda x: model(x, inference=True)), (8,)


@onnx_function
class _EqxInferenceDropoutBlock(eqx.Module):
    dropout: eqx.nn.Dropout

    def __init__(self) -> None:
        self.dropout = eqx.nn.Dropout(0.5)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(x, inference=True)


def _eqx_onnx_function() -> ExportCase:
    block = _EqxInferenceDropoutBlock()
    return (lambda x: block(x)), (2, 8)


def _nnx_call_deterministic() -> ExportCase:
    dropout = nnx.Dropout(0.5, rngs=nnx.Rngs(0))
    return (lambda x: dropout(x, deterministic=True)), (2, 8)


class _LinenDropout(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dropout(rate=0.5)(x, deterministic=True)


def _linen_call_deterministic() -> ExportCase:
    module = _LinenDropout()
    variables = module.init(jax.random.PRNGKey(0), jnp.ones((2, 8), jnp.float32))
    return (lambda x: module.apply(variables, x)), (2, 8)


@pytest.mark.parametrize(
    "build",
    [
        _eqx_call_inference,
        _eqx_call_inference_array,
        _eqx_linear_then_dropout,
        _eqx_onnx_function,
        _nnx_call_deterministic,
        _linen_call_deterministic,
    ],
    ids=[
        "eqx_inference_true",
        "eqx_inference_array_true",
        "eqx_linear_then_dropout",
        "eqx_onnx_function",
        "nnx_deterministic_true",
        "linen_deterministic_true",
    ],
)
def test_call_time_inference_dropout_exports_runnable_model(
    build: Callable[[], ExportCase],
) -> None:
    fn, shape = build()
    x = np.linspace(-2.0, 2.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)

    model, actual = _export_and_run(fn, x)

    assert _unused_initializers(model) == []
    np.testing.assert_allclose(actual, np.asarray(fn(jnp.asarray(x))), rtol=1e-6)


@onnx_function
class _EqxTrainingDropoutBlock(eqx.Module):
    dropout: eqx.nn.Dropout

    def __init__(self) -> None:
        self.dropout = eqx.nn.Dropout(0.5)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(x, inference=False, key=jax.random.PRNGKey(0))


@pytest.mark.parametrize("scope", ["graph", "onnx_function"])
def test_call_time_training_dropout_keeps_training_mode(scope: str) -> None:
    if scope == "graph":
        dropout = eqx.nn.Dropout(0.5)

        def fn(x: jax.Array) -> jax.Array:
            return dropout(x, inference=False, key=jax.random.PRNGKey(0))

    else:
        block = _EqxTrainingDropoutBlock()

        def fn(x: jax.Array) -> jax.Array:
            return block(x)

    x = np.linspace(1.0, 2.0, 64, dtype=np.float32)

    model, actual = _export_and_run(fn, x)

    (dropout_node,) = _dropout_nodes(model)
    assert len(dropout_node.input) == 3 and dropout_node.input[2]
    # Training-mode Dropout zeroes elements or scales them by 1 / (1 - 0.5).
    assert np.all(np.isclose(actual, 0.0) | np.isclose(actual, 2.0 * x))
    assert not np.allclose(actual, x)
