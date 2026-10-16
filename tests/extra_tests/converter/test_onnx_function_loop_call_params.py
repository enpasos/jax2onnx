# tests/extra_tests/converter/test_onnx_function_loop_call_params.py

"""Named model parameters enter Loop bodies through explicit carried state."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx


@onnx_function
def _select_loop_flag(x, *, deterministic=True):
    return jnp.where(deterministic, x, x * 2.0)


@onnx_function
def _increment_without_flag(x):
    return x + 1.0


def _scan_flag(x, *, deterministic=True):
    def body(carry, unused):
        result = _select_loop_flag(carry, deterministic=deterministic)
        return result, result

    return jax.lax.scan(body, x, None, length=2)


def _scan_flag_with_inputs(x, xs, *, deterministic=True):
    def body(carry, item):
        result = _select_loop_flag(carry + item, deterministic=deterministic)
        return result, result

    return jax.lax.scan(body, x, xs)


def _scan_flag_in_if(x, predicate, *, deterministic=True):
    def body(carry, unused):
        result = jax.lax.cond(
            predicate,
            lambda value: _select_loop_flag(value, deterministic=deterministic),
            lambda value: value - 1.0,
            carry,
        )
        return result, result

    return jax.lax.scan(body, x, None, length=2)


def _while_flag(x, limit, *, deterministic=True):
    def body(state):
        index, value = state
        return index + 1, _select_loop_flag(value, deterministic=deterministic)

    return jax.lax.while_loop(lambda state: state[0] < limit, body, (jnp.int32(0), x))[
        1
    ]


def _nested_loop_flag(x, *, deterministic=True):
    def body(carry, unused):
        result = _while_flag(carry, jnp.int32(2), deterministic=deterministic)
        return result, result

    return jax.lax.scan(body, x, None, length=2)


def _unused_scan_flag(x, *, deterministic=True):
    return jax.lax.scan(
        lambda carry, unused: (_increment_without_flag(carry), None),
        x,
        None,
        length=2,
    )[0]


def _unused_while_flag(x, *, deterministic=True):
    return jax.lax.while_loop(
        lambda state: state[0] < 2,
        lambda state: (state[0] + 1, _increment_without_flag(state[1])),
        (jnp.int32(0), x),
    )[1]


def _graphs(graph: onnx.GraphProto):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _graphs(attribute.g)


def _assert_explicit_loop_interfaces(model: onnx.ModelProto) -> None:
    for graph in _graphs(model.graph):
        for node in graph.node:
            if node.op_type != "Loop":
                continue
            body = next(
                attribute.g for attribute in node.attribute if attribute.name == "body"
            )
            assert len(body.input) == len(node.input)
            assert len(body.output) == len(node.output) + 1
            available = {value.name for value in body.input}
            available.update(value.name for value in body.initializer)
            for body_node in body.node:
                for value in body_node.input:
                    assert not value or value in available
                available.update(body_node.output)


def _session(model: onnx.ModelProto) -> ort.InferenceSession:
    onnx.checker.check_model(model, full_check=True)
    _assert_explicit_loop_interfaces(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


_X = np.asarray([2.0, 3.0], dtype=np.float32)


@pytest.mark.parametrize(
    "target, inputs",
    [
        (_scan_flag, {"x": _X}),
        (_scan_flag_with_inputs, {"x": _X, "xs": np.ones((2, 2), np.float32)}),
        (_scan_flag_in_if, {"x": _X, "predicate": np.asarray(True)}),
        (_scan_flag_in_if, {"x": _X, "predicate": np.asarray(False)}),
        (_while_flag, {"x": _X, "limit": np.asarray(2, np.int32)}),
        (_while_flag, {"x": _X, "limit": np.asarray(0, np.int32)}),
        (_nested_loop_flag, {"x": _X}),
    ],
    ids=[
        "scan",
        "scan-xs",
        "scan-if-true",
        "scan-if-false",
        "while",
        "while-zero",
        "nested-loops",
    ],
)
def test_loop_functions_preserve_named_runtime_flags(
    target: Callable, inputs: dict[str, np.ndarray]
) -> None:
    model = to_onnx(
        target,
        [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in inputs.values()],
        input_names=list(inputs),
        input_params={"deterministic": True},
    )
    assert model.functions
    assert "deterministic" in {value.name for value in model.graph.input}
    session = _session(model)
    for flag in (True, False):
        actual = session.run(
            None, {**inputs, "deterministic": np.asarray(flag, dtype=np.bool_)}
        )
        expected = jax.tree.leaves(target(*inputs.values(), deterministic=flag))
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected):
            np.testing.assert_allclose(result, reference)


@pytest.mark.parametrize("target", [_unused_scan_flag, _unused_while_flag])
def test_unused_model_flag_does_not_add_loop_state(target: Callable) -> None:
    model = to_onnx(
        target,
        [jax.ShapeDtypeStruct(_X.shape, _X.dtype)],
        input_names=["x"],
        input_params={"deterministic": True},
    )
    assert {value.name for value in model.graph.input} == {"x"}
    session = _session(model)
    np.testing.assert_allclose(session.run(None, {"x": _X})[0], _X + 2.0)
