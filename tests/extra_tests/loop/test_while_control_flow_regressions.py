# tests/extra_tests/loop/test_while_control_flow_regressions.py

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx


def _symbolic_vector_spec(dtype: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(jax.export.symbolic_shape("B"), dtype)


def _make_session(model: onnx.ModelProto) -> ort.InferenceSession:
    onnx.checker.check_model(model, full_check=True)
    return ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def _run_with_timeout(
    session: ort.InferenceSession,
    feeds: dict[str, np.ndarray],
    *,
    timeout_seconds: float = 5.0,
) -> list[np.ndarray]:
    run_options = ort.RunOptions()
    timed_out = threading.Event()

    def terminate() -> None:
        timed_out.set()
        run_options.terminate = True

    timer = threading.Timer(timeout_seconds, terminate)
    timer.start()
    try:
        outputs = session.run(None, feeds, run_options=run_options)
    except Exception:
        if timed_out.is_set():
            pytest.fail(f"ONNX Runtime did not finish within {timeout_seconds} seconds")
        raise
    finally:
        timer.cancel()

    assert (
        not timed_out.is_set()
    ), f"ONNX Runtime did not finish within {timeout_seconds} seconds"
    return outputs


@pytest.mark.parametrize("opset", [17, 18, 21])
def test_vmapped_while_any_is_empty_batch_safe(opset: int) -> None:
    def one(value: jax.Array) -> jax.Array:
        return jax.lax.while_loop(
            lambda current: current < 3.0,
            lambda current: current + 1.0,
            value,
        )

    fn = jax.vmap(one)
    model = to_onnx(fn, [_symbolic_vector_spec(jnp.float32)], opset=opset)
    session = _make_session(model)
    input_name = session.get_inputs()[0].name

    for values in (
        np.empty((0,), dtype=np.float32),
        np.asarray([1.0, 3.5], dtype=np.float32),
    ):
        (actual,) = _run_with_timeout(session, {input_name: values})
        expected = np.asarray(fn(values))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    assert all(node.op_type != "ReduceMax" for node in model.graph.node)


@pytest.mark.parametrize("opset", [17, 18, 21])
def test_unvmap_any_is_empty_input_safe(opset: int) -> None:
    model = to_onnx(
        eqx.internal.unvmap_any,
        [_symbolic_vector_spec(jnp.bool_)],
        opset=opset,
    )
    session = _make_session(model)
    input_name = session.get_inputs()[0].name

    for values, expected in (
        (np.empty((0,), dtype=np.bool_), False),
        (np.asarray([False, True], dtype=np.bool_), True),
        (np.asarray([False, False], dtype=np.bool_), False),
    ):
        (actual,) = session.run(None, {input_name: values})
        np.testing.assert_array_equal(actual, np.asarray(expected))

    reduction = next(node for node in model.graph.node if node.op_type == "ReduceSum")
    assert len(reduction.input) == 2
    assert all(attribute.name != "axes" for attribute in reduction.attribute)


@pytest.mark.parametrize(
    ("fn", "values", "op_type"),
    [
        (
            eqx.internal.unvmap_all,
            np.asarray([[True, True], [True, False]], dtype=np.bool_),
            "ReduceMin",
        ),
        (
            eqx.internal.unvmap_max,
            np.asarray([[1, 7], [3, 2]], dtype=np.int32),
            "ReduceMax",
        ),
    ],
    ids=["all", "max"],
)
@pytest.mark.parametrize("opset", [17, 18])
def test_unvmap_reduction_axes_follow_opset_schema(
    fn: Callable[[jax.Array], jax.Array],
    values: np.ndarray,
    op_type: str,
    opset: int,
) -> None:
    model = to_onnx(fn, [values], opset=opset)
    session = _make_session(model)
    input_name = session.get_inputs()[0].name
    (actual,) = session.run(None, {input_name: values})
    np.testing.assert_array_equal(actual, np.asarray(fn(values)))

    reduction = next(node for node in model.graph.node if node.op_type == op_type)
    axes_attributes = [
        attribute for attribute in reduction.attribute if attribute.name == "axes"
    ]
    if opset < 18:
        assert len(reduction.input) == 1
        assert len(axes_attributes) == 1
    else:
        assert len(reduction.input) == 2
        assert not axes_attributes


@pytest.mark.parametrize("opset", [17, 21])
def test_while_condition_only_runtime_capture_is_loop_carried(opset: int) -> None:
    def fn(x: jax.Array, limit: jax.Array, step: jax.Array) -> jax.Array:
        return jax.lax.while_loop(
            lambda value: value < limit,
            lambda value: value + step,
            x,
        )

    input_values = [
        np.asarray(1.0, dtype=np.float32),
        np.asarray(5.0, dtype=np.float32),
        np.asarray(2.0, dtype=np.float32),
    ]
    model = to_onnx(fn, input_values, opset=opset)
    session = _make_session(model)
    input_names = [value.name for value in session.get_inputs()]

    for values in (
        input_values,
        [
            np.asarray(2.0, dtype=np.float32),
            np.asarray(8.0, dtype=np.float32),
            np.asarray(3.0, dtype=np.float32),
        ],
    ):
        feeds = dict(zip(input_names, values))
        (actual,) = session.run(None, feeds)
        expected = np.asarray(fn(*values))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    loop = next(node for node in model.graph.node if node.op_type == "Loop")
    body = next(attribute.g for attribute in loop.attribute if attribute.name == "body")
    assert len(body.input) == len(loop.input)
    assert len(body.output) == len(loop.output) + 1
