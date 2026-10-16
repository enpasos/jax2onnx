# tests/extra_tests/loop/test_fori_loop_captures.py

"""fori_loop closure values remain explicit dependencies during export."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx


def _checked_session(model: onnx.ModelProto) -> ort.InferenceSession:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


@pytest.mark.parametrize("iterations", [0, 3])
@pytest.mark.parametrize("compiled", [False, True], ids=["plain", "jit"])
def test_fori_loop_tracks_computed_closure_operands(iterations, compiled) -> None:
    def target(x, flag, scale):
        increment = scale + 1.0

        def body(index, value):
            delta = increment * index
            return jnp.where(flag, value + delta, value - delta)

        return jax.lax.fori_loop(2, 2 + iterations, body, x)

    exported = jax.jit(target) if compiled else target
    model = to_onnx(
        exported,
        inputs=[
            jax.ShapeDtypeStruct((2,), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.bool_),
            jax.ShapeDtypeStruct((2,), jnp.float32),
        ],
        input_names=["x", "flag", "scale"],
    )
    session = _checked_session(model)
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    for flag in (True, False):
        for scale in ([1.0, 2.0], [-2.0, 3.0]):
            feeds = {
                "x": x,
                "flag": np.asarray(flag),
                "scale": np.asarray(scale, dtype=np.float32),
            }
            expected = np.asarray(target(x, feeds["flag"], feeds["scale"]))
            actual = session.run(None, feeds)[0]
            np.testing.assert_allclose(actual, expected)


def test_nested_fori_loops_track_outer_iteration_and_flag() -> None:
    def target(x, flag):
        def outer(index, value):
            def inner(inner_index, inner_value):
                increment = index + inner_index + 1
                return jnp.where(flag, inner_value + increment, inner_value - increment)

            return jax.lax.fori_loop(0, 2, inner, value)

        return jax.lax.fori_loop(1, 3, outer, x)

    model = to_onnx(
        target,
        inputs=[
            jax.ShapeDtypeStruct((2,), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        ],
        input_names=["x", "flag"],
    )
    session = _checked_session(model)
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    for flag in (True, False):
        feeds = {"x": x, "flag": np.asarray(flag)}
        actual = session.run(None, feeds)[0]
        expected = np.asarray(target(x, feeds["flag"]))
        np.testing.assert_allclose(actual, expected)


@onnx_function
def _select(value, *, deterministic=True):
    return jnp.where(deterministic, value, value * 2.0)


@pytest.mark.parametrize("iterations", [0, 2])
def test_fori_loop_threads_external_flags_as_explicit_state(iterations) -> None:
    def target(x, *, deterministic=True):
        return jax.lax.fori_loop(
            0,
            iterations,
            lambda index, value: _select(value, deterministic=deterministic),
            x,
        )

    model = to_onnx(
        target,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        input_names=["x"],
        input_params={"deterministic": True},
    )
    session = _checked_session(model)
    loop = next(node for node in model.graph.node if node.op_type == "Loop")
    body = next(attribute.g for attribute in loop.attribute if attribute.name == "body")
    assert len(body.input) == len(loop.input)
    assert len(body.output) == len(loop.output) + 1
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    for flag in (True, False):
        feeds = {"x": x, "deterministic": np.asarray(flag)}
        feeds = {value.name: feeds[value.name] for value in session.get_inputs()}
        actual = session.run(None, feeds)[0]
        expected = np.asarray(target(x, deterministic=flag))
        np.testing.assert_allclose(actual, expected)
