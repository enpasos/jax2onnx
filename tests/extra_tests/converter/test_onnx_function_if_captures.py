# tests/extra_tests/converter/test_onnx_function_if_captures.py

"""Function keyword captures preserve values across lexical If scopes."""

from __future__ import annotations

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx


@onnx_function
def _select_from_deterministic(
    x: jax.Array, *, deterministic: bool | jax.Array = True
) -> jax.Array:
    return jnp.where(deterministic, x, x * 2.0)


@onnx_function
def _select_from_gate(x: jax.Array, *, gate: jax.Array) -> jax.Array:
    return jnp.where(gate, x, x * 2.0)


def _flag_conditional(
    x: jax.Array,
    predicate: jax.Array,
    inner_predicate: jax.Array,
    inverted: jax.Array,
    *,
    renamed: bool,
    nested: bool,
) -> jax.Array:
    def captured_call(value: jax.Array) -> jax.Array:
        if renamed:
            return _select_from_gate(value, gate=inverted)
        return _select_from_deterministic(value, deterministic=inverted)

    def selected(value: jax.Array) -> jax.Array:
        if nested:
            return jax.lax.cond(
                inner_predicate, captured_call, lambda operand: operand - 3.0, value
            )
        return captured_call(value)

    return jax.lax.cond(predicate, selected, lambda value: value - 1.0, x)


@onnx_function
def _same_flag_one_if(
    x: jax.Array,
    predicate: jax.Array,
    inner_predicate: jax.Array,
    *,
    deterministic: bool | jax.Array = True,
) -> jax.Array:
    return _flag_conditional(
        x,
        predicate,
        inner_predicate,
        jnp.logical_not(deterministic),
        renamed=False,
        nested=False,
    )


@onnx_function
def _renamed_flag_one_if(
    x: jax.Array,
    predicate: jax.Array,
    inner_predicate: jax.Array,
    *,
    deterministic: bool | jax.Array = True,
) -> jax.Array:
    return _flag_conditional(
        x,
        predicate,
        inner_predicate,
        jnp.logical_not(deterministic),
        renamed=True,
        nested=False,
    )


@onnx_function
def _same_flag_two_ifs(
    x: jax.Array,
    predicate: jax.Array,
    inner_predicate: jax.Array,
    *,
    deterministic: bool | jax.Array = True,
) -> jax.Array:
    return _flag_conditional(
        x,
        predicate,
        inner_predicate,
        jnp.logical_not(deterministic),
        renamed=False,
        nested=True,
    )


@onnx_function
def _renamed_flag_two_ifs(
    x: jax.Array,
    predicate: jax.Array,
    inner_predicate: jax.Array,
    *,
    deterministic: bool | jax.Array = True,
) -> jax.Array:
    return _flag_conditional(
        x,
        predicate,
        inner_predicate,
        jnp.logical_not(deterministic),
        renamed=True,
        nested=True,
    )


def _call_same_flag_one_if(x, predicate, inner_predicate, *, deterministic=True):
    return _same_flag_one_if(x, predicate, inner_predicate, deterministic=deterministic)


def _call_renamed_flag_one_if(x, predicate, inner_predicate, *, deterministic=True):
    return _renamed_flag_one_if(
        x, predicate, inner_predicate, deterministic=deterministic
    )


def _call_same_flag_two_ifs(x, predicate, inner_predicate, *, deterministic=True):
    return _same_flag_two_ifs(
        x, predicate, inner_predicate, deterministic=deterministic
    )


def _call_renamed_flag_two_ifs(x, predicate, inner_predicate, *, deterministic=True):
    return _renamed_flag_two_ifs(
        x, predicate, inner_predicate, deterministic=deterministic
    )


def _checked_session(model: onnx.ModelProto) -> ort.InferenceSession:
    assert any(
        node.op_type == "If" for function in model.functions for node in function.node
    ), "The capture must cross an If inside an ONNX function."
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


@pytest.mark.parametrize(
    "block",
    [
        _call_same_flag_one_if,
        _call_renamed_flag_one_if,
        _call_same_flag_two_ifs,
        _call_renamed_flag_two_ifs,
    ],
    ids=[
        "one_if-same_keyword",
        "one_if-renamed",
        "two_ifs-same_keyword",
        "two_ifs-renamed",
    ],
)
def test_derived_flag_capture_keeps_variable_identity_across_if(block) -> None:
    def target(x, predicate, inner_predicate, *, deterministic=True):
        return block(x, predicate, inner_predicate, deterministic=deterministic)

    model = to_onnx(
        target,
        inputs=[
            jax.ShapeDtypeStruct((2,), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.bool_),
            jax.ShapeDtypeStruct((), jnp.bool_),
        ],
        input_names=["x", "predicate", "inner_predicate"],
        input_params={"deterministic": True},
    )
    session = _checked_session(model)
    input_names = {value.name for value in session.get_inputs()}
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    for deterministic, predicate, inner_predicate in product((True, False), repeat=3):
        feeds = {
            "x": x,
            "predicate": np.asarray(predicate, dtype=np.bool_),
            "inner_predicate": np.asarray(inner_predicate, dtype=np.bool_),
            "deterministic": np.asarray(deterministic, dtype=np.bool_),
        }
        (actual,) = session.run(
            None, {name: value for name, value in feeds.items() if name in input_names}
        )
        expected = np.asarray(
            target(x, predicate, inner_predicate, deterministic=deterministic)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@onnx_function
def _scale_from_tensor_capture(x: jax.Array, *, factor: jax.Array) -> jax.Array:
    return x * factor


def _tensor_conditional(
    x: jax.Array,
    gain: jax.Array,
    predicate: jax.Array,
    inner_predicate: jax.Array,
    *,
    nested: bool,
) -> jax.Array:
    factor = gain * 2.0 + 1.0

    def captured_call(value: jax.Array) -> jax.Array:
        return _scale_from_tensor_capture(value, factor=factor)

    def selected(value: jax.Array) -> jax.Array:
        if nested:
            return jax.lax.cond(
                inner_predicate, captured_call, lambda operand: operand - 3.0, value
            )
        return captured_call(value)

    return jax.lax.cond(predicate, selected, lambda value: value - 1.0, x)


@onnx_function
def _tensor_one_if(
    x: jax.Array, gain: jax.Array, predicate: jax.Array, inner_predicate: jax.Array
) -> jax.Array:
    return _tensor_conditional(x, gain, predicate, inner_predicate, nested=False)


@onnx_function
def _tensor_two_ifs(
    x: jax.Array, gain: jax.Array, predicate: jax.Array, inner_predicate: jax.Array
) -> jax.Array:
    return _tensor_conditional(x, gain, predicate, inner_predicate, nested=True)


def _call_tensor_one_if(x, gain, predicate, inner_predicate):
    return _tensor_one_if(x, gain, predicate, inner_predicate)


def _call_tensor_two_ifs(x, gain, predicate, inner_predicate):
    return _tensor_two_ifs(x, gain, predicate, inner_predicate)


@pytest.mark.parametrize(
    "target", [_call_tensor_one_if, _call_tensor_two_ifs], ids=["one_if", "two_ifs"]
)
def test_derived_tensor_keyword_capture_crosses_if_scopes(target) -> None:
    model = to_onnx(
        target,
        inputs=[
            jax.ShapeDtypeStruct((2,), jnp.float32),
            jax.ShapeDtypeStruct((2,), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.bool_),
            jax.ShapeDtypeStruct((), jnp.bool_),
        ],
        input_names=["x", "gain", "predicate", "inner_predicate"],
    )
    session = _checked_session(model)
    input_names = {value.name for value in session.get_inputs()}
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    for gain_values, predicate, inner_predicate in product(
        ((4.0, 5.0), (-2.0, 7.0)), (True, False), (True, False)
    ):
        gain = np.asarray(gain_values, dtype=np.float32)
        feeds = {
            "x": x,
            "gain": gain,
            "predicate": np.asarray(predicate, dtype=np.bool_),
            "inner_predicate": np.asarray(inner_predicate, dtype=np.bool_),
        }
        (actual,) = session.run(
            None, {name: value for name, value in feeds.items() if name in input_names}
        )
        expected = np.asarray(target(x, gain, predicate, inner_predicate))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
