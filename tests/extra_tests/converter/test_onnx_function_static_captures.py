# tests/extra_tests/converter/test_onnx_function_static_captures.py

"""Static configuration must select the matching specialized function body."""

from __future__ import annotations

import gc
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx.plugins import _function_keys
from jax2onnx.plugins._function_keys import snapshot_static_value, static_value_key


@onnx_function
def _ragged_configuration(x, *, axes):
    return x + axes[0][0]


@onnx_function(unique=True)
def _unique_ragged_configuration(x, *, axes):
    return x + axes[0][0]


def _call_ragged_configuration(x):
    return (
        _ragged_configuration(x, axes=((0,), (1, 2))),
        _ragged_configuration(x, axes=((1,), (0, 2))),
    )


def _call_unique_ragged_configuration(x):
    return (
        _unique_ragged_configuration(x, axes=((0,), (1, 2))),
        _unique_ragged_configuration(x, axes=((1,), (0, 2))),
    )


@onnx_function(unique=True)
def _dictionary_configuration(x, *, config):
    return x * config["scale"] + config["bias"]


def _call_dictionary_configuration(x):
    return (
        _dictionary_configuration(x, config={"scale": 2.0, "bias": 1.0}),
        _dictionary_configuration(x, config={"scale": 2.0, "bias": 1.0}),
        _dictionary_configuration(x, config={"scale": 2.0, "bias": 3.0}),
    )


def _call_mutated_dictionary_configuration(x):
    config = {"scale": 2.0, "bias": 1.0}
    first = _dictionary_configuration(x, config=config)
    config["bias"] = 3.0
    second = _dictionary_configuration(x, config=config)
    return first, second


@onnx_function(unique=True)
def _array_configuration(x, *, config):
    return x + config[0]


def _call_mutated_array_configuration(x):
    config = np.asarray([1.0, 2.0], dtype=np.float32)
    first = _array_configuration(x, config=config)
    config[0] = 3.0
    second = _array_configuration(x, config=config)
    return first, second


@onnx_function(unique=True)
def _ordered_dictionary_configuration(x, *, config):
    return x + next(iter(config.values()))


def _call_ordered_dictionary_configuration(x):
    return (
        _ordered_dictionary_configuration(x, config={"first": 1.0, "second": 2.0}),
        _ordered_dictionary_configuration(x, config={"second": 2.0, "first": 1.0}),
    )


@onnx_function(unique=True)
def _static_mode(x, *, mode):
    return x + 1.0 if mode == "left" else x * 2.0


def _call_static_mode(x):
    return _static_mode(x, mode="left"), _static_mode(x, mode="right")


@onnx_function(unique=True)
def _static_boolean(x, *, add):
    return x + 1.0 if add else x * 2.0


def _call_static_boolean(x):
    return _static_boolean(x, add=True), _static_boolean(x, add=False)


@onnx_function(unique=True)
def _static_axes(x, *, axes):
    return jnp.transpose(x, axes)


def _call_static_axes(x):
    return _static_axes(x, axes=(1, 0)), _static_axes(x, axes=(0, 1))


def _export_session(target, input_shape=(2,)):
    model = to_onnx(
        target,
        inputs=[jax.ShapeDtypeStruct(input_shape, jnp.float32)],
        input_names=["x"],
    )
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return model, session


@pytest.mark.parametrize("under_jit", [False, True], ids=["direct", "jit"])
@pytest.mark.parametrize(
    "target,transposes",
    [
        (_call_static_mode, False),
        (_call_static_boolean, False),
        (_call_static_axes, True),
    ],
    ids=["string-mode", "python-bool", "transpose-axes"],
)
def test_static_configuration_remains_static_during_abstract_eval(
    target, transposes, under_jit
):
    original_target = target
    if under_jit:

        def target(x):
            return jax.jit(original_target)(x)

    model, session = _export_session(target, input_shape=(2, 3))
    assert len(model.functions) == 2
    x = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    first, second = session.run(None, {"x": x})
    expected_first, expected_second = (x.T, x) if transposes else (x + 1.0, x * 2.0)
    np.testing.assert_array_equal(first, expected_first)
    np.testing.assert_array_equal(second, expected_second)


@pytest.mark.parametrize(
    "target", [_call_ragged_configuration, _call_unique_ragged_configuration]
)
def test_distinct_ragged_static_arguments_use_distinct_function_bodies(target):
    model, session = _export_session(target)
    assert len(model.functions) == 2
    for values in ([2.0, 3.0], [-4.0, 7.0]):
        x = np.asarray(values, dtype=np.float32)
        first, second = session.run(None, {"x": x})
        np.testing.assert_array_equal(first, x)
        np.testing.assert_array_equal(second, x + 1.0)


def test_equal_dictionary_configuration_reuses_only_the_matching_function():
    model, session = _export_session(_call_dictionary_configuration)
    assert len(model.functions) == 2
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    first, repeated, distinct = session.run(None, {"x": x})
    np.testing.assert_array_equal(first, x * 2.0 + 1.0)
    np.testing.assert_array_equal(repeated, first)
    np.testing.assert_array_equal(distinct, x * 2.0 + 3.0)


def test_dictionary_iteration_order_can_select_a_different_function_body():
    model, session = _export_session(_call_ordered_dictionary_configuration)
    assert len(model.functions) == 2
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    first, reordered = session.run(None, {"x": x})
    np.testing.assert_array_equal(first, x + 1.0)
    np.testing.assert_array_equal(reordered, x + 2.0)


@pytest.mark.parametrize(
    "target, scale",
    [
        (_call_mutated_dictionary_configuration, 2.0),
        (_call_mutated_array_configuration, 1.0),
    ],
)
def test_each_call_snapshots_its_mutable_static_configuration(target, scale):
    model, session = _export_session(target)
    assert len(model.functions) == 2
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    first, changed = session.run(None, {"x": x})
    np.testing.assert_array_equal(first, x * scale + 1.0)
    np.testing.assert_array_equal(changed, x * scale + 3.0)


def test_static_array_keys_snapshot_contents_and_preserve_scalar_types():
    values = np.asarray([0.0, 1.0], dtype=np.float32)
    original = static_value_key(values)
    assert original == static_value_key(values.copy())
    values[0] = -0.0
    assert original != static_value_key(values)
    assert static_value_key(1) != static_value_key(True)
    assert static_value_key(0.0) != static_value_key(-0.0)
    assert static_value_key(np.float32(1)) != static_value_key(1.0)


def test_array_digest_collisions_still_compare_exact_contents(monkeypatch):
    monkeypatch.setattr(_function_keys, "_array_digest", lambda value: b"collision")
    values = np.asarray([1.0, 2.0], dtype=np.float32)
    first = static_value_key(values)
    other = static_value_key(values + 1.0)
    assert hash(first) == hash(other)
    assert first != other
    assert first == static_value_key(values.copy())
    assert first != static_value_key(values.reshape(1, 2))
    assert first != static_value_key(values.view(np.int32))


def test_dtype_metadata_cannot_alias_a_different_static_configuration():
    first = np.dtype(np.int32, metadata={"offset": 1})
    second = np.dtype(np.int32, metadata={"offset": 2})
    assert first == second, "NumPy dtype equality deliberately ignores metadata."
    assert static_value_key(first) != static_value_key(second)
    assert static_value_key(np.ones(2, dtype=first)) != static_value_key(
        np.ones(2, dtype=second)
    )


def test_static_object_arrays_compare_contents_instead_of_object_addresses():
    first = np.empty((), dtype=object)
    same = np.empty((), dtype=object)
    different = np.empty((), dtype=object)
    first[()] = {"offset": (1, 2)}
    same[()] = {"offset": (1, 2)}
    different[()] = {"offset": (2, 1)}
    assert static_value_key(first) == static_value_key(same)
    assert static_value_key(first) != static_value_key(different)


class _OpaqueConfiguration:
    def __hash__(self):
        return 1

    def __repr__(self):
        return "identical representation"


def test_opaque_and_cyclic_configuration_cannot_reuse_a_specialized_body():
    value = _OpaqueConfiguration()
    reference = weakref.ref(value)
    first = static_value_key(value)
    assert first != static_value_key(value)
    assert first != static_value_key(_OpaqueConfiguration())
    del value
    gc.collect()
    assert reference() is not None, "The key must prevent object-address recycling."

    cycle: list[object] = []
    cycle.append(cycle)
    assert static_value_key(cycle) != static_value_key(cycle)
    with pytest.raises(ValueError, match="Cyclic static ONNX function"):
        snapshot_static_value(cycle)
