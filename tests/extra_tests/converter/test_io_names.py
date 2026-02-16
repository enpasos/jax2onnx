# tests/extra_tests/converter/test_io_names.py

from __future__ import annotations

import pytest
import jax.numpy as jnp

from jax2onnx.user_interface import to_onnx


def _add(x, y):
    return x + y


def _identity(x):
    return x


def _plus_one(x):
    return x + 1


def _identity_with_flag(x, deterministic: bool = True):
    return x if deterministic else x


def _constant_pair():
    return jnp.array([1.0], dtype=jnp.float32), jnp.array([2.0], dtype=jnp.float32)


def test_custom_io_names_proto_mode():
    model = to_onnx(
        _add,
        inputs=[(2, 3), (2, 3)],
        input_names=["lhs", "rhs"],
        output_names=["sum_out"],
    )

    assert [value.name for value in model.graph.input] == ["lhs", "rhs"]
    assert [value.name for value in model.graph.output] == ["sum_out"]


def test_custom_io_names_ir_mode():
    ir_model = to_onnx(
        _add,
        inputs=[(2, 3), (2, 3)],
        input_names=["lhs", "rhs"],
        output_names=["sum_out"],
        return_mode="ir",
    )

    assert [value.name for value in ir_model.graph.inputs] == ["lhs", "rhs"]
    assert [value.name for value in ir_model.graph.outputs] == ["sum_out"]


def test_custom_io_names_with_nchw_boundary():
    model = to_onnx(
        _plus_one,
        inputs=[(1, 8, 8, 3)],
        inputs_as_nchw=[0],
        outputs_as_nchw=[0],
        input_names=["image"],
        output_names=["image_nchw"],
    )

    assert [value.name for value in model.graph.input] == ["image"]
    assert [value.name for value in model.graph.output] == ["image_nchw"]


def test_custom_output_names_support_swapping_constant_initializers():
    baseline = to_onnx(_constant_pair, inputs=[], return_mode="ir")
    baseline_names = [value.name for value in baseline.graph.outputs]
    assert len(baseline_names) == 2
    assert all(value.is_initializer() for value in baseline.graph.outputs)

    swapped_names = [baseline_names[1], baseline_names[0]]
    ir_model = to_onnx(
        _constant_pair,
        inputs=[],
        output_names=swapped_names,
        return_mode="ir",
    )

    assert [value.name for value in ir_model.graph.outputs] == swapped_names
    assert sorted(ir_model.graph.initializers.keys()) == sorted(swapped_names)


def test_default_names_unchanged_when_no_custom_names():
    model = to_onnx(_identity, inputs=[(2,)])
    assert [value.name for value in model.graph.input] == ["in_0"]


def test_input_names_length_mismatch_raises():
    with pytest.raises(ValueError, match="input_names length"):
        to_onnx(_identity, inputs=[(2,)], input_names=["x", "y"])


def test_output_names_length_mismatch_raises():
    with pytest.raises(ValueError, match="output_names length"):
        to_onnx(_identity, inputs=[(2,)], output_names=["y0", "y1"])


def test_duplicate_custom_names_raises():
    with pytest.raises(ValueError, match="must be unique"):
        to_onnx(_add, inputs=[(2,), (2,)], input_names=["x", "x"])


def test_empty_custom_name_raises():
    with pytest.raises(ValueError, match="non-empty string"):
        to_onnx(_identity, inputs=[(2,)], output_names=[""])


def test_input_names_do_not_collide_with_input_params():
    with pytest.raises(ValueError, match="input_names collide"):
        to_onnx(
            _identity_with_flag,
            inputs=[(2,)],
            input_params={"deterministic": True},
            input_names=["deterministic"],
        )
