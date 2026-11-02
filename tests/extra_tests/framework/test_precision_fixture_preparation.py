# tests/extra_tests/framework/test_precision_fixture_preparation.py

from __future__ import annotations

from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx.plugins.plugin_system import PLUGIN_REGISTRY, with_requested_dtype
from tests.t_generator import generate_test_params, make_test_function


def _fixed_fp32_cancellation(x):
    rounded = x.astype(jnp.float32)
    return (rounded + jnp.float32(1)) - rounded


def test_authored_output_dtype_marker_resolves_without_widening_fixed_expectations():
    marker = with_requested_dtype()
    expected = [marker, np.float32]
    supplied = np.asarray([1.0], dtype=np.float32)
    metadata = {
        "testcase": "authored_output_dtypes",
        "callable": lambda x: (x, x.astype(jnp.float32)),
        "input_values": [supplied],
        "expected_output_dtypes": expected,
    }

    f32, f64 = generate_test_params(metadata)

    assert [np.dtype(dtype) for dtype in f32["expected_output_dtypes"]] == [
        np.dtype(np.float32),
        np.dtype(np.float32),
    ]
    assert [np.dtype(dtype) for dtype in f64["expected_output_dtypes"]] == [
        np.dtype(np.float64),
        np.dtype(np.float32),
    ]
    assert f32["expected_output_dtypes"][1] is np.float32
    assert f64["expected_output_dtypes"][1] is np.float32
    assert metadata["expected_output_dtypes"] is expected
    assert expected[0] is marker and expected[1] is np.float32
    assert metadata["input_values"][0] is supplied
    assert supplied.dtype == np.float32
    with jax.enable_x64():
        source = jax.eval_shape(f64["callable"], *f64["input_values"])
    assert [value.dtype for value in source] == [np.float64, np.float32]


def test_supplied_fp32_fixture_gets_real_fp64_input_without_rewriting_output():
    supplied = np.asarray([2**24], dtype=np.float32)
    expected = [np.float32]
    metadata = {
        "testcase": "fixed_fp32_cancellation",
        "callable": _fixed_fp32_cancellation,
        "input_values": [supplied],
        "expected_output_dtypes": expected,
    }

    f32, f64 = generate_test_params(metadata)

    assert f32["input_values"][0] is supplied
    assert f64["input_values"][0].dtype == np.float64
    np.testing.assert_array_equal(f64["input_values"][0], supplied)
    assert supplied.dtype == np.float32
    assert metadata["input_values"][0] is supplied
    assert f32["expected_output_dtypes"] is expected
    assert f64["expected_output_dtypes"] is expected
    with jax.enable_x64():
        source = jax.eval_shape(f64["callable"], f64["input_values"][0])
        result = f64["callable"](jnp.asarray(f64["input_values"][0]))
    assert source.dtype == np.float32
    np.testing.assert_array_equal(np.asarray(result), np.zeros((1,), dtype=np.float32))


def test_shape_variants_prepare_explicit_dtypes_for_the_complete_input_list():
    _, shape_f64 = generate_test_params(
        {
            "testcase": "shape_inputs",
            "callable": lambda x, y: x + y,
            "input_shapes": [(7, 30), (7, 30)],
        }
    )
    assert shape_f64["input_shapes"] == [(7, 30), (7, 30)]
    assert shape_f64["input_dtypes"] == [np.float64, np.float64]

    _, typed_f64 = generate_test_params(
        {
            "testcase": "typed_inputs",
            "callable": lambda x, i: x[i],
            "input_shapes": [(7, 30), (2,)],
            "input_dtypes": [np.float32, np.int32],
        }
    )
    assert typed_f64["input_dtypes"] == [np.float64, np.int32]


@pytest.mark.parametrize("fixed_dtype", [np.float16, jnp.bfloat16, np.float64])
def test_mixed_and_half_input_fixtures_keep_their_declared_types(fixed_dtype):
    values = [
        np.asarray([1], dtype=np.float32),
        np.asarray([1], dtype=fixed_dtype),
    ]
    _, f64 = generate_test_params(
        {
            "testcase": "mixed_inputs",
            "callable": lambda x, y: (x, y),
            "input_shapes": [(1,), (1,)],
            "input_dtypes": [np.float32, fixed_dtype],
            "input_values": values,
        }
    )
    assert f64["input_dtypes"] == [np.float32, fixed_dtype]
    assert all(
        actual is original for actual, original in zip(f64["input_values"], values)
    )


def test_fixture_preparation_preserves_weak_scalars_and_exact_integer_values():
    exact_integer = np.asarray([2**53 + 1], dtype=np.int64)
    _, f64 = generate_test_params(
        {
            "testcase": "weak_scalar_and_integer",
            "callable": lambda x, y, i: (x + y, i),
            "input_values": [np.asarray([1], dtype=np.float32), 1.0, exact_integer],
        }
    )
    assert f64["input_values"][0].dtype == np.float64
    assert type(f64["input_values"][1]) is float
    assert f64["input_values"][2] is exact_integer
    assert exact_integer[0] == 2**53 + 1


def test_weak_jax_array_keeps_its_dtype_and_weakness():
    with jax.enable_x64(False):
        supplied = jnp.asarray(1.0)
    assert supplied.weak_type
    _, f64 = generate_test_params(
        {
            "testcase": "weak_array",
            "callable": lambda x: x,
            "input_values": [supplied],
        }
    )
    assert f64["input_values"][0] is supplied
    assert supplied.dtype == np.float32
    assert supplied.weak_type


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_explicit_complex_input_precision_is_retained(dtype):
    supplied = np.asarray([1 + 2j], dtype=dtype)
    _, f64 = generate_test_params(
        {
            "testcase": "complex_input",
            "callable": lambda x: x,
            "input_values": [supplied],
        }
    )
    assert f64["input_values"][0] is supplied
    assert f64["input_values"][0].dtype == dtype


def test_generated_fp64_input_preserves_fixed_source_rounding(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, f64 = generate_test_params(
        {
            "testcase": "fixed_fp32_cancellation",
            "callable": _fixed_fp32_cancellation,
            "input_values": [np.asarray([2**24], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
            "expected_output_shapes": [(1,)],
        }
    )
    make_test_function(f64)()


@pytest.mark.parametrize(
    "component,primitive,testcase",
    [
        ("fori_loop", "lax.fori_loop", "fori_loop_example"),
        ("while_loop", "while", "while_loop_4d_and_scalar_state"),
        ("while_loop", "while", "while_loop_cnn_scalar_state_bug"),
        ("iota", "iota", "iota_float32"),
        ("iota", "iota", "iota_float64"),
    ],
)
def test_loop_and_iota_metadata_prepare_the_authored_precision(
    component, primitive, testcase
):
    import_module(f"jax2onnx.plugins.jax.lax.{component}")
    entry = next(
        case
        for case in PLUGIN_REGISTRY[primitive].metadata["testcases"]
        if case["testcase"] == testcase
    )

    for params in generate_test_params(entry):
        use_x64 = params["_jax_x64_test_setting"]
        with jax.enable_x64(use_x64):
            callable_obj = params["callable"]
            if hasattr(callable_obj, "instantiate"):
                callable_obj = callable_obj.instantiate()
            outputs = jax.tree.leaves(
                jax.eval_shape(callable_obj, *params.get("input_values", []))
            )

        expected_float = (
            np.float64 if use_x64 and testcase != "iota_float32" else np.float32
        )
        expected_dtypes = [np.dtype(expected_float)]
        if len(outputs) == 2:
            expected_dtypes.append(np.dtype(np.int32))
        assert [output.dtype for output in outputs] == expected_dtypes
        assert [np.dtype(dtype) for dtype in params["expected_output_dtypes"]] == (
            expected_dtypes
        )
        if "expected_output_shapes" in params:
            assert [output.shape for output in outputs] == params[
                "expected_output_shapes"
            ]
