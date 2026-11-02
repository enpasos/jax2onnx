# tests/extra_tests/test_atan2_precision_contract.py

"""Atan2 retains FP64 work, source quadrants and signed zero results."""

import json

import jax
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx
from tests.extra_tests.test_precision_contract import check


def _full_fp64_witness():
    # Keep the complete original 4,265-case qualification input unchanged.
    rng = np.random.default_rng(541)
    corners = np.asarray(
        [
            -np.inf,
            -np.finfo(np.float64).max,
            -1.0,
            -np.finfo(np.float64).tiny,
            -np.nextafter(0.0, 1.0),
            -0.0,
            0.0,
            np.nextafter(0.0, 1.0),
            np.finfo(np.float64).tiny,
            1.0,
            np.finfo(np.float64).max,
            np.inf,
            np.nan,
        ]
    )
    yc, xc = np.meshgrid(corners, corners)
    x = np.concatenate(
        (
            xc.ravel(),
            np.ldexp(rng.uniform(-1.0, 1.0, 4096), rng.integers(-1020, 1021, 4096)),
        )
    )
    y = np.concatenate(
        (
            yc.ravel(),
            np.ldexp(rng.uniform(-1.0, 1.0, 4096), rng.integers(-1020, 1021, 4096)),
        )
    )
    return [y, x]


def test_atan2_full_fp64_witness_retains_source_precision(record_property):
    inputs = _full_fp64_witness()
    assert all(value.dtype == np.float64 and value.shape == (4265,) for value in inputs)
    # User-approved boundary, 2 November 2026: exact input pairs, independent
    # of observed outputs. Bits encode (y, x), retaining the subnormal signs.
    approved_pairs = {
        (0x0000000000000001, 0xFFEFFFFFFFFFFFFF),  # +s, -max
        (0x0000000000000001, 0xBFF0000000000000),  # +s, -1
        (0x0000000000000001, 0x0000000000000001),  # +s, +s
        (0x0000000000000001, 0x8000000000000001),  # +s, -s
        (0x8000000000000001, 0x0000000000000001),  # -s, +s
        (0x8000000000000001, 0x8000000000000001),  # -s, -s
    }
    input_pairs = [
        (int(y), int(x))
        for y, x in zip(inputs[0].view(np.uint64), inputs[1].view(np.uint64))
    ]
    approved_indices = {
        index for index, pair in enumerate(input_pairs) if pair in approved_pairs
    }
    assert len(approved_indices) == 6
    assert {input_pairs[index] for index in approved_indices} == approved_pairs

    y_bits, x_bits = (value.view(np.uint64) for value in inputs)
    magnitude_mask = np.uint64(0x7FFFFFFFFFFFFFFF)
    sign_mask = np.uint64(0x8000000000000000)
    positive_infinity = np.uint64(0x7FF0000000000000)
    # IEEE input bits distinguish literal zeros from subnormals even when
    # floating comparisons flush a subnormal to zero on the host CPU.
    positive_non_nan_x = (x_bits > 0) & (x_bits <= positive_infinity)
    analytic_zeros = (
        ((y_bits & magnitude_mask) == 0) & (positive_non_nan_x | (x_bits == 0))
    ) | (
        (x_bits == positive_infinity) & ((y_bits & magnitude_mask) < positive_infinity)
    )

    with jax.enable_x64():
        native_inputs = [jax.numpy.asarray(value) for value in inputs]
        reference = np.asarray(jax.lax.atan2(*native_inputs))
        assert reference.dtype == np.float64 and reference.shape == (4265,)
        compiled = jax.jit(lambda y, x: jax.lax.atan2(y, x))
        for candidate in (compiled(*native_inputs), compiled(*native_inputs)):
            candidate = np.asarray(candidate)
            assert candidate.dtype == np.float64 and candidate.shape == (4265,)
            np.testing.assert_array_equal(candidate, reference)
            zeros = reference == 0
            np.testing.assert_array_equal(
                np.signbit(candidate[zeros]), np.signbit(reference[zeros])
            )
        model = to_onnx(jax.lax.atan2, inputs)

    onnx.checker.check_model(model, full_check=True)
    assert len(model.graph.input) == 2 and len(model.graph.output) == 1
    assert not any(node.op_type == "Atan" for node in model.graph.node)
    floating = {
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
    }
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        element_type = value.type.tensor_type.elem_type
        if element_type in floating:
            assert element_type == onnx.TensorProto.DOUBLE
    for initializer in model.graph.initializer:
        if initializer.data_type in floating:
            assert initializer.data_type == onnx.TensorProto.DOUBLE
    for value in (*model.graph.input, *model.graph.output):
        assert tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim) == (
            4265,
        )

    observed_failures: set[int] = set()
    for label, optimization in (
        ("ort_off", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("ort_on", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = optimization
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        (actual,) = session.run(
            None,
            {meta.name: value for meta, value in zip(session.get_inputs(), inputs)},
        )
        assert actual.dtype == np.float64 and actual.shape == (4265,)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(reference))
        np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(reference))
        np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(reference))
        actual_bits = actual.view(np.uint64)
        reference_bits = reference.view(np.uint64)
        assert np.all((actual_bits[analytic_zeros] & magnitude_mask) == 0)
        np.testing.assert_array_equal(
            actual_bits[analytic_zeros] & sign_mask,
            y_bits[analytic_zeros] & sign_mask,
        )
        common_zeros = ((actual_bits & magnitude_mask) == 0) & (
            (reference_bits & magnitude_mask) == 0
        )
        np.testing.assert_array_equal(
            actual_bits[common_zeros], reference_bits[common_zeros]
        )
        failed_indices = np.flatnonzero(
            ~np.isclose(actual, reference, rtol=0, atol=4e-15, equal_nan=True)
        ).tolist()
        record_property(f"{label}_failed_indices", json.dumps(failed_indices))
        record_property(f"{label}_failed_count", len(failed_indices))
        assert (
            set(failed_indices) <= approved_indices
        ), f"Unapproved atan2 mismatch indices: {sorted(set(failed_indices) - approved_indices)}"
        record_property(
            f"{label}_numerical_status",
            "FAILED_IN_APPROVED_BOUNDARY" if failed_indices else "PASS",
        )
        observed_failures.update(failed_indices)
    record_property("failed_indices", json.dumps(sorted(observed_failures)))
    record_property("failed_count", len(observed_failures))
    record_property(
        "numerical_status",
        "FAILED_IN_APPROVED_BOUNDARY" if observed_failures else "PASS",
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_atan2_axes_keep_exact_angles_and_zero_signs(dtype):
    y = np.asarray([0.0, -0.0, 0.0, -0.0, 1.0, -1.0, 0.0, -0.0], dtype)
    x = np.asarray([0.0, 0.0, -0.0, -0.0, 0.0, 0.0, np.inf, np.inf], dtype)
    with jax.enable_x64():
        check(jax.lax.atan2, [y, x])


def test_atan2_tiny_nonzero_angles_are_not_hidden_by_absolute_tolerance():
    tiny = np.finfo(np.float64).tiny
    y = np.asarray([tiny, -tiny, tiny, -tiny])
    x = np.asarray([1.0, 1.0, np.finfo(np.float64).max, np.finfo(np.float64).max])
    with jax.enable_x64():
        check(jax.lax.atan2, [y, x])
