# tests/extra_tests/test_framework_precision_contract.py

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

from flax import linen as nn
from flax import nnx
import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from tests.extra_tests.test_precision_contract import check


def _linear(kind: str, *, dtype: Any, param_dtype: Any = np.float32, **kwargs: Any):
    common = {
        "dtype": dtype,
        "param_dtype": param_dtype,
        "kernel_init": nnx.initializers.ones,
        "bias_init": nnx.initializers.ones,
        "rngs": nnx.Rngs(0),
        **kwargs,
    }
    if kind == "linear":
        return nnx.Linear(1, 1, **common)
    return nnx.LinearGeneral((1, 1), (1,), axis=(-2, -1), **common)


def _linear_input(kind: str, value: float, dtype: Any) -> np.ndarray:
    shape = (1, 1) if kind == "linear" else (1, 1, 1)
    return np.full(shape, value, dtype=dtype)


def _batch_norm(
    framework: str,
    *,
    dtype: Any,
    param_dtype: Any = np.float32,
    variance: float = 1.0,
    scale: float = 1.0,
    bias: float = 1.0,
    force_float32_reductions: bool = True,
    **kwargs: Any,
) -> Callable[[jax.Array], jax.Array]:
    if framework == "nnx":
        module = nnx.BatchNorm(
            1,
            dtype=dtype,
            param_dtype=param_dtype,
            use_running_average=True,
            epsilon=0.0,
            rngs=nnx.Rngs(0),
            **kwargs,
        )
        module.mean[...] = jnp.zeros((1,), dtype=jnp.float32)
        module.var[...] = jnp.full((1,), variance, dtype=jnp.float32)
        assert module.scale is not None and module.bias is not None
        module.scale[...] = jnp.full((1,), scale, dtype=param_dtype)
        module.bias[...] = jnp.full((1,), bias, dtype=param_dtype)
        return module

    linen_module = nn.BatchNorm(
        dtype=dtype,
        param_dtype=param_dtype,
        use_running_average=True,
        epsilon=0.0,
        force_float32_reductions=force_float32_reductions,
    )
    variables = {
        "params": {
            "scale": jnp.full((1,), scale, dtype=param_dtype),
            "bias": jnp.full((1,), bias, dtype=param_dtype),
        },
        "batch_stats": {
            "mean": jnp.zeros((1,), dtype=jnp.float32),
            "var": jnp.full((1,), variance, dtype=jnp.float32),
        },
    }

    def apply(x: jax.Array) -> jax.Array:
        return linen_module.apply(variables, x)

    return apply


@pytest.mark.parametrize("kind", ["linear", "linear_general"])
@pytest.mark.parametrize("compute_dtype", [np.float32, np.float64])
def test_nnx_linear_uses_configured_computation_dtype(kind, compute_dtype):
    with jax.enable_x64():
        module = _linear(kind, dtype=compute_dtype)
        _, reference = check(module, [_linear_input(kind, 2**24, np.float64)])
        assert reference.dtype == np.dtype(compute_dtype)
        assert reference.item() == (2**24 if compute_dtype is np.float32 else 2**24 + 1)
        assert module.kernel[...].dtype == np.float32


@pytest.mark.parametrize("kind", ["linear", "linear_general"])
@pytest.mark.parametrize(
    ("compute_dtype", "expected_dtype", "expected"),
    [(None, np.float32, 2049), (np.float16, np.float16, 2048)],
    ids=["infer-from-fp32-parameters", "configured-fp16"],
)
def test_nnx_linear_half_input_respects_fp32_parameters(
    kind, compute_dtype, expected_dtype, expected
):
    module = _linear(kind, dtype=compute_dtype)
    _, reference = check(module, [_linear_input(kind, 2048, np.float16)])
    assert reference.dtype == np.dtype(expected_dtype)
    assert reference.item() == expected
    assert module.kernel[...].dtype == np.float32


@pytest.mark.parametrize("kind", ["linear", "linear_general", "batch_norm"])
def test_nnx_uses_current_promotion_hook_once_per_call(kind):
    calls = []

    def promote(args, *, dtype=None):
        del dtype
        calls.append("promote")
        return tuple(
            jnp.asarray(value, jnp.float64) if value is not None else None
            for value in args
        )

    with jax.enable_x64():
        module = (
            _batch_norm("nnx", dtype=np.float32, promote_dtype=promote)
            if kind == "batch_norm"
            else _linear(kind, dtype=np.float32, promote_dtype=promote)
        )

        def fn(x):
            before = len(calls)
            result = module(x)
            assert len(calls) == before + 1
            return result

        input_kind = "linear" if kind == "batch_norm" else kind
        model, reference = check(fn, [_linear_input(input_kind, 2**24, np.float32)])
        if kind == "batch_norm":
            # BatchNorm's original normalization computes after the hook, then
            # applies self.dtype at its final return, preserving this FP32 result.
            assert reference.dtype == np.float32
            assert reference.item() == 2**24
            types = {
                value.name: value.type.tensor_type.elem_type
                for value in model.graph.value_info
            }
            work = [
                node
                for node in model.graph.node
                if node.op_type in {"Sub", "Sqrt", "Div", "Mul", "Add"}
            ]
            assert work
            assert all(
                types[node.output[0]] == onnx.TensorProto.DOUBLE for node in work
            )
        else:
            assert reference.dtype == np.float64
            assert reference.item() == 2**24 + 1


@pytest.mark.parametrize("kind", ["linear", "linear_general"])
def test_nnx_custom_contraction_keeps_promotion_and_call_order(kind):
    calls = []

    def promote(args, *, dtype=None):
        calls.append("promote")
        return tuple(
            jnp.asarray(value, dtype) if value is not None else None for value in args
        )

    def dot_general(lhs, rhs, dimension_numbers, *, precision=None, **kwargs):
        del kwargs
        calls.append("dot")
        return jax.lax.dot_general(lhs, rhs, dimension_numbers, precision=precision) + 1

    module = _linear(
        kind, dtype=np.float32, promote_dtype=promote, dot_general=dot_general
    )

    def fn(x):
        before = len(calls)
        result = module(x)
        assert calls[before:] == ["promote", "dot"]
        return result

    _, reference = check(fn, [_linear_input(kind, 2, np.float32)])
    assert reference.dtype == np.float32
    assert reference.item() == 4


@pytest.mark.parametrize("kind", ["linear", "linear_general"])
def test_nnx_parameter_conversion_preserves_shared_kernel_payload(kind):
    with jax.enable_x64():
        module = _linear(kind, dtype=np.float32, param_dtype=np.float64)
        original = jnp.full(module.kernel[...].shape, 2**24 + 1, dtype=jnp.float64)
        module.kernel[...] = original
        assert module.bias is not None
        module.bias[...] = jnp.zeros(module.bias[...].shape, dtype=jnp.float64)

        def fn(x):
            return module(x), module.kernel[...]

        model, reference = check(fn, [_linear_input(kind, 1, np.float32)])
        result, shared = reference
        assert result.dtype == np.float32 and result.item() == 2**24
        assert shared.dtype == np.float64 and shared.item() == 2**24 + 1
        assert module.kernel[...].dtype == np.float64
        np.testing.assert_array_equal(module.kernel[...], original)
        assert [value.type.tensor_type.elem_type for value in model.graph.output] == [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
        ]


@pytest.mark.parametrize("framework", ["nnx", "linen"])
@pytest.mark.parametrize(
    ("input_dtype", "compute_dtype", "value", "nnx_expected", "linen_expected"),
    [
        (np.float32, np.float64, 2**24, 2**24 + 1, 2**24),
        (np.float64, np.float32, 2**24 + 1, 2**24, 2**24 + 2),
    ],
    ids=["fp32-input-fp64-result", "fp64-input-fp32-result"],
)
def test_batch_norm_retains_framework_dtype_staging(
    framework, input_dtype, compute_dtype, value, nnx_expected, linen_expected
):
    with jax.enable_x64():
        module = _batch_norm(framework, dtype=compute_dtype)
        model, reference = check(module, [np.full((1, 1), value, dtype=input_dtype)])
        assert reference.dtype == np.dtype(compute_dtype)
        assert reference.item() == (
            nnx_expected if framework == "nnx" else linen_expected
        )
        fused = [
            node for node in model.graph.node if node.op_type == "BatchNormalization"
        ]
        assert len(fused) == int(framework == "nnx" and compute_dtype is np.float32)


@pytest.mark.parametrize(
    ("framework", "force_float32_reductions"),
    [("nnx", False), ("linen", False), ("linen", True)],
    ids=["nnx-half-coefficient", "linen-half-coefficient", "linen-fp32-coefficient"],
)
def test_half_batch_norm_retains_source_coefficient_rounding(
    framework, force_float32_reductions
):
    module = _batch_norm(
        framework,
        dtype=np.float16,
        param_dtype=np.float16,
        variance=3.0,
        scale=0.7,
        bias=0.0,
        force_float32_reductions=force_float32_reductions,
    )
    # Exact comparison requires eager, fresh/warmed JIT and both ORT settings to
    # agree. The source references determine the result; no guessed half value.
    model, reference = check(module, [np.ones((1, 1), dtype=np.float16)])
    assert reference.dtype == np.float16
    assert expect_graph(
        [
            ("Sqrt -> Div -> Mul -> Mul", {"counts": {"Mul": 2}}),
            "Sub -> Mul -> Add",
        ],
        must_absent=["BatchNormalization"],
        no_unused_inputs=True,
    )(model)


@pytest.mark.parametrize("kind", ["linear", "linear_general"])
def test_nnx_linear_preserves_wider_requested_accumulation(kind):
    module = _linear(
        kind,
        dtype=np.float16,
        param_dtype=np.float16,
        preferred_element_type=np.float32,
    )
    _, reference = check(module, [_linear_input(kind, 2048, np.float16)])
    assert reference.dtype == np.float32
    assert reference.item() == 2049


def test_half_rsqrt_coefficient_and_shared_values_keep_each_result_boundary():
    scale = jnp.asarray([0.7, 1 / 3, 1.1, 0.3], dtype=jnp.float16)

    def fn(x, factor):
        reciprocal = jax.lax.rsqrt(x)
        coefficient = reciprocal * scale
        return x, scale, reciprocal, coefficient, coefficient * factor

    values = np.asarray([3.0, 0.5, 0.1, 2.0], dtype=np.float16)
    factors = np.asarray([1.5, -2.0, 0.25, -0.5], dtype=np.float16)
    model, reference = check(fn, [values, factors])
    np.testing.assert_array_equal(reference[0], values)
    np.testing.assert_array_equal(reference[1], np.asarray(scale))
    assert all(value.dtype == np.float16 for value in reference)
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*model.graph.value_info, *model.graph.output)
    }
    # The primitive's Sqrt and reciprocal work are FP32. Its one FP16 result
    # remains observable before either source multiplication.
    assert all(
        types[node.output[0]] == onnx.TensorProto.FLOAT
        for node in model.graph.node
        if node.op_type in {"Sqrt", "Div", "Mul"}
    )
    assert expect_graph(
        [
            (
                "Sqrt -> Div -> Cast -> Mul -> Cast -> Mul -> Cast",
                {"counts": {"Sqrt": 1, "Div": 1, "Mul": 2}},
            )
        ],
        no_unused_inputs=True,
    )(model)


@pytest.mark.parametrize("case", ["finite", "exceptional"])
def test_half_rsqrt_selected_boundaries_match_native_consensus(case, record_property):
    if case == "finite":
        info = np.finfo(np.float16)
        values = np.asarray(
            [info.smallest_subnormal, info.tiny, 0.25, 1.0, 3.0, 4.0, 16.0, info.max],
            dtype=np.float16,
        )
    else:
        values = np.asarray(
            [0.0, -0.0, np.inf, -np.inf, -1.0, np.nan], dtype=np.float16
        )

    # These selected inputs establish native agreement and CPU ORT behavior;
    # they do not assert universal correctly rounded rsqrt or NaN payloads.
    qualified_boundary = (
        case == "finite"
        and jax.__version__ == "0.8.1"
        and jaxlib.__version__ == "0.8.1"
        and ort.__version__ == "1.24.1"
        and jax.default_backend() == "cpu"
    )
    if qualified_boundary:
        # Astra accepted this precise boundary under the user's delegation
        # after unchanged main and current exports matched bit-for-bit on all
        # eight inputs with both CPU ORT optimization settings.
        assert values.dtype == np.float16 and values.shape == (8,)
        input_bits = values.view(np.uint16)
        assert np.count_nonzero(input_bits == 0x7BFF) == 1
        native_input = jnp.asarray(values)
        reference = np.asarray(jax.lax.rsqrt(native_input))
        assert reference.dtype == np.float16 and reference.shape == (8,)
        reference_bits = reference.view(np.uint16)
        compiled = jax.jit(lambda value: jax.lax.rsqrt(value))
        for candidate in (compiled(native_input), compiled(native_input)):
            candidate = np.asarray(candidate)
            assert candidate.dtype == np.float16 and candidate.shape == (8,)
            # Exact bits also retain NaN/Inf classifications and zero signs.
            np.testing.assert_array_equal(candidate.view(np.uint16), reference_bits)
        model = to_onnx(jax.lax.rsqrt, [values])
        onnx.checker.check_model(model, full_check=True)
        assert len(model.graph.input) == len(model.graph.output) == 1
        for value in (*model.graph.input, *model.graph.output):
            assert value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
            assert tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim) == (
                8,
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
            assert session.get_providers() == ["CPUExecutionProvider"]
            (actual,) = session.run(None, {session.get_inputs()[0].name: values})
            assert actual.dtype == np.float16 and actual.shape == (8,)
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(reference))
            np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(reference))
            np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(reference))
            zeros = reference == 0
            np.testing.assert_array_equal(actual == 0, zeros)
            np.testing.assert_array_equal(
                np.signbit(actual[zeros]), np.signbit(reference[zeros])
            )
            actual_bits = actual.view(np.uint16)
            failed_indices = np.flatnonzero(actual_bits != reference_bits).tolist()
            record_property(f"{label}_failed_indices", json.dumps(failed_indices))
            record_property(f"{label}_failed_count", len(failed_indices))
            allowed = (
                (input_bits == 0x7BFF)
                & (reference_bits == 7169)
                & (actual_bits == 7168)
            )
            assert np.all(
                (actual_bits == reference_bits) | allowed
            ), f"Unapproved FP16 rsqrt mismatches: {failed_indices}"
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
    else:
        model, reference = check(jax.lax.rsqrt, [values])
    assert reference.dtype == np.float16
    assert expect_graph(
        [("Cast -> Sqrt -> Div -> Cast", {"counts": {"Sqrt": 1, "Div": 1}})],
        no_unused_inputs=True,
    )(model)


@pytest.mark.parametrize("case", ["finite", "exceptional"])
def test_half_multiply_selected_boundaries_preserve_shared_operands(case):
    info = np.finfo(np.float16)
    if case == "finite":
        lhs = np.asarray(
            [info.smallest_subnormal, info.tiny, 0.7, 3.0, info.max, 0.0, -0.0],
            dtype=np.float16,
        )
        rhs = np.asarray([2.0, 0.5, 1.1, 0.7, 2.0, -2.0, -2.0], dtype=np.float16)
    else:
        lhs = np.asarray([np.inf, -np.inf, np.inf, np.nan], dtype=np.float16)
        rhs = np.asarray([0.0, 2.0, -2.0, 1.0], dtype=np.float16)

    def fn(x, y):
        product = x * y
        return x, y, product, product * jnp.float16(0.25)

    _, reference = check(fn, [lhs, rhs])
    for actual, original in zip(reference[:2], (lhs, rhs)):
        np.testing.assert_array_equal(actual, original)
        zeros = original == 0
        np.testing.assert_array_equal(
            np.signbit(actual[zeros]), np.signbit(original[zeros])
        )
    assert all(value.dtype == np.float16 for value in reference)
