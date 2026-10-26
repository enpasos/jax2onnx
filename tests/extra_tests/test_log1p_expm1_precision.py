# tests/extra_tests/test_log1p_expm1_precision.py

from __future__ import annotations

from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx

# Measured worst case is 3 ulp on ONNX Runtime CPU; Log(1 + x) and Exp(x) - 1
# are off by hundreds to millions of ulp for |x| below 0.1.
_MAX_ULP_ERROR = 8.0


def _log1mexp_reference(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.where(x < np.log(2.0), np.log(-np.expm1(-x)), np.log1p(-np.exp(-x)))


def _log_sigmoid_reference(x: np.ndarray) -> np.ndarray:
    result: np.ndarray = -np.logaddexp(0.0, -x)
    return result


# Look the functions up at call time: conversion patches the jnp and jax.nn
# functions, and a captured reference would bypass those plugins.
_FUNCTIONS: dict[
    str, tuple[Callable[[jax.Array], jax.Array], Callable[[np.ndarray], np.ndarray]]
] = {
    "lax.log1p": (lambda x: jax.lax.log1p(x), np.log1p),
    "jnp.log1p": (lambda x: jnp.log1p(x), np.log1p),
    "lax.expm1": (lambda x: jax.lax.expm1(x), np.expm1),
    "jnp.expm1": (lambda x: jnp.expm1(x), np.expm1),
    "nn.log1mexp": (lambda x: jax.nn.log1mexp(x), _log1mexp_reference),
    "lax.sinh": (lambda x: jax.lax.sinh(x), np.sinh),
    "jnp.sinh": (lambda x: jnp.sinh(x), np.sinh),
    "nn.log_sigmoid": (lambda x: jax.nn.log_sigmoid(x), _log_sigmoid_reference),
}


def _finite_inputs(name: str, dtype: type) -> np.ndarray:
    info = np.finfo(dtype)
    small = np.logspace(np.log10(float(info.tiny)), 0.0, 400)
    # The geometric range starts at the smallest normal value, only 2**-14 for
    # float16, so the subnormal and normal edges are added explicitly.
    edge_cases = np.array([float(info.smallest_subnormal), float(info.tiny), 1e-3])
    # Inputs around the point where fl(1 + x) or fl(exp(x)) becomes one.
    rounding_edge = float(info.eps) * np.arange(-16, 17) / 8
    if name.endswith("log1p"):
        # geomspace overflows internally before snapping its endpoint to max.
        with np.errstate(over="ignore"):
            large = np.geomspace(10.0, float(info.max), 60)
        parts = [
            small,
            -small[small < 1.0],
            -1.0 + np.logspace(np.log10(float(info.eps)), -0.3, 200),
            np.linspace(-0.99, 10.0, 300),
            large,
            rounding_edge,
            edge_cases,
            -edge_cases,
        ]
    elif name.endswith("expm1"):
        limit = 0.999 * float(np.log(info.max))
        parts = [
            small,
            -small,
            np.linspace(-limit, limit, 400),
            rounding_edge,
            edge_cases,
            -edge_cases,
        ]
    elif name.endswith("sinh"):
        # Up to the overflow threshold, including the band where exp(|x|)
        # already overflows but sinh(x) does not.
        edge = float(np.arcsinh(np.float64(info.max)))
        band = np.linspace(np.log(float(info.max)), edge, 20, endpoint=False)
        parts = [
            small,
            -small,
            np.linspace(-edge, edge, 400)[1:-1],
            band,
            -band,
            edge_cases,
            -edge_cases,
        ]
    elif name.endswith("log_sigmoid"):
        # Above -log(tiny), the result underflows. Stopping at max / 2 keeps
        # the ulp of every expected value finite.
        largest = -float(np.log(info.tiny))
        very_negative = -np.geomspace(10.0, float(info.max) / 2, 60)
        parts = [
            small,
            -small,
            np.linspace(-largest, largest, 400),
            very_negative,
            edge_cases,
            -edge_cases,
        ]
    else:
        largest = -float(np.log(info.tiny))
        parts = [
            small,
            np.linspace(0.01, 5.0, 200),
            np.linspace(5.0, largest, 100),
            edge_cases,
        ]
    return np.unique(np.concatenate(parts).astype(dtype))


def _run(
    model: onnx.ModelProto,
    x: np.ndarray,
    optimization: ort.GraphOptimizationLevel,
) -> np.ndarray:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = optimization
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(None, {session.get_inputs()[0].name: x})
    result: np.ndarray = np.asarray(actual)
    return result


def _run_with_raw_bits(
    model: onnx.ModelProto, bits: np.ndarray, onnx_type: int
) -> np.ndarray:
    # ONNX Runtime's NumPy binding accepts bfloat16 only as raw uint16 bits.
    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out_bits = np.empty_like(bits)
    binding = session.io_binding()
    for bind, meta, array in (
        (binding.bind_ortvalue_input, session.get_inputs()[0], bits),
        (binding.bind_ortvalue_output, session.get_outputs()[0], out_bits),
    ):
        value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(array, onnx_type)
        bind(meta.name, value)
    session.run_with_iobinding(binding)
    return out_bits


def _export(name: str, x: np.ndarray) -> onnx.ModelProto:
    return to_onnx(
        _FUNCTIONS[name][0], [x], enable_double_precision=x.dtype == np.float64
    )


@pytest.fixture(
    params=[np.float16, np.float32, np.float64],
    ids=["float16", "float32", "float64"],
)
def dtype(request: pytest.FixtureRequest) -> Iterator[type]:
    original_x64 = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", request.param == np.float64)
    try:
        yield request.param
    finally:
        jax.config.update("jax_enable_x64", original_x64)


@pytest.mark.parametrize("name", list(_FUNCTIONS))
@pytest.mark.parametrize(
    "optimization",
    [
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    ],
    ids=["optimized", "unoptimized"],
)
def test_relative_accuracy_holds_near_zero_and_across_range(
    name: str, optimization: ort.GraphOptimizationLevel, dtype: type
) -> None:
    x = _finite_inputs(name, dtype)
    actual = _run(_export(name, x), x, optimization)

    # float64 NumPy is effectively exact for float16 and float32 inputs and
    # within about one ulp for float64 inputs.
    expected = _FUNCTIONS[name][1](x.astype(np.float64))
    spacing = np.spacing(np.abs(expected.astype(dtype)))
    ulp_error = np.abs(actual.astype(np.float64) - expected) / spacing

    assert actual.dtype == dtype
    worst = int(np.argmax(ulp_error))
    assert ulp_error[worst] <= _MAX_ULP_ERROR, (
        f"{ulp_error[worst]:.1f} ulp at x={x[worst]!r}: "
        f"got {actual[worst]!r}, expected {expected[worst]!r}"
    )


@pytest.mark.parametrize(
    ("name", "values", "expected"),
    [
        (
            "lax.log1p",
            [-1.0, -2.0, -np.inf, np.inf, np.nan],
            [-np.inf, np.nan, np.nan, np.inf, np.nan],
        ),
        (
            "lax.expm1",
            [np.inf, -np.inf, np.nan, 1e4, -1e4],
            [np.inf, -1.0, np.nan, np.inf, -1.0],
        ),
        (
            "nn.log1mexp",
            [0.0, -1.0, np.inf, np.nan],
            [-np.inf, np.nan, 0.0, np.nan],
        ),
        (
            "lax.sinh",
            [np.inf, -np.inf, np.nan, 1e4, -1e4],
            [np.inf, -np.inf, np.nan, np.inf, -np.inf],
        ),
        (
            "nn.log_sigmoid",
            [np.inf, -np.inf, np.nan],
            [0.0, -np.inf, np.nan],
        ),
    ],
    ids=["log1p", "expm1", "log1mexp", "sinh", "log_sigmoid"],
)
def test_exceptional_values_match_jax(
    name: str, values: list[float], expected: list[float], dtype: type
) -> None:
    x = np.asarray(values, dtype=dtype)
    actual = _run(_export(name, x), x, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=dtype))
    np.testing.assert_array_equal(actual, np.asarray(_FUNCTIONS[name][0](x)))


@pytest.mark.parametrize("name", ["lax.log1p", "lax.sinh", "jnp.sinh"])
def test_negative_zero_keeps_its_sign(name: str, dtype: type) -> None:
    x = np.array([-0.0, 0.0], dtype=dtype)
    actual = _run(_export(name, x), x, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

    np.testing.assert_array_equal(np.signbit(actual), [True, False])


_HALF_PRECISION = {
    # bfloat16 is computed in float32 and rounded once; ONNX Runtime has no
    # bfloat16 Exp or Log kernel, so the session would not start otherwise.
    "bfloat16": (jnp.bfloat16, onnx.TensorProto.BFLOAT16, 1.0),
    # float16 is computed in float16. ONNX Runtime's CPU provider upcasts it
    # internally (0.5 ulp measured); CUDA computes it natively (about 2.6 ulp).
    "float16": (np.float16, onnx.TensorProto.FLOAT16, _MAX_ULP_ERROR),
}


@pytest.mark.parametrize("name", list(_FUNCTIONS))
@pytest.mark.parametrize("half", list(_HALF_PRECISION))
@pytest.mark.parametrize(
    "enable_double_precision", [False, True], ids=["default", "double_precision"]
)
def test_half_precision_keeps_its_dtype_and_accuracy_for_every_input(
    name: str, half: str, enable_double_precision: bool
) -> None:
    dtype, onnx_type, max_ulp_error = _HALF_PRECISION[half]
    bits = np.arange(1 << 16, dtype=np.uint16)
    x = bits.view(dtype)
    model = to_onnx(
        _FUNCTIONS[name][0],
        [jax.ShapeDtypeStruct(x.shape, dtype)],
        enable_double_precision=enable_double_precision,
    )
    # Double precision widens float32 only; half-precision values keep their
    # dtype at the model boundary.
    assert model.graph.output[0].type.tensor_type.elem_type == onnx_type
    actual = _run_with_raw_bits(model, bits, onnx_type).view(dtype).astype(np.float64)

    with np.errstate(all="ignore"):
        expected = _FUNCTIONS[name][1](x.astype(np.float64))
        rounded = expected.astype(dtype).astype(np.float64)
    exceptional = ~np.isfinite(rounded)
    np.testing.assert_array_equal(actual[exceptional], rounded[exceptional])
    finite = ~exceptional
    magnitude = np.abs(rounded[finite]).astype(dtype)
    # np.spacing overflows at the largest finite value, which float16
    # log_sigmoid(-max) returns; use the gap below it there.
    with np.errstate(over="ignore"):
        spacing = np.spacing(magnitude)
    spacing = np.where(
        np.isinf(spacing), magnitude - np.nextafter(magnitude, dtype(0)), spacing
    )
    ulp_error = np.abs(actual[finite] - expected[finite]) / spacing.astype(np.float64)
    worst = int(np.argmax(ulp_error))
    assert ulp_error[worst] <= max_ulp_error, (
        f"{ulp_error[worst]:.2f} ulp at x={x[finite][worst]!r}: "
        f"got {actual[finite][worst]!r}, expected {expected[finite][worst]!r}"
    )
