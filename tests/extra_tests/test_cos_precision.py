# tests/extra_tests/test_cos_precision.py

from __future__ import annotations

from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx


def _jnp_cos(x: jax.Array) -> jax.Array:
    return jnp.cos(x)


def _lax_cos(x: jax.Array) -> jax.Array:
    return jax.lax.cos(x)


def _op_types(model: onnx.ModelProto) -> set[str]:
    return {node.op_type for node in model.graph.node}


def _run(
    model: onnx.ModelProto,
    x: np.ndarray,
    optimization: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
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


@pytest.mark.parametrize("dtype", [np.float32, np.int32], ids=["float32", "int32"])
def test_jnp_cos_emits_cos_below_float64(dtype: type) -> None:
    x: np.ndarray = np.arange(6).astype(dtype)

    model = to_onnx(lambda v: jnp.cos(v), [x])

    assert "Cos" in _op_types(model)
    assert "Sin" not in _op_types(model)
    np.testing.assert_allclose(_run(model, x), np.cos(x.astype(np.float64)), atol=1e-6)


@pytest.fixture
def enable_x64() -> Iterator[None]:
    original_x64 = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", original_x64)


@pytest.mark.parametrize("fn", [_jnp_cos, _lax_cos], ids=["jnp", "lax"])
@pytest.mark.parametrize(
    "optimization",
    [
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    ],
    ids=["optimized", "unoptimized"],
)
def test_cos_float64_is_accurate_across_argument_range(
    fn: Callable[[jax.Array], jax.Array],
    optimization: ort.GraphOptimizationLevel,
    enable_x64: None,
) -> None:
    rng = np.random.default_rng(262)
    roots = np.pi / 2 + np.arange(-5, 6) * np.pi
    finite = np.concatenate(
        [
            np.linspace(-4.0, 4.0, 101),
            rng.uniform(-1e6, 1e6, 256),
            rng.uniform(-1.0, 1.0, 1024) * np.logspace(6, 308, 1024),
            roots,
            np.nextafter(roots, -np.inf),
            np.nextafter(roots, np.inf),
            [0.0, -0.0, np.nextafter(0.0, 1.0), np.finfo(np.float64).max],
        ]
    )
    x = np.concatenate([finite, [np.nan, np.inf, -np.inf]])
    model = to_onnx(fn, [x], enable_double_precision=True)

    # All supported ORT versions provide DOUBLE Sin, whereas older CPU
    # runtimes do not provide DOUBLE Cos. Avoid a phase-shifting Add too.
    assert {"Mul", "Sin", "Sub"} <= _op_types(model)
    assert not {"Cos", "Add"} & _op_types(model)
    actual = _run(model, x, optimization)
    with np.errstate(invalid="ignore"):
        expected = np.cos(x)
    # Absolute error is the useful contract near zeros of cosine. The previous
    # sin(x + pi/2) fallback can have order-one errors on these large inputs.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15, equal_nan=True)
    np.testing.assert_allclose(
        actual, np.asarray(jax.jit(fn)(x)), rtol=0, atol=2e-15, equal_nan=True
    )
    assert actual.dtype == np.float64


@pytest.mark.parametrize("fn", [_jnp_cos, _lax_cos], ids=["jnp", "lax"])
@pytest.mark.parametrize("shape", [(), ("B",)], ids=["scalar", "symbolic"])
def test_cos_float64_preserves_scalar_and_symbolic_shapes(
    fn: Callable[[jax.Array], jax.Array],
    shape: tuple[int | str, ...],
    enable_x64: None,
) -> None:
    model = to_onnx(
        fn, [jax.ShapeDtypeStruct(shape, np.float64)], enable_double_precision=True
    )
    if shape:
        assert model.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "B"
        inputs = [np.array([1e14]), np.array([-1e100, 0.0, 1e14])]
    else:
        inputs = [np.asarray(1e14)]
    for x in inputs:
        actual = _run(model, x)
        assert actual.shape == x.shape
        np.testing.assert_allclose(actual, np.cos(x), rtol=0, atol=2e-15)


def test_jnp_cos_is_accurate_for_rotary_angles() -> None:
    # RoPE-style angle table; cos(x) as sin(x + pi/2) loses ~2.6e-5 here.
    positions: np.ndarray = np.arange(1024, dtype=np.float32)[:, None]
    frequencies = (1.0 / 10_000 ** (np.arange(0, 64, 2, dtype=np.float32) / 64))[None]
    angles = (positions * frequencies).astype(np.float32)

    model = to_onnx(lambda v: jnp.cos(v), [angles])
    expected = np.cos(angles.astype(np.float64))

    np.testing.assert_allclose(_run(model, angles), expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jnp.cos)(angles)), expected, rtol=0, atol=1e-6
    )
