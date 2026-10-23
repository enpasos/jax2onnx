# tests/extra_tests/converter/test_source_function_context.py

"""Source functions retain call-site configuration and isolate JAX trace caches."""

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
)
from tests.extra_tests.converter.precision_test_support import (
    PrototypeLimit,
    export_source_dtypes,
)


@pytest.fixture(autouse=True)
def _restore_jax_config() -> Iterator[None]:
    previous_x64 = bool(jax.config.x64_enabled)
    previous_platforms = jax.config.jax_platforms
    previous_matmul_precision = jax.config.jax_default_matmul_precision
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_platforms", previous_platforms)
        jax.config.update("jax_default_matmul_precision", previous_matmul_precision)


@onnx_function
def _cancellation(x: jax.Array) -> jax.Array:
    wide = x.astype(jnp.float64)
    return ((wide + np.float64(1)) - wide).astype(jnp.float32)


@onnx_function
def _wide_result(x: jax.Array) -> jax.Array:
    return x.astype(jnp.float64) + np.float64(1)


@onnx_function
def _nested_cancellation(x: jax.Array) -> jax.Array:
    with jax.enable_x64(True):
        return _cancellation(x)


@onnx_function
def _negated(x: jax.Array) -> jax.Array:
    return -x


@onnx_function
def _unsupported(x: jax.Array) -> jax.Array:
    return jnp.sin(x.astype(jnp.float64))


@onnx_function
def _preferred_dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        lhs,
        rhs,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _spec() -> list[jax.ShapeDtypeStruct]:
    return [jax.ShapeDtypeStruct((), np.float32)]


def _run(
    model: onnx.ModelProto, x: np.ndarray, *, optimize: bool = False
) -> list[np.ndarray]:
    reloaded = onnx.load_model_from_string(model.SerializeToString())
    onnx.checker.check_model(reloaded, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(
        reloaded.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    dtype = onnx.helper.tensor_dtype_to_np_dtype(
        reloaded.graph.input[0].type.tensor_type.elem_type
    )
    return session.run(None, {session.get_inputs()[0].name: x.astype(dtype)})


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize("local_x64", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_source_function_replays_call_site_x64_with_unchanged_result_dtype(
    ambient_x64: bool, local_x64: bool, optimize: bool
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)

    def model(x: jax.Array) -> jax.Array:
        with jax.enable_x64(local_x64):
            return _cancellation(x)

    x = np.asarray(1e8, np.float32)
    expected = np.asarray(model(jnp.asarray(x)))
    assert expected.dtype == np.float32
    assert expected == (1 if local_x64 else 0)
    exported = export_source_dtypes(model, _spec())
    (actual,) = _run(exported, x, optimize=optimize)
    np.testing.assert_array_equal(actual, expected, strict=True)
    assert bool(jax.config.x64_enabled) is ambient_x64
    assert active_precision_policy() is PrecisionPolicy.LEGACY


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_same_signature_source_calls_have_distinct_x64_specializations(
    ambient_x64: bool, optimize: bool
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)

    def model(x: jax.Array) -> tuple[jax.Array, ...]:
        with jax.enable_x64(True):
            wide = _cancellation(x)
            repeated_wide = _cancellation(x)
        with jax.enable_x64(False):
            narrow = _cancellation(x)
        return wide, narrow, repeated_wide

    x = np.asarray(1e8, np.float32)
    expected = model(jnp.asarray(x))
    exported = export_source_dtypes(model, _spec())
    assert len(exported.functions) == 2
    for actual, reference in zip(_run(exported, x, optimize=optimize), expected):
        np.testing.assert_array_equal(actual, np.asarray(reference), strict=True)
    assert bool(jax.config.x64_enabled) is ambient_x64


@pytest.mark.parametrize("optimize", [False, True])
def test_source_call_site_x64_preserves_fp64_result(optimize: bool) -> None:
    jax.config.update("jax_enable_x64", False)

    def model(x: jax.Array) -> jax.Array:
        with jax.enable_x64(True):
            return _wide_result(x)

    x = np.asarray(1e8, np.float32)
    expected = np.asarray(model(jnp.asarray(x)))
    assert expected.dtype == np.float64 and expected == 100000001
    exported = export_source_dtypes(model, _spec())
    (actual,) = _run(exported, x, optimize=optimize)
    np.testing.assert_array_equal(actual, expected, strict=True)
    assert not jax.config.x64_enabled


def test_nested_source_functions_restore_each_call_site_x64() -> None:
    jax.config.update("jax_enable_x64", False)
    x = np.asarray(1e8, np.float32)
    exported = export_source_dtypes(lambda x: _nested_cancellation(x), _spec())
    assert len(exported.functions) == 2
    (actual,) = _run(exported, x)
    np.testing.assert_array_equal(actual, np.asarray(1, np.float32), strict=True)
    assert not jax.config.x64_enabled


@pytest.mark.parametrize("optimize", [False, True])
def test_existing_fp64_formal_survives_local_x64_disabled(optimize: bool) -> None:
    jax.config.update("jax_enable_x64", True)

    def model(x: jax.Array) -> jax.Array:
        with jax.enable_x64(False):
            return _negated(x)

    x = np.asarray(1 + 2**-30, np.float64)
    expected = np.asarray(model(jnp.asarray(x)))
    assert expected.dtype == np.float64
    exported = export_source_dtypes(model, [jax.ShapeDtypeStruct((), np.float64)])
    assert len(exported.functions) == 1
    (actual,) = _run(exported, x, optimize=optimize)
    np.testing.assert_array_equal(actual, expected, strict=True)
    assert jax.config.x64_enabled


def _conditional() -> Callable[[jax.Array], jax.Array]:
    # Stable branch identities within a test exercise JAX's cached body traces;
    # distinct closures prevent another parameter case from warming those caches.
    def branch(x: jax.Array) -> jax.Array:
        return _negated(x)

    def other_branch(x: jax.Array) -> jax.Array:
        return x

    def model(x: jax.Array) -> jax.Array:
        return jax.lax.cond(x > np.float32(0), branch, other_branch, x)

    return model


@pytest.mark.parametrize("source_first", [False, True])
@pytest.mark.parametrize("legacy", [None, False, True])
@pytest.mark.parametrize("ambient_x64", [False, True])
def test_cached_cond_branches_isolate_source_and_legacy_exports(
    source_first: bool, legacy: bool | None, ambient_x64: bool
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)
    model = _conditional()
    kwargs: dict[str, Any] = (
        {} if legacy is None else {"enable_double_precision": legacy}
    )
    # A separate callable provides the untouched public-export reference.
    baseline = to_onnx(_conditional(), _spec(), **kwargs)
    assert isinstance(baseline, onnx.ModelProto)
    routes = (
        ("source", "legacy", "source")
        if source_first
        else ("legacy", "source", "legacy")
    )
    for route in routes:
        exported = (
            export_source_dtypes(model, _spec())
            if route == "source"
            else to_onnx(model, _spec(), **kwargs)
        )
        assert isinstance(exported, onnx.ModelProto)
        assert len(exported.functions) == 1
        for scalar in (-3, 3):
            x = np.asarray(scalar, np.float32)
            (actual,) = _run(exported, x)
            expected = (
                np.asarray(-abs(scalar), np.float32)
                if route == "source"
                else _run(baseline, x)[0]
            )
            np.testing.assert_array_equal(actual, expected, strict=True)
        assert bool(jax.config.x64_enabled) is ambient_x64
    assert active_precision_policy() is PrecisionPolicy.LEGACY


@pytest.mark.parametrize("warm_ordinary_jax", [False, True])
def test_source_cond_cache_does_not_escape_to_ordinary_jax(
    warm_ordinary_jax: bool,
) -> None:
    jax.config.update("jax_enable_x64", False)
    model = _conditional()
    x = jnp.asarray(3, np.float32)
    if warm_ordinary_jax:
        np.testing.assert_array_equal(model(x), np.asarray(-3, np.float32))
        np.testing.assert_array_equal(jax.jit(model)(x), np.asarray(-3, np.float32))
    exported = export_source_dtypes(model, _spec())
    assert len(exported.functions) == 1
    # A cached custom source primitive has no ordinary JAX MLIR lowering.
    for scalar in (-3, 3):
        x = jnp.asarray(scalar, np.float32)
        expected = np.asarray(-abs(scalar), np.float32)
        np.testing.assert_array_equal(model(x), expected, strict=True)
        np.testing.assert_array_equal(jax.jit(model)(x), expected, strict=True)


def test_failed_source_function_export_restores_tracing_context() -> None:
    jax.config.update("jax_enable_x64", False)

    def model(x: jax.Array) -> jax.Array:
        with jax.enable_x64(True):
            return _unsupported(x)

    with pytest.raises(PrototypeLimit, match="sin"):
        export_source_dtypes(model, _spec())
    assert not jax.config.x64_enabled
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    ordinary = _conditional()
    np.testing.assert_array_equal(
        jax.jit(ordinary)(jnp.asarray(3, np.float32)), np.asarray(-3, np.float32)
    )
    public = to_onnx(ordinary, _spec(), enable_double_precision=False)
    assert isinstance(public, onnx.ModelProto)
    (actual,) = _run(public, np.asarray(3, np.float32))
    np.testing.assert_array_equal(actual, np.asarray(-3, np.float32), strict=True)


def test_scoped_matrix_precision_requirement_survives_source_function_retrace() -> None:
    jax.config.update("jax_default_matmul_precision", None)

    def model(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        with jax.default_matmul_precision("highest"):
            return _preferred_dot(lhs, rhs)

    specs = [
        jax.ShapeDtypeStruct((1, 2), np.float16),
        jax.ShapeDtypeStruct((2, 1), np.float16),
    ]
    with pytest.raises(
        NotImplementedError,
        match=r"SOURCE dot_general: precision .*native FP64-compute Precision enum pairs",
    ):
        export_source_dtypes(model, specs)
    assert jax.config.jax_default_matmul_precision is None
    assert active_precision_policy() is PrecisionPolicy.LEGACY


def test_default_matrix_scope_overrides_ambient_highest_for_source_function() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")

    def model(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        with jax.default_matmul_precision(None):
            return _preferred_dot(lhs, rhs)

    lhs = np.asarray([[2048, 1]], np.float16)
    rhs = np.ones((2, 1), np.float16)
    exported = export_source_dtypes(
        model, [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in (lhs, rhs)]
    )
    reloaded = onnx.load_model_from_string(exported.SerializeToString())
    onnx.checker.check_model(reloaded, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        reloaded.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(
        None,
        {value.name: array for value, array in zip(session.get_inputs(), (lhs, rhs))},
    )
    np.testing.assert_array_equal(actual, np.asarray([[2049]], np.float32), strict=True)
    assert jax.config.jax_default_matmul_precision == "highest"


def test_source_function_cache_distinguishes_scoped_matrix_requirements() -> None:
    jax.config.update("jax_default_matmul_precision", None)

    def model(lhs: jax.Array, rhs: jax.Array) -> tuple[jax.Array, jax.Array]:
        ordinary = _preferred_dot(lhs, rhs)
        with jax.default_matmul_precision("highest"):
            demanding = _preferred_dot(lhs, rhs)
        return ordinary, demanding

    specs = [
        jax.ShapeDtypeStruct((1, 2), np.float16),
        jax.ShapeDtypeStruct((2, 1), np.float16),
    ]
    # Reusing the first body's definition would incorrectly accept the second
    # call despite the explicitly unsupported accumulation requirement.
    with pytest.raises(
        NotImplementedError,
        match=r"SOURCE dot_general: precision .*native FP64-compute Precision enum pairs",
    ):
        export_source_dtypes(model, specs)
    assert jax.config.jax_default_matmul_precision is None
