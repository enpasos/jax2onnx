# tests/extra_tests/framework/test_export_jit_cache_isolation.py

"""Tracing-time plugin bindings must not share native JAX JIT cache entries."""

from __future__ import annotations

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx.plugins import plugin_system
from jax2onnx.plugins.jax.numpy.arange import JnpArangePlugin


def _assert_export(source, values, expected):
    model = to_onnx(source, [values])
    onnx.checker.check_model(model, full_check=True)
    (actual,) = ReferenceEvaluator(model).run(None, {model.graph.input[0].name: values})
    np.testing.assert_array_equal(actual, expected)
    return model


@pytest.mark.parametrize("prewarm", [False, True])
def test_native_jit_before_and_after_repeated_exports(prewarm):
    original_arange = jnp.arange
    traced_arange = []

    @jax.jit
    def source(x):
        traced_arange.append(jnp.arange)
        return x + jnp.arange(x.shape[0], dtype=x.dtype)

    values = np.asarray([2, 4, 6, 8], np.float32)
    expected = np.asarray([2, 5, 8, 11], np.float32)
    if prewarm:
        np.testing.assert_array_equal(source(values), expected)
        assert traced_arange == [original_arange]

    _assert_export(source, values, expected)
    assert len(traced_arange) == int(prewarm) + 1
    assert traced_arange[-1] is not original_arange
    _assert_export(source, values, expected)
    assert len(traced_arange) == int(prewarm) + 1
    assert jnp.arange is original_arange

    np.testing.assert_array_equal(source(values), expected)
    np.testing.assert_array_equal(source(values), expected)
    assert len(traced_arange) == 2
    assert sum(fn is original_arange for fn in traced_arange) == 1


def test_leaf_binding_has_one_stable_nested_jit_namespace():
    original_arange = jnp.arange
    traced_arange = []

    @jax.jit
    def source(x):
        traced_arange.append(jnp.arange)
        return x + jnp.arange(x.shape[0], dtype=x.dtype)

    values = np.asarray([2, 4, 6, 8], np.float32)
    spec = jax.ShapeDtypeStruct(values.shape, values.dtype)
    with JnpArangePlugin.plugin_binding():
        jax.make_jaxpr(source)(spec)
        with JnpArangePlugin.plugin_binding():
            jax.make_jaxpr(source)(spec)
        jax.make_jaxpr(source)(spec)
    assert len(traced_arange) == 1
    assert traced_arange[0] is not original_arange
    assert jnp.arange is original_arange

    np.testing.assert_array_equal(source(values), np.asarray([2, 5, 8, 11], np.float32))
    assert len(traced_arange) == 2
    assert traced_arange[-1] is original_arange
    with JnpArangePlugin.plugin_binding():
        jax.make_jaxpr(source)(spec)
    assert len(traced_arange) == 2


@jax.jit
def _delayed_jit_arange(x):
    return x + jnp.arange(x.shape[0], dtype=x.dtype)


@onnx_function
def _delayed_function(x):
    return _delayed_jit_arange(x)


def test_delayed_function_body_keeps_native_jit_cache_separate():
    def source(x):
        return _delayed_function(x)

    values = np.asarray([2, 4, 6, 8], np.float32)
    expected = np.asarray([2, 5, 8, 11], np.float32)
    np.testing.assert_array_equal(source(values), expected)
    model = _assert_export(source, values, expected)
    assert model.functions
    _assert_export(source, values, expected)
    np.testing.assert_array_equal(source(values), expected)


def test_failed_export_restores_native_jit_namespace_and_bindings():
    original_arange = jnp.arange

    @jax.jit
    def source(x):
        return x + jnp.arange(x.shape[0], dtype=x.dtype)

    def failing_source(x):
        source(x)
        raise RuntimeError("failure after JIT tracing")

    values = np.asarray([2, 4, 6, 8], np.float32)
    with pytest.raises(RuntimeError, match="failure after JIT tracing"):
        to_onnx(failing_source, [values])
    assert jnp.arange is original_arange
    np.testing.assert_array_equal(source(values), np.asarray([2, 5, 8, 11], np.float32))


def test_partial_function_patch_failure_restores_before_namespace_exit(monkeypatch):
    events = []

    @contextmanager
    def record_namespace(value):
        events.append(("enter", value))
        try:
            yield
        finally:
            events.append(("exit", value))

    class TargetMeta(type):
        def __setattr__(cls, name, value):
            events.append(("assign", value))
            super().__setattr__(name, value)

    class Target(metaclass=TargetMeta):
        method = object()
        other = method

    original = Target.method
    replacement = object()

    def install(_original):
        events.append(("install", _original))
        return replacement

    def fail(_original):
        events.append(("fail", _original))
        raise RuntimeError("partial patch failure")

    monkeypatch.setattr(plugin_system, "_PLUGIN_JIT_CONTEXT", record_namespace)
    monkeypatch.setattr(
        plugin_system,
        "_iter_patch_specs",
        lambda: iter([(install, [Target], "method"), (fail, [Target], "other")]),
    )
    with pytest.raises(RuntimeError, match="partial patch failure"):
        with plugin_system.apply_monkey_patches():
            pytest.fail("patch installation should fail before entering the body")

    assert Target.method is original
    assert (Target, "method") not in plugin_system._PATCH_STATE
    assert events == [
        ("enter", True),
        ("install", original),
        ("assign", replacement),
        ("fail", original),
        ("assign", original),
        ("exit", True),
    ]
