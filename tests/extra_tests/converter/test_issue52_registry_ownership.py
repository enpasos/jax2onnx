# tests/extra_tests/converter/test_issue52_registry_ownership.py

"""Issue52 discovery observes primitive ownership without changing runtime calls."""

import importlib
import inspect
import sys
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx._compat.jax import Primitive, preserve_caller_configuration


@pytest.fixture
def fixture():
    # The historical fixture sets x64 at import. This control owns that import
    # and restores the actual caller configuration without changing the fixture.
    with preserve_caller_configuration(), jax.enable_x64(True):
        helper = importlib.import_module(
            "tests.extra_tests.helpers.issue52_scatter_payload_fixture"
        )
        helper._primitive_registry.cache_clear()
        try:
            yield helper
        finally:
            helper._primitive_registry.cache_clear()


def _state(primitive):
    fields = vars(primitive)
    return (
        "bind" in fields,
        id(fields["bind"]) if "bind" in fields else None,
        id(inspect.getattr_static(type(primitive), "bind")),
        id(inspect.getattr_static(primitive, "impl", None)),
        id(inspect.getattr_static(primitive, "abstract_eval", None)),
        "_sharding_bind_wrapped" in fields,
        (
            id(fields["_sharding_bind_wrapped"])
            if "_sharding_bind_wrapped" in fields
            else None
        ),
    )


def _preexisting_primitives():
    primitives = {}
    for module in tuple(sys.modules.values()):
        if isinstance(module, ModuleType) and module.__name__.startswith("jax"):
            for value in tuple(vars(module).values()):
                if isinstance(value, Primitive):
                    primitives[id(value)] = value
    return primitives


def test_repeated_registry_discovery_preserves_all_preexisting_bindings(fixture):
    plugin = importlib.import_module("jax2onnx.plugins.jax.numpy.cumsum")
    primitives = _preexisting_primitives()
    assert id(plugin.JnpCumSumPlugin._PRIM) in primitives
    before = {key: _state(value) for key, value in primitives.items()}
    public = jax.lax.cumsum, jnp.cumsum, plugin._ORIGINAL_JNP_CUMSUM
    first = fixture._primitive_registry()
    assert first is fixture._primitive_registry()
    assert first is fixture._primitive_registry()
    assert first[plugin.JnpCumSumPlugin._PRIM.name] is plugin.JnpCumSumPlugin._PRIM
    assert {key: _state(value) for key, value in primitives.items()} == before
    assert all(
        actual is expected
        for actual, expected in zip(
            (jax.lax.cumsum, jnp.cumsum, plugin._ORIGINAL_JNP_CUMSUM),
            public,
            strict=True,
        )
    )


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("dtype", (np.int32, np.float64))
def test_discovery_keeps_native_and_runtime_cumsum_dtype_reverse_and_vmap(
    fixture, reverse, dtype
):
    plugin = importlib.import_module("jax2onnx.plugins.jax.numpy.cumsum")
    shim, native = plugin._runtime_cumsum, plugin._ORIGINAL_JNP_CUMSUM
    values = np.asarray([[1, 2, 3], [4, 5, 6]], np.int32)
    expected_forward = np.cumsum(values, axis=1, dtype=dtype)
    expected = (
        np.flip(np.cumsum(np.flip(values, axis=1), axis=1, dtype=dtype), axis=1)
        if reverse
        else expected_forward
    )
    public = jax.lax.cumsum, jnp.cumsum
    primitive = plugin.JnpCumSumPlugin._PRIM
    before = _state(primitive)
    for after_discovery in (False, True):
        if after_discovery:
            fixture._primitive_registry()
            fixture._primitive_registry()
        native_result = native(values, axis=1, dtype=dtype)
        actual = shim(values, axis=1, reverse=reverse, dtype=dtype)
        mapped = jax.vmap(lambda row: shim(row, axis=0, reverse=reverse, dtype=dtype))(
            values
        )
        assert np.asarray(native_result).dtype == np.dtype(dtype)
        np.testing.assert_array_equal(np.asarray(native_result), expected_forward)
        for result in (actual, mapped):
            assert np.asarray(result).dtype == np.dtype(dtype)
            np.testing.assert_array_equal(np.asarray(result), expected)
        assert _state(primitive) == before
        assert all(
            actual is expected
            for actual, expected in zip(
                (jax.lax.cumsum, jnp.cumsum), public, strict=True
            )
        )


def test_discovery_keeps_the_existing_unsupported_exclusive_error(fixture):
    plugin = importlib.import_module("jax2onnx.plugins.jax.numpy.cumsum")
    shim = plugin._runtime_cumsum
    before = _state(plugin.JnpCumSumPlugin._PRIM)
    for after_discovery in (False, True):
        if after_discovery:
            fixture._primitive_registry()
            fixture._primitive_registry()
        with pytest.raises(TypeError, match="unexpected keyword argument 'exclusive'"):
            shim(np.asarray([1, 2, 3], np.int32), axis=0, exclusive=True)
    assert _state(plugin.JnpCumSumPlugin._PRIM) == before


@pytest.mark.parametrize("origin", ("discovered", "fallback"))
def test_registry_and_fallback_never_drop_unknown_keys_or_invent_required_defaults(
    fixture, monkeypatch, origin
):
    calls = []

    def implementation(value, *, required):
        calls.append(required)
        return value

    name = "issue52_required_parameter_control"
    if origin == "fallback":
        primitive = fixture._build_fallback_primitive(name, implementation)
    else:
        primitive = Primitive(name)
        primitive.def_impl(implementation)
    module = ModuleType("jax.issue52_registry_control")
    module.primitive = primitive
    monkeypatch.setitem(sys.modules, module.__name__, module)
    value = np.asarray([1], np.int32)
    before = _state(primitive)
    for after_discovery in (False, True):
        if after_discovery:
            assert fixture._primitive_registry()[name] is primitive
            fixture._primitive_registry()
        with pytest.raises(TypeError, match="unexpected keyword argument 'semantic'"):
            primitive.bind(value, required=True, semantic=True)
        with pytest.raises(TypeError, match="required keyword-only argument.*required"):
            primitive.bind(value)
        assert not calls
        np.testing.assert_array_equal(primitive.bind(value, required=True), value)
        assert calls == [True]
        calls.clear()
        assert _state(primitive) == before


@pytest.mark.parametrize(
    "message,parameters",
    [
        ("native() got an unexpected keyword argument 'semantic'", {"semantic": True}),
        ("native() missing 1 required keyword-only argument: 'transport'", {}),
    ],
)
def test_native_typeerror_identity_and_single_call_survive_discovery(
    fixture, monkeypatch, message, parameters
):
    calls = []
    sentinel = TypeError(message)

    def implementation(value, **keywords):
        calls.append(dict(keywords))
        raise sentinel

    primitive = Primitive("issue52_body_typeerror_control")
    primitive.def_impl(implementation)
    module = ModuleType("jax.issue52_body_typeerror_control")
    module.primitive = primitive
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert fixture._primitive_registry()[primitive.name] is primitive
    fixture._primitive_registry()
    with pytest.raises(TypeError) as error:
        primitive.bind(np.asarray([1], np.int32), **parameters)
    assert error.value is sentinel
    assert calls == [parameters]
