# tests/extra_tests/converter/test_source_linen_pool_forwarding.py

"""The actual Linen pool registration preserves native SOURCE calls."""

from typing import Any

from flax import linen, nnx
from flax.linen import pooling
import jax
import numpy as np
import pytest

from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    precision_policy_scope,
)
from jax2onnx.plugins._patching import apply_patches
from jax2onnx.plugins.flax.linen.pool import PoolPlugin
from jax2onnx.plugins.flax.nnx.avg_pool import AvgPoolPlugin
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)


@pytest.mark.parametrize("alias", ["linen", "pooling"])
@pytest.mark.parametrize("raises", [False, True])
def test_source_pool_registration_forwards_exact_arguments_and_native_errors(
    monkeypatch: pytest.MonkeyPatch, alias: str, raises: bool
) -> None:
    arguments = (object(), object(), object(), [2, 3], None, ((1, 0), (0, 1)))
    keywords = {"native_keyword": object()}
    result = object()
    error = ValueError("original native callable error")
    calls = []

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        if raises:
            raise error
        return result

    monkeypatch.setattr(PoolPlugin, "_ORIG_CALL", PoolPlugin._ORIG_CALL)
    monkeypatch.setattr(linen, "pool", original)
    monkeypatch.setattr(pooling, "pool", original)
    module = linen if alias == "linen" else pooling

    def invoke() -> None:
        assert PoolPlugin._ORIG_CALL is original
        assert getattr(module.pool, "__j2o_linen_pool_original__") is original
        if raises:
            with pytest.raises(ValueError) as caught:
                module.pool(*arguments, **keywords)
            assert caught.value is error
        else:
            assert module.pool(*arguments, **keywords) is result
        actual_args, actual_kwargs = calls[-1]
        assert all(
            actual is expected
            for actual, expected in zip(actual_args, arguments, strict=True)
        )
        assert actual_kwargs.keys() == keywords.keys()
        assert actual_kwargs["native_keyword"] is keywords["native_keyword"]

    with precision_policy_scope(PrecisionPolicy.SOURCE):
        with apply_patches(PoolPlugin.binding_specs()):
            outer = module.pool
            invoke()
            with apply_patches(PoolPlugin.binding_specs()):
                invoke()
            assert module.pool is outer
            invoke()
        assert linen.pool is pooling.pool is original
    assert len(calls) == 3


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("entry", ["linen", "pooling", "nnx"])
def test_pool_native_trace_and_legacy_binding_survive_repeated_activation(
    entry: str, x64: bool
) -> None:
    originals = (linen.pool, pooling.pool, nnx.avg_pool)
    global_x64 = jax.enable_x64.get_global()

    def target(value: Any) -> Any:
        if entry == "nnx":
            return nnx.avg_pool(
                value, window_shape=(2, 2), strides=None, padding="VALID"
            )
        module = linen if entry == "linen" else pooling
        return module.pool(value, 0.0, jax.lax.add, (2, 2), None, "VALID")

    with jax.enable_x64(x64):
        spec = jax.ShapeDtypeStruct((1, 4, 4, 1), np.float32)
        native = jax.make_jaxpr(target)(spec)
        manifest = typed_trace_manifest(native)
        for policy in (
            PrecisionPolicy.SOURCE,
            PrecisionPolicy.LEGACY,
            PrecisionPolicy.SOURCE,
        ):
            with precision_policy_scope(policy), _activate_plugin_worlds():
                activated = jax.make_jaxpr(target)(spec)
                if policy is PrecisionPolicy.SOURCE:
                    assert typed_trace_manifest(activated) == manifest
                    assert all(
                        actual.primitive is expected.primitive
                        for actual, expected in zip(
                            activated.jaxpr.eqns, native.jaxpr.eqns, strict=True
                        )
                    )
                else:
                    primitive = (
                        AvgPoolPlugin._PRIM if entry == "nnx" else PoolPlugin._PRIM
                    )
                    assert len(activated.jaxpr.eqns) == 1
                    assert activated.jaxpr.eqns[0].primitive is primitive
            assert (linen.pool, pooling.pool, nnx.avg_pool) == originals
        assert typed_trace_manifest(native) == manifest
    assert jax.enable_x64.get_global() is global_x64
