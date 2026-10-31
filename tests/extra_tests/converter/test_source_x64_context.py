# tests/extra_tests/converter/test_source_x64_context.py

"""LEGACY x64 global updates restore the hidden state beneath local overrides."""

from collections.abc import Iterator
from contextlib import nullcontext
import jax
import jax.numpy as jnp
import pytest

from jax2onnx.converter.conversion_api import _force_jax_x64
from jax2onnx.converter.precision_policy import active_precision_policy
from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
    _globals,
    _settings,
)


class BodyFailure(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _restore_configuration() -> Iterator[None]:
    hidden, effective = _globals(), _settings()
    callables = jnp.where, jnp.matmul, jnp.reshape, jnp.add
    policy = active_precision_policy()
    try:
        yield
    finally:
        for name, value in hidden.items():
            jax.config.update(name, value)
        assert (_globals(), _settings()) == (hidden, effective)
        assert (jnp.where, jnp.matmul, jnp.reshape, jnp.add) == callables
        assert active_precision_policy() is policy


@pytest.mark.parametrize("global_x64", [False, True])
@pytest.mark.parametrize("local_x64", [None, False, True])
@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_force_x64_restores_global_beneath_local_override(
    global_x64: bool, local_x64: bool | None, requested: bool, raises: bool
) -> None:
    jax.config.update("jax_enable_x64", global_x64)
    with nullcontext() if local_x64 is None else jax.enable_x64(local_x64):
        before = _globals(), _settings()
        failure = pytest.raises(BodyFailure) if raises else nullcontext()
        with failure:
            with _force_jax_x64(requested):
                assert jax.enable_x64.get_global() is requested
                assert jax.config.x64_enabled is (
                    requested if local_x64 is None else local_x64
                )
                if raises:
                    raise BodyFailure("restore the hidden global on a body error")
        assert (_globals(), _settings()) == before
    assert jax.enable_x64.get_global() is global_x64
    assert jax.config.x64_enabled is global_x64


@pytest.mark.parametrize("global_x64", [False, True])
@pytest.mark.parametrize("local_x64", [None, False, True])
@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("inner", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_nested_force_x64_restores_each_global_frame(
    global_x64: bool, local_x64: bool | None, outer: bool, inner: bool, raises: bool
) -> None:
    jax.config.update("jax_enable_x64", global_x64)
    with nullcontext() if local_x64 is None else jax.enable_x64(local_x64):
        before = _globals(), _settings()
        with _force_jax_x64(outer):
            assert jax.enable_x64.get_global() is outer
            outer_state = _globals(), _settings()
            with pytest.raises(BodyFailure) if raises else nullcontext():
                with _force_jax_x64(inner):
                    assert jax.enable_x64.get_global() is inner
                    assert jax.config.x64_enabled is (
                        inner if local_x64 is None else local_x64
                    )
                    if raises:
                        raise BodyFailure("inner frame failure")
            assert (_globals(), _settings()) == outer_state
        assert (_globals(), _settings()) == before
    assert jax.enable_x64.get_global() is global_x64
    assert jax.config.x64_enabled is global_x64


def test_force_x64_restores_after_local_override_inside_body() -> None:
    jax.config.update("jax_enable_x64", True)
    with jax.enable_x64(False):
        with pytest.raises(BodyFailure):
            with _force_jax_x64(False):
                with jax.enable_x64(True):
                    assert jax.enable_x64.get_global() is False
                    assert jax.config.x64_enabled is True
                    raise BodyFailure("unwind local and global frames")
        assert jax.enable_x64.get_global() is True
        assert jax.config.x64_enabled is False
    assert jax.config.x64_enabled is True


@pytest.mark.parametrize("global_x64", [False, True])
@pytest.mark.parametrize("local_x64", [None, False, True])
@pytest.mark.parametrize("requested", [False, True])
def test_force_x64_restores_after_entry_update_raises(
    global_x64: bool,
    local_x64: bool | None,
    requested: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jax.config.update("jax_enable_x64", global_x64)
    update = jax.config.update
    calls = []

    def fail_once(name: str, value: bool) -> None:
        update(name, value)
        calls.append((name, value))
        if len(calls) == 1:
            raise BodyFailure("global hook failed after updating its value")

    with nullcontext() if local_x64 is None else jax.enable_x64(local_x64):
        before = _globals(), _settings()
        monkeypatch.setattr(jax.config, "update", fail_once)
        with pytest.raises(BodyFailure, match="global hook"):
            with _force_jax_x64(requested):
                pytest.fail("A failed entry update cannot enter the body")
        assert calls == [("jax_enable_x64", requested), ("jax_enable_x64", global_x64)]
        assert (_globals(), _settings()) == before
    assert jax.enable_x64.get_global() is global_x64


@pytest.mark.parametrize("reverse", [False, True])
def test_original_precompiled_cache_cases_with_hidden_global_true(
    reverse: bool,
) -> None:
    from tests.extra_tests.converter.test_source_jit_admission import (
        test_precompiled_callable_keeps_native_legacy_source_caches,
    )

    jax.config.update("jax_enable_x64", True)
    with jax.enable_x64(False):
        before = _globals(), _settings()
        test_precompiled_callable_keeps_native_legacy_source_caches(reverse)
        assert (_globals(), _settings()) == before
        assert jax.enable_x64.get_global() is True
