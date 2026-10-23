# tests/extra_tests/framework/test_native_context_compat.py

from __future__ import annotations

from typing import Any

import pytest
from pytest import MonkeyPatch

from jax2onnx._compat import jax as compat

_HAS_NO_COLLECTIVES = (
    compat._native_context_protocol() == "interned-layout-no-collectives"
)


@pytest.mark.parametrize("raises", [False, True])
def test_recorded_native_context_restores_caller_on_success_and_exception(
    raises: bool,
) -> None:
    configurations = [(compat._native_config.compute_on_context_manager, "device")]
    if _HAS_NO_COLLECTIVES:
        configurations.append((compat._native_config.no_collectives, frozenset()))
    previous = [
        (configuration, configuration.get_local())
        for configuration, _ in configurations
    ]
    try:
        for configuration, value in configurations:
            configuration.set_local(value)
        recorded = compat.current_native_eqn_context()
        # The older mutable manager deliberately refuses conflicting compute
        # scopes; its no-conflict path still owns caller restoration.
        if compat._native_context_protocol() != "mutable":
            compat._native_config.compute_on_context_manager.set_local("device_host")
        if _HAS_NO_COLLECTIVES:
            compat._native_config.no_collectives.set_local(frozenset(("caller",)))
        before = compat.native_context_snapshot()

        def exercise() -> None:
            with compat.recorded_native_context(recorded):
                assert compat.current_native_eqn_context().compute_type == "device"
                if _HAS_NO_COLLECTIVES:
                    assert compat._native_config.no_collectives.value == frozenset()
                if raises:
                    raise ValueError("inference failed")

        if raises:
            with pytest.raises(ValueError, match="inference failed"):
                exercise()
        else:
            exercise()
        compat.require_native_context_restored(before)
    finally:
        for configuration, value in reversed(previous):
            configuration.set_local(value)


@pytest.mark.parametrize(
    "fault",
    ["compute"]
    + (
        ["no-collectives-nonempty", "no-collectives-malformed"]
        if _HAS_NO_COLLECTIVES
        else []
    ),
)
def test_native_context_rejects_unsupported_fields_before_manager(
    monkeypatch: MonkeyPatch, fault: str
) -> None:
    original = compat.current_native_eqn_context()
    context = object.__new__(type(original))
    for field in type(original).__slots__:
        if field != "__weakref__" and hasattr(original, field):
            object.__setattr__(context, field, getattr(original, field))
    field, value = {
        "compute": ("compute_type", object()),
        "no-collectives-nonempty": ("no_collectives", frozenset(("unowned",))),
        "no-collectives-malformed": ("no_collectives", set()),
    }[fault]
    object.__setattr__(context, field, value)

    def manager(_self: Any) -> Any:
        raise AssertionError("unsupported context reached its native manager")

    monkeypatch.setattr(type(original), "manager", property(manager))
    before = compat.native_context_snapshot()
    with pytest.raises(TypeError):
        with compat.recorded_native_context(context):
            raise AssertionError("unsupported context entered its native manager")
    compat.require_native_context_restored(before)


@pytest.mark.parametrize(
    "role", ["metadata"] + (["no-collectives"] if _HAS_NO_COLLECTIVES else [])
)
def test_native_context_snapshot_preserves_unset_and_equal_effective_value(
    role: str,
) -> None:
    configuration = dict(compat._native_context_configurations())[role]
    previous = configuration.get_local()
    unset = compat._native_xla_metadata.config_ext.unset
    try:
        configuration.set_local(unset)
        effective = configuration.value
        before = compat.native_context_snapshot()
        configuration.set_local(effective)
        with pytest.raises(RuntimeError, match="restored"):
            compat.require_native_context_restored(before)
        configuration.set_local(unset)
        compat.require_native_context_restored(before)
    finally:
        configuration.set_local(previous)


def test_native_context_protocol_rejects_unknown_schema(
    monkeypatch: MonkeyPatch,
) -> None:
    context_type = compat._native_core.JaxprEqnContext
    monkeypatch.setattr(context_type, "__slots__", (*context_type.__slots__, "unowned"))
    with pytest.raises(TypeError, match="schema needs an exact adapter"):
        compat._native_context_protocol()
