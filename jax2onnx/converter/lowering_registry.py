# jax2onnx/converter/lowering_registry.py

"""Prepare plugin call conventions once, at the registry boundary.

Registry reads expose the original plugins. Equation lowering uses the prepared
``(ctx, eqn) -> None`` callable. First-party registration supplies that typed
contract explicitly; only legacy assignments inspect call conventions and
interpret returned values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, replace
from inspect import signature as _inspect_signature
from types import SimpleNamespace
from typing import TypeAlias, cast
from weakref import WeakValueDictionary

from jax2onnx._compat.jax import JaxprEqn

from .output_binding import bind_returned_lowering_values
from .typing_support import (
    CanonicalLowering,
    LoweringContextProtocol,
    PrimitiveLowering,
)

LegacyCall: TypeAlias = Callable[
    [LoweringContextProtocol, JaxprEqn, object | None], object
]
ConverterBinder: TypeAlias = Callable[[object], CanonicalLowering]


def make_converter_facade(ctx: LoweringContextProtocol) -> SimpleNamespace:
    """Expose the context and builder to legacy function handlers."""
    return SimpleNamespace(builder=ctx.builder, ctx=ctx)


@dataclass(frozen=True)
class RegisteredLowering:
    plugin: object
    lower: CanonicalLowering
    bind_converter: ConverterBinder | None = None

    def with_converter(self, converter: object) -> RegisteredLowering:
        if self.bind_converter is None:
            return self
        return replace(self, lower=self.bind_converter(converter))


def _primitive_call(lower: object) -> LegacyCall:
    """Inspect only the legacy params convention, once at assignment."""
    lower_fn = cast(Callable[..., object], lower)
    try:
        signature = _inspect_signature(lower_fn)
    except (TypeError, ValueError):
        signature = None
    accepts_params = signature is not None and "params" in signature.parameters
    if accepts_params:

        def call(
            ctx: LoweringContextProtocol, eqn: JaxprEqn, converter: object | None
        ) -> object:
            return lower_fn(ctx, eqn, getattr(eqn, "params", None))

    else:

        def call(
            ctx: LoweringContextProtocol, eqn: JaxprEqn, converter: object | None
        ) -> object:
            return lower_fn(ctx, eqn)

    return call


def _function_call(get_handler: object) -> LegacyCall:
    handler_factory = cast(Callable[[object], Callable[..., object]], get_handler)

    def call(
        ctx: LoweringContextProtocol, eqn: JaxprEqn, converter: object | None
    ) -> object:
        facade = converter or make_converter_facade(ctx)
        handler = handler_factory(facade)
        return handler(facade, eqn, getattr(eqn, "params", {}) or {})

    return call


def _unsupported_call(primitive_name: str, source: str) -> LegacyCall:
    def call(
        ctx: LoweringContextProtocol, eqn: JaxprEqn, converter: object | None
    ) -> object:
        raise NotImplementedError(
            f"[{source}] Unsupported plugin type for primitive '{primitive_name}'"
        )

    return call


def prepare_legacy_call(
    plugin: object, *, primitive_name: str, source: str
) -> LegacyCall:
    """Prepare a raw compatibility call without altering its returned result."""
    lower = getattr(plugin, "lower", None)
    if callable(lower):
        return _primitive_call(lower)
    get_handler = getattr(plugin, "get_handler", None)
    if callable(get_handler):
        return _function_call(get_handler)
    return _unsupported_call(primitive_name, source)


def _bind_legacy_call(
    call: LegacyCall, primitive_name: str, converter: object | None = None
) -> CanonicalLowering:
    def lower(ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        result = call(ctx, eqn, converter)
        bind_returned_lowering_values(ctx, eqn, result, primitive_name=primitive_name)

    return lower


def prepare_lowering(
    plugin: object, *, primitive_name: str, source: str = "plugin_registry"
) -> RegisteredLowering:
    """Adapt a raw legacy plugin without inferring a canonical contract.

    Unsupported entries remain insertable for metadata/collision inspection.
    Their prepared callable reports the error when lowering actually uses it.
    """
    lower = getattr(plugin, "lower", None)
    if callable(lower):
        call = _primitive_call(lower)
        return RegisteredLowering(plugin, _bind_legacy_call(call, primitive_name))
    get_handler = getattr(plugin, "get_handler", None)
    if callable(get_handler):
        call = _function_call(get_handler)

        def bind_converter(converter: object) -> CanonicalLowering:
            return _bind_legacy_call(call, primitive_name, converter)

        return RegisteredLowering(
            plugin, _bind_legacy_call(call, primitive_name), bind_converter
        )
    return RegisteredLowering(
        plugin,
        _bind_legacy_call(_unsupported_call(primitive_name, source), primitive_name),
    )


class PluginRegistry(MutableMapping[str, object]):
    """Original plugins for callers, prepared canonical lowerings for the core."""

    def __init__(self, plugins: Mapping[str, object] | None = None) -> None:
        self._entries: dict[str, RegisteredLowering] = {}
        # Remember explicit registration across aliases and temporary replacement.
        # Weak values avoid retaining removed function targets and their captures.
        self._canonical_plugins: WeakValueDictionary[int, PrimitiveLowering] = (
            WeakValueDictionary()
        )
        if plugins is not None:
            self.update(plugins)

    def __getitem__(self, name: str) -> object:
        return self._entries[name].plugin

    def __setitem__(self, name: str, plugin: object) -> None:
        canonical = self._canonical_plugins.get(id(plugin))
        if canonical is not None and canonical is plugin:
            self._entries[name] = RegisteredLowering(canonical, canonical.lower)
        else:
            self._entries[name] = prepare_lowering(plugin, primitive_name=name)

    def register_canonical(self, name: str, plugin: PrimitiveLowering) -> None:
        """Register the typed binding contract directly, without reflection."""
        try:
            self._canonical_plugins[id(plugin)] = plugin
        except TypeError:
            # Weak references are optional; the explicit contract is not.
            # A later raw assignment can still use the legacy boundary.
            pass
        self._entries[name] = RegisteredLowering(plugin, plugin.lower)

    def __delitem__(self, name: str) -> None:
        del self._entries[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def copy(self) -> dict[str, object]:
        """Keep dictionary snapshots available to existing registry callers."""
        return dict(self.items())

    def popitem(self) -> tuple[str, object]:
        """Remove the last inserted entry, matching the original dict registry."""
        name, entry = self._entries.popitem()
        return name, entry.plugin

    def prepared_lowerings(
        self, converter: object | None = None
    ) -> Mapping[str, RegisteredLowering]:
        if converter is None:
            return self._entries
        return {
            name: entry.with_converter(converter)
            for name, entry in self._entries.items()
        }


def prepared_lowerings(
    registry: Mapping[str, object], converter: object | None = None
) -> Mapping[str, RegisteredLowering]:
    """Adapt ordinary mappings once at entry; registered plugins need no inspection."""
    prepared = (
        registry if isinstance(registry, PluginRegistry) else PluginRegistry(registry)
    )
    return prepared.prepared_lowerings(converter)
