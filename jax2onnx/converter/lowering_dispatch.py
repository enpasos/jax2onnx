# jax2onnx/converter/lowering_dispatch.py

from __future__ import annotations

import inspect
from contextlib import nullcontext
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from .output_binding import finalize_eqn_lowering_outputs
from .typing_support import (
    FunctionLowering,
    LoweringContextProtocol,
    PrimitiveLowering,
)

_LOWER_SIGNATURE_CACHE: dict[object, bool] = {}


def _lower_accepts_params(lower: Any) -> bool:
    cache_key = getattr(lower, "__func__", lower)
    try:
        cached = _LOWER_SIGNATURE_CACHE.get(cache_key)
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    try:
        accepts_params = "params" in inspect.signature(lower).parameters
    except (TypeError, ValueError):
        accepts_params = False
    try:
        _LOWER_SIGNATURE_CACHE[cache_key] = accepts_params
    except TypeError:
        pass
    return accepts_params


def make_converter_facade(ctx: LoweringContextProtocol) -> SimpleNamespace:
    """Build the minimal converter facade expected by function lowerings."""
    return SimpleNamespace(builder=getattr(ctx, "builder", None), ctx=ctx)


def get_registered_lowering_plugin(
    registry: Mapping[str, object],
    primitive_name: str,
    *,
    source: str,
    detail: str | None = None,
) -> object:
    """Return the registered plugin for ``primitive_name`` or raise a source label."""
    plugin = registry.get(primitive_name)
    if plugin is not None:
        return plugin

    detail_text = f" {detail}" if detail else ""
    raise NotImplementedError(
        f"[{source}] No plugins registered for primitive "
        f"'{primitive_name}'{detail_text}"
    )


def identify_lowering_plugin(
    plugin: object, primitive_name: str
) -> tuple[str, str | None]:
    """Return a stable plugin identifier and optional source line for metadata."""
    if plugin is None:
        return primitive_name, None

    try:
        if isinstance(plugin, PrimitiveLowering):
            lower = plugin.lower
            func_name = getattr(lower, "__name__", "lower")
            identifier = (
                f"{type(plugin).__module__}.{type(plugin).__name__}.{func_name}"
            )
            try:
                _, start_line = inspect.getsourcelines(lower)
            except (OSError, TypeError):
                return identifier, None
            return identifier, str(start_line)

        if isinstance(plugin, FunctionLowering):
            return (
                f"{type(plugin).__module__}.{type(plugin).__name__}.get_handler",
                None,
            )

        if hasattr(plugin, "__class__"):
            return f"{type(plugin).__module__}.{type(plugin).__name__}", None
    except Exception:
        pass

    return primitive_name, None


def dispatch_plugin_lowering(
    plugin: object,
    *,
    ctx: LoweringContextProtocol,
    eqn: object,
    primitive_name: str,
    source: str,
    converter: object | None = None,
) -> object:
    """Invoke a registered lowering plugin for a single JAXPR equation."""
    if isinstance(plugin, PrimitiveLowering):
        lower = plugin.lower
        if _lower_accepts_params(lower):
            return lower(ctx, eqn, getattr(eqn, "params", None))
        return lower(ctx, eqn)

    if isinstance(plugin, FunctionLowering):
        converter_facade = converter or make_converter_facade(ctx)
        handler = plugin.get_handler(converter_facade)
        return handler(
            converter_facade,
            eqn,
            getattr(eqn, "params", {}) or {},
        )

    raise NotImplementedError(
        f"[{source}] Unsupported plugin type for primitive '{primitive_name}'"
    )


def lower_equation_with_plugin(
    plugin: object,
    *,
    ctx: LoweringContextProtocol,
    eqn: object,
    primitive_name: str,
    eqn_index: int,
    source: str,
    converter: object | None = None,
) -> object:
    """Dispatch a plugin and verify the equation output contract."""
    lowering_result = dispatch_plugin_lowering(
        plugin,
        ctx=ctx,
        eqn=eqn,
        primitive_name=primitive_name,
        source=source,
        converter=converter,
    )
    finalize_eqn_lowering_outputs(
        ctx,
        eqn,
        lowering_result,
        primitive_name=primitive_name,
        eqn_index=eqn_index,
    )
    return lowering_result


def lower_jaxpr_with_plugins(
    *,
    ctx: LoweringContextProtocol,
    jaxpr: object,
    registry: Mapping[str, object],
    source: str,
    converter: object | None = None,
    missing_plugin_detail: str | None = None,
) -> None:
    """Lower every equation in a JAXPR-like object with registered plugins."""
    const_folder = getattr(ctx, "_const_folder", None)
    producer_scope = getattr(const_folder, "producer_scope", None)
    scope = producer_scope(jaxpr) if callable(producer_scope) else nullcontext()
    with scope:
        for eqn_index, eqn in enumerate(getattr(jaxpr, "eqns", ())):
            primitive_name = eqn.primitive.name
            plugin = get_registered_lowering_plugin(
                registry,
                primitive_name,
                source=source,
                detail=missing_plugin_detail,
            )
            lower_equation_with_plugin(
                plugin,
                ctx=ctx,
                eqn=eqn,
                primitive_name=primitive_name,
                eqn_index=eqn_index,
                source=source,
                converter=converter,
            )
