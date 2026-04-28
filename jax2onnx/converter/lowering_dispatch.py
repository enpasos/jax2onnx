# jax2onnx/converter/lowering_dispatch.py

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

from .typing_support import (
    FunctionLowering,
    LoweringContextProtocol,
    PrimitiveLowering,
)


def _lower_accepts_params(lower: Any) -> bool:
    try:
        return "params" in inspect.signature(lower).parameters
    except (TypeError, ValueError):
        return False


def _default_converter_facade(ctx: LoweringContextProtocol) -> SimpleNamespace:
    return SimpleNamespace(builder=getattr(ctx, "builder", None), ctx=ctx)


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
        converter_facade = converter or _default_converter_facade(ctx)
        handler = plugin.get_handler(converter_facade)
        return handler(
            converter_facade,
            eqn,
            getattr(eqn, "params", {}) or {},
        )

    raise NotImplementedError(
        f"[{source}] Unsupported plugin type for primitive '{primitive_name}'"
    )
