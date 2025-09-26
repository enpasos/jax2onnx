from __future__ import annotations

import inspect
import types
from typing import Any

from jax2onnx.plugins2.plugin_system import PLUGIN_REGISTRY2


def _call_plugin_lower(plugin: Any, ctx: Any, eqn: Any) -> None:
    """Invoke a plugin's lowering helper, forwarding params when supported."""
    lower_fn = getattr(plugin, "lower", None)
    if lower_fn is None:
        raise NotImplementedError(f"Plugin for '{plugin}' lacks a lower() method.")
    try:
        sig = inspect.signature(lower_fn)
        if "params" in sig.parameters:
            return lower_fn(ctx, eqn, getattr(eqn, "params", None))
    except (ValueError, TypeError):
        pass
    return lower_fn(ctx, eqn)


def lower_jaxpr_eqns(ctx: Any, jaxpr: Any) -> None:
    """Lower every equation in ``jaxpr`` using the registered plugins2."""
    for inner_eqn in getattr(jaxpr, "eqns", ()):
        prim = inner_eqn.primitive.name
        plugin = PLUGIN_REGISTRY2.get(prim)
        if plugin is None:
            raise NotImplementedError(
                f"[control_flow] No plugins2 registered for primitive '{prim}'"
            )
        _call_plugin_lower(plugin, ctx, inner_eqn)


def make_subgraph_context(parent_ctx: Any, *, prefix: str) -> Any:
    """Create a child IR context suitable for Loop/If subgraphs."""
    child_ctx = type(parent_ctx)(
        opset=getattr(parent_ctx.builder, "opset", 21),
        enable_double_precision=getattr(
            parent_ctx.builder, "enable_double_precision", False
        ),
        input_specs=[],
    )
    child_ctx._function_mode = True
    child_ctx._inside_function_scope = True

    # Inherit known symbolic dimension origins so nested graphs can resolve them.
    if hasattr(parent_ctx, "_sym_origin"):
        child_ctx._sym_origin = dict(getattr(parent_ctx, "_sym_origin", {}))
    if hasattr(parent_ctx, "_sym_origin_str"):
        child_ctx._sym_origin_str = dict(
            getattr(parent_ctx, "_sym_origin_str", {})
        )

    # Propagate optional knobs the parent may expose.
    for attr_name in ("loosen_internal_shapes", "_function_registry"):
        if hasattr(parent_ctx, attr_name):
            setattr(child_ctx, attr_name, getattr(parent_ctx, attr_name))

    # Prefix all fresh names so nested graphs remain unique.
    prefix_base = parent_ctx.fresh_name(prefix)
    orig_ctx_fresh = child_ctx.fresh_name
    setattr(
        child_ctx,
        "fresh_name",
        types.MethodType(
            lambda self, base, _orig=orig_ctx_fresh, _pref=prefix_base: _orig(
                f"{_pref}/{base}"
            ),
            child_ctx,
        ),
    )
    orig_builder_fresh = child_ctx.builder.fresh_name
    setattr(
        child_ctx.builder,
        "fresh_name",
        types.MethodType(
            lambda self, base, _orig=orig_builder_fresh, _pref=prefix_base: _orig(
                f"{_pref}/{base}"
            ),
            child_ctx.builder,
        ),
    )
    return child_ctx
