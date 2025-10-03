from __future__ import annotations

import inspect
import types
from collections.abc import Iterable as IterableABC
from typing import Any

import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PLUGIN_REGISTRY2


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
    """Lower every equation in ``jaxpr`` using the registered plugins."""
    for inner_eqn in getattr(jaxpr, "eqns", ()):
        prim = inner_eqn.primitive.name
        plugin = PLUGIN_REGISTRY2.get(prim)
        if plugin is None:
            raise NotImplementedError(
                f"[control_flow] No plugins registered for primitive '{prim}'"
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
    if hasattr(parent_ctx, "_keep_function_float32"):
        child_ctx._keep_function_float32 = getattr(
            parent_ctx, "_keep_function_float32", False
        )

    # Inherit known symbolic dimension origins so nested graphs can resolve them.
    if hasattr(parent_ctx, "_sym_origin"):
        child_ctx._sym_origin = dict(getattr(parent_ctx, "_sym_origin", {}))
    if hasattr(parent_ctx, "_sym_origin_str"):
        child_ctx._sym_origin_str = dict(getattr(parent_ctx, "_sym_origin_str", {}))

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


def relax_value_to_rank_only(val: ir.Value | None) -> None:
    if val is None or not isinstance(val, ir.Value):
        return
    shape_obj = getattr(val, "shape", None)
    dims = getattr(shape_obj, "dims", None)
    if dims is None and shape_obj is not None:
        try:
            dims = list(shape_obj) if isinstance(shape_obj, IterableABC) else None
        except Exception:
            dims = None
    if dims is None:
        tensor_type = getattr(val, "type", None)
        if isinstance(tensor_type, ir.TensorType):
            shape_obj = getattr(tensor_type, "shape", None)
            dims = getattr(shape_obj, "dims", None)
            if dims is None and shape_obj is not None:
                try:
                    dims = (
                        list(shape_obj) if isinstance(shape_obj, IterableABC) else None
                    )
                except Exception:
                    dims = None
    if not dims or len(dims) == 0:
        return
    if all(dim is None for dim in dims):
        return
    rank_only = ir.Shape(tuple(None for _ in dims))
    try:
        val.shape = rank_only
    except Exception:
        pass
    tensor_type = getattr(val, "type", None)
    if isinstance(tensor_type, ir.TensorType):
        dtype = getattr(tensor_type, "dtype", getattr(tensor_type, "elem_type", None))
        try:
            val.type = ir.TensorType(dtype, rank_only)
        except Exception:
            val.type = ir.TensorType(dtype)
