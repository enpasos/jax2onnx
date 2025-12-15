# jax2onnx/plugins/jax/numpy/_common.py

from __future__ import annotations

from typing import Any, Callable, cast

from jax.extend.core import Primitive

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec


BoundCallable = Callable[..., Any]


def make_jnp_primitive(name: str) -> Primitive:
    """Create a Primitive for a jax.numpy function and mark it single-result."""
    prim = Primitive(name)
    prim.multiple_results = False
    return prim


def jnp_binding_specs(
    prim: Primitive, func_name: str, *, store_attr: str = "__orig_impl__"
) -> list[AssignSpec | MonkeyPatchSpec]:
    """Return patch specs that bind ``jax.numpy.func_name`` to ``prim``.

    The monkey-patch stores the original callable on the primitive so that
    ``def_impl`` implementations can delegate without recursion concerns.
    """

    attr_name = f"{func_name}_p"
    storage_slot = f"{store_attr}_{func_name}"

    def _make_value(orig: BoundCallable | None) -> BoundCallable:
        if orig is None:
            raise RuntimeError(f"Original jnp.{func_name} not found for patching")
        setattr(prim, storage_slot, orig)

        def _bound(*args: Any, **kwargs: Any) -> Any:
            return prim.bind(*args, **kwargs)

        return _bound

    return [
        AssignSpec(
            target="jax.numpy", attr=attr_name, value=prim, delete_if_missing=True
        ),
        MonkeyPatchSpec(
            target="jax.numpy",
            attr=func_name,
            make_value=_make_value,
            delete_if_missing=False,
        ),
    ]


def get_orig_impl(
    prim: Primitive, func_name: str, store_attr: str = "__orig_impl__"
) -> BoundCallable:
    storage_slot = f"{store_attr}_{func_name}"
    orig = cast(BoundCallable | None, getattr(prim, storage_slot, None))
    if orig is None:
        # Backwards-compatible: older plugins stored without the underscore separator
        # (e.g., "__orig_impl__reshape" instead of "__orig_impl___reshape").
        legacy_slot = f"{store_attr}{func_name}"
        orig = cast(BoundCallable | None, getattr(prim, legacy_slot, None))
    if orig is None:
        raise RuntimeError(f"Original implementation for jnp.{func_name} not captured")
    return orig
