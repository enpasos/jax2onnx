# jax2onnx/plugins/flax/nnx/_source_linen_declaration.py

"""The unbound, scalar-config Linen declaration retained by native ToNNX.

This recognizes a static native declaration, never a foreign numeric carrier.
Numeric configuration and bound/exotic declarations still need an exact adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
import inspect
from types import CodeType, FunctionType, GetSetDescriptorType
from typing import Any

from flax import linen
from flax.ids import FlaxId
from flax.linen import module as linen_module


def is_static_linen_declaration(
    value: Any, *, failure: Callable[[str], Exception]
) -> bool:
    """Validate raw native storage before allowing GraphDef to retain it.

    No user getter, descriptor, hash, setup or model body is called here.
    The normal native generated hash is checked structurally, not invoked.
    """
    if not isinstance(value, linen.Module):
        return False
    owner_type = type(value)
    for name in ("__getattribute__", "__getattr__"):
        if inspect.getattr_static(owner_type, name) is not inspect.getattr_static(
            linen.Module, name
        ):
            raise failure("Linen declaration custom field access needs an adapter")
    if type(inspect.getattr_static(owner_type, "__dict__")) is not GetSetDescriptorType:
        raise failure("Linen declaration storage descriptor needs an adapter")
    raw = object.__getattribute__(value, "__dict__")
    declared = fields(value)
    names = tuple(field.name for field in declared)
    expected = (set(names) - {"parent"}) | {"_parent_ref", "_id", "_state"}
    if (
        type(raw) is not dict
        or set(raw) != expected
        or not all(field.init for field in declared)
    ):
        raise failure("Linen declaration raw storage schema needs an adapter")
    if (
        raw["_parent_ref"] is not None
        or inspect.getattr_static(owner_type, "scope") is not None
    ):
        raise failure("Bound or scoped Linen declarations need an adapter")
    if raw.get("name") is not None or type(raw["_id"]) is not FlaxId:
        raise failure("Linen declaration native identity metadata needs an adapter")
    state = raw["_state"]
    if type(state) is not linen_module._ModuleInternalState:
        raise failure("Linen declaration native evaluation state needs an adapter")
    state_raw = object.__getattribute__(state, "__dict__")
    if (
        set(state_raw)
        != {
            "in_compact_method",
            "in_setup",
            "setup_called",
            "is_initialized",
            "autoname_cursor",
            "children",
        }
        or any(
            state_raw[name] is not False
            for name in ("in_compact_method", "in_setup", "is_initialized")
        )
        or state_raw["setup_called"] is not linen_module.SetupState.NEW
        or any(
            type(state_raw[name]) is not dict or state_raw[name]
            for name in ("autoname_cursor", "children")
        )
    ):
        raise failure("Only pristine unbound Linen declarations are static")

    def inspect_config(item: Any) -> None:
        # The observed native envelope has scalar Python configuration only.
        # Arrays, nested ordinary containers containing arrays, callables and
        # all other objects are rejected without evaluating any payload/hook.
        if not any(
            type(item) is allowed
            for allowed in (type(None), bool, int, float, complex, str, bytes)
        ):
            raise failure("Linen numeric or nonscalar configuration needs an adapter")

    for field in declared:
        descriptor = inspect.getattr_static(owner_type, field.name, None)
        if field.name == "parent":
            if type(descriptor) is not linen_module.ParentDescriptor:
                raise failure("Unknown Linen parent descriptor needs an adapter")
        else:
            if hasattr(type(descriptor), "__get__"):
                raise failure("Linen configuration field descriptor needs an adapter")
            inspect_config(raw[field.name])
    # Only the native framework wrapper around a generated dataclass hash is
    # eligible. An arbitrary custom hash must not run during eligibility checks.
    hasher = inspect.getattr_static(owner_type, "__hash__")
    native_code = next(
        (
            code
            for code in linen_module._wrap_hash.__code__.co_consts
            if type(code) is CodeType and code.co_name == "wrapped"
        ),
        None,
    )
    if (
        type(hasher) is not FunctionType
        or hasher.__code__ is not native_code
        or hasher.__code__.co_freevars != ("hash_fn",)
    ):
        raise failure("Linen custom hash needs an adapter")
    closure = hasher.__closure__
    original = (
        closure[0].cell_contents if closure is not None and len(closure) == 1 else None
    )
    if any(field.hash is not None or field.compare is not True for field in declared):
        raise failure("Linen custom hash field policy needs an adapter")
    hash_fields = names
    expression = ",".join(f"self.{name}" for name in hash_fields)
    expected_code = next(
        code
        for code in compile(
            f"def __hash__(self):\n return hash(({expression},))\n",
            "<string>",
            "exec",
            dont_inherit=True,
        ).co_consts
        if type(code) is CodeType
    )
    if (
        type(original) is not FunctionType
        or original.__code__.co_code != expected_code.co_code
        or original.__code__.co_names != expected_code.co_names
        or original.__code__.co_consts != expected_code.co_consts
        or original.__code__.co_argcount != 1
        or original.__code__.co_kwonlyargcount != 0
        or original.__code__.co_freevars
        or original.__globals__.get("hash", hash) is not hash
        or original.__builtins__.get("hash") is not hash
    ):
        raise failure("Linen custom hash needs an adapter")
    return True
