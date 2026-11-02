# jax2onnx/plugins/jax/numpy/_common.py

from __future__ import annotations

from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx._compat.jax import AbstractValue, Primitive, ShapedArray


BoundCallable = Callable[..., Any]


def abstract_eval_via_jax(
    original: BoundCallable, *avals: AbstractValue
) -> ShapedArray:
    """Ask JAX about actual typed operands, including their weak scalar types."""
    specs = [
        jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type)
        for aval in avals
    ]
    source_x64 = bool(jax.config.jax_enable_x64)

    def call(*args: Any) -> Any:
        with jax.enable_x64(source_x64):
            return original(*args)

    # Provision existing FP64 formals without changing the source calculation.
    with jax.enable_x64(True):
        out = jax.eval_shape(call, *specs)
    return ShapedArray(out.shape, out.dtype, weak_type=out.weak_type)


def promoted_dtype_via_jax(*avals: AbstractValue) -> np.dtype[Any]:
    """Resolve operand promotion with JAX's weak types and active configuration."""

    def promoted_scalar(*values: Any) -> jax.Array:
        return jnp.asarray(0, dtype=jax.dtypes.result_type(*values))

    dtype: np.dtype[Any] = np.dtype(
        abstract_eval_via_jax(promoted_scalar, *avals).dtype
    )
    return dtype


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
