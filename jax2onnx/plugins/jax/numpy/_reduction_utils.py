# jax2onnx/plugins/jax/numpy/_reduction_utils.py

from __future__ import annotations

import inspect
from collections.abc import Sequence

import jax
from jax import core
from jax.extend.core import Primitive
from jax.interpreters import batching
import numpy as np

from jax2onnx.plugins.jax.numpy._common import get_orig_impl


def normalize_axis_to_axes(
    axis: int | Sequence[int] | None,
) -> tuple[tuple[int, ...] | None, bool]:
    """Convert jnp-style ``axis`` into primitive params ``axes`` + flag."""
    if axis is None:
        return None, False
    if isinstance(axis, Sequence) and not isinstance(axis, (str, bytes)):
        return tuple(int(ax) for ax in axis), True
    return (int(axis),), False


def axis_arg_from_params(
    axes: tuple[int, ...] | None, axes_is_tuple: bool
) -> int | tuple[int, ...] | None:
    """Convert primitive params back to jnp-style axis argument."""
    if axes is None:
        return None
    axis_vals = tuple(int(ax) for ax in axes)
    return axis_vals if axes_is_tuple else axis_vals[0]


def abstract_eval_via_orig_reduction(
    prim: Primitive,
    func_name: str,
    x: core.AbstractValue,
    *,
    axes: tuple[int, ...] | None = None,
    axes_is_tuple: bool = False,
    dtype: np.dtype[np.generic] | type | None = None,
    keepdims: bool = False,
    promote_integers: bool = True,
) -> core.ShapedArray:
    """Mirror JAX reduction semantics by delegating to original jnp callable."""
    shape = tuple(getattr(x, "shape", ()))
    in_dtype = np.dtype(getattr(x, "dtype", np.float32))
    shape_dtype = jax.ShapeDtypeStruct(shape, in_dtype)
    axis_arg = axis_arg_from_params(axes, axes_is_tuple)

    orig = get_orig_impl(prim, func_name)
    kwargs: dict[str, object] = {
        "axis": axis_arg,
        "keepdims": keepdims,
    }
    sig = inspect.signature(orig)
    if "dtype" in sig.parameters:
        kwargs["dtype"] = dtype
    if "promote_integers" in sig.parameters:
        kwargs["promote_integers"] = promote_integers

    out = jax.eval_shape(lambda v: orig(v, **kwargs), shape_dtype)
    out_shape = tuple(getattr(out, "shape", ()))
    out_dtype = np.dtype(getattr(out, "dtype", in_dtype))
    return core.ShapedArray(out_shape, out_dtype)


def register_reduction_batch_rule(prim: Primitive, lax_prim: Primitive) -> None:
    """Attach a reduction batcher for primitives with ``axes`` params."""

    del lax_prim  # Signature kept for call-site stability.

    def _batch_rule(
        batched_args: tuple[jax.Array, ...],
        batch_dims: tuple[int | type(batching.not_mapped), ...],
        *,
        axes: tuple[int, ...] | None = None,
        axes_is_tuple: bool = False,
        **params: object,
    ) -> tuple[jax.Array, int]:
        (operand,), (bdim,) = batched_args, batch_dims
        axis_size = operand.shape[bdim]
        operand = batching.bdim_at_front(operand, bdim, axis_size)

        slice_rank = operand.ndim - 1
        if axes is None:
            axes_full = tuple(range(1, operand.ndim))
        else:
            axes_norm = tuple(int(ax) % slice_rank for ax in axes)
            axes_full = tuple(ax + 1 for ax in axes_norm)

        out = prim.bind(
            operand,
            axes=axes_full,
            axes_is_tuple=True,
            **params,
        )
        del axes_is_tuple  # We always preserve tuple semantics in batched mode.
        return out, 0

    batching.primitive_batchers[prim] = _batch_rule
