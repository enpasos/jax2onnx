# jax2onnx/plugins/_utils.py

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import onnx_ir as ir

if TYPE_CHECKING:
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


def cast_param_like(
    ctx: "IRBuildContext",
    param: ir.Value,
    like: ir.Value,
    name_hint: str = "param_cast",
) -> ir.Value:
    """
    If `param`'s dtype differs from `like`'s dtype, insert a CastLike node to cast
    `param` to `like`'s dtype. Never casts `like`. Returns the value to use downstream.

    Safe to call when dtypes are unknown (no-op). Preserves shape.
    """
    p_ty = getattr(param, "type", None)
    l_ty = getattr(like, "type", None)
    p_dt = getattr(p_ty, "dtype", None)
    l_dt = getattr(l_ty, "dtype", None)
    if p_dt is None or l_dt is None or p_dt == l_dt:
        return param

    out = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(l_dt),
        shape=param.shape,
    )
    ctx.add_node(
        ir.Node(
            op_type="CastLike",
            domain="",
            inputs=[param, like],  # cast `param` to dtype of `like`
            outputs=[out],
            name=ctx.fresh_name("CastLike"),
        )
    )
    return out


# --- NEW: inline reshape for constant parameters (no runtime node) ---
def inline_reshape_initializer(
    ctx, val: ir.Value, new_shape: tuple[int, ...], name_hint: str
) -> ir.Value:
    """
    If `val` is a constant initializer, create a new initializer with the data
    reshaped to `new_shape` and return it. Otherwise, return `val` unchanged.
    """
    arr = getattr(val, "const_value", None)
    if arr is None:
        return val  # not a constant → caller must insert a Reshape node

    np_arr = np.asarray(arr).reshape(new_shape)
    reshaped = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=val.type,  # keep dtype
        shape=ir.Shape(tuple(int(s) for s in new_shape)),
        const_value=ir.tensor(np_arr),
    )
    ctx._initializers.append(reshaped)
    return reshaped
