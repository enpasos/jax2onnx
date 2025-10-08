# jax2onnx/plugins/_ir_shapes.py

from __future__ import annotations
from collections.abc import Iterable as IterableABC, Sequence as SequenceABC
from typing import Tuple, Union, cast

import numpy as np
import onnx_ir as ir


DimValue = Union[int, ir.SymbolicDim, None]


def _stamp_type_and_shape(v: ir.Value, dims: SequenceABC[DimValue]) -> None:
    """Ensure both meta and tensor-type shape carry symbolic names like 'B'."""
    ir_dims_list: list[DimValue] = [_to_ir_dim_for_shape(d) for d in dims]
    sh = ir.Shape(ir_dims_list)
    v.shape = sh
    dtype = v.dtype
    if dtype is not None:
        v.type = ir.TensorType(dtype)


def _prod(xs: IterableABC[int]) -> int:
    p = 1
    for v in xs:
        p *= int(v)
    return int(p)


def _as_ir_dim_label(dim: object) -> Union[str, int, None]:
    """
    Best-effort extraction of a printable dim label from onnx_ir dims.
    Handles:
      - ir.SymbolicDim('B') -> 'B'
      - objects with .param/.name/.symbol/.value
      - plain ints/strings/None
    """
    if dim is None:
        return None
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    if isinstance(dim, ir.SymbolicDim):
        value = cast(object, getattr(dim, "value", None))
        if isinstance(value, (int, np.integer)):
            return int(value)
        text = str(dim)
        return text if text else None
    if isinstance(dim, str):
        return dim
    value_attr = cast(object, getattr(dim, "value", None))
    if isinstance(value_attr, (int, np.integer)):
        return int(value_attr)
    if isinstance(value_attr, str):
        return value_attr
    text = str(dim)
    return text if text else None


def _to_ir_dim_for_shape(dim: object) -> DimValue:
    if dim is None:
        return None
    if isinstance(dim, ir.SymbolicDim):
        return dim
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    if isinstance(dim, str):
        return ir.SymbolicDim(dim)
    value_attr = cast(object, getattr(dim, "value", None))
    if isinstance(value_attr, (int, np.integer)):
        return int(value_attr)
    if isinstance(value_attr, str):
        return ir.SymbolicDim(value_attr)
    return None


def _is_static_int(d) -> bool:
    return isinstance(d, (int, np.integer)) and int(d) >= 0


def _dim_label_from_value_or_aval(
    val: ir.Value, aval_shape: SequenceABC[object], i: int
):
    """Return a readable dim label, falling back to aval metadata when needed."""
    shp = val.shape
    if shp is not None:
        shp_obj = cast(object, shp)
        if isinstance(shp_obj, ir.Shape):
            dims: Tuple[object, ...] = tuple(shp_obj.dims)
        else:
            dims = tuple(cast(SequenceABC[object], shp_obj))
        if i < len(dims):
            label = _as_ir_dim_label(dims[i])
            if label is not None:
                return label
    if i < len(aval_shape):
        label = _as_ir_dim_label(aval_shape[i])
        if label is not None:
            return label
    return None


def _ensure_value_info(ctx, v: ir.Value | None):
    if v is None:
        return
    try:
        lst = getattr(ctx, "_value_info", None) or getattr(ctx, "_value_infos", None)
        if lst is None:
            return
        if all(getattr(x, "name", None) != v.name for x in lst):
            lst.append(v)
    except Exception:
        pass


def is_shape_all_unknown(shp) -> bool:
    """True if shape is None or all dims are anonymous (None/unknown)."""
    if shp is None:
        return True
    shp_obj = cast(object, shp)
    if isinstance(shp_obj, ir.Shape):
        dims: Tuple[object, ...] = tuple(shp_obj.dims)
    elif isinstance(shp_obj, SequenceABC) and not isinstance(shp_obj, (str, bytes)):
        dims = tuple(cast(SequenceABC[object], shp_obj))
    else:
        return True
    for d in dims:
        if _as_ir_dim_label(d) is not None:
            return False
    return True
