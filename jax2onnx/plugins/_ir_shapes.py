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
        value = dim.value
        if isinstance(value, str):
            return value
        text = str(dim)
        return text if text else None
    if isinstance(dim, str):
        return dim
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
    text = str(dim)
    return ir.SymbolicDim(text) if text else None


def _is_static_int(d: object) -> bool:
    return isinstance(d, (int, np.integer)) and int(d) >= 0


def _dim_label_from_value_or_aval(
    val: ir.Value, aval_shape: SequenceABC[object], i: int
) -> Union[str, int, None]:
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


def _ensure_value_metadata(ctx: object, v: ir.Value | None) -> None:
    del ctx  # parameter retained for API compatibility
    if v is None:
        return
    shape_obj = v.shape
    dims: Tuple[object, ...] = ()
    if isinstance(shape_obj, ir.Shape):
        dims = tuple(shape_obj.dims)
    else:
        seq_like = cast(object, shape_obj)
        if isinstance(seq_like, SequenceABC) and not isinstance(seq_like, (str, bytes)):
            dims = tuple(cast(SequenceABC[object], seq_like))
            v.shape = ir.Shape(tuple(_to_ir_dim_for_shape(d) for d in dims))

    if v.type is None and v.dtype is not None:
        v.type = ir.TensorType(v.dtype)


def is_shape_all_unknown(shp: object) -> bool:
    """True if shape is None or all dims are anonymous (None/unknown)."""
    if shp is None:
        return True
    if isinstance(shp, ir.Shape):
        dims: Tuple[object, ...] = tuple(shp.dims)
    elif isinstance(shp, SequenceABC) and not isinstance(shp, (str, bytes)):
        dims = tuple(cast(SequenceABC[object], shp))
    else:
        return True
    for d in dims:
        if _as_ir_dim_label(d) is not None:
            return False
    return True
