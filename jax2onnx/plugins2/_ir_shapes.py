from __future__ import annotations
from typing import Iterable
import re
import numpy as np
import onnx_ir as ir


def _stamp_type_and_shape(v: ir.Value, dims):
    """Ensure both meta and tensor-type shape carry symbolic names like 'B'."""
    ir_dims = tuple(_to_ir_dim_for_shape(d) for d in dims)
    sh = ir.Shape(ir_dims)
    v.shape = sh
    try:
        if isinstance(getattr(v, "type", None), ir.TensorType):
            v.type = ir.TensorType(v.type.dtype, sh)
    except Exception:
        pass


def _prod(xs: Iterable[int]) -> int:
    p = 1
    for v in xs:
        p *= int(v)
    return int(p)


def _as_ir_dim_label(d):
    """
    Best-effort extraction of a printable dim label from onnx_ir dims.
    Handles:
      - ir.SymbolicDim('B') -> 'B'
      - objects with .param/.name/.symbol/.value
      - plain ints/strings/None
    """
    try:
        if isinstance(d, ir.SymbolicDim):  # type: ignore[attr-defined]
            for attr in ("param", "name", "symbol", "label"):
                v = getattr(d, attr, None)
                if v:
                    return str(v)
            s = repr(d)
            m = re.search(r"SymbolicDim\(['\"]?([A-Za-z0-9_]+)['\"]?\)", s)
            if m:
                return m.group(1)
            s = str(d)
            if s and s.isidentifier():
                return s
    except Exception:
        pass
    try:
        if hasattr(d, "param") and getattr(d, "param", None):
            return getattr(d, "param")
        if hasattr(d, "value") and getattr(d, "value", None) is not None:
            return int(getattr(d, "value"))
    except Exception:
        pass
    if isinstance(d, (int, np.integer)):
        return int(d)
    if isinstance(d, str):
        return d
    try:
        text = str(d)
        if text and text.isidentifier():
            return text
    except Exception:
        pass
    if d is None:
        return None
    return None


def _to_ir_dim_for_shape(d):
    try:
        if isinstance(d, ir.SymbolicDim):  # type: ignore[attr-defined]
            return d
        if hasattr(d, "param") and d.param:
            return ir.SymbolicDim(str(d.param))  # type: ignore[attr-defined]
        if hasattr(d, "value") and d.value is not None:
            return int(d.value)
    except Exception:
        pass
    if isinstance(d, (int, np.integer)):
        return int(d)
    if isinstance(d, str):
        return ir.SymbolicDim(d)  # type: ignore[attr-defined]
    if d is None:
        return None
    return None


def _is_static_int(d) -> bool:
    return isinstance(d, (int, np.integer)) and int(d) >= 0


def _dim_label_from_value_or_aval(val: ir.Value, aval_shape: tuple, i: int):
    """Return a readable dim label, falling back to aval metadata when needed."""

    shp = getattr(val, "shape", None)
    if shp is not None:
        dims = getattr(shp, "dims", None)
        if dims is None:
            try:
                dims = list(shp)
            except Exception:
                dims = None
        if dims is not None and i < len(dims):
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
    dims = getattr(shp, "dims", None)
    if dims is None:
        try:
            dims = list(shp)
        except Exception:
            return True
    for d in dims:
        if _as_ir_dim_label(d) is not None:
            return False
    return True
