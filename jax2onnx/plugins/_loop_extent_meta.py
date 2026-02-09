# jax2onnx/plugins/_loop_extent_meta.py

from __future__ import annotations

from typing import Any

import numpy as np

AXIS0_OVERRIDE_META_KEY: str = "loop_axis0_override"


def _static_axis0(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    dims = getattr(shape, "dims", None)
    if dims is None:
        return None
    if len(dims) == 0:
        return 0
    dim0 = dims[0]
    if isinstance(dim0, (int, np.integer)):
        return int(dim0)
    maybe = getattr(dim0, "value", None)
    if isinstance(maybe, (int, np.integer)):
        return int(maybe)
    try:
        return int(dim0)
    except Exception:
        return None


def get_axis0_override(value: Any) -> int | None:
    meta = getattr(value, "meta", None)
    if meta is None:
        return None
    maybe = meta.get(AXIS0_OVERRIDE_META_KEY)
    if isinstance(maybe, (int, np.integer)):
        return int(maybe)
    return None


def set_axis0_override(value: Any, extent: Any) -> None:
    meta = getattr(value, "meta", None)
    if meta is None:
        return
    if isinstance(extent, (int, np.integer)) and int(extent) >= 0:
        meta[AXIS0_OVERRIDE_META_KEY] = int(extent)


def propagate_axis0_override(src: Any, dest: Any) -> None:
    override = get_axis0_override(src)
    if override is None:
        return
    dest_axis0 = _static_axis0(dest)
    if dest_axis0 == 0:
        return
    if isinstance(dest_axis0, int) and dest_axis0 > 1 and dest_axis0 != override:
        return
    set_axis0_override(dest, override)
