# jax2onnx/plugins/_loop_extent_meta.py

from __future__ import annotations

from typing import Any

import numpy as np

AXIS0_OVERRIDE_META_KEY: str = "loop_axis0_override"


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
    if override is not None:
        set_axis0_override(dest, override)
