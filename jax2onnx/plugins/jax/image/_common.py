# jax2onnx/plugins/jax/image/_common.py

from __future__ import annotations

from typing import Callable

from jax.extend.core import Primitive


def make_image_primitive(name: str) -> Primitive:
    prim = Primitive(name)
    prim.multiple_results = False
    return prim


def get_orig_impl(
    prim: Primitive, func_name: str, store_attr: str = "__orig_impl__"
) -> Callable:
    storage_slot = f"{store_attr}_{func_name}"
    orig = getattr(prim, storage_slot, None)
    if orig is None:
        raise RuntimeError(
            f"Original implementation for jax.image.{func_name} not captured"
        )
    return orig
