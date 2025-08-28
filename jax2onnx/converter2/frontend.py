from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import jax


def _normalize_inputs_for_tracing(inputs: List[Any]) -> Tuple[Any, ...]:
    """
    Ensure we hand jax.make_jaxpr abstract values with shape + dtype where possible.
    Accepts jax.ShapeDtypeStruct / jax.core.ShapedArray / (tuple|list) shape.
    If only a shape is given, default to float32 for tracing purposes.
    """
    import numpy as np

    xs = []
    for spec in inputs:
        if hasattr(spec, "shape") and hasattr(spec, "dtype"):
            xs.append(spec)  # already a ShapeDtypeStruct/ShapedArray-like
        elif isinstance(spec, (tuple, list)):
            xs.append(jax.ShapeDtypeStruct(tuple(spec), np.float32))
        else:
            raise TypeError(f"Unsupported input spec type for tracing: {type(spec)}")
    return tuple(xs)


def trace_to_jaxpr(fn: Any, inputs: List[Any], input_params: Optional[Dict[str, Any]]):
    """
    Returns a closed jaxpr for fn(*inputs, **input_params).
    """
    xs = _normalize_inputs_for_tracing(inputs)

    if input_params:

        def wrapped(*args):
            return fn(*args, **input_params)

        f = wrapped
    else:
        f = fn

    closed = jax.make_jaxpr(f)(*xs)
    return closed
