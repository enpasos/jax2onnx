# jax2onnx/converter/frontend.py


from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import jax
import numpy as np


def _normalize_inputs_for_tracing(
    inputs: List[Any],
    default_float: Any | None = None,  # <- optional, positional or keyword
) -> Tuple[Any, ...]:
    """
    Ensure we hand jax.make_jaxpr abstract values with shape + dtype where possible.
    Accepts jax.ShapeDtypeStruct / jax.core.ShapedArray / (tuple|list) shape.
    If only a shape is given, default to float32 for tracing purposes.
    """
    if default_float is None:
        default_float = np.float32

    xs = []
    for spec in inputs:
        if hasattr(spec, "shape") and hasattr(spec, "dtype"):
            xs.append(spec)  # already a ShapeDtypeStruct/ShapedArray-like
        elif isinstance(spec, (tuple, list)):
            xs.append(jax.ShapeDtypeStruct(tuple(spec), default_float))
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
