# jax2onnx/ir_utils.py

from __future__ import annotations

from typing import Any, cast

import numpy as np
import onnx_ir as ir


def tensor_to_numpy(tensor: object) -> np.ndarray[Any, np.dtype[Any]] | None:
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return cast(np.ndarray[Any, np.dtype[Any]], tensor)

    numpy_method = getattr(tensor, "numpy", None)
    if callable(numpy_method):
        try:
            result = numpy_method()
        except Exception:
            return None
        if isinstance(result, np.ndarray):
            return cast(np.ndarray[Any, np.dtype[Any]], result)
        return cast(np.ndarray[Any, np.dtype[Any]], np.asarray(result))

    try:
        return cast(np.ndarray[Any, np.dtype[Any]], np.asarray(tensor))
    except Exception:
        return None


def const_value_to_numpy(value: object) -> np.ndarray[Any, np.dtype[Any]] | None:
    if not isinstance(value, ir.Value):
        return None
    tensor = ir.convenience.get_const_tensor(value)
    if tensor is None:
        return None
    return tensor_to_numpy(tensor)


def tensor_attr(name: str, tensor: object) -> ir.Attr:
    return ir.convenience.convert_attribute(
        name,
        cast(Any, tensor),
        ir.AttributeType.TENSOR,
    )
