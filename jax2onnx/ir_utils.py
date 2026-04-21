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


def ir_dtype_to_numpy(
    dtype: object,
    *,
    default: np.dtype[Any] | None = np.dtype(np.float32),
) -> np.dtype[Any] | None:
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, ir.DataType):
        try:
            return cast(np.dtype[Any], np.dtype(dtype.numpy()))
        except Exception:
            return default
    if isinstance(dtype, (int, np.integer)):
        try:
            return ir_dtype_to_numpy(ir.DataType(int(dtype)), default=default)
        except Exception:
            return default
    return default


def numpy_dtype_to_ir(
    dtype: object,
    *,
    default: ir.DataType = ir.DataType.FLOAT,
) -> ir.DataType:
    try:
        return ir.DataType.from_numpy(np.dtype(dtype))
    except Exception:
        return default
