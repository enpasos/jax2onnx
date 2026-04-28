# jax2onnx/ir_utils.py

from __future__ import annotations

from collections.abc import Sequence
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


def numpy_dtype_to_ir_with_float_policy(
    dtype: object | None,
    enable_double_precision: bool,
) -> ir.DataType:
    """
    Map a numpy dtype to an IR dtype using the converter's float precision policy.

    Float32 and unknown floating dtypes follow ``enable_double_precision``;
    float16 and float64 keep their exact ONNX IR dtype.
    """
    if dtype is None:
        return ir.DataType.DOUBLE if enable_double_precision else ir.DataType.FLOAT
    try:
        key = np.dtype(dtype)
    except Exception as exc:
        raise TypeError(f"Unsupported dtype: {dtype}") from exc
    if np.issubdtype(key, np.floating):
        if key == np.float16:
            return ir.DataType.FLOAT16
        if key == np.float32:
            return ir.DataType.DOUBLE if enable_double_precision else ir.DataType.FLOAT
        if key == np.float64:
            return ir.DataType.DOUBLE
        return ir.DataType.DOUBLE if enable_double_precision else ir.DataType.FLOAT
    try:
        return ir.DataType.from_numpy(key)
    except Exception as exc:
        raise TypeError(f"Unsupported dtype: {dtype}") from exc


def maybe_numpy_dtype(dtype: object | None) -> np.dtype[Any] | None:
    if dtype is None:
        return None
    try:
        return cast(np.dtype[Any], np.dtype(dtype))
    except TypeError:
        return None


def coerce_ir_shape_dim(
    dim: object,
    *,
    parse_integer_like: bool = True,
) -> int | str:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    if parse_integer_like:
        try:
            return int(cast(Any, dim))
        except Exception:
            pass
    return str(dim)


def coerce_ir_shape_dims(
    dims: Sequence[object],
    *,
    parse_integer_like: bool = True,
) -> tuple[int | str, ...]:
    return tuple(
        coerce_ir_shape_dim(dim, parse_integer_like=parse_integer_like) for dim in dims
    )


def ir_shape_from_dims(
    dims: Sequence[object],
    *,
    parse_integer_like: bool = True,
) -> ir.Shape:
    return ir.Shape(coerce_ir_shape_dims(dims, parse_integer_like=parse_integer_like))
