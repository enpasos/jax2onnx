# file: jax2onnx/converter/dtype_utils.py

import numpy as np
from onnx import TensorProto


def tensorproto_dtype_to_numpy(onnx_dtype: int) -> np.dtype:
    """
    Converts ONNX TensorProto data types to NumPy data types.

    Args:
        onnx_dtype: ONNX data type as an integer.

    Returns:
        Corresponding NumPy data type.
    """
    dtype_map = {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.BOOL: np.bool_,
        TensorProto.INT8: np.int8,
        TensorProto.UINT8: np.uint8,
    }
    np_dtype = dtype_map.get(onnx_dtype)
    if np_dtype is None:
        print(
            f"Warning: Unsupported ONNX dtype {onnx_dtype} encountered. Defaulting to np.float32."
        )
        return np.float32
    return np_dtype


def numpy_dtype_to_tensorproto(dtype):
    """
    Converts NumPy data types to ONNX TensorProto data types.

    Args:
        dtype: NumPy data type or equivalent representation.

    Returns:
        Corresponding ONNX TensorProto data type.

    Raises:
        TypeError: If the input dtype is unsupported.
    """
    import numpy as np
    from onnx import TensorProto

    # Normalize input
    if isinstance(dtype, int):  # Already ONNX enum
        return dtype

    if isinstance(dtype, str):
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            raise TypeError(f"Unsupported dtype string: {dtype}")

    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        dtype = np.dtype(dtype)

    if isinstance(dtype, np.dtype):
        dtype_map = {
            np.dtype("float32"): TensorProto.FLOAT,
            np.dtype("float64"): TensorProto.DOUBLE,
            np.dtype("float16"): TensorProto.FLOAT16,
            np.dtype("int32"): TensorProto.INT32,
            np.dtype("int64"): TensorProto.INT64,
            np.dtype("uint8"): TensorProto.UINT8,
            np.dtype("int8"): TensorProto.INT8,
            np.dtype("int16"): TensorProto.INT16,  # ✅ Add this line
            np.dtype("bool"): TensorProto.BOOL,
        }

        if dtype in dtype_map:
            return dtype_map[dtype]
        else:
            raise TypeError(f"Unsupported dtype: {dtype} ({type(dtype)})")

    raise TypeError(f"Unsupported dtype: {dtype} ({type(dtype)})")
