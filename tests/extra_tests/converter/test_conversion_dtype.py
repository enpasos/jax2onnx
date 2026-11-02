# tests/extra_tests/converter/test_conversion_dtype.py

import numpy as np
import onnx_ir as ir

from jax2onnx.ir_utils import numpy_dtype_to_ir


def test_numpy_dtype_to_ir_reuses_ir_integer_mapping() -> None:
    assert numpy_dtype_to_ir(np.dtype(np.int8)) == ir.DataType.INT8
    assert numpy_dtype_to_ir(np.dtype(np.uint16)) == ir.DataType.UINT16
    assert numpy_dtype_to_ir(np.dtype(np.int64)) == ir.DataType.INT64


def test_numpy_dtype_to_ir_reuses_ir_bool_mapping() -> None:
    assert numpy_dtype_to_ir(np.dtype(np.bool_)) == ir.DataType.BOOL


def test_numpy_dtype_to_ir_preserves_source_float_widths() -> None:
    assert numpy_dtype_to_ir(np.dtype(np.float16)) == ir.DataType.FLOAT16
    assert numpy_dtype_to_ir(np.dtype(np.float32)) == ir.DataType.FLOAT
    assert numpy_dtype_to_ir(np.dtype(np.float64)) == ir.DataType.DOUBLE
