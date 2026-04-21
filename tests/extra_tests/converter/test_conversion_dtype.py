# tests/extra_tests/converter/test_conversion_dtype.py

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.conversion_api import _to_ir_dtype_from_np


def test_to_ir_dtype_from_np_reuses_ir_integer_mapping() -> None:
    assert _to_ir_dtype_from_np(np.dtype(np.int8)) == ir.DataType.INT8
    assert _to_ir_dtype_from_np(np.dtype(np.uint16)) == ir.DataType.UINT16
    assert _to_ir_dtype_from_np(np.dtype(np.int64)) == ir.DataType.INT64


def test_to_ir_dtype_from_np_reuses_ir_bool_mapping() -> None:
    assert _to_ir_dtype_from_np(np.dtype(np.bool_)) == ir.DataType.BOOL


def test_to_ir_dtype_from_np_preserves_float_policy() -> None:
    assert _to_ir_dtype_from_np(np.dtype(np.float16)) == ir.DataType.FLOAT
    assert _to_ir_dtype_from_np(np.dtype(np.float32)) == ir.DataType.FLOAT
    assert _to_ir_dtype_from_np(np.dtype(np.float64)) == ir.DataType.DOUBLE
