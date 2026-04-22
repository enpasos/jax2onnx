# tests/extra_tests/converter/test_ir_attr_conversion.py

from __future__ import annotations

import numpy as np
from onnx_ir import AttributeType

from jax2onnx.converter.conversion_api import _convert_ir_attr
from jax2onnx.ir_utils import tensor_to_numpy


def test_convert_ir_attr_normalizes_numpy_scalars() -> None:
    attr = _convert_ir_attr("axis", np.int64(2))

    assert attr is not None
    assert attr.type == AttributeType.INT
    assert attr.as_int() == 2


def test_convert_ir_attr_keeps_empty_sequences_typed() -> None:
    attr = _convert_ir_attr("axes", [])

    assert attr is not None
    assert attr.type == AttributeType.INTS
    assert attr.as_ints() == ()


def test_convert_ir_attr_uses_tensor_fallback_for_arrays() -> None:
    expected = np.asarray([1, 2, 3], dtype=np.int64)

    attr = _convert_ir_attr("value", expected)

    assert attr is not None
    assert attr.type == AttributeType.TENSOR
    np.testing.assert_array_equal(tensor_to_numpy(attr.as_tensor()), expected)
