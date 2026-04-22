# tests/extra_tests/framework/test_ir_utils.py

import numpy as np
import onnx_ir as ir

from jax2onnx.ir_utils import (
    const_value_to_numpy,
    ir_dtype_to_numpy,
    numpy_dtype_to_ir,
    tensor_attr,
    tensor_to_numpy,
)


def test_tensor_to_numpy_reads_onnx_ir_tensor() -> None:
    array = np.array([1.0, 2.0], dtype=np.float32)

    actual = tensor_to_numpy(ir.tensor(array))

    assert actual is not None
    np.testing.assert_array_equal(actual, array)


def test_const_value_to_numpy_reads_initializer_value() -> None:
    array = np.array([1, 2, 3], dtype=np.int64)
    value = ir.Value(
        name="x",
        shape=ir.Shape(array.shape),
        type=ir.TensorType(ir.DataType.INT64),
        const_value=ir.tensor(array),
    )

    actual = const_value_to_numpy(value)

    assert actual is not None
    np.testing.assert_array_equal(actual, array)


def test_const_value_to_numpy_reads_constant_node_output() -> None:
    array = np.array([True, False], dtype=np.bool_)
    output = ir.Value(
        name="const_out",
        shape=ir.Shape(array.shape),
        type=ir.TensorType(ir.DataType.BOOL),
    )
    ir.Node(
        "",
        "Constant",
        [],
        [ir.Attr("value", ir.AttributeType.TENSOR, ir.tensor(array))],
        outputs=[output],
        name="const_node",
    )

    actual = const_value_to_numpy(output)

    assert actual is not None
    np.testing.assert_array_equal(actual, array)


def test_tensor_attr_converts_tensor_value() -> None:
    array = np.array([1, 2, 3], dtype=np.int64)

    attr = tensor_attr("value", ir.tensor(array))

    assert attr.type == ir.AttributeType.TENSOR
    np.testing.assert_array_equal(tensor_to_numpy(attr.as_tensor()), array)


def test_ir_dtype_to_numpy_uses_onnx_ir_dtype_mapping() -> None:
    assert ir_dtype_to_numpy(ir.DataType.INT64) == np.dtype(np.int64)
    assert ir_dtype_to_numpy(ir.DataType.BFLOAT16) == np.dtype(
        ir.DataType.BFLOAT16.numpy()
    )
    assert ir_dtype_to_numpy(int(ir.DataType.FLOAT.value)) == np.dtype(np.float32)
    assert ir_dtype_to_numpy(np.dtype(np.float16)) == np.dtype(np.float16)
    assert ir_dtype_to_numpy(object(), default=None) is None


def test_numpy_dtype_to_ir_uses_onnx_ir_dtype_mapping() -> None:
    assert numpy_dtype_to_ir(np.float32) == ir.DataType.FLOAT
    assert numpy_dtype_to_ir(np.dtype(np.float64)) == ir.DataType.DOUBLE
    assert numpy_dtype_to_ir(np.dtype(np.float16)) == ir.DataType.FLOAT16
    assert numpy_dtype_to_ir(np.dtype(np.int32)) == ir.DataType.INT32
    assert numpy_dtype_to_ir(np.dtype(np.bool_)) == ir.DataType.BOOL
    assert (
        numpy_dtype_to_ir(
            object(),
            default=ir.DataType.INT64,
        )
        == ir.DataType.INT64
    )
