# tests/extra_tests/framework/test_ir_utils.py

import numpy as np
import onnx_ir as ir

from jax2onnx.ir_utils import (
    const_value_to_numpy,
    ir_shape_from_dims,
    ir_dtype_to_numpy,
    iter_ir_functions,
    maybe_numpy_dtype,
    numpy_dtype_to_ir,
    numpy_dtype_to_ir_with_float_policy,
    tensor_attr,
    tensor_to_numpy,
)


class _SymbolicDim:
    def __str__(self) -> str:
        return "batch"


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


def test_numpy_dtype_to_ir_with_float_policy_matches_converter_rules() -> None:
    assert numpy_dtype_to_ir_with_float_policy(None, False) == ir.DataType.FLOAT
    assert numpy_dtype_to_ir_with_float_policy(None, True) == ir.DataType.DOUBLE
    assert (
        numpy_dtype_to_ir_with_float_policy(np.dtype(np.float16), True)
        == ir.DataType.FLOAT16
    )
    assert (
        numpy_dtype_to_ir_with_float_policy(np.dtype(np.float32), False)
        == ir.DataType.FLOAT
    )
    assert (
        numpy_dtype_to_ir_with_float_policy(np.dtype(np.float32), True)
        == ir.DataType.DOUBLE
    )
    assert (
        numpy_dtype_to_ir_with_float_policy(np.dtype(np.int16), True)
        == ir.DataType.INT16
    )


def test_maybe_numpy_dtype_normalizes_or_returns_none() -> None:
    assert maybe_numpy_dtype(np.float32) == np.dtype(np.float32)
    assert maybe_numpy_dtype("float64") == np.dtype(np.float64)
    assert maybe_numpy_dtype(object()) is None


def test_ir_shape_from_dims_coerces_dims_for_ir_values() -> None:
    shape = ir_shape_from_dims((np.int64(2), "3", _SymbolicDim()))

    assert shape.dims == (2, 3, "batch")


def test_ir_shape_from_dims_can_preserve_symbolic_numeric_strings() -> None:
    shape = ir_shape_from_dims(("3", np.int64(4)), parse_integer_like=False)

    assert shape.dims == ("3", 4)


def test_iter_ir_functions_handles_common_container_shapes() -> None:
    first = object()
    second = object()

    class _ValuesContainer:
        def values(self) -> tuple[object, object]:
            return (first, second)

    assert list(iter_ir_functions(None)) == []
    assert list(iter_ir_functions({"first": first, "second": second})) == [
        first,
        second,
    ]
    assert list(iter_ir_functions([first, second])) == [first, second]
    assert list(iter_ir_functions(_ValuesContainer())) == [first, second]
