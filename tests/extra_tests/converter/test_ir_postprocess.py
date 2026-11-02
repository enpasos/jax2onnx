# tests/extra_tests/converter/test_ir_postprocess.py

import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx.converter.ir_postprocess import postprocess_ir_model


def _float_tensor_value(name: str, array: np.ndarray) -> ir.Value:
    return ir.Value(
        name=name,
        shape=ir.Shape(array.shape),
        type=ir.TensorType(ir.DataType.from_numpy(array.dtype)),
        const_value=ir.tensor(array),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_postprocess_preserves_typed_constant_attributes(dtype: type) -> None:
    array = np.array([1.0, 2.0], dtype=dtype)
    output = ir.Value(
        name="y",
        shape=ir.Shape(array.shape),
        type=ir.TensorType(ir.DataType.from_numpy(array.dtype)),
    )
    constant = ir.Node(
        "",
        "Constant",
        [],
        [ir.Attr("value", ir.AttributeType.TENSOR, ir.tensor(array))],
        outputs=[output],
        name="const_node",
    )
    model = ir.Model(ir.Graph([], [output], nodes=[constant], name="g"), ir_version=10)

    postprocess_ir_model(model)

    actual = list(model.graph)[0].attributes["value"].as_tensor().numpy()
    assert actual.dtype == array.dtype
    assert actual.tobytes() == array.tobytes()
    assert output.dtype == ir.DataType.from_numpy(array.dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_postprocess_preserves_typed_initializers(dtype: type) -> None:
    graph_input = ir.Value(
        name="x",
        shape=ir.Shape((2,)),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    graph_output = ir.Value(
        name="out",
        shape=ir.Shape((2,)),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    array = np.array([3.0, 4.0], dtype=dtype)
    weight = _float_tensor_value("weight", array)
    model = ir.Model(
        ir.Graph(
            [graph_input],
            [graph_output],
            nodes=[],
            initializers=[weight],
            name="g",
        ),
        ir_version=10,
    )

    postprocess_ir_model(model)

    actual = model.graph.initializers["weight"]
    assert actual.const_value.numpy().dtype == array.dtype
    assert actual.const_value.numpy().tobytes() == array.tobytes()
    assert actual.dtype == ir.DataType.from_numpy(array.dtype)
    assert graph_input.dtype == ir.DataType.FLOAT
    assert graph_output.dtype == ir.DataType.FLOAT
