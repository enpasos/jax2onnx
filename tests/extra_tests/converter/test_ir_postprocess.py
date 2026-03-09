# tests/extra_tests/converter/test_ir_postprocess.py

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_postprocess import postprocess_ir_model


def _float_tensor_value(name: str, array: np.ndarray) -> ir.Value:
    return ir.Value(
        name=name,
        shape=ir.Shape(array.shape),
        type=ir.TensorType(ir.DataType.from_numpy(array.dtype)),
        const_value=ir.tensor(array),
    )


def test_postprocess_promotes_constant_attributes_when_requested() -> None:
    array = np.array([1.0, 2.0], dtype=np.float32)
    output = ir.Value(
        name="y",
        shape=ir.Shape(array.shape),
        type=ir.TensorType(ir.DataType.FLOAT),
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

    postprocess_ir_model(model, promote_to_double=True)

    promoted = list(model.graph)[0].attributes["value"].as_tensor().numpy()
    assert promoted.dtype == np.float64


def test_postprocess_promotes_initializers_when_requested() -> None:
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
    weight = _float_tensor_value("weight", np.array([3.0, 4.0], dtype=np.float32))
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

    postprocess_ir_model(model, promote_to_double=True)

    promoted = model.graph.initializers["weight"]
    assert promoted.const_value.numpy().dtype == np.float64
    assert promoted.dtype == ir.DataType.DOUBLE
