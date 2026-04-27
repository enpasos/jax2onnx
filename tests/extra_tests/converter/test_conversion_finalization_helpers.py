# tests/extra_tests/converter/test_conversion_finalization_helpers.py

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.conversion_api import (
    _apply_late_ir_attr_overrides,
    _attach_ir_functions,
    _finalize_model_value_shapes,
)
from jax2onnx.converter.ir_context import IRContext


def _value(name: str, shape: tuple[object, ...] = (1,)) -> ir.Value:
    return ir.Value(
        name=name,
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape(shape),
    )


def _model_with_node(node: ir.Node, inputs: list[ir.Value], outputs: list[ir.Value]):
    graph = ir.Graph(
        inputs,
        outputs,
        nodes=[node],
        opset_imports={"": 21},
        name="test_graph",
    )
    return ir.Model(graph, ir_version=10)


def test_attach_ir_functions_adds_functions_and_opset_imports() -> None:
    ctx = IRContext(opset=21, enable_double_precision=False, input_specs=[])
    fn_input = _value("fn_x")
    fn_output = _value("fn_y")
    fn_graph = ir.Graph(
        [fn_input],
        [fn_output],
        nodes=[],
        opset_imports={"": 21},
        name="fn_graph",
    )
    fn = ir.Function("custom", "Body", graph=fn_graph, attributes={})
    ctx.ir_functions.append(fn)

    model_input = _value("x")
    model_output = _value("y")
    model = ir.Model(
        ir.Graph(
            [model_input],
            [model_output],
            nodes=[],
            opset_imports={"": 21},
            name="main",
        ),
        ir_version=10,
    )

    _attach_ir_functions(model, ctx)

    functions = model.functions
    if isinstance(functions, dict):
        attached_functions = list(functions.values())
    else:
        attached_functions = list(functions)

    assert attached_functions == [fn]
    assert model.opset_imports[""] == 21
    assert model.opset_imports["custom"] == 1


def test_apply_late_ir_attr_overrides_updates_attrs_and_concat_axis() -> None:
    ctx = IRContext(opset=21, enable_double_precision=False, input_specs=[])
    lhs = _value("lhs")
    rhs = _value("rhs")
    out = _value("out")
    concat = ir.Node("", "Concat", [lhs, rhs], outputs=[out], name="concat")
    model = _model_with_node(concat, [lhs, rhs], [out])

    ctx.add_node_attr_override("concat", {"axis": np.int64(2)})

    _apply_late_ir_attr_overrides(model, ctx)

    assert concat.attributes["axis"].as_int() == 2


def test_apply_late_ir_attr_overrides_fills_missing_concat_axis() -> None:
    ctx = IRContext(opset=21, enable_double_precision=False, input_specs=[])
    lhs = _value("lhs")
    rhs = _value("rhs")
    out = _value("out")
    concat = ir.Node("", "Concat", [lhs, rhs], outputs=[out], name="concat")
    model = _model_with_node(concat, [lhs, rhs], [out])

    _apply_late_ir_attr_overrides(model, ctx)

    assert concat.attributes["axis"].as_int() == 0


def test_finalize_model_value_shapes_normalizes_symbolic_dims() -> None:
    value = _value("x", (np.int64(2), "B"))
    model = ir.Model(
        ir.Graph([value], [value], nodes=[], opset_imports={"": 21}, name="main"),
        ir_version=10,
    )

    _finalize_model_value_shapes(model)

    dims = model.graph.inputs[0].shape.dims
    assert dims[0] == 2
    assert isinstance(dims[0], int)
    assert isinstance(dims[1], ir.SymbolicDim)
    assert str(dims[1]) == "B"
