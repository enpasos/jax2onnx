# tests/extra_tests/framework/test_function_optimized_output_alias.py

"""Function formal result slots survive shared destructive optimization."""

from __future__ import annotations

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx.converter.ir_optimizations import (
    _FunctionOutputMetadata,
    _preserve_function_output_bindings,
    optimize_graph,
)
from jax2onnx.converter.precision_policy import PrecisionPolicy


def _function_alias_fixture(shape):
    formal = ir.val("formal", ir.DataType.INT32, shape)
    formal.doc_string = "Input passthrough documentation"
    formal.metadata_props["slot"] = "passthrough"
    made = ir.val("produced", ir.DataType.INT32, shape)
    repeated = ir.val("repeated", ir.DataType.INT32, shape)
    repeated.doc_string = "Second result documentation"
    repeated.metadata_props["slot"] = "second"
    sibling = ir.val("sibling", ir.DataType.INT32, shape)
    producer = ir.Node("", "Neg", [formal], outputs=[made], name="produce")
    cast = ir.Node(
        "",
        "Cast",
        [made],
        attributes=[ir.AttrInt64("to", int(ir.DataType.INT32))],
        outputs=[repeated],
        name="redundant_cast",
    )
    consumer = ir.Node("", "Neg", [made], outputs=[sibling], name="sibling_neg")
    graph = ir.Graph(
        [formal],
        [made, repeated, formal, formal, sibling],
        nodes=[producer, cast, consumer],
        name="AliasSlots",
        opset_imports={"": 23},
    )
    # Occupy the natural input-alias suffix without returning that value.
    made.name = "formal_1"
    function = ir.Function(
        "test.function_alias", "AliasSlots", graph=graph, attributes=[]
    )
    inputs = [ir.val(name, ir.DataType.INT32, shape) for name in ("x", "other")]
    calls = []
    outputs = []
    for index, value in enumerate(inputs):
        results = [
            ir.val(f"call_{index}_out_{slot}", ir.DataType.INT32, shape)
            for slot in range(5)
        ]
        calls.append(ir.Node(function.domain, function.name, [value], outputs=results))
        outputs.extend(results)
    root = ir.Graph(
        inputs,
        outputs,
        nodes=calls,
        name="caller",
        opset_imports={"": 23, function.domain: 1},
    )
    model = ir.Model(root, ir_version=10, functions=[function])
    return (
        model,
        function,
        formal,
        made,
        repeated,
        sibling,
        producer,
        cast,
        consumer,
        calls,
    )


@pytest.mark.parametrize(
    "precision_policy", [PrecisionPolicy.SOURCE, PrecisionPolicy.LEGACY]
)
@pytest.mark.parametrize("shape", [(0, 4), (2, 4)], ids=["empty", "nonempty"])
def test_function_optimized_output_aliases_preserve_real_slots(precision_policy, shape):
    (
        model,
        function,
        formal,
        made,
        repeated,
        sibling,
        producer,
        cast,
        consumer,
        calls,
    ) = _function_alias_fixture(shape)
    root, graph = model.graph, function.graph
    input_metadata = (formal.name, formal.doc_string, dict(formal.metadata_props))
    repeated_metadata = (repeated.doc_string, dict(repeated.metadata_props))
    assert optimize_graph(model, precision_policy=precision_policy) is model
    assert model.graph is root
    assert model.functions[function.identifier()] is function
    assert function.graph is graph and list(function.inputs) == [formal]
    assert list(root) == calls and all(
        call.inputs[0] is value for call, value in zip(calls, root.inputs)
    )
    assert function.outputs[0] is made and function.outputs[4] is sibling
    assert consumer.inputs[0] is made and producer.inputs[0] is formal
    assert cast.graph is None and tuple(cast.inputs) == (None,)
    assert (cast, 0) not in made.uses() and not repeated.uses()
    assert (
        formal.name,
        formal.doc_string,
        dict(formal.metadata_props),
    ) == input_metadata
    assert all(value.graph is graph for value in function.outputs)
    names = [value.name for value in function.outputs]
    assert all(names) and len(set(names)) == 5
    assert not set(names).intersection(value.name for value in function.inputs)
    assert function.outputs[1].name == "repeated"
    assert function.outputs[2].name == "formal_2"
    assert function.outputs[3].name == "formal_3"
    for slot, expected in ((1, made), (2, formal), (3, formal)):
        alias = function.outputs[slot]
        node = alias.producer()
        assert node is not None and node.graph is graph
        assert node.domain == "" and node.op_type == "Identity"
        assert list(node.inputs) == [expected] and node.outputs[0] is alias
        assert alias.type == expected.type and alias.shape == expected.shape
        assert alias.const_value is None
    assert (
        function.outputs[1].doc_string,
        dict(function.outputs[1].metadata_props),
    ) == repeated_metadata
    assert all(
        (function.outputs[slot].doc_string, dict(function.outputs[slot].metadata_props))
        == input_metadata[1:]
        for slot in (2, 3)
    )
    assert not graph.initializers and [node.op_type for node in graph] == [
        "Neg",
        "Neg",
        "Identity",
        "Identity",
        "Identity",
    ]
    first_nodes, first_outputs = tuple(graph), tuple(graph.outputs)
    first_bytes = ir.to_proto(model).SerializeToString()
    optimize_graph(model, precision_policy=precision_policy)
    assert tuple(graph) == first_nodes and tuple(graph.outputs) == first_outputs
    assert ir.to_proto(model).SerializeToString() == first_bytes
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    data = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
    other = -data - np.int32(1)
    expected = [
        value
        for operand in (data, other)
        for value in (-operand, -operand, operand, operand, operand)
    ]
    for optimization in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.graph_optimization_level = optimization
        session = ort.InferenceSession(
            proto.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        actual = session.run(
            None,
            {value.name: payload for value, payload in zip(root.inputs, (data, other))},
        )
        assert len(actual) == len(expected) == 10
        for got, want in zip(actual, expected):
            assert got.dtype == want.dtype and got.shape == want.shape
            assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("hostile", ["arity", "type", "shape", "name"])
def test_function_output_alias_rejects_contradictory_contract(hostile):
    model, function, formal, made, *_ = _function_alias_fixture((2, 4))
    graph = function.graph
    original = [
        _FunctionOutputMetadata(
            value.name,
            value.type,
            value.shape,
            value.doc_string,
            dict(value.metadata_props),
        )
        for value in graph.outputs
    ]
    if hostile == "arity":
        original.pop()
        message = "output arity"
    elif hostile == "type":
        made.type = ir.TensorType(ir.DataType.FLOAT)
        message = "declared type"
    elif hostile == "shape":
        made.shape = ir.Shape((3, 4))
        message = "declared shape"
    else:
        made.name = formal.name
        message = "Conflicting Function definitions"
    nodes, outputs = tuple(graph), tuple(graph.outputs)
    with pytest.raises(ValueError, match=message):
        _preserve_function_output_bindings(graph, original)
    assert tuple(graph) == nodes and tuple(graph.outputs) == outputs
    assert model.functions[function.identifier()] is function


def test_function_output_alias_preserves_partial_shape_refinement():
    made = ir.val("made", ir.DataType.INT32, (2, 4))
    formal = ir.val("formal", ir.DataType.INT32, (2, 4))
    producer = ir.Node("", "Neg", [formal], outputs=[made])
    graph = ir.Graph([formal], [made, made], nodes=[producer], opset_imports={"": 23})
    original = [
        _FunctionOutputMetadata("first", made.type, ir.Shape((None, 4)), None, {}),
        _FunctionOutputMetadata(
            "second",
            made.type,
            ir.Shape((None, 4)),
            "refined result",
            {"slot": "second"},
        ),
    ]
    _preserve_function_output_bindings(graph, original)
    assert graph.outputs[0] is made and made.shape == ir.Shape((2, 4))
    alias = graph.outputs[1]
    assert alias.shape == made.shape and alias.shape is not made.shape
    assert alias.producer().inputs[0] is made
    assert alias.doc_string == "refined result" and dict(alias.metadata_props) == {
        "slot": "second"
    }
