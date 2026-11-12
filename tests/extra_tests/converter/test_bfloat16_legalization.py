# tests/extra_tests/converter/test_bfloat16_legalization.py

from __future__ import annotations

import onnx
import onnx_ir as ir
import pytest
from onnx_ir.tape import Tape

from jax2onnx.converter.bfloat16_legalization import legalize_bfloat16_operators

BF16 = ir.DataType.BFLOAT16
FLOAT = ir.DataType.FLOAT


def _value(name: str, shape: list[int], dtype: ir.DataType = BF16) -> ir.Value:
    return ir.val(name, dtype=dtype, shape=shape)


def _graph(
    inputs: list[ir.Value], outputs: list[ir.Value], opset: int, name: str = "g"
) -> ir.Graph:
    return ir.Graph(inputs, outputs, nodes=[], opset_imports={"": opset}, name=name)


def _legalize(graph: ir.Graph, functions: list[ir.Function] | None = None) -> ir.Model:
    model = ir.Model(graph, ir_version=10, functions=functions or [])
    legalize_bfloat16_operators(model)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    return model


def _ops(graph: ir.Graph) -> list[str]:
    return [node.op_type for node in graph]


def _cast_target(node: ir.Node) -> ir.DataType:
    assert node.op_type == "Cast"
    return ir.DataType(node.attributes["to"].as_int())


def _conv_graph(opset: int) -> tuple[ir.Graph, ir.Value, ir.Value, ir.Value]:
    x = _value("x", [1, 3, 8, 8])
    w = _value("w", [4, 3, 3, 3])
    y = _value("y", [1, 4, 6, 6])
    y.metadata_props["source"] = "kept"
    graph = _graph([x, w], [y], opset)
    Tape(graph).op("Conv", [x, w], output=y, name="conv")
    return graph, x, w, y


def test_conv_below_opset_22_computes_in_float32_between_bfloat16_values() -> None:
    graph, x, w, _ = _conv_graph(21)
    _legalize(graph)

    assert _ops(graph) == ["Cast", "Cast", "Conv", "Cast"]
    to_x, to_w, conv, restore = graph
    assert (to_x.inputs[0], to_w.inputs[0]) == (x, w)
    assert [_cast_target(node) for node in (to_x, to_w, restore)] == [
        FLOAT,
        FLOAT,
        BF16,
    ]
    assert list(conv.inputs) == [to_x.outputs[0], to_w.outputs[0]]
    assert [value.dtype for value in conv.inputs] == [FLOAT, FLOAT]
    assert conv.outputs[0].dtype == FLOAT
    assert restore.inputs[0] is conv.outputs[0]

    # The public boundary is untouched: same inputs, and an output that keeps
    # its name, type, shape and metadata.
    assert list(graph.inputs) == [x, w]
    assert [value.dtype for value in graph.inputs] == [BF16, BF16]
    (output,) = graph.outputs
    assert output is restore.outputs[0]
    assert (output.name, output.dtype) == ("y", BF16)
    assert output.shape == ir.Shape([1, 4, 6, 6])
    assert output.metadata_props == {"source": "kept"}


def test_conv_from_opset_22_is_left_in_bfloat16() -> None:
    graph, x, w, y = _conv_graph(22)
    _legalize(graph)

    assert _ops(graph) == ["Conv"]
    (conv,) = graph
    assert list(conv.inputs) == [x, w]
    assert conv.outputs[0] is y and y.dtype == BF16


def test_other_float_types_and_custom_domains_are_left_alone() -> None:
    x = _value("x", [1, 3, 8, 8], ir.DataType.FLOAT16)
    w = _value("w", [4, 3, 3, 3], ir.DataType.FLOAT16)
    y = _value("y", [1, 4, 6, 6], ir.DataType.FLOAT16)
    graph = _graph([x, w], [y], 21)
    Tape(graph).op("Conv", [x, w], output=y)

    custom_in = _value("a", [2])
    custom_out = _value("b", [2])
    custom = ir.Graph(
        [custom_in],
        [custom_out],
        nodes=[ir.Node("custom.domain", "Conv", [custom_in], outputs=[custom_out])],
        opset_imports={"": 21, "custom.domain": 1},
    )

    for candidate in (graph, custom):
        before = _ops(candidate)
        legalize_bfloat16_operators(ir.Model(candidate, ir_version=10))
        assert _ops(candidate) == before


def test_one_cast_per_source_value_and_graph() -> None:
    x = _value("x", [1, 3, 8, 8])
    w = _value("w", [4, 3, 3, 3])
    first = _value("first", [1, 4, 6, 6])
    second = _value("second", [1, 4, 6, 6])
    graph = _graph([x, w], [first, second], 21)
    tape = Tape(graph)
    tape.op("Conv", [x, w], output=first)
    tape.op("Conv", [x, w], output=second)
    _legalize(graph)

    assert _ops(graph) == ["Cast", "Cast", "Conv", "Cast", "Conv", "Cast"]
    convs = [node for node in graph if node.op_type == "Conv"]
    assert list(convs[0].inputs) == list(convs[1].inputs)


def test_chained_operators_keep_the_bfloat16_rounding_between_them() -> None:
    x = _value("x", [1, 3, 8, 8])
    w1 = _value("w1", [3, 3, 3, 3])
    w2 = _value("w2", [4, 3, 3, 3])
    hidden = _value("hidden", [1, 3, 6, 6])
    y = _value("y", [1, 4, 4, 4])
    graph = _graph([x, w1, w2], [y], 21)
    tape = Tape(graph)
    tape.op("Conv", [x, w1], output=hidden)
    tape.op("Conv", [hidden, w2], output=y)
    _legalize(graph)

    convs = [node for node in graph if node.op_type == "Conv"]
    # float32 -> bfloat16 -> float32 is a rounding step and must stay.
    round_down = convs[0].outputs[0].consumers()[0]
    assert _cast_target(round_down) == BF16
    assert round_down.outputs[0].name == "hidden"
    widen = convs[1].inputs[0].producer()
    assert widen is not None and _cast_target(widen) == FLOAT
    assert widen.inputs[0] is round_down.outputs[0]


def test_variadic_operands_are_all_widened() -> None:
    a = _value("a", [2, 3])
    b = _value("b", [3, 4])
    c = _value("c", [4, 5])
    y = _value("y", [2, 5])
    graph = _graph([a, b, c], [y], 27)
    Tape(graph).op(
        "Einsum", [a, b, c], output=y, attributes={"equation": "ij,jk,kl->il"}
    )
    _legalize(graph)

    assert _ops(graph) == ["Cast", "Cast", "Cast", "Einsum", "Cast"]
    einsum = next(node for node in graph if node.op_type == "Einsum")
    assert [value.dtype for value in einsum.inputs] == [FLOAT, FLOAT, FLOAT]
    assert graph.outputs[0].dtype == BF16

    native = _graph(
        [_value("a", [2, 3]), _value("b", [3, 4])], [_value("y", [2, 4])], 28
    )
    Tape(native).op(
        "Einsum",
        list(native.inputs),
        output=native.outputs[0],
        attributes={"equation": "ij,jk->ik"},
    )
    _legalize(native)
    assert _ops(native) == ["Einsum"]


def test_only_outputs_of_the_excluded_type_parameter_are_restored() -> None:
    x = _value("x", [2, 6])
    k = ir.val("k", dtype=ir.DataType.INT64, shape=[1])
    values = _value("values", [2, 2])
    indices = ir.val("indices", dtype=ir.DataType.INT64, shape=[2, 2])
    graph = _graph([x, k], [values, indices], 23)
    Tape(graph).op_multi_out("TopK", [x, k], outputs=[values, indices])
    _legalize(graph)

    assert _ops(graph) == ["Cast", "TopK", "Cast"]
    top_k = next(node for node in graph if node.op_type == "TopK")
    assert top_k.inputs[1] is k
    assert top_k.outputs[1] is indices and indices.dtype == ir.DataType.INT64
    assert [value.name for value in graph.outputs] == ["values", "indices"]
    assert graph.outputs[0].dtype == BF16


def test_unused_output_gets_no_restoring_cast() -> None:
    x = _value("x", [2, 6])
    k = ir.val("k", dtype=ir.DataType.INT64, shape=[1])
    values = _value("values", [2, 2])
    indices = ir.val("indices", dtype=ir.DataType.INT64, shape=[2, 2])
    graph = _graph([x, k], [indices], 23)
    Tape(graph).op_multi_out("TopK", [x, k], outputs=[values, indices])
    _legalize(graph)

    assert _ops(graph) == ["Cast", "TopK"]
    assert values.dtype == FLOAT


def test_input_only_type_parameter_needs_no_restoring_cast() -> None:
    x = _value("x", [4])
    y = ir.val("y", dtype=ir.DataType.BOOL, shape=[4])
    graph = _graph([x], [y], 19)
    Tape(graph).op("IsInf", [x], output=y)
    _legalize(graph)

    assert _ops(graph) == ["Cast", "IsInf"]
    assert graph.outputs[0] is y


def test_control_flow_body_is_legalized_in_its_own_scope() -> None:
    x = _value("x", [1, 3, 8, 8])
    w = _value("w", [4, 3, 3, 3])
    condition = ir.val("condition", dtype=ir.DataType.BOOL, shape=[])
    y = _value("y", [1, 4, 6, 6])

    def branch(name: str) -> ir.Graph:
        out = _value(f"{name}_out", [1, 4, 6, 6])
        body = ir.Graph([], [out], nodes=[], name=name)
        Tape(body).op("Conv", [x, w], output=out)
        return body

    then_branch, else_branch = branch("then"), branch("else")
    graph = _graph([x, w, condition], [y], 21)
    Tape(graph).op(
        "If",
        [condition],
        output=y,
        attributes={"then_branch": then_branch, "else_branch": else_branch},
    )
    _legalize(graph)

    assert _ops(graph) == ["If"]
    for body in (then_branch, else_branch):
        assert _ops(body) == ["Cast", "Cast", "Conv", "Cast"]
        (body_output,) = body.outputs
        assert body_output.dtype == BF16
        assert body_output.producer() is list(body)[-1]
    # Each body holds its own casts of the captured outer values.
    then_casts = {id(node.outputs[0]) for node in then_branch if node.op_type == "Cast"}
    else_casts = {id(node.outputs[0]) for node in else_branch if node.op_type == "Cast"}
    assert then_casts.isdisjoint(else_casts)


def test_function_body_keeps_its_bfloat16_signature() -> None:
    fx = _value("fx", [1, 3, 8, 8])
    fw = _value("fw", [4, 3, 3, 3])
    fy = _value("fy", [1, 4, 6, 6])
    body = _graph([fx, fw], [fy], 21, name="body")
    Tape(body).op("Conv", [fx, fw], output=fy)
    function = ir.Function("custom", "Block", graph=body, attributes=[])

    x = _value("x", [1, 3, 8, 8])
    w = _value("w", [4, 3, 3, 3])
    y = _value("y", [1, 4, 6, 6])
    graph = ir.Graph(
        [x, w],
        [y],
        nodes=[ir.Node("custom", "Block", [x, w], outputs=[y])],
        opset_imports={"": 21, "custom": 1},
        name="g",
    )
    _legalize(graph, [function])

    assert _ops(graph) == ["Block"]
    assert [node.op_type for node in function] == ["Cast", "Cast", "Conv", "Cast"]
    assert [value.dtype for value in function.inputs] == [BF16, BF16]
    assert [(value.name, value.dtype) for value in function.outputs] == [("fy", BF16)]


def test_new_names_do_not_collide_with_existing_ones() -> None:
    x = _value("x", [1, 3, 8, 8])
    w = _value("w", [4, 3, 3, 3])
    taken = _value("x_as_float32", [1, 3, 8, 8])
    y = _value("y", [1, 4, 6, 6])
    graph = _graph([x, w], [y, taken], 21)
    tape = Tape(graph)
    tape.op("Identity", [x], output=taken)
    tape.op("Conv", [x, w], output=y)
    _legalize(graph)

    names = [value.name for node in graph for value in node.outputs]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    ("op_type", "opset", "required"),
    [("RandomNormal", 21, "opset >= 22"), ("ConstantOfShape", 19, "opset >= 20")],
)
def test_generators_typed_by_an_attribute_are_rejected(
    op_type: str, opset: int, required: str
) -> None:
    y = _value("y", [2])
    if op_type == "RandomNormal":
        graph = _graph([], [y], opset)
        Tape(graph).op(
            op_type, [], output=y, attributes={"dtype": int(BF16), "shape": [2]}
        )
    else:
        shape = ir.val("shape", dtype=ir.DataType.INT64, shape=[1])
        graph = _graph([shape], [y], opset)
        Tape(graph).op(op_type, [shape], output=y)

    with pytest.raises(NotImplementedError, match=required) as error:
        legalize_bfloat16_operators(ir.Model(graph, ir_version=10))
    assert op_type in str(error.value) and f"opset {opset}" in str(error.value)


def test_bfloat16_through_control_flow_before_opset_16_is_rejected() -> None:
    condition = ir.val("condition", dtype=ir.DataType.BOOL, shape=[])
    x = _value("x", [2])
    y = _value("y", [2])

    def branch(name: str) -> ir.Graph:
        out = _value(f"{name}_out", [2])
        body = ir.Graph([], [out], nodes=[], name=name)
        Tape(body).op("Identity", [x], output=out)
        return body

    graph = _graph([condition, x], [y], 15)
    Tape(graph).op(
        "If",
        [condition],
        output=y,
        attributes={"then_branch": branch("then"), "else_branch": branch("else")},
    )
    with pytest.raises(NotImplementedError, match="opset >= 16"):
        legalize_bfloat16_operators(ir.Model(graph, ir_version=10))


def test_operator_that_never_admits_bfloat16_is_computed_in_float32() -> None:
    x = _value("x", [4])
    y = _value("y", [4])
    graph = _graph([x], [y], 23)
    Tape(graph).op("Shrink", [x], output=y)
    _legalize(graph)

    # Shrink shares its type parameter between input and output.
    assert _ops(graph) == ["Cast", "Shrink", "Cast"]


def test_rejection_without_any_admitting_opset_names_no_opset() -> None:
    x = _value("x", [4])
    sequence = ir.Value(
        name="sequence", type=ir.SequenceType(ir.TensorType(ir.DataType.FLOAT))
    )
    graph = _graph([x], [sequence], 23)
    Tape(graph).op("SequenceConstruct", [x], output=sequence)

    with pytest.raises(NotImplementedError, match="no ONNX opset known"):
        legalize_bfloat16_operators(ir.Model(graph, ir_version=10))
