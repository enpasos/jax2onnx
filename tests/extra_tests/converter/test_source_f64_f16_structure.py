# tests/extra_tests/converter/test_source_f64_f16_structure.py

"""Independent final-IR checks for an explicitly requested RNE math graph.

These contracts are declared before emission. They neither identify a source
compiler's cast semantics nor authorize any SOURCE algorithm or registration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Any

import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.source_structure import (
    LiteralOperand,
    StructureMismatch,
    StructureUnavailable,
)
from jax2onnx.plugins.jax._f64_to_f16_double_round import emit_f64_to_f16_double_round
from jax2onnx.plugins.jax._f64_to_f16_rne import emit_f64_to_f16_rne
from jax2onnx.plugins.jax._source_f64_f16_structure import (
    F64F16DoubleRoundSpec,
    F64F16RneSpec,
    RneGraphMatch,
    check_f64_f16_double_round_graph,
    check_f64_f16_rne_graph,
)


@dataclass
class _Case:
    ctx: IRContext
    spec: F64F16RneSpec
    source: ir.Value
    result: ir.Value
    expected_literals: tuple[bytes, ...]

    @property
    def graph(self) -> ir.Graph:
        return self.ctx.builder.graph


@dataclass
class _CompositionCase:
    ctx: IRContext
    spec: F64F16DoubleRoundSpec
    source: ir.Value
    result: ir.Value
    expected_literals: tuple[bytes, ...]

    @property
    def graph(self) -> ir.Graph:
        return self.ctx.builder.graph


def _case(shape: tuple[int, ...] = (6,), *, function: bool = False) -> _Case:
    # Mathematical input and literal obligations precede every emitted node.
    spec = F64F16RneSpec(shape)
    expected = (struct.pack("<d", 2.0**24),) + tuple(
        raw
        for exponent in range(-13, 16)
        for raw in (
            struct.pack("<d", 2.0**exponent),
            struct.pack("<d", 2.0 ** (10 - exponent)),
        )
    )
    parent = IRContext(opset=23, enable_double_precision=False)
    ctx = parent.create_child_context(scope="function") if function else parent
    source = ir.Value(
        name="original_input",
        type=ir.TensorType(ir.DataType.DOUBLE),
        shape=ir.Shape(shape),
    )
    ctx.builder.inputs.append(source)
    assert not tuple(ctx.builder.graph) and not ctx.builder.initializers
    result = emit_f64_to_f16_rne(ctx, source)
    ctx.builder.outputs.append(result)
    return _Case(ctx, spec, source, result, expected)


def _nodes(case: _Case | _CompositionCase, kind: str) -> list[ir.Node]:
    return [node for node in case.graph if node.op_type == kind]


def _literal_roles(case: _Case | _CompositionCase) -> tuple[ir.Value, ...]:
    comparisons, selections = _nodes(case, "GreaterOrEqual"), _nodes(case, "Where")
    assert len(comparisons) == len(selections) == 29
    values = [selections[0].inputs[2]]
    for comparison, selection in zip(comparisons, selections):
        values.extend((comparison.inputs[1], selection.inputs[1]))
    assert len(values) == 59 and all(type(value) is ir.Value for value in values)
    return tuple(values)


def _check(case: _Case) -> RneGraphMatch:
    return check_f64_f16_rne_graph(case.spec, case.graph, case.source, case.result)


def _good(case: _Case) -> RneGraphMatch:
    result = _check(case)
    assert result.nodes == tuple(case.graph)
    assert len([node for node in result.nodes if node.op_type != "Constant"]) == 63
    assert len(result.literal_obligations) == 59
    assert {id(value) for value in result.constants} == {
        id(value) for value, _ in result.literal_obligations
    }
    assert [value for value, _ in result.literal_obligations] == list(
        _literal_roles(case)
    )
    for (_, literal), expected in zip(
        result.literal_obligations, case.expected_literals
    ):
        assert type(literal) is LiteralOperand
        assert literal.tensor.dtype == ir.DataType.DOUBLE
        assert literal.tensor.shape == literal.storage.shape == ()
        assert literal.storage.dtype == "<f8" and literal.storage.byte_count == 8
        assert literal.storage.sha256 == hashlib.sha256(expected).hexdigest()
    return result


def _set_literal(value: ir.Value, tensor: ir.Tensor) -> None:
    producer = value.producer()
    if producer is None:
        value.const_value = tensor
    else:
        assert producer.op_type == "Constant"
        producer.attributes["value"] = ir.AttrTensor("value", tensor)


@pytest.mark.parametrize("shape", [(), (1,), (6,), (3, 7), (0, 3)])
@pytest.mark.parametrize(
    "function", [False, True], ids=["initializers", "Constant-nodes"]
)
def test_actual_emitter_graph_matches_independently_declared_math(
    shape: tuple[int, ...],
    function: bool,
) -> None:
    case = _case(shape, function=function)
    match = _good(case)
    assert len(match.constants) == 59
    if function:
        assert not case.graph.initializers and len(_nodes(case, "Constant")) == 59
    else:
        assert len(case.graph.initializers) == 59


def test_names_are_not_mathematical_or_compiler_authority() -> None:
    case = _case()
    _good(case)
    for index, node in enumerate(case.graph):
        node.name = f"unrelated_{index}"
        node.metadata_props["helper_name"] = "not_an_rne_helper"
    for index, value in enumerate(
        [case.source, *[v for node in case.graph for v in node.outputs]]
    ):
        value.name = f"arbitrary_value_{index}"
    _good(case)


def test_one_exact_output_facade_identity_is_covered() -> None:
    case = _case()
    _good(case)
    alias = case.ctx.builder.op(
        "Identity",
        [case.result],
        output=ir.Value(
            name="facade",
            type=ir.TensorType(ir.DataType.FLOAT16),
            shape=ir.Shape(case.spec.shape),
        ),
    )
    case.graph.outputs[0] = alias
    match = _check(case)
    assert len(match.nodes) == 64 and match.nodes == tuple(case.graph)
    assert len(match.literal_obligations) == 59


@pytest.mark.parametrize(
    "function", [False, True], ids=["initializers", "Constant-nodes"]
)
def test_checking_is_observational_and_does_not_rewrite_ir(function: bool) -> None:
    case = _case(function=function)
    model = ir.Model(case.graph, ir_version=10)
    before = ir.to_proto(model).SerializeToString(deterministic=True)
    nodes_before = tuple(case.graph)
    meta_before = dict(case.graph.meta)
    _good(case)
    assert ir.to_proto(model).SerializeToString(deterministic=True) == before
    assert tuple(case.graph) == nodes_before and dict(case.graph.meta) == meta_before


def test_a_second_output_facade_does_not_expand_the_checked_algorithm() -> None:
    case = _case()
    _good(case)
    output = case.result
    for index in range(2):
        output = case.ctx.builder.op(
            "Identity",
            [output],
            output=ir.Value(
                name=f"facade_{index}",
                type=ir.TensorType(ir.DataType.FLOAT16),
                shape=ir.Shape(case.spec.shape),
            ),
        )
    case.graph.outputs[0] = output
    with pytest.raises(StructureMismatch):
        _check(case)


def test_computational_producer_after_consumer_is_not_a_valid_final_graph() -> None:
    case = _case()
    _good(case)
    rounded, divided = _nodes(case, "Round")[0], _nodes(case, "Div")[0]
    before = {
        id(node): (
            tuple(id(value) for value in node.inputs),
            tuple(id(value) for value in node.outputs),
        )
        for node in case.graph
    }
    case.graph.insert_after(divided, rounded)
    assert len(tuple(case.graph)) == 63
    assert tuple(case.graph).index(divided) < tuple(case.graph).index(rounded)
    assert divided.inputs[0] is rounded.outputs[0]
    assert rounded.graph is divided.graph is case.graph
    assert before == {
        id(node): (
            tuple(id(value) for value in node.inputs),
            tuple(id(value) for value in node.outputs),
        )
        for node in case.graph
    }
    with pytest.raises(StructureMismatch):
        _check(case)


def test_function_constant_after_first_consumer_is_not_a_valid_final_graph() -> None:
    case = _case(function=True)
    _good(case)
    literal = _literal_roles(case)[0]
    constant = literal.producer()
    assert constant is not None and constant.op_type == "Constant"
    first_consumer = _nodes(case, "Where")[0]
    before = {
        id(node): (
            tuple(id(value) for value in node.inputs),
            tuple(id(value) for value in node.outputs),
        )
        for node in case.graph
    }
    case.graph.insert_after(first_consumer, constant)
    assert len(tuple(case.graph)) == 122
    assert tuple(case.graph).index(first_consumer) < tuple(case.graph).index(constant)
    assert first_consumer.inputs[2] is literal and literal.producer() is constant
    assert constant.graph is first_consumer.graph is case.graph
    assert before == {
        id(node): (
            tuple(id(value) for value in node.inputs),
            tuple(id(value) for value in node.outputs),
        )
        for node in case.graph
    }
    with pytest.raises(StructureMismatch):
        _check(case)


def test_exact_storage_dedup_preserves_all_fifty_nine_literal_roles() -> None:
    case = _case()
    _good(case)
    canonical: dict[bytes, ir.Value] = {}
    for value in list(case.graph.initializers.values()):
        assert value.const_value is not None
        storage = value.const_value.numpy().tobytes()
        if storage not in canonical:
            canonical[storage] = value
            continue
        for node in tuple(case.graph):
            for index, operand in enumerate(node.inputs):
                if operand is value:
                    node.replace_input_with(index, canonical[storage])
        del case.graph.initializers[value.name]
    match = _good(case)
    assert len(match.constants) == len(case.graph.initializers) == 38
    assert len(match.literal_obligations) == 59


@pytest.mark.parametrize("role", range(59))
def test_each_literal_role_requires_exact_double_storage(role: int) -> None:
    case = _case()
    _good(case)
    value = _literal_roles(case)[role]
    original = int.from_bytes(case.expected_literals[role], "little")
    changed = np.asarray(original ^ 1, dtype="<u8").view("<f8")
    value.const_value = ir.tensor(changed)
    with pytest.raises(StructureMismatch):
        _check(case)


@pytest.mark.parametrize(
    "fault", ["subnormal-unit", "overflow-threshold", "wrong-sharing"]
)
def test_subnormal_overflow_and_wrong_sharing_are_contradictions(fault: str) -> None:
    case = _case()
    _good(case)
    selections, comparisons = _nodes(case, "Where"), _nodes(case, "GreaterOrEqual")
    if fault == "subnormal-unit":
        selections[0].inputs[2].const_value = ir.tensor(
            np.asarray(2.0**23, dtype=np.float64)
        )
    elif fault == "overflow-threshold":
        comparisons[-1].inputs[1].const_value = ir.tensor(
            np.asarray(2.0**16, dtype=np.float64)
        )
    else:
        comparisons[0].replace_input_with(1, selections[0].inputs[1])
    with pytest.raises(StructureMismatch):
        _check(case)


@pytest.mark.parametrize(
    "fault",
    [
        "comparison-ports",
        "selection-branches",
        "multiply-ports",
        "divide-ports",
        "multiply-scale",
        "divide-scale",
        "round-bypass",
        "absolute-bypass",
        "selection-chain-cut",
        "round-replacement",
        "round-domain",
        "round-overload",
        "round-extra-attribute",
        "cast-target",
    ],
)
def test_ordered_ports_and_operators_require_the_complete_math_chain(
    fault: str,
) -> None:
    case = _case()
    _good(case)
    comparisons, selections = _nodes(case, "GreaterOrEqual"), _nodes(case, "Where")
    multiply, rounded, divided = (
        _nodes(case, name)[0] for name in ("Mul", "Round", "Div")
    )
    if fault == "comparison-ports":
        left, right = comparisons[4].inputs
        comparisons[4].replace_input_with(0, right)
        comparisons[4].replace_input_with(1, left)
    elif fault == "selection-branches":
        left, right = selections[4].inputs[1:]
        selections[4].replace_input_with(1, right)
        selections[4].replace_input_with(2, left)
    elif fault == "multiply-ports":
        left, right = multiply.inputs
        multiply.replace_input_with(0, right)
        multiply.replace_input_with(1, left)
    elif fault == "divide-ports":
        left, right = divided.inputs
        divided.replace_input_with(0, right)
        divided.replace_input_with(1, left)
    elif fault == "multiply-scale":
        multiply.replace_input_with(1, selections[-2].outputs[0])
    elif fault == "divide-scale":
        divided.replace_input_with(1, selections[-2].outputs[0])
    elif fault == "round-bypass":
        divided.replace_input_with(0, multiply.outputs[0])
    elif fault == "absolute-bypass":
        comparisons[9].replace_input_with(0, case.source)
    elif fault == "selection-chain-cut":
        selections[15].replace_input_with(2, selections[10].outputs[0])
    elif fault == "round-replacement":
        rounded.op_type = "Floor"
        assert len(tuple(case.graph)) == 63
    elif fault == "round-domain":
        rounded.domain = "unrelated.domain"
    elif fault == "round-overload":
        rounded.overload = "unreviewed"
    elif fault == "round-extra-attribute":
        rounded.attributes["unknown"] = ir.AttrInt64("unknown", 1)
    else:
        case.result.producer().attributes["to"] = ir.AttrInt64(
            "to", int(ir.DataType.FLOAT.value)
        )
    with pytest.raises(StructureMismatch):
        _check(case)


@pytest.mark.parametrize(
    "fault",
    ["extra-node", "extra-initializer", "extra-input", "extra-output", "missing-round"],
)
def test_whole_graph_coverage_rejects_unaccounted_or_missing_definitions(
    fault: str,
) -> None:
    case = _case()
    _good(case)
    if fault == "extra-node":
        case.ctx.builder.op(
            "Identity",
            [case.source],
            output=ir.Value(
                name="unused",
                type=ir.TensorType(ir.DataType.DOUBLE),
                shape=ir.Shape(case.spec.shape),
            ),
        )
    elif fault == "extra-initializer":
        case.ctx.bind_const_for_var(object(), np.asarray(7.0, dtype=np.float64))
    elif fault == "extra-input":
        case.graph.inputs.append(
            ir.Value(
                name="extra",
                type=ir.TensorType(ir.DataType.DOUBLE),
                shape=ir.Shape(case.spec.shape),
            )
        )
    elif fault == "extra-output":
        case.graph.outputs.append(case.source)
    else:
        case.graph.remove(_nodes(case, "Round")[0])
    with pytest.raises(StructureMismatch):
        _check(case)


@pytest.mark.parametrize(
    "part", ["source", "result", "round", "predicate", "threshold"]
)
@pytest.mark.parametrize("fault", ["type", "shape"])
def test_known_tensor_metadata_contradictions_fail(part: str, fault: str) -> None:
    case = _case()
    _good(case)
    value = {
        "source": case.source,
        "result": case.result,
        "round": _nodes(case, "Round")[0].outputs[0],
        "predicate": _nodes(case, "GreaterOrEqual")[0].outputs[0],
        "threshold": _nodes(case, "GreaterOrEqual")[0].inputs[1],
    }[part]
    if fault == "type":
        value.type = ir.TensorType(ir.DataType.INT64)
    else:
        value.shape = ir.Shape((1, 6))
    with pytest.raises(StructureMismatch):
        _check(case)


@pytest.mark.parametrize(
    "part", ["source", "result", "round", "predicate", "threshold"]
)
@pytest.mark.parametrize("fault", ["type", "shape"])
def test_missing_tensor_metadata_is_unavailable(part: str, fault: str) -> None:
    case = _case()
    _good(case)
    value = {
        "source": case.source,
        "result": case.result,
        "round": _nodes(case, "Round")[0].outputs[0],
        "predicate": _nodes(case, "GreaterOrEqual")[0].outputs[0],
        "threshold": _nodes(case, "GreaterOrEqual")[0].inputs[1],
    }[part]
    if fault == "type":
        value.type = None
    else:
        value.shape = None
    with pytest.raises(StructureUnavailable):
        _check(case)


@pytest.mark.parametrize("part", ["source", "result", "literal"])
def test_equal_typed_values_from_another_graph_do_not_supply_ownership(
    part: str,
) -> None:
    case, other = _case(), _case()
    _good(case)
    _good(other)
    if part == "source":
        arguments = (other.source, case.result)
    elif part == "result":
        arguments = (case.source, other.result)
    else:
        _nodes(case, "GreaterOrEqual")[0].replace_input_with(
            1, _nodes(other, "GreaterOrEqual")[0].inputs[1]
        )
        arguments = (case.source, case.result)
    with pytest.raises(StructureMismatch):
        check_f64_f16_rne_graph(case.spec, case.graph, *arguments)


class _ArrayBacking:
    """Public numpy-compatible backing, permitting independent IR declarations."""

    def __init__(self, value: np.ndarray, calls: list[str]) -> None:
        self.value, self.calls = value, calls
        self.shape = value.shape

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        self.calls.append("array")
        converted = self.value.__array__(dtype)
        return converted.copy() if copy else converted


@pytest.mark.parametrize(
    "function", [False, True], ids=["initializer", "Constant-attribute"]
)
@pytest.mark.parametrize(
    "fault", ["declared-dtype", "declared-shape", "backing-dtype", "backing-shape"]
)
def test_real_tensor_declarations_and_backing_storage_must_both_match(
    function: bool,
    fault: str,
) -> None:
    case = _case(function=function)
    _good(case)
    value = _literal_roles(case)[0]
    calls: list[str] = []
    if fault in ("declared-dtype", "backing-dtype"):
        backing = np.asarray(
            2.0**24, dtype=np.float64 if fault == "declared-dtype" else np.float32
        )
        tensor = ir.Tensor(
            _ArrayBacking(backing, calls),
            dtype=(
                ir.DataType.FLOAT if fault == "declared-dtype" else ir.DataType.DOUBLE
            ),
            shape=ir.Shape(()),
        )
        assert tensor.numpy().dtype == backing.dtype
        assert tensor.dtype == (
            ir.DataType.FLOAT if fault == "declared-dtype" else ir.DataType.DOUBLE
        )
    elif fault == "declared-shape":
        tensor = ir.Tensor(np.asarray(2.0**24, dtype=np.float64), shape=ir.Shape((1,)))
        assert tensor.shape == ir.Shape((1,)) and tensor.numpy().shape == ()
    else:
        tensor = ir.Tensor(np.asarray([2.0**24], dtype=np.float64), shape=ir.Shape(()))
        assert tensor.shape == ir.Shape(()) and tensor.numpy().shape == (1,)
    _set_literal(value, tensor)
    calls.clear()
    with pytest.raises(StructureMismatch):
        _check(case)
    if fault == "declared-dtype":
        assert calls == []  # Native IR metadata rejects before reading the backing.


def test_function_constant_attribute_is_authoritative_over_pristine_cache() -> None:
    case = _case(function=True)
    _good(case)
    value = _literal_roles(case)[0]
    assert value.const_value is not None
    original_cache = value.const_value.numpy().tobytes()
    _set_literal(value, ir.tensor(np.asarray(2.0**23, dtype=np.float64)))
    assert (
        value.const_value.numpy().tobytes()
        == original_cache
        == case.expected_literals[0]
    )
    with pytest.raises(StructureMismatch):
        _check(case)


def test_missing_initializer_storage_is_unavailable() -> None:
    case = _case()
    _good(case)
    _literal_roles(case)[0].const_value = None
    with pytest.raises(StructureUnavailable):
        _check(case)


@pytest.mark.parametrize(
    "function", [False, True], ids=["initializer", "Constant-attribute"]
)
def test_unreadable_external_storage_is_unavailable(
    function: bool,
    tmp_path: Path,
) -> None:
    case = _case(function=function)
    _good(case)
    tensor = ir.ExternalTensor(
        location="absent.bin",
        offset=0,
        length=8,
        dtype=ir.DataType.DOUBLE,
        shape=ir.Shape(()),
        name="unavailable_literal",
        base_dir=tmp_path,
    )
    _set_literal(_literal_roles(case)[0], tensor)
    with pytest.raises(StructureUnavailable):
        _check(case)


def test_a_literal_runtime_input_is_not_an_immutable_scalar_obligation() -> None:
    case = _case()
    _good(case)
    literal = _literal_roles(case)[0]
    case.graph.inputs.append(literal)
    assert literal.const_value is not None
    with pytest.raises(StructureMismatch):
        _check(case)


def test_lost_initializer_definition_cannot_be_replaced_by_cached_bytes() -> None:
    case = _case()
    _good(case)
    literal = _literal_roles(case)[0]
    del case.graph.initializers[literal.name]
    assert literal.const_value is not None
    with pytest.raises((StructureMismatch, StructureUnavailable)):
        _check(case)


@pytest.mark.parametrize("fault", ["missing", "below-floor"])
def test_unknown_or_undefined_default_opset_cannot_validate(fault: str) -> None:
    case = _case()
    _good(case)
    if fault == "missing":
        del case.graph.opset_imports[""]
        expected = StructureUnavailable
    else:
        case.graph.opset_imports[""] = 11
        expected = StructureMismatch
    with pytest.raises(expected):
        _check(case)


@pytest.mark.parametrize("shape,forged", [((1,), (True,)), ((0,), (False,))])
def test_forged_frozen_spec_does_not_accept_bool_as_integer(
    shape: tuple[int, ...],
    forged: tuple[bool, ...],
) -> None:
    case = _case(shape)
    _good(case)
    object.__setattr__(case.spec, "shape", forged)
    with pytest.raises(StructureMismatch):
        _check(case)


@pytest.mark.parametrize(
    "fault", ["bool", "numpy-int", "text", "negative", "list", "foreign"]
)
def test_math_spec_dimensions_require_exact_native_nonnegative_integers(
    fault: str,
) -> None:
    calls = []

    class Foreign:
        def __int__(self) -> int:
            calls.append("int")
            raise AssertionError("foreign dimension was coerced")

        def __index__(self) -> int:
            calls.append("index")
            raise AssertionError("foreign dimension was indexed")

        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            raise AssertionError("foreign dimension was compared")

    shape = {
        "bool": (True,),
        "numpy-int": (np.int64(6),),
        "text": ("6",),
        "negative": (-1,),
        "list": [6],
        "foreign": (Foreign(),),
    }[fault]
    with pytest.raises(ValueError):
        F64F16RneSpec(shape)
    assert calls == []


def _composition_case(
    shape: tuple[int | str, ...] | None = (6,),
    *,
    function: bool = False,
    opset: int = 23,
) -> _CompositionCase:
    # The spec and literal storage obligations precede emission, independently.
    spec = F64F16DoubleRoundSpec(shape)
    expected = (struct.pack("<d", 2.0**24),) + tuple(
        raw
        for exponent in range(-13, 16)
        for raw in (
            struct.pack("<d", 2.0**exponent),
            struct.pack("<d", 2.0 ** (10 - exponent)),
        )
    )
    parent = IRContext(opset=opset, enable_double_precision=False)
    ctx = parent.create_child_context(scope="function") if function else parent
    source = ir.Value(
        name="original_input",
        type=ir.TensorType(ir.DataType.DOUBLE),
        shape=ir.Shape(shape) if shape is not None else None,
    )
    ctx.builder.inputs.append(source)
    result = emit_f64_to_f16_double_round(ctx, source)
    ctx.builder.outputs.append(result)
    return _CompositionCase(ctx, spec, source, result, expected)


def _composition_check(case: _CompositionCase, *, saved: bool = False) -> RneGraphMatch:
    return check_f64_f16_double_round_graph(
        case.spec,
        case.graph,
        case.source,
        case.result,
        allow_cast_schema_defaults=saved,
    )


def _composition_good(case: _CompositionCase) -> RneGraphMatch:
    match = _composition_check(case)
    assert match.nodes == tuple(case.graph)
    assert len([node for node in match.nodes if node.op_type != "Constant"]) == 65
    assert len(match.literal_obligations) == len(case.expected_literals) == 59
    assert [value for value, _ in match.literal_obligations] == list(
        _literal_roles(case)
    )
    for (_, literal), expected in zip(
        match.literal_obligations, case.expected_literals
    ):
        assert type(literal) is LiteralOperand
        assert literal.tensor.dtype == ir.DataType.DOUBLE
        assert literal.tensor.shape == literal.storage.shape == ()
        assert literal.storage.dtype == "<f8" and literal.storage.byte_count == 8
        assert literal.storage.sha256 == hashlib.sha256(expected).hexdigest()
    first, wide = [node for node in match.nodes if node.op_type == "Cast"][:2]
    assert first.inputs[0] is case.source and wide.inputs[0] is first.outputs[0]
    assert first.outputs[0].dtype == ir.DataType.FLOAT
    assert wide.outputs[0].dtype == ir.DataType.DOUBLE
    assert _nodes(case, "Abs")[0].inputs[0] is wide.outputs[0]
    assert _nodes(case, "Mul")[0].inputs[0] is wide.outputs[0]
    return match


@pytest.mark.parametrize("shape", [(), (0,), (6,), ("N", 7), None])
@pytest.mark.parametrize(
    "function", [False, True], ids=["initializers", "Constant-nodes"]
)
def test_composed_graph_requires_the_float_rounding_and_complete_half_math(
    shape: tuple[int | str, ...] | None, function: bool
) -> None:
    case = _composition_case(shape, function=function)
    match = _composition_good(case)
    assert len(match.constants) == 59
    assert len(match.nodes) == (124 if function else 65)


@pytest.mark.parametrize("source", ["original", "widened"])
def test_direct_wrapper_keeps_its_original_input_and_whole_graph_contract(
    source: str,
) -> None:
    case = _composition_case()
    _composition_good(case)
    value = case.source if source == "original" else _nodes(case, "Cast")[1].outputs[0]
    with pytest.raises(StructureMismatch):
        check_f64_f16_rne_graph(F64F16RneSpec((6,)), case.graph, value, case.result)


@pytest.mark.parametrize("opset", [12, 21, 23])
@pytest.mark.parametrize("cast_index", [0, 1, 2])
def test_only_saved_supported_cast_schemas_accept_exact_saturate_default(
    opset: int, cast_index: int
) -> None:
    case = _composition_case(opset=opset)
    _composition_good(case)
    _nodes(case, "Cast")[cast_index].attributes["saturate"] = ir.AttrInt64(
        "saturate", 1
    )
    with pytest.raises(StructureMismatch):
        _composition_check(case)
    if opset < 19:
        with pytest.raises(StructureMismatch):
            _composition_check(case, saved=True)
    else:
        assert _composition_check(case, saved=True).nodes == tuple(case.graph)


@pytest.mark.parametrize(
    "name,value",
    [("saturate", 0), ("saturate", 2), ("unknown", 1), ("round_mode", "up")],
)
def test_saved_cast_default_policy_rejects_other_attributes_and_values(
    name: str, value: int | str
) -> None:
    case = _composition_case()
    _composition_good(case)
    _nodes(case, "Cast")[0].attributes[name] = (
        ir.AttrInt64(name, value)
        if isinstance(value, int)
        else ir.AttrString(name, value)
    )
    with pytest.raises(StructureMismatch):
        _composition_check(case, saved=True)


@pytest.mark.parametrize(
    "fault",
    [
        "narrow-target",
        "wide-target",
        "narrow-type",
        "wide-type",
        "bypass-narrow",
        "multiply-original",
        "absolute-original",
        "foreign-input",
        "narrow-shape",
        "wide-domain",
        "wide-overload",
        "extra-attribute",
    ],
)
def test_composed_prefix_requires_exact_owners_edges_types_shapes_and_casts(
    fault: str,
) -> None:
    case = _composition_case()
    _composition_good(case)
    first, wide = _nodes(case, "Cast")[:2]
    if fault == "narrow-target":
        first.attributes["to"] = ir.AttrInt64("to", int(ir.DataType.DOUBLE.value))
    elif fault == "wide-target":
        wide.attributes["to"] = ir.AttrInt64("to", int(ir.DataType.FLOAT.value))
    elif fault == "narrow-type":
        first.outputs[0].type = ir.TensorType(ir.DataType.DOUBLE)
    elif fault == "wide-type":
        wide.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    elif fault == "bypass-narrow":
        wide.replace_input_with(0, case.source)
    elif fault == "multiply-original":
        _nodes(case, "Mul")[0].replace_input_with(0, case.source)
    elif fault == "absolute-original":
        _nodes(case, "Abs")[0].replace_input_with(0, case.source)
    elif fault == "foreign-input":
        first.replace_input_with(0, _composition_case().source)
    elif fault == "narrow-shape":
        first.outputs[0].shape = ir.Shape((1, 6))
    elif fault == "wide-domain":
        wide.domain = "unreviewed"
    elif fault == "wide-overload":
        wide.overload = "unreviewed"
    else:
        wide.attributes["unknown"] = ir.AttrInt64("unknown", 1)
    with pytest.raises(StructureMismatch):
        _composition_check(case)


@pytest.mark.parametrize("role", range(59))
@pytest.mark.parametrize(
    "function", [False, True], ids=["initializers", "Constant-nodes"]
)
def test_composed_body_checks_every_literal_role_against_independent_storage(
    role: int, function: bool
) -> None:
    case = _composition_case(function=function)
    _composition_good(case)
    changed = int.from_bytes(case.expected_literals[role], "little") ^ 1
    tensor = ir.tensor(np.asarray(changed, dtype="<u8").view("<f8"))
    _set_literal(_literal_roles(case)[role], tensor)
    with pytest.raises(StructureMismatch):
        _composition_check(case)


@pytest.mark.parametrize(
    "fault",
    [
        "extra-node",
        "extra-initializer",
        "extra-input",
        "extra-output",
        "missing-round",
        "prefix-after-consumer",
    ],
)
def test_composed_whole_graph_and_topological_coverage_cannot_skip_definitions(
    fault: str,
) -> None:
    case = _composition_case()
    _composition_good(case)
    if fault == "extra-node":
        case.ctx.builder.op(
            "Identity",
            [case.source],
            output=ir.Value(
                name="unused",
                type=ir.TensorType(ir.DataType.DOUBLE),
                shape=ir.Shape((6,)),
            ),
        )
    elif fault == "extra-initializer":
        case.ctx.bind_const_for_var(object(), np.asarray(7.0, dtype=np.float64))
    elif fault == "extra-input":
        case.graph.inputs.append(
            ir.Value(
                name="extra",
                type=ir.TensorType(ir.DataType.DOUBLE),
                shape=ir.Shape((6,)),
            )
        )
    elif fault == "extra-output":
        case.graph.outputs.append(case.source)
    elif fault == "missing-round":
        case.graph.remove(_nodes(case, "Round")[0])
    else:
        case.graph.insert_after(_nodes(case, "Abs")[0], _nodes(case, "Cast")[1])
    with pytest.raises(StructureMismatch):
        _composition_check(case)


@pytest.mark.parametrize(
    "function", [False, True], ids=["initializers", "Constant-nodes"]
)
def test_composed_checker_is_observational(function: bool) -> None:
    case = _composition_case(function=function)
    model = ir.Model(case.graph, ir_version=10)
    before = ir.to_proto(model).SerializeToString(deterministic=True)
    _composition_good(case)
    assert ir.to_proto(model).SerializeToString(deterministic=True) == before


def test_composed_exact_storage_dedup_keeps_all_fifty_nine_literal_roles() -> None:
    case = _composition_case()
    _composition_good(case)
    canonical: dict[bytes, ir.Value] = {}
    for value in list(case.graph.initializers.values()):
        assert value.const_value is not None
        storage = value.const_value.numpy().tobytes()
        if storage not in canonical:
            canonical[storage] = value
            continue
        for node in tuple(case.graph):
            for index, operand in enumerate(node.inputs):
                if operand is value:
                    node.replace_input_with(index, canonical[storage])
        del case.graph.initializers[value.name]
    match = _composition_good(case)
    assert len(match.constants) == len(case.graph.initializers) == 38
    assert len(match.literal_obligations) == 59


def test_composed_function_constant_must_precede_its_first_consumer() -> None:
    case = _composition_case(function=True)
    _composition_good(case)
    constant = _literal_roles(case)[0].producer()
    assert constant is not None
    case.graph.insert_after(_nodes(case, "Where")[0], constant)
    with pytest.raises(StructureMismatch):
        _composition_check(case)


@pytest.mark.parametrize(
    "function", [False, True], ids=["initializers", "Constant-nodes"]
)
@pytest.mark.parametrize("facades", [1, 2])
def test_composed_graph_covers_one_facade_within_the_125_node_bound(
    function: bool, facades: int
) -> None:
    case = _composition_case(function=function)
    _composition_good(case)
    output = case.result
    for index in range(facades):
        output = case.ctx.builder.op(
            "Identity",
            [output],
            output=ir.Value(
                name=f"facade_{index}",
                type=ir.TensorType(ir.DataType.FLOAT16),
                shape=ir.Shape((6,)),
            ),
        )
    case.graph.outputs[0] = output
    if facades == 2:
        with pytest.raises(StructureMismatch):
            _composition_check(case)
    else:
        match = _composition_check(case)
        assert match.nodes == tuple(case.graph) and len(match.nodes) <= 125
        assert len(match.literal_obligations) == 59


@pytest.mark.parametrize(
    "fault", ["bool", "numpy-int", "empty-symbol", "negative", "list", "foreign"]
)
def test_composed_spec_rejects_dimensions_without_coercing_foreign_values(
    fault: str,
) -> None:
    calls: list[str] = []

    class Foreign:
        def __int__(self) -> int:
            calls.append("int")
            raise AssertionError("foreign dimension was coerced")

        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            raise AssertionError("foreign dimension was compared")

    shape: Any = {
        "bool": (True,),
        "numpy-int": (np.int64(6),),
        "empty-symbol": ("",),
        "negative": (-1,),
        "list": [6],
        "foreign": (Foreign(),),
    }[fault]
    with pytest.raises(ValueError):
        F64F16DoubleRoundSpec(shape)
    assert calls == []
