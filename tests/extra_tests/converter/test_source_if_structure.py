# tests/extra_tests/converter/test_source_if_structure.py

"""Real dynamic If regions, lexical edges and composed source selectors."""

from dataclasses import fields, is_dataclass, replace
from typing import Any

import jax
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import Jaxpr, JaxprEqn, Literal, Var
from jax2onnx.converter import ir_context, source_structure as structure
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.ir_utils import iter_ir_functions
from jax2onnx.plugins import plugin_system
from jax2onnx.plugins._source_control_flow_structure import BooleanIndexRule, IfRule
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)


_CALLS: list[tuple[str, tuple[Any, ...]]] = []
_PAYLOAD = np.asarray([0x80000000, 0x7FC00001], np.uint32).view(np.float32)


def _configuration() -> tuple[Any, ...]:
    return (
        bool(jax.config.x64_enabled),
        active_precision_policy(),
        jax.config.jax_default_matmul_precision,
        str(jax.config.jax_numpy_dtype_promotion),
        bool(jax.config.jax_debug_key_reuse),
        plugin_system._IN_FUNCTION_BUILD.get(),
    )


def _choice(flag: Any, left: Any, right: Any) -> Any:
    return jax.lax.cond(
        flag,
        lambda x, y: jax.lax.sub(x, y),
        lambda x, y: jax.lax.add(x, y),
        left,
        right,
    )


@onnx_function
def _function_choice(flag: Any, left: Any, right: Any) -> Any:
    _CALLS.append(("choice", _configuration()))
    return _choice(flag, left, right)


@onnx_function
def _function_difference(left: Any, right: Any) -> Any:
    _CALLS.append(("difference", _configuration()))
    return jax.lax.sub(left, right)


def _functions_in_branches(flag: Any, left: Any, right: Any) -> Any:
    return jax.lax.cond(
        flag,
        lambda x, y: _function_difference(x, y),
        lambda x, y: _function_difference(y, x),
        left,
        right,
    )


def _nested(flag: Any, other: Any, left: Any, right: Any) -> Any:
    return jax.lax.cond(
        flag,
        lambda x, y: _choice(other, x, y),
        lambda x, y: jax.lax.sub(y, x),
        left,
        right,
    )


def _source(
    function: Any = _choice,
    *,
    flags: int = 1,
    dtype: Any = np.float32,
    shape: tuple[int, ...] = (2,),
    operands: int = 2,
) -> ir.Model:
    specs = [jax.ShapeDtypeStruct((), np.bool_)] * flags
    specs += [jax.ShapeDtypeStruct(shape, dtype)] * operands
    with jax.enable_x64(True):
        model = ui._to_onnx(
            function, specs, precision_policy=PrecisionPolicy.SOURCE, return_mode="ir"
        )
    assert isinstance(model, ir.Model)
    return model


def _proof(model: ir.Model) -> structure.StructureResult:
    return structure.validate_source_structure(model)


def _graphs(model: ir.Model) -> list[ir.Graph]:
    result: list[ir.Graph] = []

    def visit(graph: ir.Graph) -> None:
        result.append(graph)
        for node in graph:
            for attribute in node.attributes.values():
                if attribute.type is ir.AttributeType.GRAPH:
                    visit(attribute.as_graph())

    visit(model.graph)
    for definition in iter_ir_functions(model.functions):
        visit(definition.graph)
    return result


def _if_nodes(model: ir.Model) -> list[ir.Node]:
    return [
        node
        for graph in _graphs(model)
        for node in graph
        if node.op_type == "If" and not node.domain
    ]


def _validated(model: ir.Model, expected: int | None = None) -> Any:
    result = _proof(model)
    assert result.status == "validated", result
    assert result.expected_equations == result.checked_equations
    if expected is not None:
        assert result.expected_equations == expected
    assert result.runtime == result.numerics == "unverified"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    return model.meta[structure._LEDGER]


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("shape", ((), (2,), (2, 3)))
def test_dynamic_if_checks_both_original_children_and_folded_selector(
    dtype: Any, shape: tuple[int, ...]
) -> None:
    model = _source(dtype=dtype, shape=shape)
    ledger = _validated(model, 4)
    assert len(ledger.if_calls) == 1 and len(ledger.regions) == 2
    assert {key.role for key in ledger.regions} == {"false", "true"}
    assert len(ledger.scopes) == 3
    node = _if_nodes(model)[0]
    assert node.inputs[0] is model.graph.inputs[0]
    assert all(node.op_type != "Cast" for node in model.graph)
    for attribute in node.attributes.values():
        assert not attribute.as_graph().inputs
    root = ledger.scopes[0]
    assert (
        sum(type(equation.rule) is BooleanIndexRule for equation in root.equations) == 1
    )
    assert sum(type(equation.rule) is IfRule for equation in root.equations) == 1


@pytest.mark.parametrize(
    "kind", ("inside-function", "functions-in-branches", "nested", "two-invocations")
)
def test_regions_and_calls_are_expected_independently_of_physical_definitions(
    kind: str,
) -> None:
    if kind == "inside-function":
        model = _source(lambda flag, left, right: _function_choice(flag, left, right))
        ledger = _validated(model, 5)
        assert len(ledger.function_calls) == 1
        assert _if_nodes(model)[0].inputs[0].dtype is ir.DataType.BOOL
    elif kind == "functions-in-branches":
        model = _source(_functions_in_branches)
        ledger = _validated(model, 6)
        assert len(ledger.function_calls) == 2
        assert len(list(iter_ir_functions(model.functions))) == 1
    elif kind == "nested":
        model = _source(_nested, flags=2)
        ledger = _validated(model, 7)
        nodes = _if_nodes(model)
        inner = next(node for node in nodes if node.graph is not model.graph)
        assert inner.inputs[0].producer().op_type == "Cast"
        assert inner.inputs[0].producer().inputs[0].producer().op_type == "Cast"
    else:
        model = _source(
            lambda flag, other, x, y: (_choice(flag, x, y), _choice(other, y, x)),
            flags=2,
        )
        ledger = _validated(model, 8)
    assert len(ledger.regions) == 2 * len(ledger.if_calls)


def test_repeated_passthrough_literal_and_mixed_dtype_outputs_keep_order() -> None:
    def target(flag: Any, x: Any, y: Any) -> Any:
        def true(left: Any, right: Any) -> Any:
            difference = jax.lax.sub(left, right)
            return (
                difference,
                left,
                difference,
                np.float32(-0.0),
                difference.astype(np.float64),
            )

        def false(left: Any, right: Any) -> Any:
            total = jax.lax.add(left, right)
            return total, right, total, np.float32(2.0), total.astype(np.float64)

        return jax.lax.cond(flag, true, false, x, y)

    model = _source(target)
    _validated(model, 6)
    assert len(model.graph.outputs) == 5
    assert model.graph.outputs[-1].dtype is ir.DataType.DOUBLE
    for node in _if_nodes(model):
        for attr in node.attributes.values():
            assert len(attr.as_graph().outputs) == 5


def test_unused_capture_slot_stays_expected_after_recursive_dce() -> None:
    model = _source(
        lambda flag, x, y: jax.lax.cond(
            flag, lambda a, b: jax.lax.neg(a), lambda a, b: jax.lax.neg(b), x, y
        )
    )
    ledger = _validated(model, 4)
    scopes = [scope for scope in ledger.scopes if scope.region is not None]
    assert len(scopes) == 2
    assert all(len(scope.region.captures) == 2 for scope in scopes)
    assert all(
        sum(port.unused for port in scope.region.captures) == 1 for scope in scopes
    )
    assert all(len(scope.inputs) == 2 for scope in scopes)


def test_zero_operand_children_have_zero_formals_and_ordered_literal_results() -> None:
    model = _source(
        lambda flag: jax.lax.cond(
            flag, lambda: np.float32(1.0), lambda: np.float32(-2.0)
        ),
        operands=0,
    )
    ledger = _validated(model, 2)
    assert all(
        not scope.region.captures for scope in ledger.scopes if scope.region is not None
    )


def test_parent_literal_capture_retains_raw_storage_and_lexical_definition() -> None:
    def target(flag: Any, left: Any, unused: Any) -> Any:
        return jax.lax.cond(
            flag,
            lambda x, y: jax.lax.sub(x, y),
            lambda x, y: jax.lax.add(x, y),
            left,
            np.float32(-0.0),
        )

    model = _source(target, shape=())
    ledger = _validated(model, 4)
    assert all(
        any(
            isinstance(port.binding, structure.LiteralOperand)
            for port in scope.region.captures
        )
        for scope in ledger.scopes
        if scope.region is not None
    )


@pytest.mark.parametrize("fault", ("none", "missing", "wrong-cast"))
def test_observed_int32_selector_keeps_its_original_cast_obligation(fault: str) -> None:
    spec = jax.ShapeDtypeStruct((2,), np.float32)
    false = jax.make_jaxpr(jax.lax.add)(spec, spec)
    true = jax.make_jaxpr(jax.lax.sub)(spec, spec)

    def target(flag: Any, left: Any, right: Any) -> Any:
        index = jax.lax.convert_element_type(flag, np.int32)
        (value,) = jax.lax.cond_p.bind(index, left, right, branches=(false, true))
        return value, index

    model = _source(target)
    _validated(model, 4)
    cast = model.graph.outputs[1].producer()
    assert cast is not None and cast.op_type == "Cast"
    assert cast.attributes["to"].as_int() == int(ir.DataType.INT32.value)
    assert _if_nodes(model)[0].inputs[0] is model.graph.inputs[0]
    if fault == "missing":
        model.graph.remove(cast)
        result = _proof(model)
        assert result.status == "unverified" and result.reasons == (
            "final-value-witness-lost",
        )
    elif fault == "wrong-cast":
        cast.attributes["to"] = ir.AttrInt64("to", int(ir.DataType.INT64.value))
        result = _proof(model)
        assert result.status == "failed" and result.reasons == (
            "boolean-index-cast-differs",
        )


@pytest.mark.parametrize(
    "fault",
    (
        "selector",
        "branch-roles",
        "branch-op",
        "capture-order",
        "same-name",
        "output-order",
        "extra-attribute",
    ),
)
def test_resolved_if_structure_corruption_is_failed(fault: str) -> None:
    model = _source()
    _validated(model, 4)
    node = _if_nodes(model)[0]
    true_graph = node.attributes["then_branch"].as_graph()
    if fault == "selector":
        unrelated = ir.Value(
            name=model.graph.inputs[0].name,
            type=ir.TensorType(ir.DataType.BOOL),
            shape=ir.Shape(()),
        )
        node.replace_input_with(0, unrelated)
    elif fault == "branch-roles":
        false_graph = node.attributes["else_branch"].as_graph()
        node.attributes["then_branch"] = ir.AttrGraph("then_branch", false_graph)
        node.attributes["else_branch"] = ir.AttrGraph("else_branch", true_graph)
    elif fault == "branch-op":
        next(item for item in true_graph if item.op_type == "Sub").op_type = "Add"
    elif fault in ("capture-order", "same-name"):
        capture = next(
            item
            for item in true_graph
            if item.op_type == "Identity" and item.inputs[0] in model.graph.inputs
        )
        old = capture.inputs[0]
        replacement = (
            model.graph.inputs[2]
            if old is model.graph.inputs[1]
            else model.graph.inputs[1]
        )
        if fault == "same-name":
            replacement = ir.Value(name=old.name, type=old.type, shape=old.shape)
        capture.replace_input_with(0, replacement)
    elif fault == "output-order":
        capture = next(
            item.outputs[0] for item in true_graph if item.op_type == "Identity"
        )
        true_graph.outputs[0] = capture
    else:
        node.attributes["unreviewed"] = ir.AttrInt64("unreviewed", 0)
    result = _proof(model)
    assert result.status == "failed", result
    assert result.expected_equations == 4


@pytest.mark.parametrize(
    "fault", ("attachment", "region", "body-receipt", "scope", "second-attachment")
)
def test_missing_proof_never_shrinks_original_expected_children(fault: str) -> None:
    paired = fault == "second-attachment"
    model = (
        _source(
            lambda flag, other, x, y: (_choice(flag, x, y), _choice(other, y, x)),
            flags=2,
        )
        if paired
        else _source()
    )
    ledger = _validated(model, 8 if paired else 4)
    expected = ledger.expected_equations
    if fault in ("attachment", "second-attachment"):
        ledger.if_calls.pop(next(reversed(ledger.if_calls)))
    elif fault == "region":
        ledger.regions.pop(next(iter(ledger.regions)))
    elif fault == "scope":
        ledger.scopes.pop()
    else:
        next(
            scope for scope in ledger.scopes if scope.region is not None
        ).receipts.clear()
    result = _proof(model)
    assert result.status == "unverified", result
    assert result.expected_equations == expected
    assert result.checked_equations < expected


def test_forged_unused_flag_cannot_discharge_a_used_capture() -> None:
    model = _source()
    ledger = _validated(model, 4)
    scope = next(scope for scope in ledger.scopes if scope.region is not None)
    ports = scope.region.captures
    assert not ports[0].unused
    scope.region = replace(
        scope.region, captures=(replace(ports[0], unused=True), *ports[1:])
    )
    result = _proof(model)
    assert result.status == "failed", result
    assert result.expected_equations == 4


@pytest.mark.parametrize("fault", ("duplicate", "orphan-role", "unclaimed-scope"))
def test_every_physical_scope_has_one_reachable_original_invocation(fault: str) -> None:
    model = _source()
    ledger = _validated(model, 4)
    child = next(scope for scope in ledger.scopes if scope.region is not None)
    if fault == "duplicate":
        extra = replace(child, invocation=object())
    elif fault == "orphan-role":
        key = structure.RegionKey(structure.EquationHandle(object(), 1), "true")
        extra = replace(
            child, invocation=object(), region=replace(child.region, key=key)
        )
    else:
        extra = replace(child, invocation=object(), region=None)
    ledger.scopes.append(extra)
    result = _proof(model)
    assert result.status == "unverified", result
    assert result.expected_equations == 4
    assert result.checked_equations < 4


@pytest.mark.parametrize("fault", ("later-parent", "sibling-branch"))
def test_capture_must_resolve_to_the_actual_visible_ancestor_definition(
    fault: str,
) -> None:
    model = _source()
    ledger = _validated(model, 4)
    node = _if_nodes(model)[0]
    true_graph = node.attributes["then_branch"].as_graph()
    false_graph = node.attributes["else_branch"].as_graph()
    scope = next(
        scope
        for scope in ledger.scopes
        if scope.region is not None and scope.region.key.role == "true"
    )
    capture = next(
        item
        for item in true_graph
        if item.op_type == "Identity" and item.inputs[0] in model.graph.inputs
    )
    original = scope.region.captures[0]
    if fault == "later-parent":
        value = ir.Value(
            name="later_same_type",
            type=model.graph.inputs[1].type,
            shape=model.graph.inputs[1].shape,
        )
        later = ir.Node("", "Identity", [model.graph.inputs[1]], outputs=[value])
        model.graph.append(later)
    else:
        value = next(
            item.outputs[0] for item in false_graph if item.op_type == "Identity"
        )
    witness = structure._witness(value)
    capture.replace_input_with(0, value)
    # Forge the late observation too: independent source binding and actual
    # ancestry/order must still reject it, without using names as authority.
    scope.region = replace(
        scope.region,
        captures=(replace(original, outer=witness), *scope.region.captures[1:]),
    )
    result = _proof(model)
    assert result.status == "failed", result


@pytest.mark.parametrize("fault", ("signed-zero", "nan-payload"))
def test_branch_constant_storage_is_bound_to_the_original_native_payload(
    fault: str,
) -> None:
    model = _source(
        lambda flag, x, y: jax.lax.cond(
            flag,
            lambda value: jax.lax.sub(value, _PAYLOAD),
            lambda value: jax.lax.add(value, _PAYLOAD),
            x,
        )
    )
    _validated(model, 4)
    branch = _if_nodes(model)[0].attributes["then_branch"].as_graph()
    operation = next(node for node in branch if node.op_type == "Sub")
    definitions = []
    for value in operation.inputs:
        assert value is not None
        producer = value.producer()
        while producer is not None and producer.op_type == "Identity":
            value = producer.inputs[0]
            assert value is not None
            producer = value.producer()
        tensor = (
            producer.attributes["value"].as_tensor()
            if producer is not None and producer.op_type == "Constant"
            else value.const_value if value.is_initializer() else None
        )
        if tensor is not None:
            observed = tensor.numpy()
            if (
                observed.dtype == _PAYLOAD.dtype
                and observed.shape == _PAYLOAD.shape
                and observed.tobytes() == _PAYLOAD.tobytes()
            ):
                definitions.append((value, producer))
    assert len(definitions) == 1
    initializer, constant = definitions[0]
    changed = _PAYLOAD.copy().view(np.uint32)
    changed[0 if fault == "signed-zero" else 1] = (
        0 if fault == "signed-zero" else 0x7FC00002
    )
    if constant is None:
        assert initializer.is_initializer()
        initializer.const_value = ir.tensor(changed.view(np.float32))
    else:
        constant.attributes["value"] = ir.AttrTensor(
            "value", ir.tensor(changed.view(np.float32))
        )
    result = _proof(model)
    assert result.status == "failed", result
    assert result.expected_equations == 4


def test_uncovered_child_node_cannot_be_hidden_by_its_parent_attachment() -> None:
    model = _source()
    _validated(model, 4)
    graph = _if_nodes(model)[0].attributes["then_branch"].as_graph()
    source = next(node.outputs[0] for node in graph if node.op_type == "Identity")
    value = ir.Value(name="uncovered", type=source.type, shape=source.shape)
    graph.append(ir.Node("", "Identity", [source], outputs=[value]))
    result = _proof(model)
    assert (
        result.status == "unverified"
        and "artifact-node-coverage-incomplete" in result.reasons
    )


@pytest.mark.parametrize("consumed", (False, True))
def test_additional_branch_initializer_needs_original_storage_authority(
    consumed: bool,
) -> None:
    model = _source(_nested, flags=2) if consumed else _source()
    _validated(model)
    graph = _if_nodes(model)[0].attributes["then_branch"].as_graph()
    value = ir.Value(
        name="extra_unobserved_storage",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape((2,)),
        const_value=ir.tensor(np.asarray([3.0, 4.0], np.float32)),
    )
    graph.register_initializer(value)
    if consumed:
        inner = next(node for node in graph if node.op_type == "If")
        descendant = inner.attributes["then_branch"].as_graph()
        capture = next(
            node
            for node in descendant
            if node.op_type == "Identity" and node.inputs[0].dtype is ir.DataType.FLOAT
        )
        capture.replace_input_with(0, value)
    result = _proof(model)
    if consumed:
        assert result.status == "failed", result
    else:
        assert (
            result.status == "unverified"
            and "artifact-constant-coverage-incomplete" in result.reasons
        )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_both_boolean_values_execute_against_independent_original_source_samples(
    dtype: Any,
) -> None:
    model = _source(dtype=dtype)
    _validated(model, 4)
    samples = []
    for flag in (False, True):
        inputs = (
            np.asarray(flag, np.bool_),
            np.asarray([4.0, -2.0], dtype),
            np.asarray([1.0, 0.5], dtype),
        )
        with jax.enable_x64(True):
            expected = np.asarray(_choice(*inputs))
        samples.append(SourceSample(inputs, [expected], SampleCriteria(mode="bitwise")))
    for optimization in ("disabled", "all"):
        report = validate_source_samples(
            model, samples=samples, optimization=optimization
        )
        report.require_sample_acceptance()
        assert report.status is CheckStatus.VALIDATED
        assert len(report.samples) == 2
    # Supplemental sample execution never upgrades the private structural
    # result into a runtime or all-input numerical certificate.
    result = _proof(model)
    assert (
        result.status == "validated"
        and result.runtime == result.numerics == "unverified"
    )


def test_if_ledger_observation_preserves_bytes_reports_context_and_callee_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fresh_root() -> Any:
        def target(flag: Any, other: Any, x: Any, y: Any) -> Any:
            return _function_choice(flag, x, y), _functions_in_branches(other, x, y)

        return target

    configuration = _configuration()
    _CALLS.clear()
    enabled = _source(fresh_root(), flags=2)
    _validated(enabled)
    observed = tuple(_CALLS)
    assert {name for name, _ in observed} == {"choice", "difference"}
    _CALLS.clear()
    with monkeypatch.context() as patch:
        patch.setattr(ir_context, "SourceStructureLedger", lambda: None)
        disabled = _source(fresh_root(), flags=2)
    assert _proof(disabled).reasons == ("source-ledger-unavailable",)
    assert tuple(_CALLS) == observed
    assert _configuration() == configuration
    first, second = ir.to_proto(enabled), ir.to_proto(disabled)
    onnx.checker.check_model(second, full_check=True)
    assert first.SerializeToString(deterministic=True) == second.SerializeToString(
        deterministic=True
    )
    first_report, second_report = get_precision_report(enabled), get_precision_report(
        disabled
    )
    assert first_report is not None and second_report is not None
    assert first_report == second_report
    assert precision_report_to_json(first_report) == precision_report_to_json(
        second_report
    )


def test_if_ledger_retains_no_native_trace_array_value_or_owner() -> None:
    model = _source(_functions_in_branches)
    ledger = _validated(model, 6)

    def inspect(value: Any) -> None:
        assert not isinstance(
            value,
            (
                Jaxpr,
                JaxprEqn,
                Literal,
                Var,
                np.ndarray,
                jax.Array,
                ir.Value,
                ir.Node,
                ir.Graph,
            ),
        )
        if is_dataclass(value) and not isinstance(value, type):
            for member in fields(value):
                inspect(getattr(value, member.name))
        elif isinstance(value, dict):
            for key, item in value.items():
                inspect(key)
                inspect(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                inspect(item)

    inspect(vars(ledger))
    assert ledger.active == ledger.handles == ledger.pending_regions == []
