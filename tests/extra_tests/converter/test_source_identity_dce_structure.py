# tests/extra_tests/converter/test_source_identity_dce_structure.py

"""Real dead-Identity edges preserve every original source result obligation."""

from dataclasses import fields, is_dataclass, replace
from typing import Any

import jax
import numpy as np
import onnx
import onnx_ir as ir
from onnx_ir.passes import common as common_passes
import pytest

from jax2onnx.converter import _source_identity_cse as cse
from jax2onnx.converter import _source_identity_dce as dce
from jax2onnx.converter import ir_context, ir_optimizations as optimizer
from jax2onnx.converter import source_structure as structure
from jax2onnx.precision import (
    SampleCriteria,
    SourceSample,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)
from tests.extra_tests.converter import test_source_identity_cse_structure as cse_owner
from tests.extra_tests.converter import test_source_literal_if_structure as literal

_SOURCE_CALLS: list[int] = []


def _boundary(
    layout: str = "chain2",
    anchor_kind: str = "input",
    dtype: ir.DataType = ir.DataType.FLOAT,
) -> tuple[ir.Model, tuple[Any, ...], Any]:
    x = ir.Value(name="x", type=ir.TensorType(dtype), shape=ir.Shape((2,)))
    y = ir.Value(name="y", type=x.type, shape=x.shape)
    nodes = []
    initializers = []
    anchor = x
    if anchor_kind == "producer":
        anchor = ir.Value(name="anchor", type=x.type, shape=x.shape)
        nodes.append(ir.Node("", "Identity", [x], outputs=[anchor]))
    elif anchor_kind == "initializer":
        array = np.asarray(
            [-0.0, 1.0000001], np.float64 if dtype is ir.DataType.DOUBLE else np.float32
        )
        anchor = ir.Value(
            name="anchor", type=x.type, shape=x.shape, const_value=ir.tensor(array)
        )
        initializers.append(anchor)
    first = ir.Value(name="live_sub", type=x.type, shape=x.shape)
    second = ir.Value(name="live_neg", type=x.type, shape=x.shape)
    nodes.extend(
        [
            ir.Node("", "Sub", [anchor, y], outputs=[first]),
            ir.Node("", "Neg", [anchor], outputs=[second]),
        ]
    )
    dead = []
    for index in range(2 if layout == "chain2" else 3):
        parent = anchor if index == 0 else dead[0] if layout == "tree3" else dead[-1]
        value = ir.Value(name=f"dead_{index}", type=x.type, shape=x.shape)
        nodes.append(ir.Node("", "Identity", [parent], outputs=[value]))
        dead.append(value)
    graph = ir.Graph(
        [x, y],
        [first, second],
        nodes=nodes,
        initializers=initializers,
        opset_imports={"": 23},
        name="source_identity_dce_boundary",
    )
    graph.meta[structure._OWNER] = object()
    witnesses = tuple(structure._witness(value) for value in dead)
    return ir.Model(graph, ir_version=10), witnesses, structure._witness(anchor)


def _run(model: ir.Model) -> None:
    optimizer.remove_dead_nodes_ir(model)
    assert not model.graph.meta.get(dce._GAPS), model.graph.meta


def _resolved(model: ir.Model, witnesses: tuple[Any, ...]) -> dict[Any, ir.Value]:
    dce.validate_identity_dce(model.graph, structure._resolve_bindings)
    return structure._resolve_bindings(model.graph, set(witnesses))


def _immutable(value: Any) -> None:
    if value is None or type(value) in (str, int, bool, bytes, object, ir.DataType):
        return
    if type(value) is tuple:
        for item in value:
            _immutable(item)
        return
    assert is_dataclass(value), type(value)
    assert type(value) in (
        dce.IdentityDCE,
        dce._Row,
        dce._Interface,
        cse._Schema,
        cse._NodeSchema,
        cse._Port,
        structure._ValueWitness,
    )
    assert type(value).__dataclass_params__.frozen
    for field in fields(value):
        _immutable(getattr(value, field.name))


@pytest.mark.parametrize("layout", ("chain2", "chain3", "tree3"))
@pytest.mark.parametrize("anchor_kind", ("input", "producer", "initializer"))
@pytest.mark.parametrize("dtype", (ir.DataType.FLOAT, ir.DataType.DOUBLE))
@pytest.mark.parametrize("protective_clone", (False, True))
def test_actual_dead_identity_tree_has_exact_edges_live_anchor_and_clone_proof(
    layout: str, anchor_kind: str, dtype: ir.DataType, protective_clone: bool
) -> None:
    model, witnesses, anchor = _boundary(layout, anchor_kind, dtype)
    if protective_clone:
        model = model.clone()
    before = tuple(model.graph.outputs)
    _run(model)
    (receipt,) = dce.receipts(model.graph)
    _immutable(receipt)
    assert tuple(model.graph.outputs) == before
    assert tuple(row.value for row in receipt.rows) == witnesses
    assert receipt.rows[0].input == anchor
    for index, row in enumerate(receipt.rows[1:], 1):
        assert row.input == (
            witnesses[0] if layout == "tree3" else witnesses[index - 1]
        )
    found = _resolved(model, (*witnesses, anchor))
    assert all(found[witness] is found[anchor] for witness in witnesses)
    for witness in witnesses:
        with pytest.raises(structure.StructureUnavailable):
            structure._resolve_direct_bindings(model.graph, {witness})
    assert (
        all(node.op_type != "Identity" for node in model.graph)
        if anchor_kind != "producer"
        else sum(node.op_type == "Identity" for node in model.graph) == 1
    )
    clone = model.clone()
    assert dce.receipts(clone.graph)[0] is receipt
    cloned = _resolved(clone, (*witnesses, anchor))
    assert all(cloned[witness] is cloned[anchor] for witness in witnesses)
    assert cloned[anchor] is not found[anchor]
    onnx.checker.check_model(ir.to_proto(clone), full_check=True)


@pytest.mark.parametrize(
    "fault",
    (
        "dtype",
        "shape",
        "domain",
        "overload",
        "attributes",
        "missing-input",
        "input-arity",
        "output-arity",
        "nonvisible",
    ),
)
def test_invalid_original_identity_never_gains_dead_alias_authority(fault: str) -> None:
    model, witnesses, _ = _boundary(anchor_kind="producer")
    graph = model.graph
    first = next(node for node in graph if node.outputs[0].name == "dead_0")
    if fault == "dtype":
        first.outputs[0].type = ir.TensorType(ir.DataType.DOUBLE)
    elif fault == "shape":
        first.outputs[0].shape = ir.Shape((1, 2))
    elif fault == "domain":
        first.domain = "foreign"
    elif fault == "overload":
        first.overload = "foreign"
    elif fault == "attributes":
        first.attributes["foreign"] = ir.AttrInt64("foreign", 1)
    elif fault == "missing-input":
        first.replace_input_with(0, None)
    elif fault == "input-arity":
        first.resize_inputs(2)
        first.replace_input_with(1, graph.inputs[1])
    elif fault == "output-arity":
        first.resize_outputs(2)
    else:
        graph.insert_before(next(iter(graph)), first)
    optimizer.remove_dead_nodes_ir(model)
    assert not dce.receipts(graph)
    with pytest.raises(structure.StructureUnavailable):
        structure._resolve_bindings(graph, set(witnesses))


@pytest.mark.parametrize("boundary", ("output", "nested-use", "nested-output"))
def test_actual_graph_and_nested_boundaries_are_not_closed_dead_components(
    boundary: str,
) -> None:
    model, witnesses, _ = _boundary()
    graph = model.graph
    value = next(node.outputs[0] for node in graph if node.outputs[0].name == "dead_1")
    if boundary == "output":
        graph.outputs.append(value)
    else:
        nested_value = value
        nodes = []
        if boundary == "nested-use":
            nested_value = ir.Value(name="nested", type=value.type, shape=value.shape)
            nodes.append(ir.Node("", "Identity", [value], outputs=[nested_value]))
        nested = ir.Graph([], [nested_value], nodes=nodes, name="nested_boundary")
        nested.meta[structure._OWNER] = object()
        flag = ir.Value(
            name="flag", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape(())
        )
        graph.inputs.append(flag)
        output = ir.Value(name="if_result", type=value.type, shape=value.shape)
        other_value = ir.Value(name="other_nested", type=value.type, shape=value.shape)
        other_nodes = [ir.Node("", "Identity", [value], outputs=[other_value])]
        other = ir.Graph([], [other_value], nodes=other_nodes, name="other_boundary")
        other.meta[structure._OWNER] = object()
        assert nested.outputs[0] is nested_value and nested_value.graph is nested
        assert other.outputs[0] is other_value and other_value.graph is other
        assert other_value is not nested_value
        assert other_nodes[0].graph is other and other_nodes[0].inputs[0] is value
        assert any(use.node is other_nodes[0] and use.idx == 0 for use in value.uses())
        if boundary == "nested-output":
            assert nested_value is value and len(nested) == 0
            assert value.producer() is not None and value.producer().graph is graph
        graph.append(
            ir.Node(
                "",
                "If",
                [flag],
                outputs=[output],
                attributes=[
                    ir.AttrGraph("then_branch", nested),
                    ir.AttrGraph("else_branch", other),
                ],
            )
        )
        graph.outputs.append(output)
    optimizer.remove_dead_nodes_ir(model)
    assert not dce.receipts(graph)
    assert not any(
        row.value in witnesses
        for receipt in dce.receipts(graph)
        for row in receipt.rows
    )


@pytest.mark.parametrize(
    "fault",
    (
        "anchor-dtype",
        "anchor-shape",
        "anchor-origin",
        "anchor-marker",
        "owner",
        "different-anchor",
        "extra-use",
        "lost-use",
        "wrong-port",
        "consumer-domain",
        "consumer-overload",
        "consumer-attrs",
        "consumer-output-dtype",
        "consumer-output-shape",
        "consumer-arity",
        "use-order",
        "output-slots",
        "old-marker",
        "old-producer",
        "old-storage",
        "missing-receipt",
        "duplicate-receipt",
        "foreign-receipt",
        "forged-receipt",
    ),
)
def test_final_current_proof_rejects_each_changed_definition_port_and_receipt(
    fault: str,
) -> None:
    model, witnesses, anchor_witness = _boundary(anchor_kind="producer")
    _run(model)
    graph = model.graph
    anchor = _resolved(model, (anchor_witness,))[anchor_witness]
    consumer = next(node for node in graph if node.op_type == "Sub")
    if fault == "anchor-dtype":
        anchor.type = ir.TensorType(ir.DataType.DOUBLE)
    elif fault == "anchor-shape":
        anchor.shape = ir.Shape((1, 2))
    elif fault == "anchor-origin":
        anchor.producer().meta[structure._ORIGIN] = object()
    elif fault == "anchor-marker":
        anchor.meta[structure._VALUES] = ()
    elif fault == "owner":
        graph.meta[structure._OWNER] = object()
    elif fault == "different-anchor":
        anchor.producer().replace_input_with(0, graph.inputs[1])
    elif fault == "extra-use":
        value = ir.Value(name="extra_use", type=anchor.type, shape=anchor.shape)
        graph.append(ir.Node("", "Identity", [anchor], outputs=[value]))
    elif fault == "lost-use":
        consumer.replace_input_with(0, graph.inputs[1])
    elif fault == "wrong-port":
        consumer.replace_input_with(0, graph.inputs[1])
        consumer.replace_input_with(1, anchor)
    elif fault == "consumer-domain":
        consumer.domain = "foreign"
    elif fault == "consumer-overload":
        consumer.overload = "foreign"
    elif fault == "consumer-attrs":
        consumer.attributes["foreign"] = ir.AttrInt64("foreign", 1)
    elif fault == "consumer-output-dtype":
        consumer.outputs[0].type = ir.TensorType(ir.DataType.DOUBLE)
    elif fault == "consumer-output-shape":
        consumer.outputs[0].shape = ir.Shape((1, 2))
    elif fault == "consumer-arity":
        consumer.resize_inputs(3)
    elif fault == "use-order":
        later = next(node for node in graph if node.op_type == "Neg")
        graph.insert_before(consumer, later)
    elif fault == "output-slots":
        first, second = graph.outputs
        graph.outputs[0], graph.outputs[1] = second, first
    elif fault == "old-marker":
        anchor.meta[structure._VALUES] = (*anchor.meta[structure._VALUES], witnesses[0])
    elif fault == "old-producer":
        value = ir.Value(name="reintroduced", type=anchor.type, shape=anchor.shape)
        node = ir.Node("", "Identity", [anchor], outputs=[value])
        node.meta[structure._ORIGIN] = witnesses[0].origin
        graph.append(node)
    elif fault == "old-storage":
        value = ir.Value(
            name="reintroduced", const_value=ir.tensor(np.ones((2,), np.float32))
        )
        value.meta[structure._VALUES] = (witnesses[0],)
        graph.register_initializer(value)
    elif fault == "missing-receipt":
        graph.meta[dce._RECEIPTS] = ()
    elif fault == "duplicate-receipt":
        graph.meta[dce._RECEIPTS] = dce.receipts(graph) * 2
    elif fault == "foreign-receipt":
        foreign, _, _ = _boundary(anchor_kind="producer")
        _run(foreign)
        graph.meta[dce._RECEIPTS] = dce.receipts(foreign.graph)
    else:
        graph.meta[dce._RECEIPTS] = (replace(dce.receipts(graph)[0]),)
    with pytest.raises((structure.StructureUnavailable, structure.StructureMismatch)):
        _resolved(model, witnesses)


@pytest.mark.parametrize("mutation", ("partial", "new-node", "retained-port"))
def test_incomplete_actual_pass_delta_never_mints_a_receipt(mutation: str) -> None:
    model, witnesses, _ = _boundary("chain3")
    graph = model.graph
    with dce.observe_identity_dce(model):
        dead = [node for node in graph if node.outputs[0].name.startswith("dead_")]
        if mutation == "partial":
            graph.remove(dead[-1], safe=True)
        else:
            common_passes.RemoveUnusedNodesPass()(model)
            if mutation == "new-node":
                value = ir.Value(
                    name="injected",
                    type=graph.inputs[0].type,
                    shape=graph.inputs[0].shape,
                )
                graph.append(
                    ir.Node("", "Identity", [graph.inputs[0]], outputs=[value])
                )
            else:
                next(
                    node for node in graph if node.op_type == "Sub"
                ).replace_input_with(0, graph.inputs[1])
    assert not dce.receipts(graph)
    assert graph.meta[dce._GAPS]
    if mutation != "partial":
        with pytest.raises(structure.StructureUnavailable):
            structure._resolve_direct_bindings(graph, set(witnesses))


@pytest.mark.parametrize("operation", ("Add", "Constant"))
def test_unobserved_or_nonidentity_deletion_has_no_equation_elision(
    operation: str,
) -> None:
    model, witnesses, _ = _boundary()
    graph = model.graph
    result = ir.Value(
        name="dead_source_operation",
        type=graph.inputs[0].type,
        shape=graph.inputs[0].shape,
    )
    attributes = (
        []
        if operation == "Add"
        else [ir.AttrTensor("value", ir.tensor(np.ones((2,), np.float32)))]
    )
    graph.append(
        ir.Node(
            "",
            operation,
            list(graph.inputs) if operation == "Add" else [],
            outputs=[result],
            attributes=attributes,
        )
    )
    original = structure._witness(result)
    optimizer.remove_dead_nodes_ir(model)
    with pytest.raises(structure.StructureUnavailable):
        structure._resolve_bindings(graph, {original})
    if operation == "Add":
        assert graph.meta.get(dce._GAPS) == (
            "identity-dce-after:identity-dce-retained-interface-changed",
        )
        assert not dce.receipts(graph)
        with pytest.raises(structure.StructureUnavailable):
            structure._resolve_bindings(graph, set(witnesses))
        with pytest.raises(
            structure.StructureUnavailable,
            match="identity-dce-retained-interface-changed",
        ):
            _resolved(model, witnesses)
    else:
        assert not graph.meta.get(dce._GAPS)
        assert dce.receipts(graph)
        assert _resolved(model, witnesses)


def test_direct_native_dce_without_observer_stays_without_authority() -> None:
    model, witnesses, _ = _boundary()
    common_passes.RemoveUnusedNodesPass()(model)
    assert not dce.receipts(model.graph)
    with pytest.raises(structure.StructureUnavailable):
        structure._resolve_bindings(model.graph, set(witnesses))


def test_noop_boundary_does_not_mint_alias_receipts() -> None:
    model, witnesses, _ = _boundary()
    with dce.observe_identity_dce(model):
        pass
    assert not dce.receipts(model.graph)
    assert not model.graph.meta.get(dce._GAPS)
    assert structure._resolve_direct_bindings(model.graph, set(witnesses))


def test_genuine_cross_clone_opposite_anchor_receipts_reject_before_callbacks() -> None:
    original, _, _ = _boundary()
    graph = original.graph
    x, y = graph.inputs
    graph.remove(list(graph)[-1], safe=True)
    graph.remove(list(graph)[-1], safe=True)
    first = ir.Value(name="first", type=x.type, shape=x.shape)
    second = ir.Value(name="second", type=x.type, shape=x.shape)
    a = ir.Node("", "Identity", [x], outputs=[first])
    b = ir.Node("", "Identity", [first], outputs=[second])
    graph.insert_before(next(iter(graph)), a)
    graph.insert_after(a, b)
    for use in tuple(x.uses()):
        if use.node not in (a, b):
            use.node.replace_input_with(use.idx, first)
    first_witness, second_witness = structure._witness(first), structure._witness(
        second
    )
    left, right = original.clone(), original.clone()
    right_a = next(node for node in right.graph if node.outputs[0].name == "first")
    right_b = next(node for node in right.graph if node.outputs[0].name == "second")
    right_a.replace_input_with(0, right_b.outputs[0])
    right_b.replace_input_with(0, right.graph.inputs[0])
    right.graph.insert_before(right_a, right_b)
    for use in tuple(right_a.outputs[0].uses()):
        use.node.replace_input_with(use.idx, right_b.outputs[0])
    _run(left)
    _run(right)
    (lreceipt,) = dce.receipts(left.graph)
    (rreceipt,) = dce.receipts(right.graph)
    assert dce._relation(lreceipt) == {second_witness: first_witness}
    assert dce._relation(rreceipt) == {first_witness: second_witness}
    assert lreceipt.owner is rreceipt.owner
    left.graph.meta[dce._RECEIPTS] = (lreceipt, rreceipt)
    with pytest.raises(structure.StructureUnavailable, match="cross-receipt"):
        structure._resolve_bindings(left.graph, {first_witness, second_witness})


def test_independent_actual_components_compose_and_clone() -> None:
    model, witnesses, anchor = _boundary()
    graph = model.graph
    y = graph.inputs[1]
    second = []
    value = y
    for index in range(2):
        output = ir.Value(name=f"second_{index}", type=y.type, shape=y.shape)
        graph.append(ir.Node("", "Identity", [value], outputs=[output]))
        second.append(structure._witness(output))
        value = output
    _run(model)
    assert len(dce.receipts(graph)) == 2
    found = _resolved(model, (*witnesses, *second, anchor))
    assert found[witnesses[0]] is found[anchor]
    assert found[second[0]] is y and found[witnesses[0]] is not y
    assert _resolved(model.clone(), (*witnesses, *second))


def _unused_target(index: int, *, observed: bool = False) -> Any:
    repeated = literal._target(index, "repeated")

    def target(left: Any, right: Any) -> Any:
        if observed:
            _SOURCE_CALLS.append(index)
        return repeated(left, right)[0]

    return target


def _source(target: Any, dtype: Any = np.float32) -> ir.Model:
    model = literal._source(target, dtype=dtype)
    proof = structure.validate_source_structure(model)
    assert proof.status == "validated", proof
    assert proof.expected_equations == proof.checked_equations == 3
    assert proof.runtime == proof.numerics == "unverified"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    return model


@pytest.mark.parametrize("index", (0, 1))
def test_unchanged_original_unused_result_regression(index: int) -> None:
    literal.test_unused_original_results_keep_both_branch_counts(index)


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("optimization", ("disabled", "all"))
def test_original_unused_result_program_bitwise_samples_and_reload(
    index: int, dtype: Any, optimization: str
) -> None:
    target = _unused_target(index)
    model = _source(target, dtype)
    ledger, _, attachment = literal._attachment(model)
    assert ledger.expected_equations == 3
    assert len(attachment.returned) == len(attachment.results) == 3
    assert len(attachment.false.equations) == len(attachment.true.equations) == 1
    assert attachment.discharge.graph_matched_equations == 1
    assert attachment.discharge.semantically_discharged_equations == 2
    assert (
        attachment.discharge.inactive_nodes
        == attachment.discharge.inactive_storage
        == ()
    )
    assert len(model.graph.outputs) == 1
    assert len(set(dict(attachment.values).values())) == len(attachment.values)
    assert dce.receipts(model.graph) and cse.receipts(model.graph)
    inputs = (np.asarray([0.5, -2.0], dtype), np.asarray([0.25, 1.0], dtype))
    with jax.enable_x64(True):
        expected = [np.asarray(target(*inputs))]
    for candidate in (model, ir.from_proto(ir.to_proto(model))):
        report = validate_source_samples(
            candidate,
            samples=[SourceSample(inputs, expected, SampleCriteria(mode="bitwise"))],
            optimization=optimization,
        )
        report.require_sample_acceptance()
    cloned = model.clone()
    assert cloned.graph is not model.graph
    assert structure._LEDGER not in cloned.meta
    bare_proof = structure.validate_source_structure(cloned)
    assert bare_proof.status == "unverified"
    assert bare_proof.reasons == ("source-ledger-unavailable",)
    assert structure._LEDGER not in cloned.meta
    cloned.meta.update(model.meta)
    assert cloned.meta[structure._LEDGER] is ledger
    clone_proof = structure.validate_source_structure(cloned)
    assert clone_proof.status == "validated", clone_proof
    assert clone_proof.expected_equations == clone_proof.checked_equations == 3


@pytest.mark.parametrize(
    "fault",
    (
        "wrong-input",
        "reverse-edge",
        "foreign-edge",
        "mere-equality",
        "wrong-type",
        "dropped-result",
        "lost-body",
    ),
)
@pytest.mark.parametrize("index", (0, 1))
def test_literal_adapter_needs_exact_original_immediate_edge_and_all_original_results(
    fault: str, index: int
) -> None:
    model = _source(_unused_target(index))
    ledger, handle, attachment = literal._attachment(model)
    original = attachment.returned[1]
    result = attachment.results[1]
    bindings = structure._resolve_bindings(model.graph, {original, result})
    source, output = bindings[original], bindings[result]
    assert source is output
    parent = next(
        scope for scope in ledger.scopes if scope.invocation is handle.invocation
    )
    tensor = parent.equations[handle.index].outputs[1].tensor
    assert dce.eliminated_identity_edge(
        model.graph,
        original,
        result,
        source,
        output,
        tensor,
        structure._resolve_bindings,
    )
    if fault == "mere-equality":
        row = next(
            row
            for receipt in dce.receipts(model.graph)
            for row in receipt.rows
            if row.value == result
        )
        assert not dce.eliminated_identity_edge(
            model.graph,
            row.anchor,
            row.anchor,
            source,
            output,
            tensor,
            structure._resolve_bindings,
        )
    elif fault in ("wrong-input", "reverse-edge", "foreign-edge", "wrong-type"):
        if fault == "wrong-input":
            original = next(
                row.anchor
                for receipt in dce.receipts(model.graph)
                for row in receipt.rows
                if row.value == result
            )
        elif fault == "reverse-edge":
            original, result = result, original
        elif fault == "foreign-edge":
            foreign = _source(_unused_target(index))
            _, _, foreign_attachment = literal._attachment(foreign)
            original = foreign_attachment.returned[1]
        else:
            tensor = replace(tensor, dtype=ir.DataType.DOUBLE)
        with pytest.raises(
            (structure.StructureMismatch, structure.StructureUnavailable)
        ):
            dce.eliminated_identity_edge(
                model.graph,
                original,
                result,
                source,
                output,
                tensor,
                structure._resolve_bindings,
            )
    else:
        if fault == "dropped-result":
            ledger.literal_if_calls[handle] = replace(
                attachment, results=attachment.results[:1]
            )
        else:
            child = next(
                scope
                for scope in ledger.scopes
                if scope.invocation is attachment.false.invocation
            )
            child.body = None
        proof = structure.validate_source_structure(model)
        assert proof.status != "validated" and proof.expected_equations == 3


@pytest.mark.parametrize("index", (0, 1))
def test_source_ledger_on_off_keeps_proto_report_source_calls_and_one_original_dce(
    index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = common_passes.RemoveUnusedNodesPass
    calls = []

    class CountedPass:
        def __init__(self) -> None:
            self.actual = original()

        def __call__(self, model: ir.Model) -> Any:
            calls.append(model)
            return self.actual(model)

    monkeypatch.setattr(common_passes, "RemoveUnusedNodesPass", CountedPass)
    _SOURCE_CALLS.clear()
    enabled = _source(_unused_target(index, observed=True))
    enabled_calls = tuple(_SOURCE_CALLS)
    _SOURCE_CALLS.clear()
    with monkeypatch.context() as change:
        change.setattr(ir_context, "SourceStructureLedger", lambda: None)
        disabled = literal._source(_unused_target(index, observed=True))
    assert calls == [enabled, disabled]
    assert tuple(_SOURCE_CALLS) == enabled_calls and enabled_calls
    assert dce.receipts(enabled.graph) and not dce.receipts(disabled.graph)
    assert structure._LEDGER not in disabled.meta
    assert ir.to_proto(enabled).SerializeToString(deterministic=True) == ir.to_proto(
        disabled
    ).SerializeToString(deterministic=True)
    first, second = get_precision_report(enabled), get_precision_report(disabled)
    assert first is not None and first == second
    assert precision_report_to_json(first) == precision_report_to_json(second)


@pytest.mark.parametrize("extra", ("node", "storage"))
def test_dead_alias_receipts_supply_zero_physical_coverage(extra: str) -> None:
    model = _source(_unused_target(0))
    if extra == "node":
        output = ir.Value(
            name="extra",
            type=model.graph.inputs[0].type,
            shape=model.graph.inputs[0].shape,
        )
        model.graph.append(
            ir.Node("", "Neg", [model.graph.inputs[0]], outputs=[output])
        )
    else:
        model.graph.register_initializer(
            ir.Value(name="extra", const_value=ir.tensor(np.ones((2,), np.float32)))
        )
    proof = structure.validate_source_structure(model)
    assert proof.status == "unverified" and proof.expected_equations == 3
    assert any("artifact-" in reason for reason in proof.reasons)


def test_final_alias_and_source_validation_do_not_materialize_retrace_or_mutate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _source(_unused_target(1))
    before = ir.to_proto(model).SerializeToString(deterministic=True)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("final Identity DCE check allocated IR/tensor or executed source")

    with monkeypatch.context() as change:
        change.setattr(ir.Value, "__init__", forbidden)
        change.setattr(ir.Node, "__init__", forbidden)
        change.setattr(ir, "tensor", forbidden)
        change.setattr(jax, "make_jaxpr", forbidden)
        change.setattr(jax, "eval_shape", forbidden)
        change.setattr(structure, "_witness", forbidden)
        change.setattr(dce, "_current", forbidden)
        change.setattr(cse, "_origin", forbidden)
        change.setattr(ir_context.IRContext, "require_value_for_var", forbidden)
        assert structure.validate_source_structure(model).status == "validated"
    assert ir.to_proto(model).SerializeToString(deterministic=True) == before


@pytest.mark.parametrize("stage", ("discovery", "before", "after"))
@pytest.mark.parametrize("pass_raises", (False, True))
def test_observation_rejection_preserves_one_original_pass_and_exception(
    stage: str, pass_raises: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _, _ = _boundary()
    actual = common_passes.RemoveUnusedNodesPass
    sentinel = RuntimeError("original DCE exception")
    calls = []

    class OriginalPass:
        def __init__(self) -> None:
            self.actual = actual()

        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            if pass_raises:
                raise sentinel
            return self.actual(value)

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise structure.StructureUnavailable("deliberate observation rejection")

    monkeypatch.setattr(common_passes, "RemoveUnusedNodesPass", OriginalPass)
    monkeypatch.setattr(
        cse if stage == "discovery" else dce,
        (
            "_graphs"
            if stage == "discovery"
            else "_capture" if stage == "before" else "_finish"
        ),
        unavailable,
    )
    if pass_raises:
        with pytest.raises(RuntimeError) as caught:
            optimizer.remove_dead_nodes_ir(model)
        assert caught.value is sentinel
    else:
        optimizer.remove_dead_nodes_ir(model)
    assert calls == [model] and not dce.receipts(model.graph)
    if stage != "after" or not pass_raises:
        assert model.graph.meta.get(dce._GAPS)


def test_existing_debug_early_return_remains_before_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, witnesses, _ = _boundary()
    monkeypatch.setenv("JAX2ONNX_ENABLE_STACKTRACE_METADATA", "1")
    optimizer.remove_dead_nodes_ir(model)
    assert not dce.receipts(model.graph)
    assert structure._resolve_direct_bindings(model.graph, set(witnesses))


@pytest.mark.parametrize("overlap", (False, True))
def test_existing_cse_receipts_are_unchanged_and_dependencies_cannot_be_deleted(
    overlap: bool,
) -> None:
    model, old, _ = cse_owner._boundary(2, 1, internal=True)
    graph = model.graph
    anchor = graph.inputs[1]
    dead = []
    for index in range(2):
        value = ir.Value(name=f"dce_{index}", type=anchor.type, shape=anchor.shape)
        graph.append(
            ir.Node(
                "",
                "Identity",
                [anchor if overlap or index == 0 else dead[-1]],
                outputs=[value],
            )
        )
        dead.append(value)
    optimizer._run_common_subexpression_elimination_pass(model)
    prior = cse.receipts(graph)
    assert prior
    witness = structure._witness(dead[0]) if not overlap else None
    optimizer.remove_dead_nodes_ir(model)
    assert cse.receipts(graph) == prior
    if overlap:
        assert not dce.receipts(graph)
        assert graph.meta.get(dce._GAPS)
        with pytest.raises(
            (structure.StructureUnavailable, structure.StructureMismatch)
        ):
            cse.validate_identity_cse(graph, structure._resolve_bindings)
    else:
        assert witness is not None
        assert _resolved(model, (*old, witness))
        cse.validate_identity_cse(graph, structure._resolve_bindings)


@pytest.mark.parametrize("overlap", (False, True))
@pytest.mark.parametrize("protective_clone", (False, True))
def test_constant_lift_current_role_and_original_use_contract_stay_independent(
    overlap: bool, protective_clone: bool
) -> None:
    model, witnesses, _ = _boundary(anchor_kind="producer")
    graph = model.graph
    anchor_node = next(iter(graph))
    constant = ir.Value(
        name="payload", type=graph.inputs[0].type, shape=graph.inputs[0].shape
    )
    producer = ir.Node(
        "",
        "Constant",
        [],
        outputs=[constant],
        attributes=[
            ir.AttrTensor("value", ir.tensor(np.asarray([-0.0, 1.0000001], np.float32)))
        ],
    )
    graph.insert_before(anchor_node, producer)
    anchor_node.replace_input_with(0, constant)
    if overlap:
        next(
            node for node in graph if node.outputs[0].name == "dead_0"
        ).replace_input_with(0, constant)
    constant_witness = structure._witness(constant)
    if protective_clone:
        model = model.clone()
        graph = model.graph
    optimizer._run_lift_constants_to_initializers_pass(model)
    prior = graph.meta[structure._CONSTANT_LIFTS]
    optimizer.remove_dead_nodes_ir(model)
    assert graph.meta[structure._CONSTANT_LIFTS] == prior
    if overlap:
        assert not dce.receipts(graph) and graph.meta.get(dce._GAPS)
        with pytest.raises(
            (structure.StructureUnavailable, structure.StructureMismatch)
        ):
            structure._resolve_bindings(graph, {constant_witness})
    else:
        found = _resolved(model, (*witnesses, constant_witness))
        assert found[constant_witness].is_initializer()
        assert _resolved(model.clone(), (*witnesses, constant_witness))[
            constant_witness
        ].is_initializer()
