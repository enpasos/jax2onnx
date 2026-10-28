# tests/extra_tests/converter/test_source_identity_cse_structure.py

"""Actual Identity-only CSE relations retain canonical uses and output slots."""

from dataclasses import fields, is_dataclass, replace
from typing import Any
from weakref import WeakSet

import jax
import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx_ir.passes import common as common_passes

from jax2onnx import user_interface as ui
from jax2onnx.converter import _source_identity_cse as identity
from jax2onnx.converter import ir_context, ir_optimizations
from jax2onnx.converter import source_structure as structure
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.precision import (
    SampleCriteria,
    SourceSample,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)

_SOURCE_CALLS: list[int] = []


def _boundary(
    duplicates: int = 3,
    depth: int = 3,
    *,
    internal: bool = True,
    dtype: ir.DataType = ir.DataType.FLOAT,
    owned: bool = True,
) -> tuple[ir.Model, tuple[Any, ...], tuple[tuple[ir.Value, ...], ...]]:
    x = ir.Value(name="x", type=ir.TensorType(dtype), shape=ir.Shape((2,)))
    y = ir.Value(name="y", type=x.type, shape=x.shape)
    nodes = []
    levels = []
    previous = (x,) * duplicates
    for level in range(depth):
        current = []
        for index, value in enumerate(previous):
            output = ir.Value(name=f"alias_{level}_{index}", type=x.type, shape=x.shape)
            nodes.append(ir.Node("", "Identity", [value], outputs=[output]))
            current.append(output)
        levels.append(tuple(current))
        previous = tuple(current)
    outputs = list(previous)
    if internal:
        result = ir.Value(name="internal", type=x.type, shape=x.shape)
        nodes.append(ir.Node("", "Sub", [previous[-1], y], outputs=[result]))
        outputs.insert(1, result)
    graph = ir.Graph(
        [x, y],
        outputs,
        nodes=nodes,
        opset_imports={"": 23},
        name="identity_cse_boundary",
    )
    model = ir.Model(graph, ir_version=10)
    if owned:
        graph.meta[structure._OWNER] = object()
        witnesses = tuple(
            structure._witness(value) for level in levels for value in level
        )
    else:
        witnesses = ()
    return model, witnesses, tuple(levels)


def _run(model: ir.Model) -> None:
    ir_optimizations._run_common_subexpression_elimination_pass(model)
    assert not model.graph.meta.get(identity._GAPS), model.graph.meta


def _resolved(model: ir.Model, witnesses: tuple[Any, ...]) -> dict[Any, ir.Value]:
    identity.validate_identity_cse(model.graph, structure._resolve_bindings)
    return structure._resolve_bindings(model.graph, set(witnesses))


def _immutable(value: Any) -> None:
    if value is None or type(value) in (bool, int, str, bytes, object, ir.DataType):
        return
    if type(value) is tuple:
        for item in value:
            _immutable(item)
        return
    assert is_dataclass(value), type(value)
    assert type(value) in (
        identity.IdentityCSE,
        identity._Row,
        identity._Port,
        identity._Schema,
        identity._NodeSchema,
        identity._Output,
        structure._ValueWitness,
    )
    assert type(value).__dataclass_params__.frozen
    for item in fields(value):
        _immutable(getattr(value, item.name))


@pytest.mark.parametrize("duplicates", (2, 3))
@pytest.mark.parametrize("depth", (1, 3))
@pytest.mark.parametrize("internal", (False, True))
@pytest.mark.parametrize("protective_clone", (False, True))
@pytest.mark.parametrize("dtype", (ir.DataType.FLOAT, ir.DataType.DOUBLE))
def test_real_pass_canonical_relations_cover_all_internal_uses_and_ordered_outputs(
    duplicates: int,
    depth: int,
    internal: bool,
    protective_clone: bool,
    dtype: ir.DataType,
) -> None:
    model, witnesses, levels = _boundary(
        duplicates, depth, internal=internal, dtype=dtype
    )
    if protective_clone:
        model = model.clone()
    _run(model)
    found = _resolved(model, witnesses)
    assert len(identity.receipts(model.graph)) == 1
    for level in range(depth):
        group = witnesses[level * duplicates : (level + 1) * duplicates]
        assert len({found[witness] for witness in group}) == 1
        assert found[group[0]].producer().op_type == "Identity"
        for old in group[1:]:
            with pytest.raises(structure.StructureUnavailable):
                structure._resolve_direct_bindings(model.graph, {old})
    canonical = found[witnesses[-1]]
    if not protective_clone:
        assert canonical is levels[-1][0]
    bridges = [
        output
        for item in identity.receipts(model.graph)
        for output in item.outputs
        if output.bridge
    ]
    assert len(bridges) == duplicates - 1
    assert all(model.graph.outputs[item.slot] is not canonical for item in bridges)
    assert all(
        model.graph.outputs[item.slot].producer().inputs[0] is canonical
        for item in bridges
    )
    if internal:
        consumer = next(node for node in model.graph if node.op_type == "Sub")
        assert consumer.inputs[0] is canonical
        assert all(
            consumer.inputs[0] is not model.graph.outputs[item.slot] for item in bridges
        )
    for receipt in identity.receipts(model.graph):
        _immutable(receipt)
    clone = model.clone()
    assert clone.graph is not model.graph
    cloned = _resolved(clone, witnesses)
    assert all(cloned[key] is not found[key] for key in witnesses)
    assert identity.receipts(clone.graph) == identity.receipts(model.graph)
    onnx.checker.check_model(ir.to_proto(clone), full_check=True)


@pytest.mark.parametrize(
    "fault",
    (
        "absent",
        "unregistered",
        "foreign",
        "duplicate",
        "cycle",
        "old-marker",
        "keeper-origin",
        "bridge-origin",
        "keeper-op",
        "domain",
        "overload",
        "attributes",
        "arity",
        "missing-input",
        "same-type-input",
        "consumer-deleted",
        "consumer-added",
        "consumer-port",
        "consumer-owner",
        "consumer-origin",
        "output-order",
        "bridge-missing",
        "bridge-fake",
        "producer-order",
        "dtype",
        "shape",
    ),
)
def test_final_receipt_schema_edges_owners_slots_and_order_reject_corruption(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, witnesses, _ = _boundary()
    _run(model)
    found = _resolved(model, witnesses)
    receipt = identity.receipts(model.graph)[0]
    graph = model.graph
    canonical = found[witnesses[-1]]
    keeper = canonical.producer()
    consumer = next(node for node in graph if node.op_type == "Sub")
    bridge_record = next(item for item in receipt.outputs if item.bridge)
    bridge = graph.outputs[bridge_record.slot].producer()
    assert keeper is not None and bridge is not None
    if fault == "absent":
        graph.meta.pop(identity._RECEIPTS)
    elif fault == "unregistered":
        graph.meta[identity._RECEIPTS] = (replace(receipt),)
    elif fault == "foreign":
        foreign, _, _ = _boundary()
        _run(foreign)
        graph.meta[identity._RECEIPTS] = identity.receipts(foreign.graph)
    elif fault == "duplicate":
        graph.meta[identity._RECEIPTS] = (receipt, receipt)
    elif fault == "cycle":
        rows = list(receipt.rows)
        rows[0] = replace(rows[0], canonical=rows[1].value)
        forged = replace(receipt, rows=tuple(rows))
        monkeypatch.setattr(
            identity, "_REGISTERED", WeakSet((*identity._REGISTERED, forged))
        )
        graph.meta[identity._RECEIPTS] = (forged,)
    elif fault == "old-marker":
        canonical.meta[structure._VALUES] += (witnesses[-1],)
    elif fault == "keeper-origin":
        keeper.meta[structure._ORIGIN] = object()
    elif fault == "bridge-origin":
        bridge.meta[structure._ORIGIN] = object()
    elif fault == "keeper-op":
        keeper.op_type = "Neg"
    elif fault == "domain":
        keeper.domain = "foreign"
    elif fault == "overload":
        keeper.overload = "foreign"
    elif fault == "attributes":
        keeper.attributes["unexpected"] = ir.AttrInt64("unexpected", 0)
    elif fault == "arity":
        keeper.resize_inputs(2)
        keeper.replace_input_with(1, graph.inputs[1])
    elif fault == "missing-input":
        keeper.replace_input_with(0, None)
    elif fault == "same-type-input":
        keeper.replace_input_with(0, graph.inputs[1])
    elif fault == "consumer-deleted":
        graph.remove(consumer, safe=False)
    elif fault == "consumer-added":
        output = ir.Value(name="extra", type=canonical.type, shape=canonical.shape)
        graph.append(ir.Node("", "Neg", [canonical], outputs=[output]))
    elif fault == "consumer-port":
        consumer.replace_input_with(0, graph.inputs[1])
        consumer.replace_input_with(1, canonical)
    elif fault == "consumer-owner":
        graph.remove(consumer, safe=False)
        foreign_graph = ir.Graph([], [], nodes=[consumer])
        foreign_graph.meta[structure._OWNER] = object()
        assert consumer.graph is foreign_graph
    elif fault == "consumer-origin":
        consumer.meta[structure._ORIGIN] = object()
    elif fault == "output-order":
        graph.outputs[0], graph.outputs[bridge_record.slot] = (
            graph.outputs[bridge_record.slot],
            graph.outputs[0],
        )
    elif fault == "bridge-missing":
        graph.outputs[bridge_record.slot] = canonical
        graph.remove(bridge, safe=True)
    elif fault == "bridge-fake":
        output = ir.Value(name="fake", type=canonical.type, shape=canonical.shape)
        replacement = ir.Node("", "Identity", [canonical], outputs=[output])
        graph.insert_before(bridge, replacement)
        graph.outputs[bridge_record.slot] = output
        graph.remove(bridge, safe=True)
    elif fault == "producer-order":
        graph.insert_after(consumer, keeper)
    elif fault == "dtype":
        canonical.type = ir.TensorType(ir.DataType.DOUBLE)
    else:
        canonical.shape = ir.Shape((1, 2))
    with pytest.raises((structure.StructureUnavailable, structure.StructureMismatch)):
        _resolved(model, witnesses)


@pytest.mark.parametrize(
    "fault", ("new-use", "lost-use", "new-origin", "partial-removal")
)
def test_no_receipt_is_minted_after_an_incomplete_real_pass_component(
    fault: str,
) -> None:
    model, witnesses, levels = _boundary()
    with identity.observe_identity_cse(model):
        common_passes.CommonSubexpressionEliminationPass()(model)
        canonical = levels[-1][0]
        consumer = next(node for node in model.graph if node.op_type == "Sub")
        if fault == "new-use":
            output = ir.Value(name="extra", type=canonical.type, shape=canonical.shape)
            model.graph.append(ir.Node("", "Neg", [canonical], outputs=[output]))
        elif fault == "lost-use":
            consumer.replace_input_with(0, model.graph.inputs[0])
        elif fault == "new-origin":
            consumer.meta[structure._ORIGIN] = object()
        else:
            old = levels[-1][-1]
            node = old.producer()
            assert node is not None and node.graph is None
            node.replace_input_with(0, levels[-2][0])
            model.graph.append(node)
    assert not identity.receipts(model.graph)
    assert model.graph.meta.get(identity._GAPS)
    with pytest.raises(structure.StructureUnavailable):
        _resolved(model, witnesses)


@pytest.mark.parametrize("prerequisite", ("Add", "Constant"))
@pytest.mark.parametrize("owned", (False, True))
def test_equal_numeric_prerequisites_and_same_names_supply_no_identity_authority(
    prerequisite: str,
    owned: bool,
) -> None:
    x = ir.Value(
        name="same", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2,))
    )
    nodes = []
    outputs = []
    for index in range(2):
        root = ir.Value(name=f"root_{index}", type=x.type, shape=x.shape)
        if prerequisite == "Add":
            node = ir.Node("", "Add", [x, x], outputs=[root])
        else:
            node = ir.Node(
                "",
                "Constant",
                [],
                outputs=[root],
                attributes=[
                    ir.AttrTensor("value", ir.tensor(np.ones((2,), np.float32)))
                ],
            )
        nodes.append(node)
        output = ir.Value(name=f"out_{index}", type=x.type, shape=x.shape)
        nodes.append(ir.Node("", "Identity", [root], outputs=[output]))
        outputs.append(output)
    graph = ir.Graph([x], outputs, nodes=nodes, opset_imports={"": 23})
    model = ir.Model(graph, ir_version=10)
    if owned:
        graph.meta[structure._OWNER] = object()
        witness = structure._witness(outputs[-1])
    for output in outputs:
        output.name = "same"
    _run(model)
    assert not identity.receipts(graph)
    if owned:
        with pytest.raises(structure.StructureUnavailable):
            structure._resolve_bindings(graph, {witness})


def test_unused_value_removed_by_dce_has_no_invented_identity_elision() -> None:
    model, _, _ = _boundary(2, 1)
    value = ir.Value(
        name="unused",
        type=model.graph.inputs[1].type,
        shape=model.graph.inputs[1].shape,
    )
    model.graph.append(
        ir.Node("", "Identity", [model.graph.inputs[1]], outputs=[value])
    )
    witness = structure._witness(value)
    common_passes.RemoveUnusedNodesPass()(model)
    _run(model)
    with pytest.raises(structure.StructureUnavailable):
        structure._resolve_bindings(model.graph, {witness})


@pytest.mark.parametrize("protective_clone", (False, True))
def test_canonical_input_resolves_through_independently_registered_constant_lift(
    protective_clone: bool,
) -> None:
    model, witnesses, _ = _boundary()
    x = model.graph.inputs[0]
    constant = ir.Value(name="payload", type=x.type, shape=x.shape)
    producer = ir.Node(
        "",
        "Constant",
        [],
        outputs=[constant],
        attributes=[
            ir.AttrTensor("value", ir.tensor(np.asarray([-0.0, 1.0000001], np.float32)))
        ],
    )
    model.graph.insert_before(next(iter(model.graph)), producer)
    for use in tuple(x.uses()):
        use.node.replace_input_with(use.idx, constant)
    constant_witness = structure._witness(constant)
    if protective_clone:
        model = model.clone()
    _run(model)
    ir_optimizations._run_lift_constants_to_initializers_pass(model)
    resolved = _resolved(model, (*witnesses, constant_witness))
    assert resolved[constant_witness].is_initializer()
    assert model.graph.meta[structure._CONSTANT_LIFTS]
    clone = model.clone()
    assert _resolved(clone, (*witnesses, constant_witness))[
        constant_witness
    ].is_initializer()


def _target(index: int, mode: str, *, observed: bool = False) -> Any:
    def target(left: Any, right: Any) -> Any:
        if observed:
            _SOURCE_CALLS.append(index)
        if mode == "alias":

            def false(x: Any, y: Any) -> Any:
                return y, x, y

            def true(x: Any, y: Any) -> Any:
                return x, y, x

        else:

            def false(x: Any, y: Any) -> Any:
                result = x + y
                return result, x, result

            def true(x: Any, y: Any) -> Any:
                result = x - y
                return result, y, result

        return jax.lax.cond(bool(index), true, false, left, right)

    return target


def _source(target: Any, dtype: Any = np.float32) -> ir.Model:
    with jax.enable_x64(True):
        model = ui._to_onnx(
            target,
            [jax.ShapeDtypeStruct((2,), dtype)] * 2,
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
            opset=23,
        )
    assert isinstance(model, ir.Model)
    proof = structure.validate_source_structure(model)
    assert proof.status == "validated", proof
    assert proof.expected_equations == proof.checked_equations
    assert proof.runtime == proof.numerics == "unverified"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    return model


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("mode", ("alias", "repeated"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("optimization", ("disabled", "all"))
def test_original_jax_alias_samples_and_serialized_reload_remain_bitwise(
    index: int,
    mode: str,
    dtype: Any,
    optimization: str,
) -> None:
    target = _target(index, mode)
    model = _source(target, dtype)
    ledger = model.meta[structure._LEDGER]
    assert ledger.expected_equations == (1 if mode == "alias" else 3)
    attachment = next(iter(ledger.literal_if_calls.values()))
    assert len(set(dict(attachment.values).values())) == len(attachment.values)
    assert (
        attachment.discharge.inactive_nodes
        == attachment.discharge.inactive_storage
        == ()
    )
    assert identity.receipts(model.graph)
    inputs = (np.asarray([0.5, -2.0], dtype), np.asarray([0.25, 1.0], dtype))
    with jax.enable_x64(True):
        expected = [np.asarray(value) for value in target(*inputs)]
    for candidate in (model, ir.from_proto(ir.to_proto(model))):
        report = validate_source_samples(
            candidate,
            samples=[SourceSample(inputs, expected, SampleCriteria(mode="bitwise"))],
            optimization=optimization,
        )
        report.require_sample_acceptance()


@pytest.mark.parametrize("mode", ("alias", "repeated"))
def test_ledger_on_off_keeps_exact_proto_report_source_calls_and_single_real_pass(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_pass = common_passes.CommonSubexpressionEliminationPass
    pass_calls = []

    class CountedPass:
        def __init__(self) -> None:
            self.actual = actual_pass()

        def __call__(self, model: ir.Model) -> Any:
            pass_calls.append(model)
            return self.actual(model)

    monkeypatch.setattr(
        common_passes, "CommonSubexpressionEliminationPass", CountedPass
    )
    _SOURCE_CALLS.clear()
    enabled = _source(_target(0, mode, observed=True))
    enabled_calls = tuple(_SOURCE_CALLS)
    assert len(pass_calls) == 1 and pass_calls[0] is enabled
    _SOURCE_CALLS.clear()
    with monkeypatch.context() as change:
        change.setattr(ir_context, "SourceStructureLedger", lambda: None)
        with jax.enable_x64(True):
            disabled = ui._to_onnx(
                _target(0, mode, observed=True),
                [jax.ShapeDtypeStruct((2,), np.float32)] * 2,
                precision_policy=PrecisionPolicy.SOURCE,
                return_mode="ir",
                opset=23,
            )
    assert len(pass_calls) == 2 and pass_calls[1] is disabled
    assert tuple(_SOURCE_CALLS) == enabled_calls
    assert enabled_calls
    assert structure._LEDGER not in disabled.meta
    assert not identity.receipts(disabled.graph)
    assert ir.to_proto(enabled).SerializeToString(deterministic=True) == ir.to_proto(
        disabled
    ).SerializeToString(deterministic=True)
    first, second = get_precision_report(enabled), get_precision_report(disabled)
    assert first is not None and first == second
    assert precision_report_to_json(first) == precision_report_to_json(second)


def test_unchanged_pass_exception_is_not_replaced_or_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, _ = _boundary()
    problem = RuntimeError("the original CSE failure")
    calls = []

    class FailingPass:
        def __call__(self, actual: ir.Model) -> Any:
            calls.append(actual)
            raise problem

    monkeypatch.setattr(
        common_passes, "CommonSubexpressionEliminationPass", FailingPass
    )
    with pytest.raises(RuntimeError) as caught:
        ir_optimizations._run_common_subexpression_elimination_pass(model)
    assert caught.value is problem and calls == [model]
    assert not identity.receipts(model.graph)


def test_unchanged_noop_boundary_mints_no_substitution_receipt() -> None:
    model, witnesses, _ = _boundary()
    with identity.observe_identity_cse(model):
        pass
    assert not identity.receipts(model.graph)
    assert not model.graph.meta.get(identity._GAPS)
    assert structure._resolve_bindings(model.graph, set(witnesses))


@pytest.mark.parametrize("extra", ("node", "storage"))
def test_transition_receipts_never_waive_whole_artifact_coverage(extra: str) -> None:
    model = _source(_target(0, "repeated"))
    if extra == "node":
        value = ir.Value(
            name="extra",
            type=model.graph.inputs[0].type,
            shape=model.graph.inputs[0].shape,
        )
        model.graph.append(ir.Node("", "Neg", [model.graph.inputs[0]], outputs=[value]))
    else:
        value = ir.Value(name="extra", const_value=ir.tensor(np.ones((2,), np.float32)))
        model.graph.register_initializer(value)
    proof = structure.validate_source_structure(model)
    assert proof.status == "unverified", proof
    assert any("artifact-" in reason for reason in proof.reasons)
    assert proof.expected_equations == 3


def test_final_recheck_never_retraces_materializes_or_changes_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _source(_target(0, "alias"))
    before = ir.to_proto(model).SerializeToString(deterministic=True)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "Identity CSE final validation executed source or materialized a value"
        )

    monkeypatch.setattr(jax, "make_jaxpr", forbidden)
    monkeypatch.setattr(jax, "eval_shape", forbidden)
    monkeypatch.setattr(ir_context.IRContext, "require_value_for_var", forbidden)
    proof = structure.validate_source_structure(model)
    assert proof.status == "validated", proof
    assert ir.to_proto(model).SerializeToString(deterministic=True) == before


@pytest.mark.parametrize("stage", ("before", "after"))
def test_observation_failure_preserves_the_single_original_pass(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, _ = _boundary()
    original = common_passes.CommonSubexpressionEliminationPass
    calls = []

    class CountedPass:
        def __init__(self) -> None:
            self.actual = original()

        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            return self.actual(value)

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise structure.StructureUnavailable("deliberately unavailable observation")

    monkeypatch.setattr(
        common_passes, "CommonSubexpressionEliminationPass", CountedPass
    )
    monkeypatch.setattr(
        identity, "_capture" if stage == "before" else "_finish", unavailable
    )
    ir_optimizations._run_common_subexpression_elimination_pass(model)
    assert calls == [model]
    assert not identity.receipts(model.graph)
    assert model.graph.meta.get(identity._GAPS)


def test_genuine_opposite_order_clone_receipts_cannot_create_a_resolution_cycle() -> (
    None
):
    original, witnesses, _ = _boundary(2, 1, internal=False)
    left, right = original.clone(), original.clone()
    first, second = tuple(right.graph)
    right.graph.insert_before(first, second)
    _run(left)
    _run(right)
    (left_receipt,) = identity.receipts(left.graph)
    (right_receipt,) = identity.receipts(right.graph)
    assert (
        left_receipt in identity._REGISTERED and right_receipt in identity._REGISTERED
    )
    left_aliases = identity._congruence(left_receipt)
    right_aliases = identity._congruence(right_receipt)
    assert left_aliases == {witnesses[1]: witnesses[0]}
    assert right_aliases == {witnesses[0]: witnesses[1]}
    assert left_receipt.owner is right_receipt.owner
    left.graph.meta[identity._RECEIPTS] = (left_receipt, right_receipt)
    with pytest.raises(
        structure.StructureUnavailable, match="identity-cse-cross-component-dependency"
    ):
        structure._resolve_bindings(left.graph, set(witnesses))


def test_genuine_independent_components_compose_without_cross_component_aliases() -> (
    None
):
    model, first_witnesses, _ = _boundary(2, 1, internal=False)
    second = []
    for index in range(2):
        value = ir.Value(
            name=f"second_{index}",
            type=model.graph.inputs[1].type,
            shape=model.graph.inputs[1].shape,
        )
        model.graph.append(
            ir.Node("", "Identity", [model.graph.inputs[1]], outputs=[value])
        )
        model.graph.outputs.append(value)
        second.append(structure._witness(value))
    _run(model)
    assert len(identity.receipts(model.graph)) == 2
    resolved = _resolved(model, (*first_witnesses, *second))
    assert resolved[first_witnesses[0]] is resolved[first_witnesses[1]]
    assert resolved[second[0]] is resolved[second[1]]
    assert resolved[first_witnesses[0]] is not resolved[second[0]]
    clone = model.clone()
    assert _resolved(clone, (*first_witnesses, *second))


@pytest.mark.parametrize("pass_raises", (False, True))
def test_real_repeated_graph_discovery_failure_preserves_original_pass_exception(
    pass_raises: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, _ = _boundary(2, 1, internal=False)
    source = model.graph.inputs[0]
    branch_output = ir.Value(name="branch_output", type=source.type, shape=source.shape)
    shared = ir.Graph(
        [],
        [branch_output],
        nodes=[ir.Node("", "Identity", [source], outputs=[branch_output])],
    )
    shared.meta[structure._OWNER] = object()
    condition = ir.Value(
        name="condition", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape(())
    )
    model.graph.inputs.append(condition)
    output = ir.Value(name="if_output", type=source.type, shape=source.shape)
    model.graph.append(
        ir.Node(
            "",
            "If",
            [condition],
            outputs=[output],
            attributes=[
                ir.AttrGraph("then_branch", shared),
                ir.AttrGraph("else_branch", shared),
            ],
        )
    )
    model.graph.outputs.append(output)
    original = common_passes.CommonSubexpressionEliminationPass
    sentinel = RuntimeError("the original pass owns this exception")
    calls = []

    class OriginalPass:
        def __init__(self) -> None:
            self.actual = original()

        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            if pass_raises:
                raise sentinel
            return self.actual(value)

    monkeypatch.setattr(
        common_passes, "CommonSubexpressionEliminationPass", OriginalPass
    )
    if pass_raises:
        with pytest.raises(RuntimeError) as caught:
            ir_optimizations._run_common_subexpression_elimination_pass(model)
        assert caught.value is sentinel
    else:
        ir_optimizations._run_common_subexpression_elimination_pass(model)
    assert calls == [model]
    assert not identity.receipts(model.graph)
    assert model.graph.meta.get(identity._GAPS)


@pytest.mark.parametrize(
    "fault",
    (
        "op",
        "domain",
        "overload",
        "attributes",
        "arity",
        "same-type-input",
        "dtype",
        "shape",
    ),
)
def test_fresh_bridge_schema_cannot_borrow_valid_canonical_authority(
    fault: str,
) -> None:
    model, witnesses, _ = _boundary()
    _run(model)
    _resolved(model, witnesses)
    receipt = identity.receipts(model.graph)[0]
    bridge_record = next(item for item in receipt.outputs if item.bridge)
    value = model.graph.outputs[bridge_record.slot]
    node = value.producer()
    assert node is not None
    if fault == "op":
        node.op_type = "Neg"
    elif fault == "domain":
        node.domain = "foreign"
    elif fault == "overload":
        node.overload = "foreign"
    elif fault == "attributes":
        node.attributes["unexpected"] = ir.AttrInt64("unexpected", 0)
    elif fault == "arity":
        node.resize_inputs(2)
        node.replace_input_with(1, model.graph.inputs[1])
    elif fault == "same-type-input":
        node.replace_input_with(0, model.graph.inputs[1])
    elif fault == "dtype":
        value.type = ir.TensorType(ir.DataType.DOUBLE)
    else:
        value.shape = ir.Shape((1, 2))
    with pytest.raises((structure.StructureUnavailable, structure.StructureMismatch)):
        _resolved(model, witnesses)
