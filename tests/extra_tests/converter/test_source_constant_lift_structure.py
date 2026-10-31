# tests/extra_tests/converter/test_source_constant_lift_structure.py

"""Actual Constant lifting needs a verified definition transition, not stale meta."""

from dataclasses import replace
from typing import Any

import jax
import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir.passes import common as common_passes

from jax2onnx import user_interface as ui
from jax2onnx.converter import (
    ir_context,
    ir_optimizations,
    source_structure as structure,
)
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.plugins._source_structure import freeze_source_structure_rule
from jax2onnx.precision import SampleCriteria, SourceSample, validate_source_samples


def _boundary_model(
    shape: tuple[int, ...] = (2,), *, marked: bool = True
) -> tuple[ir.Model, tuple[Any, ...]]:
    value = ir.Value(
        name="input", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(shape)
    )
    nodes = []
    constants = []
    previous = value
    for index in range(2):
        array = np.ones(shape, np.float32)
        constant = ir.Value(
            name=f"constant_{index}", type=value.type, shape=ir.Shape(shape)
        )
        nodes.append(
            ir.Node(
                "",
                "Constant",
                [],
                attributes=[ir.AttrTensor("value", ir.tensor(array))],
                outputs=[constant],
            )
        )
        result = ir.Value(
            name=f"result_{index}", type=value.type, shape=ir.Shape(shape)
        )
        nodes.append(ir.Node("", "Sub", [previous, constant], outputs=[result]))
        previous = result
        constants.append(constant)
    graph = ir.Graph([value], [previous], nodes=nodes, opset_imports={"": 23})
    model = ir.Model(graph, ir_version=10)
    witnesses = tuple(structure._witness(item) for item in constants) if marked else ()
    return model, witnesses


def _lift(model: ir.Model) -> None:
    ir_optimizations._run_lift_constants_to_initializers_pass(model)
    assert not model.graph.meta.get(structure._CONSTANT_LIFT_GAPS), model.graph.meta


def _resolve(model: ir.Model, witnesses: tuple[Any, ...]) -> list[ir.Value]:
    found = structure._resolve_bindings(model.graph, set(witnesses))
    return [found[item] for item in witnesses]


@pytest.mark.parametrize("shape", ((2,), (0,), (0, 2, 3)))
def test_equal_bytes_and_empty_constants_get_distinct_registered_transitions(
    shape: tuple[int, ...],
) -> None:
    model, witnesses = _boundary_model(shape)
    originals = list(model.graph)
    _lift(model)
    resolved = _resolve(model, witnesses)
    receipts = model.graph.meta[structure._CONSTANT_LIFTS]
    assert len(receipts) == len(resolved) == 2
    assert resolved[0] is not resolved[1] and witnesses[0] != witnesses[1]
    assert receipts[0].storage == receipts[1].storage
    assert all(item in structure._REGISTERED_CONSTANT_LIFTS for item in receipts)
    assert all(
        item.before.role == "producer" and item.after.role == "initializer"
        for item in receipts
    )
    assert all(item.before.origin is not item.after.origin for item in receipts)
    assert all(node.graph is None for node in originals if node.op_type == "Constant")
    for witness, value in zip(witnesses, resolved, strict=True):
        assert value.is_initializer() and value.producer() is None
        with pytest.raises(
            structure.StructureUnavailable, match="final-value-witness-unavailable"
        ):
            structure._resolve_direct_bindings(model.graph, {witness})


def test_a_copied_stale_marker_without_the_boundary_receipt_has_no_authority() -> None:
    model, witnesses = _boundary_model()
    common_passes.LiftConstantsToInitializersPass(size_limit=0)(model)
    assert not model.graph.meta.get(structure._CONSTANT_LIFTS)
    with pytest.raises(
        structure.StructureUnavailable, match="final-value-witness-unavailable"
    ):
        _resolve(model, witnesses)


def test_inherited_staging_marker_does_not_earn_a_transition() -> None:
    model, witnesses = _boundary_model()
    constant = next(
        node.outputs[0] for node in model.graph if node.op_type == "Constant"
    )
    foreign = replace(witnesses[0], owner=object())
    constant.meta[structure._VALUES] = (foreign, witnesses[0])
    _lift(model)
    assert set(
        item.before for item in model.graph.meta[structure._CONSTANT_LIFTS]
    ) == set(witnesses)
    assert _resolve(model, witnesses)[0].is_initializer()
    with pytest.raises(
        structure.StructureUnavailable, match="final-value-witness-unavailable"
    ):
        structure._resolve_bindings(model.graph, {foreign})


@pytest.mark.parametrize("protective_clone", (False, True))
def test_boundary_receipts_follow_actual_graph_clones(protective_clone: bool) -> None:
    model, witnesses = _boundary_model()
    if protective_clone:
        model = model.clone()
    _lift(model)
    first = _resolve(model, witnesses)
    cloned = model.clone()
    second = _resolve(cloned, witnesses)
    assert all(a is not b for a, b in zip(first, second, strict=True))
    assert all(
        a.graph is model.graph and b.graph is cloned.graph
        for a, b in zip(first, second, strict=True)
    )


@pytest.mark.parametrize(
    "fault",
    (
        "declared-dtype",
        "declared-shape",
        "backing-dtype",
        "backing-shape",
        "backing-bytes",
        "lost-use",
        "extra-use",
        "wrong-port",
        "wrong-graph",
        "missing-receipt",
        "duplicate-receipt",
        "foreign-receipt",
        "unregistered-receipt",
        "replayed-initializer",
        "new-origin",
        "lost-marker",
    ),
)
def test_changed_final_definition_use_or_receipt_never_resolves(fault: str) -> None:
    model, witnesses = _boundary_model()
    _lift(model)
    first, second = _resolve(model, witnesses)
    receipts = model.graph.meta[structure._CONSTANT_LIFTS]
    consumer = next(use.node for use in first.uses())
    port = next(use.idx for use in first.uses())
    if fault == "declared-dtype":
        first.type = ir.TensorType(ir.DataType.DOUBLE)
    elif fault == "declared-shape":
        first.shape = ir.Shape((1, 2))
    elif fault == "backing-dtype":
        first.const_value = ir.tensor(np.ones((2,), np.float64))
    elif fault == "backing-shape":
        first.const_value = ir.tensor(np.ones((1, 2), np.float32))
    elif fault == "backing-bytes":
        first.const_value = ir.tensor(np.zeros((2,), np.float32))
    elif fault == "lost-use":
        consumer.replace_input_with(port, second)
    elif fault == "extra-use":
        result = ir.Value(name="extra_use", type=first.type, shape=first.shape)
        model.graph.append(ir.Node("", "Identity", [first], outputs=[result]))
    elif fault == "wrong-port":
        other = consumer.inputs[0]
        assert port == 1 and other is not first
        consumer.replace_input_with(0, first)
        consumer.replace_input_with(1, other)
    elif fault == "wrong-graph":
        assert first.graph is model.graph and first.name is not None
        assert model.graph.initializers.pop(first.name) is first
        foreign_graph = ir.Graph([], [], nodes=[], initializers=[first])
        assert first.graph is foreign_graph and foreign_graph is not model.graph
        assert foreign_graph.initializers[first.name] is first
        assert all(value is not first for value in model.graph.initializers.values())
        assert consumer.graph is model.graph and consumer.inputs[port] is first
    elif fault == "missing-receipt":
        model.graph.meta[structure._CONSTANT_LIFTS] = ()
    elif fault == "duplicate-receipt":
        model.graph.meta[structure._CONSTANT_LIFTS] = (*receipts, receipts[0])
    elif fault == "foreign-receipt":
        other, _ = _boundary_model()
        _lift(other)
        model.graph.meta[structure._CONSTANT_LIFTS] = other.graph.meta[
            structure._CONSTANT_LIFTS
        ]
    elif fault == "unregistered-receipt":
        model.graph.meta[structure._CONSTANT_LIFTS] = (
            replace(receipts[0]),
            receipts[1],
        )
    elif fault == "replayed-initializer":
        duplicate = ir.Value(
            name="replayed",
            type=first.type,
            shape=first.shape,
            const_value=first.const_value,
        )
        duplicate.meta.update(first.meta)
        model.graph.register_initializer(duplicate)
    elif fault == "new-origin":
        first.meta[structure._ORIGIN] = object()
    else:
        first.meta[structure._VALUES] = (receipts[0].after,)
    with pytest.raises((structure.StructureUnavailable, structure.StructureMismatch)):
        _resolve(model, witnesses)


@pytest.mark.parametrize(
    "fault",
    ("new-bytes", "wrong-use", "extra-use", "extra-initializer", "consumer-kind"),
)
def test_boundary_postchecks_issue_no_receipts_after_a_bad_replacement(
    fault: str,
) -> None:
    model, witnesses = _boundary_model()
    with structure.observe_constant_lifts(model):
        common_passes.LiftConstantsToInitializersPass(size_limit=0)(model)
        first = next(iter(model.graph.initializers.values()))
        consumer = next(use.node for use in first.uses())
        if fault == "new-bytes":
            first.const_value = ir.tensor(np.zeros((2,), np.float32))
        elif fault == "wrong-use":
            consumer.replace_input_with(1, model.graph.inputs[0])
        elif fault == "extra-use":
            output = ir.Value(name="extra_use", type=first.type, shape=first.shape)
            model.graph.append(ir.Node("", "Identity", [first], outputs=[output]))
        elif fault == "consumer-kind":
            consumer.op_type = "Add"
        else:
            model.graph.register_initializer(
                ir.Value(
                    name="extra",
                    type=first.type,
                    shape=first.shape,
                    const_value=first.const_value,
                )
            )
    assert model.graph.meta[structure._CONSTANT_LIFT_GAPS]
    assert not model.graph.meta.get(structure._CONSTANT_LIFTS)
    with pytest.raises((structure.StructureUnavailable, structure.StructureMismatch)):
        _resolve(model, witnesses)


def test_true_function_and_graph_output_constants_are_not_lift_receipts() -> None:
    model, witnesses = _boundary_model()
    function_model, function_witnesses = _boundary_model()
    function = ir.Function("unit", "body", graph=function_model.graph, attributes=())
    model.functions[function.identifier()] = function
    constant = next(
        node.outputs[0] for node in model.graph if node.op_type == "Constant"
    )
    model.graph.outputs.append(constant)
    _lift(model)
    assert (
        structure._resolve_bindings(model.graph, {witnesses[0]})[witnesses[0]]
        is constant
    )
    assert constant.producer() is not None
    assert not function.graph.meta.get(structure._CONSTANT_LIFTS)
    found = structure._resolve_bindings(function.graph, set(function_witnesses))
    assert all(value.producer() is not None for value in found.values())
    assert len(model.graph.meta[structure._CONSTANT_LIFTS]) == 1


def test_observation_preserves_the_existing_pass_once_and_its_exception() -> None:
    model, _ = _boundary_model()
    calls = []
    with pytest.raises(RuntimeError, match="original pass failure"):
        with structure.observe_constant_lifts(model):
            calls.append(1)
            raise RuntimeError("original pass failure")
    assert calls == [1] and not model.graph.meta.get(structure._CONSTANT_LIFTS)


def test_own_same_byte_constvars_are_ordered_and_captures_are_separate() -> None:
    left, right = np.ones((2,), np.float32), np.ones((2,), np.float32)
    native = jax.make_jaxpr(lambda value: (value, left, right))(
        jax.ShapeDtypeStruct((2,), np.float32)
    )
    assert len(native.jaxpr.constvars) == len(native.consts) == 2
    # A valid native body may retain a closed constvar unused by equations or
    # results. Preserve its original declaration instead of fabricating storage.
    body = native.jaxpr.replace(outvars=native.jaxpr.outvars[:2])
    ctx = ir_context.IRContext(
        opset=23,
        enable_double_precision=False,
        input_specs=[],
        precision_policy=PrecisionPolicy.SOURCE,
    )
    ledger = ctx._source_structure
    assert ledger is not None
    for var, array in zip(body.constvars, native.consts, strict=True):
        ctx.bind_const_for_var(var, array)
    ctx.add_input_for_invar(body.invars[0], 0)
    with ledger.scope(ctx, body, freeze_source_structure_rule):
        pass
    assert not ledger.gaps
    scope = ledger.scopes[0]
    assert len(scope.closed_constant_values) == 2
    assert scope.closed_constant_values[0] != scope.closed_constant_values[1]
    assert all(item not in scope.input_values for item in scope.closed_constant_values)
    assert scope.body.constants[0] == scope.body.constants[1]
    coverage = structure._Coverage()
    before = tuple(ctx.builder.graph.initializers.values())
    structure._audit_closed_constants(
        ctx.builder.graph, scope, ledger, structure.TensorContract.matches, coverage
    )
    assert tuple(ctx.builder.graph.initializers.values()) == before
    assert coverage.constants == set(before)
    duplicate = ir.Value(
        name="extra_same_bytes",
        type=before[0].type,
        shape=before[0].shape,
        const_value=before[0].const_value,
    )
    ctx.builder.graph.register_initializer(duplicate)
    structure._audit_closed_constants(
        ctx.builder.graph, scope, ledger, structure.TensorContract.matches, coverage
    )
    assert duplicate not in coverage.constants


def _nested_source(counter: Any, value: Any, flag: Any) -> Any:
    def body(state: Any) -> Any:
        count, payload = state
        updated = jax.lax.cond(
            flag, lambda x: x - np.float32(1), lambda x: x + np.float32(1), payload
        )
        return count + np.int32(1), updated

    return jax.lax.while_loop(
        lambda state: state[0] < np.int32(2), body, (counter, value)
    )


@pytest.mark.parametrize("protective_clone", (False, True))
def test_nested_loop_and_dynamic_if_lifts_preserve_graph_and_numeric_report(
    protective_clone: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [
        jax.ShapeDtypeStruct((), np.int32),
        jax.ShapeDtypeStruct((2,), np.float32),
        jax.ShapeDtypeStruct((), np.bool_),
    ]
    original_conversion = ui._convert_to_ir

    def conversion(**kwargs: Any) -> Any:
        assert kwargs["protective_clone"] is True
        kwargs["protective_clone"] = protective_clone
        return original_conversion(**kwargs)

    monkeypatch.setattr(ui, "_convert_to_ir", conversion)
    enabled = ui._to_onnx(
        _nested_source, specs, precision_policy=PrecisionPolicy.SOURCE, return_mode="ir"
    )
    proof = structure.validate_source_structure(enabled)
    assert proof.status == "validated", proof
    graphs = structure._constant_lift_graphs(enabled.graph)
    assert any(
        graph.meta.get(structure._CONSTANT_LIFTS)
        for graph in graphs
        if graph is not enabled.graph
    )
    with monkeypatch.context() as patch:
        patch.setattr(ir_context, "SourceStructureLedger", lambda: None)
        disabled = ui._to_onnx(
            _nested_source,
            specs,
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
        )
    assert ir.to_proto(enabled).SerializeToString(deterministic=True) == ir.to_proto(
        disabled
    ).SerializeToString(deterministic=True)
    arrays = (np.asarray(0, np.int32), np.asarray([3, 4], np.float32), np.asarray(True))
    expected = [np.asarray(value) for value in jax.tree.leaves(_nested_source(*arrays))]
    reports = []
    for model in (enabled, disabled):
        report = validate_source_samples(
            model,
            samples=[SourceSample(arrays, expected, SampleCriteria(mode="bitwise"))],
            optimization="disabled",
        )
        report.require_sample_acceptance()
        reports.append(report)
    assert reports[0] == reports[1]
    cloned = enabled.clone()
    cloned.meta.update(enabled.meta)
    assert structure.validate_source_structure(cloned).status == "validated"


@pytest.mark.parametrize(
    "fault", ("dtype", "shape", "extra-input", "duplicate-marker", "foreign-use")
)
def test_prechecks_do_not_certify_a_malformed_original_definition(fault: str) -> None:
    model, witnesses = _boundary_model()
    constants = [node for node in model.graph if node.op_type == "Constant"]
    first = constants[0].outputs[0]
    if fault == "dtype":
        first.type = ir.TensorType(ir.DataType.DOUBLE)
    elif fault == "shape":
        first.shape = ir.Shape((1, 2))
    elif fault == "extra-input":
        constants[0].resize_inputs(1)
        constants[0].replace_input_with(0, model.graph.inputs[0])
    elif fault == "duplicate-marker":
        constants[1].outputs[0].meta[structure._VALUES] = (witnesses[0],)
    else:
        result = ir.Value(name="foreign_use", type=first.type, shape=first.shape)
        foreign = ir.Graph(
            [], [result], nodes=[ir.Node("", "Identity", [first], outputs=[result])]
        )
        assert first.uses() and foreign is not model.graph
    # The existing pass is still executed once even when observation cannot
    # establish authority for its malformed incoming definition.
    with structure.observe_constant_lifts(model):
        common_passes.LiftConstantsToInitializersPass(size_limit=0)(model)
    assert model.graph.meta.get(structure._CONSTANT_LIFT_GAPS)
    assert not model.graph.meta.get(structure._CONSTANT_LIFTS)


def test_unchanged_candidate_earns_no_transition() -> None:
    model, witnesses = _boundary_model()
    nodes = tuple(model.graph)
    with structure.observe_constant_lifts(model):
        pass
    assert tuple(model.graph) == nodes
    assert not model.graph.meta.get(structure._CONSTANT_LIFTS)
    assert all(value.producer() is not None for value in _resolve(model, witnesses))


def test_lift_observation_never_retraces_or_materializes_a_native_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, witnesses = _boundary_model()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("observation requested source/literal materialization")

    monkeypatch.setattr(jax, "make_jaxpr", forbidden)
    monkeypatch.setattr(ir_context.IRContext, "bind_const_for_var", forbidden)
    monkeypatch.setattr(ir_context.IRContext, "_bind_literal_value_for_var", forbidden)
    before = len(tuple(model.graph))
    _lift(model)
    assert len(tuple(model.graph)) == before - 2
    assert len(model.graph.initializers) == len(_resolve(model, witnesses)) == 2


def _unmarked_constant_model(
    shape: tuple[int, ...],
    tensor_mode: str,
    mixed: str,
) -> tuple[ir.Model, tuple[Any, ...], tuple[ir.Node, ...]]:
    model, witnesses = _boundary_model(shape)
    witnessed = next(node for node in model.graph if node.op_type == "Constant")
    shared = (
        witnessed.attributes["value"].as_tensor()
        if tensor_mode == "witness-shared"
        else ir.tensor(np.ones(shape, np.float32))
    )
    dead = []
    for index in range(2):
        tensor = (
            ir.tensor(np.full(shape, index + 1, np.float32))
            if tensor_mode == "different"
            else (
                ir.tensor(np.ones(shape, np.float32))
                if tensor_mode == "distinct"
                else shared
            )
        )
        output = ir.Value(
            name=f"unmarked_dead_{index}",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(shape),
        )
        dead.append(
            ir.Node(
                "",
                "Constant",
                [],
                attributes=[ir.AttrTensor("value", tensor)],
                outputs=[output],
            )
        )
    used = []
    if mixed != "none":
        output = ir.Value(
            name="unmarked_used",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(shape),
        )
        tensor = (
            ir.tensor(np.ones(shape, np.float32))
            if tensor_mode in ("distinct", "different")
            else shared
        )
        used.append(
            ir.Node(
                "",
                "Constant",
                [],
                attributes=[ir.AttrTensor("value", tensor)],
                outputs=[output],
            )
        )
        result = ir.Value(name="unmarked_result", type=output.type, shape=output.shape)
        used.append(
            ir.Node("", "Sub", [model.graph.inputs[0], output], outputs=[result])
        )
        model.graph.outputs.append(result)
    for node in (*used, *dead) if mixed == "used-first" else (*dead, *used):
        model.graph.append(node)
    assert all(
        not node.outputs[0].meta.get(structure._VALUES) and not node.outputs[0].uses()
        for node in dead
    )
    return model, witnesses, tuple(dead)


@pytest.mark.parametrize("shape", ((2,), (0, 2, 3)))
@pytest.mark.parametrize(
    "tensor_mode", ("different", "distinct", "shared", "witness-shared")
)
@pytest.mark.parametrize("mixed", ("none", "used-first", "used-last"))
def test_unmarked_dead_constants_have_complete_injective_pass_accounting(
    shape: tuple[int, ...],
    tensor_mode: str,
    mixed: str,
) -> None:
    model, witnesses, dead = _unmarked_constant_model(shape, tensor_mode, mixed)
    tensors = tuple(node.attributes["value"].as_tensor() for node in dead)
    assert (tensors[0] is tensors[1]) is (tensor_mode in ("shared", "witness-shared"))
    original_nodes = tuple(model.graph)
    original_constants = tuple(
        node for node in original_nodes if node.op_type == "Constant"
    )
    original_tensors = {
        node.outputs[0]: node.attributes["value"].as_tensor()
        for node in original_constants
    }
    original_uses = {
        node.outputs[0]: tuple((use.node, use.idx) for use in node.outputs[0].uses())
        for node in original_constants
    }
    _lift(model)
    assert (
        len(model.graph.initializers)
        == len(original_constants)
        == (4 if mixed == "none" else 5)
    )
    assert len(tuple(model.graph)) == len(original_nodes) - len(original_constants)
    assert all(
        node.graph is None and not node.outputs[0].uses() for node in original_constants
    )
    receipts = model.graph.meta[structure._CONSTANT_LIFTS]
    assert len(receipts) == len(witnesses) == 2
    assert {item.before for item in receipts} == set(witnesses)
    assert all(value.is_initializer() for value in _resolve(model, witnesses))
    unmarked = [
        value
        for value in model.graph.initializers.values()
        if not value.meta.get(structure._VALUES)
    ]
    unused = [value for value in unmarked if not value.uses()]
    assert len(unused) == len(set(unused)) == 2
    assert all(
        any(value.const_value is tensor for tensor in tensors) for value in unused
    )
    assert all(
        structure.ConstantContract.from_array(value.const_value.numpy()).shape == shape
        for value in unused
    )
    for original, ports in original_uses.items():
        if not ports:
            continue
        actual = ports[0][0].inputs[ports[0][1]]
        assert actual is not original and actual in model.graph.initializers.values()
        assert actual.const_value is original_tensors[original]
        assert all(consumer.inputs[port] is actual for consumer, port in ports)
        assert set((use.node, use.idx) for use in actual.uses()) == set(ports)


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "merged",
        "extra-same-tensor",
        "extra-equal-tensor",
        "used-dead",
        "bytes",
        "shape",
        "dtype",
    ),
)
def test_unmarked_classes_cannot_hide_incomplete_or_changed_pass_deltas(
    fault: str,
) -> None:
    model, _, _ = _unmarked_constant_model(
        (2,), "shared", "used-last" if fault == "merged" else "none"
    )
    with structure.observe_constant_lifts(model):
        common_passes.LiftConstantsToInitializersPass(size_limit=0)(model)
        unmarked = [
            value
            for value in model.graph.initializers.values()
            if not value.meta.get(structure._VALUES)
        ]
        dead = [value for value in unmarked if not value.uses()]
        assert len(dead) == 2
        if fault == "missing":
            model.graph.initializers.pop(dead[0].name)
        elif fault == "merged":
            used = next(value for value in unmarked if value.uses())
            used.replace_all_uses_with(dead[0])
            model.graph.initializers.pop(used.name)
        elif fault.startswith("extra-"):
            tensor = (
                dead[0].const_value
                if fault == "extra-same-tensor"
                else ir.tensor(np.ones((2,), np.float32))
            )
            model.graph.register_initializer(
                ir.Value(
                    name="unmarked_extra",
                    type=dead[0].type,
                    shape=dead[0].shape,
                    const_value=tensor,
                )
            )
        elif fault == "used-dead":
            output = ir.Value(
                name="unexpected_dead_use", type=dead[0].type, shape=dead[0].shape
            )
            model.graph.append(ir.Node("", "Identity", [dead[0]], outputs=[output]))
        elif fault == "bytes":
            dead[0].const_value = ir.tensor(np.zeros((2,), np.float32))
        elif fault == "shape":
            dead[0].shape = ir.Shape((1, 2))
        else:
            dead[0].type = ir.TensorType(ir.DataType.DOUBLE)
    assert model.graph.meta.get(structure._CONSTANT_LIFT_GAPS)
    assert not model.graph.meta.get(structure._CONSTANT_LIFTS)
