# tests/extra_tests/converter/test_source_cse_output_aliases.py

"""Real CSE preserves original shared output classes through explicit repair."""

from dataclasses import fields, is_dataclass, replace
import gc
from typing import Any
import weakref

import jax
import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx_ir.passes import common as common_passes

from jax2onnx.converter import _source_cse_output_aliases as aliases
from jax2onnx.converter import _source_identity_cse as identity
from jax2onnx.converter import _source_leaf_cse as leaf
from jax2onnx import user_interface as ui
from jax2onnx.converter import (
    ir_context,
    ir_optimizations,
    source_structure as structure,
)
from jax2onnx.precision import get_precision_report, precision_report_to_json
from jax2onnx.converter.precision_policy import PrecisionPolicy


def _value(name: str, dtype: ir.DataType, shape: tuple[int, ...]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def _model(
    operation: str = "Sub",
    dtype: ir.DataType = ir.DataType.FLOAT,
    shape: tuple[int, ...] = (2,),
    *,
    many: bool = True,
    owned: bool = True,
) -> tuple[ir.Model, tuple[Any, ...]]:
    x, y = _value("x", dtype, shape), _value("y", dtype, shape)
    values = [_value(name, dtype, shape) for name in ("first", "second", "third")]
    nodes = []
    for index, value in enumerate(values):
        if operation == "Constant":
            array = np.full(
                shape,
                0.125,
                dtype=np.float32 if dtype is ir.DataType.FLOAT else np.float64,
            )
            nodes.append(
                ir.Node(
                    "",
                    operation,
                    [],
                    {"value": ir.AttrTensor("value", ir.tensor(array))},
                    outputs=[value],
                    name=f"original_{index}",
                )
            )
        else:
            nodes.append(
                ir.Node(
                    "",
                    operation,
                    [x] if operation == "Identity" else [x, y],
                    outputs=[value],
                    name=f"original_{index}",
                )
            )
    result = _value("internal", dtype, shape)
    nodes.append(
        ir.Node("", "Neg", [values[1]], outputs=[result], name="original_internal")
    )
    outputs = (
        [values[0], values[1], values[2], values[1], values[2], values[1], result]
        if many
        else [values[0], values[1], result, values[1]]
    )
    graph = ir.Graph(
        [x, y],
        outputs,
        nodes=nodes,
        name="source_output_alias_classes",
        opset_imports={"": 23},
    )
    model = ir.Model(graph, ir_version=10)
    if owned:
        graph.meta[structure._OWNER] = object()
        witnesses = tuple(structure._witness(value) for value in values)
    else:
        witnesses = ()
    return model, witnesses


def _full(model: ir.Model) -> None:
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def _run(model: ir.Model) -> None:
    ir_optimizations._run_common_subexpression_elimination_pass(
        model, precision_policy=PrecisionPolicy.SOURCE
    )
    assert not model.graph.meta.get(identity._GAPS), model.graph.meta
    assert not model.graph.meta.get(leaf._GAPS), model.graph.meta


def _resolve(model: ir.Model, witnesses: tuple[Any, ...]) -> dict[Any, ir.Value]:
    identity.validate_identity_cse(model.graph, structure._resolve_bindings)
    leaf.validate_leaf_cse(model.graph, structure._resolve_bindings)
    return structure._resolve_bindings(model.graph, set(witnesses))


def _immutable(value: Any) -> None:
    if value is None or type(value) in (bool, int, str, bytes, object, ir.DataType):
        return
    if type(value) is tuple:
        for item in value:
            _immutable(item)
        return
    assert is_dataclass(value) and type(value).__dataclass_params__.frozen
    assert not isinstance(value, (ir.Graph, ir.Node, ir.Value)) and not callable(value)
    for item in fields(value):
        _immutable(getattr(value, item.name))


@pytest.mark.parametrize("operation", ("Identity", "Constant", "Sub"))
@pytest.mark.parametrize(
    "dtype", (ir.DataType.FLOAT, ir.DataType.DOUBLE), ids=("FLOAT", "DOUBLE")
)
@pytest.mark.parametrize("shape", ((), (2,)), ids=("scalar", "2"))
@pytest.mark.parametrize("clone_stage", ("none", "before", "after"))
@pytest.mark.parametrize("many", (False, True), ids=("pair", "many"))
def test_real_pass_restores_exact_original_value_classes_names_and_all_slots(
    operation: str,
    dtype: ir.DataType,
    shape: tuple[int, ...],
    clone_stage: str,
    many: bool,
) -> None:
    model, witnesses = _model(operation, dtype, shape, many=many)
    if clone_stage == "before":
        model = model.clone()
    _full(model)
    original_names = tuple(value.name for value in model.graph.outputs)
    expected = ((0,), (1, 3, 5), (2, 4), (6,)) if many else ((0,), (1, 3), (2,))
    _run(model)
    assert aliases.slots(tuple(model.graph.outputs)) == expected
    assert tuple(value.name for value in model.graph.outputs) == original_names
    assert model.graph.outputs[1] is model.graph.outputs[3]
    if many:
        assert model.graph.outputs[1] is model.graph.outputs[5]
        assert model.graph.outputs[2] is model.graph.outputs[4]
        assert model.graph.outputs[1] is not model.graph.outputs[2]
    _full(model)
    found = _resolve(model, witnesses)
    assert len({found[item] for item in witnesses}) == 1
    parents = (
        identity.receipts(model.graph)
        if operation == "Identity"
        else leaf.receipts(model.graph)
    )
    assert len(parents) == 1
    receipt = parents[0]
    assert len(receipt.normalizations) == (2 if many else 1)
    assert sum(output.bridge for output in receipt.outputs) == (5 if many else 2)
    assert len({output.actual for output in receipt.outputs if output.bridge}) == (
        2 if many else 1
    )
    assert all(len(fact.raw) == len(fact.slots) for fact in receipt.normalizations)
    _immutable(receipt)
    if clone_stage == "after":
        clone = model.clone()
        assert clone.graph is not model.graph
        _full(clone)
        cloned = _resolve(clone, witnesses)
        assert all(cloned[key] is not found[key] for key in found)
        assert aliases.slots(tuple(clone.graph.outputs)) == expected
        assert tuple(value.name for value in clone.graph.outputs) == original_names
        assert (
            identity.receipts(clone.graph)
            if operation == "Identity"
            else leaf.receipts(clone.graph)
        ) == parents


@pytest.mark.parametrize("mode", ("noop", "single", "already-shared", "unowned"))
def test_noop_single_shared_and_unowned_paths_do_not_mint_normalization_facts(
    mode: str,
) -> None:
    model, witnesses = _model(many=False, owned=mode != "unowned")
    if mode == "noop":
        for index, node in enumerate(tuple(model.graph)[:3]):
            node.replace_input_with(0, model.graph.inputs[index % 2])
            node.replace_input_with(1, model.graph.inputs[(index + 1) % 2])
        # Keep only the distinct first two leaves and their genuine internal use.
        third = tuple(model.graph)[2]
        model.graph.remove(third, safe=True)
    if mode == "single":
        model.graph.outputs.pop()
    if mode == "already-shared":
        first, second = model.graph.outputs[:2]
        model.graph.outputs[1] = first
        model.graph.outputs[3] = first
        model.graph.node("original_internal").replace_input_with(0, first)
        # The original repeated class is retained; only unused duplicates disappear.
        assert not second.uses() and not second.is_graph_output()
    _full(model)
    if mode == "unowned":
        plain = ir.from_proto(ir.to_proto(model))
        common_passes.CommonSubexpressionEliminationPass()(plain)
        ir_optimizations._run_common_subexpression_elimination_pass(model)
        assert (
            ir.to_proto(model).SerializeToString()
            == ir.to_proto(plain).SerializeToString()
        )
        assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)
        return
    _run(model)
    assert all(
        not item.normalizations
        for item in (*identity.receipts(model.graph), *leaf.receipts(model.graph))
    )
    _full(model)
    if mode != "noop":
        _resolve(model, witnesses)


_PREFLIGHT: tuple[str, ...] = (
    "slot-missing",
    "slot-extra",
    "slot-order",
    "different-class",
    "canonical",
    "operation",
    "domain",
    "overload",
    "attributes",
    "arity",
    "name",
    "type",
    "shape",
    "owner",
    "preexisting",
    "extra-use",
    "captured-use",
    "name-collision",
    "visibility",
    "extra-bridge",
)


@pytest.mark.parametrize("fault", _PREFLIGHT)
def test_all_class_preflight_rejects_actual_malformed_raw_bridges_without_mutation(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _model()
    _full(model)
    actual = common_passes.CommonSubexpressionEliminationPass
    observed = []

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            result = actual()(value)
            graph = value.graph
            bridge = graph.outputs[1].producer()
            assert bridge is not None
            if fault == "slot-missing":
                graph.outputs.pop()
            elif fault == "slot-extra":
                graph.outputs.append(graph.outputs[1])
            elif fault == "slot-order":
                graph.outputs[1], graph.outputs[2] = graph.outputs[2], graph.outputs[1]
            elif fault == "different-class":
                graph.outputs[2] = graph.outputs[1]
            elif fault == "canonical":
                bridge.replace_input_with(0, graph.inputs[0])
            elif fault == "operation":
                bridge.op_type = "Neg"
            elif fault == "domain":
                bridge.domain = "foreign"
            elif fault == "overload":
                bridge.overload = "foreign"
            elif fault == "attributes":
                bridge.attributes["extra"] = ir.AttrInt64("extra", 1)
            elif fault == "arity":
                bridge.resize_inputs(2)
            elif fault == "name":
                bridge.outputs[0].name = "changed"
            elif fault == "type":
                bridge.outputs[0].type = ir.TensorType(ir.DataType.DOUBLE)
            elif fault == "shape":
                bridge.outputs[0].shape = ir.Shape((1, 2))
            elif fault == "owner":
                graph.meta[structure._OWNER] = object()
            elif fault == "preexisting":
                graph.outputs[1] = graph.outputs[0]
            elif fault == "visibility":
                producer = graph.outputs[0].producer()
                assert producer is not None
                graph.insert_after(bridge, producer)
            elif fault == "extra-bridge":
                graph.append(
                    ir.Node(
                        "",
                        "Identity",
                        [graph.outputs[0]],
                        outputs=[_value("second", ir.DataType.FLOAT, (2,))],
                        name="unrecorded_bridge",
                    )
                )
            elif fault in ("extra-use", "captured-use"):
                output = _value("extra", ir.DataType.FLOAT, (2,))
                consumer = ir.Node(
                    "", "Neg", [bridge.outputs[0]], outputs=[output], name="extra_user"
                )
                if fault == "extra-use":
                    graph.append(consumer)
                else:
                    child = ir.Graph(
                        [],
                        [output],
                        nodes=[consumer],
                        name="captured_user",
                        opset_imports={"": 23},
                    )
                    child.meta[structure._OWNER] = object()
                    holder = ir.Node(
                        "",
                        "Holder",
                        [],
                        {"body": ir.AttrGraph("body", child)},
                        outputs=[],
                        name="holder",
                    )
                    graph.append(holder)
            else:
                graph.inputs[0].name = "second"
            observed.append(aliases._physical((graph,)))
            return result

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    if fault == "owner":
        ir_optimizations._run_common_subexpression_elimination_pass(
            model, precision_policy=PrecisionPolicy.SOURCE
        )
    else:
        with pytest.raises(
            structure.StructureMismatch, match="cse-output-alias-required-repair:"
        ) as caught:
            ir_optimizations._run_common_subexpression_elimination_pass(
                model, precision_policy=PrecisionPolicy.SOURCE
            )
    assert len(observed) == 1
    if fault == "owner":
        # Optional provenance corruption cannot veto a valid SOURCE repair.
        _full(model)
        assert aliases.slots(tuple(model.graph.outputs)) == (
            (0,),
            (1, 3, 5),
            (2, 4),
            (6,),
        )
    else:
        assert aliases._physical((model.graph,)) == observed[0]
    assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)
    if fault == "owner":
        assert any(
            "cse-output-alias" in reason
            for reason in model.graph.meta.get(leaf._GAPS, ())
        )
    else:
        causes = {
            "slot-missing": "cse-output-alias-owner-or-slot-count-differs",
            "slot-extra": "cse-output-alias-owner-or-slot-count-differs",
            "slot-order": "cse-output-alias-bridge-interface-differs",
            "different-class": "cse-output-alias-unrecorded-output-occurrence",
            "canonical": "cse-output-alias-canonical-input-differs",
            "operation": "identity-cse-identity-schema-differs",
            "domain": "identity-cse-identity-schema-differs",
            "overload": "identity-cse-identity-schema-differs",
            "attributes": "identity-cse-identity-schema-differs",
            "arity": "identity-cse-identity-schema-differs",
            "name": "cse-output-alias-bridge-interface-differs",
            "type": "identity-cse-identity-schema-differs",
            "shape": "identity-cse-identity-schema-differs",
            "preexisting": "cse-output-alias-bridge-not-fresh",
            "extra-use": "cse-output-alias-bridge-interface-differs",
            "captured-use": "cse-output-alias-current-attachments-differ",
            "name-collision": "cse-output-alias-current-name-ambiguous",
            "visibility": "cse-output-alias-bridge-interface-differs",
            "extra-bridge": "cse-output-alias-current-name-ambiguous",
        }
        assert str(caught.value) == "cse-output-alias-required-repair:" + causes[fault]


@pytest.mark.parametrize(
    "fault",
    ("output-first", "output-second-class", "remove-after-disconnect", "post-verify"),
)
def test_atomic_commit_fault_restores_exact_raw_ports_order_and_output_occurrences(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _model()
    _full(model)
    actual = common_passes.CommonSubexpressionEliminationPass
    armed = []
    failed = []
    raw = []
    output_type = type(model.graph.outputs)
    set_output = output_type.__setitem__
    remove = ir.Graph.remove
    verify = aliases._verify_normalized
    writes = []

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            result = actual()(value)
            raw.append(aliases._physical((value.graph,)))
            armed.append(True)
            return result

    def assign(self: Any, index: Any, value: Any) -> None:
        set_output(self, index, value)
        if armed and not failed and type(index) is int:
            writes.append(index)
            if (fault == "output-first" and len(writes) == 1) or (
                fault == "output-second-class" and len(writes) == 4
            ):
                failed.append(True)
                raise RuntimeError("output assignment fault")

    def safe_remove(self: ir.Graph, nodes: Any, safe: bool = False) -> None:
        remove(self, nodes, safe=safe)
        if armed and not failed and fault == "remove-after-disconnect":
            failed.append(True)
            raise RuntimeError("removal completed before fault")

    def final_verify(*args: Any) -> None:
        verify(*args)
        if armed and not failed and fault == "post-verify":
            failed.append(True)
            raise RuntimeError("verified commit fault")

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    monkeypatch.setattr(output_type, "__setitem__", assign)
    monkeypatch.setattr(ir.Graph, "remove", safe_remove)
    monkeypatch.setattr(aliases, "_verify_normalized", final_verify)
    with pytest.raises(
        structure.StructureMismatch, match="cse-output-alias-required-repair:"
    ) as caught:
        ir_optimizations._run_common_subexpression_elimination_pass(
            model, precision_policy=PrecisionPolicy.SOURCE
        )
    assert len(armed) == len(raw) == len(failed) == 1
    assert aliases._physical((model.graph,)) == raw[0]
    assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)
    assert "fault" in str(caught.value)


_CURRENT: tuple[str, ...] = (
    "split",
    "merge",
    "unrelated-slot",
    "name",
    "input",
    "dtype",
    "extra-use",
    "old-origin",
    "old-marker",
    "parent-absent",
    "parent-forged",
    "foreign",
)


@pytest.mark.parametrize("fault", _CURRENT)
def test_authentic_current_class_and_parent_receipt_reject_corruption(
    fault: str,
) -> None:
    model, witnesses = _model()
    _full(model)
    _run(model)
    _full(model)
    _resolve(model, witnesses)
    graph = model.graph
    (receipt,) = leaf.receipts(graph)
    fact = receipt.normalizations[0]
    bridge = graph.outputs[1].producer()
    assert bridge is not None
    if fault == "split":
        output = _value("second", ir.DataType.FLOAT, (2,))
        graph.append(
            ir.Node(
                "",
                "Identity",
                [graph.outputs[0]],
                outputs=[output],
                name="unauthorized_split",
            )
        )
        graph.outputs[3] = output
    elif fault == "merge":
        for slot in (2, 4):
            graph.outputs[slot] = graph.outputs[1]
    elif fault == "unrelated-slot":
        graph.outputs[6] = graph.outputs[1]
    elif fault == "name":
        graph.outputs[1].name = "changed"
    elif fault == "input":
        bridge.replace_input_with(0, graph.inputs[0])
    elif fault == "dtype":
        graph.outputs[1].type = ir.TensorType(ir.DataType.DOUBLE)
    elif fault == "extra-use":
        graph.append(
            ir.Node(
                "",
                "Neg",
                [graph.outputs[1]],
                outputs=[_value("extra", ir.DataType.FLOAT, (2,))],
                name="extra_user",
            )
        )
    elif fault == "old-origin":
        output = _value("revived", ir.DataType.FLOAT, (2,))
        node = ir.Node(
            "", "Identity", [graph.outputs[0]], outputs=[output], name="revived_bridge"
        )
        node.meta[structure._ORIGIN] = fact.raw[1].value.origin
        graph.append(node)
    elif fault == "old-marker":
        graph.outputs[1].meta[structure._VALUES] += (fact.raw[1].value,)
    elif fault == "parent-absent":
        graph.meta.pop(leaf._RECEIPTS)
    elif fault == "parent-forged":
        graph.meta[leaf._RECEIPTS] = (replace(receipt),)
    else:
        foreign, _ = _model()
        _run(foreign)
        graph.meta[leaf._RECEIPTS] = leaf.receipts(foreign.graph)
    with pytest.raises((structure.StructureMismatch, structure.StructureUnavailable)):
        _resolve(model, witnesses)


@pytest.mark.parametrize("stage", ("capture", "physical", "pass-exception"))
def test_observation_failure_never_skips_repeats_or_replaces_original_pass(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _model(many=False)
    original = common_passes.CommonSubexpressionEliminationPass
    calls = []
    sentinel = RuntimeError("original exception")

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            if stage == "pass-exception":
                raise sentinel
            return original()(value)

    def unavailable(*args: Any) -> Any:
        raise structure.StructureUnavailable("bounded snapshot unavailable")

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    if stage in ("capture", "physical"):
        monkeypatch.setattr(
            aliases, "_capture" if stage == "capture" else "_physical", unavailable
        )
        if stage == "physical":
            with pytest.raises(
                structure.StructureMismatch,
                match="cse-output-alias-required-repair:bounded snapshot unavailable",
            ):
                ir_optimizations._run_common_subexpression_elimination_pass(
                    model, precision_policy=PrecisionPolicy.SOURCE
                )
        else:
            ir_optimizations._run_common_subexpression_elimination_pass(
                model, precision_policy=PrecisionPolicy.SOURCE
            )
        assert not leaf.receipts(model.graph)
        if stage == "capture":
            assert model.graph.meta.get(leaf._GAPS)
    else:
        monkeypatch.setattr(aliases.Transaction, "finish", unavailable)
        with pytest.raises(RuntimeError) as caught:
            ir_optimizations._run_common_subexpression_elimination_pass(
                model, precision_policy=PrecisionPolicy.SOURCE
            )
        assert caught.value is sentinel
    assert calls == [model]


def test_normalization_facts_are_inert_immutable_and_release_every_native_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = []
    plans = []
    capture = aliases._capture
    plan = aliases._plan

    def captured(*args: Any) -> Any:
        value = capture(*args)
        snapshots.append(weakref.ref(value))
        return value

    def planned(*args: Any) -> Any:
        values = plan(*args)
        plans.extend(weakref.ref(value) for value in values)
        return values

    monkeypatch.setattr(aliases, "_capture", captured)
    monkeypatch.setattr(aliases, "_plan", planned)
    model, witnesses = _model()
    _run(model)
    _resolve(model, witnesses)
    (receipt,) = leaf.receipts(model.graph)
    assert receipt.normalizations and all(
        reference() is None for reference in (*snapshots, *plans)
    )
    _immutable(receipt)
    facts = receipt.normalizations
    graph_reference = weakref.ref(model.graph)
    del model, receipt, witnesses
    gc.collect()
    assert graph_reference() is None
    _immutable(facts)
    assert not hasattr(aliases, "resolve_bindings") and not hasattr(
        aliases, "_REGISTERED"
    )


@pytest.mark.parametrize("operation", ("Identity", "Constant", "Sub"))
def test_actual_owned_child_pass_keeps_valid_lexical_capture_and_disjoint_owners(
    operation: str,
) -> None:
    model, witnesses = _model(operation)
    child = model.graph
    x, y = child.inputs
    # Release the original formal ownership before making these values captures.
    child.inputs.clear()
    parent = ir.Graph(
        [x, y], [], nodes=[], name="lexical_parent", opset_imports={"": 23}
    )
    parent.meta[structure._OWNER] = object()
    other = child.clone(allow_outer_scope_values=True)
    other.meta[structure._OWNER] = object()
    flag = _value("flag", ir.DataType.BOOL, ())
    parent.inputs.append(flag)
    outputs = [
        _value(f"result_{i}", ir.DataType.FLOAT, (2,))
        for i in range(len(child.outputs))
    ]
    parent.append(
        ir.Node(
            "",
            "If",
            [flag],
            {
                "then_branch": ir.AttrGraph("then_branch", child),
                "else_branch": ir.AttrGraph("else_branch", other),
            },
            outputs=outputs,
            name="lexical_if",
        )
    )
    parent.outputs.extend(outputs)
    whole = ir.Model(parent, ir_version=10)
    _full(whole)
    other_bytes = ir.to_proto(ir.Model(other, ir_version=10)).SerializeToString()
    _run(model)
    assert model.graph is child
    assert not child.inputs
    assert (
        ir.to_proto(ir.Model(other, ir_version=10)).SerializeToString() == other_bytes
    )
    assert parent.meta[structure._OWNER] is not child.meta[structure._OWNER]
    assert other.meta[structure._OWNER] is not child.meta[structure._OWNER]
    _resolve(model, witnesses)
    assert child.outputs[1] is child.outputs[3] is child.outputs[5]
    _full(whole)


def test_changed_raw_fact_between_plan_and_commit_is_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _model()
    actual = common_passes.CommonSubexpressionEliminationPass
    original_plan = aliases._plan
    raw = []

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            result = actual()(value)
            raw.append(aliases._physical((value.graph,)))
            return result

    def altered(*args: Any) -> Any:
        plans = original_plan(*args)
        args[0].graph.outputs[6] = args[0].graph.outputs[1]
        return plans

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    monkeypatch.setattr(aliases, "_plan", altered)
    with pytest.raises(
        structure.StructureMismatch,
        match="cse-output-alias-required-repair:cse-output-alias-preflight-mutated",
    ) as caught:
        ir_optimizations._run_common_subexpression_elimination_pass(
            model, precision_policy=PrecisionPolicy.SOURCE
        )
    assert len(raw) == 1 and aliases._physical((model.graph,)) == raw[0]
    assert not leaf.receipts(model.graph) and not identity.receipts(model.graph)
    assert "preflight-mutated" in str(caught.value)


def test_unverifiable_rollback_raises_explicitly_without_parent_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _model()
    verify = aliases._verify_normalized
    armed = []

    def fail_after_commit(*args: Any) -> None:
        verify(*args)
        armed.append(True)
        raise RuntimeError("commit verification fault")

    def cannot_restore(*args: Any) -> None:
        assert armed
        raise RuntimeError("public rollback fault")

    # Arm the insert fault only after the actual upstream pass has returned.
    actual = common_passes.CommonSubexpressionEliminationPass

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            result = actual()(value)
            monkeypatch.setattr(ir.Graph, "insert_before", cannot_restore)
            return result

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    monkeypatch.setattr(aliases, "_verify_normalized", fail_after_commit)
    with pytest.raises(
        structure.StructureMismatch, match="cse-output-alias-rollback-failed"
    ):
        ir_optimizations._run_common_subexpression_elimination_pass(
            model, precision_policy=PrecisionPolicy.SOURCE
        )
    assert armed == [True]
    assert not leaf.receipts(model.graph) and not identity.receipts(model.graph)


@pytest.mark.parametrize(
    "fault",
    (
        "omitted",
        "duplicate",
        "owner",
        "original",
        "slots",
        "name",
        "canonical",
        "survivor",
        "raw-order",
        "raw-slot",
        "raw-input",
        "raw-schema",
    ),
)
def test_inert_facts_cannot_omit_or_forge_the_authentic_parent_class(
    fault: str,
) -> None:
    model, witnesses = _model()
    _run(model)
    _resolve(model, witnesses)
    (receipt,) = leaf.receipts(model.graph)
    fact = receipt.normalizations[0]
    resolved = structure._resolve_bindings(model.graph, leaf._dependencies(receipt))
    facts = receipt.normalizations
    if fault == "omitted":
        facts = facts[1:]
    elif fault == "duplicate":
        facts = (fact, *facts)
    else:
        if fault == "owner":
            changed = replace(fact, owner=object())
        elif fault == "original":
            changed = replace(fact, original=receipt.outputs[0].original)
        elif fault == "slots":
            changed = replace(fact, slots=fact.slots[:-1])
        elif fault == "name":
            changed = replace(fact, name="different")
        elif fault == "canonical":
            changed = replace(fact, canonical=fact.survivor)
        elif fault == "survivor":
            changed = replace(fact, survivor=fact.canonical)
        elif fault == "raw-order":
            changed = replace(fact, raw=tuple(reversed(fact.raw)))
        elif fault == "raw-slot":
            changed = replace(
                fact, raw=(replace(fact.raw[0], slots=fact.slots), *fact.raw[1:])
            )
        elif fault == "raw-input":
            changed = replace(
                fact, raw=(replace(fact.raw[0], input=fact.survivor), *fact.raw[1:])
            )
        else:
            changed = replace(
                fact,
                raw=(
                    replace(
                        fact.raw[0], schema=identity._Schema(ir.DataType.DOUBLE, (2,))
                    ),
                    *fact.raw[1:],
                ),
            )
        facts = (changed, *facts[1:])
    with pytest.raises((structure.StructureMismatch, structure.StructureUnavailable)):
        aliases.validate_facts(model.graph, facts, receipt.outputs, resolved)
    # A copied parent is not authenticated, even when extracted facts are genuine.
    model.graph.meta[leaf._RECEIPTS] = (replace(receipt, normalizations=facts),)
    with pytest.raises(structure.StructureUnavailable):
        leaf.receipts(model.graph)


@pytest.mark.parametrize(
    "dtype", (ir.DataType.FLOAT, ir.DataType.DOUBLE), ids=("FLOAT", "DOUBLE")
)
@pytest.mark.parametrize("shape", ((), (2,)), ids=("scalar", "2"))
def test_normalized_constant_classes_compose_only_with_the_real_later_lift(
    dtype: ir.DataType, shape: tuple[int, ...]
) -> None:
    model, witnesses = _model("Constant", dtype, shape)
    graph = model.graph
    internals = [
        _value(name, dtype, shape)
        for name in ("internal_constant_a", "internal_constant_b")
    ]
    for index, value in enumerate(internals):
        array = np.full(
            shape, 0.5, dtype=np.float32 if dtype is ir.DataType.FLOAT else np.float64
        )
        graph.append(
            ir.Node(
                "",
                "Constant",
                [],
                {"value": ir.AttrTensor("value", ir.tensor(array))},
                outputs=[value],
                name=f"internal_constant_{index}",
            )
        )
        output = _value(f"live_use_{index}", dtype, shape)
        graph.append(
            ir.Node(
                "",
                "Sub",
                [graph.inputs[index], value],
                outputs=[output],
                name=f"live_consumer_{index}",
            )
        )
        graph.outputs.append(output)
    internal_witnesses = tuple(structure._witness(value) for value in internals)
    assert (
        internals[0].producer().attributes["value"].as_tensor().numpy().tobytes()
        == internals[1].producer().attributes["value"].as_tensor().numpy().tobytes()
    )
    assert (
        internals[0].producer().attributes["value"].as_tensor().numpy().tobytes()
        != graph.outputs[0].producer().attributes["value"].as_tensor().numpy().tobytes()
    )
    _full(model)
    _run(model)
    found = _resolve(model, (*witnesses, *internal_witnesses))
    assert found[internal_witnesses[0]] is found[internal_witnesses[1]]
    internal_canonical = found[internal_witnesses[0]]
    assert internal_canonical not in graph.outputs
    output_constant = graph.outputs[0]
    (receipt,) = leaf.receipts(graph)
    facts = receipt.normalizations
    original_names = tuple(value.name for value in graph.outputs)
    assert len(facts) == 2
    ir_optimizations._run_lift_constants_to_initializers_pass(model)
    (lift,) = structure._constant_lift_receipts(graph)
    assert lift.before == internal_witnesses[0]
    assert all(lift.before != output.original for output in receipt.outputs)
    assert (
        output_constant is graph.outputs[0]
        and output_constant.producer().op_type == "Constant"
    )
    assert sum(node.op_type == "Constant" for node in graph) == 1
    assert len(graph.initializers) == 1
    _full(model)
    lifted = _resolve(model, (*witnesses, *internal_witnesses))
    assert lifted[internal_witnesses[0]] is lifted[internal_witnesses[1]]
    assert lifted[internal_witnesses[0]] is next(iter(graph.initializers.values()))
    assert tuple(value.name for value in graph.outputs) == original_names
    assert graph.outputs[1] is graph.outputs[3] is graph.outputs[5]
    assert leaf.receipts(graph)[0] is receipt and receipt.normalizations is facts
    clone = model.clone()
    _full(clone)
    _resolve(clone, (*witnesses, *internal_witnesses))
    assert aliases.slots(tuple(clone.graph.outputs)) == (
        (0,),
        (1, 3, 5),
        (2, 4),
        (6,),
        (7,),
        (8,),
    )


def test_output_constant_class_is_retained_without_a_fictitious_lift() -> None:
    model, witnesses = _model("Constant")
    _full(model)
    _run(model)
    _resolve(model, witnesses)
    before = ir.to_proto(model).SerializeToString(deterministic=True)
    ir_optimizations._run_lift_constants_to_initializers_pass(model)
    assert not structure._constant_lift_receipts(model.graph)
    assert len(model.graph.initializers) == 0
    assert sum(node.op_type == "Constant" for node in model.graph) == 1
    assert ir.to_proto(model).SerializeToString(deterministic=True) == before
    _full(model)
    _resolve(model, witnesses)


@pytest.mark.parametrize("operation", ("Identity", "Constant", "Sub"))
@pytest.mark.parametrize(
    "recording", ("on", "off", "current-failure", "witness-failure")
)
def test_source_physical_bytes_and_one_pass_are_independent_of_optional_recording(
    operation: str,
    recording: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, _ = _model(operation)
    _full(baseline)
    _run(baseline)
    expected = ir.to_proto(baseline).SerializeToString(deterministic=True)
    model, _ = _model(operation, owned=recording != "off")
    _full(model)
    original = common_passes.CommonSubexpressionEliminationPass
    calls = []

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            return original()(value)

    def unavailable(*args: Any) -> Any:
        raise structure.StructureUnavailable(
            "optional recording deliberately unavailable"
        )

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    if recording in ("off", "current-failure"):
        monkeypatch.setattr(identity, "_current", unavailable)
    if recording in ("off", "witness-failure"):
        monkeypatch.setattr(structure, "_witness", unavailable)
    ir_optimizations._run_common_subexpression_elimination_pass(
        model, precision_policy=PrecisionPolicy.SOURCE
    )
    assert calls == [model]
    _full(model)
    assert ir.to_proto(model).SerializeToString(deterministic=True) == expected
    assert aliases.slots(tuple(model.graph.outputs)) == ((0,), (1, 3, 5), (2, 4), (6,))
    clone = model.clone()
    _full(clone)
    assert ir.to_proto(clone).SerializeToString(deterministic=True) == expected
    if recording != "on":
        assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)
    if recording == "off":
        assert structure._OWNER not in model.graph.meta
        assert all(
            structure._VALUES not in value.meta
            for node in model.graph
            for value in node.outputs
        )


@pytest.mark.parametrize("owned", (False, True), ids=("unowned", "incidental-owner"))
def test_legacy_default_keeps_exact_upstream_result_even_with_incidental_owner(
    owned: bool,
) -> None:
    model, _ = _model(owned=owned)
    plain, _ = _model(owned=False)
    _full(model)
    _full(plain)
    common_passes.CommonSubexpressionEliminationPass()(plain)
    ir_optimizations._run_common_subexpression_elimination_pass(model)
    assert ir.to_proto(model).SerializeToString(deterministic=True) == ir.to_proto(
        plain
    ).SerializeToString(deterministic=True)
    assert model.graph.outputs[1] is not model.graph.outputs[3]
    with pytest.raises(onnx.checker.ValidationError, match="SSA|single static"):
        _full(model)
    assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)


def test_real_optimize_graph_forwards_source_enum_without_owner_or_ambient_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _model("Sub", owned=False)
    _full(model)
    capture = aliases.capture
    original = common_passes.CommonSubexpressionEliminationPass
    policies = []
    calls = []

    def captured(
        value: ir.Model, policy: PrecisionPolicy = PrecisionPolicy.LEGACY
    ) -> Any:
        policies.append((value, policy))
        return capture(value, policy)

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            return original()(value)

    def forbidden(*args: Any) -> Any:
        raise AssertionError("physical SOURCE must not acquire a witness")

    monkeypatch.setattr(aliases, "capture", captured)
    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    monkeypatch.setattr(identity, "_current", forbidden)
    monkeypatch.setattr(structure, "_witness", forbidden)
    assert (
        ir_optimizations.optimize_graph(model, precision_policy=PrecisionPolicy.SOURCE)
        is model
    )
    assert policies == [(model, PrecisionPolicy.SOURCE)] and calls == [model]
    _full(model)
    assert model.graph.outputs[1] is model.graph.outputs[3] is model.graph.outputs[5]
    assert structure._OWNER not in model.graph.meta
    assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)


@pytest.mark.parametrize(
    "real_producer",
    (False, True),
    ids=("fake-name-foreign-runner", "real-runner-different-name"),
)
def test_dispatch_uses_genuine_producer_identity_never_a_pass_display_name(
    real_producer: bool,
) -> None:
    model, _ = _model(owned=False)
    _full(model)
    called = []

    def foreign(value: ir.Model) -> None:
        called.append(value)
        common_passes.CommonSubexpressionEliminationPass()(value)

    runner = (
        ir_optimizations._run_common_subexpression_elimination_pass
        if real_producer
        else foreign
    )
    entry = ir_optimizations._OptimizerPass(
        name=(
            "unrelated-display-name"
            if real_producer
            else "common_subexpression_elimination"
        ),
        precision_policies=frozenset({PrecisionPolicy.SOURCE}),
        model_runner=runner,
    )
    ir_optimizations._run_top_level_optimizer_pass(
        entry, model, precision_policy=PrecisionPolicy.SOURCE
    )
    if real_producer:
        _full(model)
        assert (
            not called
            and model.graph.outputs[1]
            is model.graph.outputs[3]
            is model.graph.outputs[5]
        )
    else:
        assert called == [model]
        assert model.graph.outputs[1] is not model.graph.outputs[3]
        with pytest.raises(onnx.checker.ValidationError, match="SSA|single static"):
            _full(model)


def test_actual_source_conversion_keeps_proto_report_source_and_pass_counts_on_off_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = common_passes.CommonSubexpressionEliminationPass
    capture = aliases.capture
    pass_calls = []
    policies = []
    source_calls = []
    models = []
    reports = []

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            pass_calls.append(value)
            return actual()(value)

    def captured(
        value: ir.Model, policy: PrecisionPolicy = PrecisionPolicy.LEGACY
    ) -> Any:
        policies.append(policy)
        return capture(value, policy)

    def unavailable(*args: Any) -> Any:
        raise structure.StructureUnavailable("optional current witness unavailable")

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    monkeypatch.setattr(aliases, "capture", captured)
    for mode in ("on", "off", "failure"):
        calls = []

        def observed(left: Any, right: Any) -> Any:
            calls.append(1)
            first = left - right
            second = left - right
            return first, second, -second, second

        with monkeypatch.context() as change:
            if mode == "off":
                change.setattr(ir_context, "SourceStructureLedger", lambda: None)
            elif mode == "failure":
                change.setattr(identity, "_current", unavailable)
            with jax.enable_x64(True):
                model = ui._to_onnx(
                    observed,
                    [jax.ShapeDtypeStruct((2,), np.float32)] * 2,
                    precision_policy=PrecisionPolicy.SOURCE,
                    return_mode="ir",
                    opset=23,
                )
        assert isinstance(model, ir.Model)
        _full(model)
        models.append(model)
        source_calls.append(tuple(calls))
        reports.append(get_precision_report(model))
        if mode != "on":
            assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)
        if mode == "off":
            assert structure._LEDGER not in model.meta
    assert policies == [PrecisionPolicy.SOURCE] * 3 and pass_calls == models
    assert source_calls[0] and source_calls[0] == source_calls[1] == source_calls[2]
    assert reports[0] is not None and reports[0] == reports[1] == reports[2]
    assert (
        precision_report_to_json(reports[0])
        == precision_report_to_json(reports[1])
        == precision_report_to_json(reports[2])
    )
    assert (
        len(
            {
                ir.to_proto(model).SerializeToString(deterministic=True)
                for model in models
            }
        )
        == 1
    )


def _lexical_model(
    *, actual: bool, name: str = "second"
) -> tuple[ir.Model, tuple[Any, ...], ir.Node]:
    model, witnesses = _model()
    graph = model.graph
    children = []
    for label in ("then", "else"):
        captured = graph.outputs[1] if actual else _value(name, ir.DataType.FLOAT, (2,))
        output = _value(f"{label}_value", ir.DataType.FLOAT, (2,))
        child = ir.Graph(
            [],
            [output],
            nodes=[
                ir.Node(
                    "",
                    "Identity",
                    [captured],
                    outputs=[output],
                    name=f"{label}_identity",
                )
            ],
            name=f"{label}_branch",
            opset_imports={"": 23},
        )
        child.meta[structure._OWNER] = object()
        children.append(child)
    flag = _value("condition", ir.DataType.BOOL, ())
    graph.inputs.append(flag)
    result = _value("if_value", ir.DataType.FLOAT, (2,))
    holder = ir.Node(
        "",
        "If",
        [flag],
        {
            "then_branch": ir.AttrGraph("then_branch", children[0]),
            "else_branch": ir.AttrGraph("else_branch", children[1]),
        },
        outputs=[result],
        name="lexical_holder",
    )
    graph.append(holder)
    graph.outputs.append(result)
    return model, witnesses, holder


@pytest.mark.parametrize(
    "actual", (False, True), ids=("unresolved-name-capture", "actual-value-capture")
)
def test_real_lexical_captures_are_distinguished_without_name_alias_authority(
    actual: bool,
) -> None:
    model, witnesses, _ = _lexical_model(actual=actual)
    _full(model)
    if actual:
        _run(model)
        _full(model)
        _resolve(model, witnesses)
        assert (
            model.graph.outputs[1] is model.graph.outputs[3] is model.graph.outputs[5]
        )
        canonical = model.graph.outputs[0]
        for node in model.graph:
            for attr in node.attributes.values():
                if attr.type is ir.AttributeType.GRAPH:
                    assert tuple(attr.as_graph())[0].inputs[0] is canonical
    else:
        raw = []
        original = common_passes.CommonSubexpressionEliminationPass
        # Use a plain pass once under the actual observer/normalizer wrapper.
        with pytest.MonkeyPatch.context() as change:

            class Original:
                def __call__(self, value: ir.Model) -> Any:
                    result = original()(value)
                    raw.append(aliases._physical(aliases._scope(value).graphs))
                    return result

            change.setattr(
                common_passes, "CommonSubexpressionEliminationPass", Original
            )
            with pytest.raises(
                structure.StructureMismatch,
                match="cse-output-alias-required-repair:cse-output-alias-hidden-interface-use",
            ):
                ir_optimizations._run_common_subexpression_elimination_pass(
                    model, precision_policy=PrecisionPolicy.SOURCE
                )
        assert (
            len(raw) == 1 and aliases._physical(aliases._scope(model).graphs) == raw[0]
        )
        assert model.graph.outputs[1] is not model.graph.outputs[3]
        assert not leaf.receipts(model.graph) and not identity.receipts(model.graph)


@pytest.mark.parametrize(
    "actual", (False, True), ids=("new-lexical-placeholder", "new-actual-value-use")
)
def test_new_current_child_attachment_rejects_before_physical_mutation(
    actual: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, holder = _lexical_model(actual=False, name="x")
    _full(model)
    original = common_passes.CommonSubexpressionEliminationPass
    raw = []
    transactions = []
    capture = aliases.capture

    def captured(*args: Any) -> Any:
        result = capture(*args)
        transactions.append(result)
        return result

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            result = original()(value)
            operand = (
                value.graph.outputs[1]
                if actual
                else _value("second", ir.DataType.FLOAT, (2,))
            )
            output = _value("new_child_value", ir.DataType.FLOAT, (2,))
            child = ir.Graph(
                [],
                [output],
                nodes=[
                    ir.Node(
                        "",
                        "Identity",
                        [operand],
                        outputs=[output],
                        name="new_child_identity",
                    )
                ],
                name="new_child",
                opset_imports={"": 23},
            )
            holder.attributes["then_branch"] = ir.AttrGraph("then_branch", child)
            raw.append(aliases._physical(aliases._scope(value).graphs))
            return result

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    monkeypatch.setattr(aliases, "capture", captured)
    with pytest.raises(
        structure.StructureMismatch,
        match="cse-output-alias-required-repair:cse-output-alias-current-attachments-differ",
    ):
        ir_optimizations._run_common_subexpression_elimination_pass(
            model, precision_policy=PrecisionPolicy.SOURCE
        )
    assert len(raw) == len(transactions) == 1
    assert "current-attachments-differ" in transactions[0].failure
    assert aliases._physical(aliases._scope(model).graphs) == raw[0]
    assert not leaf.receipts(model.graph) and not identity.receipts(model.graph)


def test_postplan_holder_mutation_is_rejected_and_raw_attachments_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, holder = _lexical_model(actual=False, name="x")
    _full(model)
    record = aliases.Observation.record
    capture = aliases.capture
    transactions = []
    raw = []
    changed = []

    def captured(*args: Any) -> Any:
        result = capture(*args)
        transactions.append(result)
        return result

    def altered(self: aliases.Observation, transaction: aliases.Transaction) -> None:
        record(self, transaction)
        if not changed:
            raw.append(aliases._physical(aliases._scope(model).graphs))
            output = _value("late_child_value", ir.DataType.FLOAT, (2,))
            placeholder = _value("second", ir.DataType.FLOAT, (2,))
            child = ir.Graph(
                [],
                [output],
                nodes=[
                    ir.Node(
                        "",
                        "Identity",
                        [placeholder],
                        outputs=[output],
                        name="late_child_identity",
                    )
                ],
                name="late_child",
                opset_imports={"": 23},
            )
            holder.attributes["then_branch"] = ir.AttrGraph("then_branch", child)
            changed.append(True)

    monkeypatch.setattr(aliases, "capture", captured)
    monkeypatch.setattr(aliases.Observation, "record", altered)
    with pytest.raises(
        structure.StructureMismatch,
        match="cse-output-alias-required-repair:cse-output-alias-precommit-mutated",
    ):
        ir_optimizations._run_common_subexpression_elimination_pass(
            model, precision_policy=PrecisionPolicy.SOURCE
        )
    assert len(raw) == len(transactions) == len(changed) == 1
    assert "precommit-mutated" in transactions[0].failure
    assert aliases._physical(aliases._scope(model).graphs) == raw[0]
    assert not leaf.receipts(model.graph) and not identity.receipts(model.graph)


@pytest.mark.parametrize("recording", ("on", "off", "failure"))
def test_foreign_defined_serialized_capture_rejects_without_name_authority(
    recording: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, holder = _lexical_model(actual=False)
    foreign_graphs = []
    for label in ("then", "else"):
        child = holder.attributes[f"{label}_branch"].as_graph()
        formal = _value(f"foreign_{label}_input", ir.DataType.FLOAT, (2,))
        foreign = _value("second", ir.DataType.FLOAT, (2,))
        producer = ir.Node(
            "",
            "Identity",
            [formal],
            outputs=[foreign],
            name=f"foreign_{label}_producer",
        )
        foreign_graphs.append(
            ir.Graph(
                [formal],
                [foreign],
                nodes=[producer],
                name=f"unattached_{label}",
                opset_imports={"": 23},
            )
        )
        tuple(child)[0].replace_input_with(0, foreign)
    # Serialization contains the legal outer capture, never the unattached
    # defining graphs. Their actual producer identities confer no visibility.
    _full(model)
    if recording == "off":
        for graph in aliases._scope(model).graphs:
            graph.meta.pop(structure._OWNER, None)
    elif recording == "failure":

        def unavailable(*args: Any) -> Any:
            raise structure.StructureUnavailable(
                "optional foreign recording unavailable"
            )

        monkeypatch.setattr(identity, "_current", unavailable)
    original = common_passes.CommonSubexpressionEliminationPass
    raw = []
    calls = []

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            result = original()(value)
            raw.append(aliases._physical(aliases._scope(value).graphs))
            return result

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    with pytest.raises(
        structure.StructureMismatch,
        match="cse-output-alias-required-repair:cse-output-alias-hidden-interface-use",
    ):
        ir_optimizations._run_common_subexpression_elimination_pass(
            model, precision_policy=PrecisionPolicy.SOURCE
        )
    assert calls == [model] and len(raw) == 1 and len(foreign_graphs) == 2
    assert aliases._physical(aliases._scope(model).graphs) == raw[0]
    assert model.graph.outputs[1] is not model.graph.outputs[3]
    assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)


@pytest.mark.parametrize("binding", ("sibling", "unordered", "shadowed"))
def test_current_occurrence_rejects_nonvisible_nearest_binding(
    binding: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, holder = _lexical_model(actual=False, name="x")
    _full(model)
    original = common_passes.CommonSubexpressionEliminationPass
    raw = []
    calls = []

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            result = original()(value)
            child = holder.attributes["then_branch"].as_graph()
            consumer = tuple(child)[0]
            if binding == "sibling":
                sibling = holder.attributes["else_branch"].as_graph()
                foreign = sibling.outputs[0]
                foreign.name = "second"
                consumer.replace_input_with(0, foreign)
            else:
                local = _value("second", ir.DataType.FLOAT, (2,))
                producer = ir.Node(
                    "",
                    "Identity",
                    [value.graph.inputs[0]],
                    outputs=[local],
                    name="late_local_binding",
                )
                child.append(producer)
                consumer.replace_input_with(
                    0,
                    (
                        local
                        if binding == "unordered"
                        else _value("second", ir.DataType.FLOAT, (2,))
                    ),
                )
            raw.append(aliases._physical(aliases._scope(value).graphs))
            return result

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    with pytest.raises(
        structure.StructureMismatch,
        match="cse-output-alias-required-repair:cse-output-alias-hidden-interface-use",
    ):
        ir_optimizations._run_common_subexpression_elimination_pass(
            model, precision_policy=PrecisionPolicy.SOURCE
        )
    assert calls == [model] and len(raw) == 1
    assert aliases._physical(aliases._scope(model).graphs) == raw[0]
    assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)


@pytest.mark.parametrize("stage", ("precommit", "postcommit"))
@pytest.mark.parametrize("recording", ("on", "off", "failure"))
def test_inplace_public_attribute_name_is_detected_and_exactly_restored(
    stage: str,
    recording: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, holder = _lexical_model(actual=False, name="x")
    _full(model)
    if recording == "off":
        for graph in aliases._scope(model).graphs:
            graph.meta.pop(structure._OWNER, None)
    elif recording == "failure":

        def unavailable(*args: Any) -> Any:
            raise structure.StructureUnavailable(
                "optional attribute recording unavailable"
            )

        monkeypatch.setattr(identity, "_current", unavailable)
    original = common_passes.CommonSubexpressionEliminationPass
    record = aliases.Observation.record
    verify = aliases._verify_normalized
    raw = []
    calls = []
    changes = []
    attribute = holder.attributes["then_branch"]
    child = attribute.as_graph()

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            result = original()(value)
            raw.append(aliases._physical(aliases._scope(value).graphs))
            return result

    def alter() -> None:
        if not changes:
            assert holder.attributes["then_branch"] is attribute
            attribute.name = "changed"
            changes.append(True)

    def recorded(self: aliases.Observation, transaction: aliases.Transaction) -> None:
        record(self, transaction)
        if stage == "precommit":
            alter()

    def verified(*args: Any) -> None:
        if stage == "postcommit":
            alter()
        verify(*args)

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    monkeypatch.setattr(aliases.Observation, "record", recorded)
    monkeypatch.setattr(aliases, "_verify_normalized", verified)
    cause = (
        "precommit-mutated" if stage == "precommit" else "normalized-node-fields-differ"
    )
    with pytest.raises(
        structure.StructureMismatch,
        match="cse-output-alias-required-repair:cse-output-alias-" + cause,
    ):
        ir_optimizations._run_common_subexpression_elimination_pass(
            model, precision_policy=PrecisionPolicy.SOURCE
        )
    assert calls == [model] and changes == [True] and len(raw) == 1
    assert (
        holder.attributes["then_branch"] is attribute
        and attribute.name == "then_branch"
    )
    assert attribute.as_graph() is child and attribute.type is ir.AttributeType.GRAPH
    assert attribute.ref_attr_name is None
    assert aliases._physical(aliases._scope(model).graphs) == raw[0]
    assert model.graph.outputs[1] is not model.graph.outputs[3]
    assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)


@pytest.mark.parametrize("recording", ("on", "off", "failure"))
@pytest.mark.parametrize("fault", ("bridge", "lexical", "commit"))
def test_required_physical_error_is_uniform_and_survives_optional_recording(
    recording: str,
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, holder = _lexical_model(actual=False, name="x")
    _full(model)
    if recording == "off":
        for graph in aliases._scope(model).graphs:
            graph.meta.pop(structure._OWNER, None)
    elif recording == "failure":

        def unavailable(*args: Any) -> Any:
            raise structure.StructureUnavailable("optional gate recording unavailable")

        monkeypatch.setattr(identity, "_current", unavailable)
    original = common_passes.CommonSubexpressionEliminationPass
    capture = aliases.capture
    verify = aliases._verify_normalized
    calls = []
    raw = []
    transactions = []

    def captured(*args: Any) -> Any:
        transaction = capture(*args)
        transactions.append(transaction)
        return transaction

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            result = original()(value)
            if fault == "bridge":
                value.graph.outputs[1].producer().replace_input_with(
                    0, value.graph.inputs[0]
                )
            elif fault == "lexical":
                tuple(holder.attributes["then_branch"].as_graph())[
                    0
                ].replace_input_with(0, _value("second", ir.DataType.FLOAT, (2,)))
            raw.append(aliases._physical(aliases._scope(value).graphs))
            return result

    def verified(*args: Any) -> None:
        verify(*args)
        raise RuntimeError("gate commit fault")

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    monkeypatch.setattr(aliases, "capture", captured)
    if fault == "commit":
        monkeypatch.setattr(aliases, "_verify_normalized", verified)
    cause = {
        "bridge": "cse-output-alias-canonical-input-differs",
        "lexical": "cse-output-alias-hidden-interface-use",
        "commit": "gate commit fault",
    }[fault]
    with pytest.raises(
        structure.StructureMismatch, match="cse-output-alias-required-repair:" + cause
    ):
        ir_optimizations._run_common_subexpression_elimination_pass(
            model, precision_policy=PrecisionPolicy.SOURCE
        )
    assert calls == [model] and len(raw) == len(transactions) == 1
    assert transactions[0].affected and not transactions[0].plans
    assert cause in transactions[0].failure
    assert aliases._physical(aliases._scope(model).graphs) == raw[0]
    assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)


def test_no_detected_split_does_not_turn_unrelated_snapshot_unavailability_into_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _model(many=False)
    nodes = tuple(model.graph)
    nodes[1].replace_input_with(0, model.graph.inputs[1])
    nodes[1].replace_input_with(1, model.graph.inputs[0])
    model.graph.remove(nodes[2], safe=True)
    _full(model)
    original = common_passes.CommonSubexpressionEliminationPass
    calls = []

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            return original()(value)

    def unavailable(*args: Any) -> Any:
        raise structure.StructureUnavailable("unrelated detailed capture unavailable")

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    monkeypatch.setattr(aliases, "_physical", unavailable)
    ir_optimizations._run_common_subexpression_elimination_pass(
        model, precision_policy=PrecisionPolicy.SOURCE
    )
    assert calls == [model] and model.graph.outputs[1] is model.graph.outputs[3]
    _full(model)
    assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)


def _two_holder_capture_model() -> (
    tuple[ir.Model, tuple[Any, ...], tuple[ir.Node, ...]]
):
    model, witnesses = _model()
    graph = model.graph
    outer_flag = _value("deep_outer_flag", ir.DataType.BOOL, ())
    inner_flag = _value("deep_inner_flag", ir.DataType.BOOL, ())
    graph.inputs.extend((outer_flag, inner_flag))
    captured = graph.outputs[1]
    leaves = []
    branches = []
    for outer in ("then", "else"):
        grandchildren = []
        for inner in ("then", "else"):
            output = _value(f"{outer}_{inner}_captured", ir.DataType.FLOAT, (2,))
            consumer = ir.Node(
                "",
                "Sub",
                [captured, graph.inputs[0]],
                outputs=[output],
                name=f"{outer}_{inner}_captured_sub",
            )
            child = ir.Graph(
                [],
                [output],
                nodes=[consumer],
                name=f"{outer}_{inner}_deep_branch",
                opset_imports={"": 23},
            )
            child.meta[structure._OWNER] = object()
            grandchildren.append(child)
            leaves.append(consumer)
        middle_output = _value(f"{outer}_middle_output", ir.DataType.FLOAT, (2,))
        middle_holder = ir.Node(
            "",
            "If",
            [inner_flag],
            {
                "then_branch": ir.AttrGraph("then_branch", grandchildren[0]),
                "else_branch": ir.AttrGraph("else_branch", grandchildren[1]),
            },
            outputs=[middle_output],
            name=f"{outer}_middle_if",
        )
        middle = ir.Graph(
            [],
            [middle_output],
            nodes=[middle_holder],
            name=f"{outer}_middle_branch",
            opset_imports={"": 23},
        )
        middle.meta[structure._OWNER] = object()
        branches.append(middle)
    output = _value("deep_if_output", ir.DataType.FLOAT, (2,))
    graph.append(
        ir.Node(
            "",
            "If",
            [outer_flag],
            {
                "then_branch": ir.AttrGraph("then_branch", branches[0]),
                "else_branch": ir.AttrGraph("else_branch", branches[1]),
            },
            outputs=[output],
            name="deep_outer_if",
        )
    )
    graph.outputs.append(output)
    return model, witnesses, tuple(leaves)


@pytest.mark.parametrize("recording", ("on", "off", "failure"))
def test_two_holder_capture_keeps_complete_global_edges_and_recording_parity(
    recording: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = common_passes.CommonSubexpressionEliminationPass
    calls = []

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            return original()(value)

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    baseline, baseline_witnesses, _ = _two_holder_capture_model()
    _full(baseline)
    _run(baseline)
    _resolve(baseline, baseline_witnesses)
    _full(baseline)
    expected_proto = ir.to_proto(baseline).SerializeToString(deterministic=True)
    model, witnesses, leaves = _two_holder_capture_model()
    original_outputs = tuple(
        (value.name, value.dtype, tuple(value.shape)) for value in model.graph.outputs
    )
    original_second_operands = tuple(node.inputs[1] for node in leaves)
    _full(model)
    if recording == "off":
        for graph in aliases._scope(model).graphs:
            graph.meta.pop(structure._OWNER, None)
    elif recording == "failure":

        def unavailable(*args: Any) -> Any:
            raise structure.StructureUnavailable(
                "optional nested recording unavailable"
            )

        monkeypatch.setattr(identity, "_current", unavailable)
    ir_optimizations._run_common_subexpression_elimination_pass(
        model, precision_policy=PrecisionPolicy.SOURCE
    )
    assert calls == [baseline, model]
    _full(model)
    assert ir.to_proto(model).SerializeToString(deterministic=True) == expected_proto
    assert (
        tuple(
            (value.name, value.dtype, tuple(value.shape))
            for value in model.graph.outputs
        )
        == original_outputs
    )
    assert aliases.slots(tuple(model.graph.outputs)) == (
        (0,),
        (1, 3, 5),
        (2, 4),
        (6,),
        (7,),
    )
    canonical = model.graph.outputs[0]
    assert all(node.inputs[0] is canonical for node in leaves)
    assert tuple(node.inputs[1] for node in leaves) == original_second_operands
    expected_edges = {
        (model.graph.node("original_internal"), 0),
        (model.graph.outputs[1].producer(), 0),
        (model.graph.outputs[2].producer(), 0),
        *((node, 0) for node in leaves),
    }
    assert len(expected_edges) == 7
    assert {(use.node, use.idx) for use in canonical.uses()} == expected_edges
    if recording == "on":
        _resolve(model, witnesses)
    else:
        assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)
    clone = model.clone()
    _full(clone)
    assert ir.to_proto(clone).SerializeToString(deterministic=True) == expected_proto


@pytest.mark.parametrize("fault", ("add", "drop", "port-swap"))
def test_unplanned_descendant_input_edges_reject_and_restore_all_raw_ports(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, leaves = _two_holder_capture_model()
    _full(model)
    original = common_passes.CommonSubexpressionEliminationPass
    verify = aliases._verify_normalized
    raw = []
    calls = []
    faults = []

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            result = original()(value)
            raw.append(aliases._physical(aliases._scope(value).graphs))
            return result

    def altered(*args: Any) -> None:
        node = leaves[0]
        canonical = model.graph.outputs[0]
        before_count = len(canonical.uses())
        if fault == "add":
            node.replace_input_with(1, canonical)
            assert len(canonical.uses()) == before_count + 1
        elif fault == "drop":
            node.replace_input_with(0, model.graph.inputs[1])
            assert len(canonical.uses()) == before_count - 1
        else:
            other = node.inputs[1]
            node.replace_input_with(0, other)
            node.replace_input_with(1, canonical)
            assert len(canonical.uses()) == before_count
        faults.append(True)
        verify(*args)

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    monkeypatch.setattr(aliases, "_verify_normalized", altered)
    with pytest.raises(
        structure.StructureMismatch,
        match="cse-output-alias-required-repair:cse-output-alias-normalized-value-fields-differ",
    ):
        ir_optimizations._run_common_subexpression_elimination_pass(
            model, precision_policy=PrecisionPolicy.SOURCE
        )
    assert calls == [model] and faults == [True] and len(raw) == 1
    assert aliases._physical(aliases._scope(model).graphs) == raw[0]
    assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)


@pytest.mark.parametrize("fault", ("none", "add", "drop", "port-swap"))
def test_external_global_edges_are_never_subtracted_or_silently_restored(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, leaves = _two_holder_capture_model()
    canonical = model.graph.outputs[0]
    external_output = _value("external_output", ir.DataType.FLOAT, (2,))
    external = ir.Node(
        "",
        "Sub",
        [canonical, model.graph.inputs[0]],
        outputs=[external_output],
        name="external_consumer",
    )
    external_graph = ir.Graph(
        [],
        [external_output],
        nodes=[external],
        name="external_retained_usage",
        opset_imports={"": 23},
    )
    scope = aliases._scope(model)
    assert external_graph not in scope.graphs
    for graph in scope.graphs:
        graph.meta.pop(structure._OWNER, None)
    _full(model)
    original = common_passes.CommonSubexpressionEliminationPass
    verify = aliases._verify_normalized
    raw = []
    calls = []
    faults = []

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            result = original()(value)
            raw.append(aliases._physical(aliases._scope(value).graphs))
            assert any(
                (external, 0) in saved.uses
                for local in raw[0]
                for saved in local.values
                if saved.value is canonical
            )
            assert all(external not in local.nodes for local in raw[0])
            return result

    def altered(*args: Any) -> None:
        before_count = len(canonical.uses())
        if fault == "add":
            external.replace_input_with(1, canonical)
            assert len(canonical.uses()) == before_count + 1
        elif fault == "drop":
            external.replace_input_with(0, model.graph.inputs[1])
            assert len(canonical.uses()) == before_count - 1
        elif fault == "port-swap":
            other = external.inputs[1]
            external.replace_input_with(0, other)
            external.replace_input_with(1, canonical)
            assert len(canonical.uses()) == before_count
        faults.append(True)
        verify(*args)

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    monkeypatch.setattr(aliases, "_verify_normalized", altered)
    if fault == "none":
        ir_optimizations._run_common_subexpression_elimination_pass(
            model, precision_policy=PrecisionPolicy.SOURCE
        )
        _full(model)
        expected = {
            (model.graph.node("original_internal"), 0),
            (model.graph.outputs[1].producer(), 0),
            (model.graph.outputs[2].producer(), 0),
            (external, 0),
            *((node, 0) for node in leaves),
        }
        assert len(expected) == 8
        assert {(use.node, use.idx) for use in canonical.uses()} == expected
    else:
        # This public fault lies outside captured/restorable ports. Rollback
        # must fail explicitly rather than claim full raw restoration.
        with pytest.raises(
            structure.StructureMismatch, match="cse-output-alias-rollback-failed"
        ) as caught:
            ir_optimizations._run_common_subexpression_elimination_pass(
                model, precision_policy=PrecisionPolicy.SOURCE
            )
        assert caught.value.__cause__ is not None
        assert "rollback-incomplete" in str(caught.value.__cause__)
        assert aliases._physical(aliases._scope(model).graphs) != raw[0]
        for local in raw[0]:
            assert (
                tuple(local.graph) == local.nodes
                and tuple(local.graph.outputs) == local.outputs
            )
            assert tuple(local.graph.inputs) == local.inputs
            assert tuple(local.graph.initializers.items()) == local.initializers
            for node, inputs, outputs in local.ports:
                assert tuple(node.inputs) == inputs and tuple(node.outputs) == outputs
        assert tuple(external.inputs) != (canonical, model.graph.inputs[0])
    assert calls == [model] and faults == [True] and len(raw) == 1
    assert not identity.receipts(model.graph) and not leaf.receipts(model.graph)


@pytest.mark.parametrize(
    "fault",
    (
        "duplicate-port",
        "conflicting-input",
        "conflicting-output",
        "conflicting-owner",
        "duplicate-graph",
        "value-owner",
        "value-uses",
        "value-producer",
        "value-schema",
        "value-constant",
        "raw-global-edge",
        "plan-canonical",
        "plan-survivor",
    ),
)
def test_frozen_global_raw_catalog_conflicts_reject_after_one_genuine_pass(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, witnesses, _ = _two_holder_capture_model()
    _full(model)
    original = common_passes.CommonSubexpressionEliminationPass
    capture = aliases.capture
    transactions = []
    calls = []

    class Original:
        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            return original()(value)

    def captured(*args: Any) -> Any:
        transaction = capture(*args)
        transactions.append(transaction)
        return transaction

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    monkeypatch.setattr(aliases, "capture", captured)
    _run(model)
    _full(model)
    _resolve(model, witnesses)
    assert calls == [model] and len(transactions) == 1
    transaction = transactions[0]
    assert transaction.finished and transaction.plans and transaction.raw
    raw = transaction.raw
    plans = transaction.plans
    parents = (identity.receipts(model.graph), leaf.receipts(model.graph))
    before = ir.to_proto(model).SerializeToString(deterministic=True)
    root_local = raw[0]
    first_node, first_inputs, first_outputs = root_local.ports[0]
    if fault in ("duplicate-port", "conflicting-input", "conflicting-output"):
        extra = root_local.ports[0]
        if fault == "conflicting-input":
            extra = (first_node, tuple(reversed(first_inputs)), first_outputs)
        elif fault == "conflicting-output":
            extra = (first_node, first_inputs, (model.graph.outputs[1],))
        raw = (replace(root_local, ports=(*root_local.ports, extra)), *raw[1:])
        cause = "raw-node-record-differs"
    elif fault == "conflicting-owner":
        changed = replace(root_local.node_fields[0], graph=raw[1].graph)
        raw = (
            replace(root_local, node_fields=(changed, *root_local.node_fields[1:])),
            *raw[1:],
        )
        cause = "raw-node-record-differs"
    elif fault == "duplicate-graph":
        raw = (*raw, root_local)
        cause = "raw-node-record-differs"
    elif fault in ("plan-canonical", "plan-survivor"):
        changed = replace(
            plans[0],
            **(
                {"canonical": model.graph.inputs[0]}
                if fault == "plan-canonical"
                else {"survivor": model.graph.outputs[0]}
            ),
        )
        plans = (changed, *plans[1:])
        cause = "raw-plan-binding-differs"
    else:
        canonical = model.graph.outputs[0]
        local_index = next(
            index
            for index, local in enumerate(raw)
            if index and any(saved.value is canonical for saved in local.values)
        )
        local = raw[local_index]
        value_index = next(
            index
            for index, saved in enumerate(local.values)
            if saved.value is canonical
        )
        saved = local.values[value_index]
        if fault == "value-owner":
            changed_value = replace(saved, graph=local.graph)
        elif fault == "value-uses":
            changed_value = replace(saved, uses=frozenset())
        elif fault == "value-producer":
            changed_value = replace(saved, producer=None)
        elif fault == "value-schema":
            changed_value = replace(saved, dtype=ir.DataType.DOUBLE)
        elif fault == "value-constant":
            changed_value = replace(
                saved, constant=ir.tensor(np.array([0.125, 0.125], dtype=np.float32))
            )
        else:
            removed_bridge = next(
                node
                for plan in plans
                for node in plan.bridges
                if node.outputs[0] is not plan.survivor
            )
            missing = (removed_bridge, 0)
            assert missing in saved.uses
            raw = tuple(
                replace(
                    item,
                    values=tuple(
                        (
                            replace(record, uses=record.uses - {missing})
                            if record.value is canonical
                            else record
                        )
                        for record in item.values
                    ),
                )
                for item in raw
            )
            changed_value = None
        if changed_value is not None:
            changed_local = replace(
                local,
                values=(
                    *local.values[:value_index],
                    changed_value,
                    *local.values[value_index + 1 :],
                ),
            )
            raw = (*raw[:local_index], changed_local, *raw[local_index + 1 :])
        cause = "raw-value-record-differs"
    # Adversarial frozen-snapshot verification only: no new pass, mutation or
    # receipt registration. The original genuine parent authority stays exact.
    with pytest.raises(structure.StructureMismatch, match="cse-output-alias-" + cause):
        aliases._verify_normalized(raw, plans, transaction.scope)
    assert calls == [model]
    assert ir.to_proto(model).SerializeToString(deterministic=True) == before
    assert (identity.receipts(model.graph), leaf.receipts(model.graph)) == parents
    _full(model)
