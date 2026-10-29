# tests/extra_tests/converter/test_source_leaf_cse.py

"""Real finite leaf CSE receipts preserve independent source obligations."""

from dataclasses import fields, is_dataclass, replace
from typing import Any
import weakref

import jax
import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx_ir.passes import common as common_passes

from jax2onnx import user_interface as ui
from jax2onnx.converter import _source_identity_cse as identity
from jax2onnx.converter import _source_leaf_cse as leaf
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


def _value(name: str, dtype: ir.DataType, shape: tuple[int, ...]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def _sub_model(
    dtype: ir.DataType = ir.DataType.FLOAT,
    shape: tuple[int, ...] = (2,),
    *,
    repeated: bool = False,
    owned: bool = True,
) -> tuple[ir.Model, tuple[Any, ...]]:
    x, y = _value("x", dtype, shape), _value("y", dtype, shape)
    first, second = _value("first", dtype, shape), _value("second", dtype, shape)
    result = _value("result", dtype, shape)
    nodes = [
        ir.Node("", "Sub", [x, y], outputs=[first], name="sub_first"),
        ir.Node("", "Sub", [x, y], outputs=[second], name="sub_second"),
        ir.Node("", "Sub", [second, y], outputs=[result], name="sub_result"),
    ]
    graph = ir.Graph(
        [x, y],
        [first, second, result, second] if repeated else [first, second, result],
        nodes=nodes,
        opset_imports={"": 23},
        name="leaf_cse_sub_fixture",
    )
    model = ir.Model(graph, ir_version=10)
    if not owned:
        return model, ()
    graph.meta[structure._OWNER] = object()
    return model, tuple(structure._witness(value) for value in (first, second))


def _constant_model(
    payload: np.ndarray[Any, Any],
    *,
    simultaneous: bool = False,
) -> tuple[ir.Model, tuple[Any, ...]]:
    dtype = ir.DataType.DOUBLE if payload.dtype == np.float64 else ir.DataType.FLOAT
    shape = tuple(payload.shape)
    x, y = _value("x", dtype, shape), _value("y", dtype, shape)
    constants = [_value("constant_0", dtype, shape), _value("constant_1", dtype, shape)]
    nodes = [
        ir.Node(
            "",
            "Constant",
            [],
            outputs=[value],
            attributes=[ir.AttrTensor("value", ir.tensor(payload.copy()))],
        )
        for value in constants
    ]
    outputs = []
    for index, (value, input_value) in enumerate(zip(constants, (x, y), strict=True)):
        result = _value(f"result_{index}", dtype, shape)
        nodes.append(ir.Node("", "Sub", [input_value, value], outputs=[result]))
        outputs.append(result)
    if simultaneous:
        for index in range(2):
            result = _value(f"alias_{index}", dtype, shape)
            nodes.append(ir.Node("", "Identity", [x], outputs=[result]))
            outputs.append(result)
    graph = ir.Graph(
        [x, y],
        outputs,
        nodes=nodes,
        opset_imports={"": 23},
        name="leaf_cse_constant_fixture",
    )
    graph.meta[structure._OWNER] = object()
    return ir.Model(graph, ir_version=10), tuple(
        structure._witness(value) for value in constants
    )


def _run(model: ir.Model) -> None:
    ir_optimizations._run_common_subexpression_elimination_pass(
        model, precision_policy=PrecisionPolicy.SOURCE
    )
    assert not model.graph.meta.get(leaf._GAPS), model.graph.meta


def _resolved(model: ir.Model, witnesses: tuple[Any, ...]) -> dict[Any, ir.Value]:
    leaf.validate_leaf_cse(model.graph, structure._resolve_bindings)
    return structure._resolve_bindings(model.graph, set(witnesses))


def _immutable(value: Any) -> None:
    if value is None or type(value) in (bool, int, str, bytes, object, ir.DataType):
        return
    if type(value) is tuple:
        for item in value:
            _immutable(item)
        return
    assert is_dataclass(value), type(value)
    assert type(value).__dataclass_params__.frozen
    assert not callable(value)
    for field in fields(value):
        _immutable(getattr(value, field.name))


@pytest.mark.parametrize(
    "dtype", (ir.DataType.FLOAT, ir.DataType.DOUBLE), ids=("FLOAT", "DOUBLE")
)
@pytest.mark.parametrize("shape", ((), (2,)), ids=("scalar", "2"))
@pytest.mark.parametrize("repeated", (False, True), ids=("False", "True"))
@pytest.mark.parametrize("clone_before", (False, True), ids=("False", "True"))
def test_real_sub_pass_preserves_ordered_inputs_internal_uses_and_output_slots(
    dtype: ir.DataType,
    shape: tuple[int, ...],
    repeated: bool,
    clone_before: bool,
) -> None:
    model, witnesses = _sub_model(dtype, shape, repeated=repeated)
    if clone_before:
        model = model.clone()
    _run(model)
    resolved = _resolved(model, witnesses)
    assert resolved[witnesses[0]] is resolved[witnesses[1]]
    canonical = resolved[witnesses[0]]
    assert canonical.producer().op_type == "Sub"
    assert tuple(canonical.producer().inputs) == tuple(model.graph.inputs)
    with pytest.raises(structure.StructureUnavailable):
        structure._resolve_direct_bindings(model.graph, {witnesses[1]})
    (receipt,) = leaf.receipts(model.graph)
    assert len(leaf._congruence(receipt)) == 1
    assert sum(output.bridge for output in receipt.outputs) == (2 if repeated else 1)
    assert model.graph.outputs[2].producer().inputs[0] is canonical
    _immutable(receipt)
    clone = model.clone()
    found = _resolved(clone, witnesses)
    assert found[witnesses[0]] is found[witnesses[1]]
    assert found[witnesses[0]] is not canonical
    assert leaf.receipts(clone.graph) == (receipt,)
    onnx.checker.check_model(ir.to_proto(clone), full_check=True)


@pytest.mark.parametrize("dtype", (np.float32, np.float64), ids=("float32", "float64"))
@pytest.mark.parametrize(
    "shape", ((), (0,), (2,), (10,)), ids=("scalar", "0", "2", "10")
)
@pytest.mark.parametrize("simultaneous", (False, True), ids=("False", "True"))
def test_real_bitexact_constant_merge_and_later_authentic_lift(
    dtype: Any,
    shape: tuple[int, ...],
    simultaneous: bool,
) -> None:
    payload = np.full(shape, -0.0, dtype=dtype)
    model, witnesses = _constant_model(payload, simultaneous=simultaneous)
    _run(model)
    before = _resolved(model, witnesses)
    assert before[witnesses[0]] is before[witnesses[1]]
    assert before[witnesses[0]].producer().op_type == "Constant"
    assert bool(identity.receipts(model.graph)) is simultaneous
    ir_optimizations._run_lift_constants_to_initializers_pass(model)
    found = _resolved(model, witnesses)
    assert found[witnesses[0]].is_initializer()
    assert found[witnesses[0]] is found[witnesses[1]]
    assert found[witnesses[0]].const_value.numpy().tobytes() == payload.tobytes()
    assert len(structure._constant_lift_receipts(model.graph)) == 1
    clone = model.clone()
    assert _resolved(clone, witnesses)[witnesses[0]].is_initializer()
    for receipt in leaf.receipts(clone.graph):
        _immutable(receipt)
    onnx.checker.check_model(ir.to_proto(clone), full_check=True)


@pytest.mark.parametrize(
    "difference",
    ("negative-zero", "nan-payload", "shape", "eleven"),
    ids=("negative-zero", "nan-payload", "shape", "eleven"),
)
def test_equal_names_or_numeric_values_never_authorize_unequal_constant_storage(
    difference: str,
) -> None:
    if difference == "eleven":
        payload = np.ones((11,), np.float32)
    else:
        payload = np.asarray([0x80000000, 0x7FC00001], np.uint32).view(np.float32)
    model, witnesses = _constant_model(payload)
    first, second = tuple(model.graph)[:2]
    if difference == "negative-zero":
        changed = np.asarray([0, 0x7FC00001], np.uint32).view(np.float32)
    elif difference == "nan-payload":
        changed = np.asarray([0x80000000, 0x7FC00002], np.uint32).view(np.float32)
    elif difference == "shape":
        changed = payload.reshape((1, 2))
        second.outputs[0].shape = ir.Shape((1, 2))
    else:
        changed = payload.copy()
    second.attributes["value"] = ir.AttrTensor("value", ir.tensor(changed))
    first.outputs[0].name = second.outputs[0].name = "same"
    _run(model)
    assert not leaf.receipts(model.graph)
    found = structure._resolve_bindings(model.graph, set(witnesses))
    assert found[witnesses[0]] is not found[witnesses[1]]


@pytest.mark.parametrize(
    "fault",
    (
        "absent",
        "forged",
        "foreign",
        "duplicate",
        "old-marker",
        "old-origin",
        "op",
        "domain",
        "overload",
        "attributes",
        "arity",
        "missing-input",
        "swap-inputs",
        "same-type-input",
        "dtype",
        "shape",
        "consumer-input",
        "consumer-output-type",
        "consumer-schema",
        "extra-use",
        "lost-use",
        "consumer-origin",
        "producer-order",
        "output-order",
        "output-count",
        "bridge-missing",
        "bridge-fake",
        "bridge-extra-user",
        "bridge-repeated-slot",
    ),
    ids=(
        "absent",
        "forged",
        "foreign",
        "duplicate",
        "old-marker",
        "old-origin",
        "op",
        "domain",
        "overload",
        "attributes",
        "arity",
        "missing-input",
        "swap-inputs",
        "same-type-input",
        "dtype",
        "shape",
        "consumer-input",
        "consumer-output-type",
        "consumer-schema",
        "extra-use",
        "lost-use",
        "consumer-origin",
        "producer-order",
        "output-order",
        "output-count",
        "bridge-missing",
        "bridge-fake",
        "bridge-extra-user",
        "bridge-repeated-slot",
    ),
)
def test_final_leaf_receipts_reject_changed_definition_use_interface_or_boundary(
    fault: str,
) -> None:
    model, witnesses = _sub_model(repeated=True)
    _run(model)
    found = _resolved(model, witnesses)
    graph = model.graph
    (receipt,) = leaf.receipts(graph)
    canonical = found[witnesses[0]]
    keeper = canonical.producer()
    consumer = graph.outputs[2].producer()
    bridge = graph.outputs[1].producer()
    if fault == "absent":
        graph.meta.pop(leaf._RECEIPTS)
    elif fault == "forged":
        graph.meta[leaf._RECEIPTS] = (replace(receipt),)
    elif fault == "foreign":
        foreign, _ = _sub_model()
        _run(foreign)
        graph.meta[leaf._RECEIPTS] = leaf.receipts(foreign.graph)
    elif fault == "duplicate":
        graph.meta[leaf._RECEIPTS] = (receipt, receipt)
    elif fault == "old-marker":
        canonical.meta[structure._VALUES] += (witnesses[1],)
    elif fault == "old-origin":
        output = _value("revived", canonical.dtype, tuple(canonical.shape))
        node = ir.Node("", "Sub", list(graph.inputs), outputs=[output])
        node.meta[structure._ORIGIN] = witnesses[1].origin
        graph.append(node)
    elif fault == "op":
        keeper.op_type = "Add"
    elif fault == "domain":
        keeper.domain = "foreign"
    elif fault == "overload":
        keeper.overload = "foreign"
    elif fault == "attributes":
        keeper.attributes["extra"] = ir.AttrInt64("extra", 1)
    elif fault == "arity":
        keeper.resize_inputs(3)
    elif fault == "missing-input":
        keeper.replace_input_with(0, None)
    elif fault == "swap-inputs":
        keeper.replace_input_with(0, graph.inputs[1])
        keeper.replace_input_with(1, graph.inputs[0])
    elif fault == "same-type-input":
        keeper.replace_input_with(0, graph.inputs[1])
    elif fault == "dtype":
        canonical.type = ir.TensorType(ir.DataType.DOUBLE)
    elif fault == "shape":
        canonical.shape = ir.Shape((1, 2))
    elif fault == "consumer-input":
        consumer.replace_input_with(1, graph.inputs[0])
    elif fault == "consumer-output-type":
        consumer.outputs[0].type = ir.TensorType(ir.DataType.DOUBLE)
    elif fault == "consumer-schema":
        consumer.op_type = "Add"
    elif fault == "extra-use":
        graph.append(
            ir.Node(
                "",
                "Neg",
                [canonical],
                outputs=[_value("extra", canonical.dtype, tuple(canonical.shape))],
            )
        )
    elif fault == "lost-use":
        consumer.replace_input_with(0, graph.inputs[0])
    elif fault == "consumer-origin":
        consumer.meta[structure._ORIGIN] = object()
    elif fault == "producer-order":
        graph.insert_after(consumer, keeper)
    elif fault == "output-order":
        graph.outputs[1], graph.outputs[2] = graph.outputs[2], graph.outputs[1]
    elif fault == "output-count":
        graph.outputs.append(canonical)
    elif fault == "bridge-missing":
        graph.outputs[1] = canonical
        graph.outputs[3] = canonical
        graph.remove(bridge, safe=True)
    elif fault == "bridge-fake":
        output = _value("fake", canonical.dtype, tuple(canonical.shape))
        replacement = ir.Node("", "Identity", [canonical], outputs=[output])
        graph.insert_before(bridge, replacement)
        graph.outputs[1] = output
        graph.outputs[3] = output
        graph.remove(bridge, safe=True)
    elif fault == "bridge-extra-user":
        graph.append(
            ir.Node(
                "",
                "Neg",
                [graph.outputs[1]],
                outputs=[_value("extra", canonical.dtype, tuple(canonical.shape))],
            )
        )
    else:
        graph.outputs[2] = graph.outputs[1]
    with pytest.raises((structure.StructureUnavailable, structure.StructureMismatch)):
        _resolved(model, witnesses)


@pytest.mark.parametrize(
    "fault",
    ("bits", "dtype", "shape", "lift-absent", "lift-forged", "lift-reversed"),
    ids=("bits", "dtype", "shape", "lift-absent", "lift-forged", "lift-reversed"),
)
def test_final_constant_rebinding_requires_exact_storage_and_real_forward_lift(
    fault: str,
) -> None:
    payload = np.asarray([0x80000000, 0x7FC00001], np.uint32).view(np.float32)
    model, witnesses = _constant_model(payload)
    _run(model)
    ir_optimizations._run_lift_constants_to_initializers_pass(model)
    value = _resolved(model, witnesses)[witnesses[0]]
    (lift,) = structure._constant_lift_receipts(model.graph)
    if fault == "bits":
        value.const_value = ir.tensor(
            np.asarray([0, 0x7FC00001], np.uint32).view(np.float32)
        )
    elif fault == "dtype":
        value.type = ir.TensorType(ir.DataType.DOUBLE)
    elif fault == "shape":
        value.shape = ir.Shape((1, 2))
    elif fault == "lift-absent":
        model.graph.meta.pop(structure._CONSTANT_LIFTS)
    elif fault == "lift-forged":
        model.graph.meta[structure._CONSTANT_LIFTS] = (replace(lift),)
    else:
        model.graph.meta[structure._CONSTANT_LIFTS] = (
            replace(lift, before=lift.after, after=lift.before),
        )
    with pytest.raises((structure.StructureUnavailable, structure.StructureMismatch)):
        _resolved(model, witnesses)


@pytest.mark.parametrize(
    "fault",
    (
        "extra-use",
        "lost-use",
        "schema",
        "dtype",
        "unexplained-op",
        "revival",
        "output-slot",
    ),
    ids=(
        "extra-use",
        "lost-use",
        "schema",
        "dtype",
        "unexplained-op",
        "revival",
        "output-slot",
    ),
)
def test_incomplete_actual_pass_never_mints_a_leaf_receipt(fault: str) -> None:
    model, _ = _sub_model()
    with leaf.observe_leaf_cse(model):
        with identity.observe_identity_cse(model):
            common_passes.CommonSubexpressionEliminationPass()(model)
        graph = model.graph
        canonical = graph.outputs[0]
        consumer = graph.outputs[2].producer()
        if fault == "extra-use":
            graph.append(
                ir.Node(
                    "",
                    "Neg",
                    [canonical],
                    outputs=[_value("extra", canonical.dtype, tuple(canonical.shape))],
                )
            )
        elif fault == "lost-use":
            consumer.replace_input_with(0, graph.inputs[0])
        elif fault == "schema":
            consumer.op_type = "Add"
        elif fault == "dtype":
            consumer.outputs[0].type = ir.TensorType(ir.DataType.DOUBLE)
        elif fault == "unexplained-op":
            graph.remove(consumer, safe=False)
        elif fault == "revival":
            graph.append(
                ir.Node(
                    "",
                    "Sub",
                    list(graph.inputs),
                    outputs=[_value("extra", canonical.dtype, tuple(canonical.shape))],
                )
            )
        else:
            graph.outputs[1] = graph.inputs[0]
    assert not leaf.receipts(model.graph)
    assert model.graph.meta.get(leaf._GAPS)


def test_genuine_opposite_order_clone_leaf_receipts_fail_before_resolution_callback() -> (
    None
):
    original, witnesses = _sub_model()
    left, right = original.clone(), original.clone()
    first, second, _ = tuple(right.graph)
    right.graph.insert_before(first, second)
    _run(left)
    _run(right)
    (left_receipt,) = leaf.receipts(left.graph)
    (right_receipt,) = leaf.receipts(right.graph)
    assert leaf._congruence(left_receipt) == {witnesses[1]: witnesses[0]}
    assert leaf._congruence(right_receipt) == {witnesses[0]: witnesses[1]}
    left.graph.meta[leaf._RECEIPTS] = (left_receipt, right_receipt)
    calls = []
    with pytest.raises(structure.StructureUnavailable, match="cycle"):
        leaf.validate_leaf_cse(left.graph, lambda *args: calls.append(args))
    assert not calls


def test_genuine_cross_catalog_clone_cycle_fails_before_resolution_callback() -> None:
    x, y = _value("x", ir.DataType.FLOAT, (2,)), _value("y", ir.DataType.FLOAT, (2,))
    a, b, s, t = (
        _value(name, ir.DataType.FLOAT, (2,)) for name in ("a", "b", "s", "t")
    )
    nodes = [
        ir.Node("", "Identity", [x], outputs=[a]),
        ir.Node("", "Identity", [x], outputs=[b]),
        ir.Node("", "Sub", [x, y], outputs=[s]),
        ir.Node("", "Sub", [x, y], outputs=[t]),
    ]
    graph = ir.Graph(
        [x, y],
        [a, b, s, t],
        nodes=nodes,
        opset_imports={"": 23},
        name="leaf_cse_cross_catalog_cycle_fixture",
    )
    graph.meta[structure._OWNER] = object()
    for value in (a, b, s, t):
        structure._witness(value)
    original = ir.Model(graph, ir_version=10)
    left, right = original.clone(), original.clone()
    ra, rb, rs, rt = tuple(right.graph)
    rt.replace_input_with(0, right.graph.inputs[0])
    rs.replace_input_with(0, right.graph.inputs[1])
    rs.replace_input_with(1, right.graph.inputs[0])
    right.graph.insert_before(ra, rt)
    right.graph.insert_before(ra, rb)
    ra.replace_input_with(0, rt.outputs[0])
    rb.replace_input_with(0, rt.outputs[0])
    _run(left)
    _run(right)
    assert leaf.receipts(left.graph) and not leaf.receipts(right.graph)
    assert identity.receipts(left.graph) and identity.receipts(right.graph)
    left.graph.meta[identity._RECEIPTS] = identity.receipts(right.graph)
    calls = []
    with pytest.raises(structure.StructureUnavailable, match="cycle"):
        leaf.validate_leaf_cse(left.graph, lambda *args: calls.append(args))
    assert not calls


@pytest.mark.parametrize(
    "stage",
    ("discovery", "capture", "finish", "noop", "exception"),
    ids=("discovery", "capture", "finish", "noop", "exception"),
)
def test_observer_failures_and_noop_preserve_the_single_original_pass(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _ = _sub_model()
    original = common_passes.CommonSubexpressionEliminationPass
    sentinel = RuntimeError("original pass failure")
    calls = []

    class Original:
        def __init__(self) -> None:
            self.actual = original()

        def __call__(self, value: ir.Model) -> Any:
            calls.append(value)
            if stage == "exception":
                raise sentinel
            if stage == "noop":
                return None
            return self.actual(value)

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise structure.StructureUnavailable(
            "deliberately unavailable leaf observation"
        )

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    if stage == "discovery":
        monkeypatch.setattr(identity, "_graphs", unavailable)
    elif stage in ("capture", "finish"):
        monkeypatch.setattr(
            leaf, "_capture" if stage == "capture" else "_finish", unavailable
        )
    if stage == "exception":
        with pytest.raises(RuntimeError) as caught:
            ir_optimizations._run_common_subexpression_elimination_pass(model)
        assert caught.value is sentinel
    else:
        ir_optimizations._run_common_subexpression_elimination_pass(model)
    assert calls == [model]
    assert not leaf.receipts(model.graph)
    assert bool(model.graph.meta.get(leaf._GAPS)) is (
        stage in ("discovery", "capture", "finish")
    )


def test_transient_snapshots_do_not_retain_native_ir_after_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = leaf._capture
    snapshots = []

    def capture(graph: ir.Graph) -> Any:
        result = original(graph)
        if result is not None:
            snapshots.append(weakref.ref(result))
        return result

    monkeypatch.setattr(leaf, "_capture", capture)
    model, _ = _sub_model()
    _run(model)
    assert snapshots and all(reference() is None for reference in snapshots)
    for receipt in leaf.receipts(model.graph):
        _immutable(receipt)


def _source(function: Any, dtype: Any = np.float32) -> ir.Model:
    with jax.enable_x64(True):
        model = ui._to_onnx(
            function,
            [jax.ShapeDtypeStruct((2,), dtype)] * 2,
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
            opset=23,
        )
    assert isinstance(model, ir.Model)
    return model


def _repeated(left: Any, right: Any) -> Any:
    return left - right, left, left - right


@pytest.mark.parametrize(
    "kind,expected,dtype",
    (
        ("repeated", 3, np.float32),
        ("repeated", 3, np.float64),
        ("closed", 7, np.float32),
        ("rich", 17, np.float32),
    ),
    ids=(
        "repeated-3-float32",
        "repeated-3-float64",
        "closed-7-float32",
        "rich-17-float32",
    ),
)
@pytest.mark.parametrize("optimization", ("disabled", "all"), ids=("disabled", "all"))
def test_original_repeated_closed_and_rich_counts_and_bitwise_samples_remain(
    kind: str, expected: int, dtype: Any, optimization: str
) -> None:
    from tests.extra_tests.converter.test_source_literal_if_inline_structure import (
        _target,
    )

    function = jax.jit(_repeated) if kind == "repeated" else _target(1, kind)
    model = _source(function, dtype)
    proof = structure.validate_source_structure(model)
    assert proof.status == "validated", proof
    assert proof.expected_equations == proof.checked_equations == expected
    assert leaf.receipts(model.graph)
    inputs = (np.asarray([0.5, -2.0], dtype), np.asarray([0.25, 1.0], dtype))
    with jax.enable_x64(True):
        values = function(*inputs)
    reference = [np.asarray(value) for value in values]
    for candidate in (model, ir.from_proto(ir.to_proto(model))):
        report = validate_source_samples(
            candidate,
            samples=[SourceSample(inputs, reference, SampleCriteria(mode="bitwise"))],
            optimization=optimization,
        )
        report.require_sample_acceptance()
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize(
    "kind", ("repeated", "closed", "rich"), ids=("repeated", "closed", "rich")
)
def test_ledger_on_off_retains_proto_report_trace_count_and_single_original_pass(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.extra_tests.converter.test_source_literal_if_inline_structure import (
        _target,
    )

    function = jax.jit(_repeated) if kind == "repeated" else _target(1, kind)
    source_calls = []

    def observed(left: Any, right: Any) -> Any:
        source_calls.append(1)
        return function(left, right)

    original = common_passes.CommonSubexpressionEliminationPass
    pass_calls = []

    class Counted:
        def __init__(self) -> None:
            self.actual = original()

        def __call__(self, model: ir.Model) -> Any:
            pass_calls.append(model)
            return self.actual(model)

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Counted)
    enabled = _source(observed)
    enabled_calls = tuple(source_calls)
    source_calls.clear()
    with monkeypatch.context() as change:
        change.setattr(ir_context, "SourceStructureLedger", lambda: None)
        disabled = _source(observed)
    assert source_calls and tuple(source_calls) == enabled_calls
    assert pass_calls == [enabled, disabled]
    assert structure._LEDGER not in disabled.meta
    assert not leaf.receipts(disabled.graph)
    assert ir.to_proto(enabled).SerializeToString(deterministic=True) == ir.to_proto(
        disabled
    ).SerializeToString(deterministic=True)
    first, second = get_precision_report(enabled), get_precision_report(disabled)
    assert first is not None and first == second
    assert precision_report_to_json(first) == precision_report_to_json(second)


@pytest.mark.parametrize("extra", ("node", "storage"), ids=("node", "storage"))
def test_receipts_leave_whole_artifact_coverage_and_source_denominator_intact(
    extra: str,
) -> None:
    model = _source(jax.jit(_repeated))
    before = structure.validate_source_structure(model)
    assert before.status == "validated" and before.expected_equations == 3
    if extra == "node":
        model.graph.append(
            ir.Node(
                "",
                "Neg",
                [model.graph.inputs[0]],
                outputs=[_value("extra", ir.DataType.FLOAT, (2,))],
            )
        )
    else:
        model.graph.register_initializer(
            ir.Value(name="extra", const_value=ir.tensor(np.ones((2,), np.float32)))
        )
    proof = structure.validate_source_structure(model)
    assert proof.status == "unverified" and proof.expected_equations == 3
    assert any("artifact-" in reason for reason in proof.reasons)


def test_final_validation_never_retraces_or_emits_or_mutates_ir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _source(jax.jit(_repeated))
    before = ir.to_proto(model).SerializeToString(deterministic=True)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("leaf receipt validation retraced or emitted IR")

    monkeypatch.setattr(jax, "make_jaxpr", forbidden)
    monkeypatch.setattr(jax, "eval_shape", forbidden)
    monkeypatch.setattr(ir_context.IRContext, "require_value_for_var", forbidden)
    proof = structure.validate_source_structure(model)
    assert (
        proof.status == "validated"
        and proof.expected_equations == proof.checked_equations == 3
    )
    assert ir.to_proto(model).SerializeToString(deterministic=True) == before


def test_genuine_later_identity_dce_cannot_delete_a_leaf_receipt_consumer() -> None:
    from jax2onnx.converter import _source_identity_dce as dce

    model, witnesses = _sub_model()
    canonical = model.graph.outputs[0]
    unused = _value("unused", canonical.dtype, tuple(canonical.shape))
    model.graph.append(ir.Node("", "Identity", [canonical], outputs=[unused]))
    structure._witness(unused)
    _run(model)
    assert _resolved(model, witnesses)
    ir_optimizations.remove_dead_nodes_ir(model)
    assert dce.receipts(model.graph)
    calls = []
    with pytest.raises(structure.StructureUnavailable, match="dce-overlap"):
        leaf.validate_leaf_cse(model.graph, lambda *args: calls.append(args))
    assert not calls


@pytest.mark.parametrize(
    "fault",
    ("op", "domain", "overload", "attributes", "dtype"),
    ids=("op", "domain", "overload", "attributes", "dtype"),
)
def test_outside_the_finite_profile_supplies_no_leaf_substitution_authority(
    fault: str,
) -> None:
    model, witnesses = _sub_model()
    first, second, _ = tuple(model.graph)
    for node in (first, second):
        if fault == "op":
            node.op_type = "Add"
        elif fault == "domain":
            node.domain = "foreign"
        elif fault == "overload":
            node.overload = "foreign"
        elif fault == "attributes":
            node.attributes["extra"] = ir.AttrInt64("extra", 1)
        else:
            node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT16)
    _run(model)
    assert not leaf.receipts(model.graph)
    with pytest.raises(structure.StructureUnavailable):
        structure._resolve_bindings(model.graph, set(witnesses))


def test_unowned_graphs_leave_no_observer_receipt_or_gap() -> None:
    model, _ = _sub_model(owned=False)
    before = ir.to_proto(model)
    _run(model)
    assert not leaf.receipts(model.graph)
    plain = ir.from_proto(before)
    common_passes.CommonSubexpressionEliminationPass()(plain)
    assert ir.to_proto(model).SerializeToString(deterministic=True) == ir.to_proto(
        plain
    ).SerializeToString(deterministic=True)


@pytest.mark.parametrize(
    "fault",
    ("negative-zero", "nan-payload", "shape", "declared-dtype"),
    ids=("negative-zero", "nan-payload", "shape", "declared-dtype"),
)
def test_final_constant_producer_uses_live_attribute_storage(fault: str) -> None:
    payload = np.asarray([0x80000000, 0x7FC00001], np.uint32).view(np.float32)
    model, witnesses = _constant_model(payload)
    _run(model)
    constant = _resolved(model, witnesses)[witnesses[0]]
    node = constant.producer()
    if fault == "negative-zero":
        array = np.asarray([0, 0x7FC00001], np.uint32).view(np.float32)
    elif fault == "nan-payload":
        array = np.asarray([0x80000000, 0x7FC00002], np.uint32).view(np.float32)
    elif fault == "shape":
        array = payload.reshape((1, 2))
    else:
        array = payload.astype(np.float64)
    node.attributes["value"] = ir.AttrTensor("value", ir.tensor(array))
    with pytest.raises((structure.StructureUnavailable, structure.StructureMismatch)):
        _resolved(model, witnesses)


@pytest.mark.parametrize(
    "stage", ("discovery", "capture"), ids=("discovery", "capture")
)
def test_before_observation_failure_preserves_original_pass_exception(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _ = _sub_model()
    sentinel = RuntimeError("the original pass exception")
    calls = []

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise structure.StructureUnavailable("observation failed first")

    class Original:
        def __call__(self, actual: ir.Model) -> Any:
            calls.append(actual)
            raise sentinel

    monkeypatch.setattr(common_passes, "CommonSubexpressionEliminationPass", Original)
    monkeypatch.setattr(
        identity if stage == "discovery" else leaf,
        "_graphs" if stage == "discovery" else "_capture",
        unavailable,
    )
    with pytest.raises(RuntimeError) as caught:
        ir_optimizations._run_common_subexpression_elimination_pass(model)
    assert caught.value is sentinel and calls == [model]
    assert model.graph.meta.get(leaf._GAPS)
    assert not leaf.receipts(model.graph)


@pytest.mark.parametrize("dtype", (np.float32, np.float64), ids=("float32", "float64"))
def test_sub_precongruence_uses_only_already_established_typed_leaf_classes(
    dtype: Any,
) -> None:
    model, constants = _constant_model(np.asarray([-0.0, 0.5], dtype))
    graph = model.graph
    consumers = [node for node in graph if node.op_type == "Sub"]
    consumers[1].replace_input_with(0, graph.inputs[0])
    outputs = tuple(structure._witness(node.outputs[0]) for node in consumers)
    _run(model)
    found = _resolved(model, (*constants, *outputs))
    assert found[constants[0]] is found[constants[1]]
    assert found[outputs[0]] is found[outputs[1]]
    (receipt,) = leaf.receipts(graph)
    assert len(leaf._congruence(receipt)) == 2
    ir_optimizations._run_lift_constants_to_initializers_pass(model)
    found = _resolved(model.clone(), (*constants, *outputs))
    assert found[constants[0]].is_initializer()
    assert found[outputs[0]].producer().inputs[1] is found[constants[0]]


def test_genuine_same_order_clone_receipts_cannot_duplicate_an_alias_domain() -> None:
    original, _ = _sub_model()
    left, right = original.clone(), original.clone()
    _run(left)
    _run(right)
    (first,) = leaf.receipts(left.graph)
    (second,) = leaf.receipts(right.graph)
    assert first is not second and leaf._congruence(first) == leaf._congruence(second)
    left.graph.meta[leaf._RECEIPTS] = (first, second)
    calls = []
    with pytest.raises(structure.StructureUnavailable, match="alias-domain-conflict"):
        leaf.validate_leaf_cse(left.graph, lambda *args: calls.append(args))
    assert not calls


def _nested_owner_model(
    *, extra_duplicates: bool = False
) -> tuple[ir.Model, dict[str, Any]]:
    """Build actual lexical captures with distinct root/child owner tokens."""
    a, b = _value("a", ir.DataType.FLOAT, (2,)), _value("b", ir.DataType.FLOAT, (2,))
    condition = _value("condition", ir.DataType.BOOL, ())
    x, y, z = (_value(name, ir.DataType.FLOAT, (2,)) for name in ("X", "Y", "Z"))
    c1, c2 = (_value(name, ir.DataType.FLOAT, (2,)) for name in ("C1", "C2"))
    child_nodes = [
        ir.Node("", "Sub", [x, b], outputs=[c1]),
        ir.Node("", "Sub", [a, b], outputs=[c2]),
    ]
    child_outputs = [c1, c2]
    if extra_duplicates:
        for name in ("C3", "C4"):
            value = _value(name, ir.DataType.FLOAT, (2,))
            child_nodes.append(ir.Node("", "Sub", [a, b], outputs=[value]))
            child_outputs.append(value)
    then = ir.Graph(
        [],
        child_outputs,
        nodes=child_nodes,
        opset_imports={"": 23},
        name="leaf_cse_nested_then_fixture",
    )
    else_values = [
        _value(f"E{index}", ir.DataType.FLOAT, (2,))
        for index in range(len(child_outputs))
    ]
    else_operands = ((a, b), (b, a), (a, a), (b, b))
    otherwise = ir.Graph(
        [],
        else_values,
        nodes=[
            ir.Node("", "Sub", list(operands), outputs=[value])
            for operands, value in zip(
                else_operands[: len(else_values)], else_values, strict=True
            )
        ],
        opset_imports={"": 23},
        name="leaf_cse_nested_else_fixture",
    )
    selected = [
        _value(f"selected_{index}", ir.DataType.FLOAT, (2,))
        for index in range(len(child_outputs))
    ]
    branch = ir.Node(
        "",
        "If",
        [condition],
        outputs=selected,
        attributes=[
            ir.AttrGraph("then_branch", then),
            ir.AttrGraph("else_branch", otherwise),
        ],
    )
    root = ir.Graph(
        [a, b, condition],
        [x, z, *selected],
        nodes=[
            ir.Node("", "Sub", [a, b], outputs=[x]),
            ir.Node("", "Sub", [a, b], outputs=[y]),
            ir.Node("", "Mul", [y, b], outputs=[z]),
            branch,
        ],
        opset_imports={"": 23},
        name="leaf_cse_nested_root_fixture",
    )
    for graph in (root, then, otherwise):
        graph.meta[structure._OWNER] = object()
    witnesses = {
        value.name: structure._witness(value)
        for value in (a, b, condition, x, y, z, *child_outputs, *else_values, *selected)
    }
    return ir.Model(root, ir_version=10), witnesses


def _attached_then(model: ir.Model) -> ir.Graph:
    return (
        next(node for node in model.graph if node.op_type == "If")
        .attributes["then_branch"]
        .as_graph()
    )


def _prepare_genuine_child_receipt(model: ir.Model) -> ir.Graph:
    """The independently reviewed legal clone-B construction, before real CSE."""
    root_nodes = tuple(model.graph)
    x, y = root_nodes[:2]
    x.replace_input_with(1, model.graph.inputs[0])
    child = _attached_then(model)
    c1, c2 = tuple(child)[:2]
    for node in (c1, c2):
        node.replace_input_with(0, y.outputs[0])
        node.replace_input_with(1, model.graph.inputs[1])
    child.insert_before(c1, c2)
    return child


@pytest.mark.parametrize("owner", ("root", "child"), ids=("root", "child"))
@pytest.mark.parametrize(
    "entry",
    ("validate", "resolve", "internal"),
    ids=("validate", "resolve", "internal"),
)
def test_genuine_parent_child_clone_cycle_rejects_before_every_leaf_callback(
    owner: str, entry: str
) -> None:
    original, witnesses = _nested_owner_model()
    left, right = original.clone(), original.clone()
    _run(left)
    (parent,) = leaf.receipts(left.graph)
    assert leaf._congruence(parent) == {witnesses["Y"]: witnesses["X"]}
    child_graph = _prepare_genuine_child_receipt(right)
    _run(ir.Model(child_graph, ir_version=10))
    (child,) = leaf.receipts(child_graph)
    assert leaf._congruence(child) == {witnesses["C1"]: witnesses["C2"]}
    assert parent in leaf._REGISTERED and child in leaf._REGISTERED
    assert parent.owner is right.graph.meta[structure._OWNER]
    assert child.owner is child_graph.meta[structure._OWNER]
    assert parent.owner is not child.owner
    right.graph.meta[leaf._RECEIPTS] = (parent,)
    # These genuine local catalogs are individually acyclic. Only their actual
    # parent/child dependency closure exposes Y -> C1 -> Y before callbacks.
    assert leaf._check_catalog(right.graph, leaf.receipts(right.graph))
    assert leaf._check_catalog(child_graph, leaf.receipts(child_graph))
    graph, receipt, alias = (
        (right.graph, parent, witnesses["Y"])
        if owner == "root"
        else (child_graph, child, witnesses["C1"])
    )
    calls = []
    with pytest.raises(structure.StructureUnavailable, match="cross-catalog-cycle"):
        if entry == "validate":
            leaf.validate_leaf_cse(graph, lambda *args: calls.append(args))
        elif entry == "resolve":
            leaf.resolve_leaf_bindings(
                graph, {alias}, frozenset(), lambda *args: calls.append(args)
            )
        else:
            leaf._validate(graph, receipt, lambda *args: calls.append(args))
    assert not calls


@pytest.mark.parametrize(
    "mode", ("root", "child", "both-disjoint"), ids=("root", "child", "both-disjoint")
)
@pytest.mark.parametrize("clone_after", (False, True), ids=("direct", "clone"))
def test_acyclic_parent_child_captures_keep_current_authentic_receipts(
    mode: str, clone_after: bool
) -> None:
    model, witnesses = _nested_owner_model(extra_duplicates=mode == "both-disjoint")
    if mode == "child":
        child = _prepare_genuine_child_receipt(model)
        _run(ir.Model(child, ir_version=10))
        assert not leaf.receipts(model.graph)
    else:
        _run(model)
        assert leaf.receipts(model.graph)
        child = _attached_then(model)
        if mode == "both-disjoint":
            _run(ir.Model(child, ir_version=10))
            assert leaf.receipts(child)
    if clone_after:
        model = model.clone()
        child = _attached_then(model)
    # Starting at either owner discovers the same valid lexical owner closure.
    leaf.validate_leaf_catalog(model.graph)
    leaf.validate_leaf_catalog(child)
    leaf.validate_leaf_cse(model.graph, structure._resolve_bindings)
    leaf.validate_leaf_cse(child, structure._resolve_bindings)
    if mode == "child":
        found = structure._resolve_bindings(child, {witnesses["C1"], witnesses["C2"]})
        assert found[witnesses["C1"]] is found[witnesses["C2"]]
    else:
        found = structure._resolve_bindings(
            model.graph, {witnesses["X"], witnesses["Y"]}
        )
        assert found[witnesses["X"]] is found[witnesses["Y"]]
        if mode == "both-disjoint":
            found_child = structure._resolve_bindings(
                child, {witnesses["C2"], witnesses["C3"], witnesses["C4"]}
            )
            assert len(set(found_child.values())) == 1
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize(
    "fault",
    ("missing-owner", "duplicate-owner"),
    ids=("missing-owner", "duplicate-owner"),
)
def test_global_current_owner_binding_rejects_ambiguity_before_callback(
    fault: str,
) -> None:
    model, _ = _nested_owner_model()
    _run(model)
    child = _attached_then(model)
    if fault == "missing-owner":
        child.meta.pop(structure._OWNER)
    else:
        foreign = model.clone()
        next(iter(child)).replace_input_with(0, foreign.graph.inputs[0])
    calls = []
    with pytest.raises(
        structure.StructureUnavailable, match="catalog-owner-unavailable"
    ):
        leaf.validate_leaf_cse(model.graph, lambda *args: calls.append(args))
    assert not calls


def test_original_connected_identity_leaf_consumer_deletion_remains_unverified() -> (
    None
):
    # Joint receipt publication for this exact original connected graph remains
    # an open composition obligation. Its valid ONNX graph is not proof authority.
    x, y = _value("x", ir.DataType.FLOAT, (2,)), _value("y", ir.DataType.FLOAT, (2,))
    a, b, s, t = (
        _value(name, ir.DataType.FLOAT, (2,)) for name in ("a", "b", "s", "t")
    )
    nodes = [
        ir.Node("", "Identity", [x], outputs=[a]),
        ir.Node("", "Identity", [x], outputs=[b]),
        ir.Node("", "Sub", [a, y], outputs=[s]),
        ir.Node("", "Sub", [a, y], outputs=[t]),
    ]
    graph = ir.Graph(
        [x, y],
        [a, b, s, t],
        nodes=nodes,
        opset_imports={"": 23},
        name="leaf_cse_original_connected_composition_gap",
    )
    graph.meta[structure._OWNER] = object()
    for value in (a, b, s, t):
        structure._witness(value)
    model = ir.Model(graph, ir_version=10)
    assert tuple(graph.inputs) == (x, y)
    assert tuple(graph.outputs) == (a, b, s, t)
    assert tuple(tuple(node.inputs) for node in graph) == ((x,), (x,), (a, y), (a, y))
    onnx.checker.check_model(ir.to_proto(model), full_check=True)

    # Call the unchanged real observer-wrapped pass directly once. Its recorded
    # gaps must remain unavailable observations rather than accepted receipts.
    ir_optimizations._run_common_subexpression_elimination_pass(model)
    assert graph.meta.get(identity._GAPS) == (
        "identity-cse-after:identity-cse-projected-consumer-differs",
    )
    assert graph.meta.get(leaf._GAPS) == (
        "leaf-cse-after:leaf-cse-unexplained-definition-removal",
    )
    assert not identity.receipts(graph)
    assert not leaf.receipts(graph)
    calls: list[Any] = []
    with pytest.raises(
        structure.StructureUnavailable, match="identity-cse-projected-consumer-differs"
    ):
        identity.validate_identity_cse(graph, lambda *args: calls.append(args))
    with pytest.raises(
        structure.StructureUnavailable, match="leaf-cse-unexplained-definition-removal"
    ):
        leaf.validate_leaf_cse(graph, lambda *args: calls.append(args))
    assert calls == []
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def test_catalog_discovery_ignores_unrelated_detached_operand_without_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def target(flag: Any, left: Any, right: Any) -> Any:
        return (
            jax.lax.sub(left, right),
            jax.lax.sub(left, right),
            jax.lax.cond(flag, jax.lax.sub, jax.lax.add, left, right),
        )

    with jax.enable_x64(True):
        model = ui._to_onnx(
            target,
            [
                jax.ShapeDtypeStruct((), np.bool_),
                jax.ShapeDtypeStruct((2,), np.float32),
                jax.ShapeDtypeStruct((2,), np.float32),
            ],
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
            opset=23,
        )
    assert isinstance(model, ir.Model)
    before = structure.validate_source_structure(model)
    assert before.status == "validated", before
    assert before.expected_equations == before.checked_equations > 0
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    graphs = identity._graphs(model.graph)
    assert len(graphs) == 3
    catalogs = {graph: leaf.receipts(graph) for graph in graphs}
    assert catalogs[model.graph]
    assert all(receipt in leaf._REGISTERED for receipt in catalogs[model.graph])
    registered = set(leaf._REGISTERED)
    branch = next(node for node in model.graph if node.op_type == "If")
    child = branch.attributes["then_branch"].as_graph()
    detached = _value("unrelated_detached", ir.DataType.FLOAT, (2,))
    unrelated = ir.Node(
        "",
        "Neg",
        [detached],
        outputs=[_value("unrelated_result", ir.DataType.FLOAT, (2,))],
    )
    # Discover an unowned operand before the real ancestor capture operands.
    child.insert_before(next(iter(child)), unrelated)
    assert detached.graph is None and not detached.meta
    expected_owners = {graph.meta[structure._OWNER]: graph for graph in graphs}
    calls: list[Any] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        pytest.fail("catalog discovery resolved or stamped a value")

    with monkeypatch.context() as change:
        change.setattr(structure, "_resolve_bindings", forbidden)
        change.setattr(structure, "_resolve_direct_bindings", forbidden)
        change.setattr(structure, "_witness", forbidden)
        change.setattr(leaf, "_current", forbidden)
        assert leaf._catalog_graphs(child) == expected_owners
        leaf.validate_leaf_catalog(child)
        leaf.validate_leaf_catalog(model.graph)
    assert not calls
    assert {graph: leaf.receipts(graph) for graph in graphs} == catalogs
    assert set(leaf._REGISTERED) == registered
    assert not detached.meta
    # Catalog discovery leaves this unrelated corruption to the full SOURCE
    # audit. The authentic leaf receipt cannot cover an extra artifact node.
    after = structure.validate_source_structure(model)
    assert after.status == "unverified", after
    assert after.expected_equations == before.expected_equations
    assert "artifact-node-coverage-incomplete" in after.reasons
    assert not detached.meta
    assert set(leaf._REGISTERED) == registered


@pytest.mark.parametrize(
    "entry",
    ("validate", "resolve", "internal"),
    ids=("validate", "resolve", "internal"),
)
def test_genuine_child_catalog_loses_every_actual_ancestor_discovery_edge(
    entry: str,
) -> None:
    model, witnesses = _nested_owner_model()
    child = _prepare_genuine_child_receipt(model)
    _run(ir.Model(child, ir_version=10))
    (receipt,) = leaf.receipts(child)
    assert receipt in leaf._REGISTERED
    assert leaf._congruence(receipt) == {witnesses["C1"]: witnesses["C2"]}
    assert not leaf.receipts(model.graph)
    assert leaf._catalog_graphs(child)[receipt.owner] is child
    owners = leaf._catalog_graphs(child)
    assert owners[model.graph.meta[structure._OWNER]] is model.graph
    leaf.validate_leaf_cse(child, structure._resolve_bindings)
    detached_values = []
    for node in child:
        for port, operand in enumerate(tuple(node.inputs)):
            if operand is not None and operand.graph is model.graph:
                detached = _value(operand.name, operand.dtype, tuple(operand.shape))
                node.replace_input_with(port, detached)
                detached_values.append(detached)
    assert detached_values
    assert all(value.graph is None and not value.meta for value in detached_values)
    # No remaining actual definition edge can discover the required ancestor.
    assert leaf._catalog_graphs(child) == {receipt.owner: child}
    assert leaf.receipts(child) == (receipt,) and receipt in leaf._REGISTERED
    calls = []
    with pytest.raises(
        structure.StructureUnavailable,
        match="^leaf-cse-catalog-dependency-owner-unavailable$",
    ):
        if entry == "validate":
            leaf.validate_leaf_cse(child, lambda *args: calls.append(args))
        elif entry == "resolve":
            leaf.resolve_leaf_bindings(
                child, {witnesses["C1"]}, frozenset(), lambda *args: calls.append(args)
            )
        else:
            leaf._validate(child, receipt, lambda *args: calls.append(args))
    assert not calls
    assert all(not value.meta for value in detached_values)


@pytest.mark.parametrize(
    "entry",
    ("validate", "resolve", "internal"),
    ids=("validate", "resolve", "internal"),
)
def test_detached_same_name_required_operand_keeps_current_binding_checks(
    entry: str,
) -> None:
    model, witnesses = _sub_model()
    _run(model)
    canonical = _resolved(model, witnesses)[witnesses[0]]
    (receipt,) = leaf.receipts(model.graph)
    assert receipt in leaf._REGISTERED
    consumer = model.graph.outputs[2].producer()
    assert consumer.inputs[0] is canonical
    original = consumer.inputs[1]
    assert original is model.graph.inputs[1]
    detached = _value(original.name, original.dtype, tuple(original.shape))
    consumer.replace_input_with(1, detached)
    assert detached is not original and detached.graph is None and not detached.meta
    leaf.validate_leaf_catalog(model.graph)
    assert (
        structure._resolve_direct_bindings(
            model.graph, {receipt.rows[0].normalized_inputs[1]}
        )[receipt.rows[0].normalized_inputs[1]]
        is original
    )
    with pytest.raises(
        structure.StructureMismatch,
        match="^leaf-cse-final-interface-edges-differ$",
    ):
        if entry == "validate":
            leaf.validate_leaf_cse(model.graph, structure._resolve_bindings)
        elif entry == "resolve":
            leaf.resolve_leaf_bindings(
                model.graph, {witnesses[1]}, frozenset(), structure._resolve_bindings
            )
        else:
            leaf._validate(model.graph, receipt, structure._resolve_bindings)
    assert leaf.receipts(model.graph) == (receipt,)
    assert not detached.meta


@pytest.mark.parametrize("owner", ("root", "child"), ids=("root", "child"))
@pytest.mark.parametrize(
    "entry",
    ("validate", "resolve", "internal"),
    ids=("validate", "resolve", "internal"),
)
def test_genuine_cross_scope_cycle_with_unrelated_detached_operand_rejects_callbacks(
    owner: str, entry: str
) -> None:
    original, witnesses = _nested_owner_model()
    left, right = original.clone(), original.clone()
    _run(left)
    (parent,) = leaf.receipts(left.graph)
    assert leaf._congruence(parent) == {witnesses["Y"]: witnesses["X"]}
    child_graph = _prepare_genuine_child_receipt(right)
    _run(ir.Model(child_graph, ir_version=10))
    (child,) = leaf.receipts(child_graph)
    assert leaf._congruence(child) == {witnesses["C1"]: witnesses["C2"]}
    assert parent in leaf._REGISTERED and child in leaf._REGISTERED
    assert parent.owner is right.graph.meta[structure._OWNER]
    assert child.owner is child_graph.meta[structure._OWNER]
    assert parent.owner is not child.owner
    right.graph.meta[leaf._RECEIPTS] = (parent,)
    detached = _value("unrelated_detached", ir.DataType.FLOAT, (2,))
    child_graph.insert_before(
        next(iter(child_graph)),
        ir.Node(
            "",
            "Neg",
            [detached],
            outputs=[_value("unrelated_result", ir.DataType.FLOAT, (2,))],
        ),
    )
    assert detached.graph is None and not detached.meta
    owners = leaf._catalog_graphs(child_graph)
    assert owners[parent.owner] is right.graph
    assert owners[child.owner] is child_graph
    # Individually authentic acyclic catalogs still expose their genuine
    # cross-scope cycle when the unchanged global gate combines dependencies.
    assert leaf._check_catalog(right.graph, leaf.receipts(right.graph))
    assert leaf._check_catalog(child_graph, leaf.receipts(child_graph))
    graph, receipt, alias = (
        (right.graph, parent, witnesses["Y"])
        if owner == "root"
        else (child_graph, child, witnesses["C1"])
    )
    calls = []
    with pytest.raises(
        structure.StructureUnavailable, match="^leaf-cse-cross-catalog-cycle$"
    ):
        if entry == "validate":
            leaf.validate_leaf_cse(graph, lambda *args: calls.append(args))
        elif entry == "resolve":
            leaf.resolve_leaf_bindings(
                graph, {alias}, frozenset(), lambda *args: calls.append(args)
            )
        else:
            leaf._validate(graph, receipt, lambda *args: calls.append(args))
    assert not calls
    assert not detached.meta
