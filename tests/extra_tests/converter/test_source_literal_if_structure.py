# tests/extra_tests/converter/test_source_literal_if_structure.py

"""Original literal selection, selected physical slice and unreachable accounting."""

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import jax
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import Literal
from jax2onnx.converter import _source_regions as regions
from jax2onnx.converter import ir_context, source_structure as structure
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.ir_utils import iter_ir_functions
from jax2onnx.plugins import _source_control_flow_structure as controls
from jax2onnx.plugins import plugin_system
from jax2onnx.precision import (
    SampleCriteria,
    SourceSample,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)

_CALLS: list[str] = []
_STORAGE = np.asarray([0x80000000, 0x3F800001], np.uint32).view(np.float32)


@onnx_function
def literal_selected(left: Any, right: Any, *, deterministic: Any = True) -> Any:
    _CALLS.append("selected")
    return jax.lax.cond(
        deterministic, lambda x, y: x - y, lambda x, y: x + y, left, right
    )


@onnx_function
def literal_wrapped(left: Any, right: Any) -> Any:
    _CALLS.append("wrapped")
    return literal_selected(left, right, deterministic=False)


@pytest.fixture(autouse=True)
def _module_function_bindings() -> Iterator[None]:
    targets = (literal_selected, literal_wrapped)
    previous = [
        (
            f"{target.__module__}.{target.__name__}",
            plugin_system.PLUGIN_REGISTRY.get(
                f"onnx_fn::{target.__module__}.{target.__name__}"
            ),
            plugin_system.ONNX_FUNCTION_PLUGIN_REGISTRY.get(
                f"{target.__module__}.{target.__name__}"
            ),
        )
        for target in targets
    ]
    try:
        for target in targets:
            onnx_function(target)
            qualified = f"{target.__module__}.{target.__name__}"
            plugin = plugin_system.ONNX_FUNCTION_PLUGIN_REGISTRY[qualified]
            assert globals()[target.__name__] is target
            assert plugin.target is target
            assert plugin_system.PLUGIN_REGISTRY.get("onnx_fn::" + qualified) is plugin
        yield
    finally:
        for target, (qualified, canonical, named) in zip(
            targets, previous, strict=True
        ):
            plugin_system.PLUGIN_REGISTRY.pop("onnx_fn::" + qualified, None)
            plugin_system.ONNX_FUNCTION_PLUGIN_REGISTRY.pop(qualified, None)
            if canonical is not None:
                plugin_system.PLUGIN_REGISTRY.register_canonical(
                    "onnx_fn::" + qualified, canonical
                )
            if named is not None:
                plugin_system.ONNX_FUNCTION_PLUGIN_REGISTRY[qualified] = named
            assert globals()[target.__name__] is target
            assert (
                plugin_system.PLUGIN_REGISTRY.get("onnx_fn::" + qualified) is canonical
            )
            assert plugin_system.ONNX_FUNCTION_PLUGIN_REGISTRY.get(qualified) is named


def _target(index: int, mode: str = "different") -> Any:
    def target(left: Any, right: Any) -> Any:
        if mode == "alias":
            branches = (lambda x, y: (y, x, y), lambda x, y: (x, y, x))
        elif mode == "repeated":

            def false(x: Any, y: Any) -> Any:
                result = x + y
                return result, x, result

            def true(x: Any, y: Any) -> Any:
                result = x - y
                return result, y, result

            branches = (false, true)
        elif mode == "storage":
            branches = (lambda x, y: x + _STORAGE, lambda x, y: x - _STORAGE)
        else:
            branches = (lambda x, y: x + y, lambda x, y: x - y)
        return jax.lax.cond(bool(index), branches[1], branches[0], left, right)

    return target


def _source(target: Any, *, dtype: Any = np.float32, **kwargs: Any) -> ir.Model:
    with jax.enable_x64(True):
        model = ui._to_onnx(
            target,
            [jax.ShapeDtypeStruct((2,), dtype)] * 2,
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
            opset=23,
            **kwargs,
        )
    assert isinstance(model, ir.Model)
    return model


def _proof(model: ir.Model) -> structure.StructureResult:
    return structure.validate_source_structure(model)


def _validated(model: ir.Model) -> Any:
    proof = _proof(model)
    assert proof.status == "validated", proof
    assert proof.checked_equations == proof.expected_equations
    assert proof.runtime == proof.numerics == "unverified"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    return model.meta[structure._LEDGER]


def _graphs(model: ir.Model) -> list[ir.Graph]:
    result = []

    def walk(graph: ir.Graph) -> None:
        result.append(graph)
        for node in graph:
            for attribute in node.attributes.values():
                if attribute.type is ir.AttributeType.GRAPH:
                    walk(attribute.as_graph())

    walk(model.graph)
    for function in iter_ir_functions(model.functions):
        walk(function.graph)
    return result


def _attachment(model: ir.Model) -> tuple[Any, Any, Any]:
    ledger = model.meta[structure._LEDGER]
    assert len(ledger.literal_if_calls) == 1
    handle, attachment = next(iter(ledger.literal_if_calls.items()))
    return ledger, handle, attachment


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("mode", ("different", "alias", "repeated", "storage"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_exact_selector_both_original_bodies_and_disjoint_accounting(
    index: int, mode: str, dtype: Any
) -> None:
    model = _source(_target(index, mode), dtype=dtype)
    ledger = _validated(model)
    _, handle, attachment = _attachment(model)
    assert attachment.rule.index == index
    assert type(attachment.rule) is controls.LiteralIfRule
    assert attachment.handle == handle
    assert attachment.rule.selector.storage == structure.ConstantContract.from_array(
        np.asarray(index, np.int32)
    )
    assert attachment.rule.selector.tensor.weak_type is False
    assert attachment.rule.selector.tensor.dtype is ir.DataType.INT32
    assert attachment.rule.selector.tensor.shape == ()
    originals = (attachment.false, attachment.true)
    assert all(
        body.invocation in {scope.invocation for scope in ledger.scopes}
        for body in originals
    )
    assert all(body.body.gaps == () for body in originals)
    assert all(body.region.key.parent == handle for body in originals)
    assert tuple(body.region.key.role for body in originals) == ("false", "true")
    assert {body.owner for body in originals}.isdisjoint(
        graph.meta[structure._OWNER] for graph in _graphs(model)
    )
    selected, inactive = originals[index], originals[1 - index]
    discharge = attachment.discharge
    assert discharge.inactive_invocation is inactive.invocation
    assert discharge.inactive_body == inactive.body
    assert discharge.graph_matched_equations == len(selected.equations)
    assert discharge.semantically_discharged_equations == 1 + len(inactive.equations)
    assert discharge.inactive_nodes == discharge.inactive_storage == ()
    assert (
        discharge.graph_matched_equations + discharge.semantically_discharged_equations
        == ledger.expected_equations
    )
    assert not any(node.op_type == "If" for graph in _graphs(model) for node in graph)
    assert len(model.graph.inputs) == 2
    assert not ledger.if_calls
    assert len(ledger.regions) == 2
    if mode == "different":
        assert len(attachment.false.equations) == len(attachment.true.equations) == 1
        assert attachment.false.body != attachment.true.body
        assert ledger.expected_equations == 3
    elif mode == "alias":
        assert ledger.expected_equations == 1
        assert discharge.graph_matched_equations == 0


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("mode", ("different", "alias", "repeated", "storage"))
@pytest.mark.parametrize("optimization", ("disabled", "all"))
def test_independent_original_jax_bitwise_samples_and_reload(
    index: int, mode: str, optimization: str
) -> None:
    target = _target(index, mode)
    model = _source(target)
    _validated(model)
    inputs = (np.asarray([0.5, -2.0], np.float32), np.asarray([0.25, 1.0], np.float32))
    with jax.enable_x64(True):
        original = jax.tree_util.tree_leaves(target(*inputs))
    expected = [np.asarray(value) for value in original]
    for candidate in (model, ir.from_proto(ir.to_proto(model))):
        report = validate_source_samples(
            candidate,
            samples=[SourceSample(inputs, expected, SampleCriteria(mode="bitwise"))],
            optimization=optimization,
        )
        report.require_sample_acceptance()


@pytest.mark.parametrize(
    "placement",
    ("default", "literal-true", "literal-false", "function", "scan", "while"),
)
def test_literal_selection_inside_original_function_scan_and_while(
    placement: str,
) -> None:
    def target(left: Any, right: Any) -> Any:
        if placement == "default":
            return literal_selected(left, right)
        if placement.startswith("literal-"):
            return literal_selected(
                left, right, deterministic=placement == "literal-true"
            )
        if placement == "function":
            return literal_wrapped(left, right)
        if placement == "scan":
            return jax.lax.scan(
                lambda carry, unused: (
                    literal_selected(carry, right, deterministic=False),
                    carry,
                ),
                left,
                None,
                length=2,
            )
        return jax.lax.while_loop(
            lambda state: state[0] < np.int32(2),
            lambda state: (
                state[0] + np.int32(1),
                literal_selected(state[1], right, deterministic=False),
            ),
            (np.int32(0), left),
        )

    model = _source(target)
    ledger = _validated(model)
    assert len(ledger.literal_if_calls) == 1
    assert len(ledger.regions) >= 2
    assert all(
        not graph.inputs
        for graph in _graphs(model)
        if graph.meta[structure._OWNER]
        in {
            item.owner
            for attachment in ledger.literal_if_calls.values()
            for item in (attachment.false, attachment.true)
        }
    )
    inputs = (np.asarray([0.5, -2.0], np.float32), np.asarray([0.25, 1.0], np.float32))
    with jax.enable_x64(True):
        expected = [
            np.asarray(value) for value in jax.tree_util.tree_leaves(target(*inputs))
        ]
    report = validate_source_samples(
        model,
        samples=[SourceSample(inputs, expected, SampleCriteria(mode="bitwise"))],
        optimization="disabled",
    )
    report.require_sample_acceptance()


@pytest.mark.parametrize("fault", ("dtype", "shape", "weak", "value", "payload-width"))
def test_native_selector_contract_rejects_altered_literal(fault: str) -> None:
    target = _target(0)
    native = jax.make_jaxpr(target)(*[jax.ShapeDtypeStruct((2,), np.float32)] * 2)
    equation = next(eqn for eqn in native.jaxpr.eqns if eqn.primitive is jax.lax.cond_p)
    selector = equation.invars[0]
    assert isinstance(selector, Literal)
    aval = selector.aval
    value: Any = np.int32(0)
    if fault == "dtype":
        aval = aval.update(dtype=np.dtype(np.int64))
        value = np.int64(0)
    elif fault == "shape":
        aval = aval.update(shape=(1,))
        value = np.asarray([0], np.int32)
    elif fault == "weak":
        aval = aval.update(weak_type=True)
    elif fault == "value":
        value = np.int32(2)
    else:
        value = np.int64(0)
    changed = equation.replace(invars=[Literal(value, aval), *equation.invars[1:]])
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in changed.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in changed.outvars
    )
    assert controls.freeze_control_flow_rule(changed, inputs, outputs) is None


@pytest.mark.parametrize(
    "fault",
    (
        "map",
        "map-extra-port",
        "map-drop",
        "ports",
        "alias",
        "selector",
        "discharge",
        "inactive-count",
        "body",
        "receipt",
        "lost-call",
        "foreign",
        "foreign-bound",
    ),
)
def test_frozen_attachment_corruptions_keep_original_denominator(fault: str) -> None:
    model = _source(_target(1, "repeated"))
    _validated(model)
    ledger, handle, attachment = _attachment(model)
    expected = ledger.expected_equations
    if fault == "map":
        source, target = attachment.values[0]
        changed = replace(
            attachment,
            values=((source, replace(target, origin=object())), *attachment.values[1:]),
        )
    elif fault == "map-extra-port":
        source, target = attachment.values[0]
        changed = replace(
            attachment,
            values=(
                *attachment.values,
                (replace(source, port=999), replace(target, port=999)),
            ),
        )
    elif fault == "map-drop":
        changed = replace(attachment, values=attachment.values[1:])
    elif fault == "ports":
        body = replace(
            attachment.false,
            region=replace(
                attachment.false.region, captures=attachment.false.region.captures[::-1]
            ),
        )
        changed = replace(attachment, false=body)
    elif fault == "alias":
        changed = replace(attachment, returned=attachment.returned[::-1])
    elif fault == "selector":
        changed = replace(attachment, rule=replace(attachment.rule, index=0))
    elif fault == "discharge":
        changed = replace(
            attachment,
            discharge=replace(attachment.discharge, inactive_nodes=(object(),)),
        )
    elif fault == "inactive-count":
        changed = replace(
            attachment,
            discharge=replace(
                attachment.discharge, semantically_discharged_equations=1
            ),
        )
    elif fault == "body":
        scope = next(
            scope
            for scope in ledger.scopes
            if scope.invocation is attachment.false.invocation
        )
        scope.body = replace(scope.body, equations=())
        changed = attachment
    elif fault == "receipt":
        scope = next(
            scope
            for scope in ledger.scopes
            if scope.invocation is attachment.false.invocation
        )
        scope.receipts.clear()
        changed = attachment
    elif fault == "lost-call":
        del ledger.literal_if_calls[handle]
        changed = None
    else:
        foreign = _source(_target(1, "repeated"))
        _validated(foreign)
        changed = _attachment(foreign)[2]
        if fault == "foreign-bound":
            changed = replace(
                changed, handle=handle, owner=attachment.owner, rule=attachment.rule
            )
    if changed is not None:
        ledger.literal_if_calls[handle] = changed
    proof = _proof(model)
    assert proof.status != "validated", proof
    assert proof.expected_equations == expected


@pytest.mark.parametrize(
    "fault",
    (
        "operand",
        "order",
        "inactive-marker",
        "inactive-node",
        "inactive-storage",
        "constant",
    ),
)
def test_final_selected_edges_order_and_inactive_physical_coverage(fault: str) -> None:
    model = _source(_target(1, "storage" if fault == "constant" else "different"))
    _validated(model)
    ledger, _, attachment = _attachment(model)
    graph = model.graph
    selected = next(node for node in graph if node.op_type == "Sub")
    if fault == "operand":
        selected.replace_input_with(1, selected.inputs[0])
    elif fault == "order":
        producer = selected.inputs[0].producer()
        assert producer is not None
        graph.insert_after(selected, producer)
    elif fault == "inactive-marker":
        inactive = attachment.false
        witness = inactive.receipts[0][1].outputs[0]
        selected.outputs[0].meta[structure._VALUES] += (witness,)
    elif fault == "inactive-node":
        extra = ir.Value(
            name="inactive_extra",
            type=selected.outputs[0].type,
            shape=selected.outputs[0].shape,
        )
        graph.append(ir.Node("", "Neg", [selected.outputs[0]], outputs=[extra]))
    elif fault == "inactive-storage":
        value = ir.Value(name="inactive_storage", const_value=ir.tensor(_STORAGE))
        graph.initializers[value.name] = value
    else:
        value = selected.inputs[1]
        assert value is not None
        producer = value.producer()
        if producer is not None and producer.op_type == "Identity":
            value = producer.inputs[0]
            assert value is not None
            producer = value.producer()
        tensor = ir.tensor(np.asarray([0.0, 1.0], np.float32))
        if producer is None:
            value.const_value = tensor
        else:
            assert producer.op_type == "Constant"
            producer.attributes["value"] = ir.AttrTensor("value", tensor)
    proof = _proof(model)
    assert proof.status != "validated", proof
    assert proof.expected_equations == ledger.expected_equations


def test_clone_safe_validation_preserves_producer_bound_receipts() -> None:
    original = _source(_target(0, "repeated"))
    ledger = _validated(original)
    cloned = ir.Model(
        original.graph.clone(allow_outer_scope_values=True),
        ir_version=original.ir_version,
    )
    cloned.meta.update(original.meta)
    assert cloned.graph is not original.graph
    assert _validated(cloned) is ledger


def test_observation_on_off_exact_proto_report_and_call_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fresh_root() -> Any:
        def target(left: Any, right: Any) -> Any:
            return literal_wrapped(left, right)

        return target

    _CALLS.clear()
    enabled = _source(fresh_root())
    enabled_calls = tuple(_CALLS)
    _validated(enabled)
    assert set(enabled_calls) == {"selected", "wrapped"}
    _CALLS.clear()
    with monkeypatch.context() as patch:
        patch.setattr(ir_context, "SourceStructureLedger", lambda: None)
        disabled = _source(fresh_root())
    assert tuple(_CALLS) == enabled_calls
    assert structure._LEDGER not in disabled.meta
    assert ir.to_proto(enabled).SerializeToString(deterministic=True) == ir.to_proto(
        disabled
    ).SerializeToString(deterministic=True)
    enabled_report, disabled_report = get_precision_report(
        enabled
    ), get_precision_report(disabled)
    assert enabled_report == disabled_report
    assert enabled_report is not None
    assert precision_report_to_json(enabled_report) == precision_report_to_json(
        disabled_report
    )
    assert tuple(_CALLS) == enabled_calls


def test_freeze_and_finalizer_never_retrace_execute_or_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target(1)
    native = jax.make_jaxpr(target)(*[jax.ShapeDtypeStruct((2,), np.float32)] * 2)
    model = _source(target)
    _validated(model)
    before = ir.to_proto(model).SerializeToString(deterministic=True)

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("literal selection observation retraced or materialized a value")

    monkeypatch.setattr(jax, "make_jaxpr", unavailable)
    monkeypatch.setattr(jax, "eval_shape", unavailable)
    monkeypatch.setattr(ir_context.IRContext, "require_value_for_var", unavailable)
    frozen = regions.freeze_closed_body(native)
    assert frozen.gaps == ()
    assert frozen.expected_equations == 3
    _validated(model)
    assert ir.to_proto(model).SerializeToString(deterministic=True) == before


@pytest.mark.parametrize("index", (0, 1))
def test_unused_original_results_keep_both_branch_counts(index: int) -> None:
    repeated = _target(index, "repeated")

    def target(left: Any, right: Any) -> Any:
        return repeated(left, right)[0]

    model = _source(target)
    ledger = _validated(model)
    assert ledger.expected_equations == 3
    _, _, attachment = _attachment(model)
    assert (
        len(attachment.false.body.equations) == len(attachment.true.body.equations) == 1
    )
    assert len(model.graph.outputs) == 1


def test_inactive_nested_branch_keeps_explicit_unfinished_discharge() -> None:
    def target(left: Any, right: Any) -> Any:
        return jax.lax.cond(
            True,
            lambda x, y: x - y,
            lambda x, y: literal_selected(x, y),
            left,
            right,
        )

    model = _source(target)
    proof = _proof(model)
    assert proof.status == "unverified", proof
    assert proof.expected_equations is not None and proof.expected_equations > 3
    assert any(
        "literal-if-childless-contract-unavailable" in reason
        for reason in proof.reasons
    )
    assert not any("opset" in reason or "target" in reason for reason in proof.reasons)


_SECOND_STORAGE = np.asarray([0x3F000001, 0xBF800001], np.uint32).view(np.float32)


def _closed_target(index: int) -> Any:
    """Keep genuine branch ClosedJaxpr constants, rather than hoisted captures."""
    spec = jax.ShapeDtypeStruct((2,), np.float32)
    false = jax.make_jaxpr(lambda x, unused: (x - _STORAGE) + _SECOND_STORAGE)(
        spec, spec
    )
    true = jax.make_jaxpr(lambda x, unused: (x - _SECOND_STORAGE) + _STORAGE)(
        spec, spec
    )
    assert len(false.consts) == len(true.consts) == 2

    def target(left: Any, right: Any) -> Any:
        (value,) = jax.lax.cond_p.bind(
            np.int32(index), left, right, branches=(false, true)
        )
        return value

    return target


@pytest.mark.parametrize("index", (0, 1))
def test_real_branch_closed_constants_project_exact_original_order(
    index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _closed_target(index)
    model = _source(target)
    ledger = _validated(model)
    _, _, attachment = _attachment(model)
    selected = (attachment.false, attachment.true)[index]
    assert ledger.expected_equations == 5
    assert (
        len(attachment.false.body.equations) == len(attachment.true.body.equations) == 2
    )
    assert len(selected.closed_constant_values) == 2
    original_scope = next(
        scope for scope in ledger.scopes if scope.invocation is selected.invocation
    )
    assert original_scope.closed_constant_values == selected.closed_constant_values
    assert all(
        witness.owner is selected.owner for witness in selected.closed_constant_values
    )
    mapping = dict(attachment.values)
    expected = tuple(mapping[witness] for witness in selected.closed_constant_values)
    observed = []
    audit_scope = structure._audit_scope

    def measured(graph: Any, scope: Any, *args: Any, **kwargs: Any) -> Any:
        if scope.invocation is selected.invocation:
            observed.append(scope.closed_constant_values)
            assert scope.closed_constant_values == expected
            assert all(
                witness.owner is graph.meta[structure._OWNER]
                for witness in scope.closed_constant_values
            )
        return audit_scope(graph, scope, *args, **kwargs)

    monkeypatch.setattr(structure, "_audit_scope", measured)
    assert _validated(model) is ledger
    assert observed == [expected]
    assert original_scope.closed_constant_values == selected.closed_constant_values
    assert all(
        witness.owner is selected.owner
        for witness in original_scope.closed_constant_values
    )
    inputs = (np.asarray([0.5, -2.0], np.float32), np.asarray([0.25, 1.0], np.float32))
    with jax.enable_x64(True):
        expected_values = [np.asarray(target(*inputs))]
    for optimization in ("disabled", "all"):
        report = validate_source_samples(
            model,
            samples=[
                SourceSample(inputs, expected_values, SampleCriteria(mode="bitwise"))
            ],
            optimization=optimization,
        )
        report.require_sample_acceptance()


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("fault", ("missing", "order", "foreign"))
def test_closed_constant_projection_cannot_borrow_or_reorder_original_ports(
    index: int, fault: str
) -> None:
    model = _source(_closed_target(index))
    ledger = _validated(model)
    _, _, attachment = _attachment(model)
    selected = (attachment.false, attachment.true)[index]
    original_scope = next(
        scope for scope in ledger.scopes if scope.invocation is selected.invocation
    )
    original_count = ledger.expected_equations
    if fault == "missing":
        original_scope.closed_constant_values = original_scope.closed_constant_values[
            :-1
        ]
    elif fault == "order":
        original_scope.closed_constant_values = original_scope.closed_constant_values[
            ::-1
        ]
    else:
        foreign = _source(_closed_target(index))
        _validated(foreign)
        foreign_attachment = _attachment(foreign)[2]
        original_scope.closed_constant_values = (
            foreign_attachment.false,
            foreign_attachment.true,
        )[index].closed_constant_values
    proof = _proof(model)
    assert proof.status != "validated", proof
    assert proof.expected_equations == original_count
