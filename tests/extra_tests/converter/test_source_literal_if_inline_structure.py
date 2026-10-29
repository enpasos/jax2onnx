# tests/extra_tests/converter/test_source_literal_if_inline_structure.py

"""Exact source-only literal selection over complete pure native inline trees."""

from collections.abc import Iterator
from dataclasses import fields, is_dataclass, replace
from typing import Any

import jax
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import ClosedJaxpr, Jaxpr, native_jit_primitive
from jax2onnx.converter import _source_cast_context as cast_context
from jax2onnx.converter import _source_regions as regions
from jax2onnx.converter import ir_context, source_structure as structure
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.ir_utils import iter_ir_functions
from jax2onnx.plugins import _source_control_flow_structure as controls
from jax2onnx.plugins import plugin_system
from jax2onnx.plugins.jax.core import _source_inline_structure as inline
from jax2onnx.precision import (
    SampleCriteria,
    SourceSample,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)

_FIRST = np.asarray([0x80000000, 0x3F000000], np.uint32).view(np.float32)
_SECOND = _FIRST.copy()


def _difference(left: Any, right: Any) -> Any:
    return left - right


_jitted_difference = jax.jit(_difference)
_jitted_aliases = jax.jit(lambda left, right: (left, right, left))
_jitted_literals = jax.jit(lambda left, right: (left, np.float32(-0.0), right))
_jitted_scale = jax.jit(lambda left, scale: left * scale)
_jitted_unused = jax.jit(lambda left, unused: left, keep_unused=True)
_jitted_closed = jax.jit(lambda left, right: (left - _FIRST, right - _SECOND))


def _nested(left: Any, right: Any) -> Any:
    return _jitted_difference(left, right) * left


_jitted_nested = jax.jit(_nested)


def _repeated(left: Any, right: Any) -> Any:
    return _jitted_difference(left, right), _jitted_difference(right, left)


_jitted_repeated = jax.jit(_repeated)


def _rich(left: Any, right: Any) -> Any:
    first, second = _jitted_closed(left, right)
    result = _jitted_nested(first, right)
    return result, second, result


_jitted_rich = jax.jit(_rich)


def _target(index: int, kind: str = "nested") -> Any:
    def branch(left: Any, right: Any) -> Any:
        if kind == "leaf":
            return _jitted_difference(left, right)
        if kind == "nested":
            return _jitted_nested(left, right)
        if kind == "repeated":
            return _jitted_repeated(left, right)
        if kind == "aliases":
            return _jitted_aliases(left, right)
        if kind == "literals":
            return _jitted_literals(left, right)
        if kind == "literal-argument":
            return _jitted_scale(left, np.float32(0.75))
        if kind == "unused":
            # The original JIT formal is unused; this separate original result
            # keeps its exact caller binding physically present after DCE.
            return _jitted_unused(left, right), right
        if kind == "closed":
            return _jitted_closed(left, right)
        return _jitted_rich(left, right)

    def target(left: Any, right: Any) -> Any:
        return jax.lax.cond(bool(index), branch, lambda x, y: branch(y, x), left, right)

    return target


@onnx_function
def _named_true(left: Any, right: Any) -> Any:
    return jax.lax.cond(
        True, _jitted_nested, lambda x, y: _jitted_nested(y, x), left, right
    )


@onnx_function
def _named_false(left: Any, right: Any) -> Any:
    return jax.lax.cond(
        False, _jitted_nested, lambda x, y: _jitted_nested(y, x), left, right
    )


@pytest.fixture(autouse=True)
def _module_function_bindings() -> Iterator[None]:
    targets = (_named_true, _named_false)
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
            assert globals()[target.__name__] is target and plugin.target is target
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


def _specs(dtype: Any = np.float32) -> list[Any]:
    return [jax.ShapeDtypeStruct((2,), dtype)] * 2


def _source(target: Any, dtype: Any = np.float32) -> ir.Model:
    with jax.enable_x64(True):
        model = ui._to_onnx(
            target,
            _specs(dtype),
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
            opset=23,
        )
    assert isinstance(model, ir.Model)
    return model


def _proof(model: ir.Model) -> structure.StructureResult:
    return structure.validate_source_structure(model)


def _validated(model: ir.Model) -> Any:
    proof = _proof(model)
    assert proof.status == "validated", proof
    assert proof.expected_equations == proof.checked_equations
    assert proof.runtime == proof.numerics == "unverified"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    return model.meta[structure._LEDGER]


def _attachment(model: ir.Model) -> tuple[Any, Any, Any]:
    ledger = model.meta[structure._LEDGER]
    assert len(ledger.literal_if_calls) == 1
    handle, attachment = next(iter(ledger.literal_if_calls.items()))
    return ledger, handle, attachment


def _scope(ledger: Any, snapshot: Any) -> Any:
    return next(
        scope for scope in ledger.scopes if scope.invocation is snapshot.invocation
    )


def _graphs(model: ir.Model) -> list[ir.Graph]:
    result = []

    def visit(graph: ir.Graph) -> None:
        result.append(graph)
        for node in graph:
            for attr in node.attributes.values():
                if attr.type is ir.AttributeType.GRAPH:
                    visit(attr.as_graph())
                elif attr.type is ir.AttributeType.GRAPHS:
                    for child in attr.as_graphs():
                        visit(child)

    visit(model.graph)
    for function in iter_ir_functions(model.functions):
        visit(function.graph)
    return result


def _stored_values(model: ir.Model) -> list[ir.Value]:
    return [
        value
        for graph in _graphs(model)
        for value in (
            *graph.initializers.values(),
            *(
                value
                for node in graph
                for value in node.outputs
                if node.op_type == "Constant"
            ),
        )
    ]


def _storage(value: ir.Value) -> Any:
    producer = value.producer()
    return (
        value.const_value.numpy()
        if producer is None
        else producer.attributes["value"].as_tensor().numpy()
    )


def _rejected(model: ir.Model, expected: int) -> None:
    proof = _proof(model)
    assert proof.status != "validated", proof
    assert proof.expected_equations == expected


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize(
    "kind",
    (
        "leaf",
        "nested",
        "repeated",
        "aliases",
        "literals",
        "literal-argument",
        "unused",
        "closed",
        "rich",
    ),
)
def test_both_original_inline_trees_keep_disjoint_complete_accounting(
    index: int, kind: str
) -> None:
    target = _target(index, kind)
    with jax.enable_x64(True):
        native = jax.make_jaxpr(target)(*_specs())
    original = regions.freeze_closed_body(native)
    assert original.gaps == ()
    model = _source(target)
    ledger = _validated(model)
    _, _, attachment = _attachment(model)
    assert ledger.expected_equations == original.expected_equations
    assert controls.authentic_literal_if_attachment(ledger, attachment)
    assert attachment.rule.index == index
    selected, inactive = attachment.trees[index], attachment.trees[1 - index]
    assert len(selected.scopes) == len(inactive.scopes) > 1
    assert len(
        {scope.invocation for tree in attachment.trees for scope in tree.scopes}
    ) == sum(len(tree.scopes) for tree in attachment.trees)
    assert all(
        scope.owner is tree.scopes[0].owner
        for tree in attachment.trees
        for scope in tree.scopes
    )
    assert {scope.invocation for tree in attachment.trees for scope in tree.scopes} == {
        scope.invocation for scope in ledger.scopes if scope.region is not None
    }
    for tree in attachment.trees:
        assert (
            sum(len(scope.equations) for scope in tree.scopes)
            == tree.scopes[0].body.expected_equations
        )
        assert len(tree.inline_calls) == len(tree.scopes) - 1
        for handle, call in tree.inline_calls:
            assert ledger.inline_calls[handle] is call
            assert inline.authentic_inline_attachment(ledger, handle, call)
            assert ledger.regions[call.region] in {
                scope.invocation for scope in tree.scopes
            }
    discharge = attachment.discharge
    assert discharge.graph_matched_equations == sum(
        len(scope.equations) for scope in selected.scopes
    )
    assert discharge.semantically_discharged_equations == 1 + sum(
        len(scope.equations) for scope in inactive.scopes
    )
    assert (
        discharge.graph_matched_equations + discharge.semantically_discharged_equations
        == ledger.expected_equations
    )
    assert discharge.inactive_nodes == discharge.inactive_storage == ()
    assert not any(node.op_type == "If" for graph in _graphs(model) for node in graph)
    assert len(model.graph.inputs) == 2
    inactive_owner = inactive.scopes[0].owner
    assert all(
        graph.meta[structure._OWNER] is not inactive_owner for graph in _graphs(model)
    )
    assert all(
        marker.owner is not inactive_owner
        for graph in _graphs(model)
        for value in (
            *graph.inputs,
            *graph.initializers.values(),
            *(value for node in graph for value in node.outputs),
        )
        for marker in value.meta.get(structure._VALUES, ())
    )


@pytest.mark.parametrize("index", (0, 1))
def test_selected_scope_and_inline_attachment_projection_uses_every_actual_clone_pair(
    index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _source(_target(index, "rich"))
    ledger = _validated(model)
    _, _, attachment = _attachment(model)
    selected = attachment.trees[index]
    mapping = dict(attachment.values)
    original_snapshots = {
        scope.invocation: controls.literal_if_body(_scope(ledger, scope))
        for scope in selected.scopes
    }
    seen = []
    original_audit = structure._audit_scope

    def measured(graph: Any, scope: Any, *args: Any, **kwargs: Any) -> Any:
        if scope.invocation in original_snapshots:
            snapshot = original_snapshots[scope.invocation]
            seen.append(scope.invocation)
            assert scope.owner is graph.meta[structure._OWNER]
            assert scope.input_values == tuple(
                mapping[value] for value in snapshot.input_values
            )
            assert scope.closed_constant_values == tuple(
                mapping[value] for value in snapshot.closed_constant_values
            )
            assert scope.variables == {
                key: mapping[value] for key, value in snapshot.variables
            }
            assert (
                scope.invocation is snapshot.invocation and scope.body is snapshot.body
            )
        return original_audit(graph, scope, *args, **kwargs)

    monkeypatch.setattr(structure, "_audit_scope", measured)
    assert _validated(model) is ledger
    assert set(seen) == set(original_snapshots) and len(seen) == len(original_snapshots)
    assert all(
        controls.literal_if_body(_scope(ledger, scope))
        == original_snapshots[scope.invocation]
        for scope in selected.scopes
    )


@pytest.mark.parametrize("side", ("selected", "inactive"))
@pytest.mark.parametrize(
    "fault",
    (
        "parent",
        "role",
        "capture-order",
        "formal-order",
        "result-order",
        "constant-order",
        "settings",
        "scope-missing",
        "scope-duplicate",
        "owner",
        "body-copy",
        "receipt-copy",
        "receipt-missing",
        "inline-copy",
        "inline-cross",
        "region-copy",
        "capture-type",
        "formal-type",
        "result-type",
        "constant-type",
        "invocation",
        "unknown-semantics",
        "cycle",
    ),
)
def test_original_tree_and_call_corruption_cannot_discharge_or_project(
    side: str, fault: str
) -> None:
    model = _source(_target(1, "rich"))
    ledger = _validated(model)
    _, _, attachment = _attachment(model)
    tree = attachment.trees[1 if side == "selected" else 0]
    if fault in ("constant-order", "constant-type"):
        snapshot = next(
            scope for scope in tree.scopes if len(scope.closed_constant_values) == 2
        )
    else:
        snapshot = next(
            scope for scope in tree.scopes[1:] if len(scope.input_values) >= 2
        )
    scope = _scope(ledger, snapshot)
    if fault == "parent":
        scope.region = replace(
            scope.region,
            key=replace(scope.region.key, parent=structure.EquationHandle(object(), 0)),
        )
    elif fault == "role":
        scope.region = replace(
            scope.region, key=replace(scope.region.key, role="false")
        )
    elif fault == "capture-order":
        scope.region = replace(scope.region, captures=scope.region.captures[::-1])
    elif fault == "formal-order":
        scope.input_values = scope.input_values[::-1]
    elif fault == "result-order":
        scope.result_values = scope.result_values[::-1]
        # Use a genuinely distinct ordered pair instead of a one-result no-op.
        if scope.result_values == snapshot.result_values:
            rich = next(item for item in tree.scopes if len(item.result_values) == 3)
            scope = _scope(ledger, rich)
            scope.result_values = (
                rich.result_values[1],
                rich.result_values[0],
                rich.result_values[2],
            )
    elif fault == "constant-order":
        assert scope.closed_constant_values[0] != scope.closed_constant_values[1]
        scope.closed_constant_values = scope.closed_constant_values[::-1]
    elif fault == "settings":
        index, equation = next(
            (index, equation)
            for index, equation in enumerate(scope.equations)
            if type(equation.rule) is inline.InlineRule
        )
        changed = replace(
            equation,
            rule=replace(equation.rule, keep_unused=not equation.rule.keep_unused),
        )
        scope.equations = (
            *scope.equations[:index],
            changed,
            *scope.equations[index + 1 :],
        )
    elif fault == "scope-missing":
        ledger.scopes.remove(scope)
    elif fault == "scope-duplicate":
        ledger.scopes.append(replace(scope, invocation=object()))
    elif fault == "owner":
        scope.owner = object()
    elif fault == "body-copy":
        scope.body = replace(scope.body)
    elif fault == "receipt-copy":
        index = next(iter(scope.receipts))
        scope.receipts[index] = replace(scope.receipts[index])
    elif fault == "receipt-missing":
        scope.receipts.pop(next(iter(scope.receipts)))
    elif fault in ("inline-copy", "inline-cross"):
        handle, call = tree.inline_calls[0]
        ledger.inline_calls[handle] = (
            replace(call)
            if fault == "inline-copy"
            else attachment.trees[1 if side == "inactive" else 0].inline_calls[0][1]
        )
    elif fault == "region-copy":
        scope.region = replace(scope.region)
    elif fault == "capture-type":
        port = scope.region.captures[0]
        scope.region = replace(
            scope.region,
            captures=(
                replace(
                    port,
                    tensor=replace(port.tensor, weak_type=not port.tensor.weak_type),
                ),
                *scope.region.captures[1:],
            ),
        )
    elif fault == "formal-type":
        use = scope.inputs[0]
        scope.inputs = (
            replace(
                use, tensor=replace(use.tensor, weak_type=not use.tensor.weak_type)
            ),
            *scope.inputs[1:],
        )
    elif fault == "result-type":
        use = scope.outputs[0]
        scope.outputs = (
            replace(
                use, tensor=replace(use.tensor, weak_type=not use.tensor.weak_type)
            ),
            *scope.outputs[1:],
        )
    elif fault == "constant-type":
        (tensor, storage), *rest = scope.body.constants
        scope.body = replace(
            scope.body,
            constants=(
                (replace(tensor, weak_type=not tensor.weak_type), storage),
                *rest,
            ),
        )
    elif fault == "invocation":
        scope.invocation = object()
    elif fault == "unknown-semantics":
        scope.body = replace(
            scope.body,
            equations=(
                replace(scope.body.equations[0], semantics=("opaque",)),
                *scope.body.equations[1:],
            ),
        )
    else:
        handle, call = tree.inline_calls[0]
        ledger.regions[call.region] = tree.scopes[0].invocation
        assert ledger.regions[call.region] is not snapshot.invocation
    _rejected(model, ledger.expected_equations)


@pytest.mark.parametrize("side", ("selected", "inactive"))
@pytest.mark.parametrize("field", ("inputs", "outputs", "constants", "region", "owner"))
def test_registered_inline_call_fields_cannot_borrow_equal_typed_ports(
    side: str, field: str
) -> None:
    model = _source(_target(1, "rich"))
    ledger = _validated(model)
    _, _, attachment = _attachment(model)
    tree = attachment.trees[1 if side == "selected" else 0]
    if field == "constants":
        handle, call = next(
            (handle, call)
            for handle, call in tree.inline_calls
            if len(call.constants) == 2
        )
        assert call.constants[0] != call.constants[1]
        assert (
            ledger.constants[call.constants[0]] == ledger.constants[call.constants[1]]
        )
    else:
        handle, call = tree.inline_calls[0]
    before = getattr(call, field)
    if field in ("inputs", "outputs", "constants"):
        assert len(before) >= 2 and before[0] != before[1]
        changed = (before[1], before[0], *before[2:])
    elif field == "region":
        changed = replace(
            before, parent=structure.EquationHandle(object(), handle.index)
        )
    else:
        changed = object()
    object.__setattr__(call, field, changed)
    try:
        _rejected(model, ledger.expected_equations)
    finally:
        object.__setattr__(call, field, before)


@pytest.mark.parametrize(
    "fault",
    (
        "copy",
        "foreign",
        "missing",
        "selected-drop",
        "selected-swap",
        "cross-owner",
        "port",
        "origin",
        "inactive-count",
        "snapshot-drop",
    ),
)
def test_authentic_producer_and_complete_selected_clone_map_are_required(
    fault: str,
) -> None:
    model = _source(_target(1, "rich"))
    ledger = _validated(model)
    _, handle, attachment = _attachment(model)
    expected = ledger.expected_equations
    if fault == "copy":
        ledger.literal_if_calls[handle] = replace(attachment)
    elif fault == "foreign":
        other = _source(_target(1, "rich"))
        _validated(other)
        ledger.literal_if_calls[handle] = _attachment(other)[2]
    elif fault == "missing":
        del ledger.literal_if_calls[handle]
    else:
        field = "values"
        before = attachment.values
        if fault == "selected-drop":
            changed = before[1:]
        elif fault == "selected-swap":
            (a, av), (b, bv), *rest = before
            changed = ((a, bv), (b, av), *rest)
        elif fault == "cross-owner":
            a, av = before[0]
            changed = ((a, replace(av, owner=attachment.false.owner)), *before[1:])
        elif fault == "port":
            a, av = before[0]
            changed = ((a, replace(av, port=av.port + 1)), *before[1:])
        elif fault == "origin":
            a, av = before[0]
            changed = ((a, replace(av, origin=object())), *before[1:])
        elif fault == "inactive-count":
            field, before = "discharge", attachment.discharge
            changed = replace(before, semantically_discharged_equations=1)
        else:
            field, before = "trees", attachment.trees
            changed = (replace(before[0], scopes=before[0].scopes[:-1]), before[1])
        # Exercise the invariant on the authenticated object itself, beyond the
        # distinct control that rejects an unauthenticated dataclass copy.
        object.__setattr__(attachment, field, changed)
        try:
            _rejected(model, expected)
        finally:
            object.__setattr__(attachment, field, before)
        return
    _rejected(model, expected)


@pytest.mark.parametrize(
    "fault",
    (
        "operand",
        "order",
        "capture",
        "extra-node",
        "extra-storage",
        "inactive-marker",
        "inactive-origin",
        "storage",
    ),
)
def test_final_selected_slice_and_inactive_absence_are_independent_checks(
    fault: str,
) -> None:
    model = _source(_target(1, "rich"))
    ledger = _validated(model)
    _, _, attachment = _attachment(model)
    graph = model.graph
    producer = next(node for node in graph if node.op_type == "Sub")
    if fault == "operand":
        producer.replace_input_with(1, producer.inputs[0])
    elif fault == "order":
        consumer = next(
            node
            for node in graph
            if any(value is producer.outputs[0] for value in node.inputs)
        )
        edges = {
            id(node): (
                tuple(id(value) for value in node.inputs),
                tuple(id(value) for value in node.outputs),
            )
            for node in graph
        }
        graph.insert_after(consumer, producer)
        assert list(graph).index(producer) > list(graph).index(consumer)
        assert edges == {
            id(node): (
                tuple(id(value) for value in node.inputs),
                tuple(id(value) for value in node.outputs),
            )
            for node in graph
        }
    elif fault == "capture":
        capture = next(
            node
            for node in graph
            if node.op_type == "Identity" and node.inputs[0] is graph.inputs[0]
        )
        capture.replace_input_with(0, graph.inputs[1])
    elif fault == "extra-node":
        value = ir.Value(
            name="unclaimed",
            type=producer.outputs[0].type,
            shape=producer.outputs[0].shape,
        )
        graph.append(ir.Node("", "Identity", [producer.outputs[0]], outputs=[value]))
    elif fault == "extra-storage":
        value = ir.Value(name="unclaimed_storage", const_value=ir.tensor(_FIRST.copy()))
        graph.initializers[value.name] = value
    elif fault in ("inactive-marker", "inactive-origin"):
        witness = attachment.trees[0].scopes[-1].input_values[0]
        assert witness.owner is attachment.false.owner
        if fault == "inactive-marker":
            producer.outputs[0].meta[structure._VALUES] += (witness,)
        else:
            producer.meta[structure._ORIGIN] = witness.origin
    else:
        value = next(
            value
            for value in _stored_values(model)
            if _storage(value).shape == _FIRST.shape
            and _storage(value).tobytes() == _FIRST.tobytes()
        )
        changed = _FIRST.copy().view(np.uint32)
        changed[0] = np.uint32(0)
        tensor = ir.tensor(changed.view(np.float32))
        node = value.producer()
        if node is None:
            value.const_value = tensor
        else:
            node.attributes["value"] = ir.AttrTensor("value", tensor)
    _rejected(model, ledger.expected_equations)


@pytest.mark.parametrize("kind", ("native", "function", "while", "scan"))
@pytest.mark.parametrize("index", (0, 1))
def test_inline_literal_subtrees_compose_with_existing_outer_regions(
    kind: str, index: int
) -> None:
    selected = _target(index)
    compiled = jax.jit(selected)

    def target(left: Any, right: Any) -> Any:
        if kind == "native":
            return compiled(left, right)
        if kind == "function":
            return _named_true(left, right) if index else _named_false(left, right)
        if kind == "while":
            return jax.lax.while_loop(
                lambda state: state[0] < np.int32(2),
                lambda state: (state[0] + np.int32(1), selected(state[1], right)),
                (np.int32(0), left),
            )[1]
        return jax.lax.scan(
            lambda carry, unused: (selected(carry, right), carry), left, None, length=2
        )

    model = _source(target)
    ledger = _validated(model)
    assert ledger.literal_if_calls and ledger.inline_calls
    assert all(
        len(tree.scopes) >= 3
        for call in ledger.literal_if_calls.values()
        for tree in call.trees
    )


@pytest.mark.parametrize(
    "fault", ("bare", "authentic", "copied-ledger", "foreign-ledger", "copied-inline")
)
def test_real_clones_require_the_original_authenticated_conversion_ledger(
    fault: str,
) -> None:
    model = _source(_target(0, "rich"))
    ledger = _validated(model)
    clone = model.clone()
    assert clone.graph is not model.graph
    if fault == "bare":
        assert structure._LEDGER not in clone.meta
        assert _proof(clone).reasons == ("source-ledger-unavailable",)
        return
    clone.meta.update(model.meta)
    if fault == "authentic":
        assert _validated(clone) is ledger
        return
    if fault == "copied-ledger":
        from copy import copy

        clone.meta[structure._LEDGER] = copy(ledger)
    elif fault == "foreign-ledger":
        other = _source(_target(0, "rich"))
        _validated(other)
        clone.meta[structure._LEDGER] = other.meta[structure._LEDGER]
    else:
        handle, call = next(iter(ledger.inline_calls.items()))
        ledger.inline_calls[handle] = replace(call)
    _rejected(clone, ledger.expected_equations)


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("optimization", ("disabled", "all"))
@pytest.mark.parametrize("kind", ("nested", "aliases", "literals", "literal-argument"))
def test_original_eager_fresh_jit_and_saved_target_samples_remain_independent(
    index: int, optimization: str, kind: str
) -> None:
    target = _target(index, kind)
    model = _source(target)
    _validated(model)
    inputs = (np.asarray([0.5, -2.0], np.float32), np.asarray([0.25, 1.0], np.float32))
    with jax.enable_x64(True):
        eager = [
            np.asarray(value) for value in jax.tree_util.tree_leaves(target(*inputs))
        ]
        fresh = [
            np.asarray(value)
            for value in jax.tree_util.tree_leaves(jax.jit(target)(*inputs))
        ]
    assert len(eager) == len(fresh)
    for left, right in zip(eager, fresh, strict=True):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes()
    for candidate in (model, ir.from_proto(ir.to_proto(model))):
        report = validate_source_samples(
            candidate,
            samples=[SourceSample(inputs, eager, SampleCriteria(mode="bitwise"))],
            optimization=optimization,
        )
        report.require_sample_acceptance()


def test_one_reused_definition_on_off_on_preserves_proto_reports_calls_and_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def child(left: Any, right: Any) -> Any:
        calls.append(bool(jax.config.x64_enabled))
        return left - right

    compiled = jax.jit(child)

    def target(left: Any, right: Any) -> Any:
        return jax.lax.cond(True, compiled, lambda x, y: compiled(y, x), left, right)

    deltas = []
    exports = []
    original_require = ir_context.IRContext.require_value_for_var
    requires = []

    def require(ctx: Any, var: Any, *args: Any, **kwargs: Any) -> Any:
        requires.append(None)
        return original_require(ctx, var, *args, **kwargs)

    monkeypatch.setattr(ir_context.IRContext, "require_value_for_var", require)
    specs = _specs()
    with jax.enable_x64(True):
        settings = (
            bool(jax.config.x64_enabled),
            jax.config.jax_default_matmul_precision,
        )
        for observe in (True, False, True):
            before_calls, before_requires = len(calls), len(requires)
            with monkeypatch.context() as patch:
                if not observe:
                    patch.setattr(ir_context, "SourceStructureLedger", lambda: None)
                model = ui._to_onnx(
                    target,
                    specs,
                    precision_policy=PrecisionPolicy.SOURCE,
                    return_mode="ir",
                    opset=23,
                )
            assert isinstance(model, ir.Model)
            if observe:
                _validated(model)
            else:
                assert _proof(model).reasons == ("source-ledger-unavailable",)
            deltas.append((len(calls) - before_calls, len(requires) - before_requires))
            exports.append(model)
        assert calls and all(calls)
        assert deltas[1] == deltas[2]
        assert len({item[1] for item in deltas}) == 1
        assert (
            bool(jax.config.x64_enabled),
            jax.config.jax_default_matmul_precision,
        ) == settings
    assert (
        len(
            {
                ir.to_proto(model).SerializeToString(deterministic=True)
                for model in exports
            }
        )
        == 1
    )
    reports = [get_precision_report(model) for model in exports]
    assert all(report is not None for report in reports)
    assert len({precision_report_to_json(report) for report in reports}) == 1
    contexts = [model.meta.get(cast_context.CONTEXT_KEY) for model in exports]
    assert all(
        context.original_lowering == contexts[0].original_lowering
        for context in contexts
        if context is not None
    )


@pytest.mark.parametrize("index", (0, 1))
def test_freeze_attachment_and_final_audit_add_no_native_source_or_materialization_calls(
    index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def child(left: Any, right: Any) -> Any:
        calls.append(None)
        return left - right

    compiled = jax.jit(child)

    def target(left: Any, right: Any) -> Any:
        return jax.lax.cond(
            bool(index), compiled, lambda x, y: compiled(y, x), left, right
        )

    with jax.enable_x64(True):
        native = jax.make_jaxpr(target)(*_specs())
    model = _source(target)
    ledger = _validated(model)
    before_calls = tuple(calls)
    before_proto = ir.to_proto(model).SerializeToString(deterministic=True)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("pure inline literal observation executed or materialized source")

    monkeypatch.setattr(jax, "make_jaxpr", forbidden)
    monkeypatch.setattr(jax, "eval_shape", forbidden)
    monkeypatch.setattr(ir_context.IRContext, "require_value_for_var", forbidden)
    monkeypatch.setattr(ir_context.IRContext, "_bind_literal_value_for_var", forbidden)
    for _ in range(2):
        frozen = regions.freeze_closed_body(native)
        assert frozen.expected_equations == ledger.expected_equations
        assert _validated(model) is ledger
    assert tuple(calls) == before_calls
    assert ir.to_proto(model).SerializeToString(deterministic=True) == before_proto


@pytest.mark.parametrize(
    "fault",
    ("effect", "foreign-primitive", "body-thunk", "compiler-options", "donation"),
)
def test_passive_inline_schema_rejections_never_execute_opaque_bodies(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target(1)
    model = _source(target)
    ledger = _validated(model)
    with jax.enable_x64(True):
        native = jax.make_jaxpr(target)(*_specs())
    branch = native.jaxpr.eqns[-1].params["branches"][1].jaxpr
    equation = next(
        equation
        for equation in branch.eqns
        if equation.primitive is native_jit_primitive()
    )
    params = dict(equation.params)
    changed = equation
    if fault == "effect":
        changed = equation.replace(effects=frozenset({object()}))
    elif fault == "foreign-primitive":
        changed = equation.replace(primitive=jax.lax.add_p)
    elif fault == "body-thunk":

        def thunk() -> Any:
            pytest.fail("opaque original body was evaluated")

        params["call_jaxpr_thunk"] = thunk
    elif fault == "compiler-options":
        params["compiler_options_kvs"] = (("opaque", True),)
    else:
        params["donated_invars"] = (True, *params["donated_invars"][1:])
    if fault not in ("effect", "foreign-primitive"):
        changed = changed.replace(params=params)
    monkeypatch.setattr(
        jax,
        "make_jaxpr",
        lambda *args, **kwargs: pytest.fail("passive schema check retraced"),
    )
    monkeypatch.setattr(
        jax,
        "eval_shape",
        lambda *args, **kwargs: pytest.fail("passive schema check executed"),
    )
    assert inline.native_inline_body(changed) is None
    assert _validated(model) is ledger


@pytest.mark.parametrize("family", ("function", "while", "scan", "conditional"))
def test_noninline_nested_families_keep_the_original_explicit_unfinished_boundary(
    family: str,
) -> None:
    def inactive(left: Any, right: Any) -> Any:
        if family == "function":
            return _named_false(left, right)
        if family == "while":
            return jax.lax.while_loop(
                lambda state: state[0] < np.int32(1),
                lambda state: (state[0] + np.int32(1), state[1] - right),
                (np.int32(0), left),
            )[1]
        if family == "scan":
            return jax.lax.scan(
                lambda carry, unused: (carry - right, carry), left, None, length=1
            )[0]
        return jax.lax.cond(True, lambda x, y: x - y, lambda x, y: y - x, left, right)

    def target(left: Any, right: Any) -> Any:
        return jax.lax.cond(True, _jitted_difference, inactive, left, right)

    model = _source(target)
    proof = _proof(model)
    assert proof.status == "unverified", proof
    assert proof.expected_equations is not None and proof.expected_equations > 3
    assert any(
        "literal-if-childless-contract-unavailable" in reason
        for reason in proof.reasons
    )


def test_permanent_tree_evidence_retains_no_graph_tensor_context_or_native_program() -> (
    None
):
    model = _source(_target(1, "rich"))
    ledger = _validated(model)
    _, _, attachment = _attachment(model)

    def inspect(value: Any) -> None:
        assert not isinstance(
            value,
            (ir.Graph, ir.Node, ir.Value, np.ndarray, jax.Array, ClosedJaxpr, Jaxpr),
        )
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                inspect(getattr(value, field.name))
        elif isinstance(value, (tuple, list, set, frozenset)):
            for child in value:
                inspect(child)
        elif isinstance(value, dict):
            for key, child in value.items():
                inspect(key)
                inspect(child)

    inspect(attachment)
    assert controls.authentic_literal_if_attachment(ledger, attachment)


def test_entirely_dead_unused_capture_retains_the_explicit_liveness_obligation() -> (
    None
):
    def target(left: Any, right: Any) -> Any:
        return jax.lax.cond(
            True, _jitted_unused, lambda x, y: _jitted_unused(x, y), left, right
        )

    with jax.enable_x64(True):
        native = jax.make_jaxpr(target)(*_specs())
    declaration = regions.declare_body(native)
    model = _source(target)
    proof = _proof(model)
    assert proof.expected_equations == declaration.expected_equations == 3
    assert proof.status == "unverified", proof
    assert "inline-caller-input-witness-lost" in proof.reasons
    assert proof.runtime == proof.numerics == "unverified"
    # This is remaining source liveness/transport implementation work. The
    # unused Identity capture has no registered final alias; it is no ONNX limit.


def test_existing_inactive_native_cast_key_adds_no_selected_kernel_authority() -> None:
    def target(left: Any, right: Any) -> Any:
        return jax.lax.cond(
            True, lambda x, y: y, lambda x, y: x.astype(np.float16), left, right
        )

    specs = [
        jax.ShapeDtypeStruct((2,), np.float64),
        jax.ShapeDtypeStruct((2,), np.float16),
    ]
    with jax.enable_x64(True):
        native = jax.make_jaxpr(target)(*specs)
        model = ui._to_onnx(
            target,
            specs,
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
            opset=23,
        )
    assert isinstance(model, ir.Model)
    ledger = _validated(model)
    _, _, attachment = _attachment(model)
    declaration = regions.declare_body(native)
    assert ledger.expected_equations == declaration.expected_equations == 2
    assert attachment.false.body.equations[0].semantics[0] == "native-f64-f16-convert"
    assert attachment.false.equations[0].rule is None
    assert attachment.discharge.graph_matched_equations == 0
    assert attachment.discharge.semantically_discharged_equations == 2
    assert not ledger.cast_plans and not ledger.cast_selections
    assert not any(node.op_type == "Cast" for graph in _graphs(model) for node in graph)
