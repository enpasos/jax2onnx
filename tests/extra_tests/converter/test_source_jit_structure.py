# tests/extra_tests/converter/test_source_jit_structure.py

"""Original native JIT invocations and their actual final inlined graph slices.

The expected denominator comes from the original native body, before dispatch.
A validated structure result remains independent of runtime/sample acceptance.
"""

from collections.abc import Iterator
from copy import copy
from dataclasses import fields, is_dataclass, replace
from typing import Any

import jax
from jax import export as jax_export
from jax.experimental.pjit import pjit
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Primitive, Var
from jax2onnx.converter import _source_regions as regions
from jax2onnx.converter import ir_context, source_structure as structure
from jax2onnx.converter import _source_cast_context as cast_context
from jax2onnx.converter import _source_cast_authority as cast_authority
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.ir_utils import iter_ir_functions
from jax2onnx.plugins import plugin_system
from jax2onnx.plugins.jax.core._source_inline_structure import InlineRule
from jax2onnx.plugins.jax.core.jit import JitPlugin
from jax2onnx.precision import (
    SampleCriteria,
    SourceSample,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)


def _difference(left: Any, right: Any) -> Any:
    return jax.lax.sub(left, right)


_jitted_difference = jax.jit(_difference)
_jitted_chain = jax.jit(lambda left, right: (left - right) * left)


@onnx_function
def _inline_function(left: Any, right: Any) -> Any:
    return _jitted_difference(left, right)


@onnx_function
def _inline_chain_function(left: Any, right: Any) -> Any:
    return _jitted_chain(left, right)


@onnx_function
def _function_difference(left: Any, right: Any) -> Any:
    return jax.lax.sub(left, right)


@onnx_function
def _inline_nested_function(left: Any, right: Any) -> Any:
    return _inline_function(left, right), _inline_function(right, left)


@pytest.fixture(autouse=True)
def _module_function_bindings() -> Iterator[None]:
    # Both registries must be restored: another owner may reset one registry
    # between collection and this test's first decorated Function invocation.
    targets = (
        _inline_function,
        _inline_chain_function,
        _function_difference,
        _inline_nested_function,
    )
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


def _specs(dtype: Any = np.float32, shape: tuple[int, ...] = (2, 3)) -> list[Any]:
    return [jax.ShapeDtypeStruct(shape, dtype)] * 2


def _source(
    function: Any = _jitted_difference,
    specs: list[Any] | None = None,
    **kwargs: Any,
) -> ir.Model:
    with jax.enable_x64(True):
        model = ui._to_onnx(
            function,
            _specs() if specs is None else specs,
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
            **kwargs,
        )
    assert isinstance(model, ir.Model)
    return model


def _native(function: Any, specs: list[Any] | None = None) -> Any:
    with jax.enable_x64(True):
        return jax.make_jaxpr(function)(*(_specs() if specs is None else specs))


def _proof(model: ir.Model) -> structure.StructureResult:
    return structure.validate_source_structure(model)


def _validated(model: ir.Model, expected: int | None = None) -> Any:
    proof = _proof(model)
    assert proof.status == "validated", proof
    assert proof.expected_equations == proof.checked_equations
    if expected is not None:
        assert proof.expected_equations == expected
    assert proof.runtime == proof.numerics == "unverified"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    return model.meta[structure._LEDGER]


def _graphs(model: ir.Model) -> list[ir.Graph]:
    graphs: list[ir.Graph] = []

    def visit(graph: ir.Graph) -> None:
        graphs.append(graph)
        for node in graph:
            for attribute in node.attributes.values():
                if attribute.type is ir.AttributeType.GRAPH:
                    visit(attribute.as_graph())
                elif attribute.type is ir.AttributeType.GRAPHS:
                    for child in attribute.as_graphs():
                        visit(child)

    visit(model.graph)
    for definition in iter_ir_functions(model.functions):
        visit(definition.graph)
    return graphs


def _inline_scopes(ledger: Any) -> list[Any]:
    return [
        scope
        for scope in ledger.scopes
        if scope.region is not None and scope.region.key.role == "inline-body"
    ]


def _storage(value: ir.Value) -> Any:
    producer = value.producer()
    if producer is None:
        assert value.is_initializer() and value.const_value is not None
        return value.const_value.numpy()
    assert producer.op_type == "Constant"
    return producer.attributes["value"].as_tensor().numpy()


def _replace_storage(value: ir.Value, array: Any) -> None:
    producer = value.producer()
    if producer is None:
        assert value.is_initializer()
        value.const_value = ir.tensor(array)
    else:
        assert producer.op_type == "Constant"
        producer.attributes["value"] = ir.AttrTensor("value", ir.tensor(array))


def _stored_values(model: ir.Model) -> list[ir.Value]:
    return [
        value
        for graph in _graphs(model)
        for value in (
            *graph.initializers.values(),
            *(
                out
                for node in graph
                if node.op_type == "Constant"
                for out in node.outputs
            ),
        )
    ]


def _same_type(actual: ir.Value, replacement: ir.Value) -> None:
    assert actual is not replacement
    assert actual.dtype is replacement.dtype
    assert actual.shape == replacement.shape


def _samples(
    model: ir.Model, function: Any, inputs: tuple[Any, ...], optimization: str
) -> None:
    with jax.enable_x64(True):
        expected = [
            np.asarray(value) for value in jax.tree_util.tree_leaves(function(*inputs))
        ]
    report = validate_source_samples(
        model,
        samples=[SourceSample(inputs, expected, SampleCriteria("bitwise"))],
        optimization=optimization,
    )
    report.require_sample_acceptance()
    assert _proof(model).runtime == _proof(model).numerics == "unverified"


@pytest.mark.parametrize("route", ("jit", "pjit"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("shape", ((), (2, 3), (0, 3)))
@pytest.mark.parametrize("opset", (18, 23, 26))
def test_real_native_jit_and_pjit_have_one_logical_child_in_the_parent_graph(
    route: str, dtype: Any, shape: tuple[int, ...], opset: int
) -> None:
    empty = 0 in shape
    # Direct arithmetic over empty outputs has a separate structural gap.
    # The empty transport witness is the genuine zero-equation identity body.
    body = (
        (lambda left, right: left)
        if empty and not (opset == 18 and dtype is np.float64)
        else _difference
    )
    function = (jax.jit if route == "jit" else pjit)(body)
    specs = _specs(dtype, shape)
    if opset == 18 and dtype is np.float64:
        with pytest.raises(
            SourcePrecisionError,
            match="dtype is outside the original opset18/21 typed expression",
        ) as error:
            _source(function, specs, opset=opset)
        assert error.value.category == "profile-exclusion"
        return
    original = _native(function, specs)
    declared = regions.declare_body(original)
    frozen = regions.freeze_closed_body(original)
    expected = 1 if empty else 2
    assert declared.expected_equations == frozen.expected_equations == expected
    assert frozen.gaps == ()
    assert tuple(role for role, _ in frozen.equations[0].children) == ("inline-body",)
    ledger = _validated(_source(function, specs, opset=opset), expected)
    assert len(ledger.inline_calls) == len(ledger.regions) == 1
    assert len(ledger.scopes) == 2
    root, child = ledger.scopes
    assert child.owner is root.owner
    assert child.region.key.role == "inline-body"
    assert child.body == frozen.equations[0].children[0][1]
    assert len(child.equations) == expected - 1
    assert type(root.equations[0].rule) is InlineRule
    assert not ledger.pending_regions and not ledger.active and not ledger.handles


@pytest.mark.parametrize("inline", (False, True))
@pytest.mark.parametrize("keep_unused", (False, True))
def test_native_hints_and_unused_original_formals_are_preserved(
    inline: bool, keep_unused: bool
) -> None:
    function = jax.jit(
        lambda left, right: -left, inline=inline, keep_unused=keep_unused
    )
    original = regions.freeze_closed_body(_native(function))
    model = _source(function)
    ledger = _validated(model, original.expected_equations)
    # A native inline=True hint may erase the wrapper before original freezing.
    # The audit must follow that actual native tree, without inventing a call.
    wrappers = [equation for equation in original.equations if equation.children]
    assert len(ledger.inline_calls) == len(wrappers)
    if wrappers:
        child = _inline_scopes(ledger)[0]
        rule = ledger.scopes[0].equations[0].rule
        assert type(rule) is InlineRule and rule.keep_unused is keep_unused
        assert child.body == wrappers[0].children[0][1]
        assert len(child.inputs) == 2
        assert structure._unused_inputs(child.body) == frozenset({1})
    inputs = (np.arange(6, dtype=np.float32).reshape(2, 3), np.ones((2, 3), np.float32))
    _samples(model, function, inputs, "disabled")


@pytest.mark.parametrize("kind", ("reused", "closures", "siblings", "nested"))
def test_invocations_share_a_physical_owner_without_sharing_logical_receipts(
    kind: str,
) -> None:
    first = jax.jit(_difference)
    captured = np.asarray([[0.25, 0.5, 0.75], [1, 1.25, 1.5]], np.float32)
    left_closure = jax.jit(lambda x: x - captured)
    right_closure = jax.jit(lambda x: x + captured)
    outer = jax.jit(lambda x, y: first(x, y) * y)

    def target(left: Any, right: Any) -> Any:
        if kind == "reused":
            return first(left, right), first(right, left)
        if kind == "closures":
            return left_closure(left), right_closure(right)
        if kind == "siblings":
            before = left + right
            middle = first(before, right)
            return middle * left
        return outer(left, right)

    frozen = regions.freeze_closed_body(_native(target))
    model = _source(target)
    ledger = _validated(model, frozen.expected_equations)
    children = _inline_scopes(ledger)
    assert len(children) == len(ledger.inline_calls) >= 1
    assert len({scope.invocation for scope in children}) == len(children)
    assert len({scope.region.key for scope in children}) == len(children)
    assert {scope.owner for scope in children} == {ledger.scopes[0].owner}
    assert len({id(node) for node in model.graph}) == len(tuple(model.graph))
    if kind == "reused":
        assert len(children) == 2 and children[0].body == children[1].body
        assert children[0].region.key.parent != children[1].region.key.parent
        assert children[0].result_values != children[1].result_values
    if kind == "nested":
        assert len(children) == 2
        assert children[1].region.key.parent.invocation is children[0].invocation


@pytest.mark.parametrize(
    "kind", ("identity", "aliases", "repeated", "literal", "closed")
)
def test_zero_equation_alias_repeated_and_literal_results_use_original_order(
    kind: str,
) -> None:
    captured = np.asarray([0x80000000, 0x7FC00001], np.uint32).view(np.float32)
    if kind == "identity":
        function = jax.jit(lambda left, right: left)
    elif kind == "aliases":
        function = jax.jit(lambda left, right: (left, right, left))
    elif kind == "repeated":
        function = jax.jit(lambda left, right: (left - right, left, left - right))
    elif kind == "literal":
        function = jax.jit(lambda left, right: (left, np.float32(-0.0), right))
    else:
        function = jax.jit(lambda left, right: (captured, left, captured))
    specs = _specs(shape=(2,))
    frozen = regions.freeze_closed_body(_native(function, specs))
    model = _source(function, specs)
    ledger = _validated(model, frozen.expected_equations)
    child = _inline_scopes(ledger)[0]
    original_child = frozen.equations[0].children[0][1]
    assert child.body == original_child
    assert len(child.outputs) == len(original_child.outputs)
    assert len(child.result_values) == len(original_child.outputs)
    if kind == "identity":
        assert original_child.expected_equations == 0
        assert not any(
            node.op_type == "Identity"
            for node in model.graph
            if node.outputs[0] not in model.graph.outputs
        )
    if kind == "aliases":
        assert original_child.outputs[0] == original_child.outputs[2]
        assert child.result_values[0] == child.result_values[2]
    if kind == "closed":
        assert len(original_child.constants) == 1
        found = [
            value
            for value in _stored_values(model)
            if _storage(value).tobytes() == captured.tobytes()
        ]
        assert found
    if kind != "closed":
        inputs = (
            np.asarray([1.5, -2], np.float32),
            np.asarray([0.5, 0.25], np.float32),
        )
        _samples(model, function, inputs, "disabled")


@pytest.mark.parametrize(
    "kind",
    (
        "function",
        "reused-function",
        "nested-function",
        "dynamic-if",
        "literal-if",
        "jit-function",
        "while",
        "scan",
        "jit-while",
        "jit-scan",
    ),
)
def test_inline_composition_keeps_child_slices_and_complete_original_invocation_tree(
    kind: str,
) -> None:
    def loop(left: Any, right: Any) -> Any:
        return jax.lax.while_loop(
            lambda state: state[0] < np.int32(2),
            lambda state: (state[0] + np.int32(1), _jitted_difference(state[1], right)),
            (np.int32(0), left),
        )[1]

    def scan(left: Any, right: Any) -> Any:
        return jax.lax.scan(
            lambda carry, unused: (_jitted_difference(carry, right), carry),
            left,
            None,
            length=2,
        )

    jitted_function = jax.jit(lambda left, right: _function_difference(left, right))
    jitted_loop = jax.jit(loop)
    jitted_scan = jax.jit(scan)

    def target(left: Any, right: Any, flag: Any) -> Any:
        if kind == "function":
            return _inline_function(left, right)
        if kind == "reused-function":
            return _inline_function(left, right), _inline_function(right, left)
        if kind == "nested-function":
            return _inline_nested_function(left, right)
        if kind in ("dynamic-if", "literal-if"):
            return jax.lax.cond(
                flag if kind == "dynamic-if" else np.asarray(True),
                lambda x, y: _jitted_difference(x, y),
                lambda x, y: _jitted_difference(y, x),
                left,
                right,
            )
        if kind == "jit-function":
            return jitted_function(left, right)
        if kind in ("while", "jit-while"):
            return (jitted_loop if kind == "jit-while" else loop)(left, right)
        return (jitted_scan if kind == "jit-scan" else scan)(left, right)

    specs = _specs() + [jax.ShapeDtypeStruct((), np.bool_)]
    model = _source(target, specs)
    ledger = _validated(model)
    children = _inline_scopes(ledger)
    assert len(children) == len(ledger.inline_calls) >= 1
    assert len({scope.invocation for scope in children}) == len(children)
    for child in children:
        parent = next(
            scope
            for scope in ledger.scopes
            if scope.invocation is child.region.key.parent.invocation
        )
        assert child.owner is parent.owner
        assert (
            child.body
            == parent.body.equations[child.region.key.parent.index].children[0][1]
        )
    if kind == "reused-function":
        assert len(list(iter_ir_functions(model.functions))) == 1
        assert len(ledger.function_calls) == 2 and len(children) == 1
        assert _proof(model).expected_equations == 6
    for flag in (False, True):
        inputs = (
            np.ones((2, 3), np.float32),
            np.full((2, 3), 0.25, np.float32),
            np.asarray(flag),
        )
        _samples(model, target, inputs, "disabled")


@pytest.mark.parametrize(
    "fault",
    (
        "operator",
        "operand-order",
        "same-typed-input",
        "result-order",
        "bypass",
        "shape",
        "dtype",
    ),
)
def test_final_graph_mutations_preserve_signatures_but_fail_the_original_body(
    fault: str,
) -> None:
    function = jax.jit(lambda left, right: (left - right, left + right))
    model = _source(function)
    _validated(model, 3)
    sub = next(node for node in model.graph if node.op_type == "Sub")
    if fault == "operator":
        sub.op_type = "Add"
    elif fault == "operand-order":
        first, second = sub.inputs
        _same_type(first, second)
        sub.replace_input_with(0, second)
        sub.replace_input_with(1, first)
    elif fault == "same-typed-input":
        _same_type(sub.inputs[0], sub.inputs[1])
        sub.replace_input_with(0, sub.inputs[1])
    elif fault == "result-order":
        first, second = model.graph.outputs
        _same_type(first, second)
        model.graph.outputs[0], model.graph.outputs[1] = second, first
    elif fault == "bypass":
        _same_type(model.graph.outputs[0], model.graph.inputs[0])
        model.graph.outputs[0] = model.graph.inputs[0]
    elif fault == "shape":
        sub.outputs[0].shape = ir.Shape((3, 2))
    else:
        sub.outputs[0].type = ir.TensorType(ir.DataType.DOUBLE)
    proof = _proof(model)
    assert proof.status == "failed", proof
    assert proof.expected_equations == 3 and proof.checked_equations < 3


@pytest.mark.parametrize("scope", ("root", "function", "if"))
def test_public_producer_reorder_keeps_edge_identities_and_fails_visibility(
    scope: str,
) -> None:
    chained = jax.jit(lambda left, right: (left - right) * left)

    def target(left: Any, right: Any, flag: Any) -> Any:
        if scope == "root":
            return chained(left, right)
        if scope == "function":
            return _inline_chain_function(left, right)
        return jax.lax.cond(
            flag, lambda x, y: chained(x, y), lambda x, y: chained(y, x), left, right
        )

    model = _source(target, _specs() + [jax.ShapeDtypeStruct((), np.bool_)])
    ledger = _validated(model)
    if scope == "function":
        graph = next(
            definition.graph
            for definition in iter_ir_functions(model.functions)
            if {node.op_type for node in definition.graph} >= {"Sub", "Mul"}
        )
        assert graph is not model.graph
    elif scope == "if":
        conditional = next(node for node in model.graph if node.op_type == "If")
        graph = next(
            attribute.as_graph()
            for attribute in conditional.attributes.values()
            if attribute.type is ir.AttributeType.GRAPH
            and {node.op_type for node in attribute.as_graph()} >= {"Sub", "Mul"}
        )
    else:
        graph = model.graph
    producer = next(node for node in graph if node.op_type == "Sub")
    consumer = next(
        node
        for node in graph
        if any(value is producer.outputs[0] for value in node.inputs)
    )
    before = {
        id(node): (
            tuple(id(value) for value in node.inputs),
            tuple(id(value) for value in node.outputs),
        )
        for node in graph
    }
    graph.insert_after(consumer, producer)
    assert tuple(graph).index(consumer) < tuple(graph).index(producer)
    assert before == {
        id(node): (
            tuple(id(value) for value in node.inputs),
            tuple(id(value) for value in node.outputs),
        )
        for node in graph
    }
    proof = _proof(model)
    assert proof.status == "failed", proof
    assert proof.expected_equations == ledger.expected_equations


@pytest.mark.parametrize(
    "fault",
    (
        "signed-zero",
        "nan-payload",
        "backing-dtype",
        "backing-shape",
        "declared-dtype",
        "declared-shape",
    ),
)
def test_closed_constant_storage_is_read_from_actual_final_definition(
    fault: str,
) -> None:
    captured = np.asarray([0x80000000, 0x7FC00001], np.uint32).view(np.float32)
    function = jax.jit(lambda left, right: left - captured)
    model = _source(function, _specs(shape=(2,)))
    _validated(model, 2)
    candidates = [
        value
        for value in _stored_values(model)
        if _storage(value).dtype == captured.dtype
        and _storage(value).shape == captured.shape
        and _storage(value).tobytes() == captured.tobytes()
    ]
    assert len(candidates) == 1
    value = candidates[0]
    if fault in ("signed-zero", "nan-payload"):
        changed = captured.copy().view(np.uint32)
        changed[0 if fault == "signed-zero" else 1] = np.uint32(
            0 if fault == "signed-zero" else 0x7FC00002
        )
        _replace_storage(value, changed.view(np.float32))
    elif fault == "backing-dtype":
        _replace_storage(value, captured.astype(np.float64))
    elif fault == "backing-shape":
        _replace_storage(value, captured.reshape(1, 2))
    elif fault == "declared-dtype":
        value.type = ir.TensorType(ir.DataType.DOUBLE)
    else:
        value.shape = ir.Shape((1, 2))
    proof = _proof(model)
    assert proof.status == "failed", proof
    assert proof.expected_equations == 2


@pytest.mark.parametrize("extra", ("node", "storage", "identical-storage"))
def test_shared_physical_union_rejects_unclaimed_nodes_and_storage(extra: str) -> None:
    captured = np.asarray([[0.25, 0.5, 0.75], [1, 1.25, 1.5]], np.float32)
    function = jax.jit(lambda left, right: left - captured)
    model = _source(lambda x, y: (function(x, y), function(y, x)))
    _validated(model, 4)
    if extra == "node":
        output = ir.Value(
            name="unclaimed",
            type=model.graph.inputs[0].type,
            shape=model.graph.inputs[0].shape,
        )
        model.graph.append(
            ir.Node("", "Neg", [model.graph.inputs[0]], outputs=[output])
        )
    else:
        payload = (
            captured.copy()
            if extra == "identical-storage"
            else np.ones((2, 3), np.float32)
        )
        model.graph.register_initializer(
            ir.Value(
                name="unclaimed",
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape((2, 3)),
                const_value=ir.tensor(payload),
            )
        )
    proof = _proof(model)
    assert proof.status == "unverified", proof
    assert proof.expected_equations == proof.checked_equations == 4
    assert (
        "artifact-node-coverage-incomplete"
        if extra == "node"
        else "artifact-constant-coverage-incomplete"
    ) in proof.reasons


@pytest.mark.parametrize(
    "fault",
    (
        "attachment",
        "region",
        "scope",
        "receipt",
        "duplicate",
        "unclaimed",
        "foreign-owner",
        "foreign-parent",
        "reused-attachment",
    ),
)
def test_missing_or_foreign_evidence_keeps_every_original_jit_invocation(
    fault: str,
) -> None:
    model = _source(
        lambda left, right: (
            _jitted_difference(left, right),
            _jitted_difference(right, left),
        )
    )
    ledger = _validated(model, 4)
    child = _inline_scopes(ledger)[-1]
    handle, attachment = next(reversed(ledger.inline_calls.items()))
    if fault == "attachment":
        ledger.inline_calls.pop(handle)
    elif fault == "region":
        ledger.regions.pop(child.region.key)
    elif fault == "scope":
        ledger.scopes.remove(child)
    elif fault == "receipt":
        child.receipts.clear()
    elif fault == "duplicate":
        ledger.scopes.append(replace(child, invocation=object()))
    elif fault == "unclaimed":
        ledger.scopes.append(replace(child, invocation=object(), region=None))
    elif fault == "foreign-owner":
        ledger.inline_calls[handle] = replace(attachment, owner=object())
    elif fault == "foreign-parent":
        ledger.inline_calls[handle] = replace(
            attachment,
            region=replace(
                attachment.region,
                parent=structure.EquationHandle(object(), handle.index),
            ),
        )
    else:
        first = next(iter(ledger.inline_calls.values()))
        assert first.region != attachment.region
        ledger.inline_calls[handle] = first
    proof = _proof(model)
    assert proof.status in ("unverified", "failed"), proof
    assert proof.expected_equations == 4 and proof.checked_equations < 4


@pytest.mark.parametrize("fault", ("source-use-order", "constant", "result-order"))
def test_freshening_must_preserve_the_original_typed_child_before_attachment(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = np.asarray([[0.25, 0.5, 0.75], [1, 1.25, 1.5]], np.float32)
    function = jax.jit(lambda left, right: (left - right, left + captured))
    original_freshen = JitPlugin._freshen_closed_jaxpr

    def corrupted(closed: Any) -> Any:
        fresh = original_freshen(closed)
        body = fresh.jaxpr
        if fault == "source-use-order":
            equations = list(body.eqns)
            first = equations[0]
            equations[0] = first.replace(invars=list(reversed(first.invars)))
            return ClosedJaxpr(body.replace(eqns=equations), fresh.consts)
        if fault == "constant":
            assert len(fresh.consts) == 1
            return ClosedJaxpr(body, [np.asarray(fresh.consts[0]) + np.float32(1)])
        return ClosedJaxpr(
            body.replace(outvars=list(reversed(body.outvars))), fresh.consts
        )

    monkeypatch.setattr(JitPlugin, "_freshen_closed_jaxpr", staticmethod(corrupted))
    model = _source(function)
    proof = _proof(model)
    assert proof.status in ("failed", "unverified"), proof
    assert proof.expected_equations == 3 and proof.checked_equations < 3


@pytest.mark.parametrize(
    "fault",
    (
        "foreign-primitive",
        "unknown-parameter",
        "lazy-body",
        "const-arity",
        "input-arity",
        "output-arity",
        "cycle",
    ),
)
def test_passive_original_declaration_never_calls_metadata_or_claims_an_outer_prefix(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = _native(_jitted_difference)
    equation = native.jaxpr.eqns[0]
    params = dict(equation.params)
    original_body = params["jaxpr"].jaxpr
    original_equations = tuple(original_body.eqns)
    calls: list[None] = []

    def forbidden() -> Any:
        calls.append(None)
        pytest.fail("Passive JIT observation evaluated a body thunk")

    if fault == "foreign-primitive":
        equation = equation.replace(primitive=Primitive(equation.primitive.name))
    elif fault == "unknown-parameter":
        params["unreviewed"] = forbidden
    elif fault == "lazy-body":
        params["call_jaxpr_thunk"] = forbidden
    elif fault == "const-arity":
        closed = copy(params["jaxpr"])
        closed._consts = [np.asarray(1, np.float32)]
        params["jaxpr"] = closed
    elif fault == "input-arity":
        equation = equation.replace(invars=equation.invars[:1])
    elif fault == "output-arity":
        equation = equation.replace(outvars=[])
    else:
        # ClosedJaxpr is a wrapper on the old stack and aliases Jaxpr on the
        # modern stack. Detach the native body before creating the exact cycle.
        # A shallow copy of only the wrapper would poison the cached original.
        closed = ClosedJaxpr(copy(params["jaxpr"].jaxpr), tuple(params["jaxpr"].consts))
        inner = closed.jaxpr
        assert inner is not original_body
        nested = equation.replace(
            invars=inner.invars,
            outvars=inner.outvars,
            params={**params, "jaxpr": closed},
        )
        inner._eqns = [nested]
        params["jaxpr"] = closed
    if fault != "foreign-primitive":
        equation = equation.replace(params=params)
    changed = native.jaxpr.replace(eqns=[equation])
    monkeypatch.setattr(
        jax,
        "make_jaxpr",
        lambda *args, **kwargs: pytest.fail("JIT observer retraced source"),
    )
    monkeypatch.setattr(
        jax,
        "eval_shape",
        lambda *args, **kwargs: pytest.fail("JIT observer evaluated source"),
    )
    declaration = regions.declare_body(changed)
    frozen = regions.freeze_closed_body(changed)
    assert not calls
    assert len(original_body.eqns) == len(original_equations)
    assert all(
        actual is before
        for actual, before in zip(original_body.eqns, original_equations, strict=True)
    )
    assert declaration.expected_equations is None
    assert frozen.expected_equations is None
    assert declaration.gaps and frozen.gaps


@pytest.mark.parametrize("weak", (False, True))
def test_original_ports_weak_types_literal_bits_and_hashes_are_immutable(
    weak: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = np.asarray([0x80000000, 0x7FC00001], np.uint32).view(np.float32)
    function = jax.jit(lambda left, right: (left - captured, right, np.float32(-0.0)))
    native = _native(
        function,
        [
            jax.ShapeDtypeStruct((2,), np.float32),
            jax.ShapeDtypeStruct((), np.float32, weak_type=weak),
        ],
    )
    monkeypatch.setattr(
        jax,
        "make_jaxpr",
        lambda *args, **kwargs: pytest.fail("JIT observer retraced source"),
    )
    monkeypatch.setattr(
        jax,
        "eval_shape",
        lambda *args, **kwargs: pytest.fail("JIT observer evaluated source"),
    )
    frozen = regions.freeze_closed_body(native)
    assert frozen.gaps == () and frozen.expected_equations == 2
    child = frozen.equations[0].children[0][1]
    assert child.inputs[1].weak_type is weak
    assert child.outputs[1].tensor.weak_type is weak
    assert child.outputs[2].literal is not None
    assert (
        child.outputs[2].literal.sha256
        == structure.ConstantContract.from_array(np.asarray(-0.0, np.float32)).sha256
    )
    before = hash(frozen)
    captured.view(np.uint32)[0] = 0
    assert hash(frozen) == before
    assert regions.freeze_closed_body(native) != frozen


@pytest.mark.parametrize("fault", ("owner", "witness", "body-owner"))
def test_clones_survive_real_copying_but_reject_owner_and_witness_corruption(
    fault: str,
) -> None:
    original = _source(lambda left, right: _inline_nested_function(left, right))
    ledger = _validated(original)
    cloned = original.clone()
    cloned.meta.update(original.meta)
    assert cloned.graph is not original.graph
    assert _validated(cloned, ledger.expected_equations) is ledger
    if fault == "owner":
        cloned.graph.meta[structure._OWNER] = object()
    else:
        definition = next(
            definition
            for definition in iter_ir_functions(cloned.functions)
            if any(node.op_type == "Sub" for node in definition.graph)
        )
        assert all(
            definition.graph is not old.graph
            for old in iter_ir_functions(original.functions)
        )
        if fault == "body-owner":
            definition.graph.meta[structure._OWNER] = object()
        else:
            output = next(
                node.outputs[0] for node in definition.graph if node.op_type == "Sub"
            )
            assert structure._VALUES in output.meta
            output.meta.pop(structure._VALUES)
    proof = _proof(cloned)
    assert proof.status == "unverified", proof
    assert proof.expected_equations == ledger.expected_equations
    assert _proof(original).status == "validated"


def test_inline_ledger_retains_no_native_array_or_ir_authority() -> None:
    ledger = _validated(_source(), 2)

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
            for key, child in value.items():
                inspect(key)
                inspect(child)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for child in value:
                inspect(child)

    inspect(vars(ledger))
    assert ledger.active == ledger.handles == ledger.pending_regions == []


@pytest.mark.parametrize("optimization", ("disabled", "all"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_native_numerics_and_serialized_reload_remain_independent(
    dtype: Any, optimization: str
) -> None:
    function = jax.jit(lambda left, right: ((left - right) * right, -left))
    model = _source(function, _specs(dtype))
    _validated(model, 4)
    inputs = (np.arange(6, dtype=dtype).reshape(2, 3), np.full((2, 3), 0.25, dtype))
    _samples(model, function, inputs, optimization)
    proto = onnx.load_model_from_string(ir.to_proto(model).SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    reloaded = ir.from_proto(proto)
    assert isinstance(reloaded, ir.Model)
    _samples(reloaded, function, inputs, optimization)


@pytest.mark.parametrize("source_first", (False, True))
def test_observation_on_off_preserves_proto_reports_original_lowering_and_legacy(
    source_first: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = jax.jit(lambda left, right: (left - right) * right)
    with jax.enable_x64(True):
        reference = ui._to_onnx(function, _specs(), return_mode="ir")
    if not source_first:
        with jax.enable_x64(True):
            ui._to_onnx(function, _specs(), return_mode="ir")
    enabled = _source(function)
    _validated(enabled, 3)
    with monkeypatch.context() as patch:
        patch.setattr(ir_context, "SourceStructureLedger", lambda: None)
        disabled = _source(function)
    assert _proof(disabled).reasons == ("source-ledger-unavailable",)
    assert ir.to_proto(enabled).SerializeToString(deterministic=True) == ir.to_proto(
        disabled
    ).SerializeToString(deterministic=True)
    enabled_report, disabled_report = get_precision_report(
        enabled
    ), get_precision_report(disabled)
    assert enabled_report is not None and disabled_report is not None
    assert precision_report_to_json(enabled_report) == precision_report_to_json(
        disabled_report
    )
    enabled_context = enabled.meta.get(cast_context.CONTEXT_KEY)
    disabled_context = disabled.meta.get(cast_context.CONTEXT_KEY)
    assert (None if enabled_context is None else enabled_context.original_lowering) == (
        None if disabled_context is None else disabled_context.original_lowering
    )
    with jax.enable_x64(True):
        legacy = ui._to_onnx(function, _specs(), return_mode="ir")
    assert isinstance(legacy, ir.Model) and isinstance(reference, ir.Model)
    assert ir.to_proto(legacy).SerializeToString(deterministic=True) == ir.to_proto(
        reference
    ).SerializeToString(deterministic=True)


@pytest.mark.parametrize("ambient_x64", (False, True))
@pytest.mark.parametrize("local_x64", (False, True))
def test_observation_preserves_call_site_settings_and_source_invocation_count(
    ambient_x64: bool, local_x64: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocations: list[tuple[Any, ...]] = []

    def child(left: Any, right: Any) -> Any:
        invocations.append(
            (
                bool(jax.config.x64_enabled),
                jax.config.jax_default_matmul_precision,
                active_precision_policy(),
            )
        )
        return jax.lax.sub(left, right)

    compiled = jax.jit(child)

    def target(left: Any, right: Any) -> Any:
        with jax.enable_x64(local_x64):
            return compiled(left, right)

    with jax.enable_x64(ambient_x64), jax.default_matmul_precision("highest"):
        before = (
            bool(jax.config.x64_enabled),
            jax.config.jax_default_matmul_precision,
            active_precision_policy(),
        )
        model = ui._to_onnx(
            target, _specs(), precision_policy=PrecisionPolicy.SOURCE, return_mode="ir"
        )
        assert isinstance(model, ir.Model)
        ledger = _validated(model)
        observed = tuple(invocations)
        assert observed and all(value[0] is local_x64 for value in observed)
        monkeypatch.setattr(
            jax,
            "make_jaxpr",
            lambda *args, **kwargs: pytest.fail("Final JIT audit retraced source"),
        )
        monkeypatch.setattr(
            jax,
            "eval_shape",
            lambda *args, **kwargs: pytest.fail("Final JIT audit evaluated source"),
        )
        for _ in range(2):
            assert _validated(model, ledger.expected_equations) is ledger
        assert tuple(invocations) == observed
        assert (
            bool(jax.config.x64_enabled),
            jax.config.jax_default_matmul_precision,
            active_precision_policy(),
        ) == before


@pytest.mark.parametrize("kind", ("unsupported-leaf", "symbolic", "specialized-prng"))
def test_explicit_unfinished_boundaries_never_become_a_false_validated_inline_proof(
    kind: str,
) -> None:
    if kind == "unsupported-leaf":
        # Tanh has numerical admission but needs its own structural composition.
        function = jax.jit(lambda left, right: jax.lax.tanh(left))
        specs = _specs()
    elif kind == "symbolic":
        (batch,) = jax_export.symbolic_shape("B")
        function = jax.jit(_difference)
        specs = _specs(shape=(batch, 3))
    else:

        def function(left: Any, words: Any) -> Any:
            return jax.random.uniform(
                jax.random.wrap_key_data(words, impl="threefry2x32"),
                left.shape,
                dtype=np.float32,
            )

        specs = [
            jax.ShapeDtypeStruct((2, 3), np.float32),
            jax.ShapeDtypeStruct((2,), np.uint32),
        ]
    native = _native(function, specs)
    declaration = regions.declare_body(native)
    frozen = regions.freeze_closed_body(native)
    model = _source(function, specs)
    ledger = model.meta[structure._LEDGER]
    proof = _proof(model)
    assert proof.status == "unverified", proof
    assert proof.checked_equations < (proof.expected_equations or 1)
    if kind == "symbolic":
        assert "cross-region-symbolic-contract-unavailable" in frozen.gaps
        assert declaration.expected_equations is not None
    else:
        assert (
            proof.expected_equations
            == declaration.expected_equations
            == frozen.expected_equations
        )
        assert proof.expected_equations is not None and proof.expected_equations > 1
    if kind == "specialized-prng":
        assert not ledger.inline_calls
        assert not _inline_scopes(ledger)


@pytest.mark.parametrize("fault", ("storage", "runtime-substitute", "result-swap"))
def test_literal_result_uses_its_original_typed_storage_and_order(fault: str) -> None:
    function = jax.jit(lambda left, right: (left - right, np.float32(-0.0)))
    model = _source(function, _specs(shape=()))
    _validated(model, 2)
    if fault == "storage":
        candidates = [
            value
            for value in _stored_values(model)
            if _storage(value).dtype == np.dtype(np.float32)
            and _storage(value).shape == ()
            and np.signbit(_storage(value)).item()
        ]
        assert len(candidates) == 1
        _replace_storage(candidates[0], np.asarray(0.0, np.float32))
    elif fault == "runtime-substitute":
        _same_type(model.graph.outputs[1], model.graph.inputs[1])
        model.graph.outputs[1] = model.graph.inputs[1]
    else:
        first, second = model.graph.outputs
        _same_type(first, second)
        model.graph.outputs[0], model.graph.outputs[1] = second, first
    proof = _proof(model)
    assert proof.status == "failed", proof
    assert proof.expected_equations == 2


def test_lexical_capture_cannot_be_moved_after_the_if_that_consumes_it() -> None:
    def target(left: Any, right: Any, flag: Any) -> Any:
        captured = left - right
        return jax.lax.cond(
            flag,
            lambda value: _jitted_difference(value, captured),
            lambda value: _jitted_difference(captured, value),
            left,
        )

    model = _source(target, _specs() + [jax.ShapeDtypeStruct((), np.bool_)])
    ledger = _validated(model)
    owner = model.graph
    producer = next(node for node in owner if node.op_type == "Sub")
    consumer = next(node for node in owner if node.op_type == "If")
    branches = [
        attribute.as_graph()
        for attribute in consumer.attributes.values()
        if attribute.type is ir.AttributeType.GRAPH
    ]
    assert any(
        value is producer.outputs[0]
        for branch in branches
        for node in branch
        for value in node.inputs
    )
    before = tuple(producer.outputs)
    owner.insert_after(consumer, producer)
    assert tuple(producer.outputs) == before
    proof = _proof(model)
    assert proof.status == "failed", proof
    assert proof.expected_equations == ledger.expected_equations


def test_declaration_and_freeze_add_zero_source_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[None] = []

    def child(left: Any, right: Any) -> Any:
        calls.append(None)
        return jax.lax.sub(left, right)

    function = jax.jit(child)
    native = _native(function)
    before = tuple(calls)
    assert before
    monkeypatch.setattr(
        jax,
        "make_jaxpr",
        lambda *args, **kwargs: pytest.fail("JIT observation retraced source"),
    )
    monkeypatch.setattr(
        jax,
        "eval_shape",
        lambda *args, **kwargs: pytest.fail("JIT observation evaluated source"),
    )
    declaration = regions.declare_body(native)
    first = regions.freeze_closed_body(native)
    second = regions.freeze_closed_body(native)
    assert (
        declaration.expected_equations
        == first.expected_equations
        == second.expected_equations
        == 2
    )
    assert first == second and hash(first) == hash(second)
    assert tuple(calls) == before


@pytest.mark.parametrize("observe", (False, True))
def test_scoped_cast_has_no_new_root_native_kernel_selection_authority(
    observe: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = jax.jit(lambda value: jax.lax.convert_element_type(value, np.float16))
    with monkeypatch.context() as patch:
        if not observe:
            patch.setattr(ir_context, "SourceStructureLedger", lambda: None)
        model = _source(function, [jax.ShapeDtypeStruct((2,), np.float64)])
    assert cast_authority.AUTHORITY_KEY not in model.meta
    record = model.meta.get(cast_context.CONTEXT_KEY)
    assert record is None or record.original_lowering is None
    if observe:
        ledger = model.meta[structure._LEDGER]
        assert not ledger.cast_selections
        proof = _proof(model)
        assert proof.status == "unverified", proof
        assert proof.expected_equations == 2
    else:
        assert _proof(model).reasons == ("source-ledger-unavailable",)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("opset", (23, 26))
def test_mixed_rank_original_boundary_uses_explicit_native_broadcast(
    dtype: Any, opset: int
) -> None:
    def child(vector: Any, matrix: Any) -> Any:
        expanded = jax.lax.broadcast_in_dim(vector, (2, 3), (1,))
        return jax.lax.sub(matrix, expanded)

    function = jax.jit(child)
    specs = [
        jax.ShapeDtypeStruct((3,), dtype),
        jax.ShapeDtypeStruct((2, 3), dtype),
    ]
    original = regions.freeze_closed_body(_native(function, specs))
    assert original.gaps == () and original.expected_equations == 3
    original_child = original.equations[0].children[0][1]
    assert tuple(value.logical_shape for value in original_child.inputs) == (
        (3,),
        (2, 3),
    )
    assert tuple(value.tensor.logical_shape for value in original_child.outputs) == (
        (2, 3),
    )
    assert len(original_child.equations) == 2
    model = _source(function, specs, opset=opset)
    ledger = _validated(model, 3)
    scope = _inline_scopes(ledger)[0]
    assert scope.body == original_child
    assert tuple(tuple(value.shape) for value in model.graph.inputs) == (
        (3,),
        (2, 3),
    )
    assert tuple(model.graph.outputs[0].shape) == (2, 3)
    assert tuple(port.tensor.logical_shape for port in scope.region.captures) == (
        (3,),
        (2, 3),
    )
    inputs = (
        np.asarray([0.25, 0.5, 0.75], dtype),
        np.arange(6, dtype=dtype).reshape(2, 3),
    )
    _samples(model, function, inputs, "disabled")


# Literal caller operands have no direct lookup binding. Only the existing JIT
# binder's fresh inner formals retain their already-materialized actual values.
def _literal_argument_program(kind: str) -> tuple[Any, list[None]]:
    traces: list[None] = []

    def body(left: Any, right: Any) -> Any:
        traces.append(None)
        return jax.lax.sub(left, right)

    compiled = jax.jit(body, inline=False)
    outer = jax.jit(lambda value: compiled(value, np.float32(0.75)), inline=False)

    def target(value: Any) -> Any:
        if kind == "flat":
            return compiled(value, np.float32(0.75))
        return outer(value), outer(jax.lax.neg(value))

    return target, traces


def _literal_argument_equations(closed: Any) -> list[Any]:
    from jax2onnx._compat.jax import native_jit_primitive

    equations: list[Any] = []

    def visit(native: Any, ancestors: frozenset[int]) -> None:
        body = native.jaxpr
        assert id(body) not in ancestors
        for equation in body.eqns:
            if equation.primitive is native_jit_primitive():
                if any(isinstance(var, Literal) for var in equation.invars):
                    equations.append(equation)
                visit(equation.params["jaxpr"], ancestors | {id(body)})

    visit(closed, frozenset())
    return equations


@pytest.mark.parametrize("kind", ("flat", "nested-shared"))
@pytest.mark.parametrize("source_first", (False, True))
def test_native_literal_argument_uses_one_existing_binding_per_invocation(
    kind: str, source_first: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = [jax.ShapeDtypeStruct((), np.float32)]
    expected_calls = 1 if kind == "flat" else 2
    expected_total = 2 if kind == "flat" else 7
    original_storage = structure.ConstantContract.from_array(
        np.asarray(0.75, np.float32)
    )
    original_materialize = ir_context.IRContext._bind_literal_value_for_var
    materialized: list[ir.Value] = []

    def materialize(ctx: Any, var: Any, **kwargs: Any) -> ir.Value:
        value = original_materialize(ctx, var, **kwargs)
        if (
            isinstance(var, Literal)
            and np.dtype(var.aval.dtype) == np.dtype(np.float32)
            and tuple(var.aval.shape) == ()
            and np.asarray(var.val, dtype=np.float32).tobytes()
            == np.asarray(0.75, np.float32).tobytes()
        ):
            assert ctx.lookup_value_for_var(var) is None
            materialized.append(value)
        return value

    monkeypatch.setattr(
        ir_context.IRContext, "_bind_literal_value_for_var", materialize
    )
    exports: dict[bool, tuple[ir.Model, Any, list[None]]] = {}
    trace_deltas: dict[bool, int] = {}
    for observe in (True, False) if source_first else (False, True):
        function, traces = _literal_argument_program(kind)
        native = _native(function, specs)
        literal_equations = _literal_argument_equations(native)
        assert len(literal_equations) == expected_calls
        for equation in literal_equations:
            assert len(equation.invars) == 2
            assert not isinstance(equation.invars[0], Literal)
            assert isinstance(equation.invars[1], Literal)
            literal = equation.invars[1]
            assert tuple(literal.aval.shape) == ()
            assert (
                structure.ConstantContract.from_array(
                    np.asarray(literal.val, dtype=literal.aval.dtype)
                )
                == original_storage
            )
        frozen = regions.freeze_closed_body(native)
        assert frozen.gaps == () and frozen.expected_equations == expected_total
        before_materialized, before_traces = len(materialized), len(traces)
        with monkeypatch.context() as patch:
            if not observe:
                patch.setattr(ir_context, "SourceStructureLedger", lambda: None)
            model = _source(function, specs, opset=23)
        trace_deltas[observe] = len(traces) - before_traces
        emitted = materialized[before_materialized:]
        assert len(emitted) == len({id(value) for value in emitted}) == expected_calls
        assert all(
            structure.ConstantContract.from_array(_storage(value)) == original_storage
            for value in emitted
        )
        final_literals = [
            value
            for value in _stored_values(model)
            if structure.ConstantContract.from_array(_storage(value))
            == original_storage
        ]
        assert len(final_literals) == expected_calls
        if observe:
            ledger = _validated(model, expected_total)
            literal_attachments = [
                (handle, attachment)
                for handle, attachment in ledger.inline_calls.items()
                if len(attachment.inputs) == 2
                if isinstance(
                    next(
                        scope
                        for scope in ledger.scopes
                        if scope.invocation is handle.invocation
                    )
                    .receipts[handle.index]
                    .inputs[1],
                    structure.LiteralOperand,
                )
            ]
            assert len(literal_attachments) == expected_calls
            for handle, attachment in literal_attachments:
                child = next(
                    scope
                    for scope in _inline_scopes(ledger)
                    if scope.region.key == attachment.region
                )
                assert len(child.region.captures) == len(attachment.inputs) == 2
                assert child.input_values == attachment.inputs
                port = child.region.captures[1]
                assert port.local == port.outer == attachment.inputs[1]
                assert isinstance(port.binding, structure.LiteralOperand)
                assert port.binding.storage == original_storage
                actual = structure._resolve_bindings(
                    model.graph, {attachment.inputs[1]}
                )[attachment.inputs[1]]
                assert any(actual is value for value in final_literals)
                assert actual.dtype is ir.DataType.FLOAT and tuple(actual.shape) == ()
            before_audit = tuple(traces)
            before_allocations = len(materialized)
            with monkeypatch.context() as patch:
                patch.setattr(
                    jax,
                    "make_jaxpr",
                    lambda *args, **kwargs: pytest.fail(
                        "Literal binding audit retraced source"
                    ),
                )
                patch.setattr(
                    jax,
                    "eval_shape",
                    lambda *args, **kwargs: pytest.fail(
                        "Literal binding audit evaluated source"
                    ),
                )
                _validated(model, expected_total)
                _validated(model, expected_total)
            assert tuple(traces) == before_audit
            assert len(materialized) == before_allocations
        else:
            assert _proof(model).reasons == ("source-ledger-unavailable",)
        exports[observe] = model, function, traces
    enabled, function, _ = exports[True]
    disabled = exports[False][0]
    assert trace_deltas[True] == trace_deltas[False]
    assert ir.to_proto(enabled).SerializeToString(deterministic=True) == ir.to_proto(
        disabled
    ).SerializeToString(deterministic=True)
    enabled_report, disabled_report = get_precision_report(
        enabled
    ), get_precision_report(disabled)
    assert enabled_report is not None and disabled_report is not None
    assert precision_report_to_json(enabled_report) == precision_report_to_json(
        disabled_report
    )
    enabled_context = enabled.meta.get(cast_context.CONTEXT_KEY)
    disabled_context = disabled.meta.get(cast_context.CONTEXT_KEY)
    assert (None if enabled_context is None else enabled_context.original_lowering) == (
        None if disabled_context is None else disabled_context.original_lowering
    )
    _samples(enabled, function, (np.asarray(1.5, np.float32),), "disabled")
    _samples(disabled, function, (np.asarray(1.5, np.float32),), "disabled")


@pytest.mark.parametrize(
    "fault", ("missing", "wrong-dtype", "wrong-value", "same-typed-dynamic")
)
def test_literal_argument_capture_rejects_missing_or_substituted_fresh_binding(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    function, _ = _literal_argument_program("flat")
    specs = [jax.ShapeDtypeStruct((), np.float32)]
    native = _native(function, specs)
    assert len(_literal_argument_equations(native)) == 1
    assert regions.freeze_closed_body(native).expected_equations == 2
    # This first SOURCE export is the genuine unmutated baseline. The next
    # export reuses this exact function/specs and tests only fresh IR binding
    # observation; it makes no cold/fresh native execution or cache claim.
    baseline = _source(function, specs, opset=23)
    _validated(baseline, 2)
    original_freshen = JitPlugin._freshen_closed_jaxpr
    original_lookup = ir_context.IRContext.lookup_value_for_var
    formals: dict[int, Any] = {}
    observations: list[tuple[Any, Any]] = []

    def freshen(closed: Any) -> Any:
        fresh = original_freshen(closed)
        assert len(fresh.jaxpr.invars) == 2
        formals[id(fresh.jaxpr.invars[1])] = fresh.jaxpr.invars[0]
        return fresh

    def lookup(ctx: Any, var: Any) -> ir.Value | None:
        actual = original_lookup(ctx, var)
        if id(var) not in formals or observations:
            return actual
        assert actual is not None
        assert actual.dtype is ir.DataType.FLOAT and tuple(actual.shape) == ()
        if fault == "missing":
            # Remove the actual fresh-formal binding only while the observation
            # reads it. Restore it for the original numerical lowering path.
            held = ctx.builder._var2val.pop(var)
            try:
                missing = original_lookup(ctx, var)
                assert missing is None
                observations.append((actual, missing))
                return missing
            finally:
                ctx.bind_value_for_var(var, held)
        if fault == "same-typed-dynamic":
            replacement = original_lookup(ctx, formals[id(var)])
            assert replacement is not None
            _same_type(actual, replacement)
        else:
            payload = (
                np.asarray(0.75, np.float64)
                if fault == "wrong-dtype"
                else np.asarray(0.5, np.float32)
            )
            replacement = ctx.builder.add_initializer_from_array(
                name=ctx.fresh_name("substituted_literal_argument"), array=payload
            )
            assert replacement is not actual
            if fault == "wrong-value":
                _same_type(actual, replacement)
                assert _storage(actual).tobytes() != _storage(replacement).tobytes()
        ctx.bind_value_for_var(var, replacement)
        try:
            observed = original_lookup(ctx, var)
            assert observed is replacement
            observations.append((actual, observed))
            return observed
        finally:
            ctx.bind_value_for_var(var, actual)
            assert ctx.builder._var2val[var] is actual

    monkeypatch.setattr(JitPlugin, "_freshen_closed_jaxpr", staticmethod(freshen))
    monkeypatch.setattr(ir_context.IRContext, "lookup_value_for_var", lookup)
    model = _source(function, specs, opset=23)
    assert len(observations) == 1
    proof = _proof(model)
    assert proof.status == ("unverified" if fault == "missing" else "failed"), proof
    assert proof.expected_equations == 2 and proof.checked_equations < 2
    if fault == "missing":
        assert observations[0][1] is None
        _samples(model, function, (np.asarray(1.5, np.float32),), "disabled")
    else:
        assert observations[0][0] is not observations[0][1]


def test_equal_byte_closed_constants_keep_distinct_ordered_native_bindings() -> None:
    first = np.asarray([0x80000000, 0x3F800000], np.uint32).view(np.float32)
    second = first.copy()
    assert first is not second and first.tobytes() == second.tobytes()
    function = jax.jit(lambda left, right: (left - first, right - second))
    model = _source(function, _specs(shape=(2,)))
    ledger = _validated(model, 3)
    (child,) = _inline_scopes(ledger)
    ((handle, attachment),) = ledger.inline_calls.items()
    assert len(child.body.constants) == len(child.closed_constant_values) == 2
    assert attachment.constants == child.closed_constant_values
    assert attachment.constants[0] != attachment.constants[1]
    assert (
        ledger.constants[attachment.constants[0]]
        == ledger.constants[attachment.constants[1]]
    )
    ledger.inline_calls[handle] = replace(
        attachment, constants=tuple(reversed(attachment.constants))
    )
    proof = _proof(model)
    assert proof.status == "failed", proof
    assert proof.expected_equations == 3 and proof.checked_equations == 0
    assert proof.reasons == ("inline-original-binding-order-differs",)
