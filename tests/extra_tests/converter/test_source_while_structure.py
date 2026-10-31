# tests/extra_tests/converter/test_source_while_structure.py

"""Scalar SOURCE While invocation accounting and actual final Loop edges."""

from collections.abc import Iterator
from dataclasses import fields, is_dataclass, replace
from typing import Any

import jax
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import Jaxpr, JaxprEqn, Literal, Var
from jax2onnx.converter import _source_regions as regions
from jax2onnx.converter import ir_context, source_structure as structure
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.ir_utils import iter_ir_functions
from jax2onnx.plugins import plugin_system
from jax2onnx.plugins.jax.lax._source_loop_structure import WhileRule
from jax2onnx.precision import SampleCriteria, SourceSample, validate_source_samples


def _mixed(counter: Any, value: Any, keep: Any, limit: Any, step: Any) -> Any:
    def condition(state: Any) -> Any:
        return jax.lax.lt(state[0], limit)

    def body(state: Any) -> Any:
        count, payload, flag = state
        return jax.lax.add(count, np.int32(1)), jax.lax.add(payload, step), flag

    return jax.lax.while_loop(condition, body, (counter, value, keep))


@onnx_function
def difference(x: Any, y: Any) -> Any:
    return jax.lax.sub(x, y)


@onnx_function
def wrapped(c: Any, x: Any, flag: Any, limit: Any, step: Any) -> Any:
    return _mixed(c, x, flag, limit, step)


@onnx_function
def selected(value: Any, step: Any, *, deterministic: Any = True) -> Any:
    return jax.lax.cond(
        deterministic, lambda x, y: x - y, lambda x, y: x + y, value, step
    )


@pytest.fixture(autouse=True)
def _module_function_bindings() -> Iterator[None]:
    targets = (difference, wrapped, selected)
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
    return [
        jax.ShapeDtypeStruct((), np.int32),
        jax.ShapeDtypeStruct(shape, dtype),
        jax.ShapeDtypeStruct((), np.bool_),
        jax.ShapeDtypeStruct((), np.int32),
        jax.ShapeDtypeStruct(shape, dtype),
    ]


def _source(
    function: Any = _mixed,
    *,
    dtype: Any = np.float32,
    shape: tuple[int, ...] = (2, 3),
    **kwargs: Any,
) -> ir.Model:
    with jax.enable_x64(True):
        model = ui._to_onnx(
            function,
            _specs(dtype, shape),
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
            **kwargs,
        )
    assert isinstance(model, ir.Model)
    return model


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


def _loops(model: ir.Model) -> list[ir.Node]:
    return [
        node
        for graph in _graphs(model)
        for node in graph
        if not node.domain and node.op_type == "Loop"
    ]


def _root_while_rule(model: ir.Model) -> WhileRule:
    ledger = model.meta[structure._LEDGER]
    root = next(
        scope
        for scope in ledger.scopes
        if scope.region is None and scope.owner is model.graph.meta[structure._OWNER]
    )
    rules = [
        equation.rule for equation in root.equations if type(equation.rule) is WhileRule
    ]
    assert len(rules) == 1
    return rules[0]


def _same_typed_distinct(actual: ir.Value, replacement: ir.Value) -> None:
    assert actual is not replacement
    assert actual.dtype is replacement.dtype
    assert actual.shape == replacement.shape


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("shape", ((), (2, 3), (2, 1, 3)))
@pytest.mark.parametrize("opset", (18, 23, 26))
def test_three_original_roles_share_two_physical_graphs(
    dtype: Any, shape: tuple[int, ...], opset: int
) -> None:
    if opset == 18 and dtype is np.float64:
        with pytest.raises(
            SourcePrecisionError,
            match="dtype is outside the original opset18/21 typed expression",
        ) as caught:
            _source(dtype=dtype, shape=shape, opset=opset)
        assert caught.value.category == "profile-exclusion"
        assert caught.value.opset == 18
        assert (
            caught.value.action
            == "use operations within this audited profile or add audited support for the requested operation"
        )
        return
    with jax.enable_x64(True):
        native = jax.make_jaxpr(_mixed)(*_specs(dtype, shape))
    native_while = next(
        equation
        for equation in native.jaxpr.eqns
        if equation.primitive is jax.lax.while_p
    )
    nc = native_while.params["cond_nconsts"]
    nb = native_while.params["body_nconsts"]
    model = _source(dtype=dtype, shape=shape, opset=opset)
    ledger = _validated(model, 5)
    rule = _root_while_rule(model)
    assert (rule.condition_constants, rule.body_constants, rule.carries) == (
        nc,
        nb,
        len(native_while.outvars),
    )
    assert len(ledger.while_calls) == 1
    assert len(ledger.regions) == 3
    assert len(ledger.scopes) == 4
    assert {key.role for key in ledger.regions} == {
        "initial-condition",
        "body",
        "next-condition",
    }
    children = {
        scope.region.key.role: scope
        for scope in ledger.scopes
        if scope.region is not None
    }
    assert children["initial-condition"].owner is ledger.scopes[0].owner
    assert children["body"].owner is children["next-condition"].owner
    loop = _loops(model)[0]
    body = loop.attributes["body"].as_graph()
    assert loop.inputs[0] is None
    serialized = next(
        node for node in ir.to_proto(model).graph.node if node.op_type == "Loop"
    )
    assert serialized.input[0] == ""
    assert len(loop.inputs) == len(body.inputs) == 7
    assert len(loop.outputs) == 5 and len(body.outputs) == 6
    native_order = (
        *native_while.invars[nc : nc + nb],
        *native_while.invars[:nc],
        *native_while.invars[nc + nb :],
    )
    assert [value.dtype for value in loop.inputs[2:]] == [
        structure.TensorContract.from_aval(var.aval, structure._ShapeFreezer()).dtype
        for var in native_order
    ]


@pytest.mark.parametrize("trips", (0, 1, 4))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_runtime_condition_only_zero_one_many_keeps_same_artifact(
    trips: int, dtype: Any
) -> None:
    model = _source(dtype=dtype)
    _validated(model, 5)
    inputs = (
        np.asarray(0, np.int32),
        np.arange(6, dtype=dtype).reshape(2, 3),
        np.asarray(True),
        np.asarray(trips, np.int32),
        np.full((2, 3), 0.5, dtype),
    )
    with jax.enable_x64(True):
        expected = [np.asarray(value) for value in _mixed(*inputs)]
    for optimization in ("disabled", "all"):
        report = validate_source_samples(
            model,
            samples=[SourceSample(inputs, expected, SampleCriteria(mode="bitwise"))],
            optimization=optimization,
        )
        report.require_sample_acceptance()
    assert _proof(model).runtime == _proof(model).numerics == "unverified"


@pytest.mark.parametrize(
    "kind",
    (
        "nested",
        "inside-function",
        "reused-function",
        "if-body",
        "while-in-if",
        "condition-while",
    ),
)
def test_nested_regions_reused_functions_and_same_owner_conditions(kind: str) -> None:

    def target(c: Any, x: Any, flag: Any, limit: Any, step: Any) -> Any:
        if kind == "inside-function":
            return wrapped(c, x, flag, limit, step)
        if kind == "reused-function":
            return wrapped(c, x, flag, limit, step), wrapped(c, step, flag, limit, x)
        if kind == "while-in-if":
            return jax.lax.cond(
                flag,
                lambda: _mixed(c, x, flag, limit, step),
                lambda: _mixed(c, step, flag, limit, x),
            )

        def condition(state: Any) -> Any:
            if kind == "condition-while":
                inner = jax.lax.while_loop(
                    lambda count: count < limit,
                    lambda count: count + np.int32(1),
                    state[0],
                )
                return inner < limit
            return state[0] < limit

        def body(state: Any) -> Any:
            count, payload, keep = state
            if kind == "nested":
                payload = _mixed(count, payload, keep, limit, step)[1]
            elif kind == "if-body":
                payload = jax.lax.cond(
                    keep,
                    lambda a: difference(a, step),
                    lambda a: difference(step, a),
                    payload,
                )
            return count + np.int32(1), payload, keep

        return jax.lax.while_loop(condition, body, (c, x, flag))

    model = _source(target)
    ledger = _validated(model)
    assert len(ledger.regions) == 3 * len(ledger.while_calls) + 2 * len(ledger.if_calls)
    if kind == "reused-function":
        assert len(list(iter_ir_functions(model.functions))) == 1
        assert len(ledger.function_calls) == 2
        assert len(ledger.while_calls) == 1
    if kind == "condition-while":
        owners = [scope.owner for scope in ledger.scopes]
        assert max(owners.count(owner) for owner in owners) >= 3


def test_repeated_same_typed_carries_keep_original_order() -> None:
    def target(counter: Any, value: Any, keep: Any, limit: Any, step: Any) -> Any:
        def body(state: Any) -> Any:
            count, left, right, flag = state
            difference = left - right
            return count + np.int32(1), difference, difference, flag

        return jax.lax.while_loop(
            lambda state: state[0] < limit, body, (counter, value, step, keep)
        )

    model = _source(target)
    ledger = _validated(model, 5)
    body_scope = next(
        scope
        for scope in ledger.scopes
        if scope.region is not None and scope.region.key.role == "body"
    )
    assert body_scope.result_values[1] == body_scope.result_values[2]
    loop = _loops(model)[0]
    body = loop.attributes["body"].as_graph()
    # Replacing a repeated result with an unrelated same-typed formal must fail.
    rule = _root_while_rule(model)
    first_carry = rule.body_constants + rule.condition_constants
    output_port = 1 + first_carry + 1
    replacement = body.inputs[2 + first_carry + 2]
    _same_typed_distinct(body.outputs[output_port], replacement)
    body.outputs[output_port] = replacement
    assert _proof(model).status == "failed"


@pytest.mark.parametrize("declared", (False, True))
def test_named_dynamic_flag_uses_original_explicit_ports(declared: bool) -> None:

    def target(
        counter: Any,
        value: Any,
        keep: Any,
        limit: Any,
        step: Any,
        *,
        deterministic: Any = True,
    ) -> Any:
        def body(state: Any) -> Any:
            count, payload, flag = state
            result = (
                selected(payload, step, deterministic=deterministic)
                if declared
                else payload + step
            )
            return count + np.int32(1), result, flag

        return jax.lax.while_loop(
            lambda state: state[0] < limit, body, (counter, value, keep)
        )

    model = _source(target, input_params={"deterministic": True} if declared else {})
    ledger = _validated(model)
    assert all(not attachment.lazy for attachment in ledger.while_calls.values())
    names = {value.name for value in model.graph.inputs}
    if declared:
        assert "deterministic" in names
        for flag in (False, True):
            arrays = (
                np.asarray(0, np.int32),
                np.ones((2, 3), np.float32),
                np.asarray(True),
                np.asarray(2, np.int32),
                np.full((2, 3), 0.5, np.float32),
            )
            with jax.enable_x64(True):
                expected = [
                    np.asarray(value) for value in target(*arrays, deterministic=flag)
                ]
            feed = {
                value.name: array
                for value, array in zip(model.graph.inputs[:5], arrays, strict=True)
            }
            feed["deterministic"] = np.asarray(flag)
            report = validate_source_samples(
                model,
                samples=[SourceSample(feed, expected, SampleCriteria(mode="bitwise"))],
                optimization="disabled",
            )
            report.require_sample_acceptance()
    else:
        assert "deterministic" not in names


@pytest.mark.parametrize(
    "fault",
    (
        "trip-count",
        "initial-condition",
        "next-condition",
        "body-operand",
        "constant-input",
        "constant-output",
        "carry-output",
        "header-type",
        "extra-attribute",
    ),
)
def test_resolved_same_typed_loop_corruption_fails(fault: str) -> None:
    model = _source()
    _validated(model, 5)
    loop = _loops(model)[0]
    body = loop.attributes["body"].as_graph()
    rule = _root_while_rule(model)
    first_carry = rule.body_constants + rule.condition_constants
    counter = body.inputs[2 + first_carry]
    payload = body.inputs[2 + first_carry + 1]
    step_index = next(
        index
        for index in range(rule.body_constants)
        if body.inputs[2 + index].dtype is ir.DataType.FLOAT
    )
    step = body.inputs[2 + step_index]
    if fault == "trip-count":
        value = ir.Value(
            name="finite_M",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(()),
            const_value=ir.tensor(np.asarray(100, np.int64)),
        )
        model.graph.register_initializer(value)
        loop.replace_input_with(0, value)
    elif fault == "initial-condition":
        _same_typed_distinct(loop.inputs[1], model.graph.inputs[2])
        loop.replace_input_with(1, model.graph.inputs[2])
    elif fault == "next-condition":
        predicate = next(node for node in body if node.op_type == "Less")
        _same_typed_distinct(predicate.inputs[0], counter)
        predicate.replace_input_with(0, counter)
    elif fault == "body-operand":
        operation = next(
            node
            for node in body
            if node.op_type == "Add" and node.outputs[0].dtype is ir.DataType.FLOAT
        )
        _same_typed_distinct(operation.inputs[0], step)
        operation.replace_input_with(0, step)
    elif fault == "constant-input":
        _same_typed_distinct(loop.inputs[2 + step_index], model.graph.inputs[1])
        loop.replace_input_with(2 + step_index, model.graph.inputs[1])
    elif fault == "constant-output":
        _same_typed_distinct(body.outputs[1 + step_index], payload)
        body.outputs[1 + step_index] = payload
    elif fault == "carry-output":
        _same_typed_distinct(body.outputs[1 + first_carry + 1], step)
        body.outputs[1 + first_carry + 1] = step
    elif fault == "header-type":
        body.inputs[0].type = ir.TensorType(ir.DataType.INT32)
    else:
        loop.attributes["unreviewed"] = ir.AttrInt64("unreviewed", 0)
    result = _proof(model)
    assert result.status == "failed", result
    assert result.expected_equations == 5


@pytest.mark.parametrize(
    "fault",
    (
        "attachment",
        "region",
        "scope",
        "body-receipt",
        "role-order",
        "duplicate",
        "unclaimed",
    ),
)
def test_lost_evidence_retains_every_original_invocation(fault: str) -> None:
    model = _source()
    ledger = _validated(model, 5)
    child = next(
        scope
        for scope in ledger.scopes
        if scope.region is not None and scope.region.key.role == "body"
    )
    if fault == "attachment":
        ledger.while_calls.clear()
    elif fault == "region":
        ledger.regions.pop(child.region.key)
    elif fault == "scope":
        ledger.scopes.remove(child)
    elif fault == "body-receipt":
        child.receipts.clear()
    elif fault == "role-order":
        handle, attachment = next(iter(ledger.while_calls.items()))
        first, body, last = attachment.regions
        ledger.while_calls[handle] = replace(attachment, regions=(last, body, first))
    elif fault == "duplicate":
        ledger.scopes.append(replace(child, invocation=object()))
    else:
        ledger.scopes.append(replace(child, invocation=object(), region=None))
    result = _proof(model)
    assert result.status == "unverified", result
    assert result.expected_equations == 5 and result.checked_equations < 5


def test_erased_internal_dimensions_are_derived_without_mutating_metadata() -> None:
    model = _source()
    _validated(model, 5)
    graph = _loops(model)[0].attributes["body"].as_graph()
    values = [
        *graph.inputs,
        *graph.outputs,
        *(value for node in graph for value in node.outputs),
    ]
    for value in values:
        if value.shape is not None and len(value.shape):
            value.shape = ir.Shape((None,) * len(value.shape))
            assert all(
                type(dim) is ir.SymbolicDim and dim.value is None for dim in value.shape
            )
    before = [
        (value, tuple(value.shape) if value.shape is not None else None)
        for value in values
    ]
    assert _proof(model).status == "validated"
    assert before == [
        (value, tuple(value.shape) if value.shape is not None else None)
        for value in values
    ]


def test_named_final_symbol_is_not_erased_static_metadata() -> None:
    model = _source()
    _validated(model, 5)
    graph = _loops(model)[0].attributes["body"].as_graph()
    operation = next(
        node
        for node in graph
        if node.op_type == "Add" and node.outputs[0].dtype is ir.DataType.FLOAT
    )
    operation.outputs[0].shape = ir.Shape(("B", 3))
    assert type(operation.outputs[0].shape[0]) is ir.SymbolicDim
    assert operation.outputs[0].shape[0].value == "B"
    proof = _proof(model)
    assert proof.status == "unverified", proof
    assert proof.reasons == ("while-derived-metadata-unavailable",)
    assert proof.expected_equations == 5


@pytest.mark.parametrize("producer_kind", ("initial-condition", "body-constant"))
def test_final_producer_reorders_keep_edges_but_fail_visibility(
    producer_kind: str,
) -> None:
    model = _source()
    _validated(model, 5)
    loop = _loops(model)[0]
    expected_equations = 5
    if producer_kind == "initial-condition":
        graph = model.graph
        consumer = loop
        producer = loop.inputs[1].producer()
        assert producer is not None and producer.op_type == "Less"
    else:
        # Resolve the module-global decorated callable during tracing, as in
        # the existing inside-function case. Its Function invocation adds one
        # original equation; normal lowering retains its genuine Constant.
        def target(c: Any, x: Any, flag: Any, limit: Any, step: Any) -> Any:
            return wrapped(c, x, flag, limit, step)

        model = _source(target)
        expected_equations = 6
        loop = _loops(model)[0]
        graph = loop.attributes["body"].as_graph()
        consumer = next(
            node
            for node in graph
            if node.op_type == "Add" and node.outputs[0].dtype is ir.DataType.INT32
        )
        literal = consumer.inputs[1]
        assert literal is not None
        producer = literal.producer()
        assert producer is not None and producer.op_type == "Constant"
        _validated(model, expected_equations)
    before = {
        id(node): (
            tuple(id(value) for value in node.inputs),
            tuple(id(value) for value in node.outputs),
        )
        for node in graph
    }
    graph.insert_after(consumer, producer)
    assert tuple(graph).index(consumer) < tuple(graph).index(producer)
    assert producer.graph is consumer.graph is graph
    assert before == {
        id(node): (
            tuple(id(value) for value in node.inputs),
            tuple(id(value) for value in node.outputs),
        )
        for node in graph
    }
    proof = _proof(model)
    assert proof.status == "failed", proof
    assert proof.reasons == ("while-input-not-visible-before-consumer",)
    assert proof.expected_equations == expected_equations


def test_post_export_initializer_to_constant_loses_physical_provenance() -> None:
    model = _source()
    ledger = _validated(model, 5)
    graph = _loops(model)[0].attributes["body"].as_graph()
    consumer = next(
        node
        for node in graph
        if node.op_type == "Add" and node.outputs[0].dtype is ir.DataType.INT32
    )
    literal = consumer.inputs[1]
    assert literal is not None
    assert literal.producer() is None
    assert literal.is_initializer() and literal.const_value is not None
    assert literal.name is not None
    tensor = literal.const_value
    inputs = tuple(consumer.inputs)
    markers = literal.meta.get(structure._VALUES, ())
    assert type(markers) is tuple and markers
    assert any(
        type(witness) is structure._ValueWitness and witness.role == "initializer"
        for witness in markers
    )
    lift_receipts = graph.meta.get(structure._CONSTANT_LIFTS)
    dce_receipts = graph.meta.get("jax2onnx.source.structure.identity_dce")
    del graph.initializers[literal.name]
    producer = ir.Node(
        "",
        "Constant",
        [],
        outputs=[literal],
        attributes=[ir.AttrTensor("value", tensor)],
    )
    graph.insert_before(consumer, producer)
    assert tuple(consumer.inputs) == inputs
    assert producer.outputs[0] is literal
    assert literal.const_value is tensor
    assert producer.attributes["value"].as_tensor() is tensor
    assert producer.graph is consumer.graph is graph
    assert tuple(graph).index(producer) < tuple(graph).index(consumer)
    assert literal.meta[structure._VALUES] is markers
    assert graph.meta.get(structure._CONSTANT_LIFTS) is lift_receipts
    assert graph.meta.get("jax2onnx.source.structure.identity_dce") is dce_receipts
    assert model.meta[structure._LEDGER] is ledger
    # The graph/storage rewrite is legal ONNX, but no inverse provenance
    # transition was observed or licensed after the original export.
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    proof = _proof(model)
    assert proof.status == "unverified", proof
    assert proof.reasons == ("final-value-witness-unavailable",)
    assert proof.expected_equations == 5
    assert proof.checked_equations == 0
    assert proof.runtime == proof.numerics == "unverified"
    assert literal.meta[structure._VALUES] is markers
    assert graph.meta.get(structure._CONSTANT_LIFTS) is lift_receipts


def test_final_graph_clone_preserves_shared_while_and_if_owners() -> None:
    def target(counter: Any, value: Any, keep: Any, limit: Any, step: Any) -> Any:
        def body(state: Any) -> Any:
            count, payload, flag = state
            payload = jax.lax.cond(
                flag, lambda x, y: x - y, lambda x, y: x + y, payload, step
            )
            return count + np.int32(1), payload, flag

        return jax.lax.while_loop(
            lambda state: state[0] < limit, body, (counter, value, keep)
        )

    original = _source(target)
    original_ledger = _validated(original, 8)
    cloned = ir.Model(
        original.graph.clone(allow_outer_scope_values=True),
        ir_version=original.ir_version,
    )
    cloned.meta.update(original.meta)
    assert cloned.graph is not original.graph
    assert _loops(cloned)[0] is not _loops(original)[0]
    assert (
        _loops(cloned)[0].attributes["body"].as_graph()
        is not _loops(original)[0].attributes["body"].as_graph()
    )
    cloned_ledger = _validated(cloned, 8)
    assert cloned_ledger is original_ledger
    assert len(cloned_ledger.while_calls) == len(cloned_ledger.if_calls) == 1
    for flag in (False, True):
        inputs = (
            np.asarray(0, np.int32),
            np.ones((2, 3), np.float32),
            np.asarray(flag),
            np.asarray(3, np.int32),
            np.full((2, 3), 0.5, np.float32),
        )
        with jax.enable_x64(True):
            expected = [np.asarray(value) for value in target(*inputs)]
        for optimization in ("disabled", "all"):
            report = validate_source_samples(
                cloned,
                samples=[
                    SourceSample(inputs, expected, SampleCriteria(mode="bitwise"))
                ],
                optimization=optimization,
            )
            report.require_sample_acceptance()


@pytest.mark.parametrize("fault", ("concrete", "rank", "unavailable-operator"))
def test_missing_or_conflicting_actual_shape_facts_never_borrow_source_avals(
    fault: str,
) -> None:
    model = _source()
    _validated(model, 5)
    graph = _loops(model)[0].attributes["body"].as_graph()
    operation = next(
        node
        for node in graph
        if node.op_type == "Add" and node.outputs[0].dtype is ir.DataType.FLOAT
    )
    if fault == "concrete":
        operation.outputs[0].shape = ir.Shape((3, 2))
    elif fault == "rank":
        operation.outputs[0].shape = ir.Shape((None,))
    else:
        operation.op_type = "UnreviewedShapeOperator"
        operation.outputs[0].shape = ir.Shape((None, None))
    proof = _proof(model)
    assert proof.status == (
        "unverified" if fault == "unavailable-operator" else "failed"
    ), proof
    assert proof.expected_equations == 5


@pytest.mark.parametrize("consumed", (False, True))
def test_graph_wide_node_and_constant_union_rejects_unclaimed_storage(
    consumed: bool,
) -> None:
    model = _source()
    _validated(model, 5)
    graph = _loops(model)[0].attributes["body"].as_graph()
    value = ir.Value(
        name="extra_storage",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape((2, 3)),
        const_value=ir.tensor(np.ones((2, 3), np.float32)),
    )
    graph.register_initializer(value)
    if consumed:
        operation = next(
            node
            for node in graph
            if node.op_type == "Add" and node.outputs[0].dtype is ir.DataType.FLOAT
        )
        operation.replace_input_with(0, value)
    proof = _proof(model)
    assert proof.status == ("failed" if consumed else "unverified"), proof
    if not consumed:
        assert "artifact-constant-coverage-incomplete" in proof.reasons


def test_extra_node_in_shared_parent_cannot_be_claimed_twice() -> None:
    model = _source()
    _validated(model, 5)
    output = ir.Value(
        name="uncovered",
        type=model.graph.inputs[1].type,
        shape=model.graph.inputs[1].shape,
    )
    model.graph.append(
        ir.Node("", "Identity", [model.graph.inputs[1]], outputs=[output])
    )
    assert "artifact-node-coverage-incomplete" in _proof(model).reasons


@pytest.mark.parametrize("fault", ("signed-zero", "nan-payload"))
def test_closed_body_storage_remains_exact_in_shared_loop_owner(fault: str) -> None:
    captured = np.asarray([0x80000000, 0x7FC00001], np.uint32).view(np.float32)

    def target(c: Any, x: Any, keep: Any, limit: Any, step: Any) -> Any:
        def body(state: Any) -> Any:
            return state[0] + np.int32(1), state[1] - captured, state[2]

        return jax.lax.while_loop(lambda state: state[0] < limit, body, (c, x, keep))

    model = _source(target, shape=(2,))
    _validated(model, 5)
    loop = _loops(model)[0]
    # JAX can lift the closed constant into the While operand partition. Find
    # its actual final storage, retaining the original signed-zero/NaN bytes.
    found = []
    for graph in _graphs(model):
        for value in (
            *graph.initializers.values(),
            *(out for node in graph for out in node.outputs),
        ):
            node = value.producer()
            tensor = (
                node.attributes["value"].as_tensor()
                if node is not None and node.op_type == "Constant"
                else value.const_value if value.is_initializer() else None
            )
            if (
                tensor is not None
                and tensor.numpy().dtype == captured.dtype
                and tensor.numpy().shape == captured.shape
                and tensor.numpy().tobytes() == captured.tobytes()
            ):
                found.append((value, node))
    assert found and loop.inputs[0] is None
    value, producer = found[0]
    changed = captured.copy().view(np.uint32)
    changed[0 if fault == "signed-zero" else 1] = (
        0 if fault == "signed-zero" else 0x7FC00002
    )
    if producer is None:
        value.const_value = ir.tensor(changed.view(np.float32))
    else:
        producer.attributes["value"] = ir.AttrTensor(
            "value", ir.tensor(changed.view(np.float32))
        )
    assert _proof(model).status == "failed"


@pytest.mark.parametrize("dtype", (np.int32, np.int64, np.float32, np.float64))
@pytest.mark.parametrize("operation", ("lt", "le", "gt", "ge", "eq"))
def test_scalar_numeric_predicate_rules_are_independent_of_float_arithmetic(
    dtype: Any, operation: str
) -> None:
    predicate = getattr(jax.lax, operation)
    with jax.enable_x64(True):
        model = ui._to_onnx(
            lambda left, right: predicate(left, right),
            [jax.ShapeDtypeStruct((), dtype)] * 2,
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
        )
    _validated(model, 1)


@pytest.mark.parametrize("dtype", (np.int32, np.int64))
@pytest.mark.parametrize("operation", ("add", "sub"))
def test_integer_counter_rules_preserve_scalar_dtype(
    dtype: Any, operation: str
) -> None:
    binary = getattr(jax.lax, operation)
    with jax.enable_x64(True):
        model = ui._to_onnx(
            lambda left, right: binary(left, right),
            [jax.ShapeDtypeStruct((), dtype)] * 2,
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
        )
    _validated(model, 1)
    assert model.graph.outputs[0].dtype is (
        ir.DataType.INT32 if dtype is np.int32 else ir.DataType.INT64
    )


def test_original_freeze_is_immutable_and_never_retraces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with jax.enable_x64(True):
        native = jax.make_jaxpr(_mixed)(*_specs())
    monkeypatch.setattr(
        jax,
        "make_jaxpr",
        lambda *args, **kwargs: pytest.fail("While freezer retraced source"),
    )
    monkeypatch.setattr(
        jax,
        "eval_shape",
        lambda *args, **kwargs: pytest.fail("While freezer evaluated source"),
    )
    frozen = regions.freeze_closed_body(native)
    assert frozen.gaps == () and frozen.expected_equations == 5
    original = next(equation for equation in frozen.equations if equation.children)
    while_equation = next(
        equation
        for equation in native.jaxpr.eqns
        if equation.primitive is jax.lax.while_p
    )
    assert original.semantics == (
        "while",
        while_equation.params["cond_nconsts"],
        while_equation.params["body_nconsts"],
        len(while_equation.outvars),
    )
    assert tuple(role for role, _ in original.children) == (
        "initial-condition",
        "body",
        "next-condition",
    )
    assert original.children[0][1] == original.children[2][1]
    before = hash(frozen)
    monkeypatch.setitem(while_equation.params, "unreviewed_parameter", False)
    changed = regions.freeze_closed_body(native)
    assert "while-attachment-contract-unavailable" in changed.gaps
    assert changed.expected_equations == 5 and hash(frozen) == before


@pytest.mark.parametrize("kind", ("vector", "complex"))
def test_following_typed_transport_routes_keep_their_explicit_gap(kind: str) -> None:
    if kind == "vector":
        target = jax.vmap(
            lambda x: jax.lax.while_loop(
                lambda y: y < np.float32(2), lambda y: y + np.float32(1), x
            )
        )
        spec = jax.ShapeDtypeStruct((2,), np.float32)
    else:

        def target(x):
            return jax.lax.while_loop(lambda y: np.asarray(False), lambda y: y, x)

        spec = jax.ShapeDtypeStruct((2,), np.complex64)
    with jax.enable_x64(True):
        frozen = regions.freeze_closed_body(jax.make_jaxpr(target)(spec))
    assert "while-attachment-contract-unavailable" in frozen.gaps
    assert (
        len(
            next(
                equation for equation in frozen.equations if equation.children
            ).children
        )
        == 3
    )


def test_while_ledger_retains_no_native_or_ir_authority() -> None:
    ledger = _validated(_source(), 5)

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
    assert any(
        type(equation.rule) is WhileRule for equation in ledger.scopes[0].equations
    )


def test_ledger_observation_preserves_emission_and_legacy_trip_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled = _source()
    _validated(enabled, 5)
    with monkeypatch.context() as patch:
        patch.setattr(ir_context, "SourceStructureLedger", lambda: None)
        disabled = _source()
    assert _proof(disabled).reasons == ("source-ledger-unavailable",)
    assert ir.to_proto(enabled).SerializeToString(deterministic=True) == ir.to_proto(
        disabled
    ).SerializeToString(deterministic=True)
    with jax.enable_x64(True):
        legacy = ui._to_onnx(_mixed, _specs(), return_mode="ir")
    assert _loops(legacy)[0].inputs[0] is not None


def test_canonical_literal_is_created_once_and_duplicate_storage_is_unclaimed() -> None:
    model = _source(shape=())
    _validated(model, 5)
    body = _loops(model)[0].attributes["body"].as_graph()
    integer_add = next(
        node
        for node in body
        if node.op_type == "Add" and node.outputs[0].dtype is ir.DataType.INT32
    )
    literal = integer_add.inputs[1]
    assert (
        literal is not None
        and literal.is_initializer()
        and literal.const_value is not None
    )
    assert tuple(body.initializers.values()) == (literal,)
    storage = literal.const_value.numpy()
    assert storage.dtype == np.dtype(np.int32) and storage.shape == ()
    assert storage.tobytes() == np.asarray(1, np.int32).tobytes()
    duplicate = ir.Value(
        name="unclaimed_literal_duplicate",
        type=literal.type,
        shape=literal.shape,
        const_value=ir.tensor(storage.copy()),
    )
    body.register_initializer(duplicate)
    proof = _proof(model)
    assert proof.status == "unverified", proof
    assert proof.expected_equations == proof.checked_equations == 5
    assert proof.reasons == ("artifact-constant-coverage-incomplete",)
