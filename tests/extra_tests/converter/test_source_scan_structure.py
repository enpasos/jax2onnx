# tests/extra_tests/converter/test_source_scan_structure.py

"""Forward SOURCE Scan original roles and independent final Loop proofs."""

from collections.abc import Iterator
from dataclasses import fields, is_dataclass, replace
import hashlib
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
from jax2onnx.ir_utils import iter_ir_functions
from jax2onnx.plugins import plugin_system
from jax2onnx.plugins.jax.lax._source_loop_structure import ScanRule
from jax2onnx.precision import SampleCriteria, SourceSample, validate_source_samples


def _target(value: Any, step: Any, xs: Any, other: Any, keep: Any) -> Any:
    def body(state: Any, elements: Any) -> Any:
        payload, flag = state
        left, right = elements
        changed = (payload + step) - left
        result = changed + right
        return (changed, flag), (result, flag, result)

    return jax.lax.scan(body, (value, keep), (xs, other))


@onnx_function
def scanned(value: Any, step: Any, xs: Any, other: Any, flag: Any) -> Any:
    return _target(value, step, xs, other, flag)


@onnx_function
def selected(value: Any, step: Any, *, deterministic: Any = True) -> Any:
    return jax.lax.cond(
        deterministic, lambda x, y: x - y, lambda x, y: x + y, value, step
    )


@pytest.fixture(autouse=True)
def _module_function_bindings() -> Iterator[None]:
    targets = (scanned, selected)
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


def _specs(
    length: int = 3, dtype: Any = np.float32, shape: tuple[int, ...] = (2, 3)
) -> list[Any]:
    return [
        jax.ShapeDtypeStruct(shape, dtype),
        jax.ShapeDtypeStruct(shape, dtype),
        jax.ShapeDtypeStruct((length, *shape), dtype),
        jax.ShapeDtypeStruct((length, *shape), dtype),
        jax.ShapeDtypeStruct((), np.bool_),
    ]


def _export(target: Any = _target, specs: Any = None, **kwargs: Any) -> ir.Model:
    with jax.enable_x64(True):
        model = ui._to_onnx(
            target,
            _specs() if specs is None else specs,
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
    assert proof.checked_equations == proof.expected_equations
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


def _storage(value: ir.Value) -> np.ndarray[Any, Any]:
    node = value.producer()
    if node is not None and node.op_type == "Identity":
        assert len(node.inputs) == 1 and node.inputs[0] is not None
        return _storage(node.inputs[0])
    tensor = (
        node.attributes["value"].as_tensor()
        if node is not None and node.op_type == "Constant"
        else value.const_value
    )
    assert tensor is not None
    return tensor.numpy()


def _replace_storage(value: ir.Value, array: np.ndarray[Any, Any]) -> None:
    node = value.producer()
    if node is not None and node.op_type == "Identity":
        assert len(node.inputs) == 1 and node.inputs[0] is not None
        _replace_storage(node.inputs[0], array)
        return
    if node is None:
        value.const_value = ir.tensor(array)
    else:
        node.attributes["value"] = ir.AttrTensor("value", ir.tensor(array))


def _identity_source(value: ir.Value) -> ir.Value:
    while True:
        node = value.producer()
        if node is None or node.op_type != "Identity":
            return value
        assert len(node.inputs) == 1 and node.inputs[0] is not None
        value = node.inputs[0]


def _replace_input(
    node: ir.Node, index: int, original: ir.Value, replacement: ir.Value
) -> None:
    assert node.inputs[index] is original and original is not replacement
    assert original.dtype is replacement.dtype
    node.replace_input_with(index, replacement)
    assert node.inputs[index] is replacement and node.inputs[index] is not original


def _replace_output(
    body: ir.Graph, index: int, original: ir.Value, replacement: ir.Value
) -> None:
    assert body.outputs[index] is original and original is not replacement
    assert original.dtype is replacement.dtype
    body.outputs[index] = replacement
    assert body.outputs[index] is replacement and body.outputs[index] is not original


def _target_loop_roles(model: ir.Model) -> tuple[ir.Node, ir.Graph]:
    ledger = model.meta[structure._LEDGER]
    root = next(scope for scope in ledger.scopes if scope.region is None)
    rule = next(
        equation.rule for equation in root.equations if type(equation.rule) is ScanRule
    )
    # Native JAX hoists the unchanged user carry flag before the step constant.
    assert (rule.constants, rule.carries, rule.sequences, rule.results) == (2, 1, 2, 3)
    loop = _loops(model)[0]
    body = loop.attributes["body"].as_graph()
    expected = (
        model.graph.inputs[4],
        model.graph.inputs[1],
        model.graph.inputs[0],
        model.graph.inputs[2],
        model.graph.inputs[3],
    )
    assert len(loop.inputs) == len(body.inputs) == 7
    assert all(
        actual is original
        for actual, original in zip(loop.inputs[2:], expected, strict=True)
    )
    for output, formal in ((0, 1), (1, 2), (2, 3), (4, 5), (5, 6)):
        assert _identity_source(body.outputs[output]) is body.inputs[formal]
    child = next(scope for scope in ledger.scopes if scope.region is not None)
    assert child.result_values is not None
    resolved = structure._resolve_bindings(
        body, {*child.input_values, *child.result_values}
    )
    assert all(
        resolved[child.input_values[index]] is body.inputs[index + 2]
        for index in range(3)
    )
    assert body.outputs[3] is resolved[child.result_values[0]]
    return loop, body


def _samples(model: ir.Model, target: Any, inputs: Any, optimization: str) -> None:
    with jax.enable_x64(True):
        expected = [np.asarray(value) for value in jax.tree.leaves(target(*inputs))]
    report = validate_source_samples(
        model,
        samples=[SourceSample(inputs, expected, SampleCriteria(mode="bitwise"))],
        optimization=optimization,
    )
    report.require_sample_acceptance()


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("shape", ((), (2, 3), (2, 1, 3)))
@pytest.mark.parametrize("length", (0, 1, 4))
@pytest.mark.parametrize("opset", (23, 26))
def test_original_body_and_exact_ordered_loop_ports(
    dtype: Any, shape: tuple[int, ...], length: int, opset: int
) -> None:
    model = _export(specs=_specs(length, dtype, shape), opset=opset)
    ledger = _validated(model, 4)
    assert len(ledger.scan_calls) == len(ledger.regions) == 1
    assert len(ledger.scopes) == 2
    child = next(scope for scope in ledger.scopes if scope.region is not None)
    assert child.region.key.role == "scan-body"
    assert len(child.equations) == 3
    loop = _loops(model)[0]
    body = loop.attributes["body"].as_graph()
    assert len(loop.inputs) == len(body.inputs) == 7
    assert len(loop.outputs) == 8 and len(body.outputs) == 9
    assert (
        _storage(loop.inputs[0]).shape == ()
        and _storage(loop.inputs[0]).dtype == np.int64
    )
    assert _storage(loop.inputs[0]).item() == length
    assert _storage(loop.inputs[1]).tobytes() == np.asarray(True, np.bool_).tobytes()
    gathers = [node for node in body if node.op_type == "Gather"]
    assert len(gathers) == 2
    assert [node.inputs[0] for node in gathers] == list(body.inputs[5:7])
    assert all(
        node.inputs[1] is body.inputs[0] and node.attributes["axis"].as_int() == 0
        for node in gathers
    )
    assert all(
        tuple(value.shape) == (length, *shape)
        for value in (model.graph.outputs[2], model.graph.outputs[4])
    )
    if length == 0:
        assert all(_storage(value).nbytes == 0 for value in model.graph.outputs[2:])


@pytest.mark.parametrize("length", (0, 1, 4))
@pytest.mark.parametrize("unroll", (0, 1, 2, 4))
@pytest.mark.parametrize("opset", (23, 26))
def test_no_xs_preserves_original_length_and_unroll_metadata(
    length: int, unroll: int, opset: int
) -> None:
    def target(value: Any, step: Any) -> Any:
        return jax.lax.scan(
            lambda carry, _: (carry + step, carry),
            value,
            None,
            length=length,
            unroll=unroll,
        )

    model = _export(target, [jax.ShapeDtypeStruct((2,), np.float32)] * 2, opset=opset)
    ledger = _validated(model, 2)
    rule = next(
        equation.rule
        for equation in ledger.scopes[0].equations
        if type(equation.rule) is ScanRule
    )
    assert rule.length == length and rule.unroll == unroll and rule.sequences == 0
    assert len(_loops(model)[0].inputs) == 4


def test_original_opset21_no_xs_lane_keeps_its_contract() -> None:
    def target(value: Any) -> Any:
        return jax.lax.scan(
            lambda carry, _: (carry + np.float32(1), carry), value, None, length=3
        )

    _validated(_export(target, [jax.ShapeDtypeStruct((), np.float32)], opset=21), 2)


@pytest.mark.parametrize("optimization", ("disabled", "all"))
@pytest.mark.parametrize("length", (0, 1, 4))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_original_samples_and_serialized_reload(
    length: int, dtype: Any, optimization: str
) -> None:
    model = _export(specs=_specs(length, dtype, (2,)))
    _validated(model, 4)
    inputs = (
        np.asarray([1, -2], dtype),
        np.asarray([0.5, 0.25], dtype),
        np.full((length, 2), 0.25, dtype),
        np.full((length, 2), 0.5, dtype),
        np.asarray(True),
    )
    _samples(model, _target, inputs, optimization)
    proto = onnx.load_model_from_string(ir.to_proto(model).SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    reloaded = ir.from_proto(proto)
    assert isinstance(reloaded, ir.Model)
    _samples(reloaded, _target, inputs, optimization)


@pytest.mark.parametrize(
    "route", ("zero-carry", "zero-ys", "passthrough", "literal", "unused-xs")
)
def test_distinct_source_result_and_unused_operand_roles(route: str) -> None:
    specs = [
        jax.ShapeDtypeStruct((2,), np.float32),
        jax.ShapeDtypeStruct((3, 2), np.float32),
    ]

    def target(value: Any, xs: Any) -> Any:
        if route == "zero-carry":
            return jax.lax.scan(lambda carry, x: (carry, x - value), (), xs)
        if route == "zero-ys":
            return jax.lax.scan(lambda carry, x: (carry - x, None), value, xs)
        if route == "passthrough":
            return jax.lax.scan(lambda carry, x: (carry, (x, x, carry)), value, xs)
        if route == "literal":
            return jax.lax.scan(
                lambda carry, x: (carry - x, np.asarray(False)), value, xs
            )
        return jax.lax.scan(lambda carry, x: (carry - value, carry), value, xs)

    model = _export(target, specs)
    ledger = _validated(model)
    loop = _loops(model)[0]
    child = next(scope for scope in ledger.scopes if scope.region is not None)
    if route == "zero-carry":
        rule = next(
            equation.rule
            for equation in ledger.scopes[0].equations
            if type(equation.rule) is ScanRule
        )
        assert rule.carries == 0
    if route == "zero-ys":
        assert len(loop.outputs) == len(loop.inputs) - 2
    if route == "unused-xs":
        assert child.region.captures[-1].unused
    inputs = (np.asarray([1, 2], np.float32), np.full((3, 2), 0.25, np.float32))
    _samples(model, target, inputs, "disabled")


@pytest.mark.parametrize(
    "composition",
    (
        "scan-in-if",
        "if-in-scan",
        "scan-in-while",
        "while-in-scan",
        "scan-in-scan",
        "reused-function",
    ),
)
def test_nested_and_reused_original_invocations(composition: str) -> None:

    def target(value: Any, step: Any, xs: Any, other: Any, flag: Any) -> Any:
        if composition == "scan-in-if":
            return jax.lax.cond(
                flag,
                lambda: _target(value, step, xs, other, flag),
                lambda: _target(value, step, other, xs, flag),
            )
        if composition == "reused-function":
            first = scanned(value, step, xs, other, flag)
            return scanned(first[0][0], step, xs, other, flag)
        if composition == "scan-in-while":

            def body(state: Any) -> Any:
                count, payload = state
                result = _target(payload, step, xs, other, flag)
                return count + np.int32(1), result[0][0]

            return jax.lax.while_loop(
                lambda state: state[0] < np.int32(2),
                body,
                (np.asarray(0, np.int32), value),
            )
        if composition == "while-in-scan":

            def body(carry: Any, x: Any) -> Any:
                updated = jax.lax.while_loop(
                    lambda state: state[0] < np.int32(2),
                    lambda state: (state[0] + np.int32(1), state[1] - step),
                    (np.asarray(0, np.int32), carry),
                )[1]
                return updated, x

            return jax.lax.scan(body, value, xs)
        if composition == "scan-in-scan":

            def body(carry: Any, x: Any) -> Any:
                updated, _ = jax.lax.scan(
                    lambda state, _: (state - step, state), carry, None, length=2
                )
                return updated, x

            return jax.lax.scan(body, value, xs)
        return jax.lax.scan(
            lambda carry, x: (
                jax.lax.cond(flag, lambda a, b: a - b, lambda a, b: a + b, carry, step),
                x,
            ),
            value,
            xs,
        )

    model = _export(target)
    ledger = _validated(model)
    assert ledger.scan_calls
    if composition == "reused-function":
        assert len(ledger.function_calls) == 2
    if "while" in composition:
        assert ledger.while_calls
    inputs = (
        np.ones((2, 3), np.float32),
        np.full((2, 3), 0.25, np.float32),
        np.full((3, 2, 3), 0.5, np.float32),
        np.full((3, 2, 3), 0.25, np.float32),
        np.asarray(True),
    )
    _samples(model, target, inputs, "disabled")


@pytest.mark.parametrize("declared", (False, True))
def test_named_runtime_flag_has_exact_explicit_source_ports(declared: bool) -> None:

    def target(value: Any, step: Any, *, deterministic: Any = True) -> Any:
        return jax.lax.scan(
            lambda carry, _: (
                selected(carry, step, deterministic=deterministic),
                carry,
            ),
            value,
            None,
            length=2,
        )

    model = _export(
        target,
        [jax.ShapeDtypeStruct((2,), np.float32)] * 2,
        input_params={"deterministic": True} if declared else {},
    )
    ledger = _validated(model)
    # SOURCE traces declared runtime parameters as ordinary original operands.
    # This control does not qualify positive lazy-tail numerical behavior.
    assert all(not attachment.lazy for attachment in ledger.scan_calls.values())
    assert ("deterministic" in {value.name for value in model.graph.inputs}) is declared
    for flag in (False, True) if declared else (True,):
        arrays = (np.ones((2,), np.float32), np.full((2,), 0.25, np.float32))
        with jax.enable_x64(True):
            expected = [
                np.asarray(value)
                for value in jax.tree.leaves(target(*arrays, deterministic=flag))
            ]
        feeds = {
            value.name: array
            for value, array in zip(model.graph.inputs[:2], arrays, strict=True)
        }
        if declared:
            feeds["deterministic"] = np.asarray(flag)
        report = validate_source_samples(
            model,
            samples=[SourceSample(feeds, expected, SampleCriteria(mode="bitwise"))],
            optimization="disabled",
        )
        report.require_sample_acceptance()


@pytest.mark.parametrize(
    "fault",
    (
        "length",
        "false-condition",
        "condition-passthrough",
        "gather-axis",
        "gather-iteration",
        "gather-sequence",
        "constant-input",
        "constant-output",
        "sequence-output",
        "carry-output",
        "carry-order",
        "ys-order",
        "header-dtype",
        "header-rank",
        "extra-attribute",
    ),
)
def test_actual_loop_mutations_fail_with_original_denominator(fault: str) -> None:
    model = _export()
    _validated(model, 4)
    loop, body = _target_loop_roles(model)
    gathers = [node for node in body if node.op_type == "Gather"]
    assert len(gathers) == 2 and gathers[0].inputs[0] is body.inputs[5]
    if fault == "length":
        _replace_storage(loop.inputs[0], np.asarray(4, np.int64))
    elif fault == "false-condition":
        _replace_storage(loop.inputs[1], np.asarray(False))
    elif fault == "condition-passthrough":
        _replace_output(body, 0, body.outputs[0], body.inputs[2])
    elif fault == "gather-axis":
        gathers[0].attributes["axis"] = ir.AttrInt64("axis", 1)
    elif fault == "gather-iteration":
        # A scalar INT64 of the same type is still not the iteration formal.
        replacement = ir.Value(
            name="wrong_iteration",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(()),
            const_value=ir.tensor(np.asarray(0, np.int64)),
        )
        body.register_initializer(replacement)
        _replace_input(gathers[0], 1, body.inputs[0], replacement)
    elif fault == "gather-sequence":
        _replace_input(gathers[0], 0, body.inputs[5], body.inputs[6])
    elif fault == "constant-input":
        _replace_input(loop, 3, model.graph.inputs[1], model.graph.inputs[0])
    elif fault == "constant-output":
        _replace_output(body, 2, body.outputs[2], body.inputs[4])
    elif fault == "sequence-output":
        _replace_output(body, 4, body.outputs[4], body.inputs[6])
    elif fault == "carry-output":
        _replace_output(body, 3, body.outputs[3], body.inputs[3])
    elif fault == "carry-order":
        # Substitute the actual updated payload carry, not the step constant.
        _replace_input(loop, 4, model.graph.inputs[0], model.graph.inputs[1])
    elif fault == "ys-order":
        _replace_output(body, 6, body.outputs[6], body.outputs[2])
    elif fault == "header-dtype":
        body.inputs[0].type = ir.TensorType(ir.DataType.INT32)
    elif fault == "header-rank":
        body.inputs[0].shape = ir.Shape((1,))
    else:
        loop.attributes["unreviewed"] = ir.AttrInt64("unreviewed", 0)
    proof = _proof(model)
    assert proof.status == "failed", proof
    assert proof.expected_equations == 4


@pytest.mark.parametrize("dtype", (np.bool_, np.int32, np.float32, np.float64))
@pytest.mark.parametrize("shape", ((0,), (0, 2), (2, 0, 3)))
def test_empty_storage_contract_retains_exact_dtype_and_shape(
    dtype: Any, shape: tuple[int, ...]
) -> None:
    array = np.empty(shape, dtype)
    contract = structure.ConstantContract.from_array(array)
    assert contract == structure.ConstantContract(
        np.dtype(dtype).str, shape, 0, hashlib.sha256(b"").hexdigest()
    )
    assert contract != structure.ConstantContract.from_array(
        np.empty((*shape, 1), dtype)
    )
    assert contract != structure.ConstantContract.from_array(np.empty(shape, np.uint8))


def test_empty_storage_contract_preserves_a_strided_view() -> None:
    noncontiguous = np.arange(24, dtype=np.float32).reshape(4, 6).T
    assert not noncontiguous.flags.c_contiguous
    empty = noncontiguous[:0, ::2]
    assert empty.shape == (0, 2) and empty.nbytes == 0
    assert empty.strides == (noncontiguous.strides[0], noncontiguous.strides[1] * 2)
    assert structure.ConstantContract.from_array(empty) == structure.ConstantContract(
        empty.dtype.str, (0, 2), 0, hashlib.sha256(b"").hexdigest()
    )


@pytest.mark.parametrize(
    "fault", ("shape", "dtype", "storage", "body-edge", "body-receipt")
)
def test_zero_trip_empty_storage_does_not_hide_original_body(fault: str) -> None:
    model = _export(specs=_specs(0))
    ledger = _validated(model, 4)
    empty = model.graph.outputs[2]
    if fault == "shape":
        empty.shape = ir.Shape((0, 3, 2))
        _replace_storage(empty, np.empty((0, 3, 2), np.float32))
    elif fault == "dtype":
        empty.type = ir.TensorType(ir.DataType.DOUBLE)
        _replace_storage(empty, np.empty((0, 2, 3), np.float64))
    elif fault == "storage":
        _replace_storage(empty, np.ones((1, 2, 3), np.float32))
    elif fault == "body-edge":
        body = _loops(model)[0].attributes["body"].as_graph()
        operation = next(node for node in body if node.op_type == "Sub")
        assert operation.inputs[1] is not None
        _replace_input(operation, 1, operation.inputs[1], body.inputs[3])
    else:
        child = next(scope for scope in ledger.scopes if scope.region is not None)
        child.receipts.clear()
    proof = _proof(model)
    assert proof.status == (
        "unverified" if fault == "body-receipt" else "failed"
    ), proof
    assert proof.expected_equations == 4


@pytest.mark.parametrize(
    "fault",
    (
        "attachment",
        "region",
        "scope",
        "body-receipt",
        "wrong-role",
        "duplicate",
        "foreign-owner",
        "lazy-port",
    ),
)
def test_missing_or_changed_evidence_never_shrinks_original_body(fault: str) -> None:
    model = _export()
    ledger = _validated(model, 4)
    child = next(scope for scope in ledger.scopes if scope.region is not None)
    handle, attachment = next(iter(ledger.scan_calls.items()))
    if fault == "attachment":
        ledger.scan_calls.clear()
    elif fault == "region":
        ledger.regions.pop(child.region.key)
    elif fault == "scope":
        ledger.scopes.remove(child)
    elif fault == "body-receipt":
        child.receipts.clear()
    elif fault == "wrong-role":
        ledger.scan_calls[handle] = replace(
            attachment, region=replace(attachment.region, role="body")
        )
    elif fault == "duplicate":
        ledger.scopes.append(replace(child, invocation=object()))
    elif fault == "foreign-owner":
        ledger.scan_calls[handle] = replace(attachment, owner=object())
    else:
        ledger.scan_calls[handle] = replace(
            attachment, lazy=((child.input_values[0], child.input_values[0]),)
        )
    proof = _proof(model)
    assert proof.status in ("failed", "unverified"), proof
    assert proof.expected_equations == 4 and proof.checked_equations < 4


@pytest.mark.parametrize("fault", ("concrete", "rank", "symbol", "unknown-operation"))
def test_actual_shape_facts_are_independent_of_expected_avals(fault: str) -> None:
    model = _export()
    _validated(model, 4)
    body = _loops(model)[0].attributes["body"].as_graph()
    operation = next(node for node in body if node.op_type == "Sub")
    if fault == "concrete":
        operation.outputs[0].shape = ir.Shape((3, 2))
    elif fault == "rank":
        operation.outputs[0].shape = ir.Shape((None,))
    elif fault == "symbol":
        operation.outputs[0].shape = ir.Shape(("B", 3))
    else:
        operation.op_type = "UnreviewedShapeOperator"
        operation.outputs[0].shape = ir.Shape((None, None))
    proof = _proof(model)
    assert proof.status == (
        "unverified" if fault in ("symbol", "unknown-operation") else "failed"
    ), proof
    assert proof.expected_equations == 4


def test_rank_only_body_metadata_is_derived_without_writes() -> None:
    model = _export()
    _validated(model, 4)
    body = _loops(model)[0].attributes["body"].as_graph()
    values = [
        *body.inputs,
        *body.outputs,
        *(value for node in body for value in node.outputs),
    ]
    for value in values:
        if value.shape is not None and len(value.shape):
            value.shape = ir.Shape((None,) * len(value.shape))
    before = [
        (value, tuple(value.shape) if value.shape is not None else None)
        for value in values
    ]
    _validated(model, 4)
    assert before == [
        (value, tuple(value.shape) if value.shape is not None else None)
        for value in values
    ]


def test_final_graph_clone_preserves_scan_and_nested_if_owners() -> None:
    def target(value: Any, step: Any, xs: Any, other: Any, flag: Any) -> Any:
        return jax.lax.scan(
            lambda carry, x: (
                jax.lax.cond(flag, lambda a, b: a - b, lambda a, b: a + b, carry, step),
                x,
            ),
            value,
            xs,
        )

    original = _export(target)
    ledger = _validated(original)
    cloned = ir.Model(
        original.graph.clone(allow_outer_scope_values=True),
        ir_version=original.ir_version,
    )
    cloned.meta.update(original.meta)
    assert _validated(cloned) is ledger
    assert _loops(cloned)[0] is not _loops(original)[0]
    assert (
        _loops(cloned)[0].attributes["body"].as_graph()
        is not _loops(original)[0].attributes["body"].as_graph()
    )


@pytest.mark.parametrize(
    "fault", ("extra-node", "extra-initializer", "gather-after-use")
)
def test_coverage_and_actual_producer_order_remain_required(fault: str) -> None:
    model = _export()
    _validated(model, 4)
    body = _loops(model)[0].attributes["body"].as_graph()
    if fault == "extra-node":
        output = ir.Value(
            name="uncovered", type=body.inputs[2].type, shape=body.inputs[2].shape
        )
        body.append(ir.Node("", "Identity", [body.inputs[2]], outputs=[output]))
    elif fault == "extra-initializer":
        body.register_initializer(
            ir.Value(
                name="uncovered",
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape((2, 3)),
                const_value=ir.tensor(np.ones((2, 3), np.float32)),
            )
        )
    else:
        gather = next(node for node in body if node.op_type == "Gather")
        consumer = next(
            node
            for node in body
            if any(value is gather.outputs[0] for value in node.inputs)
        )
        body.insert_after(consumer, gather)
    proof = _proof(model)
    assert proof.status == (
        "failed" if fault == "gather-after-use" else "unverified"
    ), proof
    assert proof.expected_equations == 4


@pytest.mark.parametrize(
    "mutation", ("length", "unroll", "reverse", "unknown", "linear")
)
def test_original_freeze_preserves_native_metadata_without_retracing(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with jax.enable_x64(True):
        native = jax.make_jaxpr(_target)(*_specs())
    monkeypatch.setattr(
        jax,
        "make_jaxpr",
        lambda *args, **kwargs: pytest.fail("Scan freezer retraced source"),
    )
    monkeypatch.setattr(
        jax,
        "eval_shape",
        lambda *args, **kwargs: pytest.fail("Scan freezer evaluated source"),
    )
    declaration = regions.declare_body(native)
    frozen = regions.freeze_closed_body(native)
    assert declaration.expected_equations == frozen.expected_equations == 4
    assert frozen.gaps == ()
    scan = next(equation for equation in frozen.equations if equation.children)
    assert tuple(role for role, _ in scan.children) == ("scan-body",)
    assert scan.semantics[:7] == ("scan", 2, 1, 2, 3, 3, 1)
    before = hash(frozen)
    equation = next(
        equation
        for equation in native.jaxpr.eqns
        if equation.primitive is jax.lax.scan_p
    )
    if mutation == "length":
        monkeypatch.setitem(equation.params, "length", 4)
    elif mutation == "unroll":
        monkeypatch.setitem(equation.params, "unroll", 2)
    elif mutation == "reverse":
        monkeypatch.setitem(equation.params, "reverse", True)
    elif mutation == "unknown":
        monkeypatch.setitem(equation.params, "unreviewed", False)
    elif "linear" in equation.params:
        monkeypatch.setitem(equation.params, "linear", (False,))
    else:
        monkeypatch.setitem(equation.params, "linear", ())
    changed = regions.freeze_closed_body(native)
    assert changed.expected_equations == 4 and hash(frozen) == before
    if mutation == "unroll":
        assert not changed.gaps and changed != frozen
    else:
        assert "scan-attachment-contract-unavailable" in changed.gaps


def test_scan_ledger_retains_no_native_array_or_ir_authority() -> None:
    ledger = _validated(_export(), 4)

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


def test_observation_preserves_emission_and_legacy_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled = _export()
    _validated(enabled, 4)
    with monkeypatch.context() as patch:
        patch.setattr(ir_context, "SourceStructureLedger", lambda: None)
        disabled = _export()
    assert _proof(disabled).reasons == ("source-ledger-unavailable",)
    assert ir.to_proto(enabled).SerializeToString(deterministic=True) == ir.to_proto(
        disabled
    ).SerializeToString(deterministic=True)
    with jax.enable_x64(True):
        legacy = ui._to_onnx(_target, _specs(), return_mode="ir")
    assert isinstance(legacy, ir.Model)
    loop = _loops(legacy)[0]
    assert _storage(loop.inputs[0]).item() == 3
    assert _storage(loop.inputs[1]).item() is True


@pytest.mark.parametrize("selected", ("carry", "ys", "one-of-repeated"))
def test_pruned_user_results_preserve_full_original_scan_arity(selected: str) -> None:
    def target(value: Any, step: Any, xs: Any, other: Any, keep: Any) -> Any:
        carry, ys = _target(value, step, xs, other, keep)
        if selected == "carry":
            return carry[0]
        if selected == "ys":
            return ys[0], ys[1]
        return ys[2]

    model = _export(target)
    ledger = _validated(model, 4)
    loop = _loops(model)[0]
    rule = next(
        equation.rule
        for equation in ledger.scopes[0].equations
        if type(equation.rule) is ScanRule
    )
    assert rule.constants == 2 and rule.carries == 1 and rule.results == 3
    assert len(loop.outputs) == 8


@pytest.mark.parametrize("fault", ("signed-zero", "nan-payload"))
def test_captured_storage_bits_are_checked_inside_the_scan_contract(fault: str) -> None:
    captured = np.asarray([0x80000000, 0x7FC00001], np.uint32).view(np.float32)

    def target(value: Any, xs: Any) -> Any:
        return jax.lax.scan(lambda carry, x: (carry - captured, x), value, xs)

    model = _export(
        target,
        [
            jax.ShapeDtypeStruct((2,), np.float32),
            jax.ShapeDtypeStruct((3, 2), np.float32),
        ],
    )
    _validated(model, 2)
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
                found.append(value)
    assert found
    changed = captured.copy().view(np.uint32)
    changed[0 if fault == "signed-zero" else 1] = (
        0 if fault == "signed-zero" else 0x7FC00002
    )
    _replace_storage(found[0], changed.view(np.float32))
    proof = _proof(model)
    assert proof.status == "failed", proof
    assert proof.expected_equations == 2


@pytest.mark.parametrize("fault", ("input-order", "output-order"))
def test_two_updated_same_dtype_carries_reject_actual_reordering(fault: str) -> None:
    def target(left: Any, right: Any, xs: Any) -> Any:
        def body(state: Any, x: Any) -> Any:
            first, second = state
            return (first - x, second + first), (first, second)

        return jax.lax.scan(body, (left, right), xs)

    model = _export(
        target,
        [jax.ShapeDtypeStruct((2,), np.float32)] * 2
        + [jax.ShapeDtypeStruct((3, 2), np.float32)],
    )
    ledger = _validated(model, 3)
    rule = next(
        equation.rule
        for equation in ledger.scopes[0].equations
        if type(equation.rule) is ScanRule
    )
    assert (rule.constants, rule.carries, rule.sequences) == (0, 2, 1)
    loop = _loops(model)[0]
    body = loop.attributes["body"].as_graph()
    assert (
        loop.inputs[2] is model.graph.inputs[0]
        and loop.inputs[3] is model.graph.inputs[1]
    )
    first_update, second_update = body.outputs[1].producer(), body.outputs[2].producer()
    assert first_update is not None and first_update.op_type == "Sub"
    assert second_update is not None and second_update.op_type == "Add"
    if fault == "input-order":
        first, second = loop.inputs[2:4]
        assert first is not None and second is not None
        _replace_input(loop, 2, first, second)
        _replace_input(loop, 3, second, first)
    else:
        first, second = body.outputs[1:3]
        _replace_output(body, 1, first, second)
        _replace_output(body, 2, second, first)
    proof = _proof(model)
    assert proof.status == "failed", proof
    assert proof.expected_equations == 3


def _unused_xs_export(forwarded: bool = False) -> tuple[ir.Model, Any]:
    def target(value: Any, step: Any, xs: Any) -> Any:
        return jax.lax.scan(
            lambda carry, x: (carry - step, x if forwarded else carry), value, xs
        )

    specs = [jax.ShapeDtypeStruct((2,), np.float32)] * 2 + [
        jax.ShapeDtypeStruct((3, 2), np.float32)
    ]
    return _export(target, specs), target


@pytest.mark.parametrize("forwarded", (False, True))
def test_normal_dce_unused_and_native_forwarded_xs_preserve_full_state(
    forwarded: bool,
) -> None:
    model, target = _unused_xs_export(forwarded)
    ledger = _validated(model, 2)
    child = next(scope for scope in ledger.scopes if scope.region is not None)
    rule = next(
        equation.rule
        for equation in ledger.scopes[0].equations
        if type(equation.rule) is ScanRule
    )
    assert (rule.constants, rule.carries, rule.sequences) == (1, 1, 1)
    assert rule.results == (0 if forwarded else 1)
    assert len(child.body.equations) == 1 and len(child.body.inputs) == 3
    assert structure._unused_inputs(child.body) == frozenset({2})
    port = child.region.captures[2]
    assert port.unused and port.local == child.input_values[2]
    loop = _loops(model)[0]
    body = loop.attributes["body"].as_graph()
    assert len(loop.inputs) == len(body.inputs) == 5
    assert loop.inputs[4] is model.graph.inputs[2]
    assert not any(node.op_type == "Gather" for node in body)
    assert port.local not in structure._resolve_bindings(
        body, {port.local}, frozenset({port.local})
    )
    assert _identity_source(body.outputs[3]) is body.inputs[4]
    assert (
        len(loop.outputs) == 3 + rule.results and len(body.outputs) == 4 + rule.results
    )
    inputs = (
        np.asarray([1, 2], np.float32),
        np.asarray([0.25, 0.5], np.float32),
        np.full((3, 2), 0.75, np.float32),
    )
    _samples(model, target, inputs, "disabled")


@pytest.mark.parametrize(
    "fault", ("missing-used-gather", "forged-used-as-unused", "forged-unused-as-used")
)
def test_xs_absence_requires_the_independent_original_use_def_proof(fault: str) -> None:
    model = _unused_xs_export()[0] if fault == "forged-unused-as-used" else _export()
    expected = 2 if fault == "forged-unused-as-used" else 4
    ledger = _validated(model, expected)
    child = next(scope for scope in ledger.scopes if scope.region is not None)
    if fault == "missing-used-gather":
        body = _loops(model)[0].attributes["body"].as_graph()
        gather = next(node for node in body if node.op_type == "Gather")
        assert gather.outputs[0].uses()
        body.remove(gather, safe=False)
        assert gather not in set(body)
    else:
        index = len(child.region.captures) - 1
        original = child.region.captures[index]
        assert original.unused is (fault == "forged-unused-as-used")
        ports = list(child.region.captures)
        ports[index] = replace(original, unused=not original.unused)
        child.region = replace(child.region, captures=tuple(ports))
        assert child.region.captures[index].unused is not original.unused
    proof = _proof(model)
    assert proof.status in ("failed", "unverified"), proof
    assert proof.expected_equations == expected and proof.checked_equations < expected


@pytest.mark.parametrize(
    "fault", ("alternate-producer", "wrong-gather", "extra-gather", "ambiguous-marker")
)
def test_surviving_unused_xs_producers_require_exact_gather_and_coverage(
    fault: str,
) -> None:
    model, _ = _unused_xs_export()
    ledger = _validated(model, 2)
    child = next(scope for scope in ledger.scopes if scope.region is not None)
    port = child.region.captures[2]
    body = _loops(model)[0].attributes["body"].as_graph()
    assert port.unused and not any(node.op_type == "Gather" for node in body)

    def append_producer(name: str, *, marked: bool) -> ir.Node:
        output = ir.Value(name=name, type=body.inputs[3].type, shape=ir.Shape((2,)))
        if fault == "alternate-producer":
            node = ir.Node("", "Identity", [body.inputs[3]], outputs=[output])
        else:
            node = ir.Node(
                "",
                "Gather",
                [body.inputs[4], body.inputs[0]],
                attributes=[ir.AttrInt64("axis", 1 if fault == "wrong-gather" else 0)],
                outputs=[output],
            )
        # Deliberately forged observation markers test the retained-definition
        # checks; optional absence must not bypass a surviving bad producer.
        if marked:
            node.meta[structure._ORIGIN] = port.local.origin
            output.meta[structure._VALUES] = (port.local,)
        body.append(node)
        return node

    first = append_producer("surviving_unused_slice", marked=fault != "extra-gather")
    assert first in set(body)
    if fault == "ambiguous-marker":
        second = append_producer("duplicate_unused_slice", marked=True)
        assert first is not second
    proof = _proof(model)
    assert proof.expected_equations == 2
    if fault == "extra-gather":
        assert (
            proof.status == "unverified"
            and "artifact-node-coverage-incomplete" in proof.reasons
        ), proof
        assert proof.checked_equations == 2
    elif fault == "ambiguous-marker":
        assert proof.status == "unverified" and proof.reasons == (
            "final-value-witness-unavailable",
        ), proof
        assert proof.checked_equations == 0
    else:
        assert proof.status == "failed" and proof.reasons == (
            "scan-sequence-gather-edge-or-axis-differs",
        ), proof
        assert proof.checked_equations == 0


@pytest.mark.parametrize("optimization", ("disabled", "all"))
@pytest.mark.parametrize("unroll", (0, 2))
def test_xs_nondefault_unroll_samples_and_serialized_reload(
    unroll: int, optimization: str
) -> None:
    def target(value: Any, step: Any, xs: Any) -> Any:
        return jax.lax.scan(
            lambda carry, x: ((carry + step) - x, carry), value, xs, unroll=unroll
        )

    specs = [jax.ShapeDtypeStruct((2,), np.float32)] * 2 + [
        jax.ShapeDtypeStruct((3, 2), np.float32)
    ]
    model = _export(target, specs)
    ledger = _validated(model, 3)
    rule = next(
        equation.rule
        for equation in ledger.scopes[0].equations
        if type(equation.rule) is ScanRule
    )
    assert rule.unroll == unroll and rule.sequences == 1
    inputs = (
        np.asarray([1, -2], np.float32),
        np.asarray([0.5, 0.25], np.float32),
        np.full((3, 2), 0.125, np.float32),
    )
    _samples(model, target, inputs, optimization)
    proto = onnx.load_model_from_string(ir.to_proto(model).SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    reloaded = ir.from_proto(proto)
    assert isinstance(reloaded, ir.Model)
    _samples(reloaded, target, inputs, optimization)


@pytest.mark.parametrize("optimization", ("disabled", "all"))
@pytest.mark.parametrize("length", (0, 3))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_no_xs_samples_and_serialized_reload(
    length: int, dtype: Any, optimization: str
) -> None:
    def target(value: Any, step: Any) -> Any:
        return jax.lax.scan(
            lambda carry, _: (carry + step, carry), value, None, length=length, unroll=2
        )

    model = _export(target, [jax.ShapeDtypeStruct((2,), dtype)] * 2)
    ledger = _validated(model, 2)
    rule = next(
        equation.rule
        for equation in ledger.scopes[0].equations
        if type(equation.rule) is ScanRule
    )
    assert rule.length == length and rule.unroll == 2 and rule.sequences == 0
    inputs = (np.asarray([1, -2], dtype), np.asarray([0.5, 0.25], dtype))
    _samples(model, target, inputs, optimization)
    proto = onnx.load_model_from_string(ir.to_proto(model).SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    reloaded = ir.from_proto(proto)
    assert isinstance(reloaded, ir.Model)
    _samples(reloaded, target, inputs, optimization)


@pytest.mark.parametrize("route", ("zero-trip-zero-carry", "pure-no-result"))
def test_normal_dce_missing_whole_loop_keeps_original_denominator_unavailable(
    route: str, record_property: Any
) -> None:
    length = 0 if route == "zero-trip-zero-carry" else 3

    def target(value: Any, xs: Any) -> Any:
        if route == "zero-trip-zero-carry":
            return jax.lax.scan(lambda carry, x: (carry, x - value), (), xs)
        return jax.lax.scan(lambda carry, x: (carry, None), (), xs)

    specs = [
        jax.ShapeDtypeStruct((2,), np.float32),
        jax.ShapeDtypeStruct((length, 2), np.float32),
    ]
    with jax.enable_x64(True):
        native = jax.make_jaxpr(target)(*specs)
    scans = [
        equation
        for equation in native.jaxpr.eqns
        if equation.primitive is jax.lax.scan_p
    ]
    original = regions.freeze_closed_body(native)
    # Native pre-freeze elision has no original Scan denominator. Record that
    # distinct boundary explicitly; it does not qualify missing-Loop evidence.
    if not scans:
        record_property(
            "native_scan_boundary",
            "elided-before-original-freeze; missing-Loop qualification unavailable",
        )
        assert not original.equations and original.expected_equations == 0
        pytest.skip(
            "Native tracing elided Scan before the original frozen body; no missing-Loop proof is claimed"
        )
    assert len(scans) == 1
    record_property(
        "native_scan_boundary",
        "retained-original-Scan; missing final Loop remains unavailable",
    )
    expected = 2 if route == "zero-trip-zero-carry" else 1
    assert original.expected_equations == expected
    model = _export(target, specs)
    ledger = model.meta[structure._LEDGER]
    assert ledger.expected_equations == expected and len(ledger.scan_calls) == 1
    assert not _loops(model)
    proof = _proof(model)
    assert proof.status == "unverified", proof
    assert proof.reasons == ("scope-owner-coverage-incomplete",)
    assert proof.expected_equations == expected and proof.checked_equations == 0


def test_final_graph_clone_preserves_reused_function_scan_owners() -> None:

    def target(value: Any, step: Any, xs: Any, other: Any, flag: Any) -> Any:
        first = scanned(value, step, xs, other, flag)
        return scanned(first[0][0], step, xs, other, flag)

    original = _export(target)
    ledger = _validated(original)
    definitions = list(iter_ir_functions(original.functions))
    assert len(ledger.function_calls) == 2 and len(definitions) == 1
    cloned = original.clone()
    cloned.meta.update(original.meta)
    assert _validated(cloned, ledger.expected_equations) is ledger
    cloned_definitions = list(iter_ir_functions(cloned.functions))
    assert len(cloned_definitions) == 1 and cloned_definitions[0] is not definitions[0]
    assert cloned_definitions[0].graph is not definitions[0].graph
    assert _loops(cloned)[0] is not _loops(original)[0]
    assert (
        _loops(cloned)[0].attributes["body"].as_graph()
        is not _loops(original)[0].attributes["body"].as_graph()
    )
    inputs = (
        np.ones((2, 3), np.float32),
        np.full((2, 3), 0.25, np.float32),
        np.full((3, 2, 3), 0.5, np.float32),
        np.full((3, 2, 3), 0.25, np.float32),
        np.asarray(True),
    )
    _samples(cloned, target, inputs, "disabled")


@pytest.mark.parametrize(
    "fault", ("full-sequence-input", "full-sequence-passthrough", "full-sequence-shape")
)
def test_absent_unused_slice_preserves_mandatory_full_sequence_state(
    fault: str,
) -> None:
    model, _ = _unused_xs_export()
    _validated(model, 2)
    loop = _loops(model)[0]
    body = loop.attributes["body"].as_graph()
    assert not any(node.op_type == "Gather" for node in body)
    if fault == "full-sequence-shape":
        body.inputs[4].shape = ir.Shape((3, 3))
    else:
        replacement = ir.Value(
            name="wrong_full_sequence",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape((3, 2)),
            const_value=ir.tensor(np.zeros((3, 2), np.float32)),
        )
        if fault == "full-sequence-input":
            model.graph.register_initializer(replacement)
            _replace_input(loop, 4, model.graph.inputs[2], replacement)
        else:
            body.register_initializer(replacement)
            _replace_output(body, 3, body.outputs[3], replacement)
    proof = _proof(model)
    assert proof.status == "failed", proof
    assert proof.expected_equations == 2 and proof.checked_equations == 0


def _scan_in_while_literal_capture(
    limit: Any, value: Any, step: Any, xs: Any, other: Any, flag: Any
) -> Any:
    def body(state: Any) -> Any:
        count, payload = state
        result = _target(payload, step, xs, other, flag)
        return count + np.int32(1), result[0][0]

    return jax.lax.while_loop(
        lambda state: state[0] < limit, body, (np.asarray(0, np.int32), value)
    )


@pytest.mark.parametrize(
    "fault", ("literal-storage", "capture-binding", "extra-same-bytes")
)
def test_nested_literal_capture_does_not_become_a_local_constvar(fault: str) -> None:
    specs = [jax.ShapeDtypeStruct((), np.int32), *_specs()]
    model = _export(_scan_in_while_literal_capture, specs)
    ledger = _validated(model, 8)
    initial = next(
        scope
        for scope in ledger.scopes
        if scope.region is not None and scope.region.key.role == "initial-condition"
    )
    assert initial.closed_constant_values == () and initial.body.constants == ()
    ports = initial.region.captures
    index = next(
        index
        for index, port in enumerate(ports)
        if isinstance(port.binding, structure.LiteralOperand)
    )
    port = ports[index]
    literal = structure._resolve_bindings(model.graph, {port.outer})[port.outer]
    assert port.local in initial.input_values and literal.is_initializer()
    assert literal.dtype is ir.DataType.INT32 and _storage(literal).item() == 0
    if fault == "literal-storage":
        _replace_storage(literal, np.asarray(1, np.int32))
    elif fault == "capture-binding":
        changed = list(ports)
        changed[index] = replace(
            port,
            binding=replace(
                port.binding,
                storage=structure.ConstantContract.from_array(np.asarray(1, np.int32)),
            ),
        )
        initial.region = replace(initial.region, captures=tuple(changed))
    else:
        model.graph.register_initializer(
            ir.Value(
                name="unclaimed_same_literal",
                type=literal.type,
                shape=literal.shape,
                const_value=ir.tensor(_storage(literal).copy()),
            )
        )
    proof = _proof(model)
    assert proof.expected_equations == 8
    if fault == "extra-same-bytes":
        assert proof.status == "unverified" and proof.checked_equations == 8, proof
        assert proof.reasons == ("artifact-constant-coverage-incomplete",)
    else:
        assert proof.status == "failed", proof


@pytest.mark.parametrize("limit", (0, 2))
def test_captured_var_and_literal_same_bytes_keep_distinct_source_authority(
    limit: int,
) -> None:
    specs = [jax.ShapeDtypeStruct((), np.int32), *_specs()]
    model = _export(_scan_in_while_literal_capture, specs)
    _validated(model, 8)
    inputs = (
        np.asarray(limit, np.int32),
        np.ones((2, 3), np.float32),
        np.full((2, 3), 0.25, np.float32),
        np.full((3, 2, 3), 0.5, np.float32),
        np.full((3, 2, 3), 0.25, np.float32),
        np.asarray(True),
    )
    _samples(model, _scan_in_while_literal_capture, inputs, "disabled")
