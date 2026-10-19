# tests/extra_tests/converter/test_source_square_variance.py

"""Static native real normalization, default SOURCE admission and CPU evidence."""

from collections.abc import Iterator
from dataclasses import asdict
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import jax
from jax import export
import jax.numpy as jnp
from jax.extend.core import jaxpr_as_fun
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
from onnx import numpy_helper
from onnxruntime.capi import _pybind_state  # type: ignore[import-untyped]
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.converter.source_representation import normalized_shape_dimensions
from jax2onnx.precision import CheckStatus, get_precision_report
from jax2onnx.plugins.plugin_system import FunctionPlugin
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.precision_numerical_support import (
    normalization_input,
    normalization_reference,
)

_BOUNDS = {
    "variance": {
        "float16": (0.002, 0.002),
        "float32": (4e-6, 2e-7),
        "float64": (8e-13, 2e-14),
    },
    "std": {
        "float16": (0.002, 0.002),
        "float32": (3e-6, 3e-7),
        "float64": (1e-12, 2e-14),
    },
    "layer_norm": {
        "float16": (0.002, 0.002),
        "float32": (3e-5, 3e-5),
        "float64": (2e-12, 2e-12),
    },
}
_VARIANCE_CASES = [
    ("var", "float32", (-1,), False, 0, None, "rows"),
    ("var", "float64", (-1,), False, 0, None, "rows"),
    ("var", "float16", (-1,), False, 0, None, "rows-div16"),
    ("var", "float32", (0, 2), True, 1, None, "multi"),
    ("var", "float64", (0, 2), False, 1, None, "multi"),
    ("var", "float32", (-1,), True, 8, None, "rows"),
    ("var", "float64", (-1,), False, 8, None, "rows"),
    ("std", "float32", (-1,), True, 0, None, "rows"),
    ("std", "float64", (0, 2), False, 1, None, "multi"),
    ("var", "float16", (-1,), True, 0, "float32", "rows-div16"),
]
_ORIGINAL_NUMPY = (jnp.mean, jnp.sum, jnp.var, jnp.std, jnp.maximum, jnp.sqrt)


def _globals() -> dict[str, Any]:
    return {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }


def _effective() -> tuple[Any, ...]:
    return tuple(getattr(jax.config, name) for name in _globals())


@pytest.fixture(autouse=True)
def source_settings() -> Iterator[None]:
    before, effective = _globals(), _effective()
    policy = active_precision_policy()
    try:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for name, value in before.items():
            jax.config.update(name, value)
        assert (_globals(), _effective()) == (before, effective)
        assert active_precision_policy() is policy
        assert (
            jnp.mean,
            jnp.sum,
            jnp.var,
            jnp.std,
            jnp.maximum,
            jnp.sqrt,
        ) == _ORIGINAL_NUMPY


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _payload(value: Any) -> tuple[Any, ...]:
    array = np.asarray(value)
    return array.dtype.name, array.shape, array.tobytes()


def _fresh(fn: Any) -> Any:
    return lambda *args: fn(*args)


def _context() -> dict[str, Any]:
    return {
        **{name: getattr(jax.config, name) for name in _globals()},
        "hidden": {
            name: {"global": repr(state.get_global()), "local": repr(state.get_local())}
            for name, state in (
                ("jax_enable_x64", jax.enable_x64),
                ("jax_default_matmul_precision", jax.default_matmul_precision),
                ("jax_numpy_dtype_promotion", jax.numpy_dtype_promotion),
                ("jax_numpy_rank_promotion", jax.numpy_rank_promotion),
            )
        },
    }


def _callee_identity(fn: Any) -> tuple[Any, Any]:
    return getattr(fn, "__func__", fn), getattr(fn, "__self__", None)


def _aval(var: Any) -> tuple[Any, ...]:
    return (
        str(var.aval.dtype),
        normalized_shape_dimensions(var.aval.shape),
        bool(var.aval.weak_type),
    )


def _carriers(closed: Any) -> list[Any]:
    body = getattr(closed, "jaxpr", closed)
    found = []
    for equation in body.eqns:
        if equation.primitive.name.startswith("onnx_fn::"):
            found.append(equation)
        if equation.primitive.name == "cond":
            for branch in equation.params["branches"]:
                found.extend(_carriers(branch))
    return found


def _function_outer(native: Any, source: Any, reference: Any, boundary: str) -> None:
    """Prove these two finite callers; retain every original outer trace separately."""
    a, b = native.jaxpr, source.jaxpr
    assert [_aval(v) for v in a.invars] == [_aval(v) for v in b.invars]
    assert [_aval(v) for v in a.outvars] == [_aval(v) for v in b.outvars]
    assert b.constvars == [] or tuple(b.constvars) == ()
    if boundary == "function":
        assert typed_trace_manifest(native) == typed_trace_manifest(reference)
        assert len(b.eqns) == 1 and len(_carriers(source)) == 1
        carrier = b.eqns[0]
        assert list(carrier.invars) == list(b.invars)
        assert list(carrier.outvars) == list(b.outvars)
    else:
        assert boundary == "cond"
        assert (
            [e.primitive.name for e in a.eqns]
            == [e.primitive.name for e in b.eqns]
            == ["gt", "convert_element_type", "cond"]
        )
        for old, new in zip(a.eqns[:2], b.eqns[:2], strict=True):
            assert old.params == new.params
            assert [_aval(v) for v in old.invars] == [_aval(v) for v in new.invars]
            assert [_aval(v) for v in old.outvars] == [_aval(v) for v in new.outvars]
        predicate, conversion, condition = b.eqns
        assert predicate.invars[0] is b.invars[1]
        assert _payload(predicate.invars[1].val) == _payload(np.float32(0))
        assert conversion.invars[0] is predicate.outvars[0]
        assert condition.invars[0] is conversion.outvars[0]
        assert list(condition.invars[1:]) == [b.invars[0]]
        assert list(condition.outvars) == list(b.outvars)
        branches = condition.params["branches"]
        assert len(branches) == 2
        for index, branch in enumerate(branches):
            body = branch.jaxpr
            assert len(body.invars) == 1
            assert len(body.outvars) == len(reference.jaxpr.outvars)
            assert not body.constvars
            carriers = _carriers(branch)
            assert len(carriers) == 1
            call = carriers[0]
            if index == 1:
                assert len(body.eqns) == 1
                assert call.invars[0] is body.invars[0] and list(call.outvars) == list(
                    body.outvars
                )
            else:
                assert [e.primitive.name for e in body.eqns] == [
                    "neg",
                    call.primitive.name,
                    *("neg" for _ in body.outvars),
                ]
                before, _, *after = body.eqns
                assert before.invars[0] is body.invars[0]
                assert call.invars[0] is before.outvars[0]
                for negation, result, output in zip(
                    after, call.outvars, body.outvars, strict=True
                ):
                    assert negation.invars[0] is result
                    assert negation.outvars[0] is output
    assert all(
        set(call.params) <= {"instance_key", "__jax2onnx_source_trace"}
        for call in _carriers(source)
    )


def _serialized_functions(
    model: Any, observations: list[Any], reference: Any, boundary: str
) -> None:
    definitions = {(f.domain, f.name): f for f in model.functions}
    assert definitions and len(definitions) == len(observations)
    calls: list[Any] = []

    def scope(nodes: Any, available: set[str], outputs: Any) -> None:
        for node in nodes:
            assert all(not name or name in available for name in node.input)
            if (node.domain, node.op_type) in definitions:
                function = definitions[(node.domain, node.op_type)]
                assert (
                    len(node.input)
                    == len(function.input)
                    == len(reference.jaxpr.invars)
                )
                assert (
                    len(node.output)
                    == len(function.output)
                    == len(reference.jaxpr.outvars)
                )
                calls.append(node)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    graph = attribute.g
                    assert (
                        node.op_type == "If"
                        and len(node.input) == 1
                        and not graph.input
                    )
                    scope(
                        graph.node,
                        available | {v.name for v in graph.initializer},
                        graph.output,
                    )
            assert not any(name in available for name in node.output if name)
            available.update(node.output)
        assert all(
            (value if isinstance(value, str) else value.name) in available
            for value in outputs
        )

    scope(
        model.graph.node,
        {v.name for v in model.graph.input} | {v.name for v in model.graph.initializer},
        model.graph.output,
    )
    for function in model.functions:
        assert len(set(function.input)) == len(function.input)
        assert len(set(function.output)) == len(function.output)
        scope(function.node, set(function.input), function.output)
    assert {(node.domain, node.op_type) for node in calls} == set(definitions)
    assert len(calls) == (2 if boundary == "cond" else 1)


class _SerializedScope:
    def __init__(self, graph: Any, parent: Any = None) -> None:
        self.graph = graph
        self.nodes = list(graph.node)
        self.producers = {output: node for node in self.nodes for output in node.output}
        self.types = {} if parent is None else dict(parent.types)
        self.constants = {} if parent is None else dict(parent.constants)
        for value in (*graph.input, *graph.output, *graph.value_info):
            if isinstance(value, str):
                continue
            tensor = value.type.tensor_type
            assert all(d.HasField("dim_value") for d in tensor.shape.dim)
            self.types[value.name] = (
                tensor.elem_type,
                tuple(d.dim_value for d in tensor.shape.dim),
            )
        for value in getattr(graph, "initializer", ()):
            self.constants[value.name] = numpy_helper.to_array(value)
        for node in self.nodes:
            if node.op_type == "Constant" and not node.domain:
                attributes = {a.name: a for a in node.attribute}
                assert set(attributes) == {"value"}
                assert len(node.output) == 1
                self.constants[node.output[0]] = numpy_helper.to_array(
                    attributes["value"].t
                )
        for name, value in self.constants.items():
            self.types[name] = (
                onnx.helper.np_dtype_to_tensor_dtype(value.dtype),
                value.shape,
            )

    def origin(self, name: str) -> str:
        """Only Identity is an alias, with unchanged recorded type and shape."""
        visited = set()
        while name in self.producers and self.producers[name].op_type == "Identity":
            assert name not in visited
            visited.add(name)
            node = self.producers[name]
            assert not node.domain and len(node.input) == len(node.output) == 1
            assert self.types[name] == self.types[node.input[0]]
            name = node.input[0]
        return name

    def same(self, actual: str, expected: str) -> None:
        assert self.origin(actual) == self.origin(expected)

    def op(self, name: str, operation: str, arity: int) -> Any:
        node = self.producers[self.origin(name)]
        assert not node.domain and node.op_type == operation
        assert len(node.input) == arity and len(node.output) == 1
        return node

    def typed(self, name: str, dtype: int, shape: tuple[int, ...]) -> None:
        assert self.types[name] == (dtype, shape)

    def constant(self, name: str, expected: Any) -> None:
        value = self.constants[self.origin(name)]
        expected = np.asarray(expected)
        assert (value.dtype, value.shape, value.tobytes()) == (
            expected.dtype,
            expected.shape,
            expected.tobytes(),
        )


def _serialized_shape(scope: Any, name: str, expected: tuple[int, ...]) -> None:
    name = scope.origin(name)
    scope.typed(name, onnx.TensorProto.INT64, (len(expected),))
    if name in scope.constants:
        scope.constant(name, np.asarray(expected, dtype=np.int64))
        return
    node = scope.op(name, "Concat", len(expected))
    assert {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute} == {
        "axis": 0
    }
    for operand, dimension in zip(node.input, expected, strict=True):
        scope.constant(operand, np.asarray([dimension], dtype=np.int64))


def _serialized_reshape(
    scope: _SerializedScope, name: str, shape: tuple[int, ...]
) -> str:
    node = scope.op(name, "Reshape", 2)
    assert not node.attribute
    _serialized_shape(scope, node.input[1], shape)
    assert scope.types[name][1] == shape
    assert scope.types[name][0] == scope.types[node.input[0]][0]
    return scope.origin(node.input[0])


def _serialized_expand(scope: Any, name: str, shape: tuple[int, ...]) -> str:
    node = scope.op(name, "Expand", 2)
    assert not node.attribute
    _serialized_shape(scope, node.input[1], shape)
    assert scope.types[name] == scope.types[node.input[0]]
    return _serialized_reshape(scope, node.input[0], shape)


def _serialized_reduction(scope: Any, name: str, operand: str) -> None:
    node = scope.op(name, "ReduceSum", 2)
    scope.same(node.input[0], operand)
    scope.constant(node.input[1], np.asarray([1], dtype=np.int64))
    assert {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute} == {
        "keepdims": 0
    }


def _serialized_definition(
    model: Any, observations: list[Any], expected_name: str
) -> Any:
    # These selected artifacts intentionally contain one observed definition.
    assert len(observations) == len(model.functions) == 1
    observation = observations[0]
    assert observation["function_name"] == expected_name
    assert observation["qualified_target"].rsplit(".", 1)[1] == expected_name
    assert observation["equation_primitive"] == (
        "onnx_fn::" + observation["qualified_target"]
    )
    namespace = observation["namespace"] or "custom"
    function = model.functions[0]
    assert (function.domain, function.name) == (
        f"{namespace}.{expected_name}.1",
        expected_name,
    )
    scope = _SerializedScope(function)
    for names, ports in (
        (function.input, observation["input_ports"]),
        (function.output, observation["result_ports"]),
    ):
        assert len(names) == len(ports)
        for name, (dtype, shape, weak) in zip(names, ports, strict=True):
            assert not weak
            scope.typed(
                name,
                onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype)),
                tuple(shape),
            )
    return function


def _serialized_call(
    scope: Any, function: Any, operand: str, outputs: list[str]
) -> None:
    calls = [
        node
        for node in scope.nodes
        if (node.domain, node.op_type) == (function.domain, function.name)
    ]
    assert len(calls) == 1
    call = calls[0]
    assert len(call.input) == len(function.input) == 1
    scope.same(call.input[0], operand)
    assert len(call.output) == len(function.output) == len(outputs)
    for actual, expected in zip(call.output, outputs, strict=True):
        scope.same(actual, expected)


def _serialized_conditional(model: Any) -> tuple[Any, Any, Any]:
    scope = _SerializedScope(model.graph)
    branches = [node for node in scope.nodes if node.op_type == "If"]
    assert len(branches) == 1
    branch = branches[0]
    assert not branch.domain and len(branch.input) == 1
    comparison = scope.op(branch.input[0], "Greater", 2)
    scope.same(comparison.input[0], model.graph.input[1].name)
    scope.constant(comparison.input[1], np.asarray(0, dtype=np.float32))
    scope.typed(comparison.output[0], onnx.TensorProto.BOOL, ())
    scope.typed(model.graph.input[1].name, onnx.TensorProto.FLOAT, ())
    assert len(branch.output) == len(model.graph.output)
    for actual, expected in zip(branch.output, model.graph.output, strict=True):
        scope.same(actual, expected.name)
    graphs = {a.name: a.g for a in branch.attribute}
    assert set(graphs) == {"then_branch", "else_branch"}
    assert not graphs["then_branch"].input and not graphs["else_branch"].input
    return (
        scope,
        _SerializedScope(graphs["then_branch"], scope),
        _SerializedScope(graphs["else_branch"], scope),
    )


def _serialized_witness_body(scope: Any, operand: str, outputs: list[str]) -> None:
    assert len(outputs) == 4
    centered, squares, returned_variance, normalized = outputs
    scope.typed(operand, onnx.TensorProto.FLOAT, (2, 16))
    for value in (centered, squares, normalized):
        scope.typed(value, onnx.TensorProto.FLOAT, (2, 16))
    scope.typed(returned_variance, onnx.TensorProto.FLOAT, (2, 1))
    center = scope.op(centered, "Sub", 2)
    scope.same(center.input[0], operand)
    mean = scope.op(_serialized_expand(scope, center.input[1], (2, 1)), "Div", 2)
    scope.constant(mean.input[1], np.asarray(16, dtype=np.float32))
    _serialized_reduction(scope, mean.input[0], operand)
    square = scope.op(squares, "Mul", 2)
    for value in square.input:
        scope.same(value, centered)
    rounded = _serialized_expand(scope, returned_variance, (2, 1))
    scope.typed(rounded, onnx.TensorProto.FLOAT, (2,))
    cast_back = scope.op(rounded, "Cast", 1)
    assert {
        a.name: onnx.helper.get_attribute_value(a) for a in cast_back.attribute
    } == {"to": onnx.TensorProto.FLOAT}
    mean64 = scope.op(cast_back.input[0], "Div", 2)
    scope.typed(mean64.output[0], onnx.TensorProto.DOUBLE, (2,))
    scope.constant(mean64.input[1], np.asarray(16, dtype=np.float64))
    reduction = scope.op(mean64.input[0], "ReduceSum", 2)
    widened = scope.op(reduction.input[0], "Cast", 1)
    scope.typed(widened.output[0], onnx.TensorProto.DOUBLE, (2, 16))
    assert {a.name: onnx.helper.get_attribute_value(a) for a in widened.attribute} == {
        "to": onnx.TensorProto.DOUBLE
    }
    scope.same(widened.input[0], squares)
    _serialized_reduction(scope, mean64.input[0], widened.output[0])
    consuming_division = scope.op(normalized, "Div", 2)
    scope.same(consuming_division.input[0], centered)
    sqrt = scope.op(consuming_division.input[1], "Sqrt", 1)
    add = scope.op(sqrt.input[0], "Add", 2)
    scope.same(_serialized_expand(scope, add.input[0], (2, 1)), rounded)
    scope.constant(add.input[1], np.asarray(1e-5, dtype=np.float32))
    for value in (add.input[0], add.output[0], sqrt.output[0]):
        scope.typed(value, onnx.TensorProto.FLOAT, (2, 1))


def _serialized_witness(model: Any, boundary: str, observations: list[Any]) -> None:
    if boundary == "cond":
        _, positive, negative = _serialized_conditional(model)
        operand = model.graph.input[0].name
        _serialized_witness_body(
            positive, operand, [v.name for v in positive.graph.output]
        )
        center = negative.op(negative.graph.output[0].name, "Sub", 2)
        negation = negative.op(center.input[0], "Neg", 1)
        negative.same(negation.input[0], operand)
        _serialized_witness_body(
            negative, negation.output[0], [v.name for v in negative.graph.output]
        )
        assert not model.functions
    elif boundary == "function":
        function = _serialized_definition(
            model, observations, "_square_precision_function"
        )
        _serialized_call(
            _SerializedScope(model.graph),
            function,
            model.graph.input[0].name,
            [v.name for v in model.graph.output],
        )
        _serialized_witness_body(
            _SerializedScope(function), function.input[0], list(function.output)
        )
    else:
        assert boundary == "inline" and not model.functions
        _serialized_witness_body(
            _SerializedScope(model.graph),
            model.graph.input[0].name,
            [v.name for v in model.graph.output],
        )


def _serialized_residual_body(
    scope: Any, operand: str, output: str, dtype: str
) -> None:
    dt = np.dtype(dtype)
    proto_dtype = onnx.helper.np_dtype_to_tensor_dtype(dt)
    scope.typed(operand, proto_dtype, (5, 8))
    scope.typed(output, proto_dtype, (5, 8))
    residual = scope.op(output, "Add", 2)
    scope.same(residual.input[0], operand)
    biased = scope.op(residual.input[1], "Add", 2)
    bias = _serialized_reshape(scope, biased.input[1], (1, 8))
    scope.constant(bias, np.linspace(-0.25, 0.75, 8).astype(dt))
    scaled = scope.op(biased.input[0], "Mul", 2)
    center = scope.op(scaled.input[0], "Sub", 2)
    scope.same(center.input[0], operand)
    mean_name = _serialized_reshape(scope, center.input[1], (5, 1))
    mean = scope.op(mean_name, "Div", 2)
    scope.constant(mean.input[1], np.asarray(8, dtype=dt))
    _serialized_reduction(scope, mean.input[0], operand)
    weight = scope.op(scaled.input[1], "Mul", 2)
    scale = _serialized_reshape(scope, weight.input[1], (1, 8))
    scope.constant(scale, np.linspace(0.5, 1.5, 8).astype(dt))
    reciprocal = scope.op(weight.input[0], "Div", 2)
    scope.constant(reciprocal.input[0], np.asarray(1, dtype=dt))
    sqrt = scope.op(reciprocal.input[1], "Sqrt", 1)
    add = scope.op(sqrt.input[0], "Add", 2)
    scope.constant(add.input[1], np.asarray(np.float32(0.0003), dtype=dt))
    variance = scope.op(_serialized_reshape(scope, add.input[0], (5, 1)), "Div", 2)
    scope.constant(variance.input[1], np.asarray(8, dtype=dt))
    reduction = scope.op(variance.input[0], "ReduceSum", 2)
    square = scope.op(reduction.input[0], "Mul", 2)
    square_center = scope.op(square.input[0], "Sub", 2)
    scope.same(square.input[1], square.input[0])
    scope.same(square_center.input[0], operand)
    scope.same(_serialized_expand(scope, square_center.input[1], (5, 1)), mean_name)
    _serialized_reduction(scope, variance.input[0], square.output[0])
    for value in (add.input[0], add.output[0], sqrt.output[0], reciprocal.output[0]):
        scope.typed(value, proto_dtype, (5, 1))


def _serialized_residual(
    model: Any, boundary: str, dtype: str, observations: list[Any]
) -> None:
    if boundary == "inline":
        assert not model.functions
        _serialized_residual_body(
            _SerializedScope(model.graph),
            model.graph.input[0].name,
            model.graph.output[0].name,
            dtype,
        )
        return
    function = _serialized_definition(model, observations, "_NamedResidual")
    _serialized_residual_body(
        _SerializedScope(function), function.input[0], function.output[0], dtype
    )
    if boundary == "function":
        _serialized_call(
            _SerializedScope(model.graph),
            function,
            model.graph.input[0].name,
            [model.graph.output[0].name],
        )
        return
    assert boundary == "cond"
    _, positive, negative = _serialized_conditional(model)
    operand = model.graph.input[0].name
    _serialized_call(positive, function, operand, [positive.graph.output[0].name])
    after = negative.op(negative.graph.output[0].name, "Neg", 1)
    call = negative.producers[negative.origin(after.input[0])]
    assert (call.domain, call.op_type) == (function.domain, function.name)
    assert len(call.input) == len(call.output) == 1
    before = negative.op(call.input[0], "Neg", 1)
    negative.same(before.input[0], operand)
    _serialized_call(negative, function, before.output[0], [after.input[0]])


def _nodes(model: Any) -> Iterator[Any]:
    def walk(nodes: Any) -> Iterator[Any]:
        for node in nodes:
            yield node
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    yield from walk(attribute.g.node)

    yield from walk(model.graph.node)
    for function in model.functions:
        yield from walk(function.node)


def _assert_result(actual: Any, expected: Any, criteria: Any) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    if criteria == "exact":
        dtype = {2: np.uint16, 4: np.uint32, 8: np.uint64}[expected.dtype.itemsize]
        np.testing.assert_array_equal(
            np.ascontiguousarray(actual[finite]).view(dtype),
            np.ascontiguousarray(expected[finite]).view(dtype),
        )
    else:
        np.testing.assert_allclose(
            actual[finite], expected[finite], rtol=criteria[0], atol=criteria[1]
        )


def _exercise(
    fn: Any,
    feeds: list[tuple[Any, ...]],
    tmp_path: Path,
    criteria: Any,
    record_property: Any,
    *,
    function_body: Any = None,
    function_body_specs: Any = None,
    function_body_feeds: Any = None,
    boundary: str = "inline",
    exact_zero_output: int | None = None,
    serialized_check: Any = None,
    declared_specs: Any = None,
    opset: int = 23,
) -> Any:
    """Capture native first, compare the full active world, export with default audit."""
    attempts = []

    def denied(*args: Any, **kwargs: Any) -> None:
        attempts.append(repr((args, kwargs)))
        raise AssertionError("native reference collection denies every ORT constructor")

    references = []
    body_reference = None
    body_export_reference = None
    body_observations: list[Any] = []
    exported_bodies: list[Any] = []
    native_body_payloads: list[Any] = []
    native_symbolic_body_payloads: list[Any] = []
    carrier_execution: list[Any] = []
    body_context = None
    body_identity = None
    body_state = None
    with (
        patch.object(ort, "InferenceSession", denied),
        patch.object(_pybind_state, "InferenceSession", denied),
    ):

        def exact_zero(outputs: Any, expected: Any) -> None:
            if exact_zero_output is not None:
                actual_row = np.asarray(outputs[exact_zero_output])[1]
                native_row = np.asarray(expected[exact_zero_output])[1]
                assert np.all(native_row == 0)
                _assert_result(actual_row, native_row, "exact")

        native = jax.make_jaxpr(_fresh(fn))(*feeds[0])
        manifest = typed_trace_manifest(native)
        _json(tmp_path / "native-trace.json", manifest)
        np.savez(
            tmp_path / "native-constants.npz",
            **{f"const{i}": np.asarray(v) for i, v in enumerate(native.consts)},
        )
        symbolic_native = (
            jax.make_jaxpr(_fresh(fn))(*declared_specs)
            if declared_specs is not None
            else None
        )
        for index, values in enumerate(feeds):
            eager = tuple(np.asarray(v) for v in jax.tree.leaves(fn(*values)))
            compiled = tuple(
                np.asarray(v) for v in jax.tree.leaves(jax.jit(_fresh(fn))(*values))
            )
            exact_zero(eager, eager)
            exact_zero(compiled, eager)
            references.append((eager, compiled))
            np.savez(
                tmp_path / f"native-{index}.npz",
                **{
                    f"{kind}-{i}": v
                    for kind, outputs in (("eager", eager), ("jit", compiled))
                    for i, v in enumerate(outputs)
                },
            )
        if function_body is not None:
            body_fn, body_values, state = function_body
            body_context = _context()
            body_identity = _callee_identity(body_fn)
            body_state = state()
            body_reference = jax.make_jaxpr(_fresh(body_fn))(*body_values)
            body_export_reference = (
                body_reference
                if function_body_specs is None
                else jax.make_jaxpr(_fresh(body_fn))(*function_body_specs)
            )
            body_feeds = (
                [body_values]
                if function_body_feeds is None
                else list(function_body_feeds)
            )
            if boundary == "cond" and function_body_feeds is None:
                assert len(body_values) == 1
                body_feeds.append((-body_values[0],))
            for index, values in enumerate(body_feeds):
                eager_body = body_fn(*values)
                compiled_body = jax.jit(_fresh(body_fn))(*values)
                assert jax.tree.structure(eager_body) == jax.tree.structure(
                    compiled_body
                )
                native_body_payloads.append(
                    (
                        jax.tree.structure(eager_body),
                        tuple(jax.tree.leaves(eager_body)),
                        tuple(jax.tree.leaves(compiled_body)),
                    )
                )
                exact_zero(jax.tree.leaves(eager_body), jax.tree.leaves(eager_body))
                exact_zero(jax.tree.leaves(compiled_body), jax.tree.leaves(eager_body))
                np.savez(
                    tmp_path / f"native-body-payload-{index}.npz",
                    **{
                        f"{kind}-{i}": np.asarray(v)
                        for kind, outputs in (
                            ("eager", jax.tree.leaves(eager_body)),
                            ("jit", jax.tree.leaves(compiled_body)),
                        )
                        for i, v in enumerate(outputs)
                    },
                    **{f"feed-{i}": np.asarray(v) for i, v in enumerate(values)},
                )
            _json(
                tmp_path / "native-body-reference-context.json",
                {
                    "callee_callable_id": id(body_identity[0]),
                    "callee_self_id": (
                        None if body_identity[1] is None else id(body_identity[1])
                    ),
                    "state_sha256": hashlib.sha256(
                        repr(body_state).encode()
                    ).hexdigest(),
                    "context": body_context,
                    "ordered_formals": [
                        _aval(v) for v in body_export_reference.jaxpr.invars
                    ],
                    "ordered_results": [
                        _aval(v) for v in body_export_reference.jaxpr.outvars
                    ],
                    "dynamic_capture_order": [],
                    "result_trees": [str(v[0]) for v in native_body_payloads],
                    "feed_order": (
                        (["x", "Neg(x)"] if boundary == "cond" else ["x"])
                        if function_body_feeds is None
                        else [
                            {"index": i, "shapes": [list(v.shape) for v in values]}
                            for i, values in enumerate(body_feeds)
                        ]
                    ),
                },
            )
            _json(
                tmp_path / "native-function-body.json",
                typed_trace_manifest(body_export_reference),
            )
            np.savez(
                tmp_path / "native-function-constants.npz",
                **{
                    f"const{i}": np.asarray(v)
                    for i, v in enumerate(body_export_reference.consts)
                },
            )
            if function_body_specs is not None:
                original_manifest = typed_trace_manifest(body_export_reference)
                original_constants = tuple(
                    _payload(v) for v in body_export_reference.consts
                )
                original_formals = tuple(
                    jax.ShapeDtypeStruct(
                        var.aval.shape, var.aval.dtype, weak_type=var.aval.weak_type
                    )
                    for var in body_export_reference.jaxpr.invars
                )
                original_compiled_body = export.export(
                    jax.jit(jaxpr_as_fun(body_export_reference))
                )(*original_formals)
                for index, values in enumerate(body_feeds):
                    outputs = tuple(original_compiled_body.call(*values))
                    native_symbolic_body_payloads.append(outputs)
                    np.savez(
                        tmp_path / f"native-symbolic-body-compiled-{index}.npz",
                        **{f"output{i}": np.asarray(v) for i, v in enumerate(outputs)},
                    )
                assert typed_trace_manifest(body_export_reference) == original_manifest
                assert (
                    tuple(_payload(v) for v in body_export_reference.consts)
                    == original_constants
                )
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            activated = jax.make_jaxpr(_fresh(fn))(*feeds[0])
            _json(tmp_path / "SOURCE-trace.json", typed_trace_manifest(activated))
            if symbolic_native is not None:
                symbolic_activated = jax.make_jaxpr(_fresh(fn))(*declared_specs)
                if body_reference is None:
                    assert typed_trace_manifest(
                        symbolic_activated
                    ) == typed_trace_manifest(symbolic_native)
                    assert [_payload(v) for v in symbolic_activated.consts] == [
                        _payload(v) for v in symbolic_native.consts
                    ]
                else:
                    _function_outer(
                        symbolic_native,
                        symbolic_activated,
                        body_export_reference,
                        boundary,
                    )
                _json(
                    tmp_path / "symbolic-native-trace.json",
                    typed_trace_manifest(symbolic_native),
                )
            if body_reference is None:
                assert typed_trace_manifest(activated) == manifest
                assert [_payload(v) for v in activated.consts] == [
                    _payload(v) for v in native.consts
                ]
            else:
                _function_outer(native, activated, body_reference, boundary)
            for index, (values, expected_pair) in enumerate(
                zip(feeds, references, strict=True)
            ):
                for mode, runner, expected in zip(
                    ("eager", "jit"),
                    (fn, jax.jit(_fresh(fn))),
                    expected_pair,
                    strict=True,
                ):
                    try:
                        result = runner(*values)
                        outputs = jax.tree.leaves(result)
                        np.savez(
                            tmp_path / f"SOURCE-{mode}-{index}.npz",
                            **{
                                f"output{i}": np.asarray(v)
                                for i, v in enumerate(outputs)
                            },
                        )
                        assert tuple(_payload(v) for v in outputs) == tuple(
                            _payload(v) for v in expected
                        )
                        exact_zero(outputs, expected)
                        carrier_execution.append(
                            {"feed": index, "mode": mode, "status": "EXACT"}
                        )
                    except NotImplementedError as exc:
                        # Explicit finite evidence amendment: keep the actual carrier failure;
                        # only its body payload is tested below after ordinary export/restoration.
                        if (
                            body_reference is None
                            or "MLIR translation rule for primitive 'onnx_fn::"
                            not in str(exc)
                        ):
                            raise
                        carrier_execution.append(
                            {
                                "feed": index,
                                "mode": mode,
                                "status": "NOT_ESTABLISHED",
                                "error": str(exc),
                            }
                        )
                    finally:
                        _json(tmp_path / "activated-execution.json", carrier_execution)
    assert attempts == []
    specs = (
        declared_specs
        if declared_specs is not None
        else [
            jax.ShapeDtypeStruct(
                v.shape, v.dtype, weak_type=bool(getattr(v, "weak_type", False))
            )
            for v in feeds[0]
        ]
    )
    original_method = FunctionPlugin._trace_function_body

    def observe_body(
        plugin: Any,
        callee: Any,
        equation: Any,
        captures: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        assert body_reference is not None and body_identity is not None
        identity = _callee_identity(callee)
        assert identity[0] is body_identity[0] and identity[1] is body_identity[1]
        assert _context() == body_context and state() == body_state
        assert tuple(captures.dynamic) == () and dict(captures.static_kwargs) == {}
        assert kwargs["precision_policy"] is PrecisionPolicy.SOURCE
        source_trace = kwargs["source_trace"]
        assert source_trace is not None
        assert (
            source_trace.x64_enabled,
            source_trace.default_matmul_precision,
            source_trace.numpy_dtype_promotion,
            source_trace.numpy_rank_promotion,
        ) == tuple(
            body_context[name]
            for name in (
                "jax_enable_x64",
                "jax_default_matmul_precision",
                "jax_numpy_dtype_promotion",
                "jax_numpy_rank_promotion",
            )
        )
        invocation = {
            "plugin_id": id(plugin),
            "function_name": plugin._friendly_name_base(),
            "namespace": plugin.namespace,
            "qualified_target": plugin._qualified_target,
            "callee_callable_id": id(identity[0]),
            "callee_self_id": None if identity[1] is None else id(identity[1]),
            "equation_id": id(equation),
            "captures_id": id(captures),
            "dynamic_capture_order": [],
            "static_capture_names": [],
            "native_context": body_context,
            "source_context": _context(),
            "policy": kwargs["precision_policy"].value,
            "state_sha256": hashlib.sha256(repr(body_state).encode()).hexdigest(),
        }
        result = original_method(plugin, callee, equation, captures, *args, **kwargs)
        actual = typed_trace_manifest(result)
        invocation.update(
            returned_closed_jaxpr_id=id(result),
            original_method_calls=1,
            identical_return_object=True,
            actual_complete_body=actual,
            equation_primitive=equation.primitive.name,
            input_ports=[_aval(v) for v in equation.invars],
            result_ports=[_aval(v) for v in equation.outvars],
        )
        exported_bodies.append(result)
        body_observations.append(invocation)
        _json(tmp_path / "actual-exported-function-bodies.json", body_observations)
        np.savez(
            tmp_path / f"actual-function-constants-{len(body_observations)}.npz",
            **{f"const{i}": np.asarray(v) for i, v in enumerate(result.consts)},
        )
        assert actual == typed_trace_manifest(body_export_reference)
        assert [_payload(v) for v in result.consts] == [
            _payload(v) for v in body_export_reference.consts
        ]
        assert _context() == body_context and state() == body_state
        return result

    try:
        with (
            patch.object(FunctionPlugin, "_trace_function_body", observe_body)
            if body_reference is not None
            else nullcontext()
        ):
            model = ui._to_onnx(
                fn,
                specs,
                precision_policy=PrecisionPolicy.SOURCE,
                return_mode="proto",
                opset=opset,
            )
    finally:
        assert FunctionPlugin._trace_function_body is original_method
    if body_reference is not None:
        # jaxpr_as_fun evaluates the identical returned ClosedJaxpr and constants.
        # Its fresh JIT stages that evaluator, never the original Python body.
        assert _context() == body_context and state() == body_state
        body_evidence = []
        for body_index, closed in enumerate(exported_bodies):
            trace_before = typed_trace_manifest(closed)
            constants_before = tuple(_payload(v) for v in closed.consts)
            digest_before = hashlib.sha256(
                repr((trace_before, constants_before)).encode()
            ).hexdigest()
            evaluator = jaxpr_as_fun(closed)
            if function_body_specs is not None:
                # Compile each retained body in its own native symbolic scope.
                # This is a compiled symbolic-body comparison, not a claim that
                # raw eval_jaxpr or the outer onnx_fn carrier has an MLIR rule.
                formals = tuple(
                    jax.ShapeDtypeStruct(
                        var.aval.shape, var.aval.dtype, weak_type=var.aval.weak_type
                    )
                    for var in closed.jaxpr.invars
                )
                actual_compiled_body = export.export(jax.jit(evaluator))(*formals)
            for feed_index, (values, (tree, eager, compiled)) in enumerate(
                zip(body_feeds, native_body_payloads, strict=True)
            ):
                comparisons = (
                    (
                        (
                            "compiled-symbolic-body",
                            actual_compiled_body.call,
                            native_symbolic_body_payloads[feed_index],
                        ),
                    )
                    if function_body_specs is not None
                    else (
                        ("eager", evaluator, eager),
                        ("jit", jax.jit(_fresh(evaluator)), compiled),
                    )
                )
                for mode, runner, expected in comparisons:
                    outputs = runner(*values)
                    assert len(outputs) == tree.num_leaves
                    assert jax.tree.structure(jax.tree.unflatten(tree, outputs)) == tree
                    np.savez(
                        tmp_path
                        / f"exported-body-{body_index}-{feed_index}-{mode}.npz",
                        **{f"output{i}": np.asarray(v) for i, v in enumerate(outputs)},
                    )
                    if function_body_specs is None:
                        assert tuple(_payload(v) for v in outputs) == tuple(
                            _payload(v) for v in expected
                        )
                    else:
                        for output_index, (actual, original) in enumerate(
                            zip(outputs, expected, strict=True)
                        ):
                            selected = (
                                criteria
                                if criteria == "exact"
                                or isinstance(criteria[0], (float, int))
                                else criteria[output_index]
                            )
                            _assert_result(actual, original, selected)
                    exact_zero(outputs, expected)
                    body_evidence.append(
                        {
                            "body": body_index,
                            "feed": feed_index,
                            "mode": mode,
                            "returned_closed_jaxpr_id": id(closed),
                            "status": (
                                "EXACT"
                                if function_body_specs is None
                                else "COMPILED_SYMBOLIC_BODY_CRITERIA_PASSED"
                            ),
                            "native_result_tree": str(tree),
                            "ordered_formals": [_aval(v) for v in closed.jaxpr.invars],
                            "ordered_results": [_aval(v) for v in closed.jaxpr.outvars],
                            "trace_constants_sha256": digest_before,
                        }
                    )
            assert typed_trace_manifest(closed) == trace_before
            assert tuple(_payload(v) for v in closed.consts) == constants_before
            digest_after = hashlib.sha256(
                repr(
                    (
                        typed_trace_manifest(closed),
                        tuple(_payload(v) for v in closed.consts),
                    )
                ).encode()
            ).hexdigest()
            assert digest_before == digest_after
            body_evidence[-1]["trace_constants_after_sha256"] = digest_after
        assert body_evidence and _context() == body_context and state() == body_state
        _json(
            tmp_path / "exported-body-eager-jit-equality.json",
            {
                "boundary": (
                    "actual exported-body eager/JIT equality; whole-carrier JIT equality NOT_ESTABLISHED"
                    if function_body_specs is None
                    else "retained original/actual exported symbolic-body compiled-call comparison; raw symbolic eager and whole-carrier JIT NOT_ESTABLISHED"
                ),
                "amendment_sha256": "aa64a4bbe0372cb5ff0aa435f927d1f76ddc08e0566a18fe53419d0866655acf",
                "observations": body_evidence,
            },
        )
    path = tmp_path / "source.onnx"
    onnx.save_model(model, path)
    model = onnx.load_model(path)
    onnx.checker.check_model(model, full_check=True)
    if body_reference is not None:
        assert body_observations
        _serialized_functions(model, body_observations, body_reference, boundary)
    if serialized_check is not None:
        serialized_check(model, body_observations)
    report = get_precision_report(model)
    assert report is not None and report.profile == "source-static-v1"
    assert report.opset == opset
    assert (
        next(item.version for item in model.opset_import if item.domain == "") == opset
    )
    checks = {c.category: c.status for c in report.checks}
    assert (
        checks["runtime-execution"]
        is checks["numerical-acceptance"]
        is CheckStatus.UNVERIFIED
    )
    for value, expected, declared in zip(
        model.graph.output, references[0][0], report.exported_outputs, strict=True
    ):
        assert value.type.tensor_type.elem_type == onnx.helper.np_dtype_to_tensor_dtype(
            expected.dtype
        )
        dimensions = tuple(
            d.dim_param if d.HasField("dim_param") else d.dim_value
            for d in value.type.tensor_type.shape.dim
        )
        assert dimensions == declared.shape
        if declared_specs is None:
            assert dimensions == expected.shape
    _json(tmp_path / "precision-report.json", asdict(report))
    observations, failures = [], []
    for optimize in (False, True):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if optimize
            else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        assert session.get_providers() == ["CPUExecutionProvider"]
        for index, (values, expected_pair) in enumerate(
            zip(feeds, references, strict=True)
        ):
            feed = {
                item.name: np.asarray(value)
                for item, value in zip(session.get_inputs(), values, strict=True)
            }
            outputs = session.run(None, feed)
            np.savez(
                tmp_path / f"target-{optimize}-{index}.npz",
                **{f"output{i}": value for i, value in enumerate(outputs)},
                **{f"feed-{key}": value for key, value in feed.items()},
            )
            for role, expected in zip(
                ("native-eager", "native-jit"), expected_pair, strict=True
            ):
                row = {
                    "optimization": "all" if optimize else "disabled",
                    "feed": index,
                    "reference": role,
                    "status": "PASSED",
                    "criteria": criteria,
                }
                try:
                    for output_index, (output, reference) in enumerate(
                        zip(outputs, expected, strict=True)
                    ):
                        selected = (
                            criteria
                            if criteria == "exact"
                            or isinstance(criteria[0], (float, int))
                            else criteria[output_index]
                        )
                        _assert_result(output, reference, selected)
                    exact_zero(outputs, expected)
                except AssertionError as exc:
                    row.update(status="FAILED", error=str(exc))
                    failures.append(row)
                observations.append(row)
    receipt = {
        "jax": jax.__version__,
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "ort": ort.__version__,
        "providers": ["CPUExecutionProvider"],
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "native_ORT_attempts": attempts,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "observations": observations,
        "target_failures": len(failures),
        "activated_execution": carrier_execution,
        "named_payload_boundary": (
            (
                "actual exported-body eager/JIT equality; whole-carrier JIT equality NOT_ESTABLISHED"
                if function_body_specs is None
                else "retained original/actual exported symbolic-body compiled-call comparison; raw symbolic eager and whole-carrier JIT NOT_ESTABLISHED"
            )
            if body_reference is not None
            else "complete activated eager/JIT equality"
        ),
    }
    _json(tmp_path / "runtime-observations.json", receipt)
    record_property("normalization-evidence", json.dumps(receipt, sort_keys=True))
    assert not failures, json.dumps(failures, indent=2)
    return model


@pytest.mark.parametrize(
    "dtype,weak",
    [
        ("float16", False),
        ("float32", False),
        ("float64", False),
        ("float32", True),
        ("float64", True),
    ],
)
def test_native_square_serialized_consumer(
    dtype: str, weak: bool, tmp_path: Path, record_property: Any
) -> None:
    with jax.enable_x64(dtype == "float64"):
        if weak:
            value = jnp.asarray(1.0009765625)
        else:
            dt = np.dtype(dtype)
            value = jnp.asarray(
                [
                    -0.0,
                    0.0,
                    1.0,
                    -1.0,
                    1.0009765625,
                    np.nextafter(dt.type(1), dt.type(2)),
                    -np.finfo(dt).tiny,
                    np.finfo(dt).tiny,
                    np.inf,
                    -np.inf,
                    np.nan,
                ],
                dtype=dt,
            )
        model = _exercise(
            jax.lax.square, [(value,)], tmp_path, "exact", record_property
        )
        mul = next(n for n in model.graph.node if n.op_type == "Mul")
        assert mul.input[0] == mul.input[1]


@pytest.mark.parametrize(
    "operation,dtype,axes,keepdims,correction,compute,family", _VARIANCE_CASES
)
def test_original_native_variance_and_std(
    operation: str,
    dtype: str,
    axes: Any,
    keepdims: bool,
    correction: int,
    compute: str | None,
    family: str,
    tmp_path: Path,
    record_property: Any,
) -> None:
    from tests.extra_tests.converter.test_source_layer_norm_forwarding import _rows

    value = (
        jnp.asarray(np.linspace(-2, 3, 48).reshape(2, 3, 8), dtype=dtype)
        if family == "multi"
        else _rows(jnp.dtype(dtype))
    )
    if family == "rows-div16":
        value = value / jnp.asarray(16, dtype=value.dtype)

    def fn(x: Any) -> Any:
        return getattr(jnp, operation)(
            x,
            axis=axes,
            keepdims=keepdims,
            correction=correction,
            dtype=None if compute is None else jnp.dtype(compute),
        )

    expected_dtype = np.asarray(fn(value)).dtype.name
    model = _exercise(
        fn,
        [(value,)],
        tmp_path,
        _BOUNDS["variance" if operation == "var" else "std"][expected_dtype],
        record_property,
    )
    assert any(n.op_type == "ReduceSum" for n in _nodes(model))
    assert any(n.op_type == "Mul" for n in _nodes(model))


def _square_precision_witness(x: Any) -> Any:
    mean = jax.lax.reduce_sum(x, axes=(1,)) / np.float32(16)
    centered = x - mean[:, None]
    squares = jax.lax.square(centered)
    variance64 = jax.lax.reduce_sum(squares.astype(jnp.float64), axes=(1,))
    variance = (variance64 / np.float64(16)).astype(jnp.float32)
    normalized = centered / jnp.sqrt(variance[:, None] + np.float32(1e-5))
    return centered, squares, variance[:, None], normalized


@onnx_function
def _square_precision_function(x: Any) -> Any:
    return _square_precision_witness(x)


@pytest.mark.parametrize("boundary", ["inline", "function", "cond"])
def test_native_square_keeps_exact_double_sum_and_consuming_normalization(
    boundary: str, tmp_path: Path, record_property: Any
) -> None:
    value = jnp.asarray(normalization_input())
    fn: Any = {
        "inline": _square_precision_witness,
        "function": lambda x: _square_precision_function(x),
        "cond": lambda x, selector: jax.lax.cond(
            selector > np.float32(0),
            _square_precision_witness,
            lambda y: _square_precision_witness(-y),
            x,
        ),
    }[boundary]
    feeds = (
        [(value,)]
        if boundary != "cond"
        else [(value, jnp.asarray(v, dtype=jnp.float32)) for v in (1, -1)]
    )
    for values in feeds:
        expected = normalization_reference(
            np.asarray(value if len(values) == 1 or values[1] > 0 else -value)
        )
        for result in (fn(*values), jax.jit(_fresh(fn))(*values)):
            for actual, reference in zip(result[:3], expected[:3], strict=True):
                _assert_result(actual, reference, "exact")
            np.testing.assert_array_equal(result[2], [[np.float32(8388609)], [0]])
            _assert_result(result[3], expected[3], (3e-7, 1e-7))
            assert np.all(expected[3][1] == 0)
            _assert_result(result[3][1], expected[3][1], "exact")
    model = _exercise(
        fn,
        feeds,
        tmp_path,
        ["exact", "exact", "exact", (3e-7, 1e-7)],
        record_property,
        function_body=(
            (_square_precision_function, (value,), lambda: ())
            if boundary == "function"
            else None
        ),
        boundary=boundary,
        exact_zero_output=3,
        serialized_check=lambda model, observations: _serialized_witness(
            model, boundary, observations
        ),
    )
    assert not any(n.op_type == "LayerNormalization" for n in _nodes(model))
    assert boundary != "function" or model.functions
    assert boundary != "cond" or any(n.op_type == "If" for n in model.graph.node)


def _graph(name: str, source: Any, target: Any, params: Any = None) -> Any:
    inputs = [SimpleNamespace(aval=source)]
    outputs = [SimpleNamespace(aval=target)]
    equation = SimpleNamespace(
        primitive=SimpleNamespace(name=name),
        invars=list(inputs),
        outvars=outputs,
        params=dict(params or {}),
        effects=(),
    )
    return SimpleNamespace(
        invars=inputs, constvars=(), outvars=outputs, eqns=[equation], effects=()
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "integer",
        "complex",
        "dtype",
        "shape",
        "weakness",
        "parameter",
        "placement",
        "arity",
    ],
)
def test_square_rejects_unaudited_or_corrupt_native_metadata(corruption: str) -> None:
    source = jax.core.ShapedArray((2, 3), np.float32)
    target = source
    params: dict[str, Any] = {}
    if corruption in {"integer", "complex"}:
        source = target = source.update(
            dtype=np.dtype(np.int32 if corruption == "integer" else np.complex64)
        )
    elif corruption == "dtype":
        target = source.update(dtype=np.dtype(np.float64))
    elif corruption == "shape":
        target = source.update(shape=(3, 2))
    elif corruption == "weakness":
        target = source.update(weak_type=True)
    elif corruption == "parameter":
        params["invented"] = 1
    elif corruption == "placement":
        params["out_sharding"] = "not unspecified"
    graph = _graph("square", source, target, params)
    if corruption == "arity":
        graph.eqns[0].invars.append(graph.invars[0])
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(graph, 23)


@pytest.mark.parametrize(
    "source,target",
    [
        ((3,), (3, 1)),
        ((2, 3), (2, 1, 3)),
        ((2, 1, 3), (2, 3)),
        ((), (1, 1)),
        ((1, 1), ()),
        ((3,), (1, 3)),
    ],
)
def test_static_singleton_reshape_keeps_original_order(
    source: Any, target: Any, tmp_path: Path, record_property: Any
) -> None:
    value = jnp.arange(np.prod(source, dtype=int), dtype=jnp.float32).reshape(source)
    model = _exercise(
        lambda x: jax.lax.reshape(x, target),
        [(value,)],
        tmp_path,
        "exact",
        record_property,
    )
    assert any(
        node.op_type in {"Reshape", "Identity", "Squeeze", "Unsqueeze"}
        for node in _nodes(model)
    )


@pytest.mark.parametrize(
    "corruption",
    ["regroup", "permute", "zero", "placement", "dtype", "weakness", "sizes"],
)
def test_singleton_reshape_rejects_other_transforms(corruption: str) -> None:
    source = jax.core.ShapedArray((2, 3), np.float32)
    target = source.update(shape=(2, 1, 3))
    params: dict[str, Any] = {"new_sizes": (2, 1, 3), "dimensions": None}
    if corruption == "regroup":
        params["new_sizes"] = (6, 1)
        target = target.update(shape=(6, 1))
        # Preserve the original parameter ID as its now-audited static positive.
        for opset in (16, 23):
            validate_source_jaxpr(_graph("reshape", source, target, params), opset)
        return
    elif corruption == "permute":
        params["dimensions"] = (1, 0)
    elif corruption == "zero":
        source = source.update(shape=(0, 3))
        target = target.update(shape=(0, 1, 3))
        params["new_sizes"] = target.shape
    elif corruption == "placement":
        params["sharding"] = "placed"
    elif corruption == "dtype":
        target = target.update(dtype=np.dtype(np.float64))
    elif corruption == "weakness":
        target = target.update(weak_type=True)
    else:
        params["new_sizes"] = (2, 3, 1)
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(_graph("reshape", source, target, params), 23)
    assert caught.value.primitive == "reshape"
    if corruption in {"dtype", "weakness", "sizes"}:
        assert caught.value.category == "lowering-defect"


@pytest.mark.parametrize(
    "source,target",
    [((2, 3, 4), (4, 6)), ((24,), (2, 3, 4)), ((2, 6), (3, 4))],
    ids=["flatten-tail", "split-vector", "regroup-coordinates"],
)
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("opset", [16, 23])
def test_static_reshape_preserves_arbitrary_floating_payload_bits(
    source: tuple[int, ...],
    target: tuple[int, ...],
    dtype: str,
    opset: int,
    tmp_path: Path,
    record_property: Any,
) -> None:
    dt = np.dtype(dtype)
    unsigned, payload = {
        "float16": (np.uint16, (0x8000, 0, 0x7C00, 0xFC00, 0x7E01, 0xFE02)),
        "float32": (
            np.uint32,
            (0x80000000, 0, 0x7F800000, 0xFF800000, 0x7FC00001, 0xFFC00002),
        ),
        "float64": (
            np.uint64,
            (
                0x8000000000000000,
                0,
                0x7FF0000000000000,
                0xFFF0000000000000,
                0x7FF8000000000001,
                0xFFF8000000000002,
            ),
        ),
    }[dtype]
    values = np.arange(np.prod(source), dtype=np.int32).astype(dt) / dt.type(
        8
    ) - dt.type(1)
    values.view(unsigned)[: len(payload)] = np.asarray(payload, dtype=unsigned)
    arrays = (values.reshape(source), values[::-1].copy().reshape(source))

    def reshape(x: Any) -> Any:
        return jax.lax.reshape(x, target)

    model = _exercise(
        reshape,
        [(jnp.asarray(value),) for value in arrays],
        tmp_path,
        "exact",
        record_property,
        opset=opset,
    )
    assert any(node.op_type == "Reshape" for node in _nodes(model))
    report = get_precision_report(model)
    assert report.inputs[0].traced.shape == source
    assert report.exported_outputs[0].shape == target
    for index, value in enumerate(arrays):
        expected = np.reshape(value, target)
        with np.load(tmp_path / f"native-{index}.npz") as native:
            for role in ("eager", "jit"):
                actual = native[f"{role}-0"]
                assert actual.dtype == dt and actual.shape == target
                assert actual.tobytes(order="C") == expected.tobytes(order="C")
        for role in ("eager", "jit"):
            with np.load(tmp_path / f"SOURCE-{role}-{index}.npz") as activated:
                actual = activated["output0"]
                assert actual.dtype == dt and actual.shape == target
                assert actual.tobytes(order="C") == expected.tobytes(order="C")
        for optimized in (False, True):
            with np.load(tmp_path / f"target-{optimized}-{index}.npz") as observed:
                actual = observed["output0"]
                assert actual.dtype == dt and actual.shape == target
                assert actual.tobytes(order="C") == expected.tobytes(order="C")
    evidence = {
        "comparison": "all C-order payload bytes including NaN payload/sign, infinities and signed zeros",
        "dtype": dtype,
        "source_shape": source,
        "target_shape": target,
        "opset": opset,
        "feeds": 2,
        "optimization_modes": ["disabled", "all"],
    }
    _json(tmp_path / "reshape-bit-preservation.json", evidence)
    record_property("reshape-bit-preservation", json.dumps(evidence, sort_keys=True))


@pytest.mark.parametrize("opset", [16, 23])
@pytest.mark.parametrize(
    "corruption",
    ["element-count", "overflow-count", "result-sizes", "dtype", "weakness"],
)
def test_static_reshape_rejects_inconsistent_native_metadata(
    corruption: str,
    opset: int,
) -> None:
    source = jax.core.ShapedArray((2, 3), np.float32)
    target = source.update(shape=(3, 2))
    params: dict[str, Any] = {"new_sizes": target.shape, "dimensions": None}
    if corruption == "element-count":
        target = target.update(shape=(7,))
        params["new_sizes"] = target.shape
    elif corruption == "overflow-count":
        # Both products wrap to zero in uint64; Python integer counts differ.
        source = source.update(shape=(2**32, 2**32))
        target = target.update(shape=(2**32, 2**33))
        params["new_sizes"] = target.shape
    elif corruption == "result-sizes":
        params["new_sizes"] = (6,)
    elif corruption == "dtype":
        target = target.update(dtype=np.dtype(np.float64))
    else:
        target = target.update(weak_type=True)
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(_graph("reshape", source, target, params), opset)
    assert caught.value.primitive == "reshape"
    assert caught.value.category == "lowering-defect"


@pytest.mark.parametrize(
    "dtype,opset", [("bfloat16", 23), ("float32", 26), ("int32", 23), ("complex64", 23)]
)
def test_max_never_falls_through_to_unaudited_legacy_route(
    dtype: str, opset: int
) -> None:
    value = jnp.ones((2, 3), dtype=dtype)
    if dtype == "complex64":
        source = jax.core.ShapedArray((2, 3), np.complex64)
        graph = _graph("max", source, source)
        graph.eqns[0].invars.append(graph.invars[0])
    else:
        graph = jax.make_jaxpr(lambda x: jax.lax.max(x, x))(value)
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(graph, opset)


@pytest.mark.parametrize(
    "corruption", ["dtype", "shape", "weakness", "parameter", "placement"]
)
def test_max_rejects_corrupt_native_metadata(corruption: str) -> None:
    source = jax.core.ShapedArray((2, 3), np.float32)
    target = source
    params: dict[str, Any] = {}
    if corruption == "dtype":
        target = target.update(dtype=np.dtype(np.float64))
    elif corruption == "shape":
        target = target.update(shape=(3, 2))
    elif corruption == "weakness":
        target = target.update(weak_type=True)
    elif corruption == "parameter":
        params["invented"] = 1
    else:
        params["out_sharding"] = "placed"
    graph = _graph("max", source, target, params)
    graph.eqns[0].invars.append(graph.invars[0])
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(graph, 23)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_sqrt_forwards_original_trace_context_errors_and_restores(dtype: str) -> None:
    original = jnp.sqrt
    value = jnp.asarray([0, 1, 2, 4], dtype=dtype)
    with jax.numpy_dtype_promotion("strict"), jax.numpy_rank_promotion("raise"):
        before = _context()
        native = typed_trace_manifest(jax.make_jaxpr(_fresh(original))(value))
        eager, compiled = original(value), jax.jit(_fresh(original))(value)
        with pytest.raises(TypeError) as error:
            original("invalid")
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            assert (
                typed_trace_manifest(jax.make_jaxpr(lambda x: jnp.sqrt(x))(value))
                == native
            )
            for result, reference in (
                (jnp.sqrt(value), eager),
                (jax.jit(lambda x: jnp.sqrt(x))(value), compiled),
            ):
                assert _payload(result) == _payload(reference)
            with pytest.raises(type(error.value)) as forwarded:
                jnp.sqrt("invalid")
            assert str(forwarded.value) == str(error.value)
        assert jnp.sqrt is original and _context() == before


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("operation", ["square", "max", "reduce_sum", "reshape"])
@pytest.mark.parametrize("opset", [16, 23])
def test_symbolic_normalization_primitives_keep_one_artifact_and_original_rounding(
    rank: int, operation: str, opset: int, tmp_path: Path, record_property: Any
) -> None:
    tail = {1: (), 2: (3,), 3: (1, 3), 4: (1, 3, 1)}[rank]
    feeds = []
    for batch in (2, 5):
        shape = (batch, *tail)
        values = (np.arange(np.prod(shape)).reshape(shape) % 9 - 4).astype(
            np.float32
        ) / 4
        if operation in {"square", "max"}:
            values = np.resize(
                np.asarray([-0.0, 0.0, -2, 4, np.nan, np.inf, -np.inf], np.float32),
                shape,
            )
        feeds.append((jnp.asarray(values),))
    if operation == "square":
        fn = jax.lax.square
    elif operation == "max":

        def fn(x: Any) -> Any:
            return jax.lax.max(x, -x)

    elif operation == "reduce_sum":

        def fn(x: Any) -> Any:
            return jax.lax.reduce_sum(x, axes=tuple(range(1, rank)))

    else:

        def fn(x: Any) -> Any:
            shape = (
                (x.shape[0], 1)
                if rank == 1
                else ((x.shape[0], 1, 3) if rank == 2 else (x.shape[0], 3))
            )
            return jax.lax.reshape(x, shape)

    (batch,) = export.symbolic_shape("B")
    specs = [jax.ShapeDtypeStruct((batch, *tail), np.float32)]
    model = _exercise(
        fn,
        feeds,
        tmp_path,
        "exact",
        record_property,
        declared_specs=specs,
        opset=opset,
    )
    report = get_precision_report(model)
    assert report is not None and report.inputs[0].traced.shape == ("B", *tail)
    assert report.exported_outputs[0].shape[0] == "B"
    operations = [node.op_type for node in _nodes(model)]
    if operation == "square":
        assert "Mul" in operations
    elif operation == "max":
        assert "Max" in operations and "GatherElements" in operations
    elif operation == "reduce_sum" and rank > 1:
        assert "ReduceSum" in operations
    assert "LayerNormalization" not in operations


@pytest.mark.parametrize("operation", ["square", "max", "reduce_sum", "reshape"])
@pytest.mark.parametrize(
    "corruption",
    [
        "foreign-scope",
        "derived-batch",
        "result-shape",
        "dtype",
        "weakness",
        "parameter",
        "placement",
    ],
)
def test_symbolic_normalization_rejects_changed_native_contracts(
    operation: str, corruption: str
) -> None:
    (batch,) = export.symbolic_shape("B")
    source = jax.core.ShapedArray((batch, 3, 1), np.float32)
    target = source
    params: dict[str, Any] = {}
    if operation == "reduce_sum":
        target = source.update(shape=(batch,))
        params = {"axes": (1, 2)}
    elif operation == "reshape":
        target = source.update(shape=(batch, 3))
        params = {"new_sizes": target.shape, "dimensions": None}
    if corruption == "foreign-scope":
        (other,) = export.symbolic_shape("B")
        target = target.update(shape=(other, *target.shape[1:]))
    elif corruption == "derived-batch":
        target = target.update(shape=(2 * batch, *target.shape[1:]))
    elif corruption == "result-shape":
        target = target.update(shape=(batch, 4))
    elif corruption == "dtype":
        target = target.update(dtype=np.dtype(np.float64))
    elif corruption == "weakness":
        target = target.update(weak_type=True)
    elif corruption == "parameter":
        params["invented"] = True
    else:
        params["out_sharding"] = "placed"
    graph = _graph(operation, source, target, params)
    if operation == "max":
        graph.eqns[0].invars.append(graph.invars[0])
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert caught.value.primitive == operation


@pytest.mark.parametrize("axes", [(0,), (0, 2), (1, 1), (3,)])
def test_symbolic_tail_reduction_rejects_batch_and_malformed_axes(
    axes: tuple[int, ...],
) -> None:
    (batch,) = export.symbolic_shape("B")
    source = jax.core.ShapedArray((batch, 3, 2), np.float32)
    shape = tuple(dim for axis, dim in enumerate(source.shape) if axis not in axes)
    graph = _graph("reduce_sum", source, source.update(shape=shape), {"axes": axes})
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert caught.value.primitive == "reduce_sum"
    assert caught.value.category == (
        "profile-exclusion" if 0 in axes else "lowering-defect"
    )


@pytest.mark.parametrize(
    "corruption", ["regroup", "permute", "shift-batch", "rank-five", "zero-tail"]
)
def test_symbolic_singleton_reshape_retains_transform_exclusions(
    corruption: str,
) -> None:
    (batch,) = export.symbolic_shape("B")
    source = jax.core.ShapedArray((batch, 2, 3), np.float32)
    shape: tuple[Any, ...] = (batch, 1, 2, 3)
    dimensions = None
    if corruption == "regroup":
        shape = (batch, 6)
    elif corruption == "permute":
        dimensions = (0, 2, 1)
        shape = (batch, 3, 2)
    elif corruption == "shift-batch":
        shape = (1, batch, 2, 3)
    elif corruption == "rank-five":
        shape = (batch, 1, 2, 3, 1)
    else:
        source = source.update(shape=(batch, 0, 3))
        shape = (batch, 1, 0, 3)
    graph = _graph(
        "reshape",
        source,
        source.update(shape=shape),
        {"new_sizes": shape, "dimensions": dimensions},
    )
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    if corruption == "regroup":
        assert caught.value.primitive == "reshape"
        assert caught.value.category == "profile-exclusion"


def test_symbolic_singleton_reshape_distinguishes_bad_result_metadata() -> None:
    (batch,) = export.symbolic_shape("B")
    source = jax.core.ShapedArray((batch, 3), np.float32)
    graph = _graph(
        "reshape",
        source,
        source.update(shape=(batch, 1, 3)),
        {"new_sizes": (batch, 3, 1), "dimensions": None},
    )
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert caught.value.primitive == "reshape"
    assert caught.value.category == "lowering-defect"


@pytest.mark.parametrize("rank", [1, 4])
@pytest.mark.parametrize("opset", [16, 23])
def test_symbolic_min_remains_outside_the_reviewed_max_packet(
    rank: int, opset: int
) -> None:
    (batch,) = export.symbolic_shape("B")
    source = jax.core.ShapedArray((batch, *(2 for _ in range(rank - 1))), np.float32)
    graph = _graph("min", source, source)
    graph.eqns[0].invars.append(graph.invars[0])
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, opset)
    assert caught.value.primitive == "min"
    assert caught.value.category == "profile-exclusion"
