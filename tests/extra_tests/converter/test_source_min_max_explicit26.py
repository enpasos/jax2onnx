# tests/extra_tests/converter/test_source_min_max_explicit26.py

"""Explicit26 reuses signed-zero selection; other numerical obligations stay open."""

from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
from jax import export
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter.ir_optimizations import optimize_graph
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.plugins.jax.lax import clamp as clamp_module
from jax2onnx.precision import (
    CheckStatus,
    PRECISION_METADATA_KEY,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    validate_source_samples,
)
from tests.extra_tests.converter.test_source_min_max_zero_sign import (
    _bits,
    _grid,
    _structure,
)
from tests.extra_tests.converter.test_source_model_native_bindings import _fresh
from tests.extra_tests.converter.test_source_mlp_runtime_acceptance import (
    _runtime_configuration,
)
from tests.extra_tests.converter.test_source_symbolic_contraction import (
    _isolated_graph,
    _rebind,
)


def _body(x: Any, y: Any) -> Any:
    shape = jax.lax.broadcast_shapes(x.shape, y.shape)

    def expand(value: Any) -> Any:
        dimensions = tuple(range(len(shape) - value.ndim, len(shape)))
        return jax.lax.broadcast_in_dim(value, shape, dimensions)

    return (
        jax.lax.min(x, y),
        jax.lax.max(x, y),
        jax.lax.clamp(expand(y), expand(x), expand(y)),
        x,
        y,
    )


@onnx_function(name="Explicit26ZeroSelection")
def _named(x: Any, y: Any) -> Any:
    return _body(x, y)


def _call_named(x: Any, y: Any) -> Any:
    return _named(x, y)


def _if_body(x: Any, y: Any, predicate: Any) -> Any:
    def other(pair: Any) -> Any:
        lo, hi, clamped, first, second = _call_named(*pair)
        return hi, lo, clamped, first, second

    return jax.lax.cond(predicate, lambda pair: _call_named(*pair), other, (x, y))


def _retain(returned: Any, path: Path, stage: str) -> tuple[np.ndarray, ...]:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = path.with_suffix(".json")
    event.write_text(json.dumps(dict(stage=stage, status="returned")))
    leaves = jax.tree.leaves(jax.device_get(returned))
    np.savez(path, **{f"leaf_{index}": leaf for index, leaf in enumerate(leaves)})
    event.write_text(
        json.dumps(
            dict(
                stage=stage,
                status="archived",
                leaves=len(leaves),
                tree=str(jax.tree.structure(returned)),
            )
        )
    )
    return tuple(np.asarray(value) for value in leaves)


def _same(actual: Any, expected: Any) -> None:
    assert len(actual) == len(expected), "selected output arity differs"
    for value, reference in zip(actual, expected, strict=True):
        assert value.dtype == reference.dtype and value.shape == reference.shape
        assert value.tobytes(order="C") == reference.tobytes(
            order="C"
        ), "selected output bytes differ"


def _native(function: Any, inputs: Any, directory: Path) -> tuple[Any, ...]:
    compiled = jax.jit(_fresh(function))
    references = []
    before = tuple(value.tobytes(order="C") for value in inputs)
    for stage, call in (("fresh", compiled), ("warmed", compiled), ("eager", function)):
        try:
            returned = call(*inputs)
        except BaseException as exc:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"native-{stage}-exception.json").write_text(
                json.dumps(
                    dict(status="raised", type=type(exc).__qualname__, message=str(exc))
                )
            )
            raise
        actual = _retain(returned, directory / f"native-{stage}.npz", stage)
        assert isinstance(returned, tuple) or len(actual) == 1
        references.append(actual)
    assert tuple(value.tobytes(order="C") for value in inputs) == before
    return tuple(references)


def _cpu(model: Any, inputs: Any, directory: Path) -> Any:
    blob = model.SerializeToString()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "artifact.onnx").write_bytes(blob)
    onnx.checker.check_model(model, full_check=True)
    for label, profile in (
        ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        phase = "session"
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = profile
            session = ort.InferenceSession(
                blob, options, providers=["CPUExecutionProvider"]
            )
            session.disable_fallback()
            phase = "run"
            feed = {}
            for port, value in zip(session.get_inputs(), inputs, strict=True):
                array = np.asarray(value)
                assert isinstance(array, np.ndarray)
                assert array.dtype == value.dtype and array.shape == value.shape
                assert array.tobytes(order="C") == value.tobytes(order="C")
                feed[port.name] = array
            returned = session.run(None, feed)
            actual = _retain(returned, directory / f"cpu-{label}.npz", label)
            assert session.get_providers() == ["CPUExecutionProvider"]
            assert session.get_session_options().graph_optimization_level == profile
            assert model.SerializeToString() == blob
            yield label, actual
        except BaseException as exc:
            (directory / f"cpu-{label}-exception.json").write_text(
                json.dumps(
                    dict(
                        status="raised",
                        stage=phase,
                        type=type(exc).__qualname__,
                        message=str(exc),
                    )
                )
            )
            raise


def _nodes(owner: Any) -> Any:
    for node in owner.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _nodes(attribute.g)


def _selected_structure(model: Any, *, named: bool) -> None:
    assert next(item.version for item in model.opset_import if not item.domain) == 26
    all_nodes = [
        node for owner in (model.graph, *model.functions) for node in _nodes(owner)
    ]
    assert not any(node.op_type == "Where" for node in all_nodes)
    definitions = {
        (function.domain, function.name, function.overload): function
        for function in model.functions
    }
    called = set()

    def names(ports: Any) -> tuple[str, ...]:
        return tuple(value if isinstance(value, str) else value.name for value in ports)

    def broadcast(left: Any, right: Any) -> tuple[Any, ...]:
        rank = max(len(left), len(right))
        left, right = (1,) * (rank - len(left)) + tuple(left), (1,) * (
            rank - len(right)
        ) + tuple(right)
        assert all(a == b or a == 1 or b == 1 for a, b in zip(left, right, strict=True))
        return tuple(b if a == 1 else a for a, b in zip(left, right, strict=True))

    class Scope:
        def __init__(self, owner: Any, bindings: Any, outer: Any = None) -> None:
            self.owner, self.bindings, self.outer = owner, bindings, outer
            self.nodes = {name: node for node in owner.node for name in node.output}
            ports = [
                port
                for port in (*owner.input, *owner.output)
                if not isinstance(port, str)
            ]
            self.infos = {info.name: info for info in (*ports, *owner.value_info)}
            self.constants = {
                value.name: value for value in getattr(owner, "initializer", ())
            }
            self.logical: dict[str, Any] = {}
            self.cache: dict[str, Any] = {}
            self.prepare()

        def strip(self, name: str) -> str:
            visited = set()
            while name in self.nodes and self.nodes[name].op_type == "Identity":
                assert name not in visited
                visited.add(name)
                node = self.nodes[name]
                assert len(node.input) == len(node.output) == 1 and not node.attribute
                name = node.input[0]
            return name

        def node(self, name: str, operation: str, arity: int) -> Any:
            name = self.strip(name)
            assert name in self.nodes, "selection producer is not local"
            node = self.nodes[name]
            assert (
                node.op_type == operation
                and len(node.input) == arity
                and len(node.output) == 1
            ), "wrong selection producer"
            return node

        def constant(self, name: str) -> Any:
            name = self.strip(name)
            if name in self.constants:
                return onnx.numpy_helper.to_array(self.constants[name])
            node = self.node(name, "Constant", 0)
            assert len(node.attribute) == 1 and node.attribute[0].name == "value"
            return onnx.numpy_helper.to_array(node.attribute[0].t)

        def axes(self, name: str) -> None:
            value = self.constant(name)
            assert (
                value.dtype == np.dtype(np.int64)
                and value.shape == (1,)
                and value.tolist() == [0]
            )

        def axis(self, node: Any) -> None:
            assert (
                len(node.attribute) == 1
                and node.attribute[0].name == "axis"
                and node.attribute[0].i == 0
            )

        def shape(self, name: str) -> tuple[Any, ...]:
            name = self.strip(name)
            if name in self.infos:
                return tuple(
                    dimension.dim_param or dimension.dim_value
                    for dimension in self.infos[name].type.tensor_type.shape.dim
                )
            if name in self.logical:
                _, left, right = self.logical[name]
                return broadcast(self.shape(left), self.shape(right))
            if name in self.nodes and self.nodes[name].op_type in {"Min", "Max", "Add"}:
                node = self.nodes[name]
                return broadcast(self.shape(node.input[0]), self.shape(node.input[1]))
            assert self.outer is not None, "missing exact broadcast metadata"
            return self.outer.shape(name)

        def dtype(self, name: str) -> int:
            name = self.strip(name)
            if name in self.infos:
                return self.infos[name].type.tensor_type.elem_type
            if name in self.logical:
                return self.dtype(self.logical[name][1])
            if name in self.nodes and self.nodes[name].op_type in {
                "Min",
                "Max",
                "Add",
                "Neg",
                "Reshape",
                "Expand",
            }:
                return self.dtype(self.nodes[name].input[0])
            assert self.outer is not None, "missing exact typed selection metadata"
            return self.outer.dtype(name)

        def prepare(self) -> None:
            used_bases = set()
            for gather in self.owner.node:
                if gather.op_type != "GatherElements":
                    continue
                assert len(gather.input) == 2 and len(gather.output) == 1
                self.axis(gather)
                choices = self.node(gather.input[0], "Concat", 2)
                self.axis(choices)
                base_choice = self.node(choices.input[0], "Unsqueeze", 2)
                correction_choice = self.node(choices.input[1], "Unsqueeze", 2)
                self.axes(base_choice.input[1])
                self.axes(correction_choice.input[1])
                base_name = self.strip(base_choice.input[0])
                assert base_name in self.nodes
                base = self.nodes[base_name]
                assert (
                    base.op_type in {"Min", "Max"}
                    and len(base.input) == 2
                    and len(base.output) == 1
                )
                used_bases.add(base_name)
                left, right = map(self.strip, base.input)
                dtype = self.dtype(base_name)
                assert dtype in {
                    onnx.TensorProto.FLOAT16,
                    onnx.TensorProto.FLOAT,
                    onnx.TensorProto.DOUBLE,
                }
                assert self.dtype(left) == self.dtype(right) == dtype
                index = self.node(gather.input[1], "Unsqueeze", 2)
                self.axes(index.input[1])
                cast = self.node(index.input[0], "Cast", 1)
                assert (
                    len(cast.attribute) == 1
                    and cast.attribute[0].name == "to"
                    and cast.attribute[0].i == onnx.TensorProto.INT64
                )
                ties = self.node(cast.input[0], "And", 2)
                equal_inputs = []
                for name in ties.input:
                    equal = self.node(name, "Equal", 2)
                    zero = self.constant(equal.input[1])
                    assert (
                        zero.shape == ()
                        and zero.dtype
                        == np.dtype(
                            {
                                onnx.TensorProto.FLOAT16: "float16",
                                onnx.TensorProto.FLOAT: "float32",
                                onnx.TensorProto.DOUBLE: "float64",
                            }[dtype]
                        )
                        and not _bits(zero).any()
                    )
                    equal_inputs.append(self.strip(equal.input[0]))
                assert tuple(equal_inputs) in (
                    (left, right),
                    (right, left),
                ), "zero selector operands differ from actual baseline"
                correction = self.node(correction_choice.input[0], "Gather", 2)
                self.axis(correction)
                table = self.constant(correction.input[0])
                assert (
                    table.shape == (2,)
                    and table.dtype == zero.dtype
                    and _bits(table).tolist()
                    == [0, 1 << (8 * table.dtype.itemsize - 1)]
                ), "zero correction table differs from exact typed zero words"
                sign_index = self.node(correction.input[1], "Cast", 1)
                assert (
                    len(sign_index.attribute) == 1
                    and sign_index.attribute[0].name == "to"
                    and sign_index.attribute[0].i == onnx.TensorProto.INT64
                )
                sign = self.node(
                    sign_index.input[0], "Or" if base.op_type == "Min" else "And", 2
                )
                assert not sign.attribute
                operands = []
                for name in sign.input:
                    less = self.node(name, "Less", 2)
                    reciprocal = self.node(less.input[0], "Reciprocal", 1)
                    assert not less.attribute and not reciprocal.attribute
                    sign_zero = self.constant(less.input[1])
                    assert (
                        sign_zero.shape == ()
                        and sign_zero.dtype == zero.dtype
                        and not _bits(sign_zero).any()
                    ), "zero sign comparison requires original typed positive zero"
                    operands.append(self.strip(reciprocal.input[0]))
                assert tuple(operands) in (
                    (left, right),
                    (right, left),
                ), "zero correction operands differ from actual baseline"
                outputs = [
                    node
                    for node in self.owner.node
                    if node.op_type == "Squeeze"
                    and self.strip(node.input[0]) == gather.output[0]
                ]
                assert outputs, "selection has no logical output"
                for output in outputs:
                    assert len(output.input) == 2 and len(output.output) == 1
                    self.axes(output.input[1])
                    self.logical[output.output[0]] = (base.op_type, left, right)
            assert {
                node.output[0]
                for node in self.owner.node
                if node.op_type in {"Min", "Max"}
            } == used_bases, "unselected baseline escaped its exact selection"

        def data(self, name: str, ancestors: frozenset[str] = frozenset()) -> Any:
            name = self.strip(name)
            assert name not in ancestors, "cyclic data provenance"
            if name in self.cache:
                return self.cache[name]
            if name in self.bindings:
                return self.bindings[name]
            if name in self.logical:
                operation, left, right = self.logical[name]
                result = (
                    operation,
                    self.data(left, ancestors | {name}),
                    self.data(right, ancestors | {name}),
                )
            elif name not in self.nodes:
                assert (
                    self.outer is not None
                ), "data is not rooted at actual source formals"
                result = self.outer.data(name)
            else:
                node = self.nodes[name]
                if node.op_type in {"Expand", "Reshape"}:
                    assert len(node.input) == 2 and self.dtype(name) == self.dtype(
                        node.input[0]
                    )
                    source_shape, target_shape = self.shape(node.input[0]), self.shape(
                        name
                    )
                    if node.op_type == "Expand":
                        assert (
                            broadcast(source_shape, target_shape)
                            == target_shape
                            == common_shape
                        )
                    else:
                        assert tuple(
                            dimension for dimension in source_shape if dimension != 1
                        ) == tuple(
                            dimension for dimension in target_shape if dimension != 1
                        )
                    result = self.data(node.input[0], ancestors | {name})
                elif (node.domain, node.op_type, node.overload) in definitions:
                    key = (node.domain, node.op_type, node.overload)
                    definition = definitions[key]
                    assert (
                        len(node.input) == len(definition.input) == 2
                        and len(node.output) == len(definition.output) == 5
                    )
                    bindings = {
                        formal: self.data(actual, ancestors | {name})
                        for formal, actual in zip(
                            definition.input, node.input, strict=True
                        )
                    }
                    child = Scope(definition, bindings)
                    results = tuple(child.data(output) for output in definition.output)
                    called.add(key)
                    self.cache.update(zip(node.output, results, strict=True))
                    result = self.cache[name]
                elif node.op_type == "If":
                    branches = {
                        attribute.name: attribute.g for attribute in node.attribute
                    }
                    assert set(branches) == {"then_branch", "else_branch"}
                    then = Scope(branches["then_branch"], {}, self)
                    other = Scope(branches["else_branch"], {}, self)
                    true_values = tuple(
                        then.data(output) for output in names(then.owner.output)
                    )
                    false_values = tuple(
                        other.data(output) for output in names(other.owner.output)
                    )
                    assert len(node.output) == len(true_values) == len(false_values)
                    results = tuple(
                        first if first == second else ("If", first, second)
                        for first, second in zip(true_values, false_values, strict=True)
                    )
                    self.cache.update(zip(node.output, results, strict=True))
                    result = self.cache[name]
                else:
                    raise AssertionError("unknown or unselected data provenance")
            self.cache[name] = result
            return result

    formals = names(model.graph.input)
    assert len(formals) in (2, 3)
    root = Scope(
        model.graph, {name: ("formal", index) for index, name in enumerate(formals)}
    )
    common_shape = broadcast(root.shape(formals[0]), root.shape(formals[1]))
    first, second = ("formal", 0), ("formal", 1)
    minimum, maximum = ("Min", first, second), ("Max", first, second)
    clamped = ("Min", maximum, second)
    conditional = any(node.op_type == "If" for node in model.graph.node)
    expected = (
        (("If", minimum, maximum), ("If", maximum, minimum), clamped, first, second)
        if conditional
        else (minimum, maximum, clamped, first, second)
    )
    assert (
        tuple(root.data(output) for output in names(model.graph.output)) == expected
    ), "returned selected/direct output provenance differs"
    if named:
        targets = {key for key in definitions if "Explicit26ZeroSelection" in key[1]}
        assert targets and targets <= called


def _zero_values(dtype: Any, layout: str, size: int = 2) -> Any:
    x = np.asarray([-0.0, 0.0, -0.0, 0.0], dtype=dtype)
    y = np.asarray([0.0, -0.0, -0.0, 0.0], dtype=dtype)
    if layout == "scalar":
        return x[0].copy(), y[0].copy()
    if layout == "broadcast":
        return x[:2, None].copy(), y[None, :2].copy()
    if layout == "batch":
        return np.repeat(x[None, :], size, axis=0), np.repeat(y[None, :], size, axis=0)
    return x, y


def _zero_expected(x: Any, y: Any, swapped: bool) -> Any:
    left, right = np.broadcast_arrays(np.signbit(x), np.signbit(y))
    zero, negative = np.asarray(0.0, dtype=x.dtype), np.asarray(-0.0, dtype=x.dtype)
    minimum = np.where(left | right, negative, zero)
    maximum = np.where(left & right, negative, zero)
    clamped = np.broadcast_to(y, minimum.shape).copy()
    return (
        (maximum, minimum, clamped, x, y)
        if swapped
        else (minimum, maximum, clamped, x, y)
    )


@pytest.mark.parametrize("dtype", (np.float16, np.float32, np.float64))
@pytest.mark.parametrize("layout", ("scalar", "tensor", "broadcast", "batch"))
@pytest.mark.parametrize("scope", ("inline", "named", "if"))
def test_explicit26_zero_signs_clamp_composition_and_actual_owners(
    dtype: Any, layout: str, scope: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _runtime_configuration(True):
        observed = []
        original = clamp_module.lower_source_min_max

        def observe(ctx: Any, eqn: Any, *, operation: Any) -> bool:
            assert ctx.opset == 26
            accepted = original(ctx, eqn, operation=operation)
            assert accepted
            observed.append(operation)
            return accepted

        monkeypatch.setattr(clamp_module, "lower_source_min_max", observe)
        function = {"inline": _body, "named": _call_named, "if": _if_body}[scope]
        values = _zero_values(dtype, layout)
        specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values]
        if layout == "batch":
            batch = export.symbolic_shape("B")[0]
            specs = [jax.ShapeDtypeStruct((batch, 4), dtype)] * 2
        if scope == "if":
            specs.append(jax.ShapeDtypeStruct((), np.bool_))
        samples = []
        for size in (2, 5) if layout == "batch" else (2,):
            x, y = _zero_values(dtype, layout, size)
            for predicate in (False, True) if scope == "if" else (True,):
                inputs = (
                    (x, y, np.asarray(predicate, dtype=np.bool_))
                    if scope == "if"
                    else (x, y)
                )
                expected = _zero_expected(x, y, scope == "if" and not predicate)
                directory = tmp_path / f"B{size}-branch-{predicate}"
                references = _native(function, inputs, directory)
                for reference in references:
                    _same(reference, expected)
                samples.append((inputs, expected, references, directory))
        model = ui._to_onnx(
            function, specs, precision_policy=PrecisionPolicy.SOURCE, opset=26
        )
        _selected_structure(model, named=scope != "inline")
        assert observed and len(observed) % 2 == 0
        assert all(
            observed[index : index + 2] == ["Max", "Min"]
            for index in range(0, len(observed), 2)
        )
        for inputs, expected, references, directory in samples:
            for _, actual in _cpu(model, inputs, directory):
                _same(actual, expected)
                for reference in references:
                    _same(actual, reference)


@pytest.mark.parametrize("operation", ("Min", "Max"))
@pytest.mark.parametrize("dtype", (np.float16, np.float32, np.float64))
def test_explicit26_grid_records_actual_status_and_retains_failed_raising_gate(
    operation: str, dtype: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _runtime_configuration(True):
        function = jax.lax.min if operation == "Min" else jax.lax.max
        values, real_ties = _grid(dtype)
        references = _native(function, values, tmp_path / "native")
        specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values]
        model = ui._to_onnx(
            function, specs, precision_policy=PrecisionPolicy.SOURCE, opset=26
        )
        _structure(model, operation)
        inspected = ir.from_proto(model)
        inspected.metadata_props.pop(PRECISION_METADATA_KEY)
        base = next(
            node.outputs[0] for node in inspected.graph if node.op_type == operation
        )
        ties = next(
            node.outputs[0] for node in inspected.graph if node.op_type == "And"
        )
        inspected.graph.outputs.extend([base, ties])
        optimize_graph(inspected, precision_policy=PrecisionPolicy.SOURCE)
        inspection = ir.to_proto(inspected)
        outputs = dict(_cpu(model, values, tmp_path / "facade"))
        observations = dict(_cpu(inspection, values, tmp_path / "inspection"))
        for optimization, (output,) in outputs.items():
            observed, baseline, actual_ties = observations[optimization]
            assert output.tobytes() == observed.tobytes()
            np.testing.assert_array_equal(actual_ties, real_ties, strict=True)
            assert np.array_equal(
                _bits(output)[~real_ties], _bits(baseline)[~real_ties]
            )
            for index, (reference,) in enumerate(references):
                np.testing.assert_array_equal(
                    np.isnan(output), np.isnan(reference), strict=True
                )
                assert np.array_equal(
                    _bits(output)[real_ties], _bits(reference)[real_ties]
                )
                mismatches = (_bits(output) != _bits(reference)) & ~np.isnan(reference)
                inherited = (
                    (_bits(baseline) != _bits(reference))
                    & ~np.isnan(reference)
                    & ~real_ties
                )
                np.testing.assert_array_equal(mismatches, inherited, strict=True)
                calls = []
                original_run = ort.InferenceSession.run

                def retaining_run(session: Any, *args: Any, **kwargs: Any) -> Any:
                    returned = original_run(session, *args, **kwargs)
                    _retain(
                        returned,
                        tmp_path / f"validator-{optimization}-{index}-{len(calls)}.npz",
                        "sample-validator",
                    )
                    calls.append(True)
                    return returned

                with monkeypatch.context() as patch:
                    patch.setattr(ort.InferenceSession, "run", retaining_run)
                    result = validate_source_samples(
                        model,
                        samples=[
                            SourceSample(values, [reference], SampleCriteria("bitwise"))
                        ],
                        optimization=optimization,
                    )
                assert calls
                failed = output.tobytes() != reference.tobytes()
                assert result.status is (
                    CheckStatus.FAILED if failed else CheckStatus.VALIDATED
                )
                (tmp_path / f"actual-status-{optimization}-{index}.json").write_text(
                    json.dumps(
                        dict(
                            artifact=hashlib.sha256(
                                model.SerializeToString()
                            ).hexdigest(),
                            requested_opset=26,
                            native_role=index,
                            non_nan_mismatches=np.argwhere(mismatches).tolist(),
                            target=asdict(result.target),
                            status=result.status.value,
                            historical_other_target_failures="RETAINED",
                        ),
                        indent=2,
                    )
                )
                if failed:
                    with pytest.raises(SampleAcceptanceError, match="bitwise"):
                        result.require_sample_acceptance()
                else:
                    result.require_sample_acceptance()


@pytest.mark.parametrize("operation", ("Min", "Max"))
def test_explicit26_wrong_zero_choice_is_retained_and_fails_exact_criterion(
    operation: str, tmp_path: Path
) -> None:
    with _runtime_configuration(True):
        function = jax.lax.min if operation == "Min" else jax.lax.max
        values = (
            np.asarray([-0.0, 0.0, -0.0, 0.0], np.float32),
            np.asarray([0.0, -0.0, -0.0, 0.0], np.float32),
        )
        expected = _native(function, values, tmp_path / "native")[0]
        model = ui._to_onnx(
            function,
            [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values],
            precision_policy=PrecisionPolicy.SOURCE,
            opset=26,
        )
        corrupted = copy.deepcopy(model)
        tie = next(node for node in corrupted.graph.node if node.op_type == "And")
        tie.op_type = "Xor"
        onnx.checker.check_model(corrupted, full_check=True)
        with pytest.raises(AssertionError, match="selected output bytes differ"):
            for _, actual in _cpu(corrupted, values, tmp_path / "corrupted"):
                _same(actual, expected)


@pytest.mark.parametrize("operation", ("min", "max", "clamp"))
@pytest.mark.parametrize("defect", ("dtype", "shape", "placement", "effects"))
def test_explicit26_preserves_existing_malformed_contract_rejections(
    operation: str, defect: str
) -> None:
    with _runtime_configuration(True):
        native = jax.make_jaxpr(_body)(
            jax.ShapeDtypeStruct((2, 4), np.float32),
            jax.ShapeDtypeStruct((2, 4), np.float32),
        )
        graph = _isolated_graph(native)
        equation = next(item for item in graph.eqns if item.primitive.name == operation)
        if defect == "dtype":
            _rebind(equation.outvars[0], dtype=np.float64)
        elif defect == "shape":
            _rebind(equation.outvars[0], shape=(2, 5))
        elif defect == "placement":
            equation.params["out_sharding"] = object()
        else:
            equation.effects = {object()}
        with pytest.raises(SourcePrecisionError):
            validate_source_jaxpr(graph, 26)
