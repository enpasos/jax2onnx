# tests/extra_tests/converter/test_source_tensor_matrix.py

"""Canonical SOURCE tensor×matrix contracts and bounded CPU numerical evidence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from flax import nnx
import jax
from jax import export
from jax.extend.core import jaxpr_as_fun
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
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
from jax2onnx.converter.source_representation import (
    normalized_shape_dimensions,
    representation_for_aval,
)
from jax2onnx.plugins.jax.lax.dot_general import DotGeneralPlugin
from jax2onnx.plugins.plugin_system import FunctionPlugin
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)

_DIRECT_CASES = (
    (3, np.float32, False, False, "dot", False),
    (3, np.float32, True, True, "matmul", True),
    (3, np.float64, False, True, "operator", False),
    (3, np.float64, True, False, "dot", True),
    (4, np.float32, False, True, "operator", False),
    (4, np.float32, True, False, "matmul", True),
    (4, np.float64, False, False, "dot", True),
    (4, np.float64, True, True, "operator", False),
)
_LINEAR: Any = None
_ORIGINAL_CALLS = jnp.matmul, nnx.Linear.__call__


@onnx_function(name="SourceTensorMatrixLinear")
def _named_linear(x: Any) -> Any:
    return _LINEAR(x)


def _globals() -> dict[str, Any]:
    return {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    original = _globals()
    effective = tuple(getattr(jax.config, key) for key in original)
    try:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for key, value in original.items():
            jax.config.update(key, value)
        assert tuple(getattr(jax.config, key) for key in original) == effective
        assert active_precision_policy() is PrecisionPolicy.LEGACY
        assert (jnp.matmul, nnx.Linear.__call__) == _ORIGINAL_CALLS


def _body(closed: Any) -> Any:
    inner = getattr(closed, "jaxpr", closed)
    return inner if inner is not closed else closed


def _payload(value: Any) -> tuple[Any, ...]:
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.tobytes()


def _same(actual: Any, expected: Any) -> None:
    assert _payload(actual) == _payload(expected)


def _fresh(fn: Any) -> Any:
    def call(*args: Any) -> Any:
        return fn(*args)

    return call


def _signature(variables: Any) -> tuple[Any, ...]:
    return tuple(
        (
            np.dtype(var.aval.dtype).name,
            normalized_shape_dimensions(var.aval.shape),
            bool(var.aval.weak_type),
        )
        for var in variables
    )


def _dot_contracts(closed: Any) -> list[Any]:
    result = []
    for eqn in _body(closed).eqns:
        if eqn.primitive.name == "dot_general":
            result.append(
                (
                    _signature(eqn.invars),
                    _signature(eqn.outvars),
                    repr(eqn.params["dimension_numbers"]),
                    repr(eqn.params.get("precision")),
                    str(eqn.params.get("preferred_element_type")),
                    repr(eqn.params.get("out_sharding")),
                )
            )
        for name in ("jaxpr", "call_jaxpr"):
            if name in eqn.params:
                result.extend(_dot_contracts(eqn.params[name]))
        for branch in eqn.params.get("branches", ()):
            result.extend(_dot_contracts(branch))
    return result


def _specs(rank: int, dtype: Any, symbolic: bool, captured: bool) -> list[Any]:
    leading: Any = export.symbolic_shape("B")[0] if symbolic else 2
    shape = (leading, 3, 4) if rank == 3 else (leading, 2, 3, 4)
    return [jax.ShapeDtypeStruct(shape, dtype)] + (
        [] if captured else [jax.ShapeDtypeStruct((4, 3), dtype)]
    )


def _kernel(dtype: Any) -> np.ndarray:
    return np.asarray(
        [[1, 0.25, -0.5], [1, -0.5, 0.25], [1, 0.125, 0.5], [1, 0.5, -0.25]],
        dtype,
    )


def _values(rank: int, batch: int, dtype: Any, precision: bool) -> np.ndarray:
    shape = (batch, 3, 4) if rank == 3 else (batch, 2, 3, 4)
    index = np.arange(np.prod(shape[:-1])).reshape(shape[:-1])
    if precision:
        value = np.empty(shape, dtype)
        value[..., 0] = 2**24 + index
        value[..., 1] = 1
        value[..., 2] = np.where(index % 2, -(2**24), 0)
        value[..., 3] = np.where(index % 2, 0.25, 0)
        return value
    # Every free coordinate has a different row, with exactly represented dyadics.
    return ((np.arange(np.prod(shape)).reshape(shape) - 11) / 16).astype(dtype)


def _target(captured: Any, entry: str, preferred: Any) -> Any:
    def call(x: Any, *runtime_weight: Any) -> Any:
        weight = captured if captured is not None else runtime_weight[0]
        if entry == "operator":
            result = x @ weight
        elif entry == "matmul":
            result = jnp.matmul(x, weight, preferred_element_type=preferred)
        else:
            result = jax.lax.dot_general(
                x,
                weight,
                (((x.ndim - 1,), (0,)), ((), ())),
                preferred_element_type=preferred,
            )
        # The native FP64→FP32 conversion has an actual Add consumer.
        consumed = result.astype(np.float32) + np.float32(0.5)
        return result, consumed, x, x, weight, weight

    return call


def _native_and_preserved(
    fn: Any, specs: Any, feeds: list[Any], directory: Path, *, named: bool = False
) -> tuple[Any, list[Any]]:
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    native = jax.make_jaxpr(_fresh(fn))(*specs)
    manifest = typed_trace_manifest(native)
    (directory / "native-trace.json").write_text(json.dumps(manifest, indent=2) + "\n")
    contracts = _dot_contracts(native)
    assert contracts
    for inputs, outputs, axes, precision, preferred, placement in contracts:
        lhs, rhs = inputs
        (result,) = outputs
        rank = len(lhs[1])
        assert rank in (2, 3, 4) and len(rhs[1]) == 2
        assert lhs[0] == rhs[0] == result[0] in {"float32", "float64"}
        assert result[1] == lhs[1][:-1] + (rhs[1][-1],)
        assert result[2] == (lhs[2] and rhs[2])
        assert axes == repr((((rank - 1,), (0,)), ((), ())))
        assert precision in {"None", repr((jax.lax.Precision.DEFAULT,) * 2)}
        assert preferred in {"None", result[0]}
        assert placement == "None"
    (directory / "native-dots.json").write_text(json.dumps(contracts, indent=2) + "\n")
    if named:
        outer = _body(native)
        selector = next(
            e for e in outer.eqns if e.primitive.name == "convert_element_type"
        )
        condition = next(e for e in outer.eqns if e.primitive.name == "cond")
        assert selector.invars[0] is outer.invars[1]
        assert np.dtype(selector.invars[0].aval.dtype) == np.dtype(np.bool_)
        assert np.dtype(selector.outvars[0].aval.dtype) == np.dtype(np.int32)
        assert condition.invars[0] is selector.outvars[0]
    constants = tuple(_payload(value) for value in native.consts)
    references = []
    for index, values in enumerate(feeds):
        eager = tuple(np.asarray(v).copy() for v in jax.tree.leaves(fn(*values)))
        compiled = tuple(
            np.asarray(v).copy() for v in jax.tree.leaves(jax.jit(_fresh(fn))(*values))
        )
        np.savez(
            directory / f"native-{index}.npz",
            **{f"input-{i}": np.asarray(v) for i, v in enumerate(values)},
            **{f"eager-{i}": v for i, v in enumerate(eager)},
            **{f"jit-{i}": v for i, v in enumerate(compiled)},
        )
        assert len(eager) == len(compiled)
        for actual, expected in zip(compiled, eager, strict=True):
            _same(actual, expected)
        references.append((eager, compiled))
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(_fresh(fn))(*specs)
        assert _signature(_body(activated).invars) == _signature(_body(native).invars)
        assert _signature(_body(activated).outvars) == _signature(_body(native).outvars)
        if not named:
            assert typed_trace_manifest(activated) == manifest
            assert tuple(_payload(value) for value in activated.consts) == constants
        else:
            outer = _body(activated)
            source_selector = next(
                e for e in outer.eqns if e.primitive.name == "convert_element_type"
            )
            source_cond = next(e for e in outer.eqns if e.primitive.name == "cond")
            assert source_selector.params == selector.params
            assert source_selector.invars[0] is outer.invars[1]
            assert source_cond.invars[0] is source_selector.outvars[0]
        if not named:
            for index, (values, reference) in enumerate(
                zip(feeds, references, strict=True)
            ):
                outputs = []
                for runner in (fn, jax.jit(_fresh(fn))):
                    outputs.append(
                        tuple(np.asarray(v) for v in jax.tree.leaves(runner(*values)))
                    )
                np.savez(
                    directory / f"activated-{index}.npz",
                    **{f"input-{i}": np.asarray(v) for i, v in enumerate(values)},
                    **{f"eager-{i}": v for i, v in enumerate(outputs[0])},
                    **{f"jit-{i}": v for i, v in enumerate(outputs[1])},
                )
                for actuals, expecteds in zip(outputs, reference, strict=True):
                    for actual, expected in zip(actuals, expecteds, strict=True):
                        _same(actual, expected)
        else:
            # Function primitives represent export boundaries and have no ordinary
            # CPU MLIR lowering. Their actual exported body is evaluated below;
            # ordinary whole-carrier execution is checked after activation exits.
            (directory / "named-execution-evidence.json").write_text(
                json.dumps(
                    {
                        "activated-typed-trace": "VALIDATED",
                        "activated-whole-carrier-runtime": "NOT_ESTABLISHED",
                        "actual-exported-body-runtime": "PENDING",
                        "restored-whole-carrier-runtime": "PENDING",
                    },
                    indent=2,
                )
                + "\n"
            )
    assert typed_trace_manifest(native) == manifest
    assert tuple(_payload(value) for value in native.consts) == constants
    return native, references


def _artifact(fn: Any, specs: Any, native: Any, directory: Path) -> Any:
    proto = ui._to_onnx(fn, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23)
    path = directory / "tensor-matrix.onnx"
    onnx.save_model(proto, path)
    reloaded = onnx.load_model(path)
    onnx.checker.check_model(reloaded, full_check=True)
    assert reloaded.SerializeToString() == proto.SerializeToString()
    report = get_precision_report(reloaded)
    assert report is not None and report.profile == "source-static-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    assert tuple((v.dtype, v.shape, v.weak_type) for v in report.traced_outputs) == (
        _signature(_body(native).outvars)
    )
    return reloaded


def _run(model: Any, feeds: Any, references: Any, directory: Path) -> None:
    payload = model.SerializeToString()
    counters = {"sessions": 0, "runs": 0}
    receipt = directory / "runtime-counts.json"
    receipt.write_text(json.dumps(counters) + "\n")
    try:
        for optimize in (False, True):
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                if optimize
                else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            session = ort.InferenceSession(
                payload, options, providers=["CPUExecutionProvider"]
            )
            counters["sessions"] += 1
            receipt.write_text(json.dumps(counters) + "\n")
            assert session.get_providers() == ["CPUExecutionProvider"]
            for index, (values, reference) in enumerate(
                zip(feeds, references, strict=True)
            ):
                actual = session.run(
                    None,
                    dict(
                        zip((v.name for v in session.get_inputs()), values, strict=True)
                    ),
                )
                counters["runs"] += 1
                receipt.write_text(json.dumps(counters) + "\n")
                np.savez(
                    directory / f"target-{int(optimize)}-{index}.npz",
                    **{f"input-{i}": np.asarray(v) for i, v in enumerate(values)},
                    **{f"eager-{i}": v for i, v in enumerate(reference[0])},
                    **{f"jit-{i}": v for i, v in enumerate(reference[1])},
                    **{f"actual-{i}": v for i, v in enumerate(actual)},
                )
                assert len(actual) == len(reference[0])
                for result, eager, compiled in zip(actual, *reference, strict=True):
                    assert np.isfinite(result).all() and np.isfinite(eager).all()
                    _same(result, eager)
                    _same(result, compiled)
        assert counters == {"sessions": 2, "runs": 2 * len(feeds)}
        assert model.SerializeToString() == payload
    finally:
        receipt.write_text(json.dumps(counters) + "\n")


def _origin(nodes: Any, name: str) -> str:
    producers = {output: node for node in nodes for output in node.output}
    visited = set()
    while name in producers and producers[name].op_type == "Identity":
        assert name not in visited
        visited.add(name)
        node = producers[name]
        assert not node.domain and len(node.input) == len(node.output) == 1
        name = node.input[0]
    return name


def _port(value: Any, dtype: Any, shape: Any) -> None:
    tensor = value.type.tensor_type
    assert tensor.elem_type == onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    assert tuple(
        d.dim_param if d.HasField("dim_param") else d.dim_value
        for d in tensor.shape.dim
    ) == tuple(shape)


def _direct_structure(model: Any, dtype: Any, captured: bool, weight: Any) -> None:
    nodes = list(model.graph.node)
    assert not model.functions
    operations = [node.op_type for node in nodes]
    assert set(operations) <= {"MatMul", "Identity", "Cast", "Add", "Constant"}
    assert operations.count("MatMul") == operations.count("Add") == 1
    assert operations.count("Cast") == int(dtype is np.float64)
    dot = next(node for node in nodes if node.op_type == "MatMul")
    assert _origin(nodes, dot.input[0]) == model.graph.input[0].name
    assert _origin(nodes, model.graph.output[0].name) == dot.output[0]
    if captured:
        arrays = {
            v.name: onnx.numpy_helper.to_array(v) for v in model.graph.initializer
        }
        _same(arrays[_origin(nodes, dot.input[1])], weight)
    else:
        assert _origin(nodes, dot.input[1]) == model.graph.input[1].name
    consumed = next(node for node in nodes if node.op_type == "Add")
    assert _origin(nodes, model.graph.output[1].name) == consumed.output[0]
    if dtype is np.float64:
        cast = next(node for node in nodes if node.op_type == "Cast")
        assert _origin(nodes, cast.input[0]) == dot.output[0]
        assert (
            next(a.i for a in cast.attribute if a.name == "to")
            == onnx.TensorProto.FLOAT
        )
        assert cast.output[0] in [_origin(nodes, name) for name in consumed.input]
    else:
        assert dot.output[0] in [_origin(nodes, name) for name in consumed.input]
    assert _origin(nodes, model.graph.output[2].name) == model.graph.input[0].name
    assert _origin(nodes, model.graph.output[3].name) == model.graph.input[0].name
    assert _origin(nodes, model.graph.output[4].name) == _origin(nodes, dot.input[1])
    assert _origin(nodes, model.graph.output[5].name) == _origin(nodes, dot.input[1])
    typed = {
        v.name: v
        for v in (*model.graph.input, *model.graph.output, *model.graph.value_info)
    }
    code = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    for name in (dot.input[0], dot.input[1], dot.output[0]):
        if name in typed:
            assert typed[name].type.tensor_type.elem_type == code
        else:
            assert captured and name in {v.name for v in model.graph.initializer}
            assert (
                next(v.data_type for v in model.graph.initializer if v.name == name)
                == code
            )


@pytest.mark.parametrize(
    "rank,dtype,symbolic,captured,entry,preferred",
    _DIRECT_CASES,
    ids=[
        "rank3-f32-static-runtime-dot-none",
        "rank3-f32-B-captured-matmul-same",
        "rank3-f64-static-captured-operator-none",
        "rank3-f64-B-runtime-dot-same",
        "rank4-f32-static-captured-operator-none",
        "rank4-f32-B-runtime-matmul-same",
        "rank4-f64-static-runtime-dot-same",
        "rank4-f64-B-captured-operator-none",
    ],
)
def test_canonical_tensor_matrix_keeps_free_axes_aliases_and_consumed_cast(
    rank: int,
    dtype: Any,
    symbolic: bool,
    captured: bool,
    entry: str,
    preferred: bool,
    tmp_path: Path,
) -> None:
    kernel = _kernel(dtype)
    original_kernel = _payload(kernel)
    fn = _target(kernel if captured else None, entry, dtype if preferred else None)
    feeds = []
    for batch in (2, 5) if symbolic else (2,):
        for precision in (False, True) if dtype is np.float64 else (False,):
            weight = kernel if captured or precision else kernel + dtype(batch / 16)
            x = _values(rank, batch, dtype, precision)
            feeds.append((x,) if captured else (x, weight))
    specs = _specs(rank, dtype, symbolic, captured)
    native, references = _native_and_preserved(fn, specs, feeds, tmp_path)
    dots = _dot_contracts(native)
    assert len(dots) == 1
    assert dots[0][2] == repr((((rank - 1,), (0,)), ((), ())))
    for values, (eager, _) in zip(feeds, references, strict=True):
        x, weight = values[0], kernel if captured else values[1]
        _same(eager[0], x @ weight)
        for actual, expected in zip(eager[2:], (x, x, weight, weight), strict=True):
            _same(actual, expected)
        if dtype is np.float64 and np.max(x) >= 2**24:
            coordinate = np.arange(np.prod(x.shape[:-1])).reshape(x.shape[:-1])
            expected = np.where(
                coordinate % 2, coordinate + 1.25, 2**24 + coordinate + 1
            )
            _same(eager[0][..., 0], expected.astype(np.float64))
            assert eager[0].reshape(-1, 3)[0, 0] == 2**24 + 1
    model = _artifact(fn, specs, native, tmp_path)
    _direct_structure(model, dtype, captured, kernel)
    leading: Any = "B" if symbolic else 2
    shape = (leading, 3, 4) if rank == 3 else (leading, 2, 3, 4)
    _port(model.graph.input[0], dtype, shape)
    _port(model.graph.output[0], dtype, shape[:-1] + (3,))
    _port(model.graph.output[1], np.float32, shape[:-1] + (3,))
    _run(model, feeds, references, tmp_path)
    assert _payload(kernel) == original_kernel


@contextmanager
def _observe_functions() -> Any:
    bodies, built = [], []
    original_trace = FunctionPlugin._trace_function_body
    original_build = FunctionPlugin._build_function

    def trace(plugin: Any, *args: Any, **kwargs: Any) -> Any:
        body = original_trace(plugin, *args, **kwargs)
        bodies.append(body)
        return body

    def build(plugin: Any, *args: Any, **kwargs: Any) -> Any:
        start = len(bodies)
        definition = original_build(plugin, *args, **kwargs)
        assert len(bodies) == start + 1
        built.append((definition, bodies[-1]))
        return definition

    with (
        patch.object(FunctionPlugin, "_trace_function_body", trace),
        patch.object(FunctionPlugin, "_build_function", build),
    ):
        yield built


def _serialized_arrays(model: Any) -> list[Any]:
    arrays = list(model.graph.initializer)

    def nodes(scope: Any) -> None:
        for node in scope.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    arrays.append(attribute.t)
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    arrays.extend(attribute.g.initializer)
                    nodes(attribute.g)

    nodes(model.graph)
    for function in model.functions:
        nodes(function)
    return [onnx.numpy_helper.to_array(value) for value in arrays]


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_named_nnx_linear_consumed_by_cond_keeps_bias_parameters_and_batch(
    dtype: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    linear = nnx.Linear(3, 6, dtype=dtype, param_dtype=dtype, rngs=nnx.Rngs(17))
    linear.kernel[...] = jnp.asarray((np.arange(18).reshape(3, 6) - 7) / 16, dtype)
    linear.bias[...] = jnp.asarray([0.25, -0.5, 0.75, -0.125, 0.375, -0.625], dtype)
    kernel, bias = linear.kernel[...], linear.bias[...]
    frozen = _payload(kernel), _payload(bias)
    variables = id(linear.kernel), id(linear.bias)
    monkeypatch.setattr(__import__(__name__, fromlist=["_LINEAR"]), "_LINEAR", linear)
    configuration = _globals()
    effective = tuple(getattr(jax.config, key) for key in configuration)
    original_calls = jnp.matmul, nnx.Linear.__call__, _named_linear

    def target(x: Any, positive: Any) -> Any:
        result = jax.lax.cond(
            positive,
            lambda value: _named_linear(value),
            lambda value: -_named_linear(value),
            x,
        )
        return result, x, x, linear.kernel[...], linear.bias[...]

    specs = _specs(4, dtype, True, True)
    specs[0] = jax.ShapeDtypeStruct(specs[0].shape[:-1] + (3,), dtype)
    specs.append(jax.ShapeDtypeStruct((), np.bool_))
    feeds = [
        (x, np.asarray(flag))
        for batch in (2, 5)
        for x in (_values(4, batch, dtype, False)[..., :3].copy(),)
        for flag in (True, False)
    ]
    native, references = _native_and_preserved(
        target, specs, feeds, tmp_path, named=True
    )
    native_manifest = typed_trace_manifest(native)
    native_constants = tuple(_payload(value) for value in native.consts)
    for (x, flag), (eager, _) in zip(feeds, references, strict=True):
        expected = x @ np.asarray(kernel) + np.asarray(bias)
        _same(eager[0], expected if flag else -expected)
        for actual, original in zip(eager[1:], (x, x, kernel, bias), strict=True):
            _same(actual, original)
    positive_references = {
        _payload(values[0]): (index, reference)
        for index, (values, reference) in enumerate(zip(feeds, references, strict=True))
        if bool(values[1])
    }
    assert len(positive_references) == 2
    with _observe_functions() as built:
        model = _artifact(target, specs, native, tmp_path)
    assert len(model.functions) == len(built) == 1
    definitions = {(fn.domain, fn.name): fn for fn in model.functions}
    assert len(definitions) == len(model.functions)
    body_contracts = []
    for definition, closed in built:
        function = definitions[definition.domain, definition.name]
        body_manifest = typed_trace_manifest(closed)
        body_constants = tuple(_payload(value) for value in closed.consts)
        (tmp_path / "actual-function-body-trace.json").write_text(
            json.dumps(body_manifest, indent=2) + "\n"
        )
        assert _signature(_body(closed).invars) == _signature(_body(native).invars[:1])
        assert _signature(_body(closed).outvars) == _signature(
            _body(native).outvars[:1]
        )
        body_contracts.extend(_dot_contracts(closed))
        constants = {
            id(var): _payload(value)
            for var, value in zip(_body(closed).constvars, closed.consts, strict=True)
        }
        native_dot = next(
            eqn for eqn in _body(closed).eqns if eqn.primitive.name == "dot_general"
        )
        assert constants[id(native_dot.invars[1])] == frozen[0]
        native_bias = next(
            eqn for eqn in _body(closed).eqns if eqn.primitive.name == "reshape"
        )
        assert constants[id(native_bias.invars[0])] == frozen[1]
        infos = {v.name: v for v in function.value_info}
        for names, variables_ in (
            (function.input, _body(closed).invars),
            (function.output, _body(closed).outvars),
        ):
            assert len(names) == len(variables_)
            for name, variable in zip(names, variables_, strict=True):
                representation = representation_for_aval(variable.aval)
                _port(
                    infos[name],
                    representation.physical_dtype,
                    normalized_shape_dimensions(representation.physical_shape),
                )
        operations = [node.op_type for node in function.node]
        assert operations.count("MatMul") == 1 and operations.count("Add") == 1
        assert not {"Cast", "Gemm", "Expand", "Transpose"} & set(operations)
        dot = next(node for node in function.node if node.op_type == "MatMul")
        add = next(node for node in function.node if node.op_type == "Add")
        assert dot.output[0] in [_origin(function.node, name) for name in add.input]
        assert _origin(function.node, function.output[0]) == add.output[0]
        producers = {name: node for node in function.node for name in node.output}

        def constant(name: str) -> Any:
            node = producers[_origin(function.node, name)]
            assert node.op_type == "Constant" and not node.domain
            value = next(a.t for a in node.attribute if a.name == "value")
            return onnx.numpy_helper.to_array(value)

        _same(constant(dot.input[1]), kernel)
        bias_name = next(
            name for name in add.input if _origin(function.node, name) != dot.output[0]
        )
        bias_producer = producers[_origin(function.node, bias_name)]
        if bias_producer.op_type == "Reshape":
            _same(constant(bias_producer.input[0]), bias)
        else:
            serialized_bias = constant(bias_name)
            assert serialized_bias.shape in {(6,), (1, 1, 1, 6)}
            assert serialized_bias.dtype == np.dtype(dtype)
            assert serialized_bias.tobytes() == np.asarray(bias).tobytes()
        evaluator = jaxpr_as_fun(closed)
        for index, values in enumerate(feeds):
            reference_index, positive_reference = positive_references[
                _payload(values[0])
            ]
            outputs = []
            for runner in (evaluator, jax.jit(_fresh(evaluator))):
                outputs.append(
                    tuple(np.asarray(value).copy() for value in runner(values[0]))
                )
            np.savez(
                tmp_path / f"actual-function-body-{index}.npz",
                input=values[0],
                original_carrier_flag=values[1],
                native_positive_reference_index=reference_index,
                **{f"eager-{i}": value for i, value in enumerate(outputs[0])},
                **{f"jit-{i}": value for i, value in enumerate(outputs[1])},
            )
            for actual, reference in zip(outputs, positive_reference, strict=True):
                assert len(actual) == 1
                _same(actual[0], reference[0])
        assert typed_trace_manifest(closed) == body_manifest
        assert tuple(_payload(value) for value in closed.consts) == body_constants
    assert set(body_contracts) == set(_dot_contracts(native))
    conditionals = [node for node in model.graph.node if node.op_type == "If"]
    assert len(conditionals) == 1
    condition = conditionals[0]
    assert not condition.domain and len(condition.input) == len(condition.output) == 1
    _port(model.graph.input[1], np.bool_, ())
    selector_name = _origin(model.graph.node, condition.input[0])
    producers = {name: node for node in model.graph.node for name in node.output}
    casts = []
    while selector_name != model.graph.input[1].name:
        selector_node = producers[selector_name]
        assert not selector_node.domain and selector_node.op_type == "Cast"
        assert len(selector_node.input) == len(selector_node.output) == 1
        casts.append(next(a.i for a in selector_node.attribute if a.name == "to"))
        assert len(casts) <= 2
        selector_name = _origin(model.graph.node, selector_node.input[0])
    assert casts in (
        [],
        [onnx.TensorProto.BOOL],
        [onnx.TensorProto.BOOL, onnx.TensorProto.INT32],
    )
    assert _origin(model.graph.node, model.graph.output[0].name) == condition.output[0]
    branches = {
        a.name: a.g for a in condition.attribute if a.type == onnx.AttributeProto.GRAPH
    }
    assert set(branches) == {"then_branch", "else_branch"}
    for name, branch in branches.items():
        assert not branch.input and len(branch.output) == 1
        calls = [
            node for node in branch.node if (node.domain, node.op_type) in definitions
        ]
        assert len(calls) == 1
        call = calls[0]
        assert (call.domain, call.op_type) == (function.domain, function.name)
        assert len(call.input) == len(call.output) == 1
        assert (
            _origin(model.graph.node, _origin(branch.node, call.input[0]))
            == model.graph.input[0].name
        )
        output_name = _origin(branch.node, branch.output[0].name)
        negations = [node for node in branch.node if node.op_type == "Neg"]
        if name == "then_branch":
            assert not negations and output_name == call.output[0]
        else:
            assert len(negations) == 1
            (negation,) = negations
            assert (
                not negation.domain and len(negation.input) == len(negation.output) == 1
            )
            assert _origin(branch.node, negation.input[0]) == call.output[0]
            assert output_name == negation.output[0]
    payloads = {_payload(value) for value in _serialized_arrays(model)}
    assert set(frozen) <= payloads
    evidence_path = tmp_path / "named-execution-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["actual-exported-body-runtime"] = "VALIDATED"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    assert _globals() == configuration
    assert tuple(getattr(jax.config, key) for key in configuration) == effective
    assert (jnp.matmul, nnx.Linear.__call__, _named_linear) == original_calls
    assert _LINEAR is linear
    for index, (values, reference) in enumerate(zip(feeds, references, strict=True)):
        outputs = []
        for runner in (target, jax.jit(_fresh(target))):
            result = runner(*values)
            assert jax.tree.structure(result) == jax.tree.structure(reference[0])
            outputs.append(
                tuple(np.asarray(value).copy() for value in jax.tree.leaves(result))
            )
        np.savez(
            tmp_path / f"restored-whole-carrier-{index}.npz",
            **{f"input-{i}": np.asarray(value) for i, value in enumerate(values)},
            **{f"eager-{i}": value for i, value in enumerate(outputs[0])},
            **{f"jit-{i}": value for i, value in enumerate(outputs[1])},
        )
        for actual, expected in zip(outputs, reference, strict=True):
            assert len(actual) == len(expected)
            for actual_value, expected_value in zip(actual, expected, strict=True):
                _same(actual_value, expected_value)
    evidence["restored-whole-carrier-runtime"] = "VALIDATED"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    _run(model, feeds, references, tmp_path)
    assert (_payload(kernel), _payload(bias)) == frozen
    assert (id(linear.kernel), id(linear.bias)) == variables
    assert _globals() == configuration
    assert tuple(getattr(jax.config, key) for key in configuration) == effective
    assert (jnp.matmul, nnx.Linear.__call__, _named_linear) == original_calls
    assert typed_trace_manifest(native) == native_manifest
    assert tuple(_payload(value) for value in native.consts) == native_constants


def test_rank_two_fp64_mantissa_and_dyadic_cancellation_share_one_artifact(
    tmp_path: Path,
) -> None:
    kernel = _kernel(np.float64)
    fn = _target(None, "dot", np.float64)
    feeds = [
        (np.asarray([[2**24, 1, 0, 0], [2**24 + 2, 1, 0, 0]], np.float64), kernel),
        (
            np.asarray(
                [[2**24, 1, -(2**24), 0.25], [2**24 + 2, 1, -(2**24), 0.25]], np.float64
            ),
            kernel,
        ),
    ]
    specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in feeds[0]]
    native, references = _native_and_preserved(fn, specs, feeds, tmp_path)
    _same(references[0][0][0][:, 0], np.asarray([2**24 + 1, 2**24 + 3], np.float64))
    _same(references[1][0][0][:, 0], np.asarray([1.25, 3.25], np.float64))
    model = _artifact(fn, specs, native, tmp_path)
    _direct_structure(model, np.float64, False, kernel)
    _run(model, feeds, references, tmp_path)


@pytest.mark.parametrize(
    "dtype,weak",
    [(np.float32, False), (np.float32, True), (np.float64, False), (np.float64, True)],
    ids=["f32-strong", "f32-weak", "f64-strong", "f64-weak"],
)
def test_tensor_matrix_retains_actual_native_operand_and_result_weakness(
    dtype: Any, weak: bool
) -> None:
    (batch,) = export.symbolic_shape("B")
    specs = [
        jax.ShapeDtypeStruct((batch, 2, 3, 4), dtype, weak_type=weak),
        jax.ShapeDtypeStruct((4, 3), dtype, weak_type=weak),
    ]
    fn = _target(None, "dot", None)
    native = jax.make_jaxpr(fn)(*specs)
    validate_source_jaxpr(native, 23)
    dot = next(eqn for eqn in _body(native).eqns if eqn.primitive.name == "dot_general")
    assert [bool(var.aval.weak_type) for var in (*dot.invars, *dot.outvars)] == [
        weak
    ] * 3
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(_fresh(fn))(*specs)
    assert typed_trace_manifest(activated) == typed_trace_manifest(native)


def _graph(symbolic: bool = False) -> Any:
    leading: Any = export.symbolic_shape("B")[0] if symbolic else 2
    native = jax.make_jaxpr(
        lambda a, b: jax.lax.dot_general(a, b, (((2,), (0,)), ((), ())))
    )(
        jax.ShapeDtypeStruct((leading, 3, 4), np.float32),
        jax.ShapeDtypeStruct((4, 2), np.float32),
    ).jaxpr.eqns[
        0
    ]
    inputs = [SimpleNamespace(aval=v.aval) for v in native.invars]
    outputs = [SimpleNamespace(aval=v.aval) for v in native.outvars]
    equation = SimpleNamespace(
        primitive=native.primitive,
        invars=inputs,
        outvars=outputs,
        params=dict(native.params),
        effects=native.effects,
        ctx=native.ctx,
    )
    return SimpleNamespace(
        invars=inputs, constvars=(), outvars=outputs, eqns=[equation], effects=()
    )


_EXCLUSIONS = {
    "rank-five",
    "rhs-rank",
    "noncanonical",
    "paired-axis",
    "preferred",
    "high",
    "algorithm",
    "placement",
    "unknown",
    "foreign-scope",
    "string-batch",
    "batch-expression",
    "zero",
}
_DEFECTS = tuple(
    sorted(
        _EXCLUSIONS
        | {
            "rhs-axis-range",
            "bool-axis",
            "float-axis",
            "K",
            "output-shape",
            "output-dtype",
            "output-weak",
        }
    )
)


@pytest.mark.parametrize("defect", _DEFECTS)
def test_tensor_matrix_rejects_unsupported_or_corrupt_native_contract_before_emission(
    defect: str,
    tmp_path: Path,
) -> None:
    graph = _graph(defect in {"foreign-scope", "string-batch", "batch-expression"})
    eqn = graph.eqns[0]
    lhs, rhs = eqn.invars
    output = eqn.outvars[0]

    def aval(
        variable: Any, shape: Any = None, dtype: Any = np.float32, weak: bool = False
    ) -> None:
        shape = variable.aval.shape if shape is None else shape
        if any(isinstance(dim, str) for dim in shape):
            from tests.extra_tests.converter.test_source_native_contract import (
                _forged_aval,
            )

            variable.aval = _forged_aval(
                variable.aval, shape=shape, dtype=np.dtype(dtype), weak_type=weak
            )
        else:
            variable.aval = variable.aval.update(
                shape=shape, dtype=np.dtype(dtype), weak_type=weak, sharding=None
            )

    if defect == "rank-five":
        aval(lhs, (2, 2, 2, 3, 4))
        aval(output, (2, 2, 2, 3, 2))
        eqn.params["dimension_numbers"] = (((4,), (0,)), ((), ()))
    elif defect == "rhs-rank":
        aval(rhs, (1, 4, 2))
    elif defect == "zero":
        aval(lhs, (2, 0, 4))
        aval(output, (2, 0, 2))
    elif defect in {
        "rhs-axis-range",
        "bool-axis",
        "float-axis",
        "noncanonical",
        "paired-axis",
    }:
        eqn.params["dimension_numbers"] = {
            "rhs-axis-range": (((2,), (2,)), ((), ())),
            "bool-axis": (((2,), (False,)), ((), ())),
            "float-axis": (((2.0,), (0,)), ((), ())),
            "noncanonical": (((1,), (0,)), ((), ())),
            "paired-axis": (((2,), (0,)), ((0,), (1,))),
        }[defect]
        if defect == "noncanonical":
            aval(rhs, (3, 2))
            aval(output, (2, 4, 2))
        elif defect == "paired-axis":
            aval(rhs, (4, 2))
            aval(output, (2, 3))
    elif defect == "K":
        aval(rhs, (5, 2))
    elif defect.startswith("output-"):
        aval(
            output,
            (2, 3, 7) if defect == "output-shape" else None,
            np.float64 if defect == "output-dtype" else np.float32,
            defect == "output-weak",
        )
    elif defect == "preferred":
        eqn.params["preferred_element_type"] = np.float64
        aval(output, dtype=np.float64)
    elif defect == "high":
        eqn.params["precision"] = (jax.lax.Precision.HIGHEST,) * 2
    elif defect == "algorithm":
        eqn.params["precision"] = jax.lax.DotAlgorithmPreset.F32_F32_F32
    elif defect in {"placement", "unknown"}:
        eqn.params["out_sharding" if defect == "placement" else "out_dtype"] = object()
    elif defect == "foreign-scope":
        (foreign,) = export.symbolic_shape("B")
        aval(output, (foreign, 3, 2))
    elif defect == "string-batch":
        aval(lhs, ("B", 3, 4))
        aval(output, ("B", 3, 2))
    else:
        batch = lhs.aval.shape[0]
        aval(lhs, (batch + 1, 3, 4))
        aval(output, (batch + 1, 3, 2))
    if defect in {"rank-five", "zero", "noncanonical", "paired-axis"}:
        from tests.extra_tests.converter.test_source_generalized_contraction import (
            _oracle,
            _payload,
            observe_contraction,
        )

        dimensions = eqn.params["dimension_numbers"]
        values = tuple(
            _payload(np.float32, tuple(v.aval.shape), right=bool(i))
            for i, v in enumerate(eqn.invars)
        )

        def fn(a, b) -> Any:
            return jax.lax.dot_general(a, b, dimensions)

        validate_source_jaxpr(graph, 23)
        observe_contraction(
            fn, values, tmp_path, expected=(_oracle(*values, dimensions, np.float32),)
        )
        return
    before = typed_trace_manifest(graph)
    with pytest.raises(SourcePrecisionError) as captured:
        validate_source_jaxpr(graph, 23)
    assert captured.value.category == (
        "profile-exclusion"
        if defect in _EXCLUSIONS - {"rhs-rank"}
        else "lowering-defect"
    )
    boundary = defect in {"string-batch", "batch-expression"}
    assert captured.value.location == ("input 0" if boundary else "equation 0")
    assert captured.value.primitive == (None if boundary else "dot_general")
    assert captured.value.phase == "profile-validation" and captured.value.opset == 23
    assert captured.value.requirement and captured.value.action

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "invalid tensor×matrix contract reached physical allocation or emission"
        )

    context = SimpleNamespace(
        require_value_for_var=forbidden,
        allocate_value_for_var=forbidden,
        bind_value_for_var=forbidden,
        builder=SimpleNamespace(opset=23, op=forbidden),
    )
    expected = (
        NotImplementedError
        if defect
        in {"preferred", "high", "algorithm", "placement", "unknown", "output-dtype"}
        else ValueError
    )
    with pytest.raises(expected):
        DotGeneralPlugin()._lower_source(context, eqn)
    assert typed_trace_manifest(graph) == before


@pytest.mark.parametrize(
    "port,defect",
    [
        (port, defect)
        for port in ("lhs", "rhs", "output")
        for defect in ("dtype", "shape", "unknown")
    ],
    ids=[
        f"{port}-{defect}"
        for port in ("lhs", "rhs", "output")
        for defect in ("dtype", "shape", "unknown")
    ],
)
def test_tensor_matrix_checks_physical_metadata_before_emitting_or_mutating(
    port: str, defect: str
) -> None:
    graph = _graph(True)
    equation = graph.eqns[0]
    values = {
        name: ir.Value(type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(shape))
        for name, shape in (
            ("lhs", ("B", 3, 4)),
            ("rhs", (4, 2)),
            ("output", ("B", 3, 2)),
        )
    }
    target = values[port]
    if defect == "dtype":
        target.type = ir.TensorType(ir.DataType.DOUBLE)
    else:
        shape = list(target.shape)
        shape[-1] = 7 if defect == "shape" else None
        target.shape = ir.Shape(shape)
    before = tuple(
        (id(value), value.dtype, tuple(value.shape)) for value in values.values()
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "invalid physical tensor×matrix metadata reached emission or binding"
        )

    context = SimpleNamespace(
        require_value_for_var=lambda var: values[
            "lhs" if var is equation.invars[0] else "rhs"
        ],
        allocate_value_for_var=lambda *args, **kwargs: values["output"],
        bind_value_for_var=forbidden,
        builder=SimpleNamespace(opset=23, op=forbidden),
    )
    with pytest.raises(ValueError):
        DotGeneralPlugin()._lower_source(context, equation)
    assert (
        tuple((id(value), value.dtype, tuple(value.shape)) for value in values.values())
        == before
    )
