# tests/extra_tests/converter/test_source_original_gpt_mlp_numerics.py

"""Finite original GPT_MLP FP32 evidence, separate from its artifact-only ledger.

Reviewed plan SHA256: 3b5ce15b10f279cdd784bcbe796550079356127aa564e423f9e5e8af5c55c627.
Two caller settings cover four pairwise feeds, never the eight-row Cartesian set.
The unchanged generated FP32 defaults are the new criterion, including under x64.
"""

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from flax import nnx
import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._compat.jax import Literal
from jax2onnx.converter import conversion_api
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
)
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.plugins._source_function_state import FunctionOperandLayout
from jax2onnx.plugins.plugin_system import FunctionPlugin
from jax2onnx.plugins.flax.nnx._source_function_state import NNXFunctionState
from jax2onnx.precision import (
    CheckStatus,
    get_precision_report,
    precision_report_to_json,
)
from scripts.source_precision_inventory import canonical_contract, raw_registrations
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.source_precision_support import (
    SourceSettings,
    _global_source_configuration,
)
from tests.extra_tests.converter.test_source_mlp_runtime_acceptance import (
    _callables,
    _configuration,
    _port,
)
from tests.extra_tests.converter.test_source_nnx_function_state import (
    _observe_functions,
)
from tests.extra_tests.converter.test_source_tensor_matrix import _dot_contracts

_ORIGIN = "example::examples.gpt::GPT_MLP"
_CONTRACT = "faf258485b7a1bf4a97e1a56b87e9fb472ad8eba84b973b8e303dd925c69e21a"
_PLAN = "3b5ce15b10f279cdd784bcbe796550079356127aa564e423f9e5e8af5c55c627"
_RTOL = _ATOL = 1e-5
_MAX_BYTES = 150_000_000
_STATE_PATHS = (
    "c_fc/bias",
    "c_fc/kernel",
    "c_proj/bias",
    "c_proj/kernel",
    "dropout/rngs/count",
    "dropout/rngs/key",
)
_PARAM_SHAPES = ((3072,), (768, 3072), (768,), (3072, 768))
_EXPECTED_COUNTS = dict(
    constructors=1,
    native_traces=1,
    pre_native=4,
    restored_native=2,
    exports=1,
    artifacts=1,
    callbacks=1,
    sessions=2,
    runs=4,
    feeds=2,
)


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _array(value: Any) -> dict[str, Any]:
    value = np.asarray(value)
    return dict(
        dtype=value.dtype.str, shape=list(value.shape), sha256=_sha(value.tobytes())
    )


def _used_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _budget(root: Path) -> None:
    assert _used_bytes(root) <= _MAX_BYTES, "GPT_MLP evidence storage ceiling exceeded"


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    _budget(path.parents[1] if path.parent.name.startswith("caller-") else path.parent)


def _save(root: Path, path: Path, **arrays: Any) -> dict[str, Any]:
    # One sequential compressed write; every expanded array retains a content hash.
    np.savez_compressed(path, **arrays)
    _budget(root)
    return dict(
        file=str(path.relative_to(root)),
        sha256=_sha(path.read_bytes()),
        arrays={name: _array(value) for name, value in arrays.items()},
    )


def _contract(case: Any) -> str:
    unresolved: list[str] = []
    contract = canonical_contract(
        {
            key: value
            for key, value in case.items()
            if key not in {"testcase", "description"}
        },
        unresolved=unresolved,
    )
    assert not unresolved
    return _sha(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())


def _state(owner: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    state = nnx.state(owner)
    arrays, entries = {}, []
    for path, variable in nnx.to_flat_state(state):
        leaves, tree = jax.tree.flatten(variable)
        assert len(leaves) == 1
        value = leaves[0]
        key = jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
        physical = np.asarray(jax.random.key_data(value) if key else value)
        name = "/".join(map(str, path))
        arrays[name] = physical
        entries.append(
            dict(
                path=name,
                variable_type=f"{type(variable).__module__}.{type(variable).__qualname__}",
                tree=str(tree),
                logical_dtype=str(value.dtype),
                logical_shape=list(value.shape),
                weak_type=bool(jax.typeof(value).weak_type),
                key_implementation=str(jax.random.key_impl(value)) if key else None,
                physical=_array(physical),
            )
        )
    assert tuple(arrays) == _STATE_PATHS and len(entries) == 6
    for path, shape in zip(_STATE_PATHS[:4], _PARAM_SHAPES, strict=True):
        assert arrays[path].dtype == np.float32 and arrays[path].shape == shape
    assert arrays[_STATE_PATHS[4]].dtype == arrays[_STATE_PATHS[5]].dtype == np.uint32
    assert arrays[_STATE_PATHS[4]].shape == () and arrays[_STATE_PATHS[5]].shape == (2,)
    assert int(arrays[_STATE_PATHS[4]]) == 0
    assert sum(arrays[path].nbytes for path in _STATE_PATHS[:4]) == 18_889_728
    _, tree = jax.tree.flatten(state)
    return dict(entries=entries, tree=str(tree)), arrays


def _identities(owner: Any) -> list[Any]:
    return [
        (list(path), f"{type(value).__module__}.{type(value).__qualname__}", id(value))
        for path, value in nnx.iter_graph(owner)
        if isinstance(value, (nnx.Module, nnx.Variable))
    ]


def _unchanged(
    owner: Any, baseline: Any, identities: Any, graphdef: Any
) -> dict[str, Any]:
    actual, _ = _state(owner)
    assert actual == baseline
    assert _identities(owner) == identities
    assert nnx.split(owner)[0] == graphdef
    return actual


def _input(batch: int) -> np.ndarray:
    b = np.arange(batch, dtype=np.int32)[:, None, None]
    t = np.arange(1024, dtype=np.int32)[None, :, None]
    c = np.arange(768, dtype=np.int32)[None, None, :]
    q = ((17 * c + 29 * (t % 8) + 43 * b) % 257) - 128
    scale = np.float32(0.25 if batch == 2 else 2)
    x = scale * (
        q.astype(np.float32) / np.float32(128)
        + ((c % 3) - 1).astype(np.float32) / np.float32(32)
    )
    assert x.shape == (batch, 1024, 768) and x.dtype == np.float32
    assert np.isfinite(x).all() and np.any(x) and np.any(x < 0) and np.any(x > 0)
    return x


def _output(value: Any, batch: int) -> tuple[np.ndarray, str]:
    leaves, tree = jax.tree.flatten(jax.device_get(value))
    assert len(leaves) == 1 and str(tree) == "PyTreeDef(*)"
    output = np.array(leaves[0], copy=True)
    return output, str(tree)


def _require_output(output: np.ndarray, batch: int) -> None:
    assert output.dtype == np.float32 and output.shape == (batch, 1024, 768)
    assert np.isfinite(output).all() and np.any(output) and np.ptp(output) > 0


def _difference(actual: Any, reference: Any) -> dict[str, Any]:
    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    return dict(
        max_abs=float(difference.max()),
        different_elements=int(np.count_nonzero(difference)),
        total_elements=int(actual.size),
        actual=_array(actual),
        reference=_array(reference),
        within_fixed_criterion=bool(
            np.allclose(actual, reference, rtol=_RTOL, atol=_ATOL, equal_nan=False)
        ),
    )


def _compare(actual: Any, reference: Any) -> dict[str, Any]:
    assert actual.dtype == reference.dtype == np.float32
    assert actual.shape == reference.shape
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    np.testing.assert_allclose(
        actual, reference, rtol=_RTOL, atol=_ATOL, equal_nan=False
    )
    return _difference(actual, reference)


def _fresh_jit() -> Any:
    # The retained original owner is a graph argument; BOOL stays a scalar argument.
    def original(module: Any, x: Any, deterministic: Any) -> Any:
        return module(x, deterministic=deterministic)

    return nnx.jit(original)


def _primary(closed: Any, roles: dict[int, str]) -> dict[str, Any]:
    """Project only observed primary def-use; discard only proved state forwards."""
    body = closed.jaxpr
    needed = {id(body.outvars[0])}
    equations = []
    for equation in reversed(body.eqns):
        if any(id(var) in needed for var in equation.outvars):
            equations.append(equation)
            needed.update(
                id(var) for var in equation.invars if not isinstance(var, Literal)
            )
    variables = [var for var in (*body.constvars, *body.invars) if id(var) in needed]
    assert all(id(var) in roles for var in variables)
    variables.sort(key=lambda var: roles[id(var)])
    projected = SimpleNamespace(
        jaxpr=SimpleNamespace(
            constvars=(),
            invars=variables,
            outvars=[body.outvars[0]],
            eqns=list(reversed(equations)),
            effects=body.effects,
        ),
        consts=(),
    )
    result = typed_trace_manifest(projected)
    result["input_roles"] = [roles[id(var)] for var in variables]
    assert not result["effects"] and not result["unresolved_parameters"]
    return result


def _source_float_constants(closed: Any) -> list[Any]:
    result = []
    for equation in closed.jaxpr.eqns:
        for variable in (*equation.invars, *equation.outvars):
            if jax.dtypes.issubdtype(variable.aval.dtype, np.floating):
                assert np.dtype(variable.aval.dtype) == np.float32
        for variable in equation.invars:
            if isinstance(variable, Literal) and np.issubdtype(
                variable.aval.dtype, np.floating
            ):
                result.append(_array(np.asarray(variable.val)))
        if equation.primitive.name == "integer_pow":
            # The native exponent is static metadata, not an emitted float
            # coefficient in the reviewed same-width cubic multiplication DAG.
            assert equation.params == {"y": 3}
        for name in ("jaxpr", "call_jaxpr"):
            if name in equation.params:
                nested = equation.params[name]
                result.extend(
                    _source_float_constants(
                        SimpleNamespace(jaxpr=getattr(nested, "jaxpr", nested))
                    )
                )
    return result


def _native_roles(closed: Any, arrays: Any) -> dict[int, str]:
    assert len(closed.jaxpr.invars) == 2 and len(closed.jaxpr.outvars) == 1
    roles = {
        id(closed.jaxpr.invars[0]): "x",
        id(closed.jaxpr.invars[1]): "deterministic",
    }
    for variable, value in zip(closed.jaxpr.constvars, closed.consts, strict=True):
        matches = [
            path for path in _STATE_PATHS[:4] if _array(value) == _array(arrays[path])
        ]
        assert (
            len(matches) == 1
        ), "Native constant must bind exact original parameter path and bytes"
        roles[id(variable)] = matches[0]
    assert len(closed.consts) == 4
    return roles


def _constant(model: Any, name: str) -> Any:
    initializers = {value.name: value for value in model.graph.initializer}
    if name in initializers:
        return initializers[name]
    nodes = {output: node for node in model.graph.node for output in node.output}
    node = nodes[name]
    assert node.op_type == "Constant" and not node.domain
    return next(
        attribute.t for attribute in node.attribute if attribute.name == "value"
    )


def _parameters(model: Any, call: Any, paths: list[str], arrays: Any) -> list[Any]:
    bindings = []
    for index, path in enumerate(paths, 1):
        tensor = _constant(model, call.input[index])
        actual = onnx.numpy_helper.to_array(tensor)
        assert _array(actual) == _array(
            arrays[path]
        ), "Consumed original state bytes changed"
        bindings.append(
            dict(
                path=path,
                call_operand=index,
                actual_name=call.input[index],
                physical=_array(actual),
            )
        )
    return bindings


def _serialized(
    model: Any, built: Any, trace: Any, native: Any, arrays: Any
) -> dict[str, Any]:
    assert len(model.functions) == len(built) == 1
    definition, body = built[0]
    function = model.functions[0]
    assert (function.domain, function.name) == (definition.domain, definition.name)
    assert function.name == "MLP" and function.domain.startswith("custom.MLP.")
    calls = [
        node
        for node in model.graph.node
        if (node.domain, node.op_type) == (function.domain, function.name)
    ]
    assert len(calls) == 1
    call = calls[0]
    assert len(function.input) == len(call.input) == 8
    assert len(function.output) == len(call.output) == 7
    assert [value.name for value in model.graph.input] == ["in_0", "deterministic"]
    assert len(model.graph.output) == 1 and model.graph.output[0].name == call.output[0]
    assert call.input[0] == "in_0" and call.input[-1] == "deterministic"
    _port(model.graph.input[0], np.float32, ("B", 1024, 768))
    _port(model.graph.output[0], np.float32, ("B", 1024, 768))
    _port(model.graph.input[1], np.bool_, ())
    assert len(trace.jaxpr.eqns) == 1
    equation = trace.jaxpr.eqns[0]
    layout = next(
        value
        for value in equation.params.values()
        if isinstance(value, FunctionOperandLayout)
    )
    adapter = next(
        value
        for value in equation.params.values()
        if isinstance(value, NNXFunctionState)
    )
    assert (layout.positional_count, layout.argument_count, layout.state_count) == (
        1,
        2,
        6,
    )
    assert adapter.effects == () and len(adapter.entries) == 6
    paths = ["/".join(map(str, entry.path)) for entry in adapter.entries]
    assert tuple(paths) == _STATE_PATHS
    assert [(entry.start, entry.stop) for entry in adapter.entries] == [
        (i, i + 1) for i in range(6)
    ]
    assert len(trace.source_inputs) == 2 and not trace.frozen_params
    for binding in trace.source_inputs:
        assert binding.var is trace.jaxpr.invars[binding.trace_index]
    assert [binding.keyword_name for binding in trace.source_inputs] == [
        None,
        "deterministic",
    ]
    assert equation.invars[0] is trace.jaxpr.invars[0]
    assert equation.invars[1] is trace.jaxpr.invars[1]
    constants = {
        id(var): value
        for var, value in zip(
            trace.jaxpr.constvars, trace.closed_jaxpr.consts, strict=True
        )
    }
    for variable, path in zip(equation.invars[2:], paths, strict=True):
        value = constants[id(variable)]
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jax.random.key_data(value)
        assert _array(value) == _array(arrays[path])
    assert len(body.jaxpr.invars) == 8 and len(body.jaxpr.outvars) == 7
    assert not body.jaxpr.constvars
    roles = {id(body.jaxpr.invars[0]): "x", id(body.jaxpr.invars[-1]): "deterministic"}
    roles.update(
        (id(var), path) for var, path in zip(body.jaxpr.invars[1:7], paths, strict=True)
    )
    for incoming, outgoing in zip(
        body.jaxpr.invars[1:7], body.jaxpr.outvars[1:], strict=True
    ):
        assert (
            incoming is outgoing
        ), "Only exact unchanged-state forwarding ports may be projected away"
    primary = _primary(body, roles)
    assert primary == native
    dots = _dot_contracts(body)
    assert len(dots) == 2
    for inputs, outputs, axes, precision, preferred, placement in dots:
        assert inputs[0][0] == inputs[1][0] == outputs[0][0] == "float32"
        assert axes == repr((((2,), (0,)), ((), ())))
        assert precision in {"None", repr((jax.lax.Precision.DEFAULT,) * 2)}
        assert preferred in {"None", "float32"} and placement == "None"
    infos = {value.name: value for value in function.value_info}
    ports = []
    for names, variables in (
        (function.input, body.jaxpr.invars),
        (function.output, body.jaxpr.outvars),
    ):
        for name, variable in zip(names, variables, strict=True):
            representation = representation_for_aval(variable.aval)
            shape = tuple(
                int(dim) if isinstance(dim, (int, np.integer)) else str(dim)
                for dim in representation.physical_shape
            )
            _port(infos[name], representation.physical_dtype, shape)
            ports.append(
                dict(
                    name=name,
                    logical_dtype=str(variable.aval.dtype),
                    logical_shape=list(map(str, variable.aval.shape)),
                    weak_type=bool(variable.aval.weak_type),
                    encoding=representation.encoding,
                    physical_dtype=representation.physical_dtype.name,
                    physical_shape=list(shape),
                )
            )
    operations = Counter(node.op_type for node in function.node)
    assert operations == Counter(
        dict(Add=4, Constant=6, Identity=7, MatMul=2, Mul=6, Reshape=2, Tanh=1)
    )
    all_nodes = [*model.graph.node, *function.node]
    assert not any(node.op_type == "Cast" for node in all_nodes)
    assert not any(
        value.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
        for value in [*model.graph.value_info, *function.value_info]
    )
    assert not any(
        value.data_type == onnx.TensorProto.DOUBLE for value in model.graph.initializer
    )
    assert not any(
        attribute.type == onnx.AttributeProto.TENSOR
        and attribute.t.data_type == onnx.TensorProto.DOUBLE
        for node in all_nodes
        for attribute in node.attribute
    )
    producers = {output: node for node in function.node for output in node.output}
    needed = {function.output[0]}
    for node in reversed(function.node):
        if needed.intersection(node.output):
            needed.update(node.input)
    source_constants = _source_float_constants(body)
    serialized_constants = []
    for node in function.node:
        if node.op_type == "Constant":
            tensor = next(
                attribute.t for attribute in node.attribute if attribute.name == "value"
            )
            value = onnx.numpy_helper.to_array(tensor)
            if np.issubdtype(value.dtype, np.floating):
                assert value.dtype == np.float32 and node.output[0] in needed
                serialized_constants.append(
                    dict(
                        value=_array(value),
                        consumers=[
                            dict(op=consumer.op_type, input_index=index)
                            for consumer in function.node
                            for index, name in enumerate(consumer.input)
                            if name == node.output[0]
                        ],
                    )
                )
    assert Counter(
        json.dumps(value, sort_keys=True) for value in source_constants
    ) == Counter(
        json.dumps(value["value"], sort_keys=True) for value in serialized_constants
    )
    assert set(function.input[1:5]) <= needed
    assert function.input[-1] not in needed and not set(function.input[5:7]) & needed
    matmuls = [node for node in function.node if node.op_type == "MatMul"]
    assert [node.input[1] for node in matmuls] == [function.input[2], function.input[4]]
    powers = [eqn for eqn in body.jaxpr.eqns if eqn.primitive.name == "integer_pow"]
    assert len(powers) == 1 and powers[0].params == {"y": 3}
    # Follow the actual activation from first Linear through exactly (x*x)*x,
    # then its original consumed GELU coefficients, Tanh and second Linear.
    activation = next(
        node
        for node in function.node
        if node.op_type == "Add" and node.input[0] == matmuls[0].output[0]
    )
    bias = producers[activation.input[1]]
    assert bias.op_type == "Reshape" and bias.input[0] == function.input[1]
    value = activation.output[0]
    square = next(
        node
        for node in function.node
        if node.op_type == "Mul" and list(node.input) == [value, value]
    )
    cube = next(
        node
        for node in function.node
        if node.op_type == "Mul" and node.input[0] == square.output[0]
    )
    assert list(cube.input) == [
        square.output[0],
        value,
    ], "Cubic source association changed"
    forwarded = next(
        node
        for node in function.node
        if node.op_type == "Identity" and list(node.input) == [cube.output[0]]
    )
    for name in (value, square.output[0], cube.output[0], forwarded.output[0]):
        _port(infos[name], np.float32, ("B", 1024, 3072))
    weighted = next(
        node
        for node in function.node
        if node.op_type == "Mul" and node.input[1] == forwarded.output[0]
    )
    summed = next(
        node
        for node in function.node
        if node.op_type == "Add" and list(node.input) == [value, weighted.output[0]]
    )
    scaled = next(
        node
        for node in function.node
        if node.op_type == "Mul" and node.input[1] == summed.output[0]
    )
    tangent = next(node for node in function.node if node.op_type == "Tanh")
    assert list(tangent.input) == [scaled.output[0]]
    shifted = next(
        node
        for node in function.node
        if node.op_type == "Add" and node.input[1] == tangent.output[0]
    )
    half = next(
        node
        for node in function.node
        if node.op_type == "Mul" and node.input[1] == shifted.output[0]
    )
    selected = producers[matmuls[1].input[0]]
    assert selected.op_type == "Mul" and list(selected.input) == [value, half.output[0]]
    coefficient_bindings = []
    for consumer, coefficient in (
        (weighted, 0.044715),
        (scaled, 0.7978845608028654),
        (shifted, 1),
        (half, 0.5),
    ):
        constant = producers[consumer.input[0]]
        assert constant.op_type == "Constant" and not constant.domain
        tensor = next(attr.t for attr in constant.attribute if attr.name == "value")
        actual = onnx.numpy_helper.to_array(tensor)
        assert _array(actual) == _array(np.asarray(coefficient, np.float32))
        coefficient_bindings.append(dict(consumer=consumer.name, value=_array(actual)))
    for index in range(6):
        forward = producers[function.output[index + 1]]
        assert forward.op_type == "Identity" and list(forward.input) == [
            function.input[index + 1]
        ]
    return dict(
        primary=primary,
        dots=dots,
        ports=ports,
        state_paths=paths,
        parameters=_parameters(model, call, paths, arrays),
        op_counts=dict(operations),
        actual_function=dict(
            domain=function.domain,
            name=function.name,
            inputs=list(function.input),
            outputs=list(function.output),
        ),
        actual_call=dict(inputs=list(call.input), outputs=list(call.output)),
        primary_dependencies=sorted(needed),
        arithmetic_constants=serialized_constants,
        cast_count=0,
        hidden_fp64=False,
        integer_pow3=dict(
            native_exponent=powers[0].params["y"],
            exponent_is_static_metadata_not_serialized_float=True,
            activation=value,
            square_node=square.name,
            cube_node=cube.name,
            output_identity=forwarded.name,
            association="(x*x)*x",
            dtype="float32",
            shape=["B", 1024, 3072],
            consumed_coefficients=coefficient_bindings,
            consumed_by_second_linear=True,
        ),
    )


@contextmanager
def _settings(x64: bool) -> Any:
    original, callables, policy = (
        _configuration(),
        _callables(),
        active_precision_policy(),
    )
    default_device = jax.config.jax_default_device
    globals_before = _global_source_configuration()
    device_global = jax.default_device.get_global()
    key_reuse_global = jax.debug_key_reuse.get_global()
    assert policy is PrecisionPolicy.LEGACY
    try:
        with (
            jax.enable_x64(x64),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
            jax.default_device(jax.devices("cpu")[0]),
        ):
            selected = _configuration()
            yield
            assert _configuration() == selected and _callables() == callables
            assert active_precision_policy() is policy
    finally:
        for name, value in globals_before.items():
            jax.config.update(name, value)
        jax.config.update("jax_default_device", device_global)
        jax.config.update("jax_debug_key_reuse", key_reuse_global)
        assert _configuration() == original and _callables() == callables
        assert jax.config.jax_default_device is default_device
        assert active_precision_policy() is policy


@pytest.fixture(scope="module")
def _packet_root(tmp_path_factory: Any) -> Path:
    configured = os.environ.get("JAX2ONNX_ORIGINAL_GPT_MLP_EVIDENCE_DIR")
    base = tmp_path_factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        base = base.parent
    root = Path(configured) if configured else base / "original-gpt-mlp-numerics"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.mark.parametrize(
    "caller_x64", [False, True], ids=["caller-x64-false", "caller-x64-true"]
)
def test_original_gpt_mlp_finite_numerics(caller_x64: bool, _packet_root: Path) -> None:
    root = _packet_root
    # Serialize the two finite cells even in xdist; archive original state once/stack.
    with (root / "packet.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not (
            root / "failed.json"
        ).exists(), "Earlier GPT_MLP cell failed; stop this stack"
        folder = root / f"caller-{str(caller_x64).lower()}"
        folder.mkdir(exist_ok=True)
        assert not (folder / "receipt.json").exists(), "Use a fresh evidence directory"
        counts = dict.fromkeys(_EXPECTED_COUNTS, 0)
        record: dict[str, Any] = dict(
            plan_sha256=_PLAN,
            origin=_ORIGIN,
            ordinal=0,
            contract_sha256=_CONTRACT,
            caller_x64=caller_x64,
            criterion=dict(
                mode="tolerant",
                rtol=_RTOL,
                atol=_ATOL,
                equal_nan=False,
                provenance="new reviewed criterion: unchanged generated FP32 variant defaults",
                historical_raw_criterion=None,
            ),
            max_new_evidence_bytes_per_stack=_MAX_BYTES,
            counts=counts,
            expected_counts=_EXPECTED_COUNTS,
            versions=dict(
                python=platform.python_version(),
                jax=jax.__version__,
                numpy=np.__version__,
                ort=ort.__version__,
            ),
            status="PARTIAL",
            last_successful_stage="created",
            native={},
            runtime={},
            restored={},
            export_runtime_and_numerics="UNVERIFIED",
            public_support="UNVERIFIED",
        )

        def checkpoint(stage: str) -> None:
            record["last_successful_stage"] = stage
            record["stored_bytes_before_receipt"] = _used_bytes(root)
            _json(folder / "receipt.json", record)

        try:
            assert (
                shutil.disk_usage(root).free >= _MAX_BYTES
            ), "Insufficient free disk for the frozen per-stack evidence ceiling"
            registrations, _ = raw_registrations()
            case = next(data for name, _, data in registrations if name == _ORIGIN)[
                "testcases"
            ][0]
            assert _contract(case) == _CONTRACT and case["run_only_f32_variant"]
            assert case["input_shapes"] == [("B", 1024, 768)]
            assert case["input_params"] == {"deterministic": True}
            assert not any(key.startswith(("rtol", "atol")) for key in case)
            record["original_settings"] = dict(
                settings=asdict(SourceSettings.observe()),
                key_reuse=jax.config.jax_debug_key_reuse,
                default_device=str(jax.config.jax_default_device),
            )
            with _settings(caller_x64):
                record["effective_settings"] = dict(
                    settings=asdict(SourceSettings.observe()),
                    key_reuse=jax.config.jax_debug_key_reuse,
                    default_device=str(jax.config.jax_default_device),
                )
                owner = case["callable"].instantiate()
                counts["constructors"] += 1
                assert type(owner) is case["callable"]._ctor
                assert owner.dropout.rate == 0.0
                original_method = type(owner).__call__
                baseline, arrays = _state(owner)
                identities, graphdef = _identities(owner), nnx.split(owner)[0]
                record["state"] = baseline
                record["graph_variable_identities"] = identities
                canonical = root / "canonical-state.json"
                if canonical.exists():
                    assert json.loads(canonical.read_text())["state"] == baseline
                else:
                    archive = _save(
                        root,
                        root / "canonical-state.npz",
                        **{
                            f"leaf{i}": arrays[path]
                            for i, path in enumerate(_STATE_PATHS)
                        },
                    )
                    _json(
                        canonical,
                        dict(
                            state=baseline,
                            archive=archive,
                            physical_arrays_in_state_path_order=list(_STATE_PATHS),
                        ),
                    )
                checkpoint("original-owner-and-canonical-state")
                b = jax_export.symbolic_shape("B")[0]
                specs = [
                    jax.ShapeDtypeStruct((b, 1024, 768), np.float32),
                    jax.ShapeDtypeStruct((), np.bool_),
                ]
                _unchanged(owner, baseline, identities, graphdef)
                native = jax.make_jaxpr(lambda x, flag: owner(x, deterministic=flag))(
                    *specs
                )
                counts["native_traces"] += 1
                _json(
                    folder / "observed-native-trace.json", typed_trace_manifest(native)
                )
                _unchanged(owner, baseline, identities, graphdef)
                primary = _primary(native, _native_roles(native, arrays))
                _json(folder / "native-primary-dataflow.json", primary)
                record["native_dot_contracts"] = _dot_contracts(native)
                assert len(record["native_dot_contracts"]) == 2
                before_jit = _fresh_jit()
                for batch, flag in ((2, not caller_x64), (5, caller_x64)):
                    x = _input(batch)
                    feed_file = root / f"input-B{batch}.npz"
                    if feed_file.exists():
                        with np.load(feed_file, allow_pickle=False) as saved:
                            assert _array(saved["x"]) == _array(x)
                    else:
                        _save(root, feed_file, x=x)
                    counts["feeds"] += 1
                    values, control = jnp.asarray(x), jnp.asarray(
                        np.asarray(flag, np.bool_)
                    )
                    assert values.device.platform == control.device.platform == "cpu"
                    entry = dict(
                        batch=batch,
                        deterministic=flag,
                        input=_array(x),
                        output_tree="PyTreeDef(*)",
                        roles={},
                    )
                    record["native"][str(batch)] = entry
                    for role in ("eager", "jit"):
                        _unchanged(owner, baseline, identities, graphdef)
                        result = (
                            owner(values, deterministic=control)
                            if role == "eager"
                            else before_jit(owner, values, control)
                        )
                        counts["pre_native"] += 1
                        output, tree = _output(result, batch)
                        entry["roles"][role] = _save(
                            root, folder / f"native-B{batch}-{role}.npz", y=output
                        )
                        entry["output_tree"] = tree
                        _require_output(output, batch)
                        _unchanged(owner, baseline, identities, graphdef)
                        checkpoint(f"native-B{batch}-{role}")
                        del result, output
                    with (
                        np.load(
                            folder / f"native-B{batch}-eager.npz", allow_pickle=False
                        ) as eager,
                        np.load(
                            folder / f"native-B{batch}-jit.npz", allow_pickle=False
                        ) as compiled,
                    ):
                        entry["eager_vs_jit"] = _difference(compiled["y"], eager["y"])
                    del x, values, control
                traces = []
                real_trace = conversion_api._trace_to_jaxpr

                def observe(**kwargs: Any) -> Any:
                    traced = real_trace(**kwargs)
                    traces.append(traced)
                    _json(
                        folder / f"observed-source-trace-{len(traces)}.json",
                        typed_trace_manifest(traced.closed_jaxpr),
                    )
                    return traced

                _unchanged(owner, baseline, identities, graphdef)
                with (
                    patch.object(conversion_api, "_trace_to_jaxpr", observe),
                    _observe_functions() as built,
                ):
                    body_trace = FunctionPlugin._trace_function_body
                    observed_bodies: list[Any] = []

                    def retain_body(*args: Any, **kwargs: Any) -> Any:
                        body = body_trace(*args, **kwargs)
                        observed_bodies.append(body)
                        _json(
                            folder
                            / f"observed-function-body-{len(observed_bodies)}-returned.json",
                            typed_trace_manifest(body),
                        )
                        return body

                    with patch.object(
                        FunctionPlugin, "_trace_function_body", retain_body
                    ):
                        artifact = ui._to_onnx(
                            owner,
                            [jax.ShapeDtypeStruct(("B", 1024, 768), np.float32)],
                            input_params=dict(case["input_params"]),
                            precision_policy=PrecisionPolicy.SOURCE,
                            return_mode="proto",
                            opset=23,
                        )
                        counts["exports"] += 1
                _unchanged(owner, baseline, identities, graphdef)
                assert type(owner).__call__ is original_method
                assert conversion_api._trace_to_jaxpr is real_trace
                assert len(traces) == 1
                payload = artifact.SerializeToString()
                (folder / "model.onnx").write_bytes(payload)
                counts["artifacts"] += 1
                _budget(root)
                record["artifact"] = dict(bytes=len(payload), sha256=_sha(payload))
                report = get_precision_report(artifact)
                assert report is not None and report.profile == "source-static-v1"
                report_json = precision_report_to_json(report)
                (folder / "report.json").write_text(report_json)
                checks = {check.category: check.status for check in report.checks}
                assert (
                    checks["representation"]
                    is checks["lowering"]
                    is CheckStatus.VALIDATED
                )
                assert (
                    checks["runtime-execution"]
                    is checks["numerical-acceptance"]
                    is CheckStatus.UNVERIFIED
                )
                checkpoint("artifact-and-unchanged-export-report")
                onnx.checker.check_model(
                    onnx.load_model_from_string(payload), full_check=True
                )
                try:
                    raw_callback = case["post_check_onnx_graph"](artifact)
                    counts["callbacks"] += 1
                    record["raw_structural_callback"] = dict(
                        status="passed" if raw_callback else "failed",
                        actual_result=repr(raw_callback),
                    )
                except Exception as exc:
                    counts["callbacks"] += 1
                    record["raw_structural_callback"] = dict(
                        status="error", type=type(exc).__name__, message=str(exc)
                    )
                    raise
                checkpoint("original-structural-callback-observed")
                assert (
                    raw_callback
                ), "Original raw callback remains an open structural obligation"
                _json(
                    folder / "observed-function-bodies-before-assertions.json",
                    [
                        dict(
                            domain=definition.domain,
                            name=definition.name,
                            typed_native_body=typed_trace_manifest(body),
                        )
                        for definition, body in built
                    ],
                )
                evidence = _serialized(artifact, built, traces[0], primary, arrays)
                _json(folder / "serialized-and-actual-body-evidence.json", evidence)
                record["export_internal_abstract_tracing"] = dict(
                    top_level_traces=len(traces),
                    observed_function_bodies=len(built),
                    separate_from_explicit_native_evaluations=True,
                )
                if not caller_x64:
                    corrupted = onnx.load_model_from_string(payload)
                    function = corrupted.functions[0]
                    call = next(
                        node
                        for node in corrupted.graph.node
                        if (node.domain, node.op_type)
                        == (function.domain, function.name)
                    )
                    kernel = _constant(corrupted, call.input[2])
                    assert kernel.raw_data
                    damaged = bytearray(kernel.raw_data)
                    damaged[0] ^= 1
                    kernel.raw_data = bytes(damaged)
                    with pytest.raises(
                        AssertionError, match="Consumed original state bytes changed"
                    ):
                        _parameters(corrupted, call, list(_STATE_PATHS), arrays)
                    record["negative_consumed_kernel"] = dict(
                        status="rejected",
                        path="c_fc/kernel",
                        word=0,
                        xor_byte=1,
                        sha256=_sha(kernel.raw_data),
                    )
                    del corrupted, damaged, kernel
                    corrupted = onnx.load_model_from_string(payload)
                    function = corrupted.functions[0]
                    square = next(
                        node
                        for node in function.node
                        if node.name == evidence["integer_pow3"]["square_node"]
                    )
                    cube = next(
                        node
                        for node in function.node
                        if node.name == evidence["integer_pow3"]["cube_node"]
                    )
                    cube.input[1] = square.output[0]
                    with pytest.raises(
                        AssertionError, match="Cubic source association"
                    ):
                        _serialized(corrupted, built, traces[0], primary, arrays)
                    record["negative_cubic_dataflow"] = dict(
                        status="rejected",
                        mutation="(x*x)*(x*x)",
                        counts_and_dtypes_unchanged=True,
                    )
                    del corrupted, function, square, cube
                checkpoint("observed-source-and-serialized-provenance")
                after_jit = _fresh_jit()
                for role in ("eager", "jit"):
                    values, control = jnp.asarray(_input(2)), jnp.asarray(
                        np.asarray(not caller_x64, np.bool_)
                    )
                    _unchanged(owner, baseline, identities, graphdef)
                    result = (
                        owner(values, deterministic=control)
                        if role == "eager"
                        else after_jit(owner, values, control)
                    )
                    counts["restored_native"] += 1
                    output, _ = _output(result, 2)
                    with np.load(
                        folder / f"native-B2-{role}.npz", allow_pickle=False
                    ) as saved:
                        same = _array(output) == _array(saved["y"])
                        if not same:
                            _save(
                                root,
                                folder / f"restored-B2-{role}-different.npz",
                                y=output,
                            )
                        record["restored"][role] = dict(
                            byte_identical=same,
                            actual=_array(output),
                            canonical_file=f"native-B2-{role}.npz",
                        )
                        _require_output(output, 2)
                        assert same
                    _unchanged(owner, baseline, identities, graphdef)
                    checkpoint(f"restored-original-B2-{role}")
                    del result, output, values, control
                for mode, optimization in (
                    ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
                    ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
                ):
                    options = ort.SessionOptions()
                    options.graph_optimization_level = optimization
                    options.intra_op_num_threads = options.inter_op_num_threads = 1
                    session = ort.InferenceSession(
                        payload,
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
                    counts["sessions"] += 1
                    record["runtime"][mode] = dict(
                        serialized_sha256=_sha(payload),
                        providers=session.get_providers(),
                        optimization=str(optimization),
                        feeds={},
                    )
                    checkpoint(f"session-{mode}")
                    for batch, flag in ((2, not caller_x64), (5, caller_x64)):
                        x = _input(batch)
                        _unchanged(owner, baseline, identities, graphdef)
                        result = session.run(
                            None,
                            {"in_0": x, "deterministic": np.asarray(flag, np.bool_)},
                        )
                        counts["runs"] += 1
                        assert len(result) == 1
                        target, _ = _output(result[0], batch)
                        receipt = dict(
                            batch=batch,
                            deterministic=flag,
                            target=_save(
                                root, folder / f"target-{mode}-B{batch}.npz", y=target
                            ),
                            comparisons={},
                        )
                        record["runtime"][mode]["feeds"][str(batch)] = receipt
                        _require_output(target, batch)
                        for role in ("eager", "jit"):
                            with np.load(
                                folder / f"native-B{batch}-{role}.npz",
                                allow_pickle=False,
                            ) as saved:
                                receipt["comparisons"][role] = _compare(
                                    target, saved["y"]
                                )
                        _unchanged(owner, baseline, identities, graphdef)
                        if not caller_x64 and mode == "disabled" and batch == 2:
                            damaged = target.copy()
                            with np.load(
                                folder / "native-B2-eager.npz", allow_pickle=False
                            ) as saved:
                                reference = saved["y"]
                                damaged.flat[0] = reference.flat[0] + np.float32(
                                    1
                                    + 32
                                    * (_ATOL + _RTOL * abs(float(reference.flat[0])))
                                )
                                with pytest.raises(AssertionError):
                                    _compare(damaged, reference)
                            record["negative_target_element"] = dict(
                                status="rejected",
                                element=0,
                                modified=_array(damaged),
                                criterion_unchanged=True,
                            )
                            del damaged
                        checkpoint(f"target-{mode}-B{batch}-both-original-references")
                        del result, target, x
                    del session
                assert (
                    precision_report_to_json(get_precision_report(artifact))
                    == report_json
                )
                assert type(owner).__call__ is original_method
                _unchanged(owner, baseline, identities, graphdef)
                assert _contract(case) == _CONTRACT
                record["contract_after_preparation_sha256"] = _contract(case)
                assert counts == _EXPECTED_COUNTS
                peer = root / f"caller-{str(not caller_x64).lower()}" / "receipt.json"
                if peer.exists():
                    other = json.loads(peer.read_text())
                    assert other["status"] == "BOUNDED_NUMERICAL_ACCEPTANCE"
                    for batch in ("2", "5"):
                        assert (
                            record["native"][batch]["input"]
                            == other["native"][batch]["input"]
                        )
                        assert (
                            record["native"][batch]["deterministic"]
                            != other["native"][batch]["deterministic"]
                        )
                        for role in ("eager", "jit"):
                            assert (
                                record["native"][batch]["roles"][role]["arrays"]
                                == other["native"][batch]["roles"][role]["arrays"]
                            )
                        for mode in ("disabled", "all"):
                            assert (
                                record["runtime"][mode]["feeds"][batch]["target"][
                                    "arrays"
                                ]
                                == other["runtime"][mode]["feeds"][batch]["target"][
                                    "arrays"
                                ]
                            )
                    record["paired_control_invariance"] = (
                        "byte-identical native roles and target modes for both B values; opposite BOOL and caller x64"
                    )
            record["status"] = "BOUNDED_NUMERICAL_ACCEPTANCE"
            record["scope"] = (
                "Original GPT_MLP FP32 pairwise sample only; no public SOURCE transition or other original-model closure"
            )
            checkpoint("complete-with-restored-caller-context")
        except Exception as exc:
            record["status"] = "FAILED_PARTIAL"
            record["error"] = dict(type=type(exc).__name__, message=str(exc))
            if "built" in locals():
                try:
                    _json(
                        folder / "observed-function-bodies-at-failure.json",
                        [
                            dict(
                                domain=definition.domain,
                                name=definition.name,
                                typed_native_body=typed_trace_manifest(body),
                            )
                            for definition, body in built
                        ],
                    )
                except Exception as observation_error:
                    record["failure_observation_error"] = dict(
                        type=type(observation_error).__name__,
                        message=str(observation_error),
                    )
            if "owner" in locals():
                try:
                    failure_state, _ = _state(owner)
                    record["failure_state"] = failure_state
                    record["failure_graph_variable_identities"] = _identities(owner)
                    record["failure_state_unchanged"] = failure_state == baseline
                except Exception as state_exc:
                    record["failure_state_error"] = dict(
                        type=type(state_exc).__name__, message=str(state_exc)
                    )
            # Preserve counts and failure even when writing the receipt crosses the ceiling.
            (folder / "receipt.json").write_text(
                json.dumps(record, sort_keys=True, indent=2) + "\n"
            )
            (root / "failed.json").write_text(
                json.dumps(
                    dict(cell=folder.name, error=record["error"], counts=counts),
                    indent=2,
                )
                + "\n"
            )
            raise
