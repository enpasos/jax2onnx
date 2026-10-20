# tests/extra_tests/converter/test_source_erfc.py

"""Actual SOURCE Erfc exports and bounded, explicitly configured CPU evidence.

Low precision is inspected structurally only. Numerical tolerances below belong
only to FLOAT/DOUBLE in a fresh child process; no ORT session is created here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import jax
from jax import export
from jax.extend.core import jaxpr_as_fun
import jax.numpy as jnp
import numpy as np
import onnx
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
from jax2onnx.plugins.jax.lax.erfc import ErfcPlugin
from jax2onnx.plugins.plugin_system import FunctionPlugin
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)

_DATA = Path(__file__).with_name("source_erfc_data")
_COEFFICIENT_SHA256 = "6ca6b169fd53f1f8097dabd57f1a6af0b59e0eab66534ff005db58ca16e55c29"
_DTYPES = (np.float16, jnp.bfloat16, np.float32, np.float64)
_ALLOWED = {"Add", "Sub", "Mul", "Div", "Neg", "Abs", "Exp", "Less", "Where"}
_ORIGINALS = jax.lax.erfc, jax.nn.gelu, jnp.exp, jnp.multiply
_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)
_OUTPUTS = (
    "direct_erfc",
    "t",
    "half_x",
    "gelu_erfc",
    "gelu",
    "consumer",
    "consumer_rounded",
)


@onnx_function(name="SourceErfcSharedBatch")
def _named_erfc(x: Any) -> Any:
    return jax.lax.erfc(x)


def _named_erfc_entry(x: Any) -> Any:
    # FunctionPlugin patches this module binding during activation.
    return _named_erfc(x)


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
        assert (jax.lax.erfc, jax.nn.gelu, jnp.exp, jnp.multiply) == _ORIGINALS


def _body(closed: Any) -> Any:
    inner = getattr(closed, "jaxpr", closed)
    return inner if inner is not closed else closed


def _fresh(fn: Any) -> Any:
    def call(*args: Any) -> Any:
        return fn(*args)

    return call


def _signature(variables: Any) -> tuple[Any, ...]:
    return tuple(
        (
            np.dtype(v.aval.dtype).name,
            normalized_shape_dimensions(v.aval.shape),
            bool(v.aval.weak_type),
        )
        for v in variables
    )


def _payload(value: Any) -> tuple[Any, ...]:
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.tobytes()


def _same(actual: Any, expected: Any) -> None:
    assert _payload(actual) == _payload(expected)


def _trace_roles(fn: Any, specs: Any, *, named: bool = False) -> Any:
    """Capture outside activation and check the actual activated/replayed trace."""
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    native = jax.make_jaxpr(_fresh(fn))(*specs)
    manifest = typed_trace_manifest(native)
    constants = tuple(_payload(value) for value in native.consts)
    replayed = jax.make_jaxpr(_fresh(jaxpr_as_fun(native)))(*specs)
    assert typed_trace_manifest(replayed) == manifest
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(_fresh(fn))(*specs)
        assert _signature(_body(activated).invars) == _signature(_body(native).invars)
        assert _signature(_body(activated).outvars) == _signature(_body(native).outvars)
        if named:
            assert any(
                eqn.primitive.name.startswith("onnx_fn::")
                for eqn in _body(activated).eqns
            )
        else:
            assert typed_trace_manifest(activated) == manifest
            assert tuple(_payload(value) for value in activated.consts) == constants
    assert typed_trace_manifest(native) == manifest
    assert tuple(_payload(value) for value in native.consts) == constants
    return native


def _artifact(fn: Any, specs: Any, *, opset: int = 23, named: bool = False) -> Any:
    native = _trace_roles(fn, specs, named=named)
    before = _globals()
    proto = ui._to_onnx(fn, specs, precision_policy=PrecisionPolicy.SOURCE, opset=opset)
    assert _globals() == before
    serialized = proto.SerializeToString()
    proto = onnx.load_model_from_string(serialized)
    assert proto.SerializeToString() == serialized
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
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
    return proto


def _port(value: Any, code: int, shape: Any) -> None:
    tensor = value.type.tensor_type
    assert tensor.elem_type == code
    assert tuple(
        dim.dim_param if dim.HasField("dim_param") else dim.dim_value
        for dim in tensor.shape.dim
    ) == tuple(shape)


def _serialized_constants(scope: Any) -> dict[str, Any]:
    values = {v.name: v for v in getattr(scope, "initializer", ())}
    for node in scope.node:
        if node.op_type == "Constant":
            assert not node.domain and len(node.output) == 1
            assert len(node.attribute) == 1
            attribute = node.attribute[0]
            assert (
                attribute.name == "value"
                and attribute.type == onnx.AttributeProto.TENSOR
            )
            values[node.output[0]] = attribute.t
    return values


def _expression(scope: Any, name: str, inputs: dict[str, Any]) -> Any:
    """Read the serialized dependency expression, retaining operand order."""
    constants = _serialized_constants(scope)
    producers = {name: node for node in scope.node for name in node.output}
    memo = dict(inputs)

    def visit(value: str) -> Any:
        if value in memo:
            return memo[value]
        if value in constants:
            tensor = constants[value]
            assert not tensor.dims and tensor.raw_data
            result = ("constant", tensor.data_type, tensor.raw_data.hex())
        else:
            node = producers[value]
            assert not node.domain and len(node.output) == 1
            if node.op_type == "Identity":
                assert not node.attribute and len(node.input) == 1
                result = visit(node.input[0])
            else:
                assert node.op_type in _ALLOWED | {"Cast"}
                if node.op_type == "Cast":
                    assert len(node.attribute) == 1 and node.attribute[0].name == "to"
                    result = ("Cast", node.attribute[0].i, visit(node.input[0]))
                else:
                    assert not node.attribute
                    result = (node.op_type, *(visit(v) for v in node.input))
        memo[value] = result
        return result

    return visit(name)


def _expected_expression(x: Any, dtype: str) -> Any:
    """Frozen reviewed Horner/grouping contract; never emits an ONNX graph."""
    raw = (_DATA / "coefficients.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _COEFFICIENT_SHA256
    payload = json.loads(raw)[dtype]
    code = onnx.TensorProto.FLOAT if dtype == "float32" else onnx.TensorProto.DOUBLE

    def constant(key: str) -> Any:
        return "constant", code, payload["constants"][key]["raw_le_hex"]

    def polynomial(value: Any, group: str) -> Any:
        array = payload["arrays"][group]
        result: Any = ("constant", code, array[0]["raw_le_hex"])
        for item in array[1:]:
            result = (
                "Add",
                ("Mul", result, value),
                ("constant", code, item["raw_le_hex"]),
            )
        return result

    one, zero, two = (constant(key) for key in ("one", "zero", "two"))
    square = ("Mul", x, x)
    z = ("Neg", square)
    absolute = ("Abs", x)
    exponential = ("Exp", z)
    if dtype == "float32":
        q = ("Div", one, square)
        tail = (
            "Mul",
            ("Mul", exponential, ("Div", one, absolute)),
            (
                "Where",
                ("Less", absolute, two),
                polynomial(q, "ErfcP"),
                polynomial(q, "ErfcR"),
            ),
        )
    else:
        tail = (
            "Where",
            ("Less", absolute, constant("eight")),
            (
                "Div",
                ("Mul", exponential, polynomial(absolute, "ErfcP")),
                polynomial(absolute, "ErfcQ"),
            ),
            (
                "Div",
                ("Mul", exponential, polynomial(absolute, "ErfcR")),
                polynomial(absolute, "ErfcS"),
            ),
        )
    tail = ("Where", ("Less", z, constant("negative_maxlog")), zero, tail)
    large = ("Where", ("Less", x, zero), ("Sub", two, tail), tail)
    small = ("Mul", x, polynomial(square, "ErfT"))
    if dtype == "float64":
        small = ("Div", small, polynomial(square, "ErfU"))
    return ("Where", ("Less", absolute, one), ("Sub", one, small), large)


def _assert_erfc(
    scope: Any, input_name: str, output_name: str, dtype: Any, shape: Any
) -> None:
    code = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    low = np.dtype(dtype).name in {"float16", "bfloat16"}
    work = "float64" if np.dtype(dtype).name == "float64" else "float32"
    work_code = onnx.TensorProto.DOUBLE if work == "float64" else onnx.TensorProto.FLOAT
    expected = _expected_expression(("Cast", work_code, "x") if low else "x", work)
    if low:
        expected = ("Cast", code, expected)
    assert _expression(scope, output_name, {input_name: "x"}) == expected
    operations = Counter(node.op_type for node in scope.node)
    assert set(operations) <= _ALLOWED | {"Constant", "Identity", "Cast"}
    assert operations["Cast"] == (2 if low else 0)
    typed = {
        v.name: v
        for v in (
            *scope.value_info,
            *getattr(scope, "input", ()),
            *getattr(scope, "output", ()),
        )
        if not isinstance(v, str)
    }
    constants = _serialized_constants(scope)
    for tensor in constants.values():
        assert tensor.data_type == work_code and not tensor.dims and tensor.raw_data
    payload = json.loads((_DATA / "coefficients.json").read_text())[work]
    required = {"one", "zero", "two", "negative_maxlog"}
    if work == "float64":
        required.add("eight")
    expected_bytes = {payload["constants"][key]["raw_le_hex"] for key in required} | {
        value["raw_le_hex"] for group in payload["arrays"].values() for value in group
    }
    assert {tensor.raw_data.hex() for tensor in constants.values()} == expected_bytes
    _port(typed[input_name], code, shape)
    _port(typed[output_name], code, shape)
    for node in scope.node:
        if node.op_type in {"Constant", "Identity"}:
            continue
        for output in node.output:
            assert (
                output in typed
            ), f"missing explicit intermediate metadata for {output}"
            result_code = onnx.TensorProto.BOOL if node.op_type == "Less" else work_code
            if node.op_type == "Cast":
                result_code = next(a.i for a in node.attribute if a.name == "to")
            _port(typed[output], result_code, shape)


@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize("shape", [(), (2, 3, 4)], ids=["scalar", "rank3"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=["f16", "bf16", "f32", "f64"])
def test_erfc_serialized_typed_expression(dtype: Any, shape: Any, opset: int) -> None:
    specs = [jax.ShapeDtypeStruct(shape, dtype)]
    model = _artifact(lambda x: jax.lax.erfc(x), specs, opset=opset)
    assert not model.functions
    assert next(v.version for v in model.opset_import if v.domain == "") == opset
    _assert_erfc(
        model.graph, model.graph.input[0].name, model.graph.output[0].name, dtype, shape
    )


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f16", "bf16", "f32", "f64"])
def test_plain_jit_preserves_native_erfc(dtype: Any) -> None:
    fn = jax.jit(lambda x: jax.lax.erfc(x))
    specs = [jax.ShapeDtypeStruct((2, 3, 4), dtype)]
    native = _trace_roles(fn, specs)
    assert any(e.primitive.name in {"jit", "pjit"} for e in _body(native).eqns)
    model = _artifact(fn, specs)
    _assert_erfc(
        model.graph,
        model.graph.input[0].name,
        model.graph.output[0].name,
        dtype,
        (2, 3, 4),
    )


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f16", "bf16", "f32", "f64"])
def test_native_erfc_preserves_scalar_weakness(dtype: Any) -> None:
    specs = [jax.ShapeDtypeStruct((), dtype, weak_type=True)]
    model = _artifact(lambda x: jax.lax.erfc(x), specs)
    report = get_precision_report(model)
    assert report is not None and report.traced_outputs[0].weak_type is True
    _assert_erfc(
        model.graph, model.graph.input[0].name, model.graph.output[0].name, dtype, ()
    )


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f16", "bf16", "f32", "f64"])
def test_exact_gelu_keeps_original_staging(dtype: Any) -> None:
    shape = (2, 3, 4)
    specs = [jax.ShapeDtypeStruct(shape, dtype)]
    model = _artifact(lambda x: jax.nn.gelu(x, approximate=False), specs)
    code = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    low = np.dtype(dtype).name in {"float16", "bfloat16"}

    def scalar(value: Any) -> Any:
        return "constant", code, np.asarray(value, dtype=dtype).tobytes().hex()

    t = ("Mul", ("Neg", "x"), scalar(np.sqrt(0.5)))
    local = _expected_expression(
        ("Cast", onnx.TensorProto.FLOAT, t) if low else t,
        "float64" if dtype is np.float64 else "float32",
    )
    if low:
        local = ("Cast", code, local)
    expected = ("Mul", ("Mul", scalar(0.5), "x"), local)
    assert (
        _expression(
            model.graph, model.graph.output[0].name, {model.graph.input[0].name: "x"}
        )
        == expected
    )
    assert sum(n.op_type == "Cast" for n in model.graph.node) == (2 if low else 0)
    assert not any(n.op_type in {"Erf", "Gelu"} for n in model.graph.node)
    typed = {
        value.name: value for value in (*model.graph.value_info, *model.graph.output)
    }
    for node in model.graph.node:
        if node.op_type in {"Neg", "Mul"}:
            expression = _expression(
                model.graph, node.output[0], {model.graph.input[0].name: "x"}
            )
            if expression in {t, ("Neg", "x"), ("Mul", scalar(0.5), "x"), expected}:
                _port(typed[node.output[0]], code, shape)


@contextmanager
def _observe_functions() -> Any:
    bodies, definitions = [], []
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
        definitions.append((definition, bodies[-1]))
        return definition

    with (
        patch.object(FunctionPlugin, "_trace_function_body", trace),
        patch.object(FunctionPlugin, "_build_function", build),
    ):
        yield definitions


def _named_case(dtype: Any) -> tuple[Any, Any, Any]:
    (batch,) = export.symbolic_shape("B")
    specs = [jax.ShapeDtypeStruct((batch, 2, 3), dtype)]
    native = _trace_roles(_named_erfc_entry, specs, named=True)
    with _observe_functions() as built:
        model = _artifact(_named_erfc_entry, specs, named=True)
    assert len(model.functions) == len(built) == 1
    definition, body = built[0]
    (function,) = model.functions
    assert (definition.domain, definition.name) == (function.domain, function.name)
    assert typed_trace_manifest(body) == typed_trace_manifest(native)
    assert not model.graph.initializer, "Erfc coefficients must belong to the Function"
    assert len(function.input) == len(function.output) == 1
    assert _serialized_constants(function)
    _assert_erfc(function, function.input[0], function.output[0], dtype, ("B", 2, 3))
    code = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    _port(model.graph.input[0], code, ("B", 2, 3))
    _port(model.graph.output[0], code, ("B", 2, 3))
    calls = [
        node
        for node in model.graph.node
        if (node.domain, node.op_type) == (function.domain, function.name)
    ]
    assert len(calls) == 1
    return model, native, body


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_shared_batch_named_erfc_has_local_constants(dtype: Any) -> None:
    _named_case(dtype)


@pytest.mark.parametrize(
    "defect",
    [
        "arity",
        "outputs",
        "dtype",
        "shape",
        "weak",
        "weak-metadata",
        "scope",
        "unknown",
        "accuracy",
        "placement",
        "integer",
        "complex",
        "effects",
    ],
)
def test_malformed_erfc_contracts_fail_before_lowering(defect: str) -> None:
    (batch,) = export.symbolic_shape("B")
    specs = [jax.ShapeDtypeStruct((batch, 3), np.float32)]
    original = _body(jax.make_jaxpr(lambda x: jax.lax.erfc(x))(*specs)).eqns[0]
    source = SimpleNamespace(aval=copy(original.invars[0].aval))
    result = SimpleNamespace(aval=copy(original.outvars[0].aval))
    inputs, outputs, params, effects = [source], [result], dict(original.params), ()
    if defect == "arity":
        inputs = []
    elif defect == "outputs":
        outputs = [result, result]
    elif defect == "dtype":
        result.aval = jax.ShapeDtypeStruct(specs[0].shape, np.float16)
    elif defect == "shape":
        result.aval = jax.ShapeDtypeStruct((batch, 4), np.float32)
    elif defect == "weak":
        result.aval = jax.ShapeDtypeStruct(specs[0].shape, np.float32, weak_type=True)
    elif defect == "weak-metadata":
        result.aval = SimpleNamespace(
            dtype=np.dtype(np.float32), shape=specs[0].shape, weak_type=1
        )
    elif defect == "scope":
        (foreign,) = export.symbolic_shape("B")
        result.aval = jax.ShapeDtypeStruct((foreign, 3), np.float32)
    elif defect in {"unknown", "accuracy", "placement"}:
        params[
            {
                "unknown": "unreviewed",
                "accuracy": "accuracy",
                "placement": "out_sharding",
            }[defect]
        ] = {
            "unknown": None,
            "accuracy": jax.lax.AccuracyMode.HIGHEST,
            "placement": object(),
        }[
            defect
        ]
    elif defect in {"integer", "complex"}:
        # Use a static shape so rejection exercises Erfc's dtype rule itself.
        dtype = np.int32 if defect == "integer" else np.complex64
        source.aval = result.aval = jax.ShapeDtypeStruct((2, 3), dtype)
    else:
        effects = ("effect",)
    equation = SimpleNamespace(
        primitive=original.primitive,
        invars=inputs,
        outvars=outputs,
        params=params,
        effects=effects,
    )
    graph = SimpleNamespace(
        invars=[source], constvars=(), outvars=outputs, eqns=[equation], effects=effects
    )
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert caught.value.category == (
        "lowering-defect"
        if defect in {"arity", "outputs", "dtype", "shape", "weak", "weak-metadata"}
        else "profile-exclusion"
    )


_NONDEFAULT_ACCURACIES = (
    pytest.param(False, id="false"),
    pytest.param(0, id="zero"),
    pytest.param(jax.lax.AccuracyMode.DEFAULT, id="native-default-mode"),
    pytest.param(jax.lax.AccuracyMode.HIGHEST, id="native-highest-mode"),
    pytest.param(jax.lax.Tolerance(atol=1e-6), id="native-tolerance"),
    pytest.param(object(), id="arbitrary-object"),
)


def _erfc_parameter_graph(params: dict[str, Any]) -> Any:
    native = _body(
        jax.make_jaxpr(jax.lax.erfc)(jax.ShapeDtypeStruct((2, 3), np.float32))
    )
    (equation,) = native.eqns
    return SimpleNamespace(
        invars=native.invars,
        constvars=(),
        outvars=native.outvars,
        effects=(),
        eqns=[
            SimpleNamespace(
                primitive=equation.primitive,
                invars=equation.invars,
                outvars=equation.outvars,
                effects=(),
                params=params,
            )
        ],
    )


def test_native_default_erfc_accuracy_is_absent_or_none() -> None:
    native = jax.make_jaxpr(jax.lax.erfc)(jax.ShapeDtypeStruct((2, 3), np.float32))
    (equation,) = _body(native).eqns
    assert equation.params.get("accuracy") is None
    validate_source_jaxpr(native, 23)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({}, id="absent"),
        pytest.param({"accuracy": None}, id="none"),
        pytest.param(
            {"accuracy": None, "sharding": None, "out_sharding": None},
            id="none-unspecified-placement",
        ),
    ],
)
def test_default_erfc_parameter_metadata_is_admitted(params: dict[str, Any]) -> None:
    # Synthetic metadata control; native public-default behavior is checked above.
    validate_source_jaxpr(_erfc_parameter_graph(params), 23)


class _ErfcInputReached(RuntimeError):
    pass


@pytest.mark.parametrize("policy", [PrecisionPolicy.LEGACY, PrecisionPolicy.SOURCE])
@pytest.mark.parametrize(
    "params", [{}, {"accuracy": None}, {"accuracy": None, "out_sharding": None}]
)
def test_default_erfc_accuracy_reaches_existing_lowering(
    params: dict[str, Any], policy: PrecisionPolicy
) -> None:
    def input_reached(*args: Any, **kwargs: Any) -> Any:
        raise _ErfcInputReached("existing input access")

    context = SimpleNamespace(
        precision_policy=policy,
        require_value_for_var=input_reached,
        get_value_for_var=input_reached,
        fresh_name=lambda hint: hint,
    )
    equation = SimpleNamespace(params=params, invars=[object()], outvars=[object()])
    before = dict(params)
    with pytest.raises(_ErfcInputReached, match="existing input access"):
        ErfcPlugin().lower(context, equation)
    assert params == before


class _ErfcContextPoison:
    def __init__(self, policy: PrecisionPolicy) -> None:
        self.precision_policy = policy

    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(
            f"Erfc accessed context before rejecting parameters: {name}"
        )


@pytest.mark.parametrize("accuracy", _NONDEFAULT_ACCURACIES)
def test_nondefault_erfc_accuracy_is_rejected_by_profile(accuracy: Any) -> None:
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(_erfc_parameter_graph({"accuracy": accuracy}), 23)
    assert caught.value.category == "profile-exclusion"
    assert caught.value.primitive == "erfc"


@pytest.mark.parametrize("accuracy", _NONDEFAULT_ACCURACIES)
@pytest.mark.parametrize("policy", [PrecisionPolicy.LEGACY, PrecisionPolicy.SOURCE])
def test_nondefault_erfc_accuracy_rejects_before_policy_or_ir_access(
    accuracy: Any, policy: PrecisionPolicy
) -> None:
    equation = SimpleNamespace(params={"accuracy": accuracy})
    with pytest.raises(NotImplementedError, match="accuracy"):
        ErfcPlugin().lower(_ErfcContextPoison(policy), equation)
    assert equation.params["accuracy"] is accuracy


@pytest.mark.parametrize("accuracy", _NONDEFAULT_ACCURACIES)
def test_direct_source_erfc_accuracy_rejects_before_ir_access(accuracy: Any) -> None:
    equation = SimpleNamespace(params={"accuracy": accuracy})
    with pytest.raises(NotImplementedError, match="accuracy"):
        ErfcPlugin()._lower_source(_ErfcContextPoison(PrecisionPolicy.SOURCE), equation)
    assert equation.params["accuracy"] is accuracy


@pytest.mark.parametrize("entry", ["profile", "source-plugin"])
def test_unknown_none_erfc_parameter_stays_rejected(entry: str) -> None:
    params = {"unreviewed": None}
    if entry == "profile":
        with pytest.raises(SourcePrecisionError) as caught:
            validate_source_jaxpr(_erfc_parameter_graph(params), 23)
        assert caught.value.category == "profile-exclusion"
    else:
        with pytest.raises(NotImplementedError, match="accuracy"):
            ErfcPlugin()._lower_source(
                _ErfcContextPoison(PrecisionPolicy.SOURCE),
                SimpleNamespace(params=params),
            )
    assert params == {"unreviewed": None}


def test_x64_off_double_erfc_is_configuration_conflict() -> None:
    with jax.enable_x64(False):
        with pytest.raises(SourcePrecisionError) as caught:
            ui._to_onnx(
                jax.lax.erfc,
                [jax.ShapeDtypeStruct((2, 3), np.float64)],
                precision_policy=PrecisionPolicy.SOURCE,
            )
        assert caught.value.category == "configuration-conflict"


def test_opset16_erfc_remains_outside_existing_subset() -> None:
    with pytest.raises(SourcePrecisionError) as caught:
        ui._to_onnx(
            jax.lax.erfc,
            [jax.ShapeDtypeStruct((2,), np.float32)],
            precision_policy=PrecisionPolicy.SOURCE,
            opset=16,
        )
    assert caught.value.category == "profile-exclusion"


def _seven_outputs(value: Any, zero_value: Any) -> Any:
    """Native reference and actual exporter target, including explicit consumer."""
    dtype = np.dtype(value.dtype)
    sqrt_half = np.asarray(np.sqrt(0.5), dtype=dtype)
    t = -value * sqrt_half
    half_x = 0.5 * value
    direct = jax.lax.erfc(value)
    gelu_erfc = jax.lax.erfc(t)
    gelu = jax.nn.gelu(value, approximate=False)
    at_zero = jax.lax.erfc(zero_value)
    big = jnp.asarray(16777216, dtype=jnp.float64)
    if dtype == np.dtype(np.float32):
        wide = jax.lax.convert_element_type(at_zero, jnp.float64)
        shifted = wide + big
        consumer_rounded = jax.lax.convert_element_type(shifted, jnp.float32)
        rewidened = jax.lax.convert_element_type(consumer_rounded, jnp.float64)
        delta = rewidened - big
        consumer = jax.lax.convert_element_type(delta, jnp.float32)
    else:
        consumer_rounded = at_zero + big
        consumer = consumer_rounded - big
    return direct, t, half_x, gelu_erfc, gelu, consumer, consumer_rounded


def test_configured_cpu_frozen_seven_output_diagnostics(tmp_path: Path) -> None:
    """The child owns all sessions and the process/thread-level FTZ side effect."""
    environment = os.environ.copy()
    environment.update({key: "1" for key in _THREAD_KEYS})
    environment.update(
        JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(_DATA / "configured_cpu.py"),
            str(Path(__file__).resolve()),
            str(tmp_path),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    (tmp_path / "child.stdout").write_text(completed.stdout)
    (tmp_path / "child.stderr").write_text(completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "configured-cpu.json").read_text())
    assert report["status"] == "DIAGNOSTIC_COMPLETE"
    assert report["diagnostic_execution"] == "COMPLETE"
    assert report["diagnostic_regression"] == "PASSED"
    assert report["sessions"] == 8 and report["runs"] == 12
    assert report["session_attempts"] == 8 and report["run_attempts"] == 12
    assert report["native_attempts"] == report["native_completed"] == 36
    assert report["export_attempts"] == report["exports_completed"] == 4
    assert report["low_precision_numerical_acceptance"] == "UNVERIFIED"
    fixed = [
        json.loads((tmp_path / path).read_text())
        for path in report["fixed_comparisons"]
    ]
    assert len(fixed) == 4
    assert len(report["fixed_results"]) == 4
    assert len(report["named_comparisons"]) == len(report["named_results"]) == 8
    full_pass = all(
        check["pass"]
        for comparison in fixed
        for checks in comparison["roles"].values()
        for check in checks.values()
    )
    assert report["full_seven_output_gate"] == ("PASSED" if full_pass else "FAILED")
    assert report["direct_erfc"] == report["consumer"] == "PASSED"
    assert report["fixed_exception_mantissa_consumer"] == "PASSED"
    assert report["named_shared_batch"] == report["ordinary_composition"] == "PASSED"
    for comparison, summary in zip(fixed, report["fixed_results"], strict=True):
        assert comparison["full_seven_output_pass"] == all(
            check["pass"]
            for checks in comparison["roles"].values()
            for check in checks.values()
        )
        assert comparison["fixed_contract"]["pass"]
        assert summary["outputs"] == {
            role: {
                name: "PASSED" if check["pass"] else "FAILED"
                for name, check in checks.items()
            }
            for role, checks in comparison["roles"].items()
        }
        for checks in comparison["roles"].values():
            assert checks["direct_erfc"]["pass"]
            assert checks["consumer"]["pass"] and checks["consumer_rounded"]["pass"]
        assert comparison["ordinary_composition"]["pass"]
        assert len(comparison["ordinary_composition"]["indices"]) == 10
    for path in report["named_comparisons"]:
        assert all(
            check["pass"]
            for check in json.loads((tmp_path / path).read_text()).values()
        )
    native_passes = []
    for record in report["native_diagnostics"].values():
        native = json.loads((tmp_path / record["path"]).read_text())
        assert record["statuses"] == {
            role: check["status"] for role, check in native.items()
        }
        for check in native.values():
            passed = all(item["pass"] for item in check["outputs"].values())
            assert check["status"] == ("PASSED" if passed else "FAILED")
            native_passes.append(passed)
    assert report["native_reference_agreement"] == (
        "PASSED" if all(native_passes) else "FAILED"
    )
    for dtype in ("float32", "float64"):
        assert (
            report["replayed_eager_preservation"][dtype]
            == report["native_diagnostics"][f"fixed-{dtype}"]["statuses"][
                "replayed-eager"
            ]
        )
    if report["native_reference_agreement"] == "FAILED":
        assert report["known_native_prefix_conflict"] is True
        assert not full_pass
    else:
        assert report["native_reference_agreement"] == "PASSED" and full_pass
    assert not report["unexpected_numerical_failures"]
