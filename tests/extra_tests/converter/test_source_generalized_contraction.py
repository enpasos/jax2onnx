# tests/extra_tests/converter/test_source_generalized_contraction.py

"""Finite real contraction geometry and exact native call forwarding.

BF16/integer/Boolean/complex contraction arithmetic, explicit algorithms and
attention remain unfinished obligations. Recorded failed callbacks and absent
criteria on the eleven original recipes are not numerical acceptance.
"""

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import asdict
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from flax import linen as nn, nnx
import jax
from jax.extend.core import jaxpr_as_fun
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import Primitive, Var
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import (
    _Site,
    _tensor,
    _validate_dot,
    SourcePrecisionError,
)
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import _SourceFunctionTrace
from jax2onnx.plugins.flax.nnx.linear_general import LinearGeneralPlugin
from jax2onnx.plugins.flax.linen.dense_general import DenseGeneralPlugin
from jax2onnx.plugins.flax.nnx.einsum import EinsumModulePlugin
from jax2onnx.plugins.flax.linen.einsum import EinsumPlugin
from jax2onnx.plugins.jax.numpy.einsum import JnpEinsumPlugin
from jax2onnx.plugins.jax.lax.dot_general import DotGeneralPlugin
from jax2onnx.precision import (
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from tests.extra_tests.converter import source_precision_support as source
from tests.extra_tests.converter.source_target_evidence import require_target_pair
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_native_contract import _forged_aval
from tests.extra_tests.converter.test_source_predicate_families import (
    _globals,
    _settings,
    _restore_context as _restore_context,
)
from tests.extra_tests.converter.test_source_tensor_matrix import (
    _body,
    _observe_functions,
    _origin,
    _payload as _array_payload,
    _port,
    _signature,
)

_BITWISE = SampleCriteria("bitwise")
_MATRIX = (((1,), (0,)), ((), ()))
_WIDTHS = [
    ("float32", False),
    ("float32", True),
    ("float64", True),
    ("float16", False),
    ("float16", True),
]
_FORWARDERS = (
    LinearGeneralPlugin,
    DenseGeneralPlugin,
    JnpEinsumPlugin,
    EinsumModulePlugin,
    EinsumPlugin,
)
_ORIGINAL_BINDINGS = (
    nnx.LinearGeneral.__call__,
    nn.DenseGeneral.__call__,
    jnp.einsum,
    nnx.Einsum.__call__,
    nn.Einsum.__call__,
)

_CASES = [
    ((2, 3, 4), (4, 3, 5), (((2, 1), (0, 1)), ((), ()))),
    ((2, 3, 4), (4, 5), (((2,), (0,)), ((), ()))),
    ((2, 3), (3, 4, 5), (((1,), (0,)), ((), ()))),
    ((2, 3, 4, 5), (4, 3, 5, 6), (((3,), (2,)), ((2, 1), (0, 1)))),
    ((2, 3, 2, 3, 4), (4, 3, 2, 2), (((4, 1), (0, 1)), ((2,), (2,)))),
    ((3,), (3,), (((0,), (0,)), ((), ()))),
    ((2, 3), (3,), (((1,), (0,)), ((), ()))),
    ((3,), (3, 2), (((0,), (0,)), ((), ()))),
    ((2, 3), (3, 2), (((1, 0), (0, 1)), ((), ()))),
    ((2, 3), (2, 3), (((1,), (1,)), ((0,), (0,)))),
    ((2, 3), (3, 2), (((), ()), ((1, 0), (0, 1)))),
    ((2, 3), (4, 2), (((), ()), ((), ()))),
    ((), (2, 3), (((), ()), ((), ()))),
    ((2, 3), (), (((), ()), ((), ()))),
    ((), (), (((), ()), ((), ()))),
    ((2, 3), (2, 4), (((), ()), ((0,), (0,)))),
    ((2, 0), (0, 3), _MATRIX),
    ((2, 0, 3), (3, 0, 4), (((1, 2), (1, 0)), ((), ()))),
    ((0, 2), (2, 3), _MATRIX),
    ((2, 3), (3, 0), _MATRIX),
    ((0, 2, 3), (0, 3, 4), (((2,), (1,)), ((0,), (0,)))),
    ((0,), (0,), (((0,), (0,)), ((), ()))),
    ((2, 0), (0, 3), (((), ()), ((), ()))),
    ((2, 3, 2, 4), (2, 5, 2, 4), (((3,), (3,)), ((0, 2), (0, 2)))),
    ((2, 2, 3, 5), (2, 5, 2, 4), (((3,), (1,)), ((0, 1), (0, 2)))),
]


def _payload(dtype: Any, shape: tuple[int, ...], *, right: bool = False) -> np.ndarray:
    indices = np.arange(np.prod(shape, dtype=np.int64)).reshape(shape)
    values = ((indices * (3 if right else 5)) % 13 - 6) / 8
    if np.dtype(dtype) == np.dtype(np.float64) and not right:
        values = values + (indices % 3) * 2.0**-35
    return values.astype(dtype)


def _oracle(left: Any, right: Any, dimensions: Any, dtype: Any) -> np.ndarray:
    """Independent coordinate summation; never flatten either contracting set."""
    left, right = np.asarray(left), np.asarray(right)
    (lc, rc), (lb, rb) = dimensions
    lf = [i for i in range(left.ndim) if i not in (*lc, *lb)]
    rf = [i for i in range(right.ndim) if i not in (*rc, *rb)]
    shape = (
        tuple(left.shape[i] for i in lb)
        + tuple(left.shape[i] for i in lf)
        + tuple(right.shape[i] for i in rf)
    )
    output = np.empty(shape, dtype)
    for index in np.ndindex(shape):
        li, ri = [0] * left.ndim, [0] * right.ndim
        for pos, (a, b) in enumerate(zip(lb, rb, strict=True)):
            li[a] = ri[b] = index[pos]
        for pos, a in enumerate(lf, len(lb)):
            li[a] = index[pos]
        for pos, b in enumerate(rf, len(lb) + len(lf)):
            ri[b] = index[pos]
        if not lc:
            # Exactly one product, without an invented +0 accumulator. Retain
            # the sign of an IEEE zero product in the independent oracle.
            output[index] = float(left[tuple(li)]) * float(right[tuple(ri)])
            continue
        total = Fraction(0)
        for reduced in product(*(range(left.shape[a]) for a in lc)):
            for a, b, value in zip(lc, rc, reduced, strict=True):
                li[a] = ri[b] = value
            total += Fraction.from_float(float(left[tuple(li)])) * Fraction.from_float(
                float(right[tuple(ri)])
            )
        output[index] = float(total)
    return output


def _same(actual: Any, expected: Any) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.fixture(autouse=True)
def _bindings_restored() -> Iterator[None]:
    yield
    actual = (
        nnx.LinearGeneral.__call__,
        nn.DenseGeneral.__call__,
        jnp.einsum,
        nnx.Einsum.__call__,
        nn.Einsum.__call__,
    )
    assert all(
        left is right for left, right in zip(actual, _ORIGINAL_BINDINGS, strict=True)
    )


def _save_arrays(path: Path, arrays: Any) -> None:
    np.savez(path, **{f"out{i}": np.asarray(value) for i, value in enumerate(arrays)})


def observe_contraction(
    fn: Any,
    values: Any,
    directory: Path,
    *,
    expected: Any,
    opset: int = 23,
    diagnostic: bool = False,
    named_arguments: tuple[Any, Any] | None = None,
) -> tuple[Any, Any]:
    """Original native roles precede export; every report is saved before guards."""
    directory.mkdir(parents=True, exist_ok=True)
    before_context = _globals(), _settings()
    (directory / "criteria.json").write_text(json.dumps(asdict(_BITWISE)))
    frozen = tuple(np.asarray(value).copy() for value in values)
    _save_arrays(directory / "inputs.npz", frozen)
    if expected is not None:
        _save_arrays(directory / "expected-before-target.npz", expected)

    def compiled_role(*args: Any) -> Any:
        return fn(*args)

    def native_trace_role(*args: Any) -> Any:
        return fn(*args)

    def activated_trace_role(*args: Any) -> Any:
        return fn(*args)

    native = tuple(
        tuple(np.asarray(v).copy() for v in jax.tree.leaves(call(*values)))
        for call in (fn, jax.jit(compiled_role))
    )
    for role, outputs in zip(("eager", "fresh-jit"), native, strict=True):
        _save_arrays(directory / (role + ".npz"), outputs)
    specs = [
        jax.ShapeDtypeStruct(
            value.shape, value.dtype, weak_type=bool(jax.typeof(value).weak_type)
        )
        for value in values
    ]
    original = jax.make_jaxpr(native_trace_role)(*specs)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(activated_trace_role)(*specs)
    (directory / "native-trace.txt").write_text(str(original))
    (directory / "activated-trace.txt").write_text(str(activated))
    if named_arguments is None:
        assert typed_trace_manifest(original) == typed_trace_manifest(activated)
        native_dot = None
    else:
        native_dot = _named_call_contract(original, activated, named_arguments)
    with (
        _observe_functions() if named_arguments is not None else nullcontext([])
    ) as built:
        model = ui._to_onnx(
            fn, specs, opset=opset, precision_policy=PrecisionPolicy.SOURCE
        )
    (directory / "source.onnx").write_bytes(model.SerializeToString())
    if named_arguments is not None:
        _named_function_evidence(
            model, built, native_dot, named_arguments, native, directory
        )
    onnx.checker.check_model(model, full_check=True)
    provenance = get_precision_report(model)
    assert provenance is not None and provenance.profile == "source-static-v1"
    assert (
        next(c.status for c in provenance.checks if c.category == "lowering")
        is CheckStatus.VALIDATED
    )
    reports = []
    for mode, outputs in product(("disabled", "all"), native):
        reports.append(
            validate_source_samples(
                model,
                samples=[SourceSample(frozen, outputs, _BITWISE)],
                optimization=mode,
            )
        )
    observations = [asdict(report.samples[0]) for report in reports]
    (directory / "comparisons-before-guards.json").write_text(
        json.dumps(observations, indent=2)
    )
    if diagnostic:
        for report in reports:
            if report.status is CheckStatus.FAILED:
                with pytest.raises(SampleAcceptanceError):
                    report.require_sample_acceptance()
            else:
                report.require_sample_acceptance()
    else:
        for role, outputs in enumerate(native):
            require_target_pair(
                model,
                [SourceSample(frozen, outputs, _BITWISE)],
                reports[role],
                reports[len(native) + role],
                evidence_path=directory / f"target-boundary-{role}.json",
            )
    if expected is not None:
        for outputs in native:
            assert len(outputs) == len(expected)
            for actual, wanted in zip(outputs, expected, strict=True):
                _same(actual, wanted)
    for actual, before in zip(values, frozen, strict=True):
        _same(actual, before)
    assert (_globals(), _settings()) == before_context
    return model, observations


@pytest.mark.parametrize("dtype,x64", _WIDTHS)
@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize("case", _CASES)
def test_ordered_native_geometry_scalar_outer_and_empty(
    case: Any, dtype: str, x64: bool, opset: int, tmp_path: Path
) -> None:
    lhs, rhs, dimensions = case
    with jax.enable_x64(x64):
        values = _payload(dtype, lhs), _payload(dtype, rhs, right=True)
        preferred = np.float32 if dtype == "float16" else None

        def fn(a: Any, b: Any) -> Any:
            return jax.lax.dot_general(
                a, b, dimensions, preferred_element_type=preferred
            )

        observe_contraction(
            fn,
            values,
            tmp_path,
            expected=(
                _oracle(
                    *values, dimensions, np.float32 if preferred is not None else dtype
                ),
            ),
            opset=opset,
        )


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_no_reduction_keeps_negative_zero_while_zero_reduction_is_additive_identity(
    dtype: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        preferred = np.float32 if dtype == "float16" else None
        output_dtype = np.float32 if preferred is not None else dtype
        for name, values, dimensions, wanted in (
            (
                "outer",
                (np.asarray([-0.0], dtype), np.asarray([1.0], dtype)),
                (((), ()), ((), ())),
                np.asarray([[-0.0]], output_dtype),
            ),
            (
                "zero-reduction",
                (np.empty((1, 0), dtype), np.empty((0, 1), dtype)),
                _MATRIX,
                np.asarray([[0.0]], output_dtype),
            ),
        ):

            def fn(a: Any, b: Any) -> Any:
                return jax.lax.dot_general(
                    a, b, dimensions, preferred_element_type=preferred
                )

            observe_contraction(fn, values, tmp_path / name, expected=(wanted,))


def _named_call_contract(original: Any, activated: Any, arguments: Any) -> Any:
    """Prove the actual carrier boundary without pretending it is a native dot."""
    for closed in (original, activated):
        body = _body(closed)
        assert len(body.invars) == len(body.constvars) == len(closed.consts) == 1
        assert len(body.outvars) == 4 and not body.effects
        assert body.outvars[0] is body.outvars[1]
        assert body.outvars[2] is body.invars[0]
        assert body.outvars[3] is body.constvars[0]
        _same(closed.consts[0], arguments[1])
        assert len(body.eqns) == 1
        eqn = body.eqns[0]
        assert len(eqn.invars) == 2 and len(eqn.outvars) == 1
        assert eqn.invars[0] is body.invars[0]
        assert eqn.invars[1] is body.constvars[0]
        assert eqn.outvars[0] is body.outvars[0]
    while True:
        left, right = _body(original), _body(activated)
        for attribute in ("constvars", "invars", "outvars"):
            assert _signature(getattr(left, attribute)) == _signature(
                getattr(right, attribute)
            )
        assert not left.effects and not right.effects
        native, carrier = left.eqns[0], right.eqns[0]
        assert not native.effects and not carrier.effects
        assert _signature(native.invars) == _signature(carrier.invars)
        assert _signature(native.outvars) == _signature(carrier.outvars)
        if native.primitive.name not in {"jit", "pjit"}:
            break
        assert native.primitive is carrier.primitive
        left_params = typed_trace_manifest(original)["equations"][0]["params"]
        right_params = typed_trace_manifest(activated)["equations"][0]["params"]
        keys = set(native.params) & {"jaxpr", "call_jaxpr"}
        assert len(keys) == 1 and keys == set(carrier.params) & {"jaxpr", "call_jaxpr"}
        assert {k: v for k, v in left_params.items() if k not in keys} == {
            k: v for k, v in right_params.items() if k not in keys
        }
        key = next(iter(keys))
        original, activated = native.params[key], carrier.params[key]
        for closed in (original, activated):
            body = _body(closed)
            assert not body.constvars and not getattr(closed, "consts", ())
            assert len(body.invars) == 2 and len(body.outvars) == len(body.eqns) == 1
            eqn = body.eqns[0]
            assert len(eqn.invars) == 2 and len(eqn.outvars) == 1
            assert all(a is b for a, b in zip(eqn.invars, body.invars, strict=True))
            assert eqn.outvars[0] is body.outvars[0]
    assert native.primitive is jax.lax.dot_general_p
    assert (
        carrier.primitive.name == f"onnx_fn::{_named.__module__}.{_named.__qualname__}"
    )
    assert set(carrier.params) == {"__jax2onnx_source_trace"}
    trace = carrier.params["__jax2onnx_source_trace"]
    assert type(trace) is _SourceFunctionTrace
    assert tuple(
        (np.dtype(v.dtype).name, tuple(v.shape), bool(v.weak_type))
        for v in trace.output_avals
    ) == _signature(native.outvars)
    assert (
        trace.x64_enabled,
        trace.default_matmul_precision,
        trace.numpy_dtype_promotion,
        trace.numpy_rank_promotion,
        trace.debug_key_reuse,
    ) == (
        bool(jax.config.x64_enabled),
        jax.config.jax_default_matmul_precision,
        jax.config.jax_numpy_dtype_promotion,
        jax.config.jax_numpy_rank_promotion,
        bool(jax.config.jax_debug_key_reuse),
    )
    _validate(native)
    return native


def _named_function_evidence(
    model: Any,
    built: Any,
    native_dot: Any,
    arguments: Any,
    references: Any,
    directory: Path,
) -> None:
    assert len(built) == len(model.functions) == 1
    definition, closed = built[0]
    function = model.functions[0]
    assert (function.domain, function.name) == (definition.domain, definition.name)
    before = typed_trace_manifest(closed)
    constants = tuple(_array_payload(value) for value in closed.consts)
    (directory / "actual-function-body-trace.json").write_text(
        json.dumps(before, indent=2)
    )
    body = _body(closed)
    assert not body.constvars and not closed.consts and not body.effects
    assert len(body.invars) == 2 and len(body.outvars) == len(body.eqns) == 1
    actual_dot = body.eqns[0]
    assert actual_dot.primitive is native_dot.primitive is jax.lax.dot_general_p
    assert actual_dot.params == native_dot.params
    assert actual_dot.effects == native_dot.effects
    assert _signature(actual_dot.invars) == _signature(native_dot.invars)
    assert _signature(actual_dot.outvars) == _signature(native_dot.outvars)
    assert len(actual_dot.invars) == 2 and len(actual_dot.outvars) == 1
    assert all(a is b for a, b in zip(actual_dot.invars, body.invars, strict=True))
    assert actual_dot.outvars[0] is body.outvars[0]
    _validate(actual_dot)
    evaluator = jaxpr_as_fun(closed)

    def compiled_body(*args: Any) -> Any:
        return evaluator(*args)

    outputs = tuple(
        tuple(np.asarray(v).copy() for v in runner(*arguments))
        for runner in (evaluator, jax.jit(compiled_body))
    )
    for role, actual in zip(("eager", "fresh-jit"), outputs, strict=True):
        _save_arrays(directory / ("actual-function-body-" + role + ".npz"), actual)
        assert len(actual) == 1
        for reference in references:
            _same(actual[0], reference[0])
    assert typed_trace_manifest(closed) == before
    assert tuple(_array_payload(value) for value in closed.consts) == constants
    infos = {v.name: v for v in function.value_info}
    for names, variables in (
        (function.input, body.invars),
        (function.output, body.outvars),
    ):
        assert len(names) == len(variables)
        for name, variable in zip(names, variables, strict=True):
            _port(infos[name], variable.aval.dtype, variable.aval.shape)
    assert len(model.graph.input) == 1 and len(model.graph.output) == 4
    _port(model.graph.input[0], arguments[0].dtype, arguments[0].shape)
    for port, reference in zip(model.graph.output, references[0], strict=True):
        _port(port, reference.dtype, reference.shape)
    calls = [node for node in model.graph.node if node.domain]
    assert len(calls) == 1
    call = calls[0]
    assert (call.domain, call.op_type) == (function.domain, function.name)
    assert len(call.input) == 2 and len(call.output) == 1
    assert all(
        (node.domain, node.op_type) == (function.domain, function.name)
        or (not node.domain and node.op_type in {"Identity", "Constant"})
        for node in model.graph.node
    )
    assert _origin(model.graph.node, call.input[0]) == model.graph.input[0].name
    assert _origin(model.graph.node, model.graph.output[0].name) == call.output[0]
    assert _origin(model.graph.node, model.graph.output[1].name) == call.output[0]
    assert (
        _origin(model.graph.node, model.graph.output[2].name)
        == model.graph.input[0].name
    )
    kernel = _origin(model.graph.node, call.input[1])
    assert _origin(model.graph.node, model.graph.output[3].name) == kernel
    arrays = {
        value.name: onnx.numpy_helper.to_array(value)
        for value in model.graph.initializer
    }
    for node in model.graph.node:
        if not node.domain and node.op_type == "Constant":
            assert len(node.output) == 1
            arrays[node.output[0]] = onnx.numpy_helper.to_array(
                next(a.t for a in node.attribute if a.name == "value")
            )
    _same(arrays[kernel], references[0][3])
    _same(arguments[0], references[0][2])
    _same(arguments[1], references[0][3])
    operations = [node.op_type for node in function.node]
    assert set(operations) <= {"Identity", "Constant", "Transpose", "Reshape", "MatMul"}
    assert all(not node.domain for node in function.node)
    assert operations.count("MatMul") == 1
    dot = next(node for node in function.node if node.op_type == "MatMul")
    assert _origin(function.node, function.output[0]) == dot.output[0]
    producers = {output: node for node in function.node for output in node.output}
    for name, formal in zip(dot.input, function.input, strict=True):
        visited = set()
        name = _origin(function.node, name)
        while name != formal:
            assert name not in visited
            visited.add(name)
            node = producers[name]
            assert node.op_type in {"Transpose", "Reshape"}
            assert len(node.output) == 1
            name = _origin(function.node, node.input[0])


@onnx_function(name="GeneralizedNativeContraction")
def _named(a: Any, b: Any) -> Any:
    return jax.lax.dot_general(a, b, (((2, 1), (0, 1)), ((), ())))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("compiled", [False, True])
def test_named_captured_kernel_aliases_and_jit(
    dtype: str, compiled: bool, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        left, right = _payload(dtype, (2, 3, 4)), _payload(dtype, (4, 3, 5), right=True)

        def fn(a: Any) -> Any:
            output = jax.jit(_named)(a, right) if compiled else _named(a, right)
            return output, output, a, right

        wanted = _oracle(left, right, _CASES[0][2], dtype)
        observe_contraction(
            fn,
            (left,),
            tmp_path,
            expected=(wanted, wanted, left, right),
            named_arguments=(left, right),
        )


def _wrapper(plugin: Any, original: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    if plugin is JnpEinsumPlugin:
        slot = "__orig_impl__einsum"
        monkeypatch.setitem(vars(plugin._PRIM), slot, vars(plugin._PRIM).get(slot))
    else:
        slot = "_ORIG_CALL" if plugin is EinsumModulePlugin else "_ORIGINAL_CALL"
        monkeypatch.setattr(plugin, slot, getattr(plugin, slot))
    spec = next(s for s in plugin.binding_specs() if isinstance(s, MonkeyPatchSpec))
    return spec.make_value(original)


@pytest.mark.parametrize("plugin", _FORWARDERS)
def test_exact_source_forwarding_keeps_argument_and_error_identity(
    plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    tokens = tuple(object() for _ in range(5))
    failure = ValueError("native callback failure")

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        if kwargs.get("fail"):
            raise failure
        return tokens[-1]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("SOURCE reached the private replacement primitive")

    monkeypatch.setitem(vars(plugin._PRIM), "bind", forbidden)
    wrapper = _wrapper(plugin, original, monkeypatch)
    options = dict(
        out_sharding=tokens[0],
        precision=None,
        preferred_element_type=tokens[1],
        _dot_general=tokens[2],
        optimize=[(0, 1)],
        einsum_str=tokens[3],
    )
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        assert wrapper(*tokens[:2], **options) is tokens[-1]
        with pytest.raises(ValueError) as caught:
            wrapper(*tokens[:2], **options, fail=True)
    assert caught.value is failure
    assert calls == [(tokens[:2], options), (tokens[:2], {**options, "fail": True})]
    assert all(calls[0][1][key] is value for key, value in options.items())


def _native_module(kind: int) -> tuple[Any, Any, Any]:
    if kind == 0:
        layer = nnx.LinearGeneral((3,), (2,), dtype=np.float32, rngs=nnx.Rngs(0))
    elif kind == 1:
        template = nn.DenseGeneral((2,), dtype=np.float32)
        layer = template.bind(
            template.init(jax.random.key(0), jnp.ones((2, 3), np.float32))
        )
    elif kind == 3:
        layer = nnx.Einsum("ij,jk->ik", (3, 2), dtype=np.float32, rngs=nnx.Rngs(0))
    else:
        template = nn.Einsum((3, 2), einsum_str="ij,jk->ik", dtype=np.float32)
        layer = template.bind(
            template.init(jax.random.key(0), jnp.ones((2, 3), np.float32))
        )
    return layer, jnp.asarray(_payload(np.float32, (2, 3))), _ORIGINAL_BINDINGS[kind]


@pytest.mark.parametrize("kind", range(5))
@pytest.mark.parametrize(
    "form",
    [
        "omitted",
        "keyword-input",
        "positional-none",
        "keyword-none",
        "placement-object",
        "duplicate-input",
        "unknown",
        "equation-override",
    ],
)
def test_actual_original_native_call_forms_and_old_errors(
    kind: int, form: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if kind == 2:
        original = _ORIGINAL_BINDINGS[kind]
        a = jnp.asarray(_payload(np.float32, (2, 3)))
        b = jnp.asarray(_payload(np.float32, (3, 2), right=True))
        args = ("ij,jk->ik", a, b)
        kwargs = {}
        state = ()
    else:
        layer, value, original = _native_module(kind)
        args = (layer, value)
        kwargs = {}
        parameters = (
            (layer.kernel[...],) + (() if layer.bias is None else (layer.bias[...],))
            if kind in (0, 3)
            else jax.tree.leaves(layer.variables)
        )
        state = tuple(np.asarray(v).copy() for v in parameters)
    if form == "keyword-input":
        if kind == 2:
            kwargs = {"subscripts": "ij,jk->ik"}
            args = args[1:]
        else:
            kwargs = {"inputs": args[-1]}
            args = args[:-1]
    elif form == "positional-none":
        args = (*args, None)
    elif form in {"keyword-none", "placement-object"}:
        kwargs = {"out_sharding": None if form == "keyword-none" else object()}
    elif form == "duplicate-input":
        kwargs = {"inputs": args[-1]}
    elif form == "unknown":
        kwargs = {"unreviewed_option": object()}
    elif form == "equation-override":
        kwargs = {"einsum_str": "ij,jk->ik"}
    wrapper = _wrapper(_FORWARDERS[kind], original, monkeypatch)
    outcomes = {}
    arrays = {}
    for role, call in (("native", original), ("source", wrapper)):
        try:
            if role == "native":
                output = call(*args, **kwargs)
            else:
                with precision_policy_scope(PrecisionPolicy.SOURCE):
                    output = call(*args, **kwargs)
            arrays[role] = np.asarray(output).copy()
            outcomes[role] = {
                "status": "returned",
                "dtype": arrays[role].dtype.name,
                "shape": arrays[role].shape,
            }
            _save_arrays(tmp_path / (role + ".npz"), (arrays[role],))
        except Exception as exc:
            outcomes[role] = {
                "status": "raised",
                "type": type(exc).__name__,
                "detail": str(exc),
            }
    (tmp_path / "original-native-call-evidence.json").write_text(
        json.dumps({"form": form, "outcomes": outcomes}, indent=2)
    )
    if form == "omitted":
        assert outcomes["native"]["status"] == "returned"
    assert outcomes["source"] == outcomes["native"]
    if arrays:
        _same(arrays["native"], arrays["source"])
    if kind != 2:
        after = (
            (layer.kernel[...],) + (() if layer.bias is None else (layer.bias[...],))
            if kind in (0, 3)
            else jax.tree.leaves(layer.variables)
        )
        for actual, before in zip(after, state, strict=True):
            _same(actual, before)


def _equation(dtype: Any = np.float64) -> Any:
    with jax.enable_x64(True):
        result = jax.make_jaxpr(lambda a, b: jax.lax.dot_general(a, b, _CASES[0][2]))(
            jax.ShapeDtypeStruct((2, 3, 4), dtype),
            jax.ShapeDtypeStruct((4, 3, 5), dtype),
        ).jaxpr.eqns[0]
    return SimpleNamespace(
        primitive=result.primitive,
        params=dict(result.params),
        invars=list(result.invars),
        outvars=list(result.outvars),
        effects=result.effects,
        ctx=result.ctx,
    )


def _validate(eqn: Any) -> None:
    site = _Site("contraction-negative", 23, "dot_general")
    _validate_dot(
        eqn,
        [_tensor(v, site) for v in eqn.invars],
        [_tensor(v, site) for v in eqn.outvars],
        site,
    )


@pytest.mark.parametrize(
    "fault",
    [
        "identity",
        "context",
        "effect",
        "dtype",
        "shape",
        "weak",
        "weak-int",
        "pairing",
        "unknown",
        "placement",
        "algorithm",
    ],
)
def test_native_contraction_rejects_forged_logical_contract(fault: str) -> None:
    eqn = _equation()
    if fault == "identity":
        eqn.primitive = Primitive("dot_general")
    elif fault == "context":
        eqn.ctx = SimpleNamespace(manager=object())
    elif fault == "effect":
        eqn.effects = {object()}
    elif fault in {"dtype", "shape", "weak", "weak-int"}:
        aval = eqn.outvars[0].aval
        aval = (
            aval.update(dtype=np.float32)
            if fault == "dtype"
            else (
                aval.update(shape=(5, 2))
                if fault == "shape"
                else aval.update(weak_type=True)
            )
        )
        if fault == "weak-int":
            aval = _forged_aval(aval, weak_type=1)
        eqn.outvars = [Var(aval)]
    elif fault == "pairing":
        eqn.params["dimension_numbers"] = (((2, 1), (1, 0)), ((), ()))
    elif fault in {"unknown", "placement"}:
        eqn.params["out_dtype" if fault == "unknown" else "out_sharding"] = object()
    else:
        eqn.params["precision"] = jax.lax.DotAlgorithmPreset.F64_F64_F64
    with pytest.raises(SourcePrecisionError):
        _validate(eqn)


def test_canonical_dot_inference_cannot_be_replaced_by_current_method_or_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eqn = _equation()
    original_outputs = list(eqn.outvars)
    calls = []

    def replacement(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return eqn.outvars[0].aval, frozenset()

    with monkeypatch.context() as patch:
        patch.setitem(vars(eqn.primitive), "abstract_eval", replacement)
        _validate(eqn)
        eqn.outvars = [Var(original_outputs[0].aval.update(shape=(5, 2)))]
        with pytest.raises(SourcePrecisionError):
            _validate(eqn)
        eqn.outvars = original_outputs
        assert not calls
    _validate(eqn)
    with monkeypatch.context() as patch:
        # The original native evaluator itself consults this invariant. It must
        # fail closed, rather than letting a mutable flag forge its result form.
        patch.setitem(vars(eqn.primitive), "multiple_results", True)
        with pytest.raises(SourcePrecisionError):
            _validate(eqn)
    _validate(eqn)


@pytest.mark.parametrize("ordinal", range(11))
@pytest.mark.parametrize("x64", [False, True])
def test_original_dot_rows_keep_recipes_callbacks_options_and_missing_criteria(
    ordinal: int, x64: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Original callback failures remain failures in saved product-stage results."""
    metadata = source.raw_source_case("plugin::dot_general", ordinal)
    original = metadata["callable"]
    callback = metadata.get("post_check_onnx_graph")
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance("plugin::dot_general", ordinal)
    )[int(x64)]
    errors, native = {}, {}
    captured = []
    original_export = source._export_prepared

    def save_before_original_callback(*args: Any, **kwargs: Any) -> Any:
        model = original_export(*args, **kwargs)
        serialized = model.SerializeToString()
        (tmp_path / "original-product.onnx").write_bytes(serialized)
        captured.append(serialized)
        return model

    monkeypatch.setattr(source, "_export_prepared", save_before_original_callback)
    result = None
    with source.source_configuration(scenario):
        prepared = source.prepare_source_case(metadata, scenario)
        inputs = tuple(np.asarray(v).copy() for v in prepared.arguments)
        np.savez(
            tmp_path / "original-feeds.npz",
            **{f"in{i}": v for i, v in enumerate(inputs)},
        )
        native["eager"] = prepared.reference_outputs
        try:

            def fresh(*args: Any) -> Any:
                return prepared.fn(*args)

            native["fresh-jit"] = tuple(
                np.asarray(v).copy()
                for v in jax.tree.leaves(jax.jit(fresh)(*prepared.arguments))
            )
        except Exception as exc:
            errors["fresh-jit"] = f"{type(exc).__name__}: {exc}"
            native["fresh-jit"] = ()
        for role, outputs in native.items():
            np.savez(
                tmp_path / (role + ".npz"),
                **{f"out{i}": v for i, v in enumerate(outputs)},
            )
        try:
            result = source.run_source_case(
                metadata, scenario, optimization=None, _prepared=prepared
            )
            if result.model is not None:
                (tmp_path / "original-product.onnx").write_bytes(
                    result.model.SerializeToString()
                )
        except Exception as exc:
            errors["product"] = f"{type(exc).__name__}: {exc}"
        (tmp_path / "original-product-observations.json").write_text(
            json.dumps(
                {
                    "errors": errors,
                    "criteria": None,
                    "opset": metadata.get("opset_version", 23),
                    "stage": None if result is None else result.stage,
                    "audit": None if result is None else asdict(result.audit),
                    "raw_callback_status": (
                        None if result is None else result.raw_callback_status
                    ),
                    "validation_checks": (
                        []
                        if result is None or result.validation is None
                        else [asdict(c) for c in result.validation.checks]
                    ),
                    "runtime_and_numerics": "UNVERIFIED; original recipe has no criterion and this replay executes no ORT",
                },
                indent=2,
            )
        )
    assert not errors, errors
    assert result is not None
    if result.raw_callback_status == "failed":
        assert len(captured) == 1 and result.audit.status is CheckStatus.FAILED
        assert (tmp_path / "original-product.onnx").read_bytes() == captured[0]
    assert scenario.criteria is None and prepared.criteria is None
    if result.validation is not None:
        assert result.validation.target is None and not result.validation.samples
        assert result.validation.status is CheckStatus.UNVERIFIED
    assert result.numerical_status is CheckStatus.UNVERIFIED
    assert metadata.get("opset_version", 23) == 23
    assert (
        source.raw_source_case("plugin::dot_general", ordinal)["callable"] is original
    )
    assert (
        source.raw_source_case("plugin::dot_general", ordinal).get(
            "post_check_onnx_graph"
        )
        is callback
    )
    for actual, saved in zip(prepared.arguments, inputs, strict=True):
        _same(actual, saved)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("form", ["string", "indexed", "multioperand"])
def test_native_einsum_forms_keep_decomposition_and_original_kwargs(
    dtype: str, form: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        a, b, c = (
            _payload(dtype, (2, 3)),
            _payload(dtype, (3, 4), right=True),
            _payload(dtype, (4, 2), right=True),
        )
        calls = []

        def custom(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return jax.lax.dot_general(*args, **kwargs)

        def fn(left: Any, right: Any, *tail: Any) -> Any:
            options = dict(
                optimize=[(0, 1), (0, 1)] if form == "multioperand" else False,
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=np.dtype(dtype),
                _dot_general=custom,
                out_sharding=None,
            )
            if form == "indexed":
                return jnp.einsum(left, [0, 1], right, [1, 2], [0, 2], **options)
            if form == "multioperand":
                return jnp.einsum("ij,jk,kl->il", left, right, tail[0], **options)
            return jnp.einsum("ij,jk->ik", left, right, **options)

        first = _oracle(a, b, _MATRIX, dtype)
        wanted = _oracle(first, c, _MATRIX, dtype) if form == "multioperand" else first
        values = (a, b, c) if form == "multioperand" else (a, b)
        model, _ = observe_contraction(fn, values, tmp_path, expected=(wanted,))
        assert calls
        assert sum(n.op_type == "MatMul" for n in model.graph.node) == (
            2 if form == "multioperand" else 1
        )
        assert not any(n.op_type == "Einsum" for n in model.graph.node)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("opset", [23, 26])
def test_native_einsum_retains_preferred_fp32_and_fp16_storage(
    x64: bool, opset: int, tmp_path: Path
) -> None:
    with jax.enable_x64(x64):
        values = _payload(np.float16, (2, 3, 4)), _payload(
            np.float16, (4, 3, 5), right=True
        )

        def fn(left: Any, right: Any) -> Any:
            return jnp.einsum(
                "abc,cbd->ad",
                left,
                right,
                preferred_element_type=np.float32,
                precision=jax.lax.Precision.DEFAULT,
                optimize=False,
            )

        observe_contraction(
            fn,
            values,
            tmp_path,
            expected=(_oracle(*values, _CASES[0][2], np.float32),),
            opset=opset,
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "kind", ["nnx-general", "linen-general", "nnx-einsum", "linen-einsum"]
)
def test_real_modules_keep_batch_axes_dtype_bias_parameter_bytes_and_hooks(
    dtype: str, kind: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        value = _payload(dtype, (2, 5, 2, 3))
        kernel = _payload(dtype, (2, 2, 3, 2, 4), right=True)
        bias = _payload(dtype, (2, 2, 4), right=True)
        dimensions = (((2, 3), (1, 2)), ((0,), (0,)))
        hook_calls = []

        def custom(*args: Any, **kwargs: Any) -> Any:
            hook_calls.append((args, kwargs))
            return jax.lax.dot_general(*args, **kwargs)

        if kind == "nnx-general":
            layer = nnx.LinearGeneral(
                (2, 3),
                (2, 4),
                axis=(-2, -1),
                batch_axis={0: 2},
                dtype=np.dtype(dtype),
                param_dtype=np.dtype(dtype),
                precision=jax.lax.Precision.DEFAULT,
                dot_general=custom,
                rngs=nnx.Rngs(0),
            )
            layer.kernel[...] = jnp.asarray(kernel)
            layer.bias[...] = jnp.asarray(bias)

            def fn(x) -> Any:
                return layer(inputs=x)

        elif kind == "linen-general":
            layer = nn.DenseGeneral(
                (2, 4),
                axis=(-2, -1),
                batch_dims=(0,),
                dtype=np.dtype(dtype),
                param_dtype=np.dtype(dtype),
                precision=jax.lax.Precision.DEFAULT,
                dot_general=custom,
            )
            parameters = {
                "params": {"kernel": jnp.asarray(kernel), "bias": jnp.asarray(bias)}
            }

            def fn(x) -> Any:
                return layer.apply(parameters, inputs=x)

        else:
            value = _payload(dtype, (2, 3, 4))
            kernel = _payload(dtype, (5, 4, 2), right=True)
            bias = _payload(dtype, (5, 2), right=True)
            dimensions = (((2,), (1,)), ((), ()))
            if kind == "nnx-einsum":
                layer = nnx.Einsum(
                    "nta,hab->nthb",
                    (5, 4, 2),
                    bias_shape=(5, 2),
                    dtype=np.dtype(dtype),
                    param_dtype=np.dtype(dtype),
                    precision=jax.lax.Precision.DEFAULT,
                    rngs=nnx.Rngs(0),
                )
                layer.kernel[...] = jnp.asarray(kernel)
                layer.bias[...] = jnp.asarray(bias)

                def fn(x) -> Any:
                    return layer(x)

            else:
                layer = nn.Einsum(
                    (5, 4, 2),
                    einsum_str="nta,hab->nthb",
                    dtype=np.dtype(dtype),
                    param_dtype=np.dtype(dtype),
                    precision=jax.lax.Precision.DEFAULT,
                )
                parameters = {
                    "params": {"kernel": jnp.asarray(kernel), "bias": jnp.asarray(bias)}
                }

                def fn(x) -> Any:
                    return layer.apply(parameters, x, einsum_str=None)

        frozen = kernel.tobytes(), bias.tobytes()
        wanted = _oracle(value, kernel, dimensions, dtype)
        if "einsum" in kind:
            wanted = wanted + bias.reshape(1, 1, 5, 2)
        else:
            wanted = wanted + bias.reshape(2, 1, 2, 4)
        observe_contraction(fn, (value,), tmp_path, expected=(wanted,))
        if kind in {"nnx-general", "linen-general"}:
            assert hook_calls
        if kind.startswith("nnx"):
            _same(layer.kernel[...], kernel)
            _same(layer.bias[...], bias)
        else:
            _same(parameters["params"]["kernel"], kernel)
            _same(parameters["params"]["bias"], bias)
        assert (kernel.tobytes(), bias.tobytes()) == frozen


@pytest.mark.parametrize("fault", ["dtype", "shape", "missing"])
def test_generalized_output_descriptor_rejects_before_emission(fault: str) -> None:
    eqn = _equation()
    sources = [
        ir.val(f"input{i}", dtype=ir.DataType.DOUBLE, shape=v.aval.shape)
        for i, v in enumerate(eqn.invars)
    ]
    output = ir.val("output", dtype=ir.DataType.DOUBLE, shape=(2, 5))
    if fault == "dtype":
        output.type = ir.TensorType(ir.DataType.FLOAT)
    else:
        output.shape = None if fault == "missing" else ir.Shape((5, 2))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid output metadata reached emission/binding")

    before = [(v.type, v.shape) for v in sources]
    ctx = SimpleNamespace(
        deployment_target=None,
        builder=SimpleNamespace(opset=23, op=forbidden),
        require_value_for_var=lambda var: sources[eqn.invars.index(var)],
        allocate_value_for_var=lambda *args, **kwargs: output,
        bind_value_for_var=forbidden,
        fresh_name=forbidden,
    )
    with pytest.raises(ValueError, match="physical matrix dtype/shape"):
        DotGeneralPlugin()._lower_source(ctx, eqn)
    assert before == [(v.type, v.shape) for v in sources]


@pytest.mark.parametrize(
    "shapes,dimensions,output",
    [
        (((2, 0), (0, 3)), _MATRIX, (2, 3)),
        (((0, 2), (2, 3)), _MATRIX, (0, 3)),
        (((), (2, 3)), (((), ()), ((), ())), (2, 3)),
    ],
)
def test_empty_and_outer_shortcuts_do_not_discard_explicit_algorithm_contracts(
    shapes: Any, dimensions: Any, output: Any
) -> None:
    eqn = _equation()
    eqn.invars = [
        Var(v.aval.update(shape=shape, sharding=None))
        for v, shape in zip(eqn.invars, shapes, strict=True)
    ]
    eqn.outvars = [Var(eqn.outvars[0].aval.update(shape=output, sharding=None))]
    eqn.params["dimension_numbers"] = dimensions
    eqn.params["precision"] = jax.lax.DotAlgorithmPreset.F64_F64_F64
    with pytest.raises(SourcePrecisionError):
        _validate(eqn)
    with pytest.raises(NotImplementedError, match="precision .*private subset"):
        DotGeneralPlugin()._lower_source(SimpleNamespace(deployment_target=None), eqn)


@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("fault", ["dtype", "shape", "missing"])
def test_physical_input_drift_rejects_before_allocation_or_emission(
    operand: int, fault: str
) -> None:
    eqn = _equation()
    values = [
        ir.val(f"input{i}", dtype=ir.DataType.DOUBLE, shape=v.aval.shape)
        for i, v in enumerate(eqn.invars)
    ]
    if fault == "dtype":
        values[operand].type = ir.TensorType(ir.DataType.FLOAT)
    else:
        values[operand].shape = (
            None
            if fault == "missing"
            else ir.Shape(tuple(reversed(tuple(values[operand].shape))))
        )
    before = [(v.type, v.shape) for v in values]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid physical input reached allocation/emission")

    ctx = SimpleNamespace(
        deployment_target=None,
        builder=SimpleNamespace(opset=23, op=forbidden),
        require_value_for_var=lambda var: values[eqn.invars.index(var)],
        allocate_value_for_var=forbidden,
        bind_value_for_var=forbidden,
        fresh_name=forbidden,
    )
    with pytest.raises(ValueError, match="physical matrix dtype/shape"):
        DotGeneralPlugin()._lower_source(ctx, eqn)
    assert before == [(v.type, v.shape) for v in values]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "weak", [(False, False), (False, True), (True, False), (True, True)]
)
def test_native_weak_contraction_inputs_and_result_are_preserved(
    dtype: str, weak: Any, tmp_path: Path
) -> None:
    with jax.enable_x64(dtype == "float64"):
        # Construct real weak arrays through public native broadcast, rather
        # than reconstructing avals or relying on private conversion parameters.
        values = tuple(
            jax.lax.broadcast_in_dim(
                (
                    jnp.asarray(0.25 if index == 0 else -0.5)
                    if flag
                    else jnp.asarray(np.asarray(0.25 if index == 0 else -0.5, dtype))
                ),
                shape,
                (),
            )
            for index, (shape, flag) in enumerate(
                zip(((2, 3, 4), (4, 3, 5)), weak, strict=True)
            )
        )
        assert tuple(bool(jax.typeof(value).weak_type) for value in values) == weak

        def fn(a: Any, b: Any) -> Any:
            return jax.lax.dot_general(a, b, _CASES[0][2])

        expected_weak = bool(jax.typeof(fn(*values)).weak_type)
        model, _ = observe_contraction(
            fn, values, tmp_path, expected=(_oracle(*values, _CASES[0][2], dtype),)
        )
        report = get_precision_report(model)
        assert report is not None
        assert tuple(entry.traced.weak_type for entry in report.inputs) == weak
        assert report.traced_outputs[0].weak_type is expected_weak


@pytest.mark.parametrize("direct", [False, True])
def test_shape_only_grouping_limit_is_an_implementation_limit_not_bad_source(
    direct: bool,
) -> None:
    left = (2**32, 2**32, 2)
    right = (2, 1) if direct else (1, 2)
    dimensions = (((2,), (0 if direct else 1,)), ((), ()))

    def fn(a: Any, b: Any) -> Any:
        return jax.lax.dot_general(a, b, dimensions)

    equation = jax.make_jaxpr(fn)(
        jax.ShapeDtypeStruct(left, np.float32), jax.ShapeDtypeStruct(right, np.float32)
    ).jaxpr.eqns[0]
    if direct:
        _validate(equation)
    else:
        with pytest.raises(SourcePrecisionError) as caught:
            _validate(equation)
        assert caught.value.category == "profile-exclusion"
        assert "another lowering strategy remains unfinished" in str(caught.value)
