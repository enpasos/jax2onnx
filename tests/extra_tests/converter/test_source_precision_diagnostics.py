# tests/extra_tests/converter/test_source_precision_diagnostics.py

"""The diagnostic callback checks explicit structures without certifying lowering."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx import onnx_function
from jax2onnx._compat.jax import Primitive, ShapedArray, Var
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.plugins.plugin_system import PLUGIN_REGISTRY, import_all_plugins
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter.source_precision_diagnostics import DiagnosticValidator
from tests.extra_tests.converter.source_precision_support import (
    SourceScenario,
    run_source_case,
)


@onnx_function
def _diagnostic_supported_function(x):
    return x + np.float32(1)


@onnx_function
def _diagnostic_unreviewed_function(x):
    return jax.lax.exp(x)


@pytest.fixture(autouse=True)
def _bounded_context(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import onnxruntime

    previous = bool(jax.config.x64_enabled), jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", None)
    import_all_plugins()

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("Structural diagnostic tests must not create ORT sessions")

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous[0])
        jax.config.update("jax_default_matmul_precision", previous[1])


def _trace(fn, shape=(2, 3), dtype=np.float32):
    return jax.make_jaxpr(fn)(jax.ShapeDtypeStruct(shape, dtype))


@pytest.mark.parametrize(
    "function",
    [
        lambda x: x + np.float32(2),
        lambda x: jnp.sin(x) + jnp.cos(x),
        lambda x: jnp.reshape(jnp.transpose(x), (6,)),
        lambda x: jnp.sum(x, axis=0),
        lambda x: x @ x.T,
        lambda x: x[jnp.asarray([0, 1], np.int32)],
        lambda x: jax.lax.complex(x, x),
    ],
)
def test_explicit_ordinary_families_check_real_abstract_result_signatures(
    function,
) -> None:
    observer = DiagnosticValidator()
    observer(_trace(function), 23)
    assert observer.scopes and observer.failure is None


@pytest.mark.parametrize("kind", ["jit", "cond", "while", "scan", "fori"])
def test_real_nested_scopes_match_roles_and_terminate_on_both_jax_versions(
    kind: str,
) -> None:
    functions = {
        "jit": lambda x: jax.jit(lambda y: y + np.float32(1))(x),
        "cond": lambda x: jax.lax.cond(
            jnp.sum(x) > 0, lambda y: y + 1, lambda y: y - 1, x
        ),
        "while": lambda x: jax.lax.while_loop(
            lambda y: jnp.sum(y) < 5, lambda y: y + 1, x
        ),
        "scan": lambda x: jax.lax.scan(
            lambda c, y: (c + y, c), jnp.zeros((3,), x.dtype), x
        ),
        "fori": lambda x: jax.lax.fori_loop(0, 2, lambda i, y: y + 1, x),
    }
    observer = DiagnosticValidator()
    observer(_trace(functions[kind]), 23)
    assert len(observer.scopes) >= 2
    assert observer.failure is None


def test_valid_original_jax_batched_while_remains_pending_without_an_export() -> None:
    def original(current, limit):
        return jax.lax.while_loop(
            lambda value: value < limit,
            lambda value: value + np.int32(1),
            current,
        )

    function = jax.vmap(original)
    current = np.asarray([0, 1], np.int32)
    limit = np.asarray([2, 3], np.int32)
    np.testing.assert_array_equal(function(current, limit), limit)
    closed = jax.make_jaxpr(function)(
        jax.ShapeDtypeStruct((2,), np.int32),
        jax.ShapeDtypeStruct((2,), np.int32),
    )
    observer = DiagnosticValidator()
    with pytest.raises(SourcePrecisionError, match="batched while predicate") as caught:
        observer(closed, 23)
    assert caught.value.category == "profile-exclusion"
    result = run_source_case(
        {"callable": function, "input_values": [current, limit]},
        SourceScenario("batched-while", False),
        optimization=None,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
    )
    assert result.model is None
    assert result.category == "migration-gap"
    assert result.audit.status is CheckStatus.UNVERIFIED
    assert "batched while predicate" in result.audit.detail


@pytest.mark.parametrize("corruption", ["predicate-dtype", "body-shape", "counts"])
def test_known_malformed_scalar_while_signatures_still_fail(corruption: str) -> None:
    closed = _trace(
        lambda x: jax.lax.while_loop(lambda y: jnp.sum(y) < 5, lambda y: y + 1, x)
    )
    outer = closed.jaxpr.eqns[0]
    params = dict(outer.params)
    if corruption == "counts":
        params["cond_nconsts"] = -1
    else:
        key = "cond_jaxpr" if corruption == "predicate-dtype" else "body_jaxpr"
        child = params[key]
        body = child.jaxpr
        bad = Var(
            ShapedArray((), np.int32)
            if corruption == "predicate-dtype"
            else ShapedArray((3, 2), np.float32)
        )
        params[key] = SimpleNamespace(
            jaxpr=body.replace(outvars=[bad]), consts=getattr(child, "consts", ())
        )
    with pytest.raises(SourcePrecisionError) as caught:
        DiagnosticValidator()(
            closed.jaxpr.replace(eqns=[outer.replace(params=params)]), 23
        )
    assert caught.value.category == "lowering-defect"


def test_registered_unknown_operation_and_unknown_parameter_form_remain_pending() -> (
    None
):
    closed = _trace(lambda x: x + 1)
    eqn = closed.jaxpr.eqns[0]
    unknown = Primitive("registered_but_unreviewed")
    observer = DiagnosticValidator({**PLUGIN_REGISTRY, unknown.name: object()})
    graph = closed.jaxpr.replace(eqns=[eqn.replace(primitive=unknown)])
    with pytest.raises(SourcePrecisionError) as caught:
        observer(graph, 23)
    assert caught.value.category == "profile-exclusion"
    assert "explicit diagnostic structural family" in str(caught.value)
    graph = closed.jaxpr.replace(eqns=[eqn.replace(params={"unreviewed": 1})])
    with pytest.raises(SourcePrecisionError) as caught:
        DiagnosticValidator()(graph, 23)
    assert caught.value.category == "profile-exclusion"


@pytest.mark.parametrize(
    "corruption", ["unbound", "redefined", "dtype", "arity", "effects"]
)
def test_known_structure_corruption_never_becomes_a_validated_diagnostic(
    corruption: str,
) -> None:
    closed = _trace(lambda x: x + 1)
    graph = closed.jaxpr
    eqn = graph.eqns[0]
    if corruption == "unbound":
        eqn = eqn.replace(invars=[Var(eqn.invars[0].aval), *eqn.invars[1:]])
    elif corruption == "redefined":
        eqn = eqn.replace(outvars=[graph.invars[0]])
    elif corruption == "dtype":
        bad = Var(ShapedArray((2, 3), np.int32))
        eqn = eqn.replace(outvars=[bad])
        graph = graph.replace(outvars=[bad])
    elif corruption == "arity":
        eqn = eqn.replace(invars=eqn.invars[:1])
    elif corruption == "effects":
        eqn = eqn.replace(effects={object()})
    with pytest.raises(SourcePrecisionError) as caught:
        DiagnosticValidator()(graph.replace(eqns=[eqn]), 23)
    assert caught.value.category == (
        "profile-exclusion" if corruption == "effects" else "lowering-defect"
    )


def test_unknown_jit_body_and_inconsistent_scan_signature_are_inspected_recursively() -> (
    None
):
    closed = _trace(lambda x: jax.jit(lambda y: y + 1)(x))
    outer = closed.jaxpr.eqns[0]
    key = "jaxpr" if "jaxpr" in outer.params else "call_jaxpr"
    child = outer.params[key]
    body = getattr(child, "jaxpr", child)
    unknown = Primitive("nested_registered_unreviewed")
    bad_body = body.replace(eqns=[body.eqns[0].replace(primitive=unknown)])
    params = {
        **outer.params,
        key: SimpleNamespace(jaxpr=bad_body, consts=getattr(child, "consts", ())),
    }
    with pytest.raises(SourcePrecisionError) as caught:
        DiagnosticValidator({**PLUGIN_REGISTRY, unknown.name: object()})(
            closed.jaxpr.replace(eqns=[outer.replace(params=params)]), 23
        )
    assert (
        caught.value.category == "profile-exclusion"
        and "/body/" in caught.value.location
    )
    closed = _trace(
        lambda x: jax.lax.scan(lambda c, y: (c + y, c), jnp.zeros((3,), x.dtype), x)
    )
    index = next(
        index
        for index, eqn in enumerate(closed.jaxpr.eqns)
        if eqn.primitive.name == "scan"
    )
    eqns = list(closed.jaxpr.eqns)
    eqns[index] = eqns[index].replace(params={**eqns[index].params, "length": 99})
    with pytest.raises(SourcePrecisionError, match="Nested dtype/shape/arity"):
        DiagnosticValidator()(closed.jaxpr.replace(eqns=eqns), 23)


def _metadata(fn, value):
    return {"callable": fn, "testcase": "focused", "input_values": [value]}


def test_actual_function_retrace_calls_same_observer_and_rejects_unknown_body() -> None:
    observer = DiagnosticValidator()
    result = run_source_case(
        _metadata(
            lambda x: _diagnostic_supported_function(x), np.asarray([1], np.float32)
        ),
        SourceScenario("fn", False),
        optimization=None,
        diagnostic_jaxpr_validator=observer,
    )
    assert result.model is not None, result.audit.detail
    assert len(observer.scopes) >= 2
    assert any(
        name.startswith("onnx_fn::")
        for scope in observer.scopes
        for name, _, _ in scope.operations
    )
    report = get_precision_report(result.model)
    assert report.profile == "source-diagnostic-v1"
    assert all(
        check.status is CheckStatus.UNVERIFIED
        for check in report.checks
        if check.category == "lowering"
    )
    result = run_source_case(
        _metadata(
            lambda x: _diagnostic_unreviewed_function(x), np.asarray([1], np.float32)
        ),
        SourceScenario("unknown-fn", False),
        optimization=None,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
    )
    assert result.model is None
    assert result.audit.status is CheckStatus.UNVERIFIED
    assert (
        "_diagnostic_unreviewed_function" in result.audit.detail
        and "exp" in result.audit.detail
    )


@pytest.mark.parametrize(
    "function", [lambda x: x + 1, lambda x: jax.lax.reduce_sum(x, (0,))]
)
def test_diagnostic_and_ordinary_audited_controls_keep_separate_profiles(
    function,
) -> None:
    metadata = _metadata(function, np.asarray([1, 2], np.float32))
    scenario = SourceScenario("separate-controls", False)
    for diagnostic in (False, True, False):
        result = run_source_case(
            metadata,
            scenario,
            optimization=None,
            diagnostic_jaxpr_validator=DiagnosticValidator() if diagnostic else None,
        )
        assert result.model is not None, result.audit.detail
        assert result.audit.status is (
            CheckStatus.UNVERIFIED if diagnostic else CheckStatus.VALIDATED
        )
        assert get_precision_report(result.model).profile == (
            "source-diagnostic-v1" if diagnostic else "source-static-v1"
        )
    sine = _metadata(lambda x: jax.lax.sin(x), np.asarray([1], np.float32))
    diagnostic = run_source_case(
        sine,
        scenario,
        optimization=None,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
    )
    assert diagnostic.model is not None, diagnostic.audit.detail
    ordinary = run_source_case(sine, scenario, optimization=None)
    assert ordinary.model is None and ordinary.audit.status is CheckStatus.UNVERIFIED


def test_complex_boundary_and_hashes_follow_declared_physical_representation() -> None:
    bits = np.asarray([0x80000000, 0, 0x7FC00001, 0x7FC00002], np.uint32)
    complex_values = bits.view(np.complex64)
    result = run_source_case(
        _metadata(lambda x: x, complex_values),
        SourceScenario("complex-payload", False),
        optimization=None,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
    )
    assert result.model is not None, result.audit.detail
    evidence = result.prepared.input_evidence[0]
    assert evidence.logical.dtype == "complex64"
    assert (
        evidence.encoding == "complex-real-pair"
        and evidence.physical_dtype == "float32"
        and evidence.physical_shape == (2, 2)
    )
    assert evidence.logical_sha256 != evidence.physical_sha256
    assert result.prepared.reference_outputs[0].tobytes() == complex_values.tobytes()
    real = run_source_case(
        _metadata(lambda x: x, bits.view(np.float32).reshape(2, 2)),
        SourceScenario("real-pairs", False),
        optimization=None,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
    )
    assert real.model is not None, real.audit.detail
    assert real.prepared.input_evidence[0].encoding == "identity"
    assert real.prepared.input_evidence[0].physical_shape == (2, 2)
