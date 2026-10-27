# tests/extra_tests/converter/test_source_cast_planning.py

"""Inactive cast plans from real source traces, never numeric calibration."""

from dataclasses import fields, is_dataclass, replace
from enum import Enum
import os
from types import SimpleNamespace
from typing import Any

import jax
from jax import export as jax_export
import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import ClosedJaxpr, Jaxpr, JaxprEqn, Primitive, Var
from jax2onnx.converter import conversion_api, source_structure as structure
from jax2onnx.converter import _source_cast_planning as planning
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.plugins.jax._source_f64_f16_structure import freeze_root_cast
from jax2onnx.precision import get_precision_report
from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
    _globals,
    _settings,
)


def _cast(value: Any) -> Any:
    return jax.lax.convert_element_type(value, np.float16)


def _closed(fn: Any = _cast, shape: Any = (6,), *, specs: Any = None) -> Any:
    with jax.enable_x64(True):
        return jax.make_jaxpr(fn)(*(specs or [jax.ShapeDtypeStruct(shape, np.float64)]))


def _eligibility(shape: Any = (6,)) -> planning.WholeRootCastEligibility:
    result = freeze_root_cast(_closed(shape=shape))
    assert type(result) is planning.WholeRootCastEligibility
    return result


def _source(fn: Any = _cast) -> ir.Model:
    model = ui._to_onnx(
        fn,
        [jax.ShapeDtypeStruct((6,), np.float64)],
        precision_policy=PrecisionPolicy.SOURCE,
        return_mode="ir",
    )
    assert isinstance(model, ir.Model)
    return model


@pytest.mark.parametrize("shape", [(), (6,), (17,), (3, 7), (0, 2)])
def test_actual_alpha_traces_freeze_the_whole_dynamic_cast_without_retracing(
    shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _closed(shape=shape)
    second = _closed(
        lambda renamed: jax.lax.convert_element_type(renamed, np.float16), shape
    )
    assert first.jaxpr.invars[0] is not second.jaxpr.invars[0]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("source retraced by planning")

    monkeypatch.setattr(jax, "make_jaxpr", forbidden)
    monkeypatch.setattr(jax, "eval_shape", forbidden)
    before, after = freeze_root_cast(first), freeze_root_cast(second)
    assert before is not None and after is not None
    assert before == after and hash(before) == hash(after)
    assert before.body.expected_equations == 1 and before.body.constants == ()
    assert before.body.inputs[0].logical_dtype == "float64"
    assert before.body.equations[0].outputs[0].logical_dtype == "float16"
    assert before.body.inputs[0].shape == shape
    assert before.params == (
        ("new_dtype", "float16"),
        ("sharding", None),
        ("weak_type", False),
    )
    plan = planning.resolve_cast_plan(before)
    assert planning.plan_is_current(plan)
    assert plan.algorithm is planning.UnresolvedAlgorithm.UNRESOLVED
    assert plan.reasons == ("default-source-compile-context-not-yet-proved",)


@pytest.mark.parametrize(
    "case",
    [
        "constant",
        "literal",
        "other-equation",
        "extra-input",
        "extra-output",
        "jit",
        "wrong-source",
        "wrong-result",
        "weak",
        "symbolic",
    ],
)
def test_other_real_source_content_cannot_borrow_whole_root_eligibility(
    case: str,
) -> None:
    captured = np.arange(6, dtype=np.float64)
    if case == "constant":
        trace = _closed(lambda x: jax.lax.convert_element_type(captured, np.float16))
    elif case == "literal":
        trace = _closed(lambda x: jax.lax.convert_element_type(1.0, np.float16))
    elif case == "other-equation":
        trace = _closed(lambda x: jax.lax.convert_element_type(-x, np.float16))
    elif case == "extra-input":
        trace = _closed(
            lambda x, y: _cast(x), specs=[jax.ShapeDtypeStruct((6,), np.float64)] * 2
        )
    elif case == "extra-output":
        trace = _closed(lambda x: (_cast(x), x))
    elif case == "jit":
        trace = _closed(jax.jit(_cast))
    elif case == "wrong-source":
        trace = _closed(specs=[jax.ShapeDtypeStruct((6,), np.float32)])
    elif case == "wrong-result":
        trace = _closed(lambda x: jax.lax.convert_element_type(x, np.float32))
    elif case == "weak":
        trace = _closed(specs=[jax.ShapeDtypeStruct((6,), np.float64, weak_type=True)])
    else:
        (batch,) = jax_export.symbolic_shape("B")
        trace = _closed(shape=(batch, 3))
    assert freeze_root_cast(trace) is None
    plan = planning.prepare_root_cast_plan(trace)
    assert plan.eligibility is None
    assert plan.algorithm is planning.UnresolvedAlgorithm.UNRESOLVED
    assert "whole-root-dynamic-cast-unavailable" in plan.reasons


@pytest.mark.parametrize(
    "fault",
    [
        "missing-sharding",
        "unknown-param",
        "sharding",
        "weak-zero",
        "weak-one",
        "dtype",
        "counterfeit",
    ],
)
def test_actual_native_equation_parameter_schema_is_exact(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = _closed()
    equation = trace.jaxpr.eqns[0]
    with monkeypatch.context() as patch:
        if fault == "missing-sharding":
            patch.delitem(equation.params, "sharding")
        elif fault == "unknown-param":
            patch.setitem(equation.params, "algorithm", "direct-rne")
        elif fault == "sharding":
            patch.setitem(equation.params, "sharding", "cpu")
        elif fault == "weak-zero":
            patch.setitem(equation.params, "weak_type", 0)
        elif fault == "weak-one":
            patch.setitem(equation.params, "weak_type", 1)
        elif fault == "dtype":
            patch.setitem(equation.params, "new_dtype", "float16")
        else:
            patch.setattr(equation, "primitive", Primitive("convert_element_type"))
        assert freeze_root_cast(trace) is None
    assert freeze_root_cast(trace) is not None


@pytest.mark.parametrize("port", ["input", "output"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("weak_type", 0),
        ("weak_type", 1),
        ("shape", (6.0,)),
        ("shape", (True,)),
        ("shape", (np.int64(6),)),
        ("shape", [6]),
    ],
)
def test_actual_input_and_output_aval_types_are_not_coerced(
    port: str, field: str, value: Any
) -> None:
    trace = _closed()
    aval = (trace.jaxpr.invars if port == "input" else trace.jaxpr.outvars)[0].aval
    assert type(aval) is jax.core.ShapedArray
    original = object.__getattribute__(aval, field)
    try:
        # The native constructor uses this slot primitive too. Bypass only its
        # immutability guard, without canonicalizing the malicious field value.
        object.__setattr__(aval, field, value)
        assert object.__getattribute__(aval, field) is value
        assert (trace.jaxpr.invars if port == "input" else trace.jaxpr.outvars)[
            0
        ].aval is aval
        assert freeze_root_cast(trace) is None
    finally:
        # ShapedArray is interned: restoration must also run on failed assertions.
        object.__setattr__(aval, field, original)
    assert object.__getattribute__(aval, field) is original
    assert freeze_root_cast(trace) is not None


def _changed_tensor(
    eligibility: planning.WholeRootCastEligibility, port: str, **updates: Any
) -> Any:
    body = eligibility.body
    equation = body.equations[0]
    if port == "input":
        tensor = replace(body.inputs[0], **updates)
        equation = replace(
            equation, inputs=(replace(equation.inputs[0], tensor=tensor),)
        )
        return replace(body, inputs=(tensor,), equations=(equation,))
    tensor = replace(equation.outputs[0], **updates)
    return replace(
        body,
        equations=(replace(equation, outputs=(tensor,)),),
        outputs=(replace(body.outputs[0], tensor=tensor),),
    )


@pytest.mark.parametrize("port", ["input", "output"])
@pytest.mark.parametrize(
    "updates",
    [
        {"weak_type": 0},
        {"shape": (6.0,), "logical_shape": (6.0,)},
        {"shape": (np.int64(6),), "logical_shape": (np.int64(6),)},
    ],
)
def test_equal_looking_immutable_contract_forgeries_are_rejected(
    port: str, updates: dict[str, Any]
) -> None:
    eligible = _eligibility()
    forged_body = _changed_tensor(eligible, port, **updates)
    assert (
        forged_body == eligible.body
    )  # Python/dataclass equality alone is insufficient.
    with pytest.raises(ValueError):
        planning.make_cast_eligibility(forged_body, eligible.params)
    plan = planning.resolve_cast_plan(replace(eligible, body=forged_body))
    assert (
        plan.eligibility is None and "whole-root-cast-contract-differs" in plan.reasons
    )


@pytest.mark.parametrize(
    "feature", ["", "+avx512fp16", "-avx512fp16", "known-model-name"]
)
def test_query_metadata_and_feature_strings_never_authorize_source(
    feature: str,
) -> None:
    eligible = _eligibility()
    # These are explicitly synthetic evidence-only records, not measured native HIGH.
    observed = planning.CompilationObservation(
        acquisition="same-client-empty-query",
        client_process_binding=(os.getpid(), 1),
        target=(("features", feature),),
        options=(("query-defaults-sha256", "0" * 64),),
        source_body_sha256=eligible.source_body_sha256,
        authority=planning.UnavailableSourceContext(),
    )
    plan = planning.resolve_cast_plan(eligible, observed)
    assert planning.plan_is_current(plan) and plan.observation is observed
    assert plan.algorithm is planning.UnresolvedAlgorithm.UNRESOLVED
    assert "default-source-compile-context-not-yet-proved" in plan.reasons


@pytest.mark.parametrize(
    "authority",
    [
        True,
        "default-source",
        SimpleNamespace(valid=True),
        object(),
        planning.UnavailableSourceContext("verified"),
    ],
)
def test_foreign_or_forged_context_authority_is_not_retained(authority: Any) -> None:
    plan = planning.resolve_cast_plan(
        _eligibility(), planning.CompilationObservation(authority=authority)
    )
    assert plan.observation is None
    assert plan.algorithm is planning.UnresolvedAlgorithm.UNRESOLVED
    assert "compilation-context-record-unavailable" in plan.reasons


def test_stale_source_binding_and_forged_plan_cannot_be_current() -> None:
    original, other = _eligibility((6,)), _eligibility((7,))
    context = planning.CompilationObservation(
        source_body_sha256=original.source_body_sha256
    )
    stale = planning.resolve_cast_plan(other, context)
    assert "compilation-source-binding-differs" in stale.reasons
    assert stale.algorithm is planning.UnresolvedAlgorithm.UNRESOLVED
    assert not planning.plan_is_current(replace(stale, algorithm="direct-rne"))
    assert not planning.plan_is_current(replace(stale, reasons=()))
    assert not planning.plan_is_current(replace(stale, eligibility_sha256="0" * 64))
    assert (
        planning.resolve_cast_plan(replace(original, root_ordinal=False)).eligibility
        is None
    )


def test_frozen_plan_retains_only_immutable_contracts_not_native_source_owners() -> (
    None
):
    plan = planning.prepare_root_cast_plan(_closed())

    def visit(value: Any) -> None:
        assert not isinstance(value, (Jaxpr, ClosedJaxpr, JaxprEqn, Var, np.ndarray))
        if value is None or type(value) in (str, int, bool) or isinstance(value, Enum):
            return
        if type(value) is tuple:
            for item in value:
                visit(item)
            return
        assert is_dataclass(value)
        for field in fields(value):
            visit(getattr(value, field.name))

    visit(plan)


def test_real_plan_is_independent_of_ledger_observation_and_never_activates_rne(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jax2onnx.plugins.jax import _f64_to_f16_rne

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unapproved RNE emission activated")

    monkeypatch.setattr(_f64_to_f16_rne, "emit_f64_to_f16_rne", forbidden)
    create = conversion_api._create_ir_context
    artifacts = []
    with jax.enable_x64(True):
        context_before = _globals(), _settings()
        for observe in (True, False):
            calls = []

            def model(value: Any) -> Any:
                calls.append("original-source")
                return jax.lax.convert_element_type(value, np.float16)

            with monkeypatch.context() as patch:
                if not observe:

                    def without_ledger(**kwargs: Any) -> Any:
                        ctx = create(**kwargs)
                        ctx._source_structure = None
                        return ctx

                    patch.setattr(conversion_api, "_create_ir_context", without_ledger)
                artifact = _source(model)
            assert calls == ["original-source"]
            plan = artifact.meta[planning.PLAN_KEY]
            assert plan.algorithm is planning.UnresolvedAlgorithm.UNRESOLVED
            assert plan.eligibility is not None and planning.plan_is_current(plan)
            assert [node.op_type for node in artifact.graph] == ["Cast"]
            assert artifact.graph.outputs[0].dtype is ir.DataType.FLOAT16
            artifacts.append(
                (
                    artifact,
                    ir.to_proto(artifact).SerializeToString(deterministic=True),
                    get_precision_report(artifact),
                )
            )
            assert (_globals(), _settings()) == context_before
    assert artifacts[0][1:] == artifacts[1][1:]
    assert (
        artifacts[0][0].meta[planning.PLAN_KEY]
        == artifacts[1][0].meta[planning.PLAN_KEY]
    )
    ledger = artifacts[0][0].meta[structure._LEDGER]
    assert len(ledger.cast_plans) == 1
    handle, mirrored = next(iter(ledger.cast_plans.items()))
    assert handle.invocation is ledger.scopes[0].invocation and handle.index == 0
    assert mirrored is artifacts[0][0].meta[planning.PLAN_KEY]
    proof = structure.validate_source_structure(artifacts[0][0])
    assert proof.status == "unverified" and proof.expected_equations == 1
    assert proof.runtime == proof.numerics == "unverified"


@pytest.mark.parametrize("fault", ["missing", "wrong-handle", "copied-plan"])
def test_actual_plan_mirror_cannot_move_or_disappear(fault: str) -> None:
    with jax.enable_x64(True):
        model = _source()
    ledger = model.meta[structure._LEDGER]
    handle, plan = next(iter(ledger.cast_plans.items()))
    if fault == "missing":
        ledger.cast_plans.clear()
    elif fault == "wrong-handle":
        ledger.cast_plans = {structure.EquationHandle(object(), 0): plan}
    else:
        ledger.cast_plans[handle] = replace(plan)
    proof = structure.validate_source_structure(model)
    assert proof.status == "unverified" and proof.expected_equations == 1
    assert proof.reasons == ("cast-plan-handle-coverage-incomplete",)


@onnx_function
def _function_cast(value: Any) -> Any:
    return _cast(value)


def test_function_scope_cannot_reuse_a_root_only_plan() -> None:
    with jax.enable_x64(True):
        model = _source(lambda value: _function_cast(value))
    plan = model.meta[planning.PLAN_KEY]
    assert plan.eligibility is None
    assert not model.meta[structure._LEDGER].cast_plans
