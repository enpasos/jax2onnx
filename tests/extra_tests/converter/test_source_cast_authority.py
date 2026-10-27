# tests/extra_tests/converter/test_source_cast_authority.py

"""Real producer/cleanup/context authority controls without numerical activation."""

import copy
from dataclasses import FrozenInstanceError, replace
import gc
import hashlib
import pickle
import sys
import threading
from typing import Any
import weakref

import jax
import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._compat import _cast_build_profile as builds
from jax2onnx._compat import _cast_lowering as native_lowering
from jax2onnx.converter import _source_cast_authority as authority
from jax2onnx.converter import _source_cast_context as acquisition
from jax2onnx.converter import _source_cast_planning as planning
from jax2onnx.converter import conversion_api
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError


def _direct(value: Any) -> Any:
    return jax.lax.convert_element_type(value, np.float16)


def _closure() -> Any:
    dtype = "float16"

    def original(value: Any) -> Any:
        return jax.lax.convert_element_type(value, np.dtype(dtype))

    return original


def _export(function: Any = _direct, *, opset: int = 23) -> ir.Model:
    result = ui._to_onnx(
        function,
        [jax.ShapeDtypeStruct((6,), np.float64)],
        opset=opset,
        return_mode="ir",
        precision_policy=PrecisionPolicy.SOURCE,
    )
    assert type(result) is ir.Model
    return result


def _require_reviewed_row() -> Any:
    row = next(
        (
            row
            for row in builds._PROFILES
            if row.cache_tag == sys.implementation.cache_tag
            and row.version == jax.__version__
        ),
        None,
    )
    if row is None or not row.provider_proof:
        pytest.skip("exact installed provider row has not been reviewed")
    return row


def _track_transaction(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    transactions = []
    begin = conversion_api._begin_cast_transaction

    def tracked(nonce: object, opset: int) -> Any:
        result = begin(nonce, opset)
        transactions.append(result)
        return result

    monkeypatch.setattr(conversion_api, "_begin_cast_transaction", tracked)
    return transactions


def _empty_context() -> IRContext:
    return IRContext(
        opset=23, enable_double_precision=False, precision_policy=PrecisionPolicy.SOURCE
    )


def _assert_discarded(transaction: Any) -> None:
    assert transaction._nonce is None and transaction._handoff is None
    assert transaction._permit is None and transaction._context is None
    assert transaction._proof == transaction._body_proof == ""


def test_cold_ordinary_cpu_initializes_once_without_a_replacement_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_reviewed_row()
    from jax._src import xla_bridge

    if (
        xla_bridge._backends.get("cpu") is not None
        or xla_bridge._default_backend is not None
    ):
        pytest.skip(
            "cold-client control requires a fresh process; never clear backend caches"
        )
    actual = conversion_api.RootCastContextLease
    initial = []

    class Tracked(actual):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            assert self._native is not None
            initial.append(self._native._backend_default_initial)

    monkeypatch.setattr(conversion_api, "RootCastContextLease", Tracked)
    with jax.enable_x64(True):
        model = _export()
    assert initial == [None]
    assert authority.AUTHORITY_KEY in model.meta
    assert dict(model.meta[acquisition.CONTEXT_KEY].native_fields)[
        "client_identity"
    ] == str(id(xla_bridge._backends["cpu"]))


@pytest.mark.parametrize("opset", [21, 23])
def test_real_original_producer_issues_after_release_and_keeps_single_cast(
    opset: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _require_reviewed_row()
    transactions = _track_transaction(monkeypatch)
    lower = conversion_api._lower_jaxpr_equations
    roots = []
    original_root = conversion_api.RootCastContextLease

    class TrackedRoot(original_root):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            roots.append(self)

    monkeypatch.setattr(conversion_api, "RootCastContextLease", TrackedRoot)
    seen = []

    def inspect(ctx: Any, body: Any) -> None:
        permit = ctx._source_cast_permit
        assert type(permit) is authority._CastPermit
        assert (
            roots[0]._closed and roots[0]._native is None and roots[0]._lowering is None
        )
        assert roots[0]._callable._state is None and not roots[0]._callable._members
        assert permit._handoff._equation is body.eqns[0]
        assert (
            permit._handoff._input is body.invars[0]
            and permit._handoff._output is body.outvars[0]
        )
        assert permit._input_value is ctx.require_value_for_var(body.invars[0])
        seen.append((ctx, permit, permit.audit))
        lower(ctx, body)

    monkeypatch.setattr(conversion_api, "_lower_jaxpr_equations", inspect)
    if opset == 21:
        with jax.enable_x64(True), pytest.raises(SourcePrecisionError) as failure:
            _export(opset=opset)
        error = failure.value
        assert (
            error.phase,
            error.category,
            error.location,
            error.primitive,
            error.opset,
        ) == (
            "profile-validation",
            "profile-exclusion",
            "equation 0",
            "convert_element_type",
            21,
        )
        assert error.requested_dtypes == ("float64", "float16")
        assert (
            error.detail
            == "dtype is outside the scientific opset21 operation/type intersection"
        )
        assert (
            error.requirement
            == "the owning operation's native FP32/FP64 data and Boolean/INT32/INT64 structural roles at21"
        )
        assert len(transactions) == len(roots) == 1 and not seen
        assert roots[0]._closed and roots[0]._native is None
        _assert_discarded(transactions[0])
        return
    with jax.enable_x64(True):
        model = _export(opset=opset)
    assert len(transactions) == len(seen) == 1
    ctx, permit, audit = seen[0]
    assert model.meta[authority.AUTHORITY_KEY] is audit
    assert audit.build_cohort == row.cohort and audit.opset == opset
    assert audit.algorithm in ("DIRECT_RNE_53_TO_11", "DOUBLE_ROUND_RNE_53_TO_24_TO_11")
    assert audit.numerical_activation is False
    assert audit.nan_contract == "original-criterion-specific-no-payload-certification"
    record = model.meta[acquisition.CONTEXT_KEY]
    assert record.original_lowering is not None
    assert (
        audit.native_options_sha256
        == hashlib.sha256(record.original_lowering.options).hexdigest()
    )
    assert (
        audit.native_options_sha256 != hashlib.sha256(record.native_options).hexdigest()
    )
    assert (
        audit.native_target_sha256 == hashlib.sha256(record.target_payload).hexdigest()
    )
    assert (
        model.meta[planning.PLAN_KEY].algorithm
        is planning.UnresolvedAlgorithm.UNRESOLVED
    )
    assert [node.op_type for node in model.graph] == ["Cast"]
    assert (
        ctx._source_cast_permit is None
        and permit._handoff is None
        and permit._owner is None
    )
    _assert_discarded(transactions[0])
    with pytest.raises(FrozenInstanceError):
        audit.algorithm = "forged"


@pytest.mark.parametrize(
    "fault", ["forged-record", "copied-audit", "dict", "fabricated-permit"]
)
def test_public_records_and_fabricated_objects_never_authorize(fault: str) -> None:
    ctx = _empty_context()
    if fault == "fabricated-permit":
        value = object.__new__(authority._CastPermit)
    else:
        value = {
            "algorithm": "DIRECT_RNE_53_TO_11",
            "status": "SUPPORTED_CURRENT_OPTIONS",
        }
        if fault == "copied-audit":
            value = copy.deepcopy(value)
        elif fault == "forged-record":
            value = planning.CompilationObservation(
                build=(("native_sha256", "f" * 64),),
                options=(("status", "SUPPORTED_CURRENT_OPTIONS"),),
            )
    ctx._source_cast_permit = value
    assert authority._consume_cast_permit(ctx, object()) is None
    assert (
        ctx._source_cast_permit is None
        or type(ctx._source_cast_permit) is not authority._CastPermit
    )
    assert authority._consume_cast_permit(object(), object()) is None
    with pytest.raises(ValueError, match="caller constructed"):
        authority._CastTransaction(object(), object(), 23)
    with pytest.raises(ValueError, match="caller constructed"):
        authority._CastPermit(object(), None, ctx, None, None, None)


def test_supported_looking_native_dataclass_cannot_be_staged_as_a_producer() -> None:
    module = native_lowering.NativeCastModule(
        "untrusted", "0" * 64, (6,), ("stablehlo.convert", "func.return")
    )
    record = native_lowering.NativeCastLowering(
        module, b"options", "0" * 64, (False,), (False,), False, 1, 2, False
    )
    transaction = authority._begin_cast_transaction(object(), 23)
    try:
        with pytest.raises(ValueError, match="actual native cast handoff"):
            transaction.stage(record, "f" * 64, "f" * 64)
        assert transaction._handoff is None and transaction.audit is None
    finally:
        transaction.discard()
    _assert_discarded(transaction)


@pytest.mark.parametrize(
    "fault", ["cross-context", "wrong-equation", "copy", "deepcopy", "pickle", "reuse"]
)
def test_actual_one_use_permit_cannot_be_moved_copied_serialized_or_replayed(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_reviewed_row()
    transactions = _track_transaction(monkeypatch)
    lower = conversion_api._lower_jaxpr_equations
    seen = []

    def inspect(ctx: Any, body: Any) -> None:
        permit = ctx._source_cast_permit
        assert type(permit) is authority._CastPermit
        seen.append(permit)
        if fault in ("copy", "deepcopy", "pickle"):
            operation = {
                "copy": copy.copy,
                "deepcopy": copy.deepcopy,
                "pickle": pickle.dumps,
            }[fault]
            with pytest.raises(TypeError):
                operation(permit)
            with pytest.raises(TypeError):
                operation(permit._handoff)
            with pytest.raises(TypeError):
                operation(transactions[0])
            assert authority._consume_cast_permit(ctx, body.eqns[0]) is permit.audit
        elif fault == "cross-context":
            foreign = _empty_context()
            foreign._source_cast_permit = permit
            assert authority._consume_cast_permit(foreign, body.eqns[0]) is None
            assert foreign._source_cast_permit is None
        elif fault == "wrong-equation":
            assert authority._consume_cast_permit(ctx, object()) is None
        else:
            audit = authority._consume_cast_permit(ctx, body.eqns[0])
            assert audit is permit.audit
        assert ctx._source_cast_permit is None
        assert authority._consume_cast_permit(ctx, body.eqns[0]) is None
        lower(ctx, body)

    monkeypatch.setattr(conversion_api, "_lower_jaxpr_equations", inspect)
    with jax.enable_x64(True):
        model = _export()
    assert len(seen) == 1 and seen[0]._handoff is None
    assert [node.op_type for node in model.graph] == ["Cast"]
    _assert_discarded(transactions[0])


@pytest.mark.parametrize(
    "fault", ["input-rebind", "input-dtype", "input-shape", "equation-params"]
)
def test_post_cleanup_actual_source_equation_and_bound_input_mutations_revoke_authority(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_reviewed_row()
    lower = conversion_api._lower_jaxpr_equations
    transactions = _track_transaction(monkeypatch)
    seen = []

    def inspect(ctx: Any, body: Any) -> None:
        permit = ctx._source_cast_permit
        assert type(permit) is authority._CastPermit
        equation, var = body.eqns[0], body.invars[0]
        value = ctx.require_value_for_var(var)
        dtype, shape = value.dtype, value.shape
        weak_type = equation.params["weak_type"]
        try:
            if fault == "input-rebind":
                ctx.bind_value_for_var(
                    var, ir.val("foreign_input", dtype=ir.DataType.DOUBLE, shape=[6])
                )
            elif fault == "input-dtype":
                value.dtype = ir.DataType.FLOAT
            elif fault == "input-shape":
                value.shape = ir.Shape([7])
            else:
                equation.params["weak_type"] = True
            if fault == "equation-params":
                with pytest.raises(ValueError, match="equation/typed-port"):
                    authority._consume_cast_permit(ctx, equation)
            else:
                assert authority._consume_cast_permit(ctx, equation) is None
            assert ctx._source_cast_permit is None and permit._handoff is None
            seen.append(permit)
        finally:
            ctx.bind_value_for_var(var, value)
            value.dtype, value.shape = dtype, shape
            equation.params["weak_type"] = weak_type
        lower(ctx, body)

    monkeypatch.setattr(conversion_api, "_lower_jaxpr_equations", inspect)
    with jax.enable_x64(True):
        model = _export()
    assert len(seen) == 1 and authority.AUTHORITY_KEY not in model.meta
    assert [node.op_type for node in model.graph] == ["Cast"]
    _assert_discarded(transactions[0])


@pytest.mark.parametrize(
    "fault",
    [
        "query-options",
        "historical-options",
        "module",
        "client",
        "input-propagation",
        "output-propagation",
        "logging",
    ],
)
def test_replaced_public_native_record_cannot_substitute_for_real_producer_handoff(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_reviewed_row()
    actual = acquisition.lower_original_cast_jaxpr

    def replaced(*args: Any) -> Any:
        result = actual(*args)
        if fault == "query-options":
            payload = b"query-only-options"
            return replace(
                result,
                options=payload,
                options_sha256=hashlib.sha256(payload).hexdigest(),
            )
        if fault == "historical-options":
            return replace(result, options_scope="historical-executable")
        if fault == "module":
            return replace(result, module=replace(result.module, text_sha256="0" * 64))
        if fault == "client":
            return replace(result, client_identity=result.client_identity + 1)
        if fault == "input-propagation":
            return replace(result, input_propagation=(not result.input_propagation[0],))
        if fault == "output-propagation":
            return replace(
                result, output_propagation=(not result.output_propagation[0],)
            )
        return replace(result, detailed_logging=not result.detailed_logging)

    monkeypatch.setattr(acquisition, "lower_original_cast_jaxpr", replaced)
    transactions = _track_transaction(monkeypatch)
    with jax.enable_x64(True):
        model = _export()
    assert authority.AUTHORITY_KEY not in model.meta
    assert model.meta[acquisition.CONTEXT_KEY].original_lowering is None
    _assert_discarded(transactions[0])


def test_nonprofile_cleanup_factory_code_mutation_revokes_staged_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_reviewed_row()
    from jax._src import xla_bridge

    actual_finish = acquisition.RootCastContextLease.finish
    function, original_code = (
        xla_bridge.make_cpu_client,
        xla_bridge.make_cpu_client.__code__,
    )
    staged = []

    def changed(instance: Any, *args: Any, **kwargs: Any) -> Any:
        result = actual_finish(instance, *args, **kwargs)
        assert instance.record.original_lowering is not None
        assert instance._transaction._handoff is not None
        staged.append(instance._transaction._handoff)
        function.__code__ = original_code.replace(
            co_consts=(*original_code.co_consts, "cleanup-factory-code-change")
        )
        return result

    monkeypatch.setattr(acquisition.RootCastContextLease, "finish", changed)
    transactions = _track_transaction(monkeypatch)
    try:
        with jax.enable_x64(True):
            model = _export()
    finally:
        function.__code__ = original_code
    record = model.meta[acquisition.CONTEXT_KEY]
    assert record.original_lowering is None and record.original_logging == "UNAVAILABLE"
    assert "cast-context-changed-during-cleanup" in record.reasons
    assert (
        dict(model.meta[planning.PLAN_KEY].observation.options)["original_availability"]
        == "failed-during-cleanup"
    )
    assert authority.AUTHORITY_KEY not in model.meta
    assert staged[0]._equation is None and staged[0].record is None
    _assert_discarded(transactions[0])


@pytest.mark.parametrize(
    "stage", ["trace", "validator", "context", "inputs", "lower", "finalize"]
)
def test_every_conversion_failure_discards_native_handoff_and_context_owners(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_reviewed_row()
    transactions = _track_transaction(monkeypatch)

    def failure(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("authority-" + stage)

    member = {
        "trace": "_trace_to_jaxpr",
        "context": "_create_ir_context",
        "inputs": "_bind_jaxpr_inputs",
        "lower": "_lower_jaxpr_equations",
        "finalize": "_build_and_finalize_ir_model",
    }.get(stage)
    if member is not None:
        monkeypatch.setattr(conversion_api, member, failure)
    with jax.enable_x64(True), pytest.raises(RuntimeError, match="authority-" + stage):
        conversion_api._convert_to_ir(
            fn=_direct,
            inputs=[jax.ShapeDtypeStruct((6,), np.float64)],
            input_params=None,
            model_name="cast",
            opset=23,
            enable_double_precision=False,
            record_primitive_calls_file=None,
            precision_policy=PrecisionPolicy.SOURCE,
            jaxpr_validator=failure if stage == "validator" else lambda *_: None,
        )
    assert len(transactions) == 1
    _assert_discarded(transactions[0])


@pytest.mark.parametrize(
    "prewarm", [False, True], ids=["ordinary-initial-state", "prewarmed-client"]
)
@pytest.mark.parametrize("profile", ["none", "sys", "thread"])
def test_ordinary_clients_and_stable_observation_profiles_keep_real_issuance(
    prewarm: bool, profile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_reviewed_row()
    from jax._src import xla_bridge

    before = (sys.getprofile(), threading.getprofile())
    calls = []

    def observer(frame: Any, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is _direct.__code__:
            calls.append("original")

    try:
        if profile == "sys":
            sys.setprofile(observer)
        elif profile == "thread":
            threading.setprofile(observer)
        selected = (sys.getprofile(), threading.getprofile())
        with jax.enable_x64(True):
            if prewarm:
                client = xla_bridge.get_backend("cpu")
                assert client is xla_bridge._backends["cpu"]
            model = _export()
        assert sys.getprofile() is selected[0] and threading.getprofile() is selected[1]
    finally:
        sys.setprofile(before[0])
        threading.setprofile(before[1])
    assert authority.AUTHORITY_KEY in model.meta
    if profile == "sys":
        assert calls == ["original"]
    fields = dict(model.meta[acquisition.CONTEXT_KEY].native_fields)
    assert fields["work"] == (
        "compile=1,serialize-executable=1,execute=0"
        if jax.__version__ == "0.10.2"
        else "compile=0,serialize-executable=0,execute=0"
    )


def test_successful_model_and_discarded_transaction_do_not_retain_original_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_reviewed_row()
    transactions = _track_transaction(monkeypatch)
    original = _closure()
    reference = weakref.ref(original)
    with jax.enable_x64(True):
        model = _export(original)
    assert authority.AUTHORITY_KEY in model.meta
    del original
    gc.collect()
    assert reference() is None
    _assert_discarded(transactions[0])


@pytest.mark.parametrize("opset", [20, 22, 24])
def test_other_requested_opsets_keep_evidence_and_no_private_authority(
    opset: int,
) -> None:
    _require_reviewed_row()
    if opset in (20, 22):
        with jax.enable_x64(True), pytest.raises(SourcePrecisionError) as failure:
            _export(opset=opset)
        error = failure.value
        assert (error.phase, error.category, error.location, error.opset) == (
            "input",
            "profile-exclusion",
            "export options",
            opset,
        )
        assert (
            error.detail
            == f"opset={opset} is an internal migration gap in this audited milestone"
        )
        assert (
            error.requirement == "the current private source-static-v1 admission scope"
        )
        return
    with jax.enable_x64(True):
        model = _export(opset=opset)
    assert model.meta[acquisition.CONTEXT_KEY].original_lowering is not None
    assert authority.AUTHORITY_KEY not in model.meta
    assert [node.op_type for node in model.graph] == ["Cast"]


@pytest.mark.parametrize("view", ["permit", "transaction"])
@pytest.mark.parametrize("replacement", ["equal-copy", "forged-algorithm"])
def test_real_permit_and_transaction_audit_views_reject_public_replacement(
    view: str, replacement: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_reviewed_row()
    transactions = _track_transaction(monkeypatch)
    lower = conversion_api._lower_jaxpr_equations
    seen = []

    def inspect(ctx: Any, body: Any) -> None:
        permit = ctx._source_cast_permit
        assert type(permit) is authority._CastPermit
        owner = transactions[0]
        original = permit.audit
        assert original is owner.audit is ctx._source_cast_audit
        candidate = (
            replace(original)
            if replacement == "equal-copy"
            else replace(original, algorithm="forged")
        )
        assert candidate is not original
        if replacement == "equal-copy":
            assert candidate == original
        else:
            assert (
                candidate.opset == original.opset and candidate.shape == original.shape
            )
        with pytest.raises(AttributeError):
            setattr(permit if view == "permit" else owner, "audit", candidate)
        assert permit.audit is owner.audit is ctx._source_cast_audit is original
        assert authority._consume_cast_permit(ctx, body.eqns[0]) is original
        assert ctx._source_cast_permit is None and permit._handoff is None
        assert authority._consume_cast_permit(ctx, body.eqns[0]) is None
        seen.append(original)
        lower(ctx, body)

    monkeypatch.setattr(conversion_api, "_lower_jaxpr_equations", inspect)
    with jax.enable_x64(True):
        model = _export()
    assert len(seen) == 1 and model.meta[authority.AUTHORITY_KEY] is seen[0]
    assert [node.op_type for node in model.graph] == ["Cast"]
    _assert_discarded(transactions[0])


@pytest.mark.parametrize(
    "location", ["permit-backing", "transaction-backing", "context-mirror"]
)
@pytest.mark.parametrize(
    "replacement", ["equal-copy", "forged-algorithm", "wrong-type"]
)
def test_real_permit_rejects_changed_audit_identity_and_clears_authority(
    location: str, replacement: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_reviewed_row()
    transactions = _track_transaction(monkeypatch)
    lower = conversion_api._lower_jaxpr_equations
    seen = []

    def inspect(ctx: Any, body: Any) -> None:
        permit = ctx._source_cast_permit
        assert type(permit) is authority._CastPermit
        owner = transactions[0]
        original = permit.audit
        assert original is owner.audit is ctx._source_cast_audit
        candidate = (
            replace(original)
            if replacement == "equal-copy"
            else (
                replace(original, algorithm="forged")
                if replacement == "forged-algorithm"
                else object()
            )
        )
        assert candidate is not original
        if replacement == "equal-copy":
            assert candidate == original
        elif replacement == "forged-algorithm":
            assert (
                candidate.opset == original.opset and candidate.shape == original.shape
            )
        if location == "permit-backing":
            permit._audit = candidate
        elif location == "transaction-backing":
            owner._audit = candidate
        else:
            ctx._source_cast_audit = candidate
        assert authority._consume_cast_permit(ctx, body.eqns[0]) is None
        assert ctx._source_cast_permit is None and ctx._source_cast_audit is None
        assert owner.audit is None and permit._handoff is None and permit._owner is None
        assert authority._consume_cast_permit(ctx, body.eqns[0]) is None
        _assert_discarded(owner)
        seen.append(permit)
        lower(ctx, body)

    monkeypatch.setattr(conversion_api, "_lower_jaxpr_equations", inspect)
    with jax.enable_x64(True):
        model = _export()
    assert len(seen) == 1 and authority.AUTHORITY_KEY not in model.meta
    assert [node.op_type for node in model.graph] == ["Cast"]
    _assert_discarded(transactions[0])
