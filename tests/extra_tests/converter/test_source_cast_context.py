# tests/extra_tests/converter/test_source_cast_context.py

"""Real provenance/constructor acquisition, with no numerical authority."""

from dataclasses import fields, is_dataclass
from enum import Enum
import gc
import sys
from types import FunctionType
from typing import Any
import weakref

import jax
import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._compat import jax as compat
from jax2onnx._compat import _cast_metadata as metadata
from jax2onnx.converter import conversion_api
from jax2onnx.converter import _source_cast_context as context
from jax2onnx.converter import _source_cast_planning as planning
from jax2onnx.converter import _source_cast_provenance as provenance
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.plugins.jax._source_f64_f16_structure import freeze_root_cast
from jax2onnx.precision import get_precision_report


def _direct(value: Any) -> Any:
    return jax.lax.convert_element_type(value, np.float16)


def _factory() -> Any:
    target = "float16"

    def original(value: Any) -> Any:
        return jax.lax.convert_element_type(value, np.dtype(target))

    return original


def _plan() -> planning.WholeRootCastPlan:
    with jax.enable_x64(True):
        trace = jax.make_jaxpr(_direct)(jax.ShapeDtypeStruct((6,), np.float64))
    eligible = freeze_root_cast(trace)
    assert eligible is not None
    return planning.resolve_cast_plan(eligible)


def _export(function: Any = _direct, **kwargs: Any) -> ir.Model:
    model = ui._to_onnx(
        function,
        [jax.ShapeDtypeStruct((6,), np.float64)],
        return_mode="ir",
        precision_policy=PrecisionPolicy.SOURCE,
        **kwargs,
    )
    assert type(model) is ir.Model
    return model


@pytest.mark.parametrize("factory", [lambda: _direct, _factory])
def test_actual_ordinary_body_and_closure_are_pinned_without_invocation(
    factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = factory()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("provenance invoked or traced source")

    monkeypatch.setattr(jax, "make_jaxpr", forbidden)
    monkeypatch.setattr(jax, "eval_shape", forbidden)
    lease = provenance.CallableCastLease(function)
    assert lease.proof is not None and lease.reason is None
    lease.assert_current()
    assert len(lease.proof.source_sha256) == len(lease.proof.code_sha256) == 64
    lease.close()
    assert lease._state is None and not lease._members
    with pytest.raises(ValueError, match="closed"):
        lease.assert_current()


@pytest.mark.parametrize("fault", ["helper", "config", "default", "object", "lambda"])
def test_other_python_bodies_do_not_borrow_a_one_cast_trace(fault: str) -> None:
    def helper(value: Any) -> Any:
        return _direct(value)

    def config(value: Any) -> Any:
        with jax.enable_x64(True):
            return jax.lax.convert_element_type(value, np.float16)

    def default(value: Any, dtype: Any = np.float16) -> Any:
        return jax.lax.convert_element_type(value, dtype)

    function = {
        "helper": helper,
        "config": config,
        "default": default,
        "object": object(),
        "lambda": lambda value: jax.lax.convert_element_type(value, np.float16),
    }[fault]
    lease = provenance.CallableCastLease(function)
    assert (
        lease.proof is None
        and lease.reason == "ordinary-cast-callable-provenance-unavailable"
    )
    lease.close()


@pytest.mark.parametrize("member", ["dtype", "convert_element_type"])
def test_selected_module_members_are_checked_without_calling_replacements(
    member: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = _factory()
    lease = provenance.CallableCastLease(function)
    assert lease.proof is not None
    calls = []

    def replaced(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        raise AssertionError("foreign member invoked")

    with monkeypatch.context() as patch:
        patch.setattr(np if member == "dtype" else jax.lax, member, replaced)
        with pytest.raises(ValueError):
            lease.assert_current()
        before_first = provenance.CallableCastLease(function)
        assert before_first.proof is None and not calls
        before_first.close()
    lease.assert_current()
    lease.close()


def test_closure_and_live_code_mutations_are_detected_and_owners_released() -> None:
    function = _factory()
    reference = weakref.ref(function)
    lease = provenance.CallableCastLease(function)
    assert lease.proof is not None
    cell = function.__closure__[0]
    original = cell.cell_contents
    try:
        cell.cell_contents = np.float16
        with pytest.raises(ValueError):
            lease.assert_current()
    finally:
        cell.cell_contents = original
    code = function.__code__
    try:
        function.__code__ = code.replace(
            co_consts=(*code.co_consts, "unexecuted-change")
        )
        with pytest.raises(ValueError):
            lease.assert_current()
        assert provenance.CallableCastLease(function).proof is None
    finally:
        function.__code__ = code
    lease.assert_current()
    lease.close()
    del function, cell, code
    gc.collect()
    assert reference() is None


def test_same_native_public_function_code_cannot_change_before_or_after_capture() -> (
    None
):
    function = _factory()
    lease = provenance.CallableCastLease(function)
    assert lease.proof is not None
    native = jax.lax.convert_element_type
    original = native.__code__
    try:
        native.__code__ = original.replace(
            co_consts=(*original.co_consts, "unexecuted-native-change")
        )
        assert jax.lax.convert_element_type is native
        with pytest.raises(ValueError):
            lease.assert_current()
        changed = provenance.CallableCastLease(function)
        assert changed.proof is None
        changed.close()
    finally:
        native.__code__ = original
        lease.close()


@pytest.mark.parametrize(
    "fault", ["implementation-binding", "primitive-implementation"]
)
def test_native_conversion_implementation_and_primitive_are_pinned(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jax._src.lax import lax

    lease = provenance.CallableCastLease(_factory())
    assert lease.proof is not None
    calls = []

    def foreign(*args: Any, **kwargs: Any) -> Any:
        calls.append("called")
        raise AssertionError("changed native implementation invoked")

    with monkeypatch.context() as patch:
        if fault == "implementation-binding":
            patch.setattr(lax, "_convert_element_type", foreign)
        else:
            patch.setattr(lax.convert_element_type_p, "impl", foreign)
        with pytest.raises(ValueError):
            lease.assert_current()
    lease.assert_current()
    lease.close()
    assert not calls


def test_hostile_dtype_capture_has_no_foreign_callbacks() -> None:
    calls = []

    class Hostile:
        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            raise AssertionError("foreign equality")

        def __getattribute__(self, name: str) -> Any:
            calls.append(name)
            raise AssertionError("foreign attribute")

    target = Hostile()

    def original(value: Any) -> Any:
        return jax.lax.convert_element_type(value, np.dtype(target))

    lease = provenance.CallableCastLease(original)
    assert lease.proof is None and not calls
    lease.close()


def test_actual_native_constructor_and_same_client_target_are_acquisition_only() -> (
    None
):
    with jax.enable_x64(True):
        lease = compat.begin_cpu_cast_context()
        try:
            acquired = compat.acquire_cpu_cast_metadata(lease)
            values = dict(acquired.fields)
            assert values["acquisition"] in (
                "retained-topology",
                "same-client-empty-query",
            )
            assert values["work"] == (
                "compile=1,serialize-executable=1,execute=0"
                if values["jax"] == "0.10.2"
                else "compile=0,serialize-executable=0,execute=0"
            )
            assert (
                values["native_build_authority"]
                == "unavailable-observed-file-digest-only"
            )
            assert (
                values["options_scope"]
                == "acquisition-construction-not-original-source"
            )
            assert values["options_detailed_logging"] == "false-acquisition-choice"
            assert (
                values["original_logging"]
                == values["original_resolved_shardings"]
                == "unavailable"
            )
            assert metadata.project_cpu_cast_options(acquired.options) == tuple(
                (key, value)
                for key, value in acquired.fields
                if key in dict(metadata.project_cpu_cast_options(acquired.options))
            )
            assert acquired.options and acquired.target_payload and values["triple"]
            with pytest.raises(ValueError, match="single-use"):
                compat.acquire_cpu_cast_metadata(lease)
        finally:
            lease.close()
        assert not lease._functions and not lease._configs and lease._snapshot is None


def test_a_replaced_native_constructor_is_never_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jax._src import compiler

    calls = []

    def foreign(*args: Any, **kwargs: Any) -> Any:
        calls.append("called")
        raise AssertionError("foreign native constructor")

    with jax.enable_x64(True), monkeypatch.context() as patch:
        patch.setattr(compiler, "get_compile_options", foreign)
        with pytest.raises(ValueError):
            compat.begin_cpu_cast_context()
    assert not calls


def test_profile_source_code_cannot_be_replaced_even_with_the_same_function_identity() -> (
    None
):
    from jax._src import compiler

    profile = compiler.get_latest_profile_version
    original = profile.__code__
    try:
        profile.__code__ = original.replace(
            co_consts=tuple(
                7 if type(value) is int and value == -1 else value
                for value in original.co_consts
            )
        )
        assert compiler.get_latest_profile_version is profile
        with jax.enable_x64(True), pytest.raises(ValueError):
            compat.begin_cpu_cast_context()
    finally:
        profile.__code__ = original


def test_first_backend_transition_does_not_accept_an_unregistered_foreign_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jax._src import xla_bridge

    with jax.enable_x64(True):
        lease = compat.begin_cpu_cast_context()
        try:
            with monkeypatch.context() as patch:
                patch.setattr(xla_bridge, "_default_backend", object())
                with pytest.raises(ValueError):
                    lease.assert_current()
            lease.assert_current()
        finally:
            lease.close()


@pytest.mark.parametrize("when", ["before-begin", "before-acquire"])
@pytest.mark.parametrize("member", ["backend", "constructor-module", "cache-helper"])
def test_actual_invoked_members_are_sealed_before_any_foreign_call(
    when: str, member: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jax._src import compiler, xla_bridge, compilation_cache
    from types import SimpleNamespace

    calls = []

    def foreign(*args: Any, **kwargs: Any) -> Any:
        calls.append("called")
        raise AssertionError("foreign native helper")

    with jax.enable_x64(True), monkeypatch.context() as patch:
        lease = compat.begin_cpu_cast_context() if when == "before-acquire" else None
        if member == "backend":
            patch.setattr(xla_bridge, "get_backend", foreign)
        elif member == "constructor-module":
            patch.setattr(compiler, "xc", SimpleNamespace(CompileOptions=foreign))
        else:
            patch.setattr(compilation_cache, "is_persistent_cache_enabled", foreign)
        try:
            with pytest.raises(ValueError):
                if lease is None:
                    compat.begin_cpu_cast_context()
                else:
                    compat.acquire_cpu_cast_metadata(lease)
        finally:
            if lease is not None:
                lease.close()
    assert not calls


@pytest.mark.parametrize(
    "fault", ["pgle", "profile", "persistent-cache", "disable-optimizations"]
)
def test_actual_nondefault_compilation_paths_are_unavailable_and_restored(
    fault: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jax._src import config, compiler
    from contextlib import ExitStack

    with jax.enable_x64(True), ExitStack() as stack, monkeypatch.context() as patch:
        if fault == "pgle":
            stack.enter_context(config.enable_pgle(True))
            stack.enter_context(config.pgle_profiling_runs(1))
        elif fault == "profile":
            stack.enter_context(config.jax_xla_profile_version(1))
        elif fault == "persistent-cache":
            stack.enter_context(config.compilation_cache_dir(str(tmp_path)))
        else:
            patch.setattr(compiler._DISABLE_MOST_OPTIMIZATIONS, "value", True)
        with pytest.raises(ValueError, match="nondefault"):
            compat.begin_cpu_cast_context()


def _wire_integer(value: int) -> bytes:
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _wire(field: int, value: int | bytes) -> bytes:
    if type(value) is int:
        return _wire_integer(field << 3) + _wire_integer(value)
    return _wire_integer((field << 3) | 2) + _wire_integer(len(value)) + value


@pytest.mark.parametrize(
    "fault",
    [
        "fast-math",
        "disabled-pass",
        "enabled-pass",
        "backend-extra",
        "env-override",
        "fdo",
        "duplicate",
        "missing",
        "truncated",
        "fixed-wire",
        "too-large",
    ],
)
def test_small_wire_projection_cannot_hide_actual_override_or_schema_attacks(
    fault: str,
) -> None:
    # Isolated protocol fixtures, not native options or source authority.
    debug = _wire(99, 0)
    build_tail = options_tail = b""
    if fault == "fast-math":
        debug = _wire(99, 1)
    elif fault == "disabled-pass":
        debug += _wire(30, b"constant_folding")
    elif fault == "enabled-pass":
        debug += _wire(124, b"custom")
    elif fault == "backend-extra":
        debug += _wire(500, _wire(1, b"target-features") + _wire(2, b"+avx512fp16"))
    elif fault == "env-override":
        options_tail = _wire(7, _wire(1, b"xla_cpu_enable_fast_math"))
    elif fault == "fdo":
        build_tail = _wire(14, b"untrusted-profile")
    elif fault == "duplicate":
        debug += _wire(99, 0)
    elif fault == "fixed-wire":
        debug = _wire_integer((99 << 3) | 1) + b"\0" * 8
    payload = _wire(3, _wire(3, debug) + build_tail) + options_tail
    if fault == "missing":
        payload = b""
    elif fault == "truncated":
        payload = payload[:-1]
    elif fault == "too-large":
        payload = b"\0" * 65537
    with pytest.raises(ValueError):
        metadata.project_cpu_cast_options(payload)


@pytest.mark.parametrize(
    "fault",
    ["closed", "finished", "params", "ineligible", "acquire-failure", "stale-callable"],
)
def test_conversion_owned_lease_lifecycle_and_unavailable_paths(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = _factory()
    with jax.enable_x64(True):
        lease = context.RootCastContextLease(
            function, has_input_params=fault == "params"
        )
        plan = _plan() if fault != "ineligible" else planning.resolve_cast_plan(None)
        if fault == "closed":
            lease.close()
            with pytest.raises(ValueError, match="closed"):
                lease.finish(plan)
            return
        if fault == "acquire-failure":

            def failure(native: Any) -> Any:
                raise RuntimeError("acquisition fixture failure")

            monkeypatch.setattr(context, "acquire_cpu_cast_metadata", failure)
        original = function.__closure__[0].cell_contents
        try:
            if fault == "stale-callable":
                function.__closure__[0].cell_contents = np.float16
            result = lease.finish(plan)
            assert result.algorithm is planning.UnresolvedAlgorithm.UNRESOLVED
            assert planning.plan_is_current(result)
            assert lease.record is not None
            assert (
                lease.record.original_logging
                == lease.record.original_input_shardings
                == lease.record.original_output_shardings
                == "UNAVAILABLE"
            )
            if fault in ("params", "ineligible", "acquire-failure", "stale-callable"):
                assert not lease.record.native_fields and len(lease.record.reasons) == 3
            if fault == "finished":
                with pytest.raises(ValueError, match="finalized"):
                    lease.finish(plan)
        finally:
            function.__closure__[0].cell_contents = original
            lease.close()
        assert lease._native is None and lease._callable._state is None


@pytest.mark.parametrize("stage", ["observer", "validator"])
def test_pre_finalize_pipeline_failures_release_the_actual_lease(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = []
    actual = context.RootCastContextLease

    class Tracked(actual):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(conversion_api, "RootCastContextLease", Tracked)

    def failure(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("original " + stage + " failure")

    with jax.enable_x64(True), pytest.raises(RuntimeError, match=stage):
        conversion_api._convert_to_ir(
            fn=_factory(),
            inputs=[jax.ShapeDtypeStruct((6,), np.float64)],
            input_params=None,
            model_name="cast",
            opset=23,
            enable_double_precision=False,
            record_primitive_calls_file=None,
            precision_policy=PrecisionPolicy.SOURCE,
            jaxpr_validator=failure if stage == "validator" else lambda *_: None,
            trace_observer=failure if stage == "observer" else None,
        )
    assert len(created) == 1 and created[0]._closed and created[0]._native is None
    assert created[0]._callable._state is None and created[0].record is None


def test_real_trace_runs_once_and_ledger_does_not_control_acquisition_or_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = conversion_api._create_ir_context
    artifacts = []
    for observe in (True, False):
        original = _factory()
        calls = []

        def trace(frame: Any, event: str, arg: Any) -> Any:
            if event == "call" and frame.f_code is original.__code__:
                calls.append("original")
            return None

        def without_ledger(**kwargs: Any) -> Any:
            ctx = create(**kwargs)
            ctx._source_structure = None
            return ctx

        with jax.enable_x64(True), monkeypatch.context() as patch:
            if not observe:
                patch.setattr(conversion_api, "_create_ir_context", without_ledger)
            before = sys.gettrace()
            sys.settrace(trace)
            try:
                model = _export(original)
            finally:
                sys.settrace(before)
        assert calls == ["original"]
        record = model.meta[context.CONTEXT_KEY]
        assert record.callable_proof is not None and record.native_fields
        assert (
            model.meta[planning.PLAN_KEY].algorithm
            is planning.UnresolvedAlgorithm.UNRESOLVED
        )
        assert [node.op_type for node in model.graph] == ["Cast"]
        artifacts.append(
            (
                ir.to_proto(model).SerializeToString(deterministic=True),
                get_precision_report(model),
                record,
            )
        )
    assert artifacts[0] == artifacts[1]


def test_noncast_export_and_immutable_result_do_not_retain_native_owners() -> None:
    def other(value: Any) -> Any:
        return -value

    with jax.enable_x64(True):
        model = _export(other)
        cast = _export(_factory())
    assert model.meta[context.CONTEXT_KEY].callable_proof is None
    assert model.meta[planning.PLAN_KEY].eligibility is None

    def visit(value: Any) -> None:
        assert not isinstance(
            value,
            (
                FunctionType,
                np.ndarray,
                compat.Jaxpr,
                compat.ClosedJaxpr,
                compat.JaxprEqn,
                compat.Var,
            ),
        )
        if (
            value is None
            or type(value) in (str, int, bool, bytes)
            or isinstance(value, Enum)
        ):
            return
        if type(value) is tuple:
            for child in value:
                visit(child)
            return
        assert is_dataclass(value)
        for field in fields(value):
            visit(getattr(value, field.name))

    visit(cast.meta[context.CONTEXT_KEY])
    visit(cast.meta[planning.PLAN_KEY])
