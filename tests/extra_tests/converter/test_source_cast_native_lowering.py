# tests/extra_tests/converter/test_source_cast_native_lowering.py

"""Actual original-trace IR/options, without compiling that source in producer."""

from dataclasses import fields, is_dataclass
from enum import Enum
import sys
import threading
from types import FunctionType
from typing import Any

import jax
import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._compat import jax as compat
from jax2onnx.converter import _source_cast_context as context
from jax2onnx.converter import _source_cast_planning as planning
from jax2onnx.converter import conversion_api
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.precision import get_precision_report


def _factory() -> Any:
    target = "float16"

    def original(value: Any) -> Any:
        return jax.lax.convert_element_type(value, np.dtype(target))

    return original


def _export(function: Any, declaration: Any) -> ir.Model:
    result = ui._to_onnx(
        function,
        [declaration],
        return_mode="ir",
        precision_policy=PrecisionPolicy.SOURCE,
    )
    assert type(result) is ir.Model
    return result


@pytest.mark.parametrize(
    "shape", [(), (1,), (2,), (4,), (8,), (17,), (19,), (65,), (3, 7), (0, 2)]
)
@pytest.mark.parametrize("profiled", [False, True])
def test_actual_original_trace_produces_native_ir_and_real_options_once(
    shape: tuple[int, ...], profiled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jax._src import compiler
    from jax._src.interpreters import pxla
    from jaxlib import _jax as native

    function = _factory()
    declaration = jax.ShapeDtypeStruct(shape, np.float64)
    original_lower = context.lower_original_cast_jaxpr
    original_acquire = context.acquire_cpu_cast_metadata
    events = []
    source_calls = []
    denied = []

    def observer(frame: Any, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is function.__code__:
            source_calls.append("source")

    def trace(frame: Any, event: str, arg: Any) -> Any:
        if event == "call" and frame.f_code is function.__code__:
            source_calls.append("source")
        return None

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        denied.append("attempt")
        raise AssertionError("source IR/options stage attempted compilation/execution")

    def acquire(*args: Any, **kwargs: Any) -> Any:
        result = original_acquire(*args, **kwargs)
        events.append(("target", dict(result.fields)["work"]))
        return result

    def lower(*args: Any, **kwargs: Any) -> Any:
        # These are unused entry points, blocked only during the added phase.
        # The old separately approved empty target query has already finished.
        with monkeypatch.context() as patch:
            patch.setattr(compiler, "compile_or_get_cached", forbidden)
            patch.setattr(compiler, "backend_compile_and_load", forbidden)
            patch.setattr(pxla.MeshComputation, "compile", forbidden)
            patch.setattr(native.Client, "compile_and_load", forbidden)
            patch.setattr(native.Client, "serialize_executable", forbidden)
            patch.setattr(native.LoadedExecutable, "execute_sharded", forbidden)
            result = original_lower(*args, **kwargs)
        events.append(("source-ir", result.resolver_requests))
        return result

    with jax.enable_x64(True), monkeypatch.context() as patch:
        patch.setattr(context, "acquire_cpu_cast_metadata", acquire)
        patch.setattr(context, "lower_original_cast_jaxpr", lower)
        before_profile, before_thread, before_trace = (
            sys.getprofile(),
            threading.getprofile(),
            sys.gettrace(),
        )
        try:
            if profiled:
                sys.setprofile(observer)
                threading.setprofile(observer)
            else:
                sys.settrace(trace)
            result = _export(function, declaration)
            assert sys.getprofile() is (observer if profiled else before_profile)
            assert threading.getprofile() is (observer if profiled else before_thread)
        finally:
            sys.setprofile(before_profile)
            threading.setprofile(before_thread)
            sys.settrace(before_trace)
    assert source_calls == ["source"] and denied == []
    record = result.meta[context.CONTEXT_KEY]
    assert record.original_lowering is not None, record.reasons
    actual = record.original_lowering
    assert actual.module.shape == shape and actual.module.operations == (
        "stablehlo.convert",
        "func.return",
    )
    assert (
        actual.resolver_requests == 1
        and actual.source_compile_requests == actual.source_execute_requests == 0
    )
    assert (
        actual.options
        and len(actual.options) <= 65536
        and len(actual.module.text.encode()) <= 65536
    )
    assert (
        record.original_logging
        == record.original_input_shardings
        == record.original_output_shardings
        == "AVAILABLE"
    )
    assert record.reasons == ()
    assert type(actual.detailed_logging) is bool
    assert len(actual.input_propagation) == len(actual.output_propagation) == 1
    assert events == [
        (
            "target",
            (
                "compile=1,serialize-executable=1,execute=0"
                if jax.__version__ == "0.10.2"
                else "compile=0,serialize-executable=0,execute=0"
            ),
        ),
        ("source-ir", 1),
    ]
    assert (
        result.meta[planning.PLAN_KEY].algorithm
        is planning.UnresolvedAlgorithm.UNRESOLVED
    )
    assert [node.op_type for node in result.graph] == ["Cast"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("_dll", object()),
        ("_sharding", object()),
        ("manual_axis_type", object()),
        ("is_ref", True),
        ("weak_type", 0),
        ("shape", (True,)),
        ("dtype", np.dtype(np.float32)),
    ],
)
def test_original_declaration_metadata_cannot_be_lost_by_normalization(
    field: str, value: Any
) -> None:
    original = jax.ShapeDtypeStruct((6,), np.float64)
    saved = object.__getattribute__(original, field)
    try:
        object.__setattr__(original, field, value)
        with pytest.raises(ValueError):
            compat.pin_cast_input_declaration([original])
    finally:
        object.__setattr__(original, field, saved)
    lease = compat.pin_cast_input_declaration([original])
    assert lease.shape == (6,)
    lease.close()


def test_captured_original_declaration_is_checked_after_source_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = jax.ShapeDtypeStruct((6,), np.float64)
    original_shape = declaration.shape
    function = _factory()

    def changed(trace: Any) -> None:
        object.__setattr__(declaration, "shape", (7,))

    with jax.enable_x64(True):
        try:
            model = conversion_api._convert_to_ir(
                fn=function,
                inputs=[declaration],
                input_params=None,
                model_name="cast",
                opset=23,
                enable_double_precision=False,
                record_primitive_calls_file=None,
                precision_policy=PrecisionPolicy.SOURCE,
                jaxpr_validator=lambda *_: None,
                trace_observer=changed,
            )
        finally:
            object.__setattr__(declaration, "shape", original_shape)
    record = model.meta[context.CONTEXT_KEY]
    assert record.original_lowering is None and record.original_logging == "UNAVAILABLE"
    assert record.reasons
    assert (
        model.meta[planning.PLAN_KEY].algorithm
        is planning.UnresolvedAlgorithm.UNRESOLVED
    )


def test_modern_original_memory_space_slot_is_pinned_when_present() -> None:
    declaration = jax.ShapeDtypeStruct((6,), np.float64)
    if "_memory_space" not in vars(type(declaration)):
        lease = compat.pin_cast_input_declaration([declaration])
        lease.close()
        return
    saved = object.__getattribute__(declaration, "_memory_space")
    try:
        object.__setattr__(declaration, "_memory_space", object())
        with pytest.raises(ValueError):
            compat.pin_cast_input_declaration([declaration])
    finally:
        object.__setattr__(declaration, "_memory_space", saved)


@pytest.mark.parametrize(
    "fault",
    [
        "resolver",
        "opaque-cache",
        "registered-rule",
        "cpu-rule",
        "test-listener",
        "dump-destination",
        "logging-helper",
    ],
)
def test_native_first_use_replacements_and_diagnostics_are_not_called(
    fault: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from jax._src import pjit, util, config, compiler
    from jax._src.interpreters import mlir
    from jax._src.lax import lax

    calls = []

    def foreign(*args: Any, **kwargs: Any) -> Any:
        calls.append("called")
        raise AssertionError("foreign native lowering callback")

    with monkeypatch.context() as patch:
        if fault == "resolver":
            patch.setattr(pjit, "_resolve_and_lower", foreign)
        elif fault == "opaque-cache":
            patch.setattr(pjit, "_pjit_lower", foreign)
        elif fault == "registered-rule":
            patch.setitem(
                mlir._lowerings,
                lax.convert_element_type_p,
                mlir.LoweringRuleEntry(foreign, True),
            )
        elif fault == "cpu-rule":
            cpu = dict.get(mlir._platform_specific_lowerings, "cpu")
            if cpu is None:
                cpu = {}
                patch.setitem(mlir._platform_specific_lowerings, "cpu", cpu)
            patch.setitem(
                cpu, lax.convert_element_type_p, mlir.LoweringRuleEntry(foreign, True)
            )
        elif fault == "test-listener":
            patch.setattr(util, "test_event_listener", foreign)
        elif fault == "logging-helper":
            patch.setattr(compiler, "use_detailed_logging", foreign)
        else:
            patch.setattr(config.jax_dump_ir_to, "value", str(tmp_path))
        with jax.enable_x64(True), pytest.raises(ValueError):
            compat.begin_original_cast_lowering(
                [jax.ShapeDtypeStruct((6,), np.float64)], "plain_cast"
            )
    assert calls == [] and list(tmp_path.iterdir()) == []


def test_ordinary_warmed_native_jit_is_preserved_and_not_used_as_options_authority() -> (
    None
):
    function = _factory()
    values = np.arange(6, dtype=np.float64)
    with jax.enable_x64(True):
        eager = np.asarray(function(values))
        compiled = jax.jit(function)
        warmed = np.asarray(compiled(values))
        model = _export(function, jax.ShapeDtypeStruct((6,), np.float64))
    assert eager.tobytes() == warmed.tobytes()
    record = model.meta[context.CONTEXT_KEY]
    assert record.original_lowering is not None, record.reasons
    assert type(record.original_lowering.preexisting_executable) is bool
    assert (
        record.original_lowering.options_scope
        == "current-native-original-trace-default-construction-not-historical-executable"
    )
    assert (
        model.meta[planning.PLAN_KEY].algorithm
        is planning.UnresolvedAlgorithm.UNRESOLVED
    )


def test_original_native_lowering_survives_ledger_off_with_identical_bytes_and_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = conversion_api._create_ir_context
    results = []
    for observe in (True, False):

        def without(**kwargs: Any) -> Any:
            ctx = create(**kwargs)
            ctx._source_structure = None
            return ctx

        with jax.enable_x64(True), monkeypatch.context() as patch:
            if not observe:
                patch.setattr(conversion_api, "_create_ir_context", without)
            model = _export(_factory(), jax.ShapeDtypeStruct((6,), np.float64))
        record = model.meta[context.CONTEXT_KEY]
        assert record.original_lowering is not None, record.reasons
        results.append(
            (
                ir.to_proto(model).SerializeToString(deterministic=True),
                get_precision_report(model),
                record.original_lowering,
            )
        )
    assert results[0] == results[1]


def test_changed_profile_is_unavailable_and_is_never_replaced_by_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_acquire = context.acquire_cpu_cast_metadata

    def first(*args: Any) -> None:
        return None

    def second(*args: Any) -> None:
        return None

    def changed(*args: Any, **kwargs: Any) -> Any:
        sys.setprofile(second)
        return original_acquire(*args, **kwargs)

    before = sys.getprofile()
    try:
        sys.setprofile(first)
        with jax.enable_x64(True), monkeypatch.context() as patch:
            patch.setattr(context, "acquire_cpu_cast_metadata", changed)
            model = _export(_factory(), jax.ShapeDtypeStruct((6,), np.float64))
        assert sys.getprofile() is second
        assert model.meta[context.CONTEXT_KEY].original_lowering is None
        assert model.meta[context.CONTEXT_KEY].reasons
    finally:
        sys.setprofile(before)


def test_native_options_result_retains_only_immutable_data() -> None:
    with jax.enable_x64(True):
        record = _export(_factory(), jax.ShapeDtypeStruct((6,), np.float64)).meta[
            context.CONTEXT_KEY
        ]
    assert record.original_lowering is not None, record.reasons

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

    visit(record)


def test_cleanup_profile_change_stays_unavailable_without_overwriting_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jax2onnx._compat import _cast_metadata

    actual_close = _cast_metadata.CpuCastContext.close

    def first(*args: Any) -> None:
        return None

    def second(*args: Any) -> None:
        return None

    def cleanup(instance: Any) -> bool:
        held = actual_close(instance)
        sys.setprofile(second)  # Adversarial fixture; the adapter never sets it.
        return held

    before = sys.getprofile()
    try:
        sys.setprofile(first)
        with jax.enable_x64(True), monkeypatch.context() as patch:
            patch.setattr(_cast_metadata.CpuCastContext, "close", cleanup)
            model = _export(_factory(), jax.ShapeDtypeStruct((6,), np.float64))
        record = model.meta[context.CONTEXT_KEY]
        assert sys.getprofile() is second
        assert (
            record.original_lowering is None
            and record.original_logging == "UNAVAILABLE"
        )
        assert "ambient-profile-changed-during-cleanup" in record.reasons
        plan = model.meta[planning.PLAN_KEY]
        assert plan.algorithm is planning.UnresolvedAlgorithm.UNRESOLVED
        assert plan.observation is not None
        assert not any(
            key.startswith("original_native_") for key, _ in plan.observation.options
        )
        assert (
            dict(plan.observation.options)["original_availability"]
            == "failed-during-cleanup"
        )
        assert planning.plan_is_current(plan)
    finally:
        sys.setprofile(before)


def test_post_native_return_callable_mutation_invalidates_original_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function = _factory()
    original_target = function.__closure__[0].cell_contents
    actual = context.lower_original_cast_jaxpr
    events = []
    calls = []

    def trace(frame: Any, event: str, arg: Any) -> Any:
        if event == "call" and frame.f_code is function.__code__:
            calls.append("original")
        return None

    def change(*args: Any, **kwargs: Any) -> Any:
        result = actual(*args, **kwargs)
        events.append(
            (
                result.resolver_requests,
                result.source_compile_requests,
                result.source_execute_requests,
            )
        )
        function.__closure__[0].cell_contents = np.float16
        return result

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        with jax.enable_x64(True), monkeypatch.context() as patch:
            patch.setattr(context, "lower_original_cast_jaxpr", change)
            model = _export(function, jax.ShapeDtypeStruct((6,), np.float64))
    finally:
        function.__closure__[0].cell_contents = original_target
        sys.settrace(previous)
    assert calls == ["original"] and events == [(1, 0, 0)]
    record = model.meta[context.CONTEXT_KEY]
    plan = model.meta[planning.PLAN_KEY]
    assert record.original_lowering is None and record.callable_proof is None
    assert (
        record.original_logging
        == record.original_input_shardings
        == record.original_output_shardings
        == "UNAVAILABLE"
    )
    assert record.native_fields and record.native_options and record.target_payload
    assert record.reasons and plan.algorithm is planning.UnresolvedAlgorithm.UNRESOLVED
    assert plan.observation is not None
    assert not any(
        key.startswith("original_native_") for key, _ in plan.observation.options
    )
