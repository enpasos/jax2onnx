# tests/extra_tests/converter/test_source_named_parameter_facade.py

"""Audited named runtime operands and original-source preparation stay aligned."""

from collections.abc import Iterator
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx, user_interface as ui
from jax2onnx.converter import conversion_api
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)
from tests.extra_tests.converter.source_precision_support import (
    SourceScenario,
    SourceSettings,
    _global_source_configuration,
    prepare_source_case,
    run_source_case,
)
from tests.extra_tests.converter.test_source_dropout_parameters import (
    DropoutDiagnosticValidator,
)


_BITWISE = SampleCriteria("bitwise")


@pytest.fixture(autouse=True)
def _configuration() -> Iterator[None]:
    previous = _global_source_configuration()
    try:
        yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)


def _source(fn: Any, inputs: Any, params: Any, **options: Any) -> Any:
    return ui._to_onnx(
        fn,
        inputs,
        input_params=params,
        precision_policy=PrecisionPolicy.SOURCE,
        **options,
    )


@onnx_function(name="NamedNumericBinding")
def _numeric(x: Any, *, value: Any) -> Any:
    return x + value


def _controlled(x: Any, *, value: Any, flag: Any) -> Any:
    return jax.lax.cond(
        flag,
        lambda operand: _numeric(operand, value=value),
        lambda operand: operand * value,
        x,
    )


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("value", [True, 7, 1.5, complex(1, 2)])
def test_python_scalar_dtype_and_weakness_come_from_jax(value: Any, x64: bool) -> None:
    with jax.enable_x64(x64):
        expected = jax.eval_shape(lambda operand: operand, value)
        artifact = _source(lambda *, operand: operand, [], {"operand": value})
        report = get_precision_report(artifact)
        assert report is not None
        binding = report.inputs[0]
        assert (
            binding.supplied.dtype
            == binding.traced.dtype
            == np.dtype(expected.dtype).name
        )
        assert (
            binding.supplied.weak_type == binding.traced.weak_type == expected.weak_type
        )
        assert binding.exported.name == "operand"
        assert artifact.graph.output[0].name == "operand"


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "int32",
        "int64",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    ],
)
@pytest.mark.parametrize("mode", ["proto", "ir", "file"])
def test_named_tensor_ports_keep_types_order_and_aliases(
    dtype: str, mode: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        params = {
            "second": jax.ShapeDtypeStruct((2, 3), dtype),
            "first": jax.ShapeDtypeStruct((), dtype),
        }
        model = _source(
            lambda *, first, second: (second, first, second),
            [],
            params,
            return_mode=mode,
            output_path=tmp_path / "named.onnx",
            output_names=["matrix", "scalar", "matrix_alias"],
        )
        proto = (
            ir.to_proto(model)
            if isinstance(model, ir.Model)
            else onnx.load(model) if isinstance(model, str) else model
        )
        report = get_precision_report(model)
        assert report is not None
        assert [entry.exported.name for entry in report.inputs] == ["second", "first"]
        assert [entry.traced.dtype for entry in report.inputs] == [dtype, dtype]
        assert [entry.traced.weak_type for entry in report.inputs] == [False, False]
        assert [entry.name for entry in proto.graph.input] == ["second", "first"]
        onnx.checker.check_model(proto, full_check=True)


@pytest.mark.parametrize("scope", ["root", "function", "if"])
def test_one_symbolic_artifact_keeps_actual_named_bindings(
    scope: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Freeze finite representable native eager/JIT references and criterion before export.
    fn = {
        "root": lambda x, *, value, flag: jax.lax.select(flag, x + value, x * value),
        "function": lambda x, *, value, flag: jax.lax.select(
            flag, _numeric(x, value=value), x * value
        ),
        "if": _controlled,
    }[scope]
    references = []
    feeds = []
    for batch in (2, 5):
        for flag in (True, False, True):
            x = np.arange(batch * 3, dtype=np.float32).reshape(batch, 3)
            value = np.full((batch, 3), 2, np.float32)
            feed = {"features": x, "value": value, "flag": np.asarray(flag)}
            eager = np.asarray(
                fn(jnp.asarray(x), value=jnp.asarray(value), flag=jnp.asarray(flag))
            )
            compiled = np.asarray(
                jax.jit(fn)(
                    jnp.asarray(x), value=jnp.asarray(value), flag=jnp.asarray(flag)
                )
            )
            assert eager.tobytes() == compiled.tobytes()
            feeds.append(feed)
            references.append(eager)
    evidence = Path(os.environ.get("JAX2ONNX_NAMED_EVIDENCE_DIR", tmp_path)) / scope
    evidence.mkdir(parents=True, exist_ok=True)
    np.savez(
        evidence / "native-before-export.npz",
        **{f"y{i}": value for i, value in enumerate(references)},
        **{
            f"feed{i}_{name}": value
            for i, feed in enumerate(feeds)
            for name, value in feed.items()
        },
    )
    before = SourceSettings.observe()
    (batch,) = export.symbolic_shape("B")
    traces = []
    original = conversion_api._trace_to_jaxpr

    def observe(**kwargs: Any) -> Any:
        traced = original(**kwargs)
        traces.append(traced)
        return traced

    monkeypatch.setattr(conversion_api, "_trace_to_jaxpr", observe)
    validator_calls = []
    real_validator = ui.validate_source_jaxpr

    def validate(jaxpr: Any, opset: int) -> None:
        validator_calls.append([eqn.primitive.name for eqn in jaxpr.eqns])
        real_validator(jaxpr, opset)

    monkeypatch.setattr(ui, "validate_source_jaxpr", validate)
    manifest = {
        "criterion": asdict(_BITWISE),
        "settings": asdict(before),
        "hidden_global_settings": _global_source_configuration(),
        "scope": scope,
        "reference_sha256": hashlib.sha256(
            (evidence / "native-before-export.npz").read_bytes()
        ).hexdigest(),
    }
    (evidence / "prefreeze.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2)
    )
    artifact = _source(
        fn,
        [jax.ShapeDtypeStruct((batch, 3), np.float32)],
        {"value": jax.ShapeDtypeStruct((batch, 3), np.float32), "flag": True},
        input_names=["features"],
    )
    payload = artifact.SerializeToString()
    (evidence / "model.onnx").write_bytes(payload)
    report = get_precision_report(artifact)
    assert report is not None
    (evidence / "report.json").write_text(precision_report_to_json(report))
    assert len(traces) == 1
    trace = traces[0]
    assert trace.frozen_params == {}
    assert [binding.keyword_name for binding in trace.source_inputs] == [
        None,
        "value",
        "flag",
    ]
    assert all(
        binding.var is var
        for binding, var in zip(trace.source_inputs, trace.jaxpr.invars, strict=True)
    )
    assert trace.sds_list[0].shape[0] is batch and trace.sds_list[1].shape[0] is batch
    for optimization in ("disabled", "all"):
        result = validate_source_samples(
            artifact,
            samples=[
                SourceSample(feed, (reference,), _BITWISE)
                for feed, reference in zip(feeds, references, strict=True)
            ],
            optimization=optimization,
        )
        assert all(
            check.status is CheckStatus.VALIDATED
            for sample in result.samples
            for check in sample.checks
            if check.category == "numerical-acceptance"
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            if optimization == "disabled"
            else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session = ort.InferenceSession(
            payload, options, providers=["CPUExecutionProvider"]
        )
        actual = [session.run(None, feed)[0] for feed in feeds]
        assert all(
            y.tobytes() == ref.tobytes()
            for y, ref in zip(actual, references, strict=True)
        )
        np.savez(
            evidence / f"actual-{optimization}.npz",
            **{f"y{i}": value for i, value in enumerate(actual)},
        )
    assert payload == artifact.SerializeToString()
    assert SourceSettings.observe() == before
    (evidence / "receipt.json").write_text(
        json.dumps(
            {
                **manifest,
                "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                "actual_variable_bindings": True,
                "root_captures": len(traces),
                "validator_invocations": validator_calls,
                "bindings": [
                    {
                        "trace_index": binding.trace_index,
                        "positional_index": binding.positional_index,
                        "keyword_name": binding.keyword_name,
                        "var": str(binding.var),
                        "actual_var_identity_matches": binding.var
                        is trace.jaxpr.invars[binding.trace_index],
                    }
                    for binding in trace.source_inputs
                ],
                "executions_per_session": len(feeds),
            },
            sort_keys=True,
            indent=2,
        )
    )


@pytest.mark.parametrize("corruption", ["dtype", "rank", "missing", "extra"])
def test_invalid_named_samples_fail_before_runtime(
    corruption: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _source(
        lambda *, value: value, [], {"value": jax.ShapeDtypeStruct((2,), np.int32)}
    )
    feeds = {"value": np.asarray([1, 2], np.int32)}
    if corruption == "dtype":
        feeds["value"] = feeds["value"].astype(np.float32)
    elif corruption == "rank":
        feeds["value"] = feeds["value"].reshape(1, 2)
    elif corruption == "missing":
        feeds.clear()
    else:
        feeds["extra"] = np.asarray(1, np.int32)
    monkeypatch.setattr(
        ort,
        "InferenceSession",
        lambda *args, **kwargs: pytest.fail("invalid sample reached ORT"),
    )
    result = validate_source_samples(
        artifact,
        samples=[SourceSample(feeds, (np.asarray([1, 2], np.int32),), _BITWISE)],
        optimization="disabled",
    )
    assert any(
        check.status is CheckStatus.FAILED
        for sample in result.samples
        for check in sample.checks
    )


def test_native_scope_collision_fails_before_callable() -> None:
    (first,) = export.symbolic_shape("B")
    (foreign,) = export.symbolic_shape("B")
    with pytest.raises(SourcePrecisionError, match="scope"):
        _source(
            lambda *args, **kwargs: pytest.fail("foreign scope reached callable"),
            [jax.ShapeDtypeStruct((first, 3), np.float32)],
            {"value": jax.ShapeDtypeStruct((foreign, 3), np.float32)},
        )


@pytest.mark.parametrize("kind", ["missing", "unknown", "positional-only", "duplicate"])
def test_python_callable_binding_errors_stay_real(kind: str) -> None:
    def positional_only(value: Any, /) -> Any:
        return value

    fn, inputs, params, message = {
        "missing": (lambda *, value: value, [], {}, "missing"),
        "unknown": (lambda: np.float32(1), [], {"value": 1}, "unexpected"),
        "positional-only": (positional_only, [], {"value": 1}, "positional"),
        "duplicate": (
            lambda value: value,
            [jax.ShapeDtypeStruct((), np.int32)],
            {"value": 1},
            "multiple values",
        ),
    }[kind]
    with pytest.raises(TypeError, match=message):
        _source(fn, inputs, params)


def test_numeric_runtime_capture_survives_cache_interleaving() -> None:
    for dtype in (np.float32, np.float64, np.float32):
        with jax.enable_x64(True), jax.numpy_dtype_promotion("standard"):
            spec = jax.ShapeDtypeStruct((2,), dtype)
            x = np.asarray([2, 4], dtype)
            y = np.asarray([1, 2], dtype)
            np.testing.assert_array_equal(_numeric(x, value=y), x + y)
            to_onnx(lambda a: a + dtype(1), [spec])
            ui._to_onnx(
                lambda a, *, other: a + other,
                [spec],
                input_params={"other": spec},
                precision_policy=PrecisionPolicy.SOURCE,
                diagnostic_jaxpr_validator=DropoutDiagnosticValidator(),
            )
            artifact = _source(
                lambda a, *, other: _numeric(a, value=other), [spec], {"other": spec}
            )
            result = validate_source_samples(
                artifact,
                samples=[SourceSample({"in_0": x, "other": y}, (x + y,), _BITWISE)],
                optimization="disabled",
            )
            result.require_sample_acceptance()


def test_preparation_forwards_runtime_parameter_values_and_detects_payload_changes() -> (
    None
):
    metadata = {
        "callable": lambda x, *, value, flag=True: jax.lax.select(
            flag, x + value, x * value
        ),
        "input_shapes": [(2,)],
        "input_values": [np.asarray([2, 4], np.float32)],
        "input_params": {"value": np.asarray([1, 2], np.float32), "flag": False},
    }
    scenario = SourceScenario("named", False, criteria=_BITWISE)
    prepared = prepare_source_case(metadata, scenario)
    assert tuple(prepared.named_specs) == ("value", "flag")
    assert len(prepared.original_jaxpr.jaxpr.invars) == 3
    assert len(prepared.input_evidence) == 3
    result = run_source_case(metadata, scenario, _prepared=prepared)
    assert result.numerical_status is CheckStatus.VALIDATED
    changed = replace(
        prepared,
        named_arguments={**prepared.named_arguments, "flag": jnp.asarray(True)},
    )
    invalid = run_source_case(metadata, scenario, _prepared=changed)
    assert (
        invalid.stage == "source-preparation"
        and invalid.audit.status is CheckStatus.FAILED
    )
    changed_reference = replace(
        prepared, reference_outputs=(np.asarray([99, 99], np.float32),)
    )
    invalid = run_source_case(metadata, scenario, _prepared=changed_reference)
    assert (
        invalid.stage == "source-preparation"
        and invalid.audit.status is CheckStatus.FAILED
    )


@pytest.mark.parametrize(
    "declaration",
    [
        0,
        1.0,
        np.ones((1,), bool),
        np.asarray(1, np.int32),
        jax.ShapeDtypeStruct((1,), bool),
        jax.ShapeDtypeStruct((), np.int32),
    ],
)
def test_original_invalid_select_predicates_keep_source_or_profile_errors(
    declaration: Any,
) -> None:
    from tests.extra_tests.converter.test_source_runtime_parameters import _select

    with pytest.raises(
        (TypeError, ValueError), match="(?i)bool|shape|predicate|selection"
    ):
        _source(
            _select, [jax.ShapeDtypeStruct((2,), np.float32)], {"flag": declaration}
        )


def test_named_declaration_does_not_admit_symbolic_subtraction() -> None:
    with pytest.raises(SourcePrecisionError, match="symbolic operation") as captured:
        _source(
            lambda x, *, value: x - value,
            [jax.ShapeDtypeStruct(("B", 3), np.float32)],
            {"value": jax.ShapeDtypeStruct(("B", 3), np.float32)},
        )
    assert captured.value.primitive == "sub"


def test_named_disabled_x64_fails_before_trace_or_artifact(tmp_path: Path) -> None:
    with jax.enable_x64(False):
        with pytest.raises(SourcePrecisionError, match="canonicalize"):
            _source(
                lambda **kwargs: pytest.fail("dtype conflict reached callable"),
                [],
                {"wide": np.asarray(1, np.float64)},
                return_mode="file",
                output_path=tmp_path / "never.onnx",
            )
    assert not (tmp_path / "never.onnx").exists()


def test_actual_bindings_reject_same_typed_keyword_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = conversion_api._trace_to_jaxpr

    def swap(**kwargs: Any) -> Any:
        trace = real(**kwargs)
        return replace(
            trace,
            source_inputs=tuple(
                replace(binding, var=trace.source_inputs[1 - index].var)
                for index, binding in enumerate(trace.source_inputs)
            ),
        )

    monkeypatch.setattr(conversion_api, "_trace_to_jaxpr", swap)
    with pytest.raises(RuntimeError, match="root bindings"):
        _source(
            lambda *, first, second: (first, second),
            [],
            {"first": np.float32(1), "second": np.float32(2)},
        )


def test_named_metadata_payload_change_fails_before_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = {
        "callable": lambda *, flag: flag,
        "input_shapes": [],
        "input_params": {"flag": True},
    }
    scenario = SourceScenario("original", False, criteria=_BITWISE)
    prepared = prepare_source_case(metadata, scenario)
    monkeypatch.setattr(
        ui,
        "_to_onnx",
        lambda *args, **kwargs: pytest.fail("changed metadata reached export"),
    )
    result = run_source_case(
        {**metadata, "input_params": {"flag": False}}, scenario, _prepared=prepared
    )
    assert (
        result.stage == "source-preparation"
        and result.audit.status is CheckStatus.FAILED
    )


def test_named_sample_declarations_are_separate_from_actual_values() -> None:
    metadata = {"callable": lambda *, value: value, "input_shapes": []}
    value = np.asarray([2, 4], np.int32)
    scenario = SourceScenario(
        "declared",
        False,
        criteria=_BITWISE,
        input_params={"value": jax.ShapeDtypeStruct((2,), np.int32)},
        parameter_values={"value": value},
    )
    prepared = prepare_source_case(metadata, scenario)
    result = run_source_case(metadata, scenario, _prepared=prepared)
    assert result.numerical_status is CheckStatus.VALIDATED


def test_static_named_mixed_samples_check_each_feed_before_existing_session_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = np.asarray([2, 4], np.int32)
    artifact = _source(
        lambda *, value: value, [], {"value": jax.ShapeDtypeStruct((2,), np.int32)}
    )
    real = ort.InferenceSession
    calls = {"sessions": 0, "runs": 0}

    class CountedSession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls["sessions"] += 1
            self.original = real(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.original, name)

        def run(self, *args: Any, **kwargs: Any) -> Any:
            calls["runs"] += 1
            return self.original.run(*args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", CountedSession)
    samples = [
        SourceSample({"wrong": value}, (value,), _BITWISE),
        SourceSample({"value": value}, (value,), _BITWISE),
        SourceSample({"value": value.astype(np.float32)}, (value,), _BITWISE),
        SourceSample({"value": value.reshape(1, 2)}, (value,), _BITWISE),
        SourceSample({"value": np.asarray([1], np.int32)}, (value,), _BITWISE),
    ]
    result = validate_source_samples(artifact, samples=samples, optimization="disabled")
    assert calls == {"sessions": 1, "runs": 1}
    assert result.samples[1].status is CheckStatus.VALIDATED
    assert all(
        sample.status is CheckStatus.FAILED
        for index, sample in enumerate(result.samples)
        if index != 1
    )
    assert all(
        any(
            check.category == "numerical-acceptance"
            and check.status is CheckStatus.UNVERIFIED
            for check in sample.checks
        )
        for index, sample in enumerate(result.samples)
        if index != 1
    )


def test_static_complex_named_samples_use_actual_physical_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logical = np.asarray([complex(1, 2), complex(3, 4)], np.complex64)
    physical = np.stack((logical.real, logical.imag), axis=-1)
    artifact = _source(
        lambda *, value: value, [], {"value": jax.ShapeDtypeStruct((2,), np.complex64)}
    )
    report = get_precision_report(artifact)
    assert report is not None and report.inputs[0].traced.dtype == "complex64"
    assert report.inputs[0].exported.dtype == "float32"
    result = validate_source_samples(
        artifact,
        samples=[SourceSample({"value": physical}, (physical,), _BITWISE)],
        optimization="disabled",
    )
    result.require_sample_acceptance()
    monkeypatch.setattr(
        ort,
        "InferenceSession",
        lambda *args, **kwargs: pytest.fail(
            "invalid physical complex feed reached ORT"
        ),
    )
    result = validate_source_samples(
        artifact,
        samples=[
            SourceSample({"value": logical}, (physical,), _BITWISE),
            SourceSample({"value": physical.astype(np.float64)}, (physical,), _BITWISE),
        ],
        optimization="all",
    )
    assert all(sample.status is CheckStatus.FAILED for sample in result.samples)


def test_named_export_preserves_hidden_globals_under_all_caller_contexts() -> None:
    previous = _global_source_configuration()
    with (
        jax.enable_x64(True),
        jax.default_matmul_precision("highest"),
        jax.numpy_dtype_promotion("strict"),
        jax.numpy_rank_promotion("raise"),
    ):
        before = SourceSettings.observe()
        artifact = _source(lambda *, value: value, [], {"value": 1.5})
        report = get_precision_report(artifact)
        assert report is not None and report.inputs[0].traced.dtype == "float64"
        assert report.inputs[0].traced.weak_type is True
        assert SourceSettings.observe() == before
        assert _global_source_configuration() == previous
    assert _global_source_configuration() == previous


@pytest.mark.parametrize("kind", ["extent", "few", "many"])
def test_static_positional_contract_rejects_before_optional_runtime_import(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    artifact = _source(lambda value: value, [jax.ShapeDtypeStruct((2,), np.int32)], {})
    value = np.asarray([1, 2], np.int32)
    values = {"extent": (value[:1],), "few": (), "many": (value, value)}[kind]
    original = builtins.__import__

    def checked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnxruntime":
            pytest.fail("invalid static feed imported optional runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    result = validate_source_samples(
        artifact,
        samples=[SourceSample(values, (value,), _BITWISE)],
        optimization="disabled",
    )
    sample = result.samples[0]
    assert sample.outputs == ()
    assert any(
        check.category == "sample-input" and check.status is CheckStatus.FAILED
        for check in sample.checks
    )
    assert any(
        check.category == "numerical-acceptance"
        and check.status is CheckStatus.UNVERIFIED
        for check in sample.checks
    )
    from jax2onnx.precision import SampleAcceptanceError

    with pytest.raises(SampleAcceptanceError):
        result.require_sample_acceptance()


@pytest.mark.parametrize("shape", [(), (0, 3)])
def test_static_scalar_empty_and_reordered_named_feeds_keep_physical_values(
    shape: tuple[int, ...],
) -> None:
    first = np.full(shape, 2, np.int32)
    second = np.full(shape, 4, np.int32)
    # Identity references are fixed independently before the exporter is invoked.
    expected = (
        np.asarray(jax.jit(lambda x: x)(first)),
        np.asarray(jax.jit(lambda x: x)(second)),
    )
    artifact = _source(
        lambda *, first, second: (first, second),
        [],
        {
            "first": jax.ShapeDtypeStruct(shape, np.int32),
            "second": jax.ShapeDtypeStruct(shape, np.int32),
        },
    )
    result = validate_source_samples(
        artifact,
        samples=[SourceSample({"second": second, "first": first}, expected, _BITWISE)],
        optimization="all",
    )
    result.require_sample_acceptance()


@pytest.mark.parametrize("field", ["name", "dtype", "shape", "order"])
def test_static_report_mismatch_fails_before_runtime(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jax2onnx.precision import attach_precision_report

    artifact = _source(
        lambda *, first, second: (first, second),
        [],
        {
            "first": jax.ShapeDtypeStruct((2,), np.int32),
            "second": jax.ShapeDtypeStruct((2,), np.int32),
        },
    )
    report = get_precision_report(artifact)
    assert report is not None
    if field == "order":
        inputs = tuple(
            replace(entry, index=index)
            for index, entry in enumerate(reversed(report.inputs))
        )
    else:
        changed = {"name": "changed", "dtype": "float32", "shape": (3,)}[field]
        inputs = (
            replace(
                report.inputs[0],
                exported=replace(report.inputs[0].exported, **{field: changed}),
            ),
            report.inputs[1],
        )
    attach_precision_report(artifact, replace(report, inputs=inputs))
    monkeypatch.setattr(
        ort,
        "InferenceSession",
        lambda *args, **kwargs: pytest.fail("static report mismatch reached runtime"),
    )
    value = np.asarray([1, 2], np.int32)
    result = validate_source_samples(
        artifact,
        samples=[
            SourceSample({"first": value, "second": value}, (value, value), _BITWISE)
        ],
        optimization="disabled",
    )
    assert result.status is CheckStatus.FAILED and result.samples == ()
    assert any(
        check.category == "source-shape-contract" and check.status is CheckStatus.FAILED
        for check in result.checks
    )


def test_keyword_role_corruption_rejected_even_with_same_typed_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = conversion_api._trace_to_jaxpr

    def swap(**kwargs: Any) -> Any:
        trace = real(**kwargs)
        return replace(
            trace,
            source_inputs=tuple(
                replace(
                    binding, keyword_name=trace.source_inputs[1 - index].keyword_name
                )
                for index, binding in enumerate(trace.source_inputs)
            ),
        )

    monkeypatch.setattr(conversion_api, "_trace_to_jaxpr", swap)
    with pytest.raises(RuntimeError, match="roles/names/order"):
        _source(
            lambda *, first, second: (first, second),
            [],
            {"first": np.float32(1), "second": np.float32(2)},
        )


@pytest.mark.parametrize(
    ("ambient", "declared", "value"),
    [
        (False, True, 1.5),
        (True, False, 1.5),
        (False, True, np.asarray(1.5, np.float64)),
        (True, True, np.asarray(1.5, np.float64)),
    ],
)
def test_named_revalidation_uses_scenario_configuration_and_preserves_evidence(
    ambient: bool, declared: bool, value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.extra_tests.converter import source_precision_support as support

    with jax.enable_x64(ambient), jax.default_matmul_precision("highest"):
        before = SourceSettings.observe()
        hidden = _global_source_configuration()
        bodies = []

        def model(*, value: Any) -> Any:
            bodies.append(SourceSettings.observe())
            return value

        metadata = {
            "callable": model,
            "input_shapes": [],
            "input_params": {"value": value},
        }
        scenario = SourceScenario("scenario-named", declared, criteria=_BITWISE)
        prepared = prepare_source_case(metadata, scenario)
        original = support._export_prepared
        exports = []

        def observe(*args: Any, **kwargs: Any) -> Any:
            exports.append(SourceSettings.observe())
            return original(*args, **kwargs)

        monkeypatch.setattr(support, "_export_prepared", observe)
        frozen_inputs = tuple(
            np.asarray(argument).tobytes() for argument in prepared.all_arguments
        )
        frozen_references = tuple(
            array.tobytes() for array in prepared.reference_outputs
        )
        frozen_evidence = (prepared.input_evidence, prepared.reference_evidence)
        result = run_source_case(metadata, scenario, _prepared=prepared)
        assert result.numerical_status is CheckStatus.VALIDATED
        assert len(exports) == 1 and len(bodies) == 3
        assert all(settings.x64_enabled is declared for settings in [*exports, *bodies])
        report = get_precision_report(result.model)
        assert report is not None and report.caller_x64_enabled is declared
        assert report.inputs[0].traced.dtype == ("float64" if declared else "float32")
        assert report.inputs[0].traced.weak_type is (type(value) is float)
        assert (
            tuple(np.asarray(argument).tobytes() for argument in prepared.all_arguments)
            == frozen_inputs
        )
        assert (
            tuple(array.tobytes() for array in prepared.reference_outputs)
            == frozen_references
        )
        assert (prepared.input_evidence, prepared.reference_evidence) == frozen_evidence
        assert SourceSettings.observe() == before
        assert _global_source_configuration() == hidden


@pytest.mark.parametrize("ambient", [False, True])
def test_named_scenario_rejects_explicit_fp64_when_declared_x64_is_disabled(
    ambient: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.extra_tests.converter import source_precision_support as support

    with jax.enable_x64(ambient):
        before = SourceSettings.observe()
        hidden = _global_source_configuration()
        bodies = []

        def model(*, value: Any) -> Any:
            bodies.append(True)
            return value

        metadata = {
            "callable": model,
            "input_shapes": [],
            "input_params": {"value": np.asarray(1.5, np.float64)},
        }
        monkeypatch.setattr(
            support,
            "_export_prepared",
            lambda *args, **kwargs: pytest.fail(
                "invalid named scenario reached export"
            ),
        )
        result = run_source_case(
            metadata, SourceScenario("invalid-wide", False, criteria=_BITWISE)
        )
        assert (
            result.stage == "source-preparation"
            and result.audit.status is CheckStatus.FAILED
        )
        assert "canonicalize" in result.audit.detail
        assert not bodies
        assert SourceSettings.observe() == before
        assert _global_source_configuration() == hidden
