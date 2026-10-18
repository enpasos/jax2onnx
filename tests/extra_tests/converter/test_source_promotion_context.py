# tests/extra_tests/converter/test_source_promotion_context.py

"""Named functions retain actual JAX promotion contexts and caller provenance."""

from collections.abc import Callable, Iterator
from dataclasses import replace
import json
from pathlib import Path
from typing import Any
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx import user_interface
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.plugins.plugin_system import _SourceFunctionTrace
from jax2onnx.precision import (
    PRECISION_METADATA_KEY,
    PrecisionReport,
    PrecisionReportError,
    get_precision_report,
    precision_report_to_json,
)


def _settings() -> tuple[bool, Any, str, str]:
    return (
        bool(jax.config.x64_enabled),
        jax.config.jax_default_matmul_precision,
        str(jax.config.jax_numpy_dtype_promotion),
        str(jax.config.jax_numpy_rank_promotion),
    )


@pytest.fixture(autouse=True)
def _restore_jax_settings() -> Iterator[None]:
    x64, matrix, dtype, rank = _settings()
    platforms = jax.config.jax_platforms
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", x64)
        jax.config.update("jax_default_matmul_precision", matrix)
        jax.config.update("jax_numpy_dtype_promotion", dtype)
        jax.config.update("jax_numpy_rank_promotion", rank)
        jax.config.update("jax_platforms", platforms)


@onnx_function
def _mixed(value: Any) -> Any:
    return value + np.asarray(1, np.float32)


@onnx_function
def _broadcast(value: Any) -> Any:
    return value + np.asarray([1, 2], np.float32)


@onnx_function
def _nested_mixed(value: Any) -> Any:
    return _mixed(value)


@onnx_function
def _nested_broadcast(value: Any) -> Any:
    return _broadcast(value)


def _model(kind: str, scope: str, local: str) -> Callable[..., Any]:
    def callee(value: Any) -> Any:
        if kind == "dtype":
            return _nested_mixed(value) if scope == "nested" else _mixed(value)
        return _nested_broadcast(value) if scope == "nested" else _broadcast(value)

    state = jax.numpy_dtype_promotion if kind == "dtype" else jax.numpy_rank_promotion

    def target(value: Any, *extra: Any) -> Any:
        with state(local):
            if scope == "cond":
                return jax.lax.cond(
                    extra[0] > np.float32(0), callee, lambda x: -callee(x), value
                )
            return callee(value)

    return target


def _values(kind: str) -> list[np.ndarray]:
    return [
        (
            np.asarray([2048, 1], np.float16)
            if kind == "dtype"
            else np.asarray([[10, 20], [30, 40]], np.float32)
        )
    ]


def _export(
    fn: Callable[..., Any],
    values: list[np.ndarray],
    *,
    reference_first: bool = True,
    **options: Any,
) -> tuple[onnx.ModelProto, list[warnings.WarningMessage]]:
    caller = _settings()
    specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values]

    def reference() -> tuple[Any, list[np.ndarray]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            native = jax.make_jaxpr(fn)(*specs).jaxpr
            expected = [
                np.asarray(v)
                for v in jax.tree.leaves(fn(*(jnp.asarray(v) for v in values)))
            ]
        return native, expected

    native, expected = None, []
    if reference_first:
        native, expected = reference()
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        artifact = user_interface._to_onnx(
            fn, specs, precision_policy=PrecisionPolicy.SOURCE, **options
        )
    if not reference_first:
        native, expected = reference()
    assert native is not None
    assert _settings() == caller
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    report = get_precision_report(artifact)
    assert report is not None
    assert report.caller_numpy_dtype_promotion == caller[2]
    assert report.caller_numpy_rank_promotion == caller[3]
    assert tuple(value.dtype for value in report.traced_outputs) == tuple(
        str(var.aval.dtype) for var in native.outvars
    )
    if isinstance(artifact, ir.Model):
        proto = ir.to_proto(artifact)
    elif isinstance(artifact, str):
        proto = onnx.load(artifact)
    else:
        proto = artifact
    proto = onnx.load_model_from_string(proto.SerializeToString())
    assert get_precision_report(proto) == report
    onnx.checker.check_model(proto, full_check=True)
    for optimization in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = settings.inter_op_num_threads = 1
        settings.graph_optimization_level = optimization
        session = ort.InferenceSession(
            proto.SerializeToString(), settings, providers=["CPUExecutionProvider"]
        )
        actual = session.run(
            None,
            {item.name: value for item, value in zip(session.get_inputs(), values)},
        )
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected):
            assert result.dtype == reference.dtype and result.shape == reference.shape
            assert result.tobytes() == reference.tobytes()
    return proto, emitted


@pytest.mark.parametrize("kind", ["dtype", "rank"])
@pytest.mark.parametrize("scope", ["direct", "nested", "cond"])
def test_named_functions_replay_local_promotion_including_both_if_arms(
    kind: str, scope: str
) -> None:
    target = _model(kind, scope, "standard" if kind == "dtype" else "allow")
    with jax.numpy_dtype_promotion("strict"), jax.numpy_rank_promotion("raise"):
        for selector in (1, -1) if scope == "cond" else (None,):
            values = _values(kind)
            if selector is not None:
                values.append(np.asarray(selector, np.float32))
            model, _ = _export(target, values)
            assert len(model.functions) >= (2 if scope == "nested" else 1)


@pytest.mark.parametrize("mode", ["proto", "ir", "file"])
@pytest.mark.parametrize("kind", ["dtype", "rank"])
def test_new_caller_fields_round_trip_independently_of_local_function_settings(
    mode: str, kind: str, tmp_path: Path
) -> None:
    target = _model(kind, "nested", "standard" if kind == "dtype" else "allow")
    options: dict[str, Any] = {"return_mode": mode}
    if mode == "file":
        options["output_path"] = tmp_path / "promotion.onnx"
    with jax.numpy_dtype_promotion("strict"), jax.numpy_rank_promotion("raise"):
        _export(target, _values(kind), **options)


@onnx_function
def _same_dtype(value: Any) -> Any:
    return value + np.float32(1)


@pytest.mark.parametrize("kind", ["dtype", "rank"])
@pytest.mark.parametrize("reverse", [False, True])
def test_one_named_callable_specializes_for_both_promotion_contexts(
    kind: str, reverse: bool
) -> None:
    modes = ("standard", "strict") if kind == "dtype" else ("allow", "warn")
    if reverse:
        modes = modes[::-1]
    state = jax.numpy_dtype_promotion if kind == "dtype" else jax.numpy_rank_promotion

    def callee(value: Any) -> Any:
        return _same_dtype(value) if kind == "dtype" else _broadcast(value)

    def target(value: Any) -> Any:
        with state(modes[0]):
            first = callee(value)
        with state(modes[1]):
            second = callee(value)
        with state(modes[0]):
            repeated = callee(value)
        return first, second, repeated

    values = [np.asarray([1, 2], np.float32)] if kind == "dtype" else _values(kind)
    for _ in range(2):
        model, _ = _export(target, values)
        assert len(model.functions) == 2


@pytest.mark.parametrize("source_first", [False, True])
def test_named_promotion_context_cache_isolates_ordinary_legacy_and_source(
    source_first: bool,
) -> None:
    target = _model("dtype", "nested", "standard")
    values = _values("dtype")
    specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values]
    with jax.numpy_dtype_promotion("standard"), jax.numpy_rank_promotion("allow"):
        if source_first:
            _export(target, values)
        ordinary = jax.make_jaxpr(target)(*specs).jaxpr
        to_onnx(target, specs, enable_double_precision=False)
        with jax.numpy_dtype_promotion("strict"), jax.numpy_rank_promotion("raise"):
            _export(target, values)
        after = jax.make_jaxpr(target)(*specs).jaxpr
        assert tuple(v.aval for v in after.outvars) == tuple(
            v.aval for v in ordinary.outvars
        )
        np.testing.assert_array_equal(
            target(jnp.asarray(values[0])),
            np.asarray([2049, 2], np.float32),
            strict=True,
        )


@pytest.mark.parametrize("kind", ["dtype", "rank"])
@pytest.mark.parametrize("scope", ["direct", "nested", "cond"])
def test_native_strict_and_raise_rejections_are_preserved(
    kind: str, scope: str
) -> None:
    target = _model(kind, scope, "strict" if kind == "dtype" else "raise")
    values = _values(kind)
    if scope == "cond":
        values.append(np.asarray(1, np.float32))
    specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values]
    before = _settings()
    with pytest.raises((TypeError, ValueError)) as original:
        jax.make_jaxpr(target)(*specs)
    with pytest.raises(type(original.value)) as exported:
        user_interface._to_onnx(target, specs, precision_policy=PrecisionPolicy.SOURCE)
    assert ("promotion" if kind == "dtype" else "broadcast") in str(exported.value)
    assert _settings() == before
    assert active_precision_policy() is PrecisionPolicy.LEGACY


def test_rank_warn_is_observable_and_recorded_under_outer_raise() -> None:
    target = _model("rank", "nested", "warn")
    with jax.numpy_rank_promotion("raise"):
        # JAX can cache its rank-warning work. Observe a fresh shape's first
        # SOURCE trace, then compute the unpatched native reference separately.
        model, emitted = _export(
            target, [np.ones((7, 2), np.float32)], reference_first=False
        )
    assert any("automatic rank promotion" in str(item.message) for item in emitted)
    report = get_precision_report(model)
    assert report is not None
    assert report.caller_numpy_rank_promotion == "raise"
    assert any(
        item.category == "UserWarning" and "automatic rank promotion" in item.message
        for item in report.warnings
    )


@pytest.mark.parametrize("fail", [False, True])
def test_replayed_function_context_restores_every_setting_on_success_and_failure(
    fail: bool,
) -> None:
    trace = _SourceFunctionTrace((), True, "highest", "standard", "warn")
    with (
        jax.enable_x64(False),
        jax.default_matmul_precision(None),
        jax.numpy_dtype_promotion("strict"),
        jax.numpy_rank_promotion("raise"),
    ):
        before = _settings()
        try:
            with trace.activate():
                assert _settings() == (True, "highest", "standard", "warn")
                if fail:
                    raise RuntimeError("controlled body failure")
        except RuntimeError:
            assert fail
        assert _settings() == before


def _empty_report() -> PrecisionReport:
    # Existing positional construction keeps its argument order and defaults.
    return PrecisionReport("test", "test", 23, False, None, (), (), ())


def _metadata_model(data: dict[str, Any]) -> onnx.ModelProto:
    graph = onnx.helper.make_graph([], "metadata", [], [])
    model = onnx.helper.make_model(graph)
    onnx.helper.set_model_props(model, {PRECISION_METADATA_KEY: json.dumps(data)})
    return model


@pytest.mark.parametrize("missing", ["both", "dtype", "rank"])
def test_schema_one_old_artifacts_keep_unrecorded_promotion_as_none(
    missing: str,
) -> None:
    report = _empty_report()
    assert report.caller_numpy_dtype_promotion is None
    assert report.caller_numpy_rank_promotion is None
    data = json.loads(
        precision_report_to_json(
            replace(
                report,
                caller_numpy_dtype_promotion="strict",
                caller_numpy_rank_promotion="raise",
            )
        )
    )
    for kind in ("dtype", "rank"):
        if missing in ("both", kind):
            data.pop(f"caller_numpy_{kind}_promotion")
    data["unrelated_optional_extra"] = {"kept_compatible": True}
    recovered = get_precision_report(_metadata_model(data))
    assert recovered is not None and recovered.schema_version == 1
    assert recovered.caller_numpy_dtype_promotion == (
        None if missing in ("both", "dtype") else "strict"
    )
    assert recovered.caller_numpy_rank_promotion == (
        None if missing in ("both", "rank") else "raise"
    )


@pytest.mark.parametrize("dtype", [None, "standard", "strict"])
@pytest.mark.parametrize("rank", [None, "allow", "warn", "raise"])
def test_optional_promotion_values_round_trip(
    dtype: str | None, rank: str | None
) -> None:
    report = replace(
        _empty_report(),
        caller_numpy_dtype_promotion=dtype,
        caller_numpy_rank_promotion=rank,
    )
    assert (
        get_precision_report(
            _metadata_model(json.loads(precision_report_to_json(report)))
        )
        == report
    )


@pytest.mark.parametrize(
    "field", ["caller_numpy_dtype_promotion", "caller_numpy_rank_promotion"]
)
@pytest.mark.parametrize("invalid", ["unknown", 17, True, [], {}])
def test_recognized_promotion_fields_reject_invalid_values_in_constructor_and_reader(
    field: str, invalid: Any
) -> None:
    with pytest.raises(PrecisionReportError, match="promotion setting"):
        replace(_empty_report(), **{field: invalid})
    data = json.loads(precision_report_to_json(_empty_report()))
    data[field] = invalid
    with pytest.raises(PrecisionReportError, match="promotion setting"):
        get_precision_report(_metadata_model(data))
