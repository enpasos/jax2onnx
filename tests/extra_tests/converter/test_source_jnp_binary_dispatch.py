# tests/extra_tests/converter/test_source_jnp_binary_dispatch.py

"""Finite native SOURCE bindings; admission and target evidence stay separate."""

import ast
from collections.abc import Callable, Iterator
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx, user_interface
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.precision import (
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from tests.extra_tests.converter.source_precision_diagnostics import (
    DiagnosticValidator,
    typed_trace_manifest,
)

_GROUP = ("add", "minimum", "maximum")
_ORIGINALS = {
    name: getattr(jnp, name) for name in (*_GROUP, "sin", "where", "full", "arange")
}
# Measured 2026-10-18 for the signed-zero fixture below, x86_64 CPU, off/all.
# Pin both the unchanged native reference and the actual target payload; any
# other mismatch fails. Future genuine exact matches take the normal pass path.
_SIGNED_ZERO_TARGETS = {("0.11.2", "1.31.0"), ("0.10.2", "1.29.0")}
_SIGNED_ZERO_FAILURES = {
    ("minimum", "float32"): (
        "8981b1ce3c1ebafcb3f2bd72ee6214734a64336993b689108df617263646c2b9",
        "d0bafbde44016a69689aea24eaa924490133ce007ac95ac400633e51bfdb4620",
    ),
    ("maximum", "float32"): (
        "da136ba782e4f82fc804cc1038d87d33497d2a372dd54a702a5fc7412d90fa72",
        "d0bafbde44016a69689aea24eaa924490133ce007ac95ac400633e51bfdb4620",
    ),
    ("minimum", "float64"): (
        "24fba6049f20db437af67d662045ff593bcc1098a23359f1e381aa80b74c7f0b",
        "1e77c15524a3fa1e005a13c2f3987328d319f3007500e304e4592920caaeab8c",
    ),
    ("maximum", "float64"): (
        "2817e48c8a6f0f3e549aeb5bb938f6c8675b9f14246a781e16496138d57135a8",
        "1e77c15524a3fa1e005a13c2f3987328d319f3007500e304e4592920caaeab8c",
    ),
}


def _globals() -> dict[str, Any]:
    return {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }


def _settings() -> tuple[Any, ...]:
    return (
        bool(jax.config.x64_enabled),
        jax.config.jax_default_matmul_precision,
        jax.config.jax_numpy_dtype_promotion,
        jax.config.jax_numpy_rank_promotion,
        jax.config.jax_platforms,
    )


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    previous = _globals()
    try:
        with (
            jax.enable_x64(False),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for key, value in previous.items():
            jax.config.update(key, value)
        assert all(
            getattr(jnp, name) is original for name, original in _ORIGINALS.items()
        )
        assert active_precision_policy() is PrecisionPolicy.LEGACY


def _spec(value: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        value.shape, value.dtype, weak_type=bool(getattr(value, "weak_type", False))
    )


def _body(graph: Any) -> Any:
    inner = getattr(graph, "jaxpr", graph)
    return inner if inner is not graph else graph


def _signature(variables: Any) -> tuple[Any, ...]:
    return tuple(
        (np.dtype(var.aval.dtype).name, tuple(var.aval.shape), bool(var.aval.weak_type))
        for var in variables
    )


def _leaves(graph: Any) -> list[Any]:
    leaves = []
    for eqn in _body(graph).eqns:
        child = eqn.params.get("jaxpr", eqn.params.get("call_jaxpr"))
        if child is not None:
            leaves.extend(_leaves(child))
        elif "branches" in eqn.params:
            for branch in eqn.params["branches"]:
                leaves.extend(_leaves(branch))
        elif not eqn.primitive.name.startswith("onnx_fn::"):
            leaves.append(
                (
                    eqn.primitive.name,
                    _signature(eqn.invars),
                    _signature(eqn.outvars),
                    tuple(
                        (key, repr(value)) for key, value in sorted(eqn.params.items())
                    ),
                )
            )
    return leaves


def _target(name: str) -> Callable[..., Any]:
    def model(left: Any, right: Any) -> Any:
        return getattr(jnp, name)(left, right)

    return model


def _reference(
    fn: Callable[..., Any], values: tuple[Any, ...]
) -> tuple[Any, list[np.ndarray], list[np.ndarray]]:
    assert all(getattr(jnp, name) is original for name, original in _ORIGINALS.items())
    original = jax.make_jaxpr(fn)(*map(_spec, values))
    eager = [np.asarray(value) for value in jax.tree.leaves(fn(*values))]
    compiled = [np.asarray(value) for value in jax.tree.leaves(jax.jit(fn)(*values))]
    for a, b in zip(eager, compiled, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
    return original, eager, compiled


def _export(
    fn: Callable[..., Any],
    values: tuple[Any, ...],
    *,
    diagnostic: bool = True,
    mode: str = "proto",
    path: Path | None = None,
    named: bool = False,
) -> tuple[onnx.ModelProto, list[np.ndarray], list[np.ndarray]]:
    original, eager, compiled = _reference(fn, values)
    caller, global_before = _settings(), _globals()
    validator, seen = DiagnosticValidator(), []

    def observe(graph: Any, opset: int) -> None:
        if not seen:
            assert _signature(_body(graph).invars) == _signature(_body(original).invars)
            assert _signature(_body(graph).outvars) == _signature(
                _body(original).outvars
            )
            if not named:
                # The older-stack callback supplies a bare Jaxpr, without
                # ClosedJaxpr payloads. Shared constant bytes are checked in IR
                # and independent runtime outputs below, not invented here.
                assert typed_trace_manifest(_body(graph)) == typed_trace_manifest(
                    _body(original)
                )
        seen.append(graph)
        validator(graph, opset)

    options: dict[str, Any] = {"return_mode": mode}
    if path is not None:
        options["output_path"] = path
    if diagnostic:
        options["diagnostic_jaxpr_validator"] = observe
    artifact = user_interface._to_onnx(
        fn, list(map(_spec, values)), precision_policy=PrecisionPolicy.SOURCE, **options
    )
    if diagnostic:
        assert set(leaf for graph in seen for leaf in _leaves(graph)) == set(
            _leaves(original)
        )
    assert _settings() == caller and _globals() == global_before
    if isinstance(artifact, ir.Model):
        artifact = ir.to_proto(artifact)
    elif isinstance(artifact, str):
        artifact = onnx.load(artifact)
    proto = onnx.load_model_from_string(artifact.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None
    assert report.profile == (
        "source-diagnostic-v1" if diagnostic else "source-static-v1"
    )
    assert tuple(
        (value.dtype, value.shape, value.weak_type) for value in report.traced_outputs
    ) == _signature(_body(original).outvars)
    assert next(
        check.status for check in report.checks if check.category == "lowering"
    ) is (CheckStatus.UNVERIFIED if diagnostic else CheckStatus.VALIDATED)
    return proto, eager, compiled


def _samples(
    model: onnx.ModelProto,
    values: tuple[Any, ...],
    expected: list[np.ndarray],
    *,
    mode: str = "disabled",
    criteria: SampleCriteria = SampleCriteria("bitwise"),
) -> Any:
    report = validate_source_samples(
        model, samples=[SourceSample(values, expected, criteria)], optimization=mode
    )
    assert (
        report.target is not None and report.target.runtime_version == ort.__version__
    )
    assert [(value.dtype, value.shape) for value in report.samples[0].outputs] == [
        (value.dtype.name, value.shape) for value in expected
    ]
    return report


@pytest.mark.parametrize("source_native", [False, True])
@pytest.mark.parametrize("policy", [PrecisionPolicy.LEGACY, PrecisionPolicy.SOURCE])
def test_private_binder_opt_in_forwards_unchanged_arguments(
    source_native: bool, policy: PrecisionPolicy, monkeypatch: Any
) -> None:
    primitive = make_jnp_primitive("test.binary_source_binding")
    left, right, option = object(), object(), {"unchanged": object()}
    calls = []

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append(("native", args, kwargs))
        return option

    def bind(*args: Any, **kwargs: Any) -> Any:
        calls.append(("custom", args, kwargs))
        return option

    monkeypatch.setattr(primitive, "bind", bind)
    specs = jnp_binding_specs(
        primitive, "binary_source_binding", source_native=source_native
    )
    wrapper = next(
        spec for spec in specs if isinstance(spec, MonkeyPatchSpec)
    ).make_value(original)
    with precision_policy_scope(policy):
        assert wrapper(left, right, original_option=option) is option
    expected = (
        "native" if source_native and policy is PrecisionPolicy.SOURCE else "custom"
    )
    assert calls == [(expected, (left, right), {"original_option": option})]
    assert get_orig_impl(primitive, "binary_source_binding") is original


def test_native_error_is_propagated_once_without_retry(monkeypatch: Any) -> None:
    primitive = make_jnp_primitive("test.binary_source_error")
    failure = ValueError("native original error")
    calls = []

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        raise failure

    monkeypatch.setattr(
        primitive,
        "bind",
        lambda *args, **kwargs: pytest.fail("native error retried custom binding"),
    )
    wrapper = next(
        spec
        for spec in jnp_binding_specs(
            primitive, "binary_source_error", source_native=True
        )
        if isinstance(spec, MonkeyPatchSpec)
    ).make_value(original)
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        with pytest.raises(ValueError) as actual:
            wrapper(1, keyword=2)
    assert actual.value is failure and calls == [((1,), {"keyword": 2})]


def test_only_the_six_reviewed_callers_opt_in() -> None:
    directory = Path(__file__).resolve().parents[3] / "jax2onnx/plugins/jax/numpy"
    selected = []
    for path in directory.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "jnp_binding_specs"
            ):
                for keyword in node.keywords:
                    if keyword.arg == "source_native":
                        assert (
                            isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True
                        )
                        selected.append(path.stem)
    assert sorted(selected) == ["add", "cos", "exp", "maximum", "minimum", "sin"]


def _values(case: str) -> tuple[Any, ...]:
    if case == "weak16":
        return jnp.asarray([0.5, -1.5], np.float16), jnp.asarray(1.0)
    if case in {"integer_float", "x64_integer_float"}:
        return jnp.asarray([2, -1], np.int32), jnp.asarray([0.5, -0.5], np.float32)
    if case == "weak_weak":
        return jnp.asarray(1.25), jnp.asarray(-0.5)
    if case == "strict":
        return jnp.asarray([0.5, -1.5], np.float16), jnp.asarray(
            [1.0, -0.5], np.float32
        )
    if case in {"rank_raise", "broadcast"}:
        return jnp.asarray([0.5, -1.5], np.float32), jnp.asarray(
            [[1.0, -0.5]], np.float32
        )
    return jnp.asarray([0.5, -1.5], np.float32), jnp.asarray([1.0, -0.5], np.float32)


@pytest.mark.parametrize("name", _GROUP)
@pytest.mark.parametrize(
    "case",
    [
        "weak16",
        "integer_float",
        "weak_weak",
        "strict",
        "rank_raise",
        "x64_integer_float",
    ],
)
def test_original_promotion_weakness_and_error_contracts(name: str, case: str) -> None:
    with (
        jax.enable_x64(case == "x64_integer_float"),
        jax.numpy_dtype_promotion("strict" if case == "strict" else "standard"),
        jax.numpy_rank_promotion("raise" if case == "rank_raise" else "allow"),
    ):
        values, fn = _values(case), _target(name)
        caller, global_before = _settings(), _globals()
        if case in {"strict", "rank_raise"}:
            with pytest.raises(ValueError) as original_error:
                jax.make_jaxpr(fn)(*map(_spec, values))
            if case == "strict":
                assert type(original_error.value).__name__ == "TypePromotionError"
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                with pytest.raises(type(original_error.value)) as actual_error:
                    jax.make_jaxpr(fn)(*map(_spec, values))
            assert (
                str(actual_error.value).splitlines()[0]
                == str(original_error.value).splitlines()[0]
            )
        else:
            original, _, _ = _reference(fn, values)
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                actual = jax.make_jaxpr(fn)(*map(_spec, values))
            assert typed_trace_manifest(actual) == typed_trace_manifest(original)
            assert _signature(_body(actual).outvars)[0][2] is (case == "weak_weak")
        assert _settings() == caller and _globals() == global_before


@pytest.mark.parametrize("name", _GROUP)
@pytest.mark.parametrize(
    "case,mode",
    [
        ("same", "proto"),
        ("weak16", "ir"),
        ("integer_float", "proto"),
        ("broadcast", "file"),
    ],
)
def test_real_facade_and_unchanged_numerical_references(
    name: str, case: str, mode: str, tmp_path: Path
) -> None:
    values = _values(case)
    diagnostic = name != "add" or case == "integer_float"
    proto, eager, compiled = _export(
        _target(name),
        values,
        diagnostic=diagnostic,
        mode=mode,
        path=tmp_path / "binary.onnx" if mode == "file" else None,
    )
    for source in (eager, compiled):
        for optimization in ("disabled", "all"):
            _samples(
                proto, values, source, mode=optimization
            ).require_sample_acceptance()


def test_shared_canonical_literal_and_distinct_consumer_casts() -> None:
    with jax.enable_x64(True):
        literal = np.asarray([16777217, 2147483647], np.int32)
        original_bytes = literal.tobytes()
        values = (jnp.asarray([0.5, -1.5], np.float16), jnp.asarray(1.0))

        def model(left: Any, weak: Any) -> Any:
            return (
                jnp.add(left, weak),
                jnp.minimum(left, weak),
                jnp.maximum(left, weak),
                literal,
                jnp.add(literal, np.int32(1)),
                jnp.add(literal, np.float64(0)),
                jnp.minimum(literal, np.float64(2147483648)),
                jnp.maximum(literal, np.float64(0)),
            )

        proto, eager, _ = _export(model, values)
        assert (
            literal.tobytes() == original_bytes and eager[3].tobytes() == original_bytes
        )
        assert eager[5].dtype == np.float64 and eager[5][0] == 16777217
        for optimization in ("disabled", "all"):
            _samples(
                proto, values, eager, mode=optimization
            ).require_sample_acceptance()
        canonical = [
            onnx.numpy_helper.to_array(value)
            for value in proto.graph.initializer
            if value.data_type == onnx.TensorProto.INT32 and tuple(value.dims) == (2,)
        ]
        assert any(value.tobytes() == original_bytes for value in canonical)


@onnx_function
def _named_group(left: Any, right: Any) -> Any:
    return jnp.add(left, right), jnp.minimum(left, right), jnp.maximum(left, right)


@onnx_function
def _nested_group(left: Any, right: Any) -> Any:
    with jax.numpy_dtype_promotion("standard"), jax.numpy_rank_promotion("allow"):
        return _named_group(left, right)


@pytest.mark.parametrize("scope", ["nested", "cond"])
def test_function_local_settings_and_control_flow(scope: str) -> None:
    values = (
        jnp.asarray([0.5, -1.5], np.float16),
        jnp.asarray([[1.0, -0.5]], np.float32),
    )

    def model(left: Any, right: Any) -> Any:
        if scope == "nested":
            return _nested_group(left, right)
        return jax.lax.cond(
            left[0] > np.float16(0),
            lambda args: _nested_group(*args),
            lambda args: _nested_group(*args),
            (left, right),
        )

    with (
        jax.numpy_dtype_promotion("strict"),
        jax.numpy_rank_promotion("raise"),
        jax.default_matmul_precision("highest"),
    ):
        proto, eager, _ = _export(model, values, named=True)
        assert len(proto.functions) == 2
        _samples(proto, values, eager).require_sample_acceptance()


@pytest.mark.parametrize("reverse", [False, True])
def test_native_legacy_source_cache_orders_without_clearing(reverse: bool) -> None:
    values = _values("same")
    fn = _named_group
    sequence = ("native", "legacy", "source", "native")
    for phase in reversed(sequence) if reverse else sequence:
        if phase == "native":
            original, eager, compiled = _reference(fn, values)
            assert {leaf[0] for leaf in _leaves(original)} >= {"add", "min", "max"}
            for a, b in zip(eager, compiled, strict=True):
                assert a.tobytes() == b.tobytes()
        elif phase == "legacy":
            expected = [np.asarray(value) for value in fn(*values)]
            _samples(
                to_onnx(fn, list(map(_spec, values))), values, expected
            ).require_sample_acceptance()
        else:
            proto, eager, _ = _export(fn, values, named=True)
            _samples(proto, values, eager).require_sample_acceptance()


@pytest.mark.parametrize("name", _GROUP)
@pytest.mark.parametrize("transform", ["grad", "vmap"])
def test_native_gradient_and_batching_paths(
    name: str, transform: str, record_property: Any
) -> None:
    operation = _target(name)
    if transform == "grad":
        values = (jnp.asarray(0.25, np.float32), jnp.asarray(1.0, np.float32))
        fn = jax.grad(operation, argnums=0)
    else:
        values = (
            jnp.asarray([[0.25, -1.0], [2.0, 0.5]], np.float32),
            jnp.asarray([[1.0, 0.5], [-1.0, 2.0]], np.float32),
        )
        fn = jax.vmap(operation)
    original, _, _ = _reference(fn, values)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        actual = jax.make_jaxpr(fn)(*map(_spec, values))
    assert typed_trace_manifest(actual) == typed_trace_manifest(original)
    proto, eager, compiled = _export(fn, values)
    for reference_kind, source in (("eager", eager), ("jit", compiled)):
        for optimization in ("disabled", "all"):
            report = _samples(proto, values, source, mode=optimization)
            report.require_sample_acceptance()
            record_property(
                f"{reference_kind}-{optimization}",
                json.dumps(
                    {
                        "target": asdict(report.target),
                        "status": report.status.value,
                        "artifact": report.artifact_sha256,
                        "samples": [asdict(sample) for sample in report.samples],
                    }
                ),
            )


@pytest.mark.parametrize("name", _GROUP)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("probe", ["nonfinite", "signed_zero"])
def test_original_special_value_criteria(
    name: str, dtype: Any, probe: str, record_property: Any
) -> None:
    with jax.enable_x64(dtype is np.float64):
        if probe == "nonfinite":
            values = (
                jnp.asarray([np.nan, 1.0, np.nan, np.inf, -np.inf], dtype),
                jnp.asarray([1.0, np.nan, np.nan, 1.0, 1.0], dtype),
            )
        else:
            values = (
                jnp.asarray([-0.0, 0.0, -0.0, 0.0], dtype),
                jnp.asarray([0.0, -0.0, -0.0, 0.0], dtype),
            )
        criteria = SampleCriteria("exact" if probe == "nonfinite" else "bitwise")
        proto, eager, compiled = _export(_target(name), values)
        for optimization in ("disabled", "all"):
            report = _samples(
                proto, values, eager, mode=optimization, criteria=criteria
            )
            record_property(
                optimization,
                json.dumps(
                    {
                        "target": asdict(report.target),
                        "status": report.status.value,
                        "criteria": asdict(criteria),
                        "artifact": report.artifact_sha256,
                        "samples": [asdict(sample) for sample in report.samples],
                        "source_x64": bool(jax.config.x64_enabled),
                    }
                ),
            )
            if report.status is CheckStatus.FAILED:
                assert (
                    probe == "signed_zero"
                    and (name, np.dtype(dtype).name) in _SIGNED_ZERO_FAILURES
                )
                assert (jax.__version__, ort.__version__) in _SIGNED_ZERO_TARGETS
                assert (
                    report.target is not None
                    and report.target.architecture == "x86_64"
                    and report.target.providers == ("CPUExecutionProvider",)
                )
                assert all(
                    check.status is CheckStatus.VALIDATED for check in report.checks
                )
                assert [
                    (check.category, check.status) for check in report.samples[0].checks
                ] == [
                    ("runtime-execution", CheckStatus.VALIDATED),
                    ("numerical-acceptance", CheckStatus.FAILED),
                ]
                reference_hash, actual_hash = _SIGNED_ZERO_FAILURES[
                    (name, np.dtype(dtype).name)
                ]
                assert report.samples[0].references[0].sha256 == reference_hash
                assert report.samples[0].outputs[0].sha256 == actual_hash
                assert eager[0].tobytes() == compiled[0].tobytes()
                with pytest.raises(SampleAcceptanceError, match="bitwise"):
                    report.require_sample_acceptance()
                continue
            report.require_sample_acceptance()
            _samples(
                proto, values, compiled, mode=optimization, criteria=criteria
            ).require_sample_acceptance()


def test_nonselected_binder_and_existing_profile_exclusions_remain() -> None:
    weak = jnp.asarray(1.0)
    for policy in (PrecisionPolicy.LEGACY, PrecisionPolicy.SOURCE):
        with precision_policy_scope(policy), _activate_plugin_worlds():
            trace = jax.make_jaxpr(lambda x: jnp.exp(x))(_spec(weak))
        native = policy is PrecisionPolicy.SOURCE
        assert _leaves(trace)[0][0] == ("exp" if native else "jax.numpy.exp")
        assert _signature(_body(trace).outvars) == (("float32", (), native),)
    for name in ("minimum", "maximum"):
        with pytest.raises(SourcePrecisionError) as failure:
            user_interface._to_onnx(
                _target(name),
                list(map(_spec, _values("same"))),
                precision_policy=PrecisionPolicy.SOURCE,
            )
        assert failure.value.category == "profile-exclusion"
        assert failure.value.primitive in {"min", "max"}


def test_where_full_adapters_and_dynamic_arange_compatibility_remain() -> None:
    def adapters(c: Any, a: Any, b: Any) -> Any:
        return jnp.where(c, a, b), jnp.full((2,), 1.0)

    values = (
        jnp.asarray([True, False]),
        jnp.asarray([True, True]),
        jnp.asarray([False, False]),
    )
    _reference(adapters, values)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        trace = jax.make_jaxpr(adapters)(*map(_spec, values))
    assert {leaf[0] for leaf in _leaves(trace)} >= {"select_n", "jax.numpy.full"}
    native_where = jax.make_jaxpr(_ORIGINALS["where"])(*map(_spec, values))
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        source_where = jax.make_jaxpr(jnp.where)(*map(_spec, values))
    assert typed_trace_manifest(source_where) == typed_trace_manifest(native_where)

    def dynamic(start: Any) -> Any:
        return jnp.arange(start, start + 3, dtype=jnp.float32)

    argument = jnp.asarray(2.0, np.float32)
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.make_jaxpr(dynamic)(_spec(argument))
    with pytest.raises(jax.errors.ConcretizationTypeError):
        user_interface._to_onnx(
            dynamic,
            [_spec(argument)],
            precision_policy=PrecisionPolicy.SOURCE,
            diagnostic_jaxpr_validator=DiagnosticValidator(),
        )
    # Existing exporter capability is retained in LEGACY, not retired because
    # the native SOURCE call cannot abstract-trace these dynamic bounds.
    legacy = to_onnx(dynamic, [_spec(argument)])
    _samples(
        legacy, (argument,), [np.asarray([2.0, 3.0, 4.0], np.float32)]
    ).require_sample_acceptance()


def test_original_ufunc_keyword_errors_and_nested_restoration() -> None:
    values = _values("same")
    for name in _GROUP:
        original = _ORIGINALS[name]
        with pytest.raises(TypeError) as expected:
            original(*values, unknown_keyword=True)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            outer = getattr(jnp, name)
            with pytest.raises(type(expected.value)):
                with _activate_plugin_worlds():
                    getattr(jnp, name)(*values, unknown_keyword=True)
            assert getattr(jnp, name) is outer
        assert getattr(jnp, name) is original
