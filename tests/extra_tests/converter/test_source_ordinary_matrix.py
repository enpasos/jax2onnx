# tests/extra_tests/converter/test_source_ordinary_matrix.py

"""Finite ordinary matrix lowering; independent source and CPU target evidence.

This diagnostic suite grants no product-profile admission. Native references
precede activation, and a valid typed MatMul is separate from numerical evidence.
"""

from collections.abc import Callable, Iterator
from copy import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
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
from jax2onnx.plugins.jax.lax.dot_general import DotGeneralPlugin
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

_ORIGINAL_MATMUL = jnp.matmul
_CANONICAL = (((1,), (0,)), ((), ()))
# These criteria precede execution. FP32 reuses the existing selected 64-term
# dot fixture's criteria; FP64 allows a conservative 64-product roundoff budget
# including absolute error near cancellation. Neither is a universal bound.
_LONG_CRITERIA = {
    "float32": SampleCriteria("tolerant", rtol=2e-6, atol=1e-5),
    "float64": SampleCriteria("tolerant", rtol=2e-14, atol=1e-13),
}
# Measured 2026-10-18, x86_64 CPU, both optimization modes, exactly the
# smallest-subnormal fixture below. Other samples/targets must pass as usual.
# Measured 2026-10-21: 8 original x86_64 CPU reports remain FAILED.
# Local audit: /tmp/jax2onnx-astra-jax012-subnormal-witness-audit-v1.json
# JAX 0.8.1 and 0.10.2 / ORT 1.31.0: 48 additional FAILED reports total
# (24 per stack: 8 matrix + 16 Min/Max), both CPU modes and raising gates.
# These failed witnesses do not establish numerical acceptance.
_SUBNORMAL_TARGETS = {
    ("0.11.2", "1.31.0"),
    ("0.10.2", "1.29.0"),
    ("0.12.0", "1.31.0"),
    ("0.10.2", "1.31.0"),
    ("0.8.1", "1.31.0"),
}
# Measured ArrayEvidence payload hashes of shape [1,1], output bits 2; this
# pins the observed deviation without requiring a future backend to retain it.
_SUBNORMAL_OUTPUTS = {
    "float32": "ac7b234ce1d59d510d1a948d88956681cbdece6c13846f54c3eec1b5e7361d45",
    "float64": "8f1e60ba7b894ea137c17f1ab6a93f769bb852466eadd4e9d0a06dbf113fa7af",
}


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    previous = {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }
    try:
        with (
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for key, value in previous.items():
            jax.config.update(key, value)


def _body(graph: Any) -> Any:
    inner = getattr(graph, "jaxpr", graph)
    return inner if inner is not graph else graph


def _signature(values: Any) -> tuple[Any, ...]:
    return tuple(
        (np.dtype(var.aval.dtype).name, tuple(var.aval.shape), var.aval.weak_type)
        for var in values
    )


def _dots(graph: Any) -> list[Any]:
    result = []
    for eqn in _body(graph).eqns:
        if eqn.primitive.name == "dot_general":
            result.append(
                (
                    _signature(eqn.invars),
                    _signature(eqn.outvars),
                    tuple(
                        (key, repr(value)) for key, value in sorted(eqn.params.items())
                    ),
                )
            )
        for key in ("jaxpr", "call_jaxpr"):
            if key in eqn.params:
                result.extend(_dots(eqn.params[key]))
        for branch in eqn.params.get("branches", ()):
            result.extend(_dots(branch))
    return result


def _spec(value: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        value.shape, value.dtype, weak_type=bool(getattr(value, "weak_type", False))
    )


def _reference(fn: Callable[..., Any], values: tuple[Any, ...]) -> tuple[Any, Any, Any]:
    assert jnp.matmul is _ORIGINAL_MATMUL
    args = tuple(jnp.asarray(value) for value in values)
    original = jax.make_jaxpr(fn)(*map(_spec, args))
    eager = [np.asarray(value) for value in jax.tree.leaves(fn(*args))]
    compiled = [np.asarray(value) for value in jax.tree.leaves(jax.jit(fn)(*args))]
    for left, right in zip(eager, compiled, strict=True):
        assert left.dtype == right.dtype and left.shape == right.shape
    return original, eager, compiled


def _export(
    fn: Callable[..., Any],
    values: tuple[Any, ...],
    *,
    opset: int = 23,
    mode: str = "proto",
    path: Path | None = None,
    named: bool = False,
) -> tuple[onnx.ModelProto, list[np.ndarray], list[np.ndarray]]:
    original, eager, compiled = _reference(fn, values)
    validator = DiagnosticValidator()
    seen = []

    def observe(graph: Any, selected_opset: int) -> None:
        if not seen:
            assert _signature(_body(graph).invars) == _signature(_body(original).invars)
            assert _signature(_body(graph).outvars) == _signature(
                _body(original).outvars
            )
            if not named:
                assert typed_trace_manifest(graph) == typed_trace_manifest(original)
        seen.append(graph)
        validator(graph, selected_opset)

    options: dict[str, Any] = {"return_mode": mode}
    if path is not None:
        options["output_path"] = path
    artifact = user_interface._to_onnx(
        fn,
        [_spec(value) for value in values],
        opset=opset,
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=observe,
        **options,
    )
    assert set(dot for graph in seen for dot in _dots(graph)) == set(_dots(original))
    if isinstance(artifact, ir.Model):
        proto = ir.to_proto(artifact)
    elif isinstance(artifact, str):
        proto = onnx.load(artifact)
    else:
        proto = artifact
    proto = onnx.load_model_from_string(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-diagnostic-v1"
    assert (
        next(check.status for check in report.checks if check.category == "lowering")
        is CheckStatus.UNVERIFIED
    )
    assert tuple(
        (out.dtype, out.shape, out.weak_type) for out in report.traced_outputs
    ) == _signature(_body(original).outvars)
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    return proto, eager, compiled


def _check_samples(
    model: onnx.ModelProto,
    values: tuple[Any, ...],
    expected: list[np.ndarray],
    *,
    criteria: SampleCriteria = SampleCriteria("bitwise"),
    optimize: str = "disabled",
) -> Any:
    report = validate_source_samples(
        model,
        samples=[SourceSample(values, expected, criteria)],
        optimization=optimize,
    )
    report.require_sample_acceptance()
    assert report.status is CheckStatus.VALIDATED
    assert (
        report.target is not None and report.target.runtime_version == ort.__version__
    )
    return report


def _dot(left: Any, right: Any, **kwargs: Any) -> Any:
    return jax.lax.dot_general(left, right, _CANONICAL, **kwargs)


def _canonical_values(
    dtype: Any, batch: tuple[int, ...] = ()
) -> tuple[np.ndarray, ...]:
    left = (np.arange(np.prod(batch + (3, 4))).reshape(batch + (3, 4)) - 7) / 4
    right = (np.arange(np.prod(batch + (4, 2))).reshape(batch + (4, 2)) - 3) / 8
    return left.astype(dtype), right.astype(dtype)


@pytest.mark.parametrize(
    "x64,dtype", [(False, np.float32), (True, np.float32), (True, np.float64)]
)
@pytest.mark.parametrize("preferred", [False, True])
@pytest.mark.parametrize("opset", [23, 26])
def test_original_b3_matrix_contract_and_shared_inputs(
    x64: bool, dtype: Any, preferred: bool, opset: int
) -> None:
    with jax.enable_x64(x64):
        values = _canonical_values(dtype)

        def model(left: Any, right: Any) -> Any:
            result = _dot(
                left, right, preferred_element_type=dtype if preferred else None
            )
            return result, left, right

        proto, eager, compiled = _export(model, values, opset=opset)
        nodes = list(proto.graph.node)
        assert [node.op_type for node in nodes].count("MatMul") == 1
        assert not {"Gemm", "Cast", "Transpose"} & {node.op_type for node in nodes}
        code = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        for value in (*proto.graph.input, *proto.graph.output, *proto.graph.value_info):
            assert value.type.tensor_type.elem_type == code
        for original in (eager, compiled):
            np.testing.assert_array_equal(
                original[0], values[0] @ values[1], strict=True
            )
            for actual, expected in zip(original[1:], values, strict=True):
                assert actual.tobytes() == expected.tobytes()
        for optimize in ("disabled", "all"):
            _check_samples(proto, values, eager, optimize=optimize)


_LAYOUTS = [
    (((0,), (0,)), ((), ())),
    (((1,), (1,)), ((), ())),
    (((0,), (1,)), ((), ())),
    (((1,), (2,)), ((2,), (0,))),
    (((0,), (1,)), ((3, 1), (2, 0))),
]


def _layout_values(dimensions: Any, dtype: Any) -> tuple[np.ndarray, ...]:
    (lc, rc), (lb, rb) = dimensions
    rank = len(lb) + 2
    lperm = lb + tuple(i for i in range(rank) if i not in lc + lb) + lc
    rperm = rb + rc + tuple(i for i in range(rank) if i not in rc + rb)
    left, right = _canonical_values(dtype, (2, 5)[: len(lb)])
    return np.transpose(left, np.argsort(lperm)), np.transpose(right, np.argsort(rperm))


@pytest.mark.parametrize("dimensions", _LAYOUTS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_noncanonical_contracting_and_batch_axes(dimensions: Any, dtype: Any) -> None:
    with jax.enable_x64(dtype is np.float64):
        values = _layout_values(dimensions, dtype)

        def model(left: Any, right: Any) -> Any:
            return jax.lax.dot_general(left, right, dimensions), left, right

        proto, eager, compiled = _export(model, values)
        assert sum(node.op_type == "Transpose" for node in proto.graph.node) in (1, 2)
        assert not {"Cast", "Gemm"} & {node.op_type for node in proto.graph.node}
        for left, right in zip(eager, compiled, strict=True):
            assert left.tobytes() == right.tobytes()
        _check_samples(proto, values, eager)


@pytest.mark.parametrize("entry", ["matmul", "operator", "dot_general"])
@pytest.mark.parametrize("batch", [(), (2,), (2, 5)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_entry_paths_and_equal_batches(
    entry: str, batch: tuple[int, ...], dtype: Any
) -> None:
    with jax.enable_x64(dtype is np.float64):
        values = _canonical_values(dtype, batch)
        dims = (((len(batch) + 1,), (len(batch),)), (tuple(range(len(batch))),) * 2)

        def model(left: Any, right: Any) -> Any:
            if entry == "matmul":
                return jnp.matmul(left, right)
            if entry == "operator":
                return left @ right
            return jax.lax.dot_general(left, right, dims)

        proto, eager, compiled = _export(model, values)
        assert any(node.op_type == "MatMul" for node in proto.graph.node)
        _check_samples(proto, values, eager)
        for left, right in zip(eager, compiled, strict=True):
            assert left.tobytes() == right.tobytes()


@onnx_function
def _named(left: Any, right: Any) -> Any:
    return jnp.matmul(left, right)


@onnx_function
def _nested(left: Any, right: Any) -> Any:
    return _named(left, right)


@onnx_function
def _named_default(left: Any, right: Any) -> Any:
    return jnp.matmul(left, right, precision=jax.lax.Precision.DEFAULT)


@onnx_function
def _named_high(left: Any, right: Any) -> Any:
    return jnp.matmul(left, right, precision=jax.lax.Precision.HIGH)


@onnx_function
def _named_highest(left: Any, right: Any) -> Any:
    return jnp.matmul(left, right, precision=jax.lax.Precision.HIGHEST)


@onnx_function
def _named_mixed_precision(left: Any, right: Any) -> Any:
    return jnp.matmul(
        left, right, precision=(jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST)
    )


@onnx_function
def _named_preset(left: Any, right: Any) -> Any:
    return jnp.matmul(left, right, precision=jax.lax.DotAlgorithmPreset.F32_F32_F32)


@onnx_function
def _named_algorithm(left: Any, right: Any) -> Any:
    return jnp.matmul(
        left, right, precision=jax.lax.DotAlgorithm(np.float32, np.float32, np.float32)
    )


@pytest.mark.parametrize("scope", ["function", "nested", "if"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_owned_constant_and_named_nested_scopes(scope: str, dtype: Any) -> None:
    with jax.enable_x64(dtype is np.float64):
        values = _canonical_values(dtype)
        weight = values[1].copy()
        original_bytes = weight.tobytes()

        def model(left: Any) -> Any:
            if scope == "function":
                result = _named(left, weight)
            elif scope == "nested":
                result = _nested(left, weight)
            else:
                result = jax.lax.cond(
                    left[0, 0] > dtype(0),
                    lambda x: _named(x, weight),
                    lambda x: _named(x, weight),
                    left,
                )
            return result, left, weight

        proto, eager, _ = _export(model, values[:1], named=True)
        assert proto.functions and weight.tobytes() == original_bytes
        assert all(not function.attribute for function in proto.functions)
        _check_samples(proto, values[:1], eager)
        assert eager[-1].tobytes() == original_bytes


@pytest.mark.parametrize("mode", ["proto", "ir", "file"])
def test_shared_facade_return_modes(mode: str, tmp_path: Path) -> None:
    proto, eager, _ = _export(
        lambda a, b: jnp.matmul(a, b),
        _canonical_values(np.float32),
        mode=mode,
        path=tmp_path / "matrix.onnx" if mode == "file" else None,
    )
    _check_samples(proto, _canonical_values(np.float32), eager)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "probe",
    [
        "mantissa",
        "cancellation",
        "finite_range",
        "subnormal",
        "nan",
        "inf",
        "signed_zero",
    ],
)
def test_exact_selected_numerical_probes(
    dtype: Any, probe: str, record_property: Any
) -> None:
    with jax.enable_x64(dtype is np.float64):
        info = np.finfo(dtype)
        cases = {
            "mantissa": (
                [2**24, 1] if dtype is np.float64 else [1 + 2**-11, 0],
                [1, 1],
            ),
            "cancellation": ([1024, -1024, 0.25], [1, 1, 1]),
            "finite_range": ([info.max / 4, info.max / 4], [1, 1]),
            "subnormal": ([info.smallest_subnormal, info.smallest_subnormal], [1, 1]),
            "nan": ([np.nan, 1], [1, 1]),
            "inf": ([np.inf, 1], [1, 1]),
            "signed_zero": ([-0.0, -0.0], [1, 1]),
        }
        left, right = cases[probe]
        values = (np.asarray([left], dtype), np.asarray(right, dtype).reshape(-1, 1))
        proto, eager, compiled = _export(lambda a, b: jnp.matmul(a, b), values)
        # Payloads of NaNs are not established by the original source contract.
        criteria = SampleCriteria("exact" if probe == "nan" else "bitwise")
        for a, b in zip(eager, compiled, strict=True):
            np.testing.assert_array_equal(a, b, strict=True)
            if probe != "nan":
                assert a.tobytes() == b.tobytes()
        for optimization in ("disabled", "all"):
            report = validate_source_samples(
                proto,
                samples=[SourceSample(values, eager, criteria)],
                optimization=optimization,
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
                        "source_default_matmul_precision": jax.config.jax_default_matmul_precision,
                    }
                ),
            )
            assert [
                (value.dtype, value.shape) for value in report.samples[0].outputs
            ] == [(value.dtype.name, value.shape) for value in eager]
            if report.status is CheckStatus.FAILED:
                assert (
                    probe == "subnormal"
                    and (jax.__version__, ort.__version__) in _SUBNORMAL_TARGETS
                    and report.target is not None
                    and report.target.architecture == "x86_64"
                    and report.target.providers == ("CPUExecutionProvider",)
                )
                assert (
                    eager[0].tobytes()
                    == compiled[0].tobytes()
                    == bytes(np.dtype(dtype).itemsize)
                )
                assert (
                    report.samples[0].outputs[0].sha256
                    == _SUBNORMAL_OUTPUTS[np.dtype(dtype).name]
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
                assert (
                    report.samples[0].checks[-1].detail
                    == "Output 0: values fail bitwise comparison"
                )
                with pytest.raises(SampleAcceptanceError, match="bitwise"):
                    report.require_sample_acceptance()
            else:
                report.require_sample_acceptance()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_declared_longer_reduction_criteria(dtype: Any, record_property: Any) -> None:
    with jax.enable_x64(dtype is np.float64):
        criteria = _LONG_CRITERIA[np.dtype(dtype).name]
        rng = np.random.default_rng(314)
        values = (
            rng.normal(size=(2, 3, 64)).astype(dtype),
            rng.normal(size=(2, 64, 4)).astype(dtype),
        )
        proto, eager, compiled = _export(lambda a, b: jnp.matmul(a, b), values)
        np.testing.assert_allclose(
            eager[0], compiled[0], rtol=criteria.rtol, atol=criteria.atol
        )
        for optimization in ("disabled", "all"):
            report = _check_samples(
                proto, values, eager, criteria=criteria, optimize=optimization
            )
            _check_samples(
                proto, values, compiled, criteria=criteria, optimize=optimization
            )
            record_property(
                optimization,
                json.dumps(
                    {
                        "target": asdict(report.target),
                        "status": report.status.value,
                        "criteria": asdict(criteria),
                        "artifact": report.artifact_sha256,
                    }
                ),
            )


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("dtypes", [(np.float16, np.float32), (np.float32, np.float64)])
def test_mixed_wrapper_preserves_native_trace_but_lowering_remains_pending(
    named: bool, dtypes: Any
) -> None:
    with jax.enable_x64(True):
        values = (np.ones((3, 4), dtypes[0]), np.ones((4, 2), dtypes[1]))
        fn = _named if named else lambda a, b: jnp.matmul(a, b)
        original, _, _ = _reference(fn, values)
        observed = []
        validator = DiagnosticValidator()

        def inspect(graph: Any, opset: int) -> None:
            validator(graph, opset)
            observed.extend(_dots(graph))

        with pytest.raises(
            NotImplementedError, match="operand/result types .*private subset"
        ):
            user_interface._to_onnx(
                fn,
                list(map(_spec, values)),
                precision_policy=PrecisionPolicy.SOURCE,
                diagnostic_jaxpr_validator=inspect,
            )
        assert observed and set(observed) == set(_dots(original))
        with jax.numpy_dtype_promotion("strict"):
            with pytest.raises(ValueError) as native_error:
                jax.make_jaxpr(fn)(*map(_spec, values))
            assert type(native_error.value).__name__ == "TypePromotionError"
            with pytest.raises(type(native_error.value)):
                user_interface._to_onnx(
                    fn,
                    list(map(_spec, values)),
                    precision_policy=PrecisionPolicy.SOURCE,
                    diagnostic_jaxpr_validator=DiagnosticValidator(),
                )


@pytest.mark.parametrize("scope", ["inline", "function", "local"])
@pytest.mark.parametrize(
    "precision",
    [
        jax.lax.Precision.DEFAULT,
        jax.lax.Precision.HIGH,
        jax.lax.Precision.HIGHEST,
        (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST),
        jax.lax.DotAlgorithmPreset.F32_F32_F32,
        jax.lax.DotAlgorithm(np.float32, np.float32, np.float32),
        "caller-highest",
    ],
)
def test_matrix_precision_requests_reach_the_lowerer(
    scope: str, precision: Any, record_property: Any
) -> None:
    values = _canonical_values(np.float32)
    requested = None if precision == "caller-highest" else precision
    callee = {
        jax.lax.Precision.DEFAULT: _named_default,
        jax.lax.Precision.HIGH: _named_high,
        jax.lax.Precision.HIGHEST: _named_highest,
        (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST): _named_mixed_precision,
        jax.lax.DotAlgorithmPreset.F32_F32_F32: _named_preset,
        jax.lax.DotAlgorithm(np.float32, np.float32, np.float32): _named_algorithm,
        "caller-highest": _named,
    }[precision]

    def fn(a: Any, b: Any) -> Any:
        if scope == "local":
            with jax.default_matmul_precision("highest"):
                return callee(a, b)
        return (
            callee(a, b)
            if scope == "function"
            else jnp.matmul(a, b, precision=requested)
        )

    with jax.default_matmul_precision(
        "highest" if precision == "caller-highest" else None
    ):
        original = jax.make_jaxpr(fn)(*map(_spec, values))
        assert _dots(original)
        # An unavailable native mandatory algorithm is honest source/runtime
        # evidence; its original trace is still compared before exclusion.
        try:
            fn(*(jnp.asarray(value) for value in values)).block_until_ready()
        except (ValueError, RuntimeError) as exc:
            record_property("native_runtime_failure", f"{type(exc).__name__}: {exc}")
        seen = []
        validator = DiagnosticValidator()

        def observe(graph: Any, opset: int) -> None:
            validator(graph, opset)
            seen.extend(_dots(graph))

        if precision == jax.lax.Precision.DEFAULT:
            proto, eager, _ = _export(fn, values, named=scope != "inline")
            _check_samples(proto, values, eager)
        else:
            with pytest.raises(NotImplementedError, match="precision .*private subset"):
                user_interface._to_onnx(
                    fn,
                    list(map(_spec, values)),
                    precision_policy=PrecisionPolicy.SOURCE,
                    diagnostic_jaxpr_validator=observe,
                )
            assert seen and set(seen) == set(_dots(original))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("named", [False, True])
def test_native_legacy_source_cache_orders_without_clearing(
    reverse: bool, named: bool
) -> None:
    values = _canonical_values(np.float32)
    fn = _named if named else lambda a, b: jnp.matmul(a, b)
    sequence = ("native", "legacy", "source", "native")
    if reverse:
        sequence = tuple(reversed(sequence))
    expected = values[0] @ values[1]
    for kind in sequence:
        if kind == "native":
            trace, eager, compiled = _reference(fn, values)
            assert _dots(trace)
            np.testing.assert_array_equal(eager[0], expected, strict=True)
            np.testing.assert_array_equal(compiled[0], expected, strict=True)
        elif kind == "legacy":
            model = to_onnx(fn, list(map(_spec, values)))
            _check_samples(model, values, [expected])
        else:
            model, eager, _ = _export(fn, values, named=named)
            _check_samples(model, values, eager)
        assert jnp.matmul is _ORIGINAL_MATMUL


def test_matmul_preserves_original_keyword_and_placement_forms() -> None:
    values = _canonical_values(np.float32)
    placement = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])

    def fn(a: Any, b: Any) -> Any:
        return jnp.matmul(
            a=a,
            b=b,
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=np.float32,
            out_sharding=None,
        )

    model, eager, _ = _export(fn, values)
    _check_samples(model, values, eager)

    # Native JAX rejects SingleDeviceSharding for this argument on both stacks.
    # Delegation must preserve that error rather than silently discard the hint.
    def invalid_placement(a: Any, b: Any) -> Any:
        return jnp.matmul(a, b, out_sharding=placement)

    with pytest.raises(TypeError) as native_error:
        jax.make_jaxpr(invalid_placement)(*map(_spec, values))
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        with pytest.raises(type(native_error.value), match="out_sharding"):
            jax.make_jaxpr(invalid_placement)(*map(_spec, values))
    weak = tuple(
        jax.lax.broadcast_in_dim(jnp.asarray(1.0), value.shape, ()) for value in values
    )
    original = jax.make_jaxpr(lambda a, b: jnp.matmul(a, b))(*map(_spec, weak))
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(lambda a, b: jnp.matmul(a, b))(*map(_spec, weak))
    assert typed_trace_manifest(original) == typed_trace_manifest(activated)


def test_explicit_fp64_input_is_rejected_before_source_narrowing() -> None:
    with jax.enable_x64(False):
        with pytest.raises((ValueError, SourcePrecisionError), match="(float64|x64)"):
            user_interface._to_onnx(
                lambda a, b: jnp.matmul(a, b),
                list(map(_spec, _canonical_values(np.float64))),
                precision_policy=PrecisionPolicy.SOURCE,
                diagnostic_jaxpr_validator=DiagnosticValidator(),
            )


@pytest.mark.parametrize(
    "defect",
    [
        "unknown",
        "out_dtype",
        "out_sharding",
        "range",
        "overlap",
        "duplicate_batch",
        "paired",
        "result",
        "zero",
        "symbolic",
        "unequal_rank",
        "multi_contract",
        "mixed",
        "wider",
        "integer",
    ],
)
def test_ordinary_local_boundary_rejects_malformed_and_pending_forms_before_emission(
    defect: str,
) -> None:
    def variable(shape: Any, dtype: Any = np.float32) -> Any:
        return SimpleNamespace(
            aval=SimpleNamespace(shape=shape, dtype=dtype, weak_type=False)
        )

    params: dict[str, Any] = {
        "dimension_numbers": _CANONICAL,
        "precision": None,
        "preferred_element_type": None,
    }
    eqn = SimpleNamespace(
        invars=[variable((3, 4)), variable((4, 2))],
        outvars=[variable((3, 2))],
        params=params,
    )
    expected = NotImplementedError
    if defect in {"unknown", "out_dtype"}:
        params["unknown" if defect == "unknown" else "out_dtype"] = np.dtype(np.float64)
    elif defect == "out_sharding":
        params["out_sharding"] = object()
    elif defect == "range":
        params["dimension_numbers"] = (((2,), (0,)), ((), ()))
        expected = ValueError
    elif defect == "overlap":
        params["dimension_numbers"] = (((1,), (0,)), ((1,), (0,)))
        expected = ValueError
    elif defect == "duplicate_batch":
        eqn.invars = [variable((2, 3, 4, 5)), variable((2, 3, 5, 6))]
        params["dimension_numbers"] = (((3,), (2,)), ((0, 0), (0, 1)))
        expected = ValueError
    elif defect == "paired":
        eqn.invars[1] = variable((5, 2))
        expected = ValueError
    elif defect == "result":
        eqn.outvars[0] = variable((2, 3))
        expected = ValueError
    elif defect in {"zero", "symbolic"}:
        eqn.invars[0] = variable((0 if defect == "zero" else "B", 4))
    elif defect == "unequal_rank":
        eqn.invars[1] = variable((1, 4, 2))
    elif defect == "multi_contract":
        params["dimension_numbers"] = (((0, 1), (0, 1)), ((), ()))
    elif defect == "mixed":
        eqn.invars[0] = variable((3, 4), np.float16)
    elif defect == "wider":
        eqn.outvars[0] = variable((3, 2), np.float64)
        params["preferred_element_type"] = np.float64
    else:
        eqn.invars = [variable((3, 4), np.int32), variable((4, 2), np.int32)]
        eqn.outvars = [variable((3, 2), np.int32)]
    with pytest.raises(expected, match="SOURCE dot_general"):
        DotGeneralPlugin()._lower_source(SimpleNamespace(), eqn)


@pytest.mark.parametrize(
    "defect", ["input_type", "input_shape", "output_type", "output_shape"]
)
def test_ordinary_physical_metadata_is_checked_without_mutating_operands(
    defect: str,
) -> None:
    values = _canonical_values(np.float32)
    eqn = jax.make_jaxpr(_dot)(*map(_spec, values)).jaxpr.eqns[0]
    sources = [
        ir.Value(
            name=f"input_{i}",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(value.shape),
        )
        for i, value in enumerate(values)
    ]
    output = ir.Value(
        name="output", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((3, 2))
    )
    if defect == "input_type":
        sources[0].type = ir.TensorType(ir.DataType.DOUBLE)
    elif defect == "input_shape":
        sources[0].shape = ir.Shape((4, 3))
    elif defect == "output_type":
        output.type = ir.TensorType(ir.DataType.DOUBLE)
    else:
        output.shape = ir.Shape((2, 3))
    before = [(copy(value.type), copy(value.shape)) for value in sources]
    ctx = SimpleNamespace(
        require_value_for_var=lambda var: sources[eqn.invars.index(var)],
        allocate_value_for_var=lambda *args, **kwargs: output,
    )
    with pytest.raises(ValueError, match="physical matrix dtype/shape"):
        DotGeneralPlugin()._lower_source(ctx, eqn)
    assert [(value.type, value.shape) for value in sources] == before
