# tests/extra_tests/converter/test_source_min_max_zero_sign.py

"""Finite zero signs, baseline preservation and honest failed target evidence."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir.traversal import RecursiveGraphIterator

from jax2onnx import onnx_function, to_onnx, user_interface as ui
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.ir_optimizations import optimize_graph
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.ir_utils import const_value_to_numpy, iter_ir_functions
from jax2onnx.plugins.jax.lax._min_max_utils import lower_source_min_max
from jax2onnx.plugins.jax.lax import min as min_module, max as max_module
from jax2onnx.precision import (
    CheckStatus,
    PRECISION_METADATA_KEY,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from tests.extra_tests.converter.source_precision_diagnostics import DiagnosticValidator

_ORIGINALS = (jnp.minimum, jnp.maximum)
_TARGETS = {("0.11.2", "1.31.0"), ("0.10.2", "1.29.0")}


def _globals() -> dict[str, Any]:
    return {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }


def _effective() -> tuple[Any, ...]:
    return tuple(getattr(jax.config, key) for key in _globals())


@pytest.fixture(autouse=True)
def _settings() -> Iterator[None]:
    previous, effective = _globals(), _effective()
    try:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for key, value in previous.items():
            jax.config.update(key, value)
        assert _effective() == effective
        assert (jnp.minimum, jnp.maximum) == _ORIGINALS
        assert active_precision_policy() is PrecisionPolicy.LEGACY


def _function(operation: str) -> Any:
    def model(x: Any, y: Any) -> Any:
        return getattr(jnp, "minimum" if operation == "Min" else "maximum")(x, y)

    return model


def _spec(value: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        value.shape, value.dtype, weak_type=bool(getattr(value, "weak_type", False))
    )


def _references(
    fn: Any, values: Any
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    assert (jnp.minimum, jnp.maximum) == _ORIGINALS
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    eager = tuple(np.asarray(v) for v in jax.tree.leaves(fn(*values)))
    compiled = tuple(np.asarray(v) for v in jax.tree.leaves(jax.jit(fn)(*values)))
    return eager, compiled


def _export(
    fn: Any, values: Any, *, mode: str = "ir", path: Path | None = None, opset: int = 23
) -> tuple[Any, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    eager, compiled = _references(fn, values)
    before = (_globals(), _effective())
    model = ui._to_onnx(
        fn,
        list(map(_spec, values)),
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
        return_mode=mode,
        output_path=path,
        opset=opset,
    )
    assert (_globals(), _effective()) == before
    proto = _proto(model)
    onnx.checker.check_model(proto, full_check=True)
    if path is not None and mode != "file":
        onnx.save(proto, path)
    report = get_precision_report(model)
    assert report is not None and report.profile == "source-diagnostic-v1"
    assert (
        next(c.status for c in report.checks if c.category == "lowering")
        is CheckStatus.UNVERIFIED
    )
    return model, eager, compiled


def _proto(model: Any) -> onnx.ModelProto:
    if isinstance(model, ir.Model):
        return ir.to_proto(model)
    return model if isinstance(model, onnx.ModelProto) else onnx.load(model)


def _bits(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(value)
    return array.view(
        {2: np.uint16, 4: np.uint32, 8: np.uint64}[array.dtype.itemsize]
    ).reshape(array.shape)


def _record(record_property: Any, label: str, report: Any, **extra: Any) -> None:
    record_property(
        label,
        json.dumps(
            {
                "artifact": report.artifact_sha256,
                "target": asdict(report.target),
                "status": report.status.value,
                "samples": [asdict(s) for s in report.samples],
                **extra,
            }
        ),
    )


def _accept(
    model: Any, values: Any, references: Any, *, record_property: Any = None
) -> None:
    for kind, expected in zip(("eager", "jit"), references, strict=True):
        for optimization in ("disabled", "all"):
            result = validate_source_samples(
                model,
                samples=[SourceSample(values, expected, SampleCriteria("bitwise"))],
                optimization=optimization,
            )
            result.require_sample_acceptance()
            if record_property is not None:
                _record(record_property, f"{kind}-{optimization}", result)


def _structure(model: Any, operation: str) -> None:
    proto = _proto(model)
    nodes = list(proto.graph.node)
    assert len(nodes) == (15 if operation == "Min" else 12)
    assert sum(n.op_type == operation for n in nodes) == 1
    assert sum(n.op_type == "GatherElements" for n in nodes) == 1
    assert not any(n.op_type == "Where" for n in nodes)
    casts = [n for n in nodes if n.op_type == "Cast"]
    assert (
        len(casts) == 1
        and next(a.i for a in casts[0].attribute if a.name == "to")
        == onnx.TensorProto.INT64
    )
    assert sorted(n.op_type for n in nodes).count("Unsqueeze") == 3
    assert len(proto.graph.initializer) == 2
    zero = next(
        v for v in proto.graph.initializer if v.data_type != onnx.TensorProto.INT64
    )
    axes = next(
        v for v in proto.graph.initializer if v.data_type == onnx.TensorProto.INT64
    )
    assert (
        onnx.numpy_helper.to_array(zero).shape == ()
        and not _bits(onnx.numpy_helper.to_array(zero)).any()
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(axes), np.array([0], np.int64), strict=True
    )


def _run(
    proto: onnx.ModelProto, values: Any, optimization: str
) -> tuple[np.ndarray, ...]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if optimization == "disabled"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = ort.InferenceSession(
        proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    session.disable_fallback()
    return tuple(
        session.run(
            None, dict(zip((v.name for v in proto.graph.input), values, strict=True))
        )
    )


@pytest.mark.parametrize("operation", ["Min", "Max"])
@pytest.mark.parametrize(
    "dtype,mode", [(np.float16, "ir"), (np.float32, "proto"), (np.float64, "file")]
)
def test_historical_zero_pairs_pass_actual_new_artifact(
    operation: str,
    dtype: Any,
    mode: str,
    tmp_path: Path,
    record_property: Any,
) -> None:
    values = (
        np.array([-0.0, 0.0, -0.0, 0.0], dtype),
        np.array([0.0, -0.0, -0.0, 0.0], dtype),
    )
    original = tuple(v.tobytes() for v in values)
    model, eager, compiled = _export(
        _function(operation), values, mode=mode, path=tmp_path / "zeros.onnx"
    )
    _structure(model, operation)
    for reference in (eager[0], compiled[0]):
        expected = np.array(
            [-0.0, -0.0, -0.0, 0.0] if operation == "Min" else [0.0, 0.0, -0.0, 0.0],
            dtype,
        )
        assert (
            reference.dtype == expected.dtype
            and reference.tobytes() == expected.tobytes()
        )
    _accept(model, values, (eager, compiled), record_property=record_property)
    assert tuple(v.tobytes() for v in values) == original


@pytest.mark.parametrize("operation", ["Min", "Max"])
def test_scalar_empty_broadcast_no_tie_and_repeated_operand(operation: str) -> None:
    scalar = (np.array(-0.0, np.float32), np.array(0.0, np.float32))
    empty = (np.zeros(0, np.float32), np.zeros(0, np.float32))
    broadcast = (
        np.array([[-0.0], [0.0], [-2.0]], np.float32),
        np.array([[0.0, -0.0, 1.0]], np.float32),
    )
    ordinary = (
        np.array([-3.0, 2.0, 8.0], np.float32),
        np.array([-1.0, 4.0, 6.0], np.float32),
    )
    repeated = np.array([-0.0, 0.0, -2.0, 3.0], np.float32)
    for values in (scalar, empty, broadcast, ordinary, (repeated, repeated)):
        model, eager, compiled = _export(_function(operation), values)
        _accept(model, values, (eager, compiled))


def _grid(dtype: Any) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    integer, exponent, mantissa = {
        np.float16: (np.uint16, 5, 10),
        np.float32: (np.uint32, 8, 23),
        np.float64: (np.uint64, 11, 52),
    }[dtype]
    sign, infinity, one = (
        1 << (exponent + mantissa),
        ((1 << exponent) - 1) << mantissa,
        ((1 << (exponent - 1)) - 1) << mantissa,
    )
    raw = [
        0,
        sign,
        1,
        sign | 1,
        2,
        sign | 2,
        (1 << mantissa) - 1,
        sign | ((1 << mantissa) - 1),
        1 << mantissa,
        sign | (1 << mantissa),
        one,
        sign | one,
        one + 1,
        sign | (one + 1),
        infinity - 1,
        sign | (infinity - 1),
        infinity,
        sign | infinity,
        infinity | (1 << (mantissa - 1)) | 0x12,
        sign | infinity | (1 << (mantissa - 1)) | 0x34,
        infinity | 1,
    ]
    vector = np.asarray(raw, dtype=integer).view(dtype)
    zero = np.array([value in (0, sign) for value in raw], bool)
    return (vector[:, None].copy(), vector[None, :].copy()), zero[:, None] & zero[
        None, :
    ]


@pytest.mark.parametrize("operation", ["Min", "Max"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_grid_obligations_are_separate_and_subnormal_mismatch_map_remains_failed(
    operation: str,
    dtype: Any,
    record_property: Any,
    tmp_path: Path,
) -> None:
    values, real_ties = _grid(dtype)
    artifact_path = tmp_path / "source-grid.onnx"
    model, eager, compiled = _export(_function(operation), values, path=artifact_path)
    _structure(model, operation)
    inspected = ir.from_proto(ir.to_proto(model))
    inspected.metadata_props.pop(PRECISION_METADATA_KEY)
    base = next(n.outputs[0] for n in inspected.graph if n.op_type == operation)
    ties = next(n.outputs[0] for n in inspected.graph if n.op_type == "And")
    inspected.graph.outputs.extend([base, ties])
    # Exercise the actual registered SOURCE optimizer with these observations
    # retained. This inspection artifact has an extended signature and no
    # source precision report; the ordinary facade artifact stays unchanged.
    optimize_graph(inspected, precision_policy=PrecisionPolicy.SOURCE)
    inspected_proto = ir.to_proto(inspected)
    onnx.checker.check_model(inspected_proto, full_check=True)
    inspection_path = tmp_path / "inspected-grid.onnx"
    onnx.save(inspected_proto, inspection_path)
    for optimization in ("disabled", "all"):
        (output,) = _run(_proto(model), values, optimization)
        observed, baseline, actual_ties = _run(inspected_proto, values, optimization)
        np.savez(
            tmp_path / f"grid-{optimization}.npz",
            x=values[0],
            y=values[1],
            eager=eager[0],
            jit=compiled[0],
            output=output,
            baseline=baseline,
            ties=actual_ties,
        )
        assert output.tobytes() == observed.tobytes()
        np.testing.assert_array_equal(actual_ties, real_ties, strict=True)
        assert np.array_equal(_bits(output)[~real_ties], _bits(baseline)[~real_ties])
        for kind, reference in (("eager", eager[0]), ("jit", compiled[0])):
            np.testing.assert_array_equal(
                np.isnan(output), np.isnan(reference), strict=True
            )
            assert np.array_equal(_bits(output)[real_ties], _bits(reference)[real_ties])
            mismatches = (_bits(output) != _bits(reference)) & ~np.isnan(reference)
            inherited = (
                (_bits(baseline) != _bits(reference))
                & ~np.isnan(reference)
                & ~real_ties
            )
            np.testing.assert_array_equal(mismatches, inherited, strict=True)
            assert np.count_nonzero(mismatches) == (0 if dtype is np.float16 else 108)
            record_property(
                f"grid-{kind}-{optimization}",
                json.dumps(
                    {
                        "artifact": hashlib.sha256(
                            _proto(model).SerializeToString()
                        ).hexdigest(),
                        "inspection_artifact": hashlib.sha256(
                            inspected_proto.SerializeToString()
                        ).hexdigest(),
                        "dtype": np.dtype(dtype).name,
                        "operation": operation,
                        "artifact_path": str(artifact_path),
                        "inspection_path": str(inspection_path),
                        "runtime_version": ort.__version__,
                        "jax_version": jax.__version__,
                        "optimization": optimization,
                        "zero_signs": "validated",
                        "baseline_bytes_outside_ties": "validated",
                        "nan_classification": "validated",
                        "native_non_nan_status": (
                            "validated" if dtype is np.float16 else "failed"
                        ),
                        "native_non_nan_mismatches": np.argwhere(mismatches).tolist(),
                    }
                ),
            )


@pytest.mark.parametrize("operation", ["Min", "Max"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_retained_signed_subnormal_failure_has_exact_artifact_and_raising_gate(
    operation: str,
    dtype: Any,
    record_property: Any,
    tmp_path: Path,
) -> None:
    assert (jax.__version__, ort.__version__) in _TARGETS
    integer = np.uint32 if dtype is np.float32 else np.uint64
    sign = 1 << (np.dtype(dtype).itemsize * 8 - 1)
    expected_bits = sign if operation == "Min" else 0
    actual_bits = expected_bits | 1
    values = (
        np.array([expected_bits], dtype=integer).view(dtype),
        np.array([actual_bits], dtype=integer).view(dtype),
    )
    before = tuple(v.tobytes() for v in values)
    artifact_path = tmp_path / "source-subnormal.onnx"
    model, eager, compiled = _export(_function(operation), values, path=artifact_path)
    assert _bits(eager[0]).tolist() == [expected_bits] == _bits(compiled[0]).tolist()
    for optimization in ("disabled", "all"):
        for kind, reference in (("eager", eager), ("jit", compiled)):
            result = validate_source_samples(
                model,
                samples=[SourceSample(values, reference, SampleCriteria("bitwise"))],
                optimization=optimization,
            )
            assert result.status is CheckStatus.FAILED
            assert (
                hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                == result.artifact_sha256
            )
            assert (
                result.target.runtime_version == ort.__version__
                and result.target.providers == ("CPUExecutionProvider",)
            )
            assert all(c.status is CheckStatus.VALIDATED for c in result.checks)
            assert [(c.category, c.status) for c in result.samples[0].checks] == [
                ("runtime-execution", CheckStatus.VALIDATED),
                ("numerical-acceptance", CheckStatus.FAILED),
            ]
            (actual,) = _run(_proto(model), values, optimization)
            assert _bits(actual).tolist() == [actual_bits]
            _record(
                record_property,
                f"subnormal-{kind}-{optimization}",
                result,
                jax_version=jax.__version__,
                artifact_path=str(artifact_path),
                operation=operation,
                reference_bits=[hex(expected_bits)],
                actual_bits=[hex(actual_bits)],
            )
            with pytest.raises(SampleAcceptanceError, match="bitwise"):
                result.require_sample_acceptance()
    assert tuple(v.tobytes() for v in values) == before


def _preserve_canonical(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    results = []
    for module in (min_module, max_module):
        original = module.lower_source_min_max

        def lower(
            ctx: Any, eqn: Any, *, operation: str, _original: Any = original
        ) -> bool:
            operands = [ctx.require_value_for_var(v) for v in eqn.invars]
            previous = [
                (
                    v.type,
                    tuple(v.shape),
                    (
                        None
                        if const_value_to_numpy(v) is None
                        else const_value_to_numpy(v).tobytes()
                    ),
                )
                for v in operands
            ]
            accepted = _original(ctx, eqn, operation=operation)
            assert accepted
            for var, value, snapshot in zip(
                eqn.invars, operands, previous, strict=True
            ):
                assert ctx.lookup_value_for_var(var) is value
                assert (
                    value.type,
                    tuple(value.shape),
                    (
                        None
                        if const_value_to_numpy(value) is None
                        else const_value_to_numpy(value).tobytes()
                    ),
                ) == snapshot
            result = ctx.lookup_value_for_var(eqn.outvars[0])
            assert all(result is not value for value in operands)
            assert result.dtype == ir.DataType.from_numpy(eqn.outvars[0].aval.dtype)
            assert tuple(result.shape) == tuple(eqn.outvars[0].aval.shape)
            results.append(result)
            return accepted

        monkeypatch.setattr(module, "lower_source_min_max", lower)
    return results


def test_weak_fp16_promotion_shared_typed_constants_and_canonical_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _preserve_canonical(monkeypatch)
    constant = np.array([-0.0, 0.0, 0.0, -0.0], np.float16)
    scientific = np.array([1 + 2**-40, -0.0, 2.0, 3.0], np.float64)
    snapshots = (constant.tobytes(), scientific.tobytes())
    weak = jnp.asarray(0.0)
    assert weak.weak_type
    values = (jnp.array([-0.0, 0.0, -1.0, 1.0], np.float16), weak)

    def model(x: Any, w: Any) -> Any:
        return jnp.minimum(x, w), jnp.maximum(x, constant), constant, scientific, x, w

    model_ir, eager, compiled = _export(model, values)
    assert len(observed) == 2
    assert [a.dtype.name for a in eager[:2]] == ["float16", "float16"]
    _accept(model_ir, values, (eager, compiled))
    assert (constant.tobytes(), scientific.tobytes()) == snapshots
    report = get_precision_report(model_ir)
    assert (
        report.inputs[1].supplied.weak_type is True
        and report.inputs[1].traced.weak_type is True
    )
    for node in model_ir.graph:
        if node.op_type == "Cast":
            target = node.attributes["to"].as_int()
            assert target in (int(ir.DataType.INT64), int(ir.DataType.FLOAT16))
    actual_constants = [
        const_value_to_numpy(v) for v in model_ir.graph.initializers.values()
    ]
    assert any(
        v.dtype == np.float64 and v.tobytes() == scientific.tobytes()
        for v in actual_constants
    )
    assert any(
        v.dtype == np.float16 and v.tobytes() == constant.tobytes()
        for v in actual_constants
    )


@onnx_function(name="ZeroSignScoped")
def _scoped(x: Any, y: Any, flag: Any) -> Any:
    return jax.lax.cond(
        flag,
        lambda z: (jnp.minimum(*z), jnp.maximum(*z)),
        lambda z: (jnp.maximum(*z), jnp.minimum(*z)),
        (x, y),
    )


def test_named_function_child_composition_owns_constants_and_fresh_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _preserve_canonical(monkeypatch)

    def model(x: Any, y: Any, flag: Any) -> Any:
        lo, hi = _scoped(x, y, flag)
        return lo, hi, jnp.minimum(lo, hi), x, y

    x, y = np.array([-0.0, 0.0, -0.0, 0.0], np.float64), np.array(
        [0.0, -0.0, -0.0, 0.0], np.float64
    )
    model_ir, eager, compiled = _export(model, (x, y, np.array(True)))
    assert len(observed) == 5
    functions = tuple(iter_ir_functions(model_ir.functions))
    assert len(functions) == 1
    (function,) = functions
    assert all(v.graph is function.graph for v in (*function.inputs, *function.outputs))
    assert all(v.graph is not model_ir.graph for v in function.inputs)
    nodes = [
        *RecursiveGraphIterator(model_ir.graph),
        *RecursiveGraphIterator(function.graph),
    ]
    local_zero_constants = []
    for node in nodes:
        assert node.graph is not None and all(
            v.graph is node.graph for v in node.outputs
        )
        if node.op_type == "Constant":
            value = const_value_to_numpy(node.outputs[0])
            if value is not None and value.dtype == np.float64:
                assert value.shape == () and not _bits(value).any()
                local_zero_constants.append(node.outputs[0])
    assert len({id(v) for v in local_zero_constants}) == len(local_zero_constants)
    assert len({v.graph for v in local_zero_constants}) == 2
    assert all(
        v.graph is not function.graph and v.graph is not model_ir.graph
        for v in local_zero_constants
    )
    for flag in (True, False):
        values = (x, y, np.array(flag))
        references = (eager, compiled) if flag else _references(model, values)
        _accept(model_ir, values, references)


@pytest.mark.parametrize("route", ["legacy", "integer", "opset26", "bf16"])
def test_unselected_routes_keep_their_existing_direct_operators(route: str) -> None:
    dtype = (
        np.int32
        if route == "integer"
        else jnp.bfloat16 if route == "bf16" else np.float32
    )
    values = (np.array([-1, 2], dtype=dtype), np.array([3, -4], dtype=dtype))

    def model(x: Any, y: Any) -> Any:
        return jnp.minimum(x, y), jnp.maximum(x, y)

    if route == "legacy":
        artifact = to_onnx(
            model,
            list(map(_spec, values)),
            return_mode="ir",
            enable_double_precision=False,
        )
        eager, compiled = _references(model, values)
    else:
        artifact, eager, compiled = _export(
            model, values, opset=26 if route == "opset26" else 23
        )
    proto = _proto(artifact)
    assert [n.op_type for n in proto.graph.node] == ["Min", "Max"]
    if route in ("legacy", "integer"):
        _accept(artifact, values, (eager, compiled))
    elif route == "bf16":
        result = validate_source_samples(
            artifact, samples=[SourceSample(values, eager)], optimization="disabled"
        )
        assert result.status is CheckStatus.FAILED
        assert any(
            c.category == "runtime-load" and "NOT_IMPLEMENTED" in c.detail
            for c in result.checks
        )
        with pytest.raises(SampleAcceptanceError):
            result.require_sample_acceptance()


@pytest.mark.parametrize("defect", ["dtype", "shape", "result"])
def test_invalid_canonical_representation_fails_before_emitting_nodes(
    defect: str,
) -> None:
    graph = jax.make_jaxpr(jax.lax.min)(
        jax.ShapeDtypeStruct((2,), np.float32), jax.ShapeDtypeStruct((2,), np.float32)
    ).jaxpr
    ctx = IRContext(
        opset=23, enable_double_precision=False, precision_policy=PrecisionPolicy.SOURCE
    )
    for index, var in enumerate(graph.invars):
        ctx.add_input_for_invar(var, index)
    eqn = graph.eqns[0]
    if defect == "dtype":
        ctx.require_value_for_var(graph.invars[0]).dtype = ir.DataType.DOUBLE
    elif defect == "shape":
        ctx.require_value_for_var(graph.invars[0]).shape = ir.Shape((3,))
    else:
        eqn = SimpleNamespace(
            invars=eqn.invars,
            outvars=[SimpleNamespace(aval=jax.ShapeDtypeStruct((3,), np.float32))],
        )
    with pytest.raises(ValueError, match="SOURCE Min"):
        lower_source_min_max(ctx, eqn, operation="Min")
    assert not tuple(ctx.builder.graph) and not ctx.builder.initializers


@pytest.mark.parametrize("operation", ["Min", "Max"])
def test_internal_lowerer_preserves_symbolic_ir_metadata_without_admission(
    operation: str,
) -> None:
    (batch,) = export.symbolic_shape("B")
    spec = jax.ShapeDtypeStruct((batch, 3), np.float32)
    graph = jax.make_jaxpr(jax.lax.min if operation == "Min" else jax.lax.max)(
        spec, spec
    ).jaxpr
    ctx = IRContext(
        opset=23, enable_double_precision=False, precision_policy=PrecisionPolicy.SOURCE
    )
    for index, var in enumerate(graph.invars):
        ctx.add_input_for_invar(var, index)
    assert lower_source_min_max(ctx, graph.eqns[0], operation=operation)
    result = ctx.require_value_for_var(graph.outvars[0])
    assert result.shape[0].value == "B" and result.shape[1] == 3
    choices = next(n.outputs[0] for n in ctx.builder.graph if n.op_type == "Concat")
    assert tuple(str(v) for v in choices.shape) == ("2", "B", "3")
    producer, metadata = result.producer(), (result.type, tuple(result.shape))
    assert lower_source_min_max(ctx, graph.eqns[0], operation=operation)
    assert ctx.require_value_for_var(graph.outvars[0]) is not result
    assert (
        result.producer() is producer and (result.type, tuple(result.shape)) == metadata
    )
    with pytest.raises(SourcePrecisionError):
        ui._to_onnx(
            _function(operation), [spec, spec], precision_policy=PrecisionPolicy.SOURCE
        )
