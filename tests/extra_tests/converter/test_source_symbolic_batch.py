# tests/extra_tests/converter/test_source_symbolic_batch.py

"""One shared symbolic batch survives audited Add, Function and sample boundaries."""

from __future__ import annotations

from collections.abc import Iterator
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
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx import precision, user_interface as ui
from jax2onnx._compat.jax import atomic_symbolic_dimension
from jax2onnx.converter import conversion_api
from jax2onnx.converter.function_scope import FunctionScope
from jax2onnx.converter.source_deployment import DeploymentTarget
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.converter.source_representation import (
    normalized_shape_dimensions,
    representation_for_aval,
)
from jax2onnx.ir_utils import iter_ir_functions
from jax2onnx.plugins.plugin_system import FunctionPlugin
from jax2onnx.precision import (
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)


def _settings() -> tuple[Any, ...]:
    return tuple(getattr(jax.config, name) for name in _globals())


def _globals() -> dict[str, Any]:
    return {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    previous = _globals()
    effective = _settings()
    try:
        with (
            jax.enable_x64(False),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)
        assert _settings() == effective
        assert active_precision_policy() is PrecisionPolicy.LEGACY


def _add(x: Any, y: Any) -> Any:
    return jax.lax.add(x, y)


@onnx_function(name="SharedBatchAdd")
def _named_add(x: Any, y: Any) -> Any:
    return jax.lax.add(x, y)


def _composed(x: Any, y: Any) -> Any:
    with (
        jax.enable_x64(True),
        jax.default_matmul_precision("highest"),
        jax.numpy_dtype_promotion("standard"),
        jax.numpy_rank_promotion("allow"),
    ):
        return _named_add(x, y)


def _source(fn: Any = _add, inputs: Any = None, **kwargs: Any) -> Any:
    return ui._to_onnx(
        fn,
        inputs if inputs is not None else [("B", 3), ("B", 3)],
        precision_policy=PrecisionPolicy.SOURCE,
        **kwargs,
    )


def _values(batch: int, dtype: Any = np.float32) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(batch * 3).reshape(batch, 3) / 8 - 2).astype(dtype)
    y = (np.arange(batch * 3, 0, -1).reshape(batch, 3) / 16 + 1).astype(dtype)
    assert not np.array_equal(x, y)
    return x, y


def _reference(fn: Any, values: Any) -> tuple[np.ndarray, ...]:
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    eager = tuple(np.asarray(v) for v in jax.tree.leaves(fn(*values)))
    compiled = tuple(np.asarray(v) for v in jax.tree.leaves(jax.jit(fn)(*values)))
    for actual, expected in zip(compiled, eager, strict=True):
        np.testing.assert_array_equal(actual, expected, strict=True)
        assert actual.tobytes() == expected.tobytes()
    return eager


def _samples(fn: Any = _add, dtype: Any = np.float32) -> tuple[SourceSample, ...]:
    return tuple(
        SourceSample(values, _reference(fn, values), SampleCriteria("bitwise"))
        for batch in (2, 5)
        for values in (_values(batch, dtype),)
    )


def _check_samples(model: Any, samples: Any) -> None:
    for optimization in ("disabled", "all"):
        result = validate_source_samples(
            model, samples=samples, optimization=optimization
        )
        assert result.status is CheckStatus.VALIDATED, result
        assert len(result.samples) == len(samples)
        result.require_sample_acceptance()


def test_one_audited_add_artifact_runs_two_batches_without_specialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _samples()
    trace = conversion_api._trace_to_jaxpr
    traced = []

    def observe(**kwargs: Any) -> Any:
        traced.append(kwargs["fn"])
        return trace(**kwargs)

    monkeypatch.setattr(conversion_api, "_trace_to_jaxpr", observe)
    model = _source(input_names=["lhs", "rhs"], output_names=["sum"])
    assert traced == [_add]
    report = get_precision_report(model)
    assert report is not None and report.profile == "source-static-v1"
    assert all(entry.supplied.dtype is None for entry in report.inputs)
    assert all(entry.normalized.dtype == "float32" for entry in report.inputs)
    assert all(entry.traced.shape == ("B", 3) for entry in report.inputs)
    assert report.exported_outputs[0].shape == ("B", 3)
    assert {c.category: c.status for c in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    digest = hashlib.sha256(model.SerializeToString()).hexdigest()
    _check_samples(model, samples)
    assert traced == [_add]
    assert hashlib.sha256(model.SerializeToString()).hexdigest() == digest


@pytest.mark.parametrize(
    "tail",
    [(), (3, 2, 2), (3, 2, 2, 2), (3, 2, 2, 2, 2)],
    ids=["rank-one", "rank-four", "rank-five", "rank-six"],
)
def test_shared_batch_rank_one_and_four_keep_one_artifact(
    tail: tuple[int, ...],
) -> None:
    samples = []
    for batch in (2, 5):
        shape = (batch, *tail)
        left = (np.arange(np.prod(shape)).reshape(shape) - 3).astype(np.float32)
        right = np.flip(left, axis=0).copy() / np.float32(4)
        values = (left, right)
        samples.append(
            SourceSample(values, _reference(_add, values), SampleCriteria("bitwise"))
        )
    model = _source(inputs=[("B", *tail), ("B", *tail)])
    onnx.checker.check_model(model, full_check=True)
    report = get_precision_report(model)
    assert report is not None
    assert all(entry.traced.shape == ("B", *tail) for entry in report.inputs)
    assert report.exported_outputs[0].shape == ("B", *tail)
    for value in (*model.graph.input, *model.graph.output):
        dims = value.type.tensor_type.shape.dim
        assert dims[0].dim_param == "B"
        assert tuple(dim.dim_value for dim in dims[1:]) == tail
    payload = model.SerializeToString()
    _check_samples(model, samples)
    assert model.SerializeToString() == payload


@pytest.mark.parametrize("native_first", [False, True])
def test_mixed_declarations_share_verified_native_scope_in_both_orders(
    native_first: bool,
) -> None:
    (batch,) = export.symbolic_shape("B")
    native = jax.ShapeDtypeStruct((batch, 3), np.float32)
    string = jax.ShapeDtypeStruct(("B", 3), np.float32)
    inputs = [native, string] if native_first else [string, native]
    specs = conversion_api._as_sds_list(
        inputs, False, precision_policy=PrecisionPolicy.SOURCE
    )
    assert specs[0].shape[0].scope is batch.scope
    assert specs[1].shape[0].scope is batch.scope
    assert specs[0 if native_first else 1].shape[0] is batch
    samples = _samples()
    model = _source(inputs=inputs)
    _check_samples(model, samples)


def test_actual_named_child_contract_and_source_legacy_native_cache_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weak = jnp.broadcast_to(jnp.asarray(0.125), (2, 3))
    assert weak.weak_type
    (batch,) = export.symbolic_shape("B")
    native_weak = jax.make_jaxpr(_add)(weak, weak)
    assert all(
        v.aval.weak_type
        for v in (*native_weak.jaxpr.invars, *native_weak.jaxpr.outvars)
    )
    specs = [jax.ShapeDtypeStruct((batch, 3), weak.dtype, weak_type=weak.weak_type)] * 2
    weak_values = [
        tuple(
            jnp.broadcast_to(jnp.asarray(value), (size, 3)) for value in (0.125, 0.75)
        )
        for size in (2, 5)
    ]
    samples = [
        SourceSample(values, _reference(_composed, values), SampleCriteria("bitwise"))
        for values in weak_values
    ]
    scopes = []
    original_begin = FunctionScope.begin
    original_validator = ui.validate_source_jaxpr
    validations = []
    body_traces = []
    original_body_trace = FunctionPlugin._trace_function_body

    def trace_body(self: Any, *args: Any, **kwargs: Any) -> Any:
        closed = original_body_trace(self, *args, **kwargs)
        source = kwargs.get("source_trace")
        if source is not None:
            assert source.x64_enabled is True
            assert source.default_matmul_precision == "highest"
            assert source.numpy_dtype_promotion == "standard"
            assert source.numpy_rank_promotion == "allow"
            body_traces.append(closed)
        return closed

    def begin(self: FunctionScope, values: Any) -> Any:
        formals = original_begin(self, values)
        if self.ctx.precision_policy is PrecisionPolicy.SOURCE:
            assert self.ctx.precision_policy is self.parent.precision_policy
            for parent, formal in zip(values, formals, strict=True):
                assert parent is not formal
                origin = self.ctx.get_symbolic_dim_origin(formal.shape[0])
                assert origin is not None and any(
                    origin.value is item for item in formals
                )
            scopes.append(self)
        return formals

    def validate(
        graph: Any, opset: int, *, deployment_target: DeploymentTarget | None = None
    ) -> None:
        original_validator(graph, opset, deployment_target=deployment_target)
        validations.append(graph)

    monkeypatch.setattr(FunctionScope, "begin", begin)
    monkeypatch.setattr(ui, "validate_source_jaxpr", validate)
    monkeypatch.setattr(FunctionPlugin, "_trace_function_body", trace_body)
    with jax.numpy_dtype_promotion("strict"), jax.numpy_rank_promotion("raise"):
        before = _settings()
        first = _source(_composed, specs, return_mode="ir")
        legacy = to_onnx(_composed, specs, return_mode="ir")
        second = _source(_composed, specs, return_mode="ir")
        assert _settings() == before
    assert legacy is not first and first is not second
    assert len(scopes) == 2
    assert len(body_traces) == 2
    children = [
        g for g in validations if any(e.primitive.name == "add" for e in g.eqns)
    ]
    assert len(children) == 2
    for graph in children:
        assert all(v.aval.weak_type for v in (*graph.invars, *graph.outvars))
        symbols = [
            atomic_symbolic_dimension(v.aval.shape[0])
            for v in (*graph.invars, *graph.outvars)
        ]
        assert all(label == "B" and scope is symbols[0][1] for label, scope in symbols)
    for model in (first, second):
        (function,) = tuple(iter_ir_functions(model.functions))
        assert all(
            v.graph is function.graph for v in (*function.inputs, *function.outputs)
        )
        assert all(v.graph is not model.graph for v in function.inputs)
        assert all(
            normalized_shape_dimensions(v.shape) == ("B", 3)
            for v in (*function.inputs, *function.outputs)
        )
        report = get_precision_report(model)
        assert report is not None and all(e.traced.weak_type for e in report.inputs)
        _check_samples(model, samples)
    for values in weak_values:
        _reference(_composed, values)
    _reference(_composed, (weak, weak))


@pytest.mark.parametrize("return_mode", ["ir", "proto", "file"])
def test_symbolic_boundaries_aliases_names_and_json_file_roundtrip(
    return_mode: str, tmp_path: Path
) -> None:
    def aliases(x: Any, y: Any) -> Any:
        return _add(x, y), x, x

    model = _source(
        aliases,
        return_mode=return_mode,
        output_path=tmp_path / "batch.onnx",
        input_names=["left", "right"],
        output_names=["total", "first", "again"],
    )
    proto = (
        ir.to_proto(model)
        if isinstance(model, ir.Model)
        else model if isinstance(model, onnx.ModelProto) else onnx.load(model)
    )
    onnx.checker.check_model(proto, full_check=True)
    assert [v.name for v in proto.graph.input] == ["left", "right"]
    assert [v.name for v in proto.graph.output] == ["total", "first", "again"]
    for value in (*proto.graph.input, *proto.graph.output):
        dims = value.type.tensor_type.shape.dim
        assert dims[0].HasField("dim_param") and dims[0].dim_param == "B"
        assert dims[1].HasField("dim_value") and dims[1].dim_value == 3
    report = get_precision_report(model)
    assert report is not None and report == get_precision_report(proto)
    json.loads(precision.precision_report_to_json(report))
    _check_samples(model, _samples(aliases))


@pytest.mark.parametrize("corruption", ["rename", "concrete", "missing", "K", "dtype"])
def test_symbolic_serialized_boundary_corruption_fails_before_report(
    corruption: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialize = ir.to_proto
    model_corruptions = 0
    leaf_delegations = 0

    def corrupt(value: Any) -> Any:
        nonlocal model_corruptions, leaf_delegations
        proto = serialize(value)
        if not isinstance(value, ir.Model):
            if isinstance(value, ir.Value):
                leaf_delegations += 1
            return proto
        tensor = proto.graph.output[0].type.tensor_type
        if corruption == "rename":
            tensor.shape.dim[0].dim_param = "Other"
        elif corruption == "concrete":
            tensor.shape.dim[0].dim_value = 2
        elif corruption == "missing":
            tensor.shape.dim[0].Clear()
        elif corruption == "K":
            tensor.shape.dim[1].dim_value = 4
        else:
            tensor.elem_type = onnx.TensorProto.DOUBLE
        model_corruptions += 1
        return proto

    monkeypatch.setattr(ir, "to_proto", corrupt)
    with pytest.raises(SourcePrecisionError, match="boundary dtype or shape differs"):
        _source()
    # Default proto export serializes the frozen reference and actual Model.
    # Apply the same fault to both so the original boundary guard is exercised.
    assert model_corruptions == 2
    assert leaf_delegations > 0


def test_symbolic_descriptor_strictly_rejects_unknown_ir_dimensions() -> None:
    (batch,) = export.symbolic_shape("B")
    representation = representation_for_aval(
        jax.ShapeDtypeStruct((batch, 3), np.float32)
    )
    for shape in (("Other", 3), (2, 3), (None, 3), ("B", 4), ("B",)):
        value = ir.Value(type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(shape))
        assert not representation.matches_ir_value(value)
    assert representation.matches_ir_value(
        ir.Value(type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(("B", 3)))
    )


def test_fp64_mantissa_contract_and_disabled_x64_pretrace_rejection() -> None:
    specs = [jax.ShapeDtypeStruct(("B", 3), np.float64)] * 2
    called = []

    def model(x: Any, y: Any) -> Any:
        called.append(True)
        return _add(x, y)

    with pytest.raises(SourcePrecisionError, match="x64 is disabled"):
        _source(model, specs)
    assert not called
    with jax.enable_x64(True):
        samples = []
        for batch in (2, 5):
            x = np.full((batch, 3), 1 + 2**-40, np.float64)
            y = np.full((batch, 3), 2**-42, np.float64)
            samples.append(
                SourceSample(
                    (x, y), _reference(_add, (x, y)), SampleCriteria("bitwise")
                )
            )
        artifact = _source(inputs=specs)
        _check_samples(artifact, samples)
        assert get_precision_report(artifact).exported_outputs[0].dtype == "float64"


def test_static_zero_extents_keep_the_existing_audited_contract() -> None:
    values = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
    samples = [
        SourceSample(values, _reference(_add, values), SampleCriteria("bitwise"))
    ]
    model = _source(inputs=[(0, 3), (0, 3)])
    _check_samples(model, samples)


@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16])
def test_low_precision_batch_schema_report_and_actual_cpu_outcome(dtype: Any) -> None:
    samples = _samples(dtype=dtype)
    model = _source(inputs=[jax.ShapeDtypeStruct(("B", 3), dtype)] * 2)
    report = get_precision_report(model)
    assert (
        report is not None and report.exported_outputs[0].dtype == np.dtype(dtype).name
    )
    onnx.checker.check_model(model, full_check=True)
    for optimization in ("disabled", "all"):
        result = validate_source_samples(
            model, samples=samples, optimization=optimization
        )
        if dtype is np.float16:
            result.require_sample_acceptance()
        else:
            assert result.status is CheckStatus.FAILED
            assert any(
                c.category == "runtime-load"
                and c.status is CheckStatus.FAILED
                and "NOT_IMPLEMENTED" in c.detail
                for c in result.checks
            )
            with pytest.raises(SampleAcceptanceError):
                result.require_sample_acceptance()


@pytest.mark.parametrize("named", [False, True])
def test_real_sample_binding_rejects_broadcasts_before_session_and_each_run(
    named: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnxruntime as ort

    model = _source(input_names=["x", "y"])
    constructor = ort.InferenceSession
    # ONNX dim_param is an annotation, so the raw runtime accepts this
    # out-of-contract broadcast. SOURCE evidence must still reject it.
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    raw = constructor(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    broadcast_x, broadcast_y = _values(1)[0], _values(5)[1]
    (broadcast_output,) = raw.run(None, {"x": broadcast_x, "y": broadcast_y})
    np.testing.assert_array_equal(
        broadcast_output, np.add(broadcast_x, broadcast_y), strict=True
    )
    sessions = []
    runs = []

    def session(*args: Any, **kwargs: Any) -> Any:
        actual = constructor(*args, **kwargs)
        sessions.append(actual)
        run = actual.run

        def observed_run(*args: Any, **kwargs: Any) -> Any:
            runs.append(True)
            return run(*args, **kwargs)

        actual.run = observed_run
        return actual

    monkeypatch.setattr(ort, "InferenceSession", session)
    invalid = [
        (_values(1)[0], _values(5)[1]),
        (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)),
        (np.zeros((2, 4), np.float32), _values(2)[1]),
        (np.zeros((2, 3, 1), np.float32), _values(2)[1]),
        (_values(2)[0].astype(np.float64), _values(2)[1]),
    ]

    def sample(values: Any) -> SourceSample:
        feeds = dict(zip(("x", "y"), values, strict=True)) if named else values
        return SourceSample(
            feeds, (np.zeros((5, 3), np.float32),), SampleCriteria("bitwise")
        )

    result = validate_source_samples(
        model, samples=[sample(v) for v in invalid], optimization="disabled"
    )
    assert not sessions and not runs
    assert len(result.samples) == len(invalid)
    assert all(
        any(
            c.category == "sample-input" and c.status is CheckStatus.FAILED
            for c in item.checks
        )
        for item in result.samples
    )
    assert all(
        any(
            c.category == "numerical-acceptance" and c.status is CheckStatus.UNVERIFIED
            for c in item.checks
        )
        for item in result.samples
    )
    valid = list(_samples())
    if named:
        valid = [
            SourceSample(
                dict(zip(("y", "x"), reversed(s.inputs), strict=True)),
                s.expected_outputs,
                s.criteria,
            )
            for s in valid
        ]
    # The first invalid sample, the valid B=2 sample, another invalid sample and
    # valid B=5 sample exercise both pre-session and existing-session gates.
    mixed = [sample(invalid[0]), valid[0], sample(invalid[0]), valid[1]]
    result = validate_source_samples(model, samples=mixed, optimization="all")
    assert len(sessions) == 1 and len(runs) == 2
    assert [s.status for s in result.samples] == [
        CheckStatus.FAILED,
        CheckStatus.VALIDATED,
        CheckStatus.FAILED,
        CheckStatus.VALIDATED,
    ]
    with pytest.raises(SampleAcceptanceError):
        result.require_sample_acceptance()


def test_actual_output_must_match_the_bound_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnxruntime as ort

    model = _source()
    constructor = ort.InferenceSession

    def session(*args: Any, **kwargs: Any) -> Any:
        actual = constructor(*args, **kwargs)
        actual.run = lambda *a, **k: [np.zeros((5, 3), np.float32)]
        return actual

    monkeypatch.setattr(ort, "InferenceSession", session)
    result = validate_source_samples(
        model, samples=[_samples()[0]], optimization="disabled"
    )
    assert result.status is CheckStatus.FAILED
    assert any(
        c.category == "sample-output" and c.status is CheckStatus.FAILED
        for c in result.samples[0].checks
    )
    assert any(
        c.category == "numerical-acceptance" and c.status is CheckStatus.UNVERIFIED
        for c in result.samples[0].checks
    )
    with pytest.raises(SampleAcceptanceError):
        result.require_sample_acceptance()


def test_native_symbol_adapter_rejects_unknown_constraint_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (batch,) = export.symbolic_shape("B")
    assert atomic_symbolic_dimension(batch) == ("B", batch.scope)
    monkeypatch.setattr(batch.scope, "_explicit_constraints", None)
    with pytest.raises(ValueError, match="unconstrained scope"):
        atomic_symbolic_dimension(batch)


def test_forged_reference_batch_and_report_signature_fail_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnxruntime as ort

    model = _source()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid SOURCE contract created a runtime session")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    result = validate_source_samples(
        model,
        samples=[SourceSample(_values(2), (np.zeros((5, 3), np.float32),))],
        optimization="disabled",
    )
    assert result.status is CheckStatus.FAILED
    assert result.samples[0].checks[0].category == "sample-input"
    report = get_precision_report(model)
    assert report is not None
    metadata = json.loads(precision.precision_report_to_json(report))
    metadata["inputs"][0]["exported"]["shape"][0] = "Other"
    for item in model.metadata_props:
        if item.key == precision.PRECISION_METADATA_KEY:
            item.value = json.dumps(metadata)
    result = validate_source_samples(model, samples=_samples(), optimization="all")
    assert any(
        c.category == "source-shape-contract" and c.status is CheckStatus.FAILED
        for c in result.checks
    )


@pytest.mark.parametrize(
    "kind",
    [
        "second",
        "nonleading",
        "affine",
        "constraint",
        "scopes",
        "rank",
        "K-zero",
        "unknown",
    ],
)
def test_finite_declaration_exclusions_fail_before_model_invocation(kind: str) -> None:
    called = []

    def model(x: Any, y: Any) -> Any:
        called.append(True)
        return _add(x, y)

    (batch,) = export.symbolic_shape("B")
    shapes: Any = [("B", 3), ("B", 3)]
    if kind == "second":
        shapes[1] = ("C", 3)
    elif kind == "nonleading":
        shapes = [(3, "B")] * 2
    elif kind == "affine":
        shapes = [(2 * batch, 3)] * 2
    elif kind == "constraint":
        (constrained,) = export.symbolic_shape("B", constraints=("B >= 2",))
        shapes = [(constrained, 3)] * 2
    elif kind == "scopes":
        (other,) = export.symbolic_shape("B")
        shapes = [(batch, 3), (other, 3)]
    elif kind == "rank":
        # Retain the original parameter ID. Valid higher ranks have positive
        # coverage; a repeated symbolic axis remains an invalid declaration.
        shapes = [("B", 3, 2, "B", 2)] * 2
    elif kind == "K-zero":
        shapes = [("B", 0)] * 2
    else:
        shapes = [(object(), 3)] * 2
    specs = [jax.ShapeDtypeStruct(shape, np.float32) for shape in shapes]
    with pytest.raises(SourcePrecisionError) as error:
        _source(model, specs)
    assert error.value.category == "profile-exclusion"
    assert not called
    assert active_precision_policy() is PrecisionPolicy.LEGACY


def test_symbolic_subtract_and_reduce_sum_remain_explicit_exclusions() -> None:
    """Original subtraction and batch reductions retain typed admission."""

    def subtract(x: Any, y: Any) -> Any:
        return jax.lax.sub(x, y)

    samples = _samples(subtract)
    model = _source(subtract)
    onnx.checker.check_model(model, full_check=True)
    report = get_precision_report(model)
    assert report is not None and report.profile == "source-static-v1"
    assert all(
        value.shape == ("B", 3)
        and value.dtype == "float32"
        and value.weak_type is False
        for value in (
            *(entry.traced for entry in report.inputs),
            *report.traced_outputs,
        )
    )
    assert [node.op_type for node in model.graph.node] == ["Sub"]
    for value in (*model.graph.input, *model.graph.output):
        tensor = value.type.tensor_type
        assert tensor.elem_type == onnx.TensorProto.FLOAT
        assert tensor.shape.dim[0].dim_param == "B"
        assert tensor.shape.dim[1].dim_value == 3
    payload = model.SerializeToString()
    _check_samples(model, samples)
    assert model.SerializeToString() == payload

    reduced = _source(lambda x, y: jax.lax.reduce_sum(x, axes=(0,)))
    onnx.checker.check_model(reduced, full_check=True)
    reduction_report = get_precision_report(reduced)
    assert (
        reduction_report is not None and reduction_report.profile == "source-static-v1"
    )
    assert reduction_report.traced_outputs[0].shape == (3,)
    assert reduction_report.traced_outputs[0].dtype == "float32"
    assert any(node.op_type == "ReduceSum" for node in reduced.graph.node)
    with pytest.raises(SourcePrecisionError) as error:
        _source(lambda x, y: jax.lax.erf(x))
    assert error.value.phase == "profile-validation"
    assert error.value.category == "profile-exclusion"
    assert error.value.primitive == "erf"


def test_symbolic_constant_cond_accepts_original_batch_scope() -> None:
    def fn(x: Any, y: Any) -> Any:
        return jax.lax.cond(np.bool_(True), lambda: _add(x, y), lambda: x)

    # Retain the original constant-True conditional and both B-shaped operands.
    # Native eager/JIT references for B=2/5 precede the common SOURCE export.
    samples = _samples(fn)
    model = _source(fn)
    _check_samples(model, samples)


def test_symbolic_add_operand_broadcast_remains_an_explicit_exclusion() -> None:
    """The reviewed singleton tail now broadcasts without specializing B."""

    samples = []
    for batch in (2, 5):
        x, y = _values(batch)
        values = (x, y[:, :1])
        samples.append(
            SourceSample(values, _reference(_add, values), SampleCriteria("bitwise"))
        )
    model = _source(inputs=[("B", 3), ("B", 1)])
    onnx.checker.check_model(model, full_check=True)
    report = get_precision_report(model)
    assert report is not None and report.profile == "source-static-v1"
    assert [entry.traced.shape for entry in report.inputs] == [("B", 3), ("B", 1)]
    assert report.traced_outputs[0].shape == ("B", 3)
    assert all(
        value.dtype == "float32" and value.weak_type is False
        for value in (
            *(entry.traced for entry in report.inputs),
            *report.traced_outputs,
        )
    )
    assert [node.op_type for node in model.graph.node] == ["Add"]
    for value, tail in zip(
        (*model.graph.input, *model.graph.output), (3, 1, 3), strict=True
    ):
        tensor = value.type.tensor_type
        assert tensor.elem_type == onnx.TensorProto.FLOAT
        assert tensor.shape.dim[0].dim_param == "B"
        assert tensor.shape.dim[1].dim_value == tail
    payload = model.SerializeToString()
    _check_samples(model, samples)
    assert model.SerializeToString() == payload


@pytest.mark.parametrize(
    "defect", ["output-symbol", "shape", "dtype", "params", "effects"]
)
def test_symbolic_profile_checks_actual_typed_add_contract(defect: str) -> None:
    (batch,) = export.symbolic_shape("B")
    shape = (batch, 3)
    spec = jax.ShapeDtypeStruct(shape, np.float32)
    inputs = [SimpleNamespace(aval=spec), SimpleNamespace(aval=spec)]
    output_shape, dtype, params, effects = shape, np.float32, {}, ()
    if defect == "output-symbol":
        (other,) = export.symbolic_shape("Other", scope=batch.scope)
        output_shape = (other, 3)
    elif defect == "shape":
        output_shape = (batch, 4)
    elif defect == "dtype":
        dtype = np.float16
    elif defect == "params":
        params = {"unknown_semantics": True}
    else:
        effects = ("side-effect",)
    output = SimpleNamespace(aval=jax.ShapeDtypeStruct(output_shape, dtype))
    equation = SimpleNamespace(
        primitive=SimpleNamespace(name="add"),
        invars=inputs,
        outvars=[output],
        params=params,
        effects=effects,
    )
    graph = SimpleNamespace(
        invars=inputs, constvars=(), outvars=[output], eqns=[equation], effects=()
    )
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(graph, 23)


def test_direct_core_checks_mixed_scope_and_finite_contract() -> None:
    (batch,) = export.symbolic_shape("B")
    options = dict(
        fn=_add,
        input_params=None,
        model_name="batch",
        opset=23,
        enable_double_precision=False,
        record_primitive_calls_file=None,
        precision_policy=PrecisionPolicy.SOURCE,
        jaxpr_validator=validate_source_jaxpr,
    )
    model = conversion_api._convert_to_ir(
        inputs=[
            jax.ShapeDtypeStruct(("B", 3), np.float32),
            jax.ShapeDtypeStruct((batch, 3), np.float32),
        ],
        **options,
    )
    assert all(
        normalized_shape_dimensions(value.shape) == ("B", 3)
        for value in model.graph.inputs
    )
    (other,) = export.symbolic_shape("B")
    with pytest.raises(ValueError, match="shared native symbolic scope"):
        conversion_api._convert_to_ir(
            inputs=[
                jax.ShapeDtypeStruct((batch, 3), np.float32),
                jax.ShapeDtypeStruct((other, 3), np.float32),
            ],
            **options,
        )


@pytest.mark.parametrize("dtype", [np.bool_, np.uint32])
def test_direct_core_symbolic_typed_alias_accepts_original_contract(dtype: Any) -> None:
    def fn(x: Any) -> Any:
        return x

    samples = []
    for batch in (2, 5):
        payload = np.asarray(
            [False, True, True] if dtype is np.bool_ else [0, 2**31, 2**32 - 1],
            dtype,
        )
        values = (np.resize(payload, (batch, 3)),)
        samples.append(
            SourceSample(values, _reference(fn, values), SampleCriteria("bitwise"))
        )
    # Retain the original direct-core entry point and every original option.
    model = conversion_api._convert_to_ir(
        fn=fn,
        inputs=[jax.ShapeDtypeStruct(("B", 3), dtype)],
        input_params=None,
        model_name="batch",
        opset=23,
        enable_double_precision=False,
        record_primitive_calls_file=None,
        precision_policy=PrecisionPolicy.SOURCE,
        jaxpr_validator=validate_source_jaxpr,
    )
    expected_dtype = ir.DataType.BOOL if dtype is np.bool_ else ir.DataType.UINT32
    for value in (*model.graph.inputs, *model.graph.outputs):
        assert value.dtype == expected_dtype
        assert normalized_shape_dimensions(value.shape) == ("B", 3)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    # The core result has no facade report. The exact same identity callable and
    # signature use the established facade/sample API for runtime acceptance.
    artifact = _source(fn, inputs=[jax.ShapeDtypeStruct(("B", 3), dtype)])
    _check_samples(artifact, samples)
