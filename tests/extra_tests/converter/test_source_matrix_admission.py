# tests/extra_tests/converter/test_source_matrix_admission.py

"""Audited finite ordinary matrices; target sample acceptance remains independent."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, to_onnx, user_interface
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.plugins.jax.lax.dot_general import DotGeneralPlugin
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter import test_source_ordinary_matrix as d2a
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)


_CANONICAL = (((1,), (0,)), ((), ()))
_WEIGHT = np.arange(8, dtype=np.float32).reshape(4, 2) / np.float32(8)
_MATMUL = jnp.matmul


@onnx_function(name="AuditedCapturedMatrix")
def _named(left: Any) -> Any:
    return jnp.matmul(left, _WEIGHT)


@onnx_function(name="AuditedNestedMatrix")
def _nested(left: Any) -> Any:
    return _named(left)


@onnx_function(name="AuditedMatrixSettings")
def _settings_matrix(left: Any, right: Any) -> Any:
    return jnp.matmul(left, right)


def _globals() -> dict[str, Any]:
    return {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }


def _effective() -> tuple[Any, ...]:
    return tuple(getattr(jax.config, name) for name in _globals())


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    previous, effective = _globals(), _effective()
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
        assert (_globals(), _effective()) == (previous, effective)
        assert jnp.matmul is _MATMUL
        assert active_precision_policy() is PrecisionPolicy.LEGACY


def _export(
    fn: Any,
    values: tuple[Any, ...],
    *,
    opset: int = 23,
    mode: str = "proto",
    path: Path | None = None,
    named: bool = False,
    artifacts: list[Any] | None = None,
) -> tuple[onnx.ModelProto, list[np.ndarray], list[np.ndarray]]:
    # Native references precede activation; eager and JIT are separate references.
    original, eager, compiled = d2a._reference(fn, values)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(fn)(*map(d2a._spec, values))
    assert d2a._signature(d2a._body(activated).invars) == d2a._signature(
        d2a._body(original).invars
    )
    assert d2a._signature(d2a._body(activated).outvars) == d2a._signature(
        d2a._body(original).outvars
    )
    if not named:
        assert typed_trace_manifest(activated) == typed_trace_manifest(original)
    options: dict[str, Any] = {"return_mode": mode}
    if path is not None:
        options["output_path"] = path
    # No diagnostic callback: admission and lowering use the real audited path.
    settings = _effective(), _globals()
    try:
        artifact = user_interface._to_onnx(
            fn,
            list(map(d2a._spec, values)),
            opset=opset,
            precision_policy=PrecisionPolicy.SOURCE,
            **options,
        )
    finally:
        assert (_effective(), _globals()) == settings
    if artifacts is not None:
        artifacts.append(artifact)
    proto = (
        ir.to_proto(artifact)
        if isinstance(artifact, ir.Model)
        else onnx.load(artifact) if isinstance(artifact, str) else artifact
    )
    proto = onnx.load_model_from_string(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-static-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    assert tuple(
        (v.dtype, v.shape, v.weak_type) for v in report.traced_outputs
    ) == d2a._signature(d2a._body(original).outvars)
    assert not any("FP16" in message for message in report.adaptations)
    if isinstance(artifact, ir.Model):
        for graph in [
            artifact.graph,
            *(function.graph for function in artifact.functions.values()),
        ]:
            for node in graph:
                assert node.graph is graph
                assert all(value.producer().graph is graph for value in node.outputs)
    return proto, eager, compiled


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rank", [2, 3, 4])
@pytest.mark.parametrize("opset", [23, 26])
def test_audited_rank_dtype_and_preference_contract(
    dtype: Any, rank: int, opset: int
) -> None:
    with jax.enable_x64(dtype is np.float64):
        values = d2a._canonical_values(dtype, (2, 5)[: rank - 2])
        batch = tuple(range(rank - 2))
        dimensions = (((rank - 1,), (rank - 2,)), (batch, batch))
        for preferred in (None, dtype):

            def model(left: Any, right: Any) -> Any:
                return (
                    jax.lax.dot_general(
                        left, right, dimensions, preferred_element_type=preferred
                    ),
                    left,
                    right,
                )

            proto, eager, _ = _export(model, values, opset=opset)
            assert [node.op_type for node in proto.graph.node] == ["MatMul"]
            for actual, expected in zip(eager[1:], values, strict=True):
                assert (
                    actual.dtype == expected.dtype
                    and actual.tobytes() == expected.tobytes()
                )
            d2a._check_samples(proto, values, eager)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dimensions", d2a._LAYOUTS)
def test_ordered_noncanonical_axes_need_only_value_preserving_transposes(
    dtype: Any, dimensions: Any
) -> None:
    with jax.enable_x64(dtype is np.float64):
        values = d2a._layout_values(dimensions, dtype)

        def model(left: Any, right: Any) -> Any:
            return jax.lax.dot_general(left, right, dimensions), left, right

        proto, eager, _ = _export(model, values)
        assert sum(n.op_type == "MatMul" for n in proto.graph.node) == 1
        assert sum(n.op_type == "Transpose" for n in proto.graph.node) in (1, 2)
        assert {n.op_type for n in proto.graph.node} <= {"MatMul", "Transpose"}
        for optimization in ("disabled", "all"):
            d2a._check_samples(proto, values, eager, optimize=optimization)


@pytest.mark.parametrize("entry", ["lax", "jnp", "operator"])
@pytest.mark.parametrize(
    "precision", [None, jax.lax.Precision.DEFAULT, (jax.lax.Precision.DEFAULT,) * 2]
)
def test_native_entry_and_default_spellings(entry: str, precision: Any) -> None:
    values = d2a._canonical_values(np.float32)

    def model(left: Any, right: Any) -> Any:
        if entry == "operator":
            return left @ right
        if entry == "jnp":
            return jnp.matmul(left, right, precision=precision)
        return jax.lax.dot_general(left, right, _CANONICAL, precision=precision)

    proto, eager, _ = _export(model, values)
    d2a._check_samples(proto, values, eager)


@pytest.mark.parametrize("preferred", [False, True])
def test_actual_weak_operands_and_result_metadata_are_preserved(
    preferred: bool,
) -> None:
    values = tuple(
        jax.lax.broadcast_in_dim(jnp.asarray(1.0), shape, ())
        for shape in [(3, 4), (4, 2)]
    )
    assert all(value.weak_type for value in values)

    def model(left: Any, right: Any) -> Any:
        return jax.lax.dot_general(
            left,
            right,
            _CANONICAL,
            preferred_element_type=np.float32 if preferred else None,
        )

    proto, eager, _ = _export(model, values)
    report = get_precision_report(proto)
    assert report is not None and all(entry.traced.weak_type for entry in report.inputs)
    d2a._check_samples(proto, tuple(np.asarray(value) for value in values), eager)


@pytest.mark.parametrize("mode", ["proto", "ir", "file"])
def test_real_audited_artifact_forms_and_explicit_source_conversions(
    mode: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        values = d2a._canonical_values(np.float32)

        def model(left: Any, right: Any) -> Any:
            return left.astype(jnp.float64) @ right.astype(jnp.float64), left, right

        proto, eager, _ = _export(
            model,
            values,
            mode=mode,
            path=tmp_path / "audited.onnx" if mode == "file" else None,
        )
        assert [n.op_type for n in proto.graph.node] == ["Cast", "Cast", "MatMul"]
        assert all(
            next(a.i for a in n.attribute if a.name == "to") == onnx.TensorProto.DOUBLE
            for n in proto.graph.node
            if n.op_type == "Cast"
        )
        d2a._check_samples(proto, values, eager)


@pytest.mark.parametrize("scope", ["named", "nested", "cond"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_captured_weights_actual_child_signatures_and_conditional_branches(
    scope: str, dtype: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    with jax.enable_x64(dtype is np.float64):
        weight = _WEIGHT.astype(dtype)
        frozen = weight.tobytes()
        monkeypatch.setattr(
            __import__(__name__, fromlist=["_WEIGHT"]), "_WEIGHT", weight
        )
        left = d2a._canonical_values(dtype)[0]

        def model(selector: Any, x: Any) -> Any:
            if scope == "cond":
                result = jax.lax.cond(
                    selector > dtype(0), lambda a: _named(a), lambda a: _named(-a), x
                )
            else:
                result = _named(x) if scope == "named" else _nested(x)
            return result, x, weight

        for selector in (-1, 1) if scope == "cond" else (1,):
            values = (np.asarray(selector, dtype), left)
            artifacts: list[Any] = []
            proto, eager, _ = _export(
                model, values, named=True, mode="ir", artifacts=artifacts
            )
            assert proto.functions and weight.tobytes() == frozen
            matmul_functions = [
                f for f in proto.functions if any(n.op_type == "MatMul" for n in f.node)
            ]
            assert len(matmul_functions) == 1
            function = matmul_functions[0]
            assert [n.op_type for n in function.node if n.op_type != "Constant"] == [
                "MatMul"
            ]
            payloads = [
                onnx.numpy_helper.to_array(a.t)
                for f in proto.functions
                for n in f.node
                for a in n.attribute
                if a.type == onnx.AttributeProto.TENSOR
            ]
            assert any(
                v.dtype == weight.dtype
                and v.shape == weight.shape
                and v.tobytes() == frozen
                for v in payloads
            )
            actual_ir = artifacts[0]
            child = next(
                function.graph
                for function in actual_ir.functions.values()
                if any(node.op_type == "MatMul" for node in function.graph)
            )
            expected_type = (
                ir.DataType.DOUBLE if dtype is np.float64 else ir.DataType.FLOAT
            )
            assert [(value.dtype, tuple(value.shape)) for value in child.inputs] == [
                (expected_type, (3, 4))
            ]
            assert [(value.dtype, tuple(value.shape)) for value in child.outputs] == [
                (expected_type, (3, 2))
            ]
            canonical = {
                value.name: onnx.numpy_helper.to_array(value)
                for value in proto.graph.initializer
            }
            assert canonical[proto.graph.output[2].name].tobytes() == frozen
            assert canonical[proto.graph.output[2].name].shape == weight.shape
            assert eager[2].tobytes() == frozen
            d2a._check_samples(proto, values, eager)


def test_named_cache_and_local_default_override_restore_all_enclosing_settings() -> (
    None
):
    values = d2a._canonical_values(np.float32)
    before, hidden = _effective(), _globals()

    def default_call(left: Any, right: Any) -> Any:
        with jax.default_matmul_precision(None):
            return _settings_matrix(left, right)

    def high_call(left: Any, right: Any) -> Any:
        with jax.default_matmul_precision("highest"):
            return _settings_matrix(left, right)

    with (
        jax.enable_x64(True),
        jax.default_matmul_precision("highest"),
        jax.numpy_dtype_promotion("strict"),
        jax.numpy_rank_promotion("raise"),
    ):
        enclosing = _effective()
        proto, eager, _ = _export(default_call, values, named=True)
        d2a._check_samples(proto, values, eager)
        # Public LEGACY temporarily changes globals; preserve their hidden values
        # as well as enclosing effective contexts in this cross-policy harness.
        legacy_globals = _globals()
        try:
            legacy = to_onnx(default_call, list(map(d2a._spec, values)))
        finally:
            for name, value in legacy_globals.items():
                jax.config.update(name, value)
        onnx.checker.check_model(legacy, full_check=True)
        _export(default_call, values, named=True)
        d2a._reference(default_call, values)
        with pytest.raises(SourcePrecisionError) as caught:
            _export(high_call, values, named=True)
        assert (
            caught.value.category == "profile-exclusion"
            and caught.value.primitive == "dot_general"
        )
        assert _effective() == enclosing and _globals() == hidden
    assert (_effective(), _globals()) == (before, hidden)


def test_native_strict_dtype_and_rank_errors_are_preserved() -> None:
    with jax.enable_x64(True), jax.numpy_dtype_promotion("strict"):
        values = (np.ones((3, 4), np.float32), np.ones((4, 2), np.float64))

        def fn(a: Any, b: Any) -> Any:
            return jnp.matmul(a, b)

        with pytest.raises(ValueError) as expected:
            jax.make_jaxpr(fn)(*map(d2a._spec, values))
        with pytest.raises(type(expected.value)):
            user_interface._to_onnx(
                fn,
                list(map(d2a._spec, values)),
                precision_policy=PrecisionPolicy.SOURCE,
            )
    with jax.numpy_rank_promotion("raise"):
        values = (np.ones((2, 3, 4), np.float32), np.ones((4, 2), np.float32))

        def fn(a: Any, b: Any) -> Any:
            return jnp.matmul(a, b)

        # Native matmul owns its batching rules even under strict rank promotion.
        original = jax.make_jaxpr(fn)(*map(d2a._spec, values))
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            actual = jax.make_jaxpr(fn)(*map(d2a._spec, values))
        assert typed_trace_manifest(actual) == typed_trace_manifest(original)
        with pytest.raises(SourcePrecisionError, match="dimensions.*outside"):
            user_interface._to_onnx(
                fn,
                list(map(d2a._spec, values)),
                precision_policy=PrecisionPolicy.SOURCE,
            )

        def rank_error(a: Any, b: Any) -> Any:
            return jnp.matmul(a, b) + np.asarray([1, 2], np.float32)

        specs = list(map(d2a._spec, d2a._canonical_values(np.float32)))
        with pytest.raises(ValueError) as expected:
            jax.make_jaxpr(rank_error)(*specs)
        with pytest.raises(type(expected.value)) as actual:
            user_interface._to_onnx(
                rank_error, specs, precision_policy=PrecisionPolicy.SOURCE
            )
        assert str(actual.value) == str(expected.value)


def _graph() -> Any:
    def variable(shape: Any, dtype: Any = np.float32) -> Any:
        return SimpleNamespace(
            aval=SimpleNamespace(shape=shape, dtype=dtype, weak_type=False)
        )

    left, right, result = variable((3, 4)), variable((4, 2)), variable((3, 2))
    equation = SimpleNamespace(
        primitive=SimpleNamespace(name="dot_general"),
        invars=[left, right],
        outvars=[result],
        params=dict(
            dimension_numbers=_CANONICAL, precision=None, preferred_element_type=None
        ),
        effects=(),
    )
    return SimpleNamespace(
        invars=[left, right],
        constvars=(),
        outvars=[result],
        eqns=[equation],
        effects=(),
    )


@pytest.mark.parametrize(
    "defect,category,location",
    [
        ("arity", "lowering-defect", "equation 0"),
        ("missing", "lowering-defect", "equation 0"),
        ("dimensions", "lowering-defect", "equation 0"),
        ("paired-count", "lowering-defect", "equation 0"),
        ("batch-count", "lowering-defect", "equation 0"),
        ("boolean-axis", "lowering-defect", "equation 0"),
        ("range", "lowering-defect", "equation 0"),
        ("duplicate", "lowering-defect", "equation 0"),
        ("overlap", "lowering-defect", "equation 0"),
        ("paired", "lowering-defect", "equation 0"),
        ("batch-paired", "lowering-defect", "equation 0"),
        ("result-type", "lowering-defect", "equation 0"),
        ("result-shape", "lowering-defect", "equation 0"),
        ("zero", "profile-exclusion", "equation 0"),
        ("negative", "profile-exclusion", "input 0"),
        ("boolean-extent", "profile-exclusion", "input 0"),
        ("symbolic", "profile-exclusion", "input 0"),
        ("unequal-rank", "profile-exclusion", "equation 0"),
        ("rank-five", "profile-exclusion", "equation 0"),
        ("multi-contract", "profile-exclusion", "equation 0"),
        ("fp16", "profile-exclusion", "equation 0"),
        ("bf16", "profile-exclusion", "equation 0"),
        ("integer", "profile-exclusion", "input 0"),
        ("complex", "profile-exclusion", "input 0"),
        ("mixed", "profile-exclusion", "equation 0"),
        ("preferred-wide", "profile-exclusion", "equation 0"),
        ("unknown", "profile-exclusion", "equation 0"),
        ("out_dtype", "profile-exclusion", "equation 0"),
        ("sharding", "profile-exclusion", "equation 0"),
        ("placement", "profile-exclusion", "equation 0"),
        ("high", "profile-exclusion", "equation 0"),
        ("highest", "profile-exclusion", "equation 0"),
        ("mixed-precision", "profile-exclusion", "equation 0"),
        ("preset", "profile-exclusion", "equation 0"),
        ("algorithm", "profile-exclusion", "equation 0"),
    ],
)
def test_early_attributed_admission_failures_never_reach_lowering(
    defect: str, category: str, location: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph()
    eqn = graph.eqns[0]
    lhs, rhs = graph.invars
    output = graph.outvars[0]
    params = eqn.params
    if defect == "arity":
        eqn.invars.pop()
    elif defect == "missing":
        del params["dimension_numbers"]
    elif defect == "dimensions":
        params["dimension_numbers"] = ((1,), (0,))
    elif defect == "paired-count":
        params["dimension_numbers"] = (((1,), ()), ((), ()))
    elif defect == "batch-count":
        lhs.aval.shape, rhs.aval.shape, output.aval.shape = (
            (2, 3, 4),
            (2, 4, 2),
            (2, 3, 2),
        )
        params["dimension_numbers"] = (((2,), (1,)), ((0,), ()))
    elif defect in {"boolean-axis", "range", "duplicate", "overlap"}:
        params["dimension_numbers"] = {
            "boolean-axis": (((True,), (0,)), ((), ())),
            "range": (((2,), (0,)), ((), ())),
            "duplicate": (((1, 1), (0, 0)), ((), ())),
            "overlap": (((1,), (0,)), ((1,), (1,))),
        }[defect]
    elif defect == "paired":
        rhs.aval.shape = (5, 2)
    elif defect == "batch-paired":
        lhs.aval.shape, rhs.aval.shape, output.aval.shape = (
            (2, 3, 4),
            (5, 4, 2),
            (2, 3, 2),
        )
        params["dimension_numbers"] = (((2,), (1,)), ((0,), (0,)))
    elif defect == "result-type":
        output.aval.dtype = np.float64
    elif defect == "result-shape":
        output.aval.shape = (3, 3)
    elif defect in {"zero", "negative", "boolean-extent", "symbolic"}:
        lhs.aval.shape = (
            {"zero": 0, "negative": -1, "boolean-extent": True, "symbolic": "B"}[
                defect
            ],
            4,
        )
    elif defect == "unequal-rank":
        rhs.aval.shape = (1, 4, 2)
    elif defect == "rank-five":
        lhs.aval.shape, rhs.aval.shape = (1, 1, 1, 3, 4), (1, 1, 1, 4, 2)
    elif defect == "multi-contract":
        lhs.aval.shape, rhs.aval.shape = (2, 3, 4), (3, 4, 5)
        params["dimension_numbers"] = (((1, 2), (0, 1)), ((), ()))
    elif defect in {"fp16", "bf16", "integer", "complex"}:
        lhs.aval.dtype = rhs.aval.dtype = output.aval.dtype = {
            "fp16": np.float16,
            "bf16": jnp.bfloat16,
            "integer": np.int32,
            "complex": np.complex64,
        }[defect]
    elif defect == "mixed":
        rhs.aval.dtype = np.float64
    elif defect == "preferred-wide":
        params["preferred_element_type"] = np.float64
        output.aval.dtype = np.float64
    elif defect in {"unknown", "out_dtype", "sharding", "placement"}:
        params["out_sharding" if defect == "placement" else defect] = (
            object() if defect == "placement" else None
        )
    else:
        params["precision"] = {
            "high": (jax.lax.Precision.HIGH,) * 2,
            "highest": (jax.lax.Precision.HIGHEST,) * 2,
            "mixed-precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST),
            "preset": jax.lax.DotAlgorithmPreset.F32_F32_F32,
            "algorithm": jax.lax.DotAlgorithm(np.float32, np.float32, np.float32),
        }[defect]
    before = typed_trace_manifest(graph)
    placement = params.get("out_sharding")
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        pytest.fail("Admission failures must precede lowerer/builder execution")

    monkeypatch.setattr(DotGeneralPlugin, "lower", forbidden)
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert not calls
    assert caught.value.category == category and caught.value.location == location
    assert caught.value.primitive == (
        "dot_general" if location.startswith("equation") else None
    )
    assert typed_trace_manifest(graph) == before
    assert params.get("out_sharding") is placement


@pytest.mark.parametrize(
    "dot_effect,graph_effect", [(True, False), (True, True), (False, True)]
)
def test_effect_attribution_localizes_only_actual_effectful_dot_equations(
    dot_effect: bool, graph_effect: bool
) -> None:
    graph = _graph()
    graph.eqns[0].effects = ("effect",) if dot_effect else ()
    graph.effects = ("effect",) if graph_effect else ()
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert (caught.value.category, caught.value.location, caught.value.primitive) == (
        ("lowering-defect", "equation 0", "dot_general")
        if dot_effect
        else ("profile-exclusion", "graph", None)
    )


def test_existing_fp16_branch_and_absent_optional_parameters_remain_valid() -> None:
    graph = _graph()
    for var in graph.invars:
        var.aval.dtype = np.float16
    graph.eqns[0].params["preferred_element_type"] = np.float32
    validate_source_jaxpr(graph, 23)
    for key in ("precision", "preferred_element_type"):
        del graph.eqns[0].params[key]
    for var in graph.invars:
        var.aval.dtype = np.float32
    validate_source_jaxpr(graph, 26)


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
def test_audited_artifacts_retain_exact_d2a_probes_and_raising_subnormal_gate(
    dtype: Any, probe: str, monkeypatch: pytest.MonkeyPatch, record_property: Any
) -> None:
    # Reuse the unchanged exact samples, target pins, bit checks and criteria.
    # Only the exporter differs: this helper always calls the audited path.
    monkeypatch.setattr(d2a, "_export", _export)
    d2a.test_exact_selected_numerical_probes(dtype, probe, record_property)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_audited_64_term_probe_keeps_the_predeclared_d2a_tolerance(
    dtype: Any, monkeypatch: pytest.MonkeyPatch, record_property: Any
) -> None:
    monkeypatch.setattr(d2a, "_export", _export)
    d2a.test_declared_longer_reduction_criteria(dtype, record_property)
