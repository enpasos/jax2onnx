# tests/extra_tests/converter/test_source_finalization.py

"""Required SOURCE finalization fails before metadata or file publication."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, user_interface
from jax2onnx.converter import conversion_api
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.precision import get_precision_report


@pytest.fixture(autouse=True)
def _restore_jax_settings() -> Iterator[None]:
    settings = {
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        "jax_numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion),
        "jax_numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
        "jax_platforms": jax.config.jax_platforms,
    }
    try:
        yield
    finally:
        for name, value in settings.items():
            jax.config.update(name, value)


def _graph(value: Any) -> Any:
    return value + value


@onnx_function
def _inner(value: Any) -> Any:
    return value + value


@onnx_function
def _outer(value: Any) -> Any:
    return _inner(value)


def _named(value: Any) -> Any:
    return _inner(value)


def _nested(value: Any) -> Any:
    return _outer(value)


_MODELS = {"graph": _graph, "function": _named, "nested": _nested}


def _export(
    fn: Any, *, diagnostic: bool = False, legacy: bool = False, **options: Any
) -> Any:
    return user_interface._to_onnx(
        fn,
        [jax.ShapeDtypeStruct((2,), np.float32)],
        model_name="strict_finalization",
        precision_policy=PrecisionPolicy.LEGACY if legacy else PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=validate_source_jaxpr if diagnostic else None,
        **options,
    )


def _options(mode: str, path: Path) -> dict[str, Any]:
    return {"return_mode": mode, **({"output_path": path} if mode == "file" else {})}


def _proto(artifact: Any) -> onnx.ModelProto:
    if isinstance(artifact, ir.Model):
        return ir.to_proto(artifact)
    if isinstance(artifact, str):
        return onnx.load(artifact)
    return artifact


def _check_cpu(artifact: Any, expected: np.ndarray) -> None:
    proto = _proto(artifact)
    onnx.checker.check_model(proto, full_check=True)
    value = np.asarray([-2, 1], np.float32)
    for level in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = level
        session = ort.InferenceSession(
            proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        (actual,) = session.run(None, {session.get_inputs()[0].name: value})
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()


def _assert_error(
    error: SourcePrecisionError, *, cause: Exception | None = None
) -> None:
    assert error.phase == "finalization"
    assert error.category == "lowering-defect"
    assert error.opset == 23
    assert error.requirement and error.action
    assert (
        error.__cause__ is cause if cause is not None else error.__cause__ is not None
    )
    assert active_precision_policy() is PrecisionPolicy.LEGACY


def _fail_shape(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    def fail(model: ir.Model) -> None:
        raise failure

    monkeypatch.setattr(conversion_api, "_finalize_model_value_shapes", fail)


def _fail_function(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    original = conversion_api._fix_concat_axis_in_graph

    def fail(graph: ir.Graph) -> None:
        if graph.name != "strict_finalization" and any(
            node.op_type == "Add" for node in graph
        ):
            raise failure
        original(graph)

    monkeypatch.setattr(conversion_api, "_fix_concat_axis_in_graph", fail)


@pytest.mark.parametrize("diagnostic", [False, True], ids=["audited", "diagnostic"])
@pytest.mark.parametrize("scope", ["graph", "function", "nested"])
@pytest.mark.parametrize("mode", ["proto", "ir", "file"])
def test_source_shape_finalization_failure_aborts_all_artifacts(
    scope: str,
    mode: str,
    diagnostic: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("injected shape finalization failure")
    _fail_shape(monkeypatch, failure)
    path = tmp_path / "existing.onnx"
    path.write_bytes(b"preexisting artifact")
    with pytest.raises(SourcePrecisionError, match="shape finalization") as caught:
        _export(_MODELS[scope], diagnostic=diagnostic, **_options(mode, path))
    _assert_error(caught.value, cause=failure)
    assert "strict_finalization" in caught.value.location
    assert path.read_bytes() == b"preexisting artifact"


@pytest.mark.parametrize("diagnostic", [False, True], ids=["audited", "diagnostic"])
@pytest.mark.parametrize("scope", ["function", "nested"])
@pytest.mark.parametrize("mode", ["proto", "ir", "file"])
def test_source_required_function_postprocess_failure_is_contextual(
    scope: str,
    mode: str,
    diagnostic: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("injected required function postprocess failure")
    _fail_function(monkeypatch, failure)
    path = tmp_path / "existing.onnx"
    path.write_bytes(b"preexisting artifact")
    with pytest.raises(SourcePrecisionError, match="function postprocess") as caught:
        _export(_MODELS[scope], diagnostic=diagnostic, **_options(mode, path))
    _assert_error(caught.value, cause=failure)
    assert (
        "function" in caught.value.location and "inner" in caught.value.location.lower()
    )
    assert path.read_bytes() == b"preexisting artifact"


@pytest.mark.parametrize("stage", ["shape", "function"])
@pytest.mark.parametrize("mode", ["proto", "ir", "file"])
def test_legacy_retains_nonfatal_finalization_behavior(
    stage: str, mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = RuntimeError("legacy control")
    (_fail_shape if stage == "shape" else _fail_function)(monkeypatch, failure)
    model = _export(_nested, legacy=True, **_options(mode, tmp_path / "legacy.onnx"))
    assert get_precision_report(model) is None
    _check_cpu(model, np.asarray([-4, 2], np.float32))


def test_failed_source_finalization_never_creates_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_shape(monkeypatch, RuntimeError("abort before save"))
    path = tmp_path / "new.onnx"
    with pytest.raises(SourcePrecisionError):
        _export(_graph, return_mode="file", output_path=path)
    assert not path.exists()


@pytest.mark.parametrize("diagnostic", [False, True])
def test_hit_counter_failures_remain_best_effort(
    diagnostic: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = []

    def fail() -> None:
        observed.append(True)
        raise RuntimeError("optional hit counter")

    monkeypatch.setattr(conversion_api.ps2, "_consume_onnx_function_hits", fail)
    model = _export(_graph, diagnostic=diagnostic)
    assert observed and get_precision_report(model) is not None
    _check_cpu(model, np.asarray([-4, 2], np.float32))


@onnx_function
def _leaky_inner(value: Any) -> Any:
    return jax.nn.leaky_relu(value, negative_slope=0.5)


@onnx_function
def _leaky_outer(value: Any) -> Any:
    return _leaky_inner(value)


def _leaky_model(scope: str) -> Any:
    def target(value: Any) -> Any:
        if scope == "nested":
            return _leaky_outer(value)
        return jax.nn.leaky_relu(value, negative_slope=0.5)

    return target


def _validate_alpha_trace(jaxpr: Any, opset: int) -> None:
    remaining = []
    assert not jaxpr.effects and opset == 23
    for eqn in jaxpr.eqns:
        if eqn.primitive.name != "jax.nn.leaky_relu":
            remaining.append(eqn)
            continue
        assert not eqn.effects and len(eqn.invars) == len(eqn.outvars) == 1
        assert eqn.params == {"negative_slope": 0.5}
        assert all(
            np.dtype(var.aval.dtype) == np.float32 and tuple(var.aval.shape) == (2,)
            for var in (*eqn.invars, *eqn.outvars)
        )
    validate_source_jaxpr(jaxpr.replace(eqns=remaining), opset)


def _stage_alpha_override(monkeypatch: pytest.MonkeyPatch) -> None:
    apply = conversion_api._apply_late_ir_attr_overrides

    def stage(model: ir.Model, ctx: IRContext) -> None:
        graphs = [(model.graph, ctx.attr_overrides)]
        graphs.extend((fn.graph, fn._attr_overrides) for fn in model.functions.values())
        for graph, overrides in graphs:
            for node in graph.all_nodes():
                if node.op_type == "LeakyRelu":
                    if "alpha" in node.attributes:
                        del node.attributes["alpha"]
                    overrides[node.name] = {"alpha": np.float32(0.5)}
        apply(model, ctx)

    monkeypatch.setattr(conversion_api, "_apply_late_ir_attr_overrides", stage)


def _alpha_export(scope: str, **options: Any) -> Any:
    return user_interface._to_onnx(
        _leaky_model(scope),
        [jax.ShapeDtypeStruct((2,), np.float32)],
        model_name="strict_finalization",
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=_validate_alpha_trace,
        **options,
    )


@pytest.mark.parametrize("scope", ["graph", "nested"])
@pytest.mark.parametrize("mode", ["proto", "ir", "file"])
def test_valid_explicit_attribute_override_preserves_actual_jax_semantics(
    scope: str, mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = np.asarray(_leaky_model(scope)(jax.numpy.asarray([-2, 1], np.float32)))
    assert expected.tobytes() == np.asarray([-1, 1], np.float32).tobytes()
    _stage_alpha_override(monkeypatch)
    artifact = _alpha_export(scope, **_options(mode, tmp_path / "positive.onnx"))
    _check_cpu(artifact, expected)


@pytest.mark.parametrize("scope", ["graph", "nested"])
@pytest.mark.parametrize("mode", ["proto", "ir", "file"])
def test_nonconvertible_explicit_attribute_cannot_silently_keep_schema_default(
    scope: str, mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage_alpha_override(monkeypatch)
    convert = conversion_api._convert_ir_attr
    monkeypatch.setattr(
        conversion_api,
        "_convert_ir_attr",
        lambda name, value: None if name == "alpha" else convert(name, value),
    )
    path = tmp_path / "existing.onnx"
    path.write_bytes(b"preexisting artifact")
    with pytest.raises(
        SourcePrecisionError, match="explicit attribute override"
    ) as caught:
        _alpha_export(scope, **_options(mode, path))
    _assert_error(caught.value)
    assert caught.value.primitive == "LeakyRelu"
    assert "alpha" in caught.value.location
    assert ("function" if scope == "nested" else "graph") in caught.value.location
    assert isinstance(caught.value.__cause__, ValueError)
    assert path.read_bytes() == b"preexisting artifact"


def test_legacy_defaulted_attribute_is_checker_valid_but_numerically_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stage_alpha_override(monkeypatch)
    monkeypatch.setattr(conversion_api, "_convert_ir_attr", lambda name, value: None)
    model = _export(_leaky_model("graph"), legacy=True)
    onnx.checker.check_model(model, full_check=True)
    assert all(
        attr.name != "alpha"
        for attr in next(
            n for n in model.graph.node if n.op_type == "LeakyRelu"
        ).attribute
    )
    # The schema default is 0.01, but the original JAX program requests 0.5.
    _check_cpu(model, np.asarray([-0.02, 1], np.float32))
    assert not np.array_equal(
        np.asarray([-0.02, 1], np.float32), np.asarray([-1, 1], np.float32)
    )


def test_source_skips_override_for_a_node_actually_removed_by_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lower = conversion_api._lower_jaxpr_equations

    def add_dead_node(ctx: IRContext, jaxpr: Any) -> None:
        lower(ctx, jaxpr)
        ctx.builder.op(
            "Identity",
            [ctx.builder.inputs[0]],
            name="removed_override_target",
            output=ir.val("dead_output", dtype=ir.DataType.FLOAT, shape=[2]),
        )
        ctx.add_node_attr_override(
            "removed_override_target", {"unrepresentable": object()}
        )

    monkeypatch.setattr(conversion_api, "_lower_jaxpr_equations", add_dead_node)
    model = _export(_graph)
    assert not any(node.name == "removed_override_target" for node in model.graph.node)
    _check_cpu(model, np.asarray([-4, 2], np.float32))


def test_source_does_not_apply_global_overrides_to_a_different_function_graph() -> None:
    ctx = IRContext(
        opset=23, enable_double_precision=False, precision_policy=PrecisionPolicy.SOURCE
    )
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2])
    model = ir.Model(
        ir.Graph([x], [x], nodes=[], opset_imports={"": 23}, name="main"), ir_version=10
    )
    fx, fy = ir.val("fx", dtype=ir.DataType.FLOAT, shape=[2]), ir.val(
        "fy", dtype=ir.DataType.FLOAT, shape=[2]
    )
    node = ir.Node("", "LeakyRelu", [fx], outputs=[fy], name="other_graph_target")
    function = ir.Function(
        "test",
        "OtherGraph",
        graph=ir.Graph(
            [fx], [fy], nodes=[node], opset_imports={"": 23}, name="function"
        ),
        attributes={},
    )
    function._attr_overrides = {}
    model.functions[function.identifier()] = function
    ctx.add_node_attr_override("other_graph_target", {"alpha": object()})
    conversion_api._apply_late_ir_attr_overrides(model, ctx)
    assert "alpha" not in node.attributes
