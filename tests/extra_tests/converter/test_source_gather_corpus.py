# tests/extra_tests/converter/test_source_gather_corpus.py

"""One original Gather diagnostic artifact; raw and numerical failures stay separate."""

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime
import pytest

from jax2onnx._compat.jax import ClosedJaxpr
from jax2onnx.converter.precision_policy import active_precision_policy
from jax2onnx.precision import CheckStatus, SampleAcceptanceError, get_precision_report
from scripts import source_precision_probe as probe
from tests.extra_tests.converter import source_precision_corpus_oracles as oracles
from tests.extra_tests.converter import source_precision_support as source
from tests.extra_tests.converter.source_precision_diagnostics import (
    DiagnosticValidator,
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
    _globals,
    _settings,
)


_COUNTERPART = oracles.GATHER_COUNTERPARTS[0]


@pytest.fixture(autouse=True)
def _preserve_and_deny_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    hidden, effective = _globals(), _settings()
    callables = jnp.matmul, jnp.reshape, jnp.add
    policy = active_precision_policy()
    attempts = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        attempts.append((args, kwargs))
        pytest.fail("Gather corpus structure must never create an ORT session")

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    try:
        yield
    finally:
        for name, value in hidden.items():
            jax.config.update(name, value)
        assert not attempts
        assert (_globals(), _settings()) == (hidden, effective)
        assert (jnp.matmul, jnp.reshape, jnp.add) == callables
        assert active_precision_policy() is policy


@pytest.fixture(scope="module")
def original() -> Any:
    metadata = source.raw_source_case("plugin::gather", 0)
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance("plugin::gather", 0)
    )[1]
    before = (
        _globals(),
        _settings(),
        (jnp.matmul, jnp.reshape, jnp.add),
        active_precision_policy(),
    )
    attempts = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        attempts.append((args, kwargs))
        raise AssertionError("Gather fixture cannot execute ORT")

    with patch.object(onnxruntime, "InferenceSession", forbidden):
        prepared = source.prepare_source_case(metadata, scenario)
        validator = DiagnosticValidator()
        activated = []

        def observe(closed: Any, opset: int) -> None:
            activated.append(json.loads(json.dumps(typed_trace_manifest(closed))))
            validator(closed, opset)

        result = source.run_source_case(
            metadata,
            scenario,
            optimization=None,
            diagnostic_jaxpr_validator=observe,
            _prepared=prepared,
            _corpus_counterpart=_COUNTERPART,
        )
    assert not attempts
    assert (
        _globals(),
        _settings(),
        (jnp.matmul, jnp.reshape, jnp.add),
        active_precision_policy(),
    ) == before
    assert result.model is not None, result.audit.detail
    expected_root = json.loads(prepared.original_trace_json)
    assert activated[0] == expected_root
    native_where_body = expected_root["equations"][-1]["params"]["jaxpr"]
    assert native_where_body in activated
    assert all(trace in (expected_root, native_where_body) for trace in activated)
    assert validator.failure is None and validator.scopes
    return metadata, scenario, prepared, result


def test_actual_raw_gather_has_separate_failed_callback_and_validated_source_structure(
    original: Any, tmp_path: Path
) -> None:
    metadata, scenario, prepared, result = original
    assert result.stage == "diagnostic-artifact-checked"
    assert result.schema is CheckStatus.VALIDATED
    assert result.audit.status is CheckStatus.UNVERIFIED
    assert result.raw_callback_status == "failed"
    assert result.corpus_structure.accepted
    assert result.corpus_structure.source_structure is CheckStatus.VALIDATED
    assert result.numerical_status is CheckStatus.UNVERIFIED
    assert result.validation is not None and not result.validation.samples
    assert result.validation.target is None
    with pytest.raises(SampleAcceptanceError, match="unverified"):
        result.validation.require_sample_acceptance()
    assert result.export_settings == prepared.settings
    assert scenario.criteria is prepared.criteria is None
    assert metadata["callable"] is prepared.fn
    assert prepared.settings.x64_enabled is True
    assert prepared.settings.default_matmul_precision is None
    assert [
        item.logical_sha256
        for item in (*prepared.input_evidence, *prepared.reference_evidence)
    ] == list(oracles._GATHER_VALUE_SHA256)
    report = get_precision_report(result.model)
    assert report is not None and report.profile == "source-diagnostic-v1"
    assert {item.category: item.status for item in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.UNVERIFIED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    path = tmp_path / "diagnostic.onnx"
    onnx.save_model(result.model, path)
    reloaded = onnx.load_model(path)
    onnx.checker.check_model(reloaded, full_check=True)
    assert _COUNTERPART.check(prepared, reloaded).accepted
    assert get_precision_report(reloaded) == report
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == result.validation.artifact_sha256
    )


def test_ordinary_probe_retains_original_failure_and_x64_off_configuration_conflict(
    original: Any,
) -> None:
    metadata, scenario, prepared, _ = original
    raw = source.run_source_case(
        metadata,
        scenario,
        optimization=None,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
        _prepared=prepared,
    )
    assert raw.model is None and raw.schema is CheckStatus.VALIDATED
    assert raw.raw_callback_status == "failed" and raw.corpus_structure is None
    off = source.raw_source_scenarios(metadata, scenario.provenance)[0]
    result = source.run_source_case(
        metadata,
        off,
        optimization=None,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
        _corpus_counterpart=_COUNTERPART,
    )
    assert result.model is result.prepared is None
    assert result.stage == "source-preparation" and result.category == "migration-gap"
    assert "canonicalize" in result.audit.detail


@pytest.mark.parametrize("optimization", ["disabled", "all"])
def test_gather_counterpart_requires_actual_none_optimization(
    original: Any, optimization: str
) -> None:
    metadata, scenario, prepared, _ = original
    result = source.run_source_case(
        metadata,
        scenario,
        optimization=optimization,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
        _prepared=prepared,
        _corpus_counterpart=_COUNTERPART,
    )
    assert result.model is None and "optimization=None" in result.audit.detail


def test_integer_input_delegates_to_actual_converter_without_admitting_product_gather(
    original: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, scenario, prepared, _ = original
    actual_export = source._export_prepared
    calls = []

    def delegate(observed: Any, **kwargs: Any) -> Any:
        calls.append(observed)
        return actual_export(observed, **kwargs)

    monkeypatch.setattr(source, "_export_prepared", delegate)
    result = source.run_source_case(
        metadata,
        scenario,
        optimization=None,
        _prepared=prepared,
        _corpus_counterpart=_COUNTERPART,
    )
    assert result.model is None and result.corpus_structure is None
    assert len(calls) == 1 and calls[0] is prepared
    assert result.stage != "profile-admission"
    assert result.raw_callback_status == "not-executed"
    assert result.audit.status is CheckStatus.UNVERIFIED


@pytest.mark.parametrize("kind", ["new-equal", "wrong-row", "subclass", "duck"])
def test_only_exact_reviewed_gather_singleton_is_accepted(
    original: Any, kind: str
) -> None:
    _, _, prepared, result = original

    class Child(oracles.GatherCounterpart):
        pass

    class Duck:
        def check(self, *args: Any) -> Any:
            pytest.fail("A duck-typed checker must never run")

    if kind == "duck":
        metadata, scenario = original[:2]
        rejected = source.run_source_case(
            metadata,
            scenario,
            optimization=None,
            diagnostic_jaxpr_validator=DiagnosticValidator(),
            _prepared=prepared,
            _corpus_counterpart=Duck(),
        )
        assert rejected.model is None and "actual reviewed" in rejected.audit.detail
        return
    counterfeit = (
        Child(_COUNTERPART.case)
        if kind == "subclass"
        else oracles.GatherCounterpart(
            replace(_COUNTERPART.case, ordinal=1)
            if kind == "wrong-row"
            else _COUNTERPART.case
        )
    )
    evidence = counterfeit.check(prepared, result.model)
    assert evidence.source_structure is CheckStatus.FAILED
    assert "singleton" in evidence.source_detail


@pytest.mark.parametrize("outcome", ["absent", "passed", "failed", "error"])
def test_changed_raw_callback_never_acquires_source_structure(
    original: Any, outcome: str
) -> None:
    metadata, _, prepared, result = original
    broken = dict(metadata)
    if outcome == "absent":
        broken.pop("post_check_onnx_graph")
    elif outcome in {"passed", "failed"}:
        broken["post_check_onnx_graph"] = lambda _: outcome == "passed"
    else:

        def raising(_: Any) -> bool:
            raise RuntimeError("original callback corruption")

        broken["post_check_onnx_graph"] = raising
    evidence = _COUNTERPART.check(replace(prepared, metadata=broken), result.model)
    assert evidence.raw_callback_status == outcome
    assert evidence.source_structure is CheckStatus.FAILED and not evidence.accepted


@pytest.mark.parametrize(
    "defect",
    [
        "origin",
        "ordinal",
        "context",
        "component",
        "testcase",
        "contract",
        "opset",
        "variant",
        "callable",
        "same-file-callable",
        "source-path",
        "source-bytes",
        "criteria",
        "scenario-criteria",
        "x64",
        "precision",
        "promotion",
        "rank",
        "spec-weak",
        "spec-dtype",
        "spec-shape",
        "trace-weak",
        "trace-dtype",
        "trace-shape",
        "feed-bytes",
        "feed-shape",
        "feed-dtype",
        "feed-weak",
        "reference-bytes",
        "reference-shape",
        "reference-dtype",
        "reference-weak",
        "evidence",
        "recipe",
        "recipe-weak",
        "stored-trace",
        "actual-trace",
        "jit-parameter",
        "output-tree",
    ],
)
def test_gather_source_binding_rejects_forged_preparation(
    original: Any, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, scenario, prepared, result = original
    broken = prepared
    if defect in {"origin", "ordinal"}:
        provenance = replace(
            scenario.provenance,
            **({"origin": "plugin::other"} if defect == "origin" else {"ordinal": 1}),
        )
        broken = replace(prepared, scenario=replace(scenario, provenance=provenance))
    elif defect in {
        "context",
        "component",
        "testcase",
        "contract",
        "opset",
        "variant",
        "callable",
    }:
        updated = dict(metadata)
        key, value = {
            "context": ("context", "other"),
            "component": ("component", "other"),
            "testcase": ("testcase", "other"),
            "contract": ("expected_output_shapes", [(3, 2)]),
            "opset": ("opset_version", 22),
            "variant": ("run_only_f64_variant", False),
            "callable": ("callable", lambda *args: prepared.fn(*args)),
        }[defect]
        updated[key] = value
        broken = replace(prepared, metadata=updated)
    elif defect == "same-file-callable":

        def surrogate(*args: Any) -> Any:
            return prepared.fn(*args)

        broken = replace(prepared, fn=surrogate)
        original_sourcefile = oracles.inspect.getsourcefile
        monkeypatch.setattr(
            oracles.inspect,
            "getsourcefile",
            lambda fn: (
                str(oracles.REPOSITORY_ROOT / oracles.GATHER_SOURCE_PATH)
                if fn is surrogate
                else original_sourcefile(fn)
            ),
        )
    elif defect == "source-path":
        monkeypatch.setattr(
            oracles.inspect, "getsourcefile", lambda _: str(Path(__file__))
        )
    elif defect == "source-bytes":
        original_read = Path.read_bytes

        def changed(path: Path) -> bytes:
            data = original_read(path)
            return (
                data + b"\n"
                if path == oracles.REPOSITORY_ROOT / oracles.GATHER_SOURCE_PATH
                else data
            )

        monkeypatch.setattr(Path, "read_bytes", changed)
    elif defect in {"criteria", "scenario-criteria"}:
        criterion = source.SampleCriteria("bitwise")
        broken = replace(
            prepared,
            **(
                {"criteria": criterion}
                if defect == "criteria"
                else {"scenario": replace(scenario, criteria=criterion)}
            ),
        )
    elif defect in {"x64", "precision", "promotion", "rank"}:
        key, value = {
            "x64": ("x64_enabled", False),
            "precision": ("default_matmul_precision", "float32"),
            "promotion": ("numpy_dtype_promotion", "strict"),
            "rank": ("numpy_rank_promotion", "raise"),
        }[defect]
        broken = replace(prepared, settings=replace(prepared.settings, **{key: value}))
    elif defect.startswith("spec-"):
        specs = list(prepared.input_specs)
        specs[0] = jax.ShapeDtypeStruct(
            (3, 4) if defect == "spec-shape" else (4, 3),
            np.float32 if defect == "spec-dtype" else np.float64,
            weak_type=defect == "spec-weak",
        )
        broken = replace(prepared, input_specs=tuple(specs))
    elif defect.startswith("trace-"):
        key, value = {
            "trace-weak": ("weak_type", True),
            "trace-dtype": ("dtype", "float32"),
            "trace-shape": ("shape", (3, 4)),
        }[defect]
        broken = replace(
            prepared,
            traced_inputs=(
                replace(prepared.traced_inputs[0], **{key: value}),
                prepared.traced_inputs[1],
            ),
        )
    elif defect.startswith("feed-"):
        values = list(prepared.arguments)
        array = np.asarray(values[0]).copy()
        if defect == "feed-bytes":
            array[0, 0] += 1
        elif defect == "feed-shape":
            array = array.reshape(3, 4)
        elif defect == "feed-dtype":
            array = array.astype(np.float32)
        else:

            class Weak(np.ndarray):
                weak_type = True

            array = array.view(Weak)
        values[0] = array
        broken = replace(prepared, arguments=tuple(values))
    elif defect.startswith("reference-"):
        array = prepared.reference_outputs[0].copy()
        if defect == "reference-bytes":
            array[0, 0] += 1
        elif defect == "reference-shape":
            array = array.reshape(3, 2)
        elif defect == "reference-dtype":
            array = array.astype(np.float32)
        else:

            class WeakReference(np.ndarray):
                weak_type = True

            array = array.view(WeakReference)
        broken = replace(prepared, reference_outputs=(array,))
    elif defect == "evidence":
        broken = replace(
            prepared,
            input_evidence=(
                replace(prepared.input_evidence[0], logical_sha256="0" * 64),
                prepared.input_evidence[1],
            ),
        )
    elif defect in {"recipe", "recipe-weak"}:
        array = np.asarray(scenario.input_values[0]).copy()
        if defect == "recipe":
            array[:] = 0
        else:

            class WeakRecipe(np.ndarray):
                weak_type = True

            array = array.view(WeakRecipe)
        broken = replace(
            prepared,
            scenario=replace(
                scenario,
                input_values=(array, scenario.input_values[1]),
            ),
        )
    elif defect == "stored-trace":
        manifest = json.loads(prepared.original_trace_json)
        manifest["equations"][-1]["params"]["inline"] = "changed"
        broken = replace(prepared, original_trace_json=json.dumps(manifest))
    elif defect in {"actual-trace", "jit-parameter"}:
        body = prepared.original_jaxpr.jaxpr
        equations = list(body.eqns)
        if defect == "actual-trace":
            equations = equations[:-1]
        else:
            jit = equations[-1]
            equations[-1] = jit.replace(params={**jit.params, "inline": True})
        closed = ClosedJaxpr(
            body.replace(eqns=equations), prepared.original_jaxpr.consts
        )
        broken = replace(prepared, original_jaxpr=closed)
    else:
        broken = replace(prepared, output_tree="PyTreeDef((*,))")
    evidence = _COUNTERPART.check(broken, result.model)
    assert evidence.source_structure is CheckStatus.FAILED, defect
    assert not evidence.accepted


def test_gather_is_one_exact_mapping_in_unchanged_26_cell_selector() -> None:
    case = _COUNTERPART.case
    assert case is oracles.CORPUS_REPLAY_CASES[5]
    assert (
        oracles.counterpart_for(case.origin, case.ordinal, case.contract_sha256)
        is _COUNTERPART
    )
    assert oracles.counterpart_for(case.origin, 1, case.contract_sha256) is None
    assert oracles.counterpart_for(case.origin, case.ordinal, "0" * 64) is None
    assert (
        oracles.APPROVED_SELECTOR_SHA256
        == "1b4f2f47d8651d8ee4e7967e7654af520c11d40f96c4836178e6ecd9720d8d09"
    )
    assert len(oracles.CORPUS_REPLAY_CASES) == 13
    assert len(oracles.DOT_COUNTERPARTS) == 4 and len(oracles.LINEAR_COUNTERPARTS) == 2
    assert probe.CORPUS_CONTROL_REVISION == "source-static-add-ordinary-dot-v4"


def _node(model: onnx.ModelProto, op: str, *, last: bool = False) -> onnx.NodeProto:
    matches = [node for node in model.graph.node if node.op_type == op]
    assert matches, op
    return matches[-1 if last else 0]


def _constant(model: onnx.ModelProto, name: str) -> onnx.TensorProto:
    for tensor in model.graph.initializer:
        if tensor.name == name:
            return tensor
    for node in model.graph.node:
        if node.op_type == "Constant" and list(node.output) == [name]:
            return next(
                attribute.t for attribute in node.attribute if attribute.name == "value"
            )
    raise AssertionError(f"Expected the actually observed constant producer for {name}")


def _change_constant(model: onnx.ModelProto, name: str, value: Any) -> None:
    tensor = _constant(model, name)
    tensor.CopyFrom(onnx.numpy_helper.from_array(np.asarray(value), name=tensor.name))


@pytest.mark.parametrize(
    "defect",
    [
        "ignored-data",
        "ignored-indices",
        "index-threshold",
        "index-offset",
        "index-select",
        "native-index-cast",
        "broadcast-axis",
        "broadcast-order",
        "gather-axis",
        "gather-indices",
        "multiply-two",
        "sine-input",
        "cosine-input",
        "cosine-half",
        "cosine-square",
        "cosine-double",
        "cosine-one",
        "cosine-subtract",
        "sine-as-cosine",
        "add-branches",
        "add-order",
        "sine-bypass",
        "cosine-bypass",
        "greater-threshold",
        "greater-direction",
        "where-condition",
        "where-arms",
        "false-scalar",
        "false-negative-zero",
        "false-dtype",
        "copy-index-source",
        "copy-index-type",
        "copy-stack-axis",
        "copy-gather-axis",
        "copy-squeeze-axis",
        "copy-bypass",
        "copy-output-binding",
        "unused-zero",
        "unused-node",
        "unused-constant",
        "intermediate-fp32",
        "extra-input",
        "extra-output",
    ],
)
def test_gather_counterpart_rejects_valid_schema_live_graph_corruptions(
    original: Any, defect: str
) -> None:
    _, _, prepared, result = original
    model = deepcopy(result.model)
    graph = model.graph
    gather, multiply = _node(model, "Gather"), _node(model, "Mul")
    sine, half_sine = _node(model, "Sin"), _node(model, "Sin", last=True)
    greater, copy = _node(model, "Greater"), _node(model, "GatherElements")
    summed = next(
        node
        for node in graph.node
        if node.op_type == "Add" and node.output[0] == greater.input[0]
    )
    producers = {name: node for node in graph.node for name in node.output}
    stack = producers[copy.input[0]]
    index_unsqueeze = producers[copy.input[1]]
    copy_index = producers[index_unsqueeze.input[0]]
    false_arm = producers[stack.input[0]]
    squeeze = producers[graph.output[0].name]
    assert stack.op_type == "Concat" and copy_index.op_type == "Cast"
    assert false_arm.op_type == "Unsqueeze" and squeeze.op_type == "Squeeze"
    cosine = producers[summed.input[1]]
    doubled = producers[cosine.input[1]]
    square = producers[doubled.input[0]]
    half_angle = producers[half_sine.input[0]]
    assert cosine.op_type == "Sub" and half_angle.op_type == "Mul"
    if defect == "ignored-data":
        name = "counterpart_substituted_data"
        graph.initializer.append(
            onnx.numpy_helper.from_array(np.asarray(prepared.arguments[0]), name=name)
        )
        gather.input[0] = name
    elif defect == "ignored-indices":
        name = "counterpart_substituted_indices"
        graph.initializer.append(
            onnx.numpy_helper.from_array(np.asarray(prepared.arguments[1]), name=name)
        )
        for node in graph.node:
            for index, value in enumerate(node.input):
                if value == graph.input[1].name:
                    node.input[index] = name
    elif defect in {
        "index-threshold",
        "index-offset",
        "index-select",
        "native-index-cast",
    }:
        if defect == "index-threshold":
            less = _node(model, "Less")
            original_value = onnx.numpy_helper.to_array(_constant(model, less.input[1]))
            _change_constant(model, less.input[1], np.full_like(original_value, -1))
        elif defect == "index-offset":
            add = next(
                node
                for node in graph.node
                if node.op_type == "Add" and node.output[0] != summed.output[0]
            )
            original_value = onnx.numpy_helper.to_array(_constant(model, add.input[1]))
            _change_constant(model, add.input[1], np.full_like(original_value, 3))
        elif defect == "index-select":
            native = _node(model, "Where")
            native.input[1], native.input[2] = native.input[2], native.input[1]
        else:
            cast = _node(model, "Cast")
            next(
                attribute for attribute in cast.attribute if attribute.name == "to"
            ).i = onnx.TensorProto.INT64
    elif defect == "gather-axis":
        next(
            attribute for attribute in gather.attribute if attribute.name == "axis"
        ).i = -2
    elif defect in {"broadcast-axis", "broadcast-order"}:
        index_reshape = producers[gather.input[1]]
        index_expand = producers[index_reshape.input[0]]
        index_broadcast = producers[index_expand.input[0]]
        concats = [
            producers[index_expand.input[1]],
            producers[index_broadcast.input[1]],
        ]
        assert len(concats) == 2 and all(node.op_type == "Concat" for node in concats)
        if defect == "broadcast-axis":
            next(
                attribute
                for attribute in concats[0].attribute
                if attribute.name == "axis"
            ).i = -1
        else:
            for node in concats:
                node.input[0], node.input[1] = node.input[1], node.input[0]
    elif defect == "gather-indices":
        gather.input[1] = graph.input[1].name
    elif defect == "multiply-two":
        value = onnx.numpy_helper.to_array(_constant(model, multiply.input[1]))
        _change_constant(model, multiply.input[1], np.full_like(value, 3.0))
    elif defect in {"sine-input", "cosine-input"}:
        (sine if defect == "sine-input" else half_angle).input[0] = gather.output[0]
    elif defect in {"cosine-half", "cosine-double", "cosine-one"}:
        role = {
            "cosine-half": half_angle,
            "cosine-double": doubled,
            "cosine-one": cosine,
        }[defect]
        port = 0 if defect == "cosine-one" else 1
        value = onnx.numpy_helper.to_array(_constant(model, role.input[port]))
        _change_constant(model, role.input[port], np.full_like(value, 1.25))
    elif defect == "cosine-square":
        square.input[1] = sine.output[0]
    elif defect == "cosine-subtract":
        cosine.input[0], cosine.input[1] = cosine.input[1], cosine.input[0]
    elif defect == "sine-as-cosine":
        sine.op_type = "Cos"
    elif defect == "add-branches":
        summed.input[1] = summed.input[0]
    elif defect == "add-order":
        summed.input[0], summed.input[1] = summed.input[1], summed.input[0]
    elif defect in {"sine-bypass", "cosine-bypass"}:
        summed.input[0 if defect == "sine-bypass" else 1] = multiply.output[0]
    elif defect == "greater-threshold":
        value = onnx.numpy_helper.to_array(_constant(model, greater.input[1]))
        _change_constant(model, greater.input[1], np.full_like(value, 0.25))
    elif defect == "greater-direction":
        greater.input[0], greater.input[1] = greater.input[1], greater.input[0]
    elif defect == "where-condition":
        name = "counterpart_always_true"
        graph.initializer.append(
            onnx.numpy_helper.from_array(np.ones((2, 3), np.bool_), name=name)
        )
        copy_index.input[0] = name
    elif defect == "where-arms":
        stack.input[0], stack.input[1] = stack.input[1], stack.input[0]
    elif defect in {"false-scalar", "false-negative-zero", "false-dtype"}:
        name, restored = false_arm.input[0], "counterpart_zero64"
        if defect == "false-scalar":
            _change_constant(model, name, np.asarray(0.0, np.float64))
            shape = "counterpart_zero_shape"
            graph.initializer.append(
                onnx.numpy_helper.from_array(np.asarray([2, 3], np.int64), name=shape)
            )
            graph.node.insert(
                list(graph.node).index(false_arm),
                onnx.helper.make_node("Expand", [name, shape], [restored]),
            )
            false_arm.input[0] = restored
        elif defect == "false-dtype":
            # Keep a valid schema using an explicit narrowing/widening route.
            _change_constant(model, name, np.zeros((2, 3), np.float32))
            graph.node.insert(
                list(graph.node).index(false_arm),
                onnx.helper.make_node(
                    "Cast", [name], [restored], to=onnx.TensorProto.DOUBLE
                ),
            )
            false_arm.input[0] = restored
        else:
            _change_constant(model, name, np.full((2, 3), -0.0, np.float64))
    elif defect == "copy-index-source":
        name = "counterpart_copy_indices"
        graph.initializer.append(
            onnx.numpy_helper.from_array(np.zeros((1, 2, 3), np.int32), name=name)
        )
        copy.input[1] = name
    elif defect == "copy-index-type":
        next(
            attribute for attribute in copy_index.attribute if attribute.name == "to"
        ).i = onnx.TensorProto.INT64
    elif defect in {"copy-stack-axis", "copy-gather-axis"}:
        changed = stack if defect == "copy-stack-axis" else copy
        next(
            attribute for attribute in changed.attribute if attribute.name == "axis"
        ).i = 1
    elif defect == "copy-squeeze-axis":
        # A distinct valid spelling of axis 0 must still fail the literal signature.
        name = "counterpart_squeeze_axis"
        graph.initializer.append(
            onnx.numpy_helper.from_array(np.asarray([-3], np.int64), name=name)
        )
        squeeze.input[1] = name
    elif defect == "copy-bypass":
        squeeze.input[0] = false_arm.output[0]
    elif defect == "copy-output-binding":
        graph.output[0].name = summed.output[0]
    elif defect in {"unused-zero", "unused-constant"}:
        value = (
            np.zeros((2, 3), np.float64)
            if defect == "unused-zero"
            else np.asarray(123, np.int64)
        )
        graph.initializer.append(
            onnx.numpy_helper.from_array(value, name="counterpart_unused")
        )
    elif defect == "unused-node":
        graph.node.append(
            onnx.helper.make_node(
                "Identity", [graph.input[0].name], ["counterpart_unused"]
            )
        )
    elif defect == "intermediate-fp32":
        position = list(graph.node).index(multiply)
        original_input = multiply.input[0]
        graph.node.insert(
            position,
            onnx.helper.make_node(
                "Cast",
                [original_input],
                ["counterpart_narrow"],
                to=onnx.TensorProto.FLOAT,
            ),
        )
        graph.node.insert(
            position + 1,
            onnx.helper.make_node(
                "Cast",
                ["counterpart_narrow"],
                ["counterpart_widen"],
                to=onnx.TensorProto.DOUBLE,
            ),
        )
        multiply.input[0] = "counterpart_widen"
    elif defect == "extra-input":
        graph.input.append(
            onnx.helper.make_tensor_value_info(
                "counterpart_extra", onnx.TensorProto.DOUBLE, [2, 3]
            )
        )
    else:
        graph.output.append(deepcopy(graph.input[0]))
    # Remove stale inferred intermediate declarations before independent inference.
    del graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    with pytest.raises(ValueError, match="exact reviewed native graph"):
        _COUNTERPART._check_model(model)
    evidence = _COUNTERPART.check(prepared, model)
    assert not evidence.accepted and evidence.source_structure is CheckStatus.FAILED


def test_historical_v1_literal_identity_is_preserved() -> None:
    assert oracles.fingerprint(oracles._GATHER_GRAPH_TEMPLATE_V1) == (
        "473035f7270ed43d80142efa6ec6c0af099e9c61970ade73537fb4e0934ff408"
    )
    assert _COUNTERPART.identifier == (
        "gather-trig-native-copy-source-counterpart-v2:plugin::gather:0"
    )


def test_current_v2_rejects_schema_valid_historical_v1(original: Any) -> None:
    _, _, prepared, result = original
    model = deepcopy(result.model)
    graph = model.graph
    greater = _node(model, "Greater")
    copy = _node(model, "GatherElements")
    producers = {name: node for node in graph.node for name in node.output}
    stack = producers[copy.input[0]]
    false_arm = producers[stack.input[0]]
    false_name = false_arm.input[0]
    output_name = graph.output[0].name
    del graph.node[19:]
    del graph.initializer[13:]
    graph.node.append(
        onnx.helper.make_node(
            "Where", [greater.output[0], greater.input[0], false_name], [output_name]
        )
    )
    del graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    assert oracles._gather_graph_signature(model) == oracles._GATHER_GRAPH_TEMPLATE_V1
    with pytest.raises(ValueError, match="exact reviewed native graph"):
        _COUNTERPART._check_model(model)
    evidence = _COUNTERPART.check(prepared, model)
    assert evidence.raw_callback_status == "failed"
    assert "exact reviewed native graph" in evidence.source_detail
    assert not evidence.accepted and evidence.source_structure is CheckStatus.FAILED


def test_schema_failure_is_distinct_from_valid_schema_structure_rejection(
    original: Any,
) -> None:
    _, _, prepared, result = original
    model = deepcopy(result.model)
    gather = _node(model, "Gather")
    gather.attribute.append(onnx.helper.make_attribute("unreviewed_attribute", 1))
    del model.graph.value_info[:]
    with pytest.raises(onnx.checker.ValidationError):
        onnx.checker.check_model(model, full_check=True)
    evidence = _COUNTERPART.check(prepared, model)
    assert evidence.source_structure is CheckStatus.FAILED
    assert not evidence.accepted
