# tests/extra_tests/converter/test_source_leading_batch_corpus.py

"""Original five-anchor preparation and strict current-counterpart provenance."""

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime
import pytest

from jax2onnx.converter.precision_policy import active_precision_policy
from jax2onnx.plugins.plugin_system import construct_and_call
from jax2onnx.precision import (
    CheckStatus,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from scripts import source_precision_probe as probe
from scripts.source_precision_inventory import raw_registrations
from tests.extra_tests.converter import source_precision_corpus_oracles as oracles
from tests.extra_tests.converter import source_precision_support as source
from tests.extra_tests.converter import (
    test_source_leading_batch_matrix_bias as production,
)


@pytest.fixture(autouse=True)
def _preserve_and_deny_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    hidden, effective = production._globals(), production._settings()
    originals = jnp.matmul, jnp.reshape, jnp.add

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "Five-anchor preparation is artifact-only; ORT remains separately released"
        )

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    try:
        with jax.enable_x64(False), jax.default_matmul_precision(None):
            yield
    finally:
        for name, value in hidden.items():
            jax.config.update(name, value)
        assert (production._globals(), production._settings()) == (hidden, effective)
        assert (jnp.matmul, jnp.reshape, jnp.add) == originals
        assert active_precision_policy().value == "legacy"


def _metadata(case: Any) -> dict[str, Any]:
    origin, ordinal, digest = case
    metadata = source.raw_source_case(origin, ordinal)
    unresolved: list[str] = []
    contract = oracles.canonical_contract(
        {
            key: value
            for key, value in metadata.items()
            if key not in {"testcase", "description", "context", "component"}
        },
        unresolved=unresolved,
    )
    assert not unresolved and oracles.fingerprint(contract) == digest
    return metadata


def _scenario(case: Any, metadata: Any, x64: bool, batch: int = 2) -> Any:
    raw = source.raw_source_scenarios(metadata, source.SourceProvenance(*case[:2]))[
        int(x64)
    ]
    assert raw.criteria is None and raw.model_dtype is source.PRESERVE_FACTORY_DTYPE
    values = production._feeds(metadata["input_shapes"], batch)
    return replace(
        raw, input_values=values, concrete_shapes=tuple(v.shape for v in values)
    )


@pytest.mark.parametrize("case", production._CASES)
@pytest.mark.parametrize("x64", [False, True])
def test_exact_five_raw_sources_prepare_one_instance_artifact_and_two_batch_feeds(
    case: Any, x64: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = _metadata(case)
    recipe = metadata["callable"]
    counts = []
    if case[0] == "plugin::nnx.linear":
        original = type(recipe).instantiate

        def instantiate(self: Any) -> Any:
            counts.append(self)
            return original(self)

        monkeypatch.setattr(type(recipe), "instantiate", instantiate)
        assert recipe._dtype is None and recipe._instance is None
    scenario = _scenario(case, metadata, x64)
    prepared = source.prepare_source_case(metadata, scenario)
    assert prepared.criteria is None and prepared.scenario.criteria is None
    assert prepared.settings.x64_enabled is x64
    assert prepared.settings.default_matmul_precision is None
    assert all(
        p.dtype == "float32" and p.weak_type is False
        for p in (*prepared.traced_inputs, *prepared.traced_outputs)
    )
    counterpart = oracles.counterpart_for(*case)
    if case[0] == "plugin::nnx.linear":
        assert counts == [recipe] and type(counterpart) is oracles.LinearCounterpart
        raw = source.run_source_case(
            metadata, scenario, optimization=None, _prepared=prepared
        )
        assert raw.model is None and raw.raw_callback_status == "failed"
        assert (
            raw.schema is CheckStatus.VALIDATED
            and raw.audit.status is CheckStatus.FAILED
        )
    else:
        assert counterpart is None and prepared.fn is recipe
    result = source.run_source_case(
        metadata,
        scenario,
        optimization=None,
        _prepared=prepared,
        _corpus_counterpart=counterpart,
    )
    assert (
        result.stage == "artifact-checked"
        and result.audit.status is CheckStatus.VALIDATED
    )
    assert result.schema is CheckStatus.VALIDATED and result.model is not None
    assert result.raw_callback_status == (
        "failed" if counterpart is not None else "passed"
    )
    assert result.numerical_status is CheckStatus.UNVERIFIED
    if counterpart is not None:
        assert result.corpus_structure.accepted
        arrays = [onnx.numpy_helper.to_array(t) for t in result.model.graph.initializer]
        kernel = np.asarray(prepared.fn.kernel[...])
        assert any(
            v.shape == kernel.shape and v.tobytes() == kernel.tobytes() for v in arrays
        )
        assert production._sha(kernel) == production._KERNEL_SHA
        if case[1] == 0:
            bias = np.asarray(prepared.fn.bias[...])
            assert any(
                v.size == bias.size and v.tobytes() == bias.tobytes() for v in arrays
            )
            assert production._sha(bias) == production._BIAS_SHA
        else:
            assert prepared.fn.bias is None
        assert counts == [recipe] and recipe._dtype is None and recipe._instance is None
    report = get_precision_report(result.model)
    assert report.profile == "source-static-v1"
    assert {c.category: c.status for c in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    snapshot = result.model.SerializeToString()
    samples = []
    for batch in (2, 5):
        concrete = _scenario(case, metadata, x64, batch)
        with source.source_configuration(concrete):
            specs, arguments = source._input_data(metadata, concrete, prepared.identity)
            references = tuple(
                np.asarray(v) for v in jax.tree.leaves(prepared.fn(*arguments))
            )
            # Historical-context references stay separate; no numeric predicate is adopted.
            with jax.default_matmul_precision("float32"):
                historical = tuple(
                    np.asarray(v) for v in jax.tree.leaves(prepared.fn(*arguments))
                )
        assert all(v.dtype == np.float32 and v.shape[0] == batch for v in references)
        assert [(v.dtype, v.shape) for v in historical] == [
            (v.dtype, v.shape) for v in references
        ]
        assert all(
            str(spec.shape[0]) == "B"
            for spec in specs
            if not isinstance(spec.shape[0], int)
        )
        if batch == 2:
            for actual, frozen in zip(
                references, prepared.reference_outputs, strict=True
            ):
                assert (
                    actual.tobytes() == frozen.tobytes() and not frozen.flags.writeable
                )
        samples.append(SourceSample(arguments, references))
    checked = validate_source_samples(result.model, samples=samples, optimization=None)
    assert checked.status is CheckStatus.UNVERIFIED
    assert result.model.SerializeToString() == snapshot
    for index, shape in enumerate(metadata["input_shapes"]):
        assert report.inputs[index].exported.shape == tuple(shape)
        if shape[0] != "B":
            assert (
                np.asarray(samples[0].inputs[index]).tobytes()
                != np.asarray(samples[1].inputs[index]).tobytes()
            )
            assert (
                result.model.graph.input[index].name in result.model.graph.node[0].input
            )
    assert report.exported_outputs[0].shape[0] == "B"
    onnx.save_model(result.model, tmp_path / "one-symbolic-artifact.onnx")
    assert (
        onnx.load(tmp_path / "one-symbolic-artifact.onnx").SerializeToString()
        == snapshot
    )
    if counts:
        assert counts == [recipe]


@pytest.mark.parametrize(
    "shapes", [((1, 3), (5, 3)), ((0, 3), (0, 3)), ((2, 4), (2, 3))]
)
def test_shared_batch_and_static_tail_fail_before_constructor(shapes: Any) -> None:
    calls = []

    def constructor() -> Any:
        calls.append(True)
        return lambda a, b: a + b

    metadata = {
        "callable": construct_and_call(constructor),
        "input_shapes": [("B", 3)] * 2,
    }
    with pytest.raises(source.SourcePreparationError):
        source.prepare_source_case(
            metadata, source.SourceScenario("invalid", False, concrete_shapes=shapes)
        )
    assert not calls


def test_declared_static_singleton_is_strict_before_constructor() -> None:
    calls = []

    def constructor() -> Any:
        calls.append(True)
        return lambda a, b: a + b

    metadata = {
        "callable": construct_and_call(constructor),
        "input_shapes": [("B", 3, 4), (1, 3, 4)],
    }
    with pytest.raises(source.SourcePreparationError, match="static dimension"):
        source.prepare_source_case(
            metadata,
            source.SourceScenario(
                "invalid", False, concrete_shapes=((5, 3, 4), (5, 3, 4))
            ),
        )
    assert not calls


def test_forged_reference_batch_cannot_be_encoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = {"callable": lambda a, b: a + b, "input_shapes": [("B", 3)] * 2}
    monkeypatch.setattr(
        source.jax, "device_get", lambda value: np.zeros((5, 3), np.float32)
    )
    with pytest.raises(source.SourcePreparationError, match="shared batch binding"):
        source.prepare_source_case(
            metadata,
            source.SourceScenario("forged", False, concrete_shapes=((2, 3), (2, 3))),
        )


def test_supplied_symbolic_preparation_rechecks_reference_without_criteria() -> None:
    case = production._CASES[0]
    metadata = _metadata(case)
    scenario = _scenario(case, metadata, False)
    prepared = source.prepare_source_case(metadata, scenario)
    forged = replace(prepared, reference_outputs=(np.zeros((5, 4), np.float32),))
    result = source.run_source_case(
        metadata, scenario, optimization=None, _prepared=forged
    )
    assert result.stage == "source-preparation" and result.model is None
    assert result.audit.status is CheckStatus.FAILED
    assert "shared batch binding" in result.audit.detail


@pytest.mark.parametrize(
    "defect", ["hidden-symbol", "malformed-body", "malformed-aval"]
)
def test_actual_original_trace_cannot_be_hidden_by_concrete_descriptions(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = {"callable": lambda a, b: a + b, "input_shapes": [("B", 3)] * 2}
    scenario = source.SourceScenario(
        "hidden-original-B",
        False,
        concrete_shapes=((2, 3), (2, 3)),
        require_reviewed_criteria=True,
    )
    prepared = source.prepare_source_case(metadata, scenario)
    assert prepared.criteria is None and scenario.criteria is None
    forged = replace(
        prepared,
        input_specs=tuple(
            jax.ShapeDtypeStruct((2, 3), np.float32) for _ in prepared.input_specs
        ),
        traced_inputs=tuple(
            replace(item, shape=(2, 3)) for item in prepared.traced_inputs
        ),
        traced_outputs=tuple(
            replace(item, shape=(2, 3)) for item in prepared.traced_outputs
        ),
    )
    if defect == "malformed-body":
        forged = replace(forged, original_jaxpr=SimpleNamespace())
    elif defect == "malformed-aval":
        forged = replace(
            forged,
            original_jaxpr=SimpleNamespace(
                invars=[SimpleNamespace(aval=object())],
                constvars=(),
                outvars=(),
            ),
        )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Contradictory original evidence must fail before export")

    monkeypatch.setattr(source, "_export_prepared", forbidden)
    result = source.run_source_case(
        metadata, scenario, optimization=None, _prepared=forged
    )
    assert result.stage == "source-preparation" and result.model is None
    assert result.audit.status is CheckStatus.FAILED


def _dot() -> tuple[Any, Any, Any]:
    metadata = source.raw_source_case("plugin::dot_general", 0)
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance("plugin::dot_general", 0)
    )[0]
    return metadata, scenario, source.prepare_source_case(metadata, scenario)


def test_only_explicit_counterpart_types_can_supply_structure() -> None:
    metadata, scenario, prepared = _dot()

    class Duck:
        def check(self, *args: Any) -> Any:
            pytest.fail("Duck-typed counterpart must never be invoked")

    result = source.run_source_case(
        metadata,
        scenario,
        optimization=None,
        _prepared=prepared,
        _corpus_counterpart=Duck(),
    )
    assert result.model is None and result.audit.status is CheckStatus.FAILED
    assert (
        "actual reviewed DotCounterpart, LinearCounterpart, GatherCounterpart or NormalizationCounterpart"
        in result.audit.detail
    )


def test_v2_dot_descriptors_preserve_raw_mappings_and_change_archived_fingerprints() -> (
    None
):
    expected = [
        ((3, 4), (4, 2), (3, 2), 0),
        ((3, 4), (2, 4), (3, 2), 1),
        ((3, 3), (3, 3), (3, 3), 0),
        ((3, 3), (3, 3), (3, 3), 1),
    ]
    for counterpart, mapping in zip(oracles.DOT_COUNTERPARTS, expected, strict=True):
        assert (
            counterpart.lhs_shape,
            counterpart.rhs_shape,
            counterpart.output_shape,
            counterpart.rhs_contract_axis,
        ) == mapping
        archived = oracles.fingerprint(
            {
                "identifier": f"static-dot-source-counterpart-v1:{counterpart.case.origin}:{counterpart.case.ordinal}",
                "mapping": asdict(counterpart),
                "source_sha256": "03799a9949277f9ad934abfac1b5dee7f2d3e3510600fba2d453244ac8a24790",
                "oracle_file_sha256": "8ea0557b0ded1b3317fbd23fde5dbfa9c550738e2a1f20f7b54e291d4a794be8",
            }
        )
        if counterpart.case.ordinal == 0:
            assert (
                archived
                == "6a8f289e93735fe3e8f927a002c61b66fd90646df308d9133228286f12b2c5ea"
            )
        assert counterpart.identifier.startswith("static-dot-source-counterpart-v2:")
        assert counterpart.sha256 != archived
        assert (
            oracles.counterpart_for(
                counterpart.case.origin,
                counterpart.case.ordinal,
                counterpart.case.contract_sha256,
            )
            is counterpart
        )


@pytest.mark.parametrize("defect", ["module-hash", "other-checkout"])
def test_current_counterpart_rejects_stale_or_other_checkout_source(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata, scenario, prepared = _dot()
    if defect == "module-hash":
        monkeypatch.setattr(oracles, "DOT_SOURCE_SHA256", "0" * 64)
    else:
        original_path = Path(inspect.getsourcefile(prepared.fn))
        copied = tmp_path / "another-checkout" / oracles.DOT_SOURCE_PATH
        copied.parent.mkdir(parents=True)
        copied.write_bytes(original_path.read_bytes())
        monkeypatch.setattr(oracles.inspect, "getsourcefile", lambda fn: str(copied))
    result = source.run_source_case(
        metadata,
        scenario,
        optimization=None,
        _prepared=prepared,
        _corpus_counterpart=oracles.DOT_COUNTERPARTS[0],
    )
    assert (
        result.model is None
        and result.corpus_structure.source_structure is CheckStatus.FAILED
    )
    assert "source bytes/path changed" in result.corpus_structure.source_detail
    assert result.raw_callback_status == "failed"


def test_real_v3_live_controls_reject_v1_and_stale_control_records() -> None:
    """Current v5 controls keep the historical test and all v1/v2/v3/v4 rejects."""
    registrations, _ = raw_registrations()
    blob = probe.DEFAULT_BASELINE.read_bytes()
    plan = probe.build_probe_plan(
        json.loads(blob), baseline_sha256=hashlib.sha256(blob).hexdigest()
    )
    controls = probe._corpus_product_controls(plan, registrations)
    assert (
        len(controls) == 4
        and probe.CORPUS_CONTROL_REVISION == "source-static-add-ordinary-dot-v5"
    )
    assert probe._corpus_controls_admitted(controls)
    for defect in ("identifier", "digest", "v3-digest", "v1", "v2", "v3", "v4"):
        broken = deepcopy(controls)
        if defect == "identifier":
            broken[2]["corpus_structure"][
                "counterpart_identifier"
            ] = "static-dot-source-counterpart-v1:plugin::dot_general:0"
        elif defect == "digest":
            broken[2]["corpus_structure"][
                "counterpart_sha256"
            ] = "6a8f289e93735fe3e8f927a002c61b66fd90646df308d9133228286f12b2c5ea"
        elif defect == "v3-digest":
            broken[2]["corpus_structure"][
                "counterpart_sha256"
            ] = "7a75bc04f47a300181f7df403311378dbeb3393f2f17e4b71f5c9ba376952637"
        else:
            broken[2]["control_revision"] = "source-static-add-ordinary-dot-" + defect
        assert not probe._corpus_controls_admitted(broken)
    assert (
        controls[2]["raw_callback_status"]
        == controls[3]["raw_callback_status"]
        == "failed"
    )


@pytest.mark.parametrize("defect", ["kernel", "bias", "port", "cast", "missing"])
def test_native_linear_counterpart_rejects_parameter_and_wiring_corruption(
    defect: str,
) -> None:
    case = production._CASES[3]
    metadata = _metadata(case)
    scenario = _scenario(case, metadata, False)
    prepared = source.prepare_source_case(metadata, scenario)
    counterpart = oracles.LINEAR_COUNTERPARTS[0]
    result = source.run_source_case(
        metadata,
        scenario,
        optimization=None,
        _prepared=prepared,
        _corpus_counterpart=counterpart,
    )
    assert result.model is not None
    model = deepcopy(result.model)
    if defect in {"kernel", "bias"}:
        expected_shape = (128, 64) if defect == "kernel" else (64,)
        tensor = next(
            t for t in model.graph.initializer if tuple(t.dims) == expected_shape
        )
        values = onnx.numpy_helper.to_array(tensor).copy()
        values.flat[0] += np.float32(0.25)
        tensor.CopyFrom(onnx.numpy_helper.from_array(values, name=tensor.name))
    elif defect == "port":
        model.graph.node[0].input[0] = model.graph.node[0].input[1]
    elif defect == "cast":
        model.graph.node[0].op_type = "Cast"
    else:
        del model.graph.node[1]
    evidence = counterpart.check(prepared, model)
    assert evidence.source_structure is CheckStatus.FAILED
