# tests/extra_tests/converter/source_precision_corpus_numerics.py

"""Fixed FP32 numerical counterparts; historical source criteria stay unset.

This tooling neither admits lowerings nor changes public comparison semantics.
One captured target output must satisfy all three approved comparisons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import platform
from typing import Any, Final

import jax
import numpy as np
import onnxruntime as ort

from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    _array_evidence,
    _compare,
    get_precision_report,
)
from scripts import source_precision_probe as artifact_probe
from scripts.source_precision_inventory import canonical_contract
from tests.extra_tests.converter.source_precision_corpus_oracles import (
    CORPUS_REPLAY_CASES,
    CorpusCase,
    counterpart_for,
    fingerprint,
)
from tests.extra_tests.converter.source_precision_diagnostics import DiagnosticValidator
from tests.extra_tests.converter.source_precision_support import (
    PreparedSourceCase,
    SourceProvenance,
    _properties,
    prepare_source_case,
    raw_source_scenarios,
    run_source_case,
    value_evidence,
)


CRITERION_VERSION: Final = "source-corpus-fp32-conjunction-v1"
APPROVAL_SHA256: Final = (
    "40188c1f0aa34edb3324c8b28494d07226bf59f98898bbb3eeee382323a92ee0"
)
COMMITTED_SOURCE_LINEAGE: Final = "fc3f932315a39e9e631e7fd0788a220e48440c77"
# The approved prose called this mode 'allclose'. The unchanged public spelling
# is 'tolerant'; it implements precisely the approved actual-first np.allclose.
CRITERIA: Final = SampleCriteria("tolerant", rtol=1e-5, atol=1e-5, equal_nan=True)
TARGET_MODES: Final = ("disabled", "all")
_OUTPUT_SHAPES: Final = {
    ("plugin::dot_general", 0): (3, 2),
    ("plugin::dot_general", 1): (3, 2),
    ("plugin::dot_general", 2): (3, 3),
    ("plugin::dot_general", 3): (3, 3),
    ("plugin::dot_general", 6): (2, 2, 3, 5),
    ("plugin::jax.numpy.add", 0): (3,),
    ("plugin::jax.numpy.add", 1): (3,),
    ("plugin::jax.numpy.add", 2): (2, 3),
    ("plugin::jax.numpy.add", 3): (3, 4),
    ("plugin::jax.numpy.matmul", 2): (3, 5),
    ("plugin::jax.numpy.matmul", 4): (2, 3, 5),
    ("plugin::jax.numpy.matmul", 8): (3, 2, 5),
}
NUMERICAL_CASES: Final = tuple(
    case
    for case in CORPUS_REPLAY_CASES
    if (case.origin, case.ordinal) in _OUTPUT_SHAPES
)


class NumericalPreflightError(ValueError):
    """No target execution is justified by the supplied frozen evidence."""


def json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def criterion_descriptor() -> dict[str, Any]:
    return {
        "version": CRITERION_VERSION,
        "public_mode": "tolerant",
        "criteria": asdict(CRITERIA),
        "predicates": [
            "allclose(actual, raw_reference)",
            "allclose(raw_reference, actual)",
            "allclose(historical_context_reference, actual)",
        ],
        "exact_output_count_dtype_shape": True,
        "dtype": "float32",
        "cases": [asdict(case) for case in NUMERICAL_CASES],
        "historical_criteria_unchanged": None,
    }


def selected_case(origin: str, ordinal: int, contract_sha256: str) -> CorpusCase:
    matches = [
        case
        for case in NUMERICAL_CASES
        if (case.origin, case.ordinal, case.contract_sha256)
        == (origin, ordinal, contract_sha256)
    ]
    if len(matches) != 1:
        raise NumericalPreflightError(
            "Not one of the twelve approved origin/ordinal/contracts"
        )
    return matches[0]


def strong_fp32_trace(trace: Any) -> None:
    if isinstance(trace, dict):
        if trace.get("effects") or trace.get("unresolved_parameters"):
            raise NumericalPreflightError("Effects or unresolved trace lineage")
        for value in trace.values():
            strong_fp32_trace(value)
    elif isinstance(trace, (tuple, list)):
        if (
            len(trace) == 3
            and isinstance(trace[0], str)
            and isinstance(trace[1], list)
            and isinstance(trace[2], bool)
        ):
            if trace[0] != "float32" or trace[2]:
                raise NumericalPreflightError("Every traced tensor must be strong FP32")
        for value in trace:
            strong_fp32_trace(value)


def reference_context_changes(
    raw: Any, harness: Any, path: str = "trace"
) -> list[dict[str, Any]]:
    """Allow only None→HIGHEST dot precision from the declared float32 context."""
    if raw == harness:
        return []
    if (
        isinstance(raw, dict)
        and isinstance(harness, dict)
        and raw.keys() == harness.keys()
    ):
        changes = []
        if raw.get("primitive") == harness.get("primitive") == "dot_general":
            before, after = raw["params"].get("precision"), harness["params"].get(
                "precision"
            )
            if before != after:
                expected = [
                    canonical_contract(jax.lax.Precision.HIGHEST, unresolved=[])
                ] * 2
                if before is not None or after != expected:
                    raise NumericalPreflightError(
                        "Unexpected historical-context dot precision"
                    )
                changes.append(
                    dict(path=path + ".params.precision", before=before, after=after)
                )
                harness = {
                    **harness,
                    "params": {**harness["params"], "precision": before},
                }
        for key in raw:
            changes.extend(
                reference_context_changes(raw[key], harness[key], path + "." + key)
            )
        return changes
    if isinstance(raw, list) and isinstance(harness, list) and len(raw) == len(harness):
        return [
            change
            for index, (a, b) in enumerate(zip(raw, harness, strict=True))
            for change in reference_context_changes(a, b, f"{path}[{index}]")
        ]
    raise NumericalPreflightError(
        f"Unexpected historical-context source/typing/parameter change at {path}"
    )


def prepare_harness_reference(
    raw: PreparedSourceCase,
) -> tuple[PreparedSourceCase, list[dict[str, Any]]]:
    feed = tuple(np.array(value, copy=True) for value in raw.arguments)
    if any(value.dtype != np.dtype(np.float32) for value in feed):
        raise NumericalPreflightError(
            "Historical reference requires the identical FP32 feed"
        )
    scenario = replace(
        raw.scenario,
        name="historical-context-reference",
        caller_x64_enabled=False,
        default_matmul_precision="float32",
        input_values=feed,
    )
    harness = prepare_source_case(raw.metadata, scenario)
    if (
        harness.input_evidence != raw.input_evidence
        or harness.traced_outputs != raw.traced_outputs
    ):
        raise NumericalPreflightError(
            "Historical context changed feed/output signatures"
        )
    original, control = json.loads(raw.original_trace_json), json.loads(
        harness.original_trace_json
    )
    strong_fp32_trace(original)
    strong_fp32_trace(control)
    changes = reference_context_changes(original, control)
    if (
        harness.settings.x64_enabled
        or harness.settings.default_matmul_precision != "float32"
    ):
        raise NumericalPreflightError(
            "Historical native reference used another context"
        )
    return harness, changes


def _output_contract(groups: Any, shapes: tuple[tuple[int, ...], ...]) -> bool:
    return not (
        any(len(values) != len(shapes) for values in groups)
        or any(
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype(np.float32)
            or value.shape != shape
            for values in groups
            for value, shape in zip(values, shapes)
        )
    )


def compare_outputs(
    actual: Any, raw: Any, harness: Any, shapes: tuple[tuple[int, ...], ...]
) -> dict[str, Any]:
    checks: dict[str, Any] = dict(
        output_contract=False,
        source_actual_first=None,
        raw_reference_first=None,
        historical_reference_first=None,
    )
    if not _output_contract((actual, raw, harness), shapes):
        return {
            **checks,
            "status": "failed",
            "detail": "Output count/dtype/shape differs before comparison",
        }
    checks["output_contract"] = True
    checks["source_actual_first"] = all(
        _compare(a, r, CRITERIA) is None for a, r in zip(actual, raw, strict=True)
    )
    checks["raw_reference_first"] = all(
        bool(np.allclose(r, a, rtol=1e-5, atol=1e-5, equal_nan=True))
        for a, r in zip(actual, raw, strict=True)
    )
    checks["historical_reference_first"] = all(
        bool(np.allclose(h, a, rtol=1e-5, atol=1e-5, equal_nan=True))
        for a, h in zip(actual, harness, strict=True)
    )
    return {
        **checks,
        "status": "validated" if all(checks.values()) else "failed",
        "detail": "All three fixed predicates consume the same captured output",
    }


def write_new(path: Path, blob: bytes) -> dict[str, Any]:
    with path.open("xb") as stream:
        stream.write(blob)
    path.chmod(0o444)
    return {"path": str(path.resolve()), "sha256": sha256(blob)}


def retain_arrays(directory: Path, role: str, values: Any) -> list[dict[str, Any]]:
    retained = []
    for index, value in enumerate(values):
        array = np.array(value, copy=True)
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        retained.append(
            {
                **write_new(directory / f"{role}-{index}.npy", buffer.getvalue()),
                "array": asdict(_array_evidence(f"{role}-{index}", array)),
            }
        )
    return retained


def load_arrays(records: Any) -> tuple[np.ndarray, ...]:
    arrays = []
    for record in records:
        blob = Path(record["path"]).read_bytes()
        if sha256(blob) != record["sha256"]:
            raise NumericalPreflightError("Frozen array payload changed")
        array = np.load(io.BytesIO(blob), allow_pickle=False)
        if json_value(
            asdict(_array_evidence(record["array"]["name"], array))
        ) != json_value(record["array"]):
            raise NumericalPreflightError("Frozen array evidence changed")
        array.setflags(write=False)
        arrays.append(array)
    return tuple(arrays)


def _scenario_fingerprint(row: Any, declared: Any) -> tuple[str, Any]:
    supplied = tuple(
        value_evidence(
            v,
            _properties(
                v if hasattr(v, "dtype") and hasattr(v, "shape") else np.asarray(v)
            ),
        )
        for v in declared.input_values or ()
    )
    return (
        fingerprint(
            dict(
                name=declared.name,
                x64=declared.caller_x64_enabled,
                matmul=declared.default_matmul_precision,
                factory="preserve",
                specs=[asdict(_properties(s)) for s in declared.input_specs or ()],
                supplied=[asdict(v) for v in supplied],
                generation_provenance=[row.origin, row.ordinal],
                criteria=None,
            )
        ),
        supplied,
    )


def prepare_numerical_cell(
    row: Any, metadata: Any, accepted: Any, directory: Path
) -> dict[str, Any]:
    case = selected_case(row.origin, row.ordinal, row.contract_sha256)
    identity = (row.origin, row.ordinal, accepted["scenario"])
    if (
        accepted["origin"] != case.origin
        or accepted["ordinal"] != case.ordinal
        or accepted["scenario"] not in artifact_probe.SCENARIOS
        or row.criteria is not None
    ):
        raise NumericalPreflightError("Original cell/criteria identity changed")
    old = accepted["history"][-1]
    declared = raw_source_scenarios(
        metadata, SourceProvenance(case.origin, case.ordinal)
    )[artifact_probe.SCENARIOS.index(accepted["scenario"])]
    scenario_sha, supplied = _scenario_fingerprint(row, declared)
    if (
        scenario_sha != old["scenario_sha256"]
        or row.contract_sha256 != old["row_contract_sha256"]
    ):
        raise NumericalPreflightError("Original scenario or contract changed")
    raw = prepare_source_case(metadata, declared)
    current = dict(
        row_contract_sha256=row.contract_sha256,
        scenario_sha256=scenario_sha,
        environment=asdict(artifact_probe.ProbeEnvironment.observe()),
        requested_x64=declared.caller_x64_enabled,
        requested_matmul_precision=declared.default_matmul_precision,
        actual_settings=asdict(raw.settings),
        original_trace_json=raw.original_trace_json,
        original_trace_sha256=sha256(raw.original_trace_json.encode()),
        supplied_inputs=[asdict(v) for v in supplied],
        inputs=[asdict(v) for v in raw.input_evidence],
        references=[asdict(v) for v in raw.reference_evidence],
    )
    if any(json_value(current[key]) != old[key] for key in current):
        raise NumericalPreflightError(
            "Original source/feed/reference/settings/environment drift"
        )
    if json_value([asdict(v) for v in raw.traced_outputs]) != [
        dict(
            name=None,
            dtype="float32",
            shape=list(_OUTPUT_SHAPES[(case.origin, case.ordinal)]),
            weak_type=False,
        )
    ]:
        raise NumericalPreflightError(
            "Output is not the exact approved strong FP32 shape"
        )
    harness, changes = prepare_harness_reference(raw)
    observer = DiagnosticValidator()
    checked = run_source_case(
        metadata,
        declared,
        optimization=None,
        diagnostic_jaxpr_validator=observer,
        _prepared=raw,
        _corpus_counterpart=counterpart_for(
            case.origin, case.ordinal, case.contract_sha256
        ),
    )
    if (
        checked.model is None
        or checked.schema is not CheckStatus.VALIDATED
        or checked.audit.status is not CheckStatus.UNVERIFIED
    ):
        raise NumericalPreflightError(
            f"Diagnostic artifact remains pending/failed: {checked.stage}: {checked.audit.detail}"
        )
    blob = checked.model.SerializeToString()
    export = dict(
        artifact_sha256=sha256(blob),
        export_settings=asdict(checked.export_settings),
        export_report=asdict(get_precision_report(checked.model)),
        corpus_structure=(
            asdict(checked.corpus_structure) if checked.corpus_structure else None
        ),
        raw_callback_status=checked.raw_callback_status,
    )
    if (
        any(json_value(value) != old[key] for key, value in export.items())
        or export["artifact_sha256"] != accepted["artifact_sha256"]
    ):
        raise NumericalPreflightError(
            "Accepted artifact/structure/export evidence changed"
        )
    directory.mkdir()
    return json_value(
        dict(
            identity=list(identity),
            case=asdict(case),
            criterion_version=CRITERION_VERSION,
            accepted_cell_sha256=fingerprint(accepted),
            previous_evidence_sha256=fingerprint(old),
            raw=current,
            export=export,
            model=write_new(directory / "artifact.onnx", blob),
            feed=retain_arrays(directory, "feed", raw.arguments),
            raw_reference=retain_arrays(directory, "raw", raw.reference_outputs),
            historical_reference=retain_arrays(
                directory, "historical", harness.reference_outputs
            ),
            historical_context=dict(
                name=harness.scenario.name,
                settings=asdict(harness.settings),
                trace=json.loads(harness.original_trace_json),
                trace_sha256=sha256(harness.original_trace_json.encode()),
                input_evidence=[asdict(v) for v in harness.input_evidence],
                reference_evidence=[asdict(v) for v in harness.reference_evidence],
                permitted_precision_changes=changes,
                reference_bytes_equal=[
                    np.asarray(a).tobytes() == np.asarray(b).tobytes()
                    for a, b in zip(
                        raw.reference_outputs, harness.reference_outputs, strict=True
                    )
                ],
            ),
            input_names=[v.name for v in checked.model.graph.input],
            input_shapes=[list(v.shape) for v in raw.traced_inputs],
            output_names=[v.name for v in checked.model.graph.output],
            output_shapes=[list(v.shape) for v in raw.traced_outputs],
            payload_directory=str(directory.resolve()),
            lowering_audit="unverified",
            state="prepared",
        )
    )


@dataclass(frozen=True)
class FrozenPayloads:
    feed: tuple[np.ndarray, ...]
    raw: tuple[np.ndarray, ...]
    historical: tuple[np.ndarray, ...]


def load_frozen_payloads(cell: Any) -> FrozenPayloads:
    return FrozenPayloads(
        load_arrays(cell["feed"]),
        load_arrays(cell["raw_reference"]),
        load_arrays(cell["historical_reference"]),
    )


def execute_frozen_cell(
    cell: Any,
    mode: str,
    preparation: Path,
    preparation_sha256: str,
    *,
    payloads: FrozenPayloads,
) -> dict[str, Any]:
    """One direct real session/run; a reserved target directory forbids retries."""
    selected_case(
        **{k: cell["case"][k] for k in ("origin", "ordinal", "contract_sha256")}
    )
    if (
        cell["identity"][:2] != [cell["case"]["origin"], cell["case"]["ordinal"]]
        or cell["identity"][2] not in artifact_probe.SCENARIOS
        or mode not in TARGET_MODES
        or cell["criterion_version"] != CRITERION_VERSION
        or cell["output_shapes"]
        != [list(_OUTPUT_SHAPES[(cell["case"]["origin"], cell["case"]["ordinal"])])]
    ):
        raise NumericalPreflightError("Frozen numerical identity/target changed")
    frozen_blob = preparation.read_bytes()
    if sha256(frozen_blob) != preparation_sha256:
        raise NumericalPreflightError("Immutable preparation bytes changed")
    frozen = json.loads(frozen_blob)
    if (
        frozen["approval_sha256"] != APPROVAL_SHA256
        or frozen["criterion"] != criterion_descriptor()
        or cell not in frozen["cells"]
    ):
        raise NumericalPreflightError("Immutable preparation/criterion changed")
    blob = Path(cell["model"]["path"]).read_bytes()
    if (
        sha256(blob) != cell["model"]["sha256"]
        or sha256(blob) != cell["export"]["artifact_sha256"]
    ):
        raise NumericalPreflightError("Frozen executable artifact changed")
    feed, raw, harness = payloads.feed, payloads.raw, payloads.historical
    for role, values in (
        ("feed", feed),
        ("raw_reference", raw),
        ("historical_reference", harness),
    ):
        frozen_arrays = load_arrays(cell[role])
        if len(values) != len(frozen_arrays) or any(
            a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes()
            for a, b in zip(values, frozen_arrays, strict=True)
        ):
            raise NumericalPreflightError(
                "In-memory payload differs from frozen source arrays"
            )
    shapes = tuple(tuple(v) for v in cell["output_shapes"])
    if (
        not _output_contract((raw, harness), shapes)
        or len(feed) != len(cell["input_names"])
        or any(
            a.dtype != np.dtype(np.float32) or a.shape != tuple(s)
            for a, s in zip(feed, cell["input_shapes"], strict=True)
        )
    ):
        raise NumericalPreflightError("Missing/mismatched frozen feed or reference")
    target_dir = Path(cell["payload_directory"]) / mode
    target_dir.mkdir()
    level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if mode == "disabled"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = level
    result = dict(
        identity=cell["identity"],
        criterion_version=CRITERION_VERSION,
        mode=mode,
        artifact_sha256=sha256(blob),
        preparation_sha256=preparation_sha256,
        runtime="failed",
        numerics="unverified",
        lowering_audit="unverified",
        stage="session-load",
        run_attempts=0,
        successful_runs=0,
        target=dict(
            ort_version=ort.__version__,
            architecture=platform.machine(),
            requested_provider="CPUExecutionProvider",
            optimization=level.name,
        ),
        outputs=[],
        comparison=None,
    )
    try:
        session = ort.InferenceSession(
            blob, options, providers=["CPUExecutionProvider"]
        )
        session.disable_fallback()
        actual = session.get_session_options()
        result["target"].update(
            providers=session.get_providers(),
            provider_options=session.get_provider_options(),
            intra_op_threads=actual.intra_op_num_threads,
            inter_op_threads=actual.inter_op_num_threads,
            enable_mem_pattern=actual.enable_mem_pattern,
            execution_mode=actual.execution_mode.name,
            optimization=actual.graph_optimization_level.name,
            fallback_disabled=True,
        )
        result["stage"] = "target-configuration"
        if session.get_providers() != ["CPUExecutionProvider"] or (
            actual.intra_op_num_threads,
            actual.inter_op_num_threads,
            actual.enable_mem_pattern,
            actual.execution_mode,
            actual.graph_optimization_level,
        ) != (1, 1, False, ort.ExecutionMode.ORT_SEQUENTIAL, level):
            raise NumericalPreflightError("Actual CPU target options differ")
        result["stage"] = "feed-schema"
        for ports, names, dimensions in (
            (session.get_inputs(), cell["input_names"], cell["input_shapes"]),
            (session.get_outputs(), cell["output_names"], cell["output_shapes"]),
        ):
            if [(v.name, v.type, v.shape) for v in ports] != [
                (n, "tensor(float)", s) for n, s in zip(names, dimensions, strict=True)
            ]:
                raise NumericalPreflightError(
                    "Actual target schema differs from exact frozen FP32 boundaries"
                )
        result["stage"] = "execution"
        result["run_attempts"] = 1
        captured = session.run(
            cell["output_names"], dict(zip(cell["input_names"], feed, strict=True))
        )
        outputs = tuple(np.array(v, copy=True) for v in captured)
        result["successful_runs"] = 1
        result["runtime"] = "validated"
        result["outputs"] = retain_arrays(target_dir, "actual", outputs)
        result["stage"] = "comparison"
        result["comparison"] = compare_outputs(outputs, raw, harness, shapes)
        result["numerics"] = result["comparison"]["status"]
    except Exception as exc:
        result["detail"] = f"{type(exc).__name__}: {exc}"
        if result["runtime"] == "validated":
            result["numerics"] = "failed"
    try:
        result["artifact_bytes_unchanged"] = (
            Path(cell["model"]["path"]).read_bytes() == blob
        )
        result["preparation_bytes_unchanged"] = preparation.read_bytes() == frozen_blob
        result["feed_bytes_unchanged"] = all(
            a.tobytes() == b.tobytes()
            for a, b in zip(feed, load_arrays(cell["feed"]), strict=True)
        )
        result["reference_bytes_unchanged"] = all(
            a.tobytes() == b.tobytes()
            for role, values in (
                ("raw_reference", raw),
                ("historical_reference", harness),
            )
            for a, b in zip(values, load_arrays(cell[role]), strict=True)
        )
    except Exception as exc:
        result.update(
            artifact_bytes_unchanged=False,
            preparation_bytes_unchanged=False,
            feed_bytes_unchanged=False,
            reference_bytes_unchanged=False,
            detail=f"Frozen payload verification failed: {type(exc).__name__}: {exc}",
        )
    if not all(
        result[k]
        for k in (
            "artifact_bytes_unchanged",
            "preparation_bytes_unchanged",
            "feed_bytes_unchanged",
            "reference_bytes_unchanged",
        )
    ):
        result["runtime"] = "failed"
        result["numerics"] = "failed"
    write_new(
        target_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return result
