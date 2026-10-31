# tests/extra_tests/framework/test_source_cast_failure_replay.py

"""Qualify failure replay provenance separately from original numerical gates."""

import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/replay_source_cast_failure.py"
_LOW = "003c003c003c00bcff7b007c"
_HIGH = "003c003c013c00bcff7b007c"


def _module() -> Any:
    spec = spec_from_file_location("_cast_replay_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value))


def _array(storage: str, dtype: str) -> dict[str, Any]:
    raw = bytes.fromhex(storage)
    size = 8 if dtype == "float64" else 2
    return {
        "identity": 123,
        "dtype": dtype,
        "dtype_str": "<f8" if size == 8 else "<f2",
        "shape": [6],
        "storage_hex": storage,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "element_storage_hex": [
            raw[i : i + size].hex() for i in range(0, len(raw), size)
        ],
    }


def _fixture(
    tmp_path: Path, *, model: bytes = b"not an ONNX model", mixed: bool = False
) -> tuple[Any, Path, Path]:
    module = _module()
    receipts = tmp_path / "receipts"
    folder = receipts / "batch-0001"
    folder.mkdir(parents=True)
    phases = [
        {"when": "setup", "outcome": "passed"},
        {
            "when": "call",
            "outcome": "failed",
            "longrepr": "SampleAcceptanceError: values fail bitwise comparison",
        },
        {"when": "teardown", "outcome": "passed"},
    ]
    _write(
        folder / "receipt.json",
        {
            "kind": "batch",
            "exitstatus": 1,
            "full_collection_checked": True,
            "collection_errors": [],
            "assigned_ids": [module.NODE_ID],
            "selected_ids": [module.NODE_ID],
            "phases": {module.NODE_ID: phases},
        },
    )
    (folder / "events.jsonl").write_text(
        "\n".join(
            json.dumps({"kind": "phase", "nodeid": module.NODE_ID, **phase})
            for phase in phases
        )
        + "\n"
    )
    tree = ET.Element("testsuites")
    suite = ET.SubElement(tree, "testsuite")
    case = ET.SubElement(
        suite,
        "testcase",
        classname="tests.extra_tests.converter.test_source_conversion_semantics",
        name=module.NODE_ID.split("::")[1],
    )
    ET.SubElement(case, "failure").text = phases[1]["longrepr"]
    (folder / "junit.xml").write_bytes(ET.tostring(tree))
    (folder / module.MODEL).write_bytes(model)
    sources = {}
    for relative in module._SOURCE_FILES:
        path = _ROOT / relative
        payload = path.read_bytes()
        sources[relative] = {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    eager = _array(_LOW if mixed else _HIGH, "float16")
    fresh_jit = _array(_HIGH, "float16")
    runtimes, comparisons = [], []
    for index, reference in enumerate((eager, fresh_jit) if mixed else (eager,)):
        labels = {
            **module._LABELS,
            "optimization": "disabled",
            "reference_index": index,
        }
        actual = _array(_LOW, "float16")
        runtimes.append(
            {
                "labels": labels,
                "inputs": {"in_0": _array(module._INPUT_HEX, "float64")},
                "outputs": [actual],
                "providers": ["CPUExecutionProvider"],
            }
        )
        different = [2] if reference["storage_hex"] != _LOW else []
        comparisons.append(
            {
                "labels": labels,
                "criteria": module._CRITERIA,
                "actual": actual,
                "reference": reference,
                "different_indices": different,
                "original_result": (
                    "values fail bitwise comparison" if different else None
                ),
            }
        )
    _write(
        folder / module.OBSERVATION,
        {
            "nodeid": module.NODE_ID,
            "status": "ORIGINAL_CALL_WORDS_OBSERVED",
            "parameter_order": module._LABELS,
            "observer_errors": [],
            "byteorder": "little",
            "jax_config_before": {},
            "jax_config_end": {},
            "function_code_provenance": {
                role: {
                    "name": name,
                    "filename": str(_ROOT / relative),
                    "marshal_sha256": "0" * 64,
                    "firstlineno": 1,
                    "function_identity": 1,
                    "code_identity": 2,
                }
                for role, name, relative in (
                    (
                        "artifact",
                        "_artifact",
                        "tests/extra_tests/converter/test_source_pointwise_mul_exp.py",
                    ),
                    ("runtime", "_run_cpu_samples", "jax2onnx/precision.py"),
                    ("compare", "_compare", "jax2onnx/precision.py"),
                )
            },
            **{name: True for name in module._FENCES},
            "native_references_recomputed_by_observer": False,
            "inputs_criteria_or_assertions_changed": False,
            "numerical_acceptance_claimed_by_observer": False,
            "source_files_before": sources,
            "source_files_end": sources,
            "artifact_events": [
                {
                    "labels": module._LABELS,
                    "sha256": hashlib.sha256(model).hexdigest(),
                    "native_eager": [eager],
                    "native_fresh_jit": [fresh_jit],
                    "graph_ops": [
                        {
                            "op_type": "Cast",
                            "input": ["in_0"],
                            "output": ["out"],
                            "attributes": [{"name": "to", "type": 2, "int": 10}],
                        }
                    ],
                }
            ],
            "runtime_events": runtimes,
            "comparison_events": comparisons,
            "event_commit": "original-merge-commit",
        },
    )
    return module, receipts, folder


def _without_framework_imports(
    receipts: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    program = """import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'jax', 'jaxlib', 'jax2onnx', 'numpy', 'onnx', 'onnxruntime', 'flax', 'scipy'}:
        raise AssertionError('framework import before preflight: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], '--receipts-dir', sys.argv[2], '--output', sys.argv[3]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    return subprocess.run(
        [sys.executable, "-B", "-c", program, str(_SCRIPT), str(receipts), str(output)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_unrelated_missing_witness_imports_no_framework(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    completed = _without_framework_imports(tmp_path / "missing", output)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["status"] == "NOT_APPLICABLE" and result["replays"] == []


@pytest.mark.parametrize(
    "forgery",
    (
        "fence",
        "raw-sha",
        "element-words",
        "six-inputs",
        "criterion",
        "phase-events",
        "junit",
        "duplicate-witness",
    ),
)
def test_forged_witness_fails_before_any_framework_import(
    forgery: str, tmp_path: Path
) -> None:
    module, receipts, folder = _fixture(tmp_path)
    path = folder / module.OBSERVATION
    observed = json.loads(path.read_text())
    if forgery == "fence":
        observed["code_identity_fence_held"] = False
    elif forgery == "raw-sha":
        record = observed["artifact_events"][0]["native_eager"][0]
        record["sha256"] = module._record(record, "float16")["typed_sha256"]
    elif forgery == "element-words":
        observed["runtime_events"][0]["outputs"][0]["element_storage_hex"][2] = "013c"
    elif forgery == "six-inputs":
        observed["runtime_events"][0]["inputs"]["in_0"] = _array("00" * 48, "float64")
    elif forgery == "criterion":
        observed["comparison_events"][0]["criteria"]["mode"] = "exact"
    elif forgery == "phase-events":
        (folder / "events.jsonl").write_text("")
    elif forgery == "junit":
        (folder / "junit.xml").write_text("<testsuites/>")
    else:
        duplicate = receipts / "batch-0002"
        duplicate.mkdir()
        (duplicate / module.OBSERVATION).write_text(path.read_text())
    _write(path, observed)
    output = tmp_path / "result.json"
    completed = _without_framework_imports(receipts, output)
    result = json.loads(output.read_text())
    assert completed.returncode == 1
    assert result["status"] == "UNVERIFIED" and result["replays"] == []
    assert "framework import" not in result["message"]


def test_historical_observer_identity_stays_separate_from_current_replay(
    tmp_path: Path,
) -> None:
    module, receipts, folder = _fixture(tmp_path)
    path = folder / module.OBSERVATION
    observed = json.loads(path.read_text())
    for key in ("source_files_before", "source_files_end"):
        observed[key]["scripts/_source_cast_observer.py"]["sha256"] = "0" * 64
    _write(path, observed)
    witness = module._preflight(receipts)
    assert (
        witness["replay_source_matches_original"]["scripts/_source_cast_observer.py"]
        is False
    )
    assert witness["replay_source_matches_original"]["jax2onnx/precision.py"] is True
    assert (
        witness["original_source_files"]["scripts/_source_cast_observer.py"]["sha256"]
        == "0" * 64
    )


def _cast_model() -> bytes:
    import onnx

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Cast", ["in_0"], ["out"], to=onnx.TensorProto.FLOAT16)],
        "replay",
        [onnx.helper.make_tensor_value_info("in_0", onnx.TensorProto.DOUBLE, [6])],
        [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT16, [6])],
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 23)], ir_version=10
    ).SerializeToString()


def test_real_reports_record_both_acceptance_results_without_recomputing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jax
    import jax2onnx.precision  # noqa: F401 - import before forbidding native calls

    def source_forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("replay recomputed a native source reference")

    monkeypatch.setattr(jax, "jit", source_forbidden)
    monkeypatch.setattr(jax.lax, "convert_element_type", source_forbidden)
    module, receipts, _ = _fixture(tmp_path, model=_cast_model(), mixed=True)
    output = tmp_path / "result.json"
    assert module.main(["--receipts-dir", str(receipts), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "REPLAY_OBSERVED"
    assert (
        result["source_callable_called"]
        is result["native_references_recomputed"]
        is False
    )
    assert result["raw_target_outputs_captured"] is False
    assert len(result["replays"]) == 4
    for index, replay in enumerate(result["replays"]):
        assert replay["state"] == "COMPLETE"
        assert replay["status"] == ("validated" if index % 2 == 0 else "failed")
        assert replay["acceptance"]["returned"] is (index % 2 == 0)
        if index % 2:
            assert (
                replay["acceptance"]["exception_type"]
                == "jax2onnx.precision.SampleAcceptanceError"
            )
            assert "bitwise comparison" in replay["acceptance"]["message"]
        sample = replay["samples"][0]
        assert sample["inputs"][0]["sha256"] == result["frozen_inputs"]["typed_sha256"]
        assert (
            sample["inputs"][0]["sha256"]
            != result["frozen_inputs"]["raw_storage_sha256"]
        )
        assert (
            sample["references"][0]["sha256"]
            == result["frozen_references"][replay["reference"]]["typed_sha256"]
        )
        assert any(
            check["category"] == "runtime-execution" and check["status"] == "validated"
            for check in sample["checks"]
        )
    assert (
        result["evidence_fence_held"]
        is result["replay_source_fence_held"]
        is result["artifact_fence_held"]
        is True
    )


def test_partial_runtime_exception_keeps_checkpoint_and_other_actual_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jax2onnx.precision as precision

    module, receipts, _ = _fixture(tmp_path, model=_cast_model())
    output = tmp_path / "result.json"
    original = precision.validate_source_samples
    calls = []

    def interrupted_second(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["optimization"])
        if len(calls) == 2:
            partial = json.loads(output.read_text())
            assert partial["replays"][0]["state"] == "COMPLETE"
            assert partial["replays"][1]["state"] == "STARTED"
            raise RuntimeError("diagnostic second-call interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(precision, "validate_source_samples", interrupted_second)
    assert module.main(["--receipts-dir", str(receipts), "--output", str(output)]) == 1
    result = json.loads(output.read_text())
    assert calls == ["disabled", "disabled", "all", "all"]
    assert result["status"] == "UNVERIFIED"
    assert [entry["state"] for entry in result["replays"]] == [
        "COMPLETE",
        "UNVERIFIED",
        "COMPLETE",
        "COMPLETE",
    ]
    assert result["replays"][1]["message"] == "diagnostic second-call interruption"
    assert result["evidence_fence_held"] is result["artifact_fence_held"] is True


def test_graph_gate_rejects_forged_model_before_precision_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    module, receipts, _ = _fixture(tmp_path)
    original_import = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name != "jax2onnx.precision", "precision imported before graph gate"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    output = tmp_path / "result.json"
    assert module.main(["--receipts-dir", str(receipts), "--output", str(output)]) == 1
    result = json.loads(output.read_text())
    assert result["status"] == "UNVERIFIED" and result["replays"] == []
    assert "precision imported" not in result["message"]


@pytest.mark.parametrize(
    "fault",
    (
        "unexpected-runtime",
        "unexpected-attribute",
        "validated-rejected",
        "failed-accepted",
        "unverified-accepted",
    ),
)
def test_acceptance_exceptions_and_status_contradictions_remain_unverified(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import jax2onnx.precision as precision

    module, receipts, _ = _fixture(tmp_path, model=_cast_model(), mixed=True)
    output = tmp_path / "result.json"
    original_validate = precision.validate_source_samples
    original_acceptance = precision.SourceValidationReport.require_sample_acceptance
    calls = []

    def validate(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["optimization"])
        report = original_validate(*args, **kwargs)
        if fault == "unverified-accepted" and len(calls) == 1:
            report = replace(
                report,
                checks=(
                    *report.checks,
                    precision.PrecisionCheck(
                        "diagnostic-control",
                        precision.CheckStatus.UNVERIFIED,
                        "explicit unverified control",
                    ),
                ),
            )
        return report

    def acceptance(report: Any) -> None:
        if fault == "failed-accepted" and len(calls) == 2:
            return
        if len(calls) == 1:
            if fault == "unexpected-runtime":
                raise RuntimeError("unexpected acceptance runtime exception")
            if fault == "unexpected-attribute":
                raise AttributeError("unexpected acceptance attribute exception")
            if fault == "validated-rejected":
                raise precision.SampleAcceptanceError(
                    "contradictory validated rejection"
                )
            if fault == "unverified-accepted":
                return
        original_acceptance(report)

    monkeypatch.setattr(precision, "validate_source_samples", validate)
    monkeypatch.setattr(
        precision.SourceValidationReport, "require_sample_acceptance", acceptance
    )
    assert module.main(["--receipts-dir", str(receipts), "--output", str(output)]) == 1
    result = json.loads(output.read_text())
    assert calls == ["disabled", "disabled", "all", "all"]
    assert result["status"] == "UNVERIFIED"
    failed = 1 if fault == "failed-accepted" else 0
    assert [
        index
        for index, entry in enumerate(result["replays"])
        if entry["state"] == "UNVERIFIED"
    ] == [failed]
    assert all(
        entry["state"] == "COMPLETE"
        for index, entry in enumerate(result["replays"])
        if index != failed
    )
    assert "acceptance" in result["replays"][failed]


def test_expected_unverified_acceptance_error_is_an_observed_negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import jax2onnx.precision as precision

    module, receipts, _ = _fixture(tmp_path, model=_cast_model(), mixed=True)
    output = tmp_path / "result.json"
    original = precision.validate_source_samples
    calls = []

    def validate(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["optimization"])
        report = original(*args, **kwargs)
        if len(calls) == 1:
            return replace(
                report,
                checks=(
                    *report.checks,
                    precision.PrecisionCheck(
                        "diagnostic-control",
                        precision.CheckStatus.UNVERIFIED,
                        "explicit unverified control",
                    ),
                ),
            )
        return report

    monkeypatch.setattr(precision, "validate_source_samples", validate)
    assert module.main(["--receipts-dir", str(receipts), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    first = result["replays"][0]
    assert first["state"] == "COMPLETE" and first["status"] == "unverified"
    assert first["acceptance"]["returned"] is False
    assert (
        first["acceptance"]["exception_type"]
        == "jax2onnx.precision.SampleAcceptanceError"
    )


@pytest.mark.parametrize(
    "target", ("receipt", "observer", "model", "source", "symlink")
)
def test_colliding_output_preserves_original_bytes_before_any_checkpoint(
    target: str, tmp_path: Path
) -> None:
    module, receipts, folder = _fixture(tmp_path)
    paths = {
        "receipt": folder / "receipt.json",
        "observer": folder / module.OBSERVATION,
        "model": folder / module.MODEL,
        "source": tmp_path / "source.py",
    }
    paths["source"].write_bytes(b"original source bytes\n")
    original = {path: path.read_bytes() for path in paths.values()}
    if target == "symlink":
        destination = tmp_path / "output-link.json"
        destination.symlink_to(paths["receipt"])
    else:
        destination = paths[target]
    assert (
        module.main(["--receipts-dir", str(receipts), "--output", str(destination)])
        == 1
    )
    assert {path: path.read_bytes() for path in paths.values()} == original
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("shape", ([6.0], [True]))
def test_array_shapes_require_exact_json_integer_dimensions(
    shape: list[Any], tmp_path: Path
) -> None:
    module, receipts, folder = _fixture(tmp_path)
    path = folder / module.OBSERVATION
    observed = json.loads(path.read_text())
    observed["artifact_events"][0]["native_eager"][0]["shape"] = shape
    _write(path, observed)
    with pytest.raises(ValueError, match="typed shape"):
        module._preflight(receipts)


def test_jsonl_rejects_duplicate_keys_and_boolean_exitstatus(tmp_path: Path) -> None:
    module, receipts, folder = _fixture(tmp_path)
    path = folder / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt["exitstatus"] = True
    _write(path, receipt)
    with pytest.raises(ValueError, match="failure receipt"):
        module._preflight(receipts)
    receipt["exitstatus"] = 1
    _write(path, receipt)
    events = folder / "events.jsonl"
    events.write_text(
        events.read_text().replace(
            '{"kind": "phase",', '{"kind": "ignored", "kind": "phase",', 1
        )
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        module._preflight(receipts)


def test_file_read_is_capped_when_file_grows_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    module = _module()
    path = tmp_path / "growing"
    path.write_bytes(b"small")
    original = module.os.fdopen
    requests = []

    @contextmanager
    def fdopen(descriptor: int, mode: str) -> Any:
        with original(descriptor, mode) as stream:

            class Reader:
                def fileno(self) -> int:
                    return stream.fileno()

                def read(self, count: int) -> bytes:
                    requests.append(count)
                    with path.open("ab") as growing:
                        growing.write(b"x" * 128)
                    return stream.read(count)

            yield Reader()

    monkeypatch.setattr(module.os, "fdopen", fdopen)
    with pytest.raises(ValueError, match="changed/oversized"):
        module._read(path, 16)
    assert requests == [17]


def test_file_replacement_between_stat_and_open_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    path = tmp_path / "original"
    path.write_bytes(b"original")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"different")
    original = module.os.open

    def swapped(file: Path, flags: int) -> int:
        replacement.replace(path)
        return original(file, flags)

    monkeypatch.setattr(module.os, "open", swapped)
    with pytest.raises(ValueError, match="replaced"):
        module._read(path)


def test_replay_context_is_bounded_current_provenance_without_environment_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    directory = tmp_path / ".git"
    directory.mkdir()
    (directory / "HEAD").write_text("a" * 40 + "\n")
    monkeypatch.setattr(module, "_ROOT", tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "current-ci-identity")
    monkeypatch.setenv("XLA_FLAGS", "x" * 10000)
    before = dict(module.os.environ)
    context = module._replay_context()
    assert dict(module.os.environ) == before
    assert context["checkout"]["head"] == "a" * 40
    assert context["environment"]["GITHUB_SHA"]["value"] == "current-ci-identity"
    assert len(context["environment"]["XLA_FLAGS"]["value"]) == 8192
    assert context["environment"]["XLA_FLAGS"]["truncated"] is True
    assert context["python"] == sys.version
    assert context["executable"] == sys.executable
    assert context["build"] and context["compiler"] and context["soabi"]
    assert (
        context["binary_identity_claimed"]
        is context["executed_kernel_identity_claimed"]
        is False
    )
