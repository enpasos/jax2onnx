#!/usr/bin/env python3
# scripts/replay_source_cast_failure.py

"""Supplement one original failed cast with four target-only reference replays.

The original pytest failure remains authoritative. Missing unrelated evidence
does nothing; a claimed but incomplete witness fails closed before importing
frameworks. Neither the source callable nor its native references run here.
"""

import argparse
from dataclasses import asdict
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import platform
import signal
import stat
import sys
import sysconfig
import tempfile
from typing import Any
import xml.etree.ElementTree as ET


NODE_ID = (
    "tests/extra_tests/converter/test_source_conversion_semantics.py::"
    "test_floating_conversion_rounding_and_strict_special_value_gates"
    "[ties-float16-float64]"
)
OBSERVATION = "source-f64-f16-cast-observation.json"
MODEL = "source-f64-f16-cast-artifact.onnx"
_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (
    "scripts/run_pytest_batches.py",
    "scripts/_source_cast_observer.py",
    "tests/extra_tests/converter/test_source_conversion_semantics.py",
    "tests/extra_tests/converter/test_source_pointwise_mul_exp.py",
    "tests/extra_tests/converter/test_source_predicate_families.py",
    "jax2onnx/precision.py",
    "jax2onnx/_compat/jax.py",
    "jax2onnx/plugins/jax/lax/convert_element_type.py",
)
_INPUT_HEX = (
    "000000000002f03fffffffffff01f03f010000000002f03f"
    "000000000002f0bf0000000000fcef400000000000feef40"
)
_LABELS = {"group": "ties", "source": "float64", "target": "float16"}
_CRITERIA = {"mode": "bitwise", "rtol": None, "atol": None, "equal_nan": True}
_FENCES = (
    "previous_profile_restored",
    "native_context_restored",
    "source_settings_restored",
    "source_fence_held",
    "code_identity_fence_held",
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _read(path: Path, limit: int = 32 * 1024 * 1024) -> bytes:
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode), f"missing/linked {path.name}")
    _require(before.st_size <= limit, f"oversized {path.name}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"replaced {path.name}",
        )
        payload = stream.read(limit + 1)
        finished = os.fstat(stream.fileno())
    after = path.lstat()

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            value.st_mode,
        )

    _require(
        len(payload) <= limit
        and len(payload) == before.st_size
        and identity(before)
        == identity(opened)
        == identity(finished)
        == identity(after),
        f"changed/oversized {path.name}",
    )
    return payload


def _loads(payload: bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(payload, object_pairs_hook=unique)


def _json(path: Path) -> Any:
    return _loads(_read(path))


def _checkout_identity() -> dict[str, str]:
    """Read Git identity without invoking Git, hooks, or another interpreter."""
    try:
        directory = _ROOT / ".git"
        if directory.is_file():
            pointer = _read(directory, 4096).decode().strip()
            _require(pointer.startswith("gitdir: "), "unrecognized Git directory")
            directory = (_ROOT / pointer[8:]).resolve()
        head = _read(directory / "HEAD", 4096).decode().strip()
        reference = ""
        if head.startswith("ref: "):
            reference = head[5:]
            _require(
                reference.startswith("refs/") and ".." not in Path(reference).parts,
                "unrecognized Git ref",
            )
            common = directory
            if (directory / "commondir").is_file():
                common = (
                    directory / _read(directory / "commondir", 4096).decode().strip()
                ).resolve()
            loose = common / reference
            if loose.is_file():
                head = _read(loose, 4096).decode().strip()
            else:
                entries = [
                    line.split()
                    for line in _read(common / "packed-refs", 2 * 1024 * 1024)
                    .decode()
                    .splitlines()
                ]
                matches = [
                    parts[0]
                    for parts in entries
                    if len(parts) == 2 and parts[1] == reference
                ]
                _require(len(matches) == 1, "Git ref unavailable/ambiguous")
                head = matches[0]
        _require(
            len(head) in (40, 64) and len(bytes.fromhex(head)) * 2 == len(head),
            "invalid Git HEAD",
        )
        return {"head": head, "symbolic_ref": reference}
    except (OSError, UnicodeError, ValueError) as error:
        return {"unavailable": type(error).__name__}


def _replay_context() -> dict[str, Any]:
    """Bounded current-process facts; no binary or executed-kernel identity claim."""
    environment_names = (
        "GITHUB_SHA",
        "GITHUB_REF",
        "GITHUB_HEAD_REF",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_JOB",
        "GITHUB_WORKFLOW",
        "ImageOS",
        "ImageVersion",
        "XLA_FLAGS",
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPY_DISABLE_CPU_FEATURES",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    )
    environment = {}
    for name in environment_names:
        value = os.environ.get(name)
        environment[name] = {
            "value": value[:8192] if value is not None else None,
            "truncated": value is not None and len(value) > 8192,
        }
    cpu: dict[str, str] = {}
    try:
        with Path("/proc/cpuinfo").open() as stream:
            first = stream.read(16384).split("\n\n", 1)[0]
        for line in first.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in (
                "vendor_id",
                "model name",
                "cpu family",
                "model",
                "stepping",
                "microcode",
                "flags",
                "Features",
                "CPU implementer",
                "CPU part",
            ):
                cpu[key.strip()] = value.strip()[:4096]
    except OSError as error:
        cpu["unavailable"] = type(error).__name__
    return {
        "scope": "Current supplemental replay process, distinct from original-call provenance.",
        "python": sys.version[:2048],
        "executable": sys.executable[:4096],
        "prefix": sys.prefix[:4096],
        "base_prefix": sys.base_prefix[:4096],
        "build": [value[:512] for value in platform.python_build()],
        "compiler": platform.python_compiler()[:512],
        "soabi": str(sysconfig.get_config_var("SOABI"))[:512],
        "byteorder": sys.byteorder,
        "architecture": platform.machine()[:256],
        "checkout": _checkout_identity(),
        "environment": environment,
        "cpu": cpu,
        "pid": os.getpid(),
        "binary_identity_claimed": False,
        "executed_kernel_identity_claimed": False,
    }


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path): _sha(_read(path)) for path in paths}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _record(record: dict[str, Any], dtype: str) -> dict[str, Any]:
    itemsize = 8 if dtype == "float64" else 2
    _require(
        type(record["identity"]) is int
        and record["identity"] > 0
        and record["dtype"] == dtype
        and record["dtype_str"] == ("<f8" if itemsize == 8 else "<f2")
        and type(record["shape"]) is list
        and len(record["shape"]) == 1
        and type(record["shape"][0]) is int
        and record["shape"][0] == 6,
        "array typed shape differs",
    )
    raw = bytes.fromhex(record["storage_hex"])
    _require(
        len(raw) == 6 * itemsize and raw.hex() == record["storage_hex"],
        "array byte count/encoding differs",
    )
    _require(_sha(raw) == record["sha256"], "raw storage SHA differs")
    elements = [raw[i : i + itemsize].hex() for i in range(0, len(raw), itemsize)]
    _require(elements == record["element_storage_hex"], "element storage differs")
    header = {key: record[key] for key in ("dtype", "dtype_str", "shape")}
    return {
        **header,
        "storage_hex": raw.hex(),
        "raw_storage_sha256": _sha(raw),
        "typed_sha256": _sha(json.dumps(header, sort_keys=True).encode() + b"\0" + raw),
    }


def _failed(receipt: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (node, phase)
        for node, phases in receipt["phases"].items()
        for phase in phases
        if phase["outcome"] == "failed"
    ]


def _preflight(receipts: Path) -> dict[str, Any] | None:
    folders = list(islice(receipts.glob("batch-*"), 513))
    _require(len(folders) <= 512, "batch count exceeds finite bound")
    folders.sort()
    candidates = []
    for folder in folders:
        observation = folder / OBSERVATION
        receipt_path = folder / "receipt.json"
        # An observer file itself claims this exact node and cannot be ignored.
        if observation.exists():
            candidates.append(folder)
        elif receipt_path.is_file():
            receipt = _json(receipt_path)
            if any(node == NODE_ID for node, _ in _failed(receipt)):
                candidates.append(folder)
    if not candidates:
        return None
    _require(len(candidates) == 1, "ambiguous original cast witnesses")
    folder = candidates[0]
    paths = [
        folder / name
        for name in ("receipt.json", "events.jsonl", "junit.xml", OBSERVATION, MODEL)
    ]
    before = _snapshot(paths)
    receipt = _json(paths[0])
    failures = _failed(receipt)
    # A passing observed original is unrelated to this failure-only diagnostic.
    if not any(node == NODE_ID for node, _ in failures):
        observed = _json(paths[3])
        _require(
            observed["nodeid"] == NODE_ID
            and observed["status"] == "ORIGINAL_CALL_WORDS_OBSERVED"
            and all(
                event["original_result"] is None
                for event in observed["comparison_events"]
            ),
            "original cast observer contradicts the failure receipt",
        )
        return None
    _require(
        len(failures) == 1 and failures[0][0] == NODE_ID,
        "failure is not the single original cast call",
    )
    failure = failures[0][1]
    _require(
        receipt["kind"] == "batch"
        and type(receipt["exitstatus"]) is int
        and receipt["exitstatus"] == 1
        and receipt["full_collection_checked"] is True
        and not receipt["collection_errors"]
        and NODE_ID in receipt["assigned_ids"]
        and len(set(receipt["assigned_ids"])) == len(receipt["assigned_ids"])
        and receipt["assigned_ids"] == receipt["selected_ids"]
        and failure["when"] == "call"
        and "values fail bitwise comparison" in failure["longrepr"],
        "original failure receipt differs",
    )
    phases = receipt["phases"][NODE_ID]
    _require(
        [(p["when"], p["outcome"]) for p in phases]
        == [("setup", "passed"), ("call", "failed"), ("teardown", "passed")],
        "original call phases incomplete",
    )
    events = [_loads(line) for line in _read(paths[1]).splitlines()]
    failed_events = [e for e in events if e.get("outcome") == "failed"]
    _require(
        failed_events == [{"kind": "phase", "nodeid": NODE_ID, **failure}]
        and [
            {k: v for k, v in e.items() if k not in ("kind", "nodeid")}
            for e in events
            if e.get("kind") == "phase" and e.get("nodeid") == NODE_ID
        ]
        == phases,
        "phase events disagree with receipt",
    )
    tree = ET.fromstring(_read(paths[2]))
    failed_cases = [
        case
        for case in tree.iter("testcase")
        if case.find("failure") is not None or case.find("error") is not None
    ]
    _require(len(failed_cases) == 1, "JUnit failure count differs")
    case = failed_cases[0]
    _require(
        case.attrib["classname"]
        == "tests.extra_tests.converter.test_source_conversion_semantics"
        and case.attrib["name"] == NODE_ID.split("::")[1]
        and case.find("failure") is not None
        and "values fail bitwise comparison" in ET.tostring(case, encoding="unicode"),
        "JUnit original failure differs",
    )
    _require(
        sum(
            item.attrib.get("name") == case.attrib["name"]
            and item.attrib.get("classname") == case.attrib["classname"]
            for item in tree.iter("testcase")
        )
        == 1,
        "ambiguous original JUnit case",
    )
    observed = _json(paths[3])
    _require(
        observed["nodeid"] == NODE_ID
        and observed["status"] == "ORIGINAL_CALL_WORDS_OBSERVED"
        and observed["parameter_order"] == _LABELS
        and observed["byteorder"] == "little"
        and not observed["observer_errors"]
        and all(observed[name] is True for name in _FENCES)
        and all(
            observed[name] is False
            for name in (
                "native_references_recomputed_by_observer",
                "inputs_criteria_or_assertions_changed",
                "numerical_acceptance_claimed_by_observer",
            )
        ),
        "original observer is incomplete or changed its source",
    )
    source_before = observed["source_files_before"]
    _require(
        source_before == observed["source_files_end"], "original source fence differs"
    )
    _require(
        observed["jax_config_before"] == observed["jax_config_end"],
        "original setting fence differs",
    )
    _require(
        set(source_before) == set(_SOURCE_FILES), "original source file set differs"
    )
    provenance = observed["function_code_provenance"]
    _require(
        set(provenance) == {"artifact", "runtime", "compare"},
        "original code provenance incomplete",
    )
    for role, name, relative in (
        (
            "artifact",
            "_artifact",
            "tests/extra_tests/converter/test_source_pointwise_mul_exp.py",
        ),
        ("runtime", "_run_cpu_samples", "jax2onnx/precision.py"),
        ("compare", "_compare", "jax2onnx/precision.py"),
    ):
        code = provenance[role]
        _require(
            code["name"] == name
            and type(code["function_identity"]) is int
            and code["function_identity"] > 0
            and type(code["code_identity"]) is int
            and code["code_identity"] > 0
            and code["filename"].endswith("/" + relative)
            and len(bytes.fromhex(code["marshal_sha256"])) == 32
            and type(code["firstlineno"]) is int
            and code["firstlineno"] > 0,
            "original code provenance differs",
        )
    source_paths = [_ROOT / name for name in _SOURCE_FILES]
    source_matches_original = {}
    for name, path in zip(_SOURCE_FILES, source_paths, strict=True):
        payload = _read(path)
        _require(
            type(source_before[name]["bytes"]) is int
            and source_before[name]["bytes"] >= 0
            and len(bytes.fromhex(source_before[name]["sha256"])) == 32,
            "original source digest record differs",
        )
        source_matches_original[name] = (
            _sha(payload) == source_before[name]["sha256"]
            and len(payload) == source_before[name]["bytes"]
        )
    _require(
        source_matches_original["jax2onnx/precision.py"],
        "active sample validator differs from original evidence",
    )
    artifacts = observed["artifact_events"]
    _require(
        len(artifacts) == 1 and artifacts[0]["labels"] == _LABELS,
        "artifact observation differs",
    )
    artifact = artifacts[0]
    operations = artifact["graph_ops"]
    _require(
        len(operations) == 1
        and operations[0]["op_type"] == "Cast"
        and operations[0]["input"] == ["in_0"]
        and len(operations[0]["output"]) == 1
        and operations[0]["attributes"] == [{"name": "to", "type": 2, "int": 10}],
        "original observed Cast differs",
    )
    _require(
        type(operations[0]["attributes"][0]["type"]) is int
        and type(operations[0]["attributes"][0]["int"]) is int,
        "original Cast attribute integer encoding differs",
    )
    _require(
        _sha(_read(paths[4], 1024 * 1024)) == artifact["sha256"], "artifact SHA differs"
    )
    references = {}
    for name in ("native_eager", "native_fresh_jit"):
        _require(len(artifact[name]) == 1, "native reference arity differs")
        references[name] = _record(artifact[name][0], "float16")
    runtime, comparisons = observed["runtime_events"], observed["comparison_events"]
    _require(
        1 <= len(runtime) <= 4 and len(runtime) == len(comparisons),
        "original runtime/comparison count differs",
    )
    for index, (run, comparison) in enumerate(zip(runtime, comparisons, strict=True)):
        labels = {
            **_LABELS,
            "optimization": ("disabled" if index < 2 else "all"),
            "reference_index": index % 2,
        }
        _require(
            run["labels"] == comparison["labels"] == labels
            and type(run["labels"]["reference_index"]) is int
            and type(comparison["labels"]["reference_index"]) is int
            and run["providers"] == ["CPUExecutionProvider"],
            "original target labels differ",
        )
        _require(
            set(run["inputs"]) == {"in_0"} and len(run["outputs"]) == 1,
            "original runtime arity differs",
        )
        inputs = _record(run["inputs"]["in_0"], "float64")
        _require(inputs["storage_hex"] == _INPUT_HEX, "original six input words differ")
        actual = _record(run["outputs"][0], "float16")
        _require(
            actual == _record(comparison["actual"], "float16"),
            "observed actual output differs",
        )
        reference = references[("native_eager", "native_fresh_jit")[index % 2]]
        _require(
            reference == _record(comparison["reference"], "float16")
            and comparison["criteria"] == _CRITERIA,
            "original native reference/criterion differs",
        )
        different = [
            i
            for i in range(6)
            if actual["storage_hex"][i * 4 : i * 4 + 4]
            != reference["storage_hex"][i * 4 : i * 4 + 4]
        ]
        _require(
            comparison["different_indices"] == different
            and all(type(value) is int for value in comparison["different_indices"]),
            "original differing indices disagree",
        )
        _require(
            comparison["original_result"]
            == ("values fail bitwise comparison" if different else None),
            "original comparison result disagrees",
        )
        _require(
            bool(different) == (index == len(runtime) - 1),
            "original first failure order differs",
        )
    _require(_snapshot(paths) == before, "original evidence changed during preflight")
    return {
        "folder": folder,
        "model": paths[4],
        "artifact_sha256": artifact["sha256"],
        "graph_ops": operations,
        "inputs": inputs,
        "references": references,
        "evidence_paths": paths,
        "evidence_before": before,
        "source_paths": [*source_paths, Path(__file__).resolve()],
        "original_source_files": source_before,
        "replay_source_matches_original": source_matches_original,
        "original_event_commit": observed.get("event_commit"),
        "original_failure": failure,
    }


def _model_gate(path: Path, operations: list[dict[str, Any]]) -> None:
    # Official protobuf parsing occurs only after the stdlib evidence preflight.
    import onnx

    model = onnx.load_model_from_string(_read(path, 1024 * 1024))
    graph = model.graph
    _require(
        len(graph.node) == len(graph.input) == len(graph.output) == 1
        and not graph.initializer
        and not graph.sparse_initializer
        and not model.functions,
        "model is not the single original Cast",
    )
    node = graph.node[0]
    _require(
        node.op_type == "Cast"
        and node.domain == ""
        and node.overload == ""
        and list(node.input) == [graph.input[0].name]
        and list(node.output) == [graph.output[0].name]
        and len(node.attribute) == 1,
        "model Cast wiring differs",
    )
    _require(
        len(model.opset_import) == 1
        and model.opset_import[0].domain == ""
        and model.opset_import[0].version == 23,
        "model opset differs",
    )
    attribute = node.attribute[0]
    _require(
        attribute.name == "to"
        and attribute.type == onnx.AttributeProto.INT
        and attribute.i == onnx.TensorProto.FLOAT16,
        "model Cast destination differs",
    )
    for port, dtype in (
        (graph.input[0], onnx.TensorProto.DOUBLE),
        (graph.output[0], onnx.TensorProto.FLOAT16),
    ):
        _require(
            port.type.HasField("tensor_type")
            and port.type.tensor_type.elem_type == dtype
            and len(port.type.tensor_type.shape.dim) == 1
            and port.type.tensor_type.shape.dim[0].HasField("dim_value")
            and port.type.tensor_type.shape.dim[0].dim_value == 6,
            "model typed shape differs",
        )
    _require(graph.input[0].name == "in_0", "model original input name differs")
    _require(
        list(node.output) == operations[0]["output"],
        "model and observer output names differ",
    )


def _replay(witness: dict[str, Any], result: dict[str, Any], checkpoint: Any) -> None:
    _model_gate(witness["model"], witness["graph_ops"])
    import numpy as np

    from jax2onnx.precision import (
        CheckStatus,
        SampleAcceptanceError,
        SampleCriteria,
        SourceSample,
        validate_source_samples,
    )

    def array(record: dict[str, Any]) -> Any:
        return np.frombuffer(
            bytes.fromhex(record["storage_hex"]), dtype=record["dtype_str"]
        ).reshape(record["shape"])

    for optimization in ("disabled", "all"):
        for name, reference in witness["references"].items():
            entry: dict[str, Any] = {
                "optimization": optimization,
                "reference": name,
                "state": "STARTED",
            }
            result["replays"].append(entry)
            checkpoint()
            try:
                report = validate_source_samples(
                    str(witness["model"]),
                    samples=[
                        SourceSample(
                            (array(witness["inputs"]),),
                            (array(reference),),
                            SampleCriteria("bitwise"),
                        )
                    ],
                    optimization=optimization,
                )
                entry.update(
                    status=report.status.value,
                    artifact_sha256=report.artifact_sha256,
                    target=asdict(report.target) if report.target is not None else None,
                    checks=[asdict(check) for check in report.checks],
                    samples=[asdict(sample) for sample in report.samples],
                )
                checkpoint()
                try:
                    report.require_sample_acceptance()
                except SampleAcceptanceError as error:
                    entry["acceptance"] = {
                        "returned": False,
                        "exception_type": type(error).__module__
                        + "."
                        + type(error).__qualname__,
                        "message": str(error),
                    }
                    _require(
                        report.status in (CheckStatus.FAILED, CheckStatus.UNVERIFIED),
                        "validated replay report unexpectedly rejected acceptance",
                    )
                except Exception as error:
                    entry["acceptance"] = {
                        "returned": False,
                        "exception_type": type(error).__module__
                        + "."
                        + type(error).__qualname__,
                        "message": str(error),
                    }
                    raise
                else:
                    entry["acceptance"] = {"returned": True}
                    _require(
                        report.status is CheckStatus.VALIDATED,
                        "nonvalidated replay report unexpectedly accepted",
                    )
                _require(
                    report.artifact_sha256 == witness["artifact_sha256"],
                    "replay artifact digest differs",
                )
                _require(len(report.samples) == 1, "replay sample evidence missing")
                sample = report.samples[0]
                _require(
                    len(sample.inputs) == len(sample.references) == 1
                    and sample.inputs[0].sha256 == witness["inputs"]["typed_sha256"]
                    and sample.references[0].sha256 == reference["typed_sha256"]
                    and asdict(sample.criteria) == _CRITERIA,
                    "replay typed input/reference evidence differs",
                )
                entry["state"] = "COMPLETE"
            except Exception as error:
                entry.update(
                    state="UNVERIFIED",
                    exception_type=type(error).__module__
                    + "."
                    + type(error).__qualname__,
                    message=str(error),
                )
            checkpoint()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    destination = args.output or args.receipts_dir / "source-cast-failure-replay.json"
    # Never checkpoint, including an error result, over any existing evidence.
    reserved = {
        "receipt.json",
        "request.json",
        "run.json",
        "events.jsonl",
        "junit.xml",
        OBSERVATION,
        MODEL,
    }
    source_paths = [_ROOT / name for name in _SOURCE_FILES]
    forbidden = {path.resolve() for path in (*source_paths, Path(__file__).resolve())}
    if (
        destination.exists()
        or destination.is_symlink()
        or destination.resolve() in forbidden
        or (
            destination.resolve().is_relative_to(args.receipts_dir.resolve())
            and destination.name in reserved
        )
    ):
        print(
            "UNVERIFIED: rejected existing/colliding diagnostic output", file=sys.stderr
        )
        return 1
    result: dict[str, Any] = {
        "status": "UNVERIFIED",
        "scope": "Supplemental target replay of frozen original-call references; original pytest outcome remains authoritative.",
        "source_callable_called": False,
        "native_references_recomputed": False,
        "raw_target_outputs_captured": False,
        "output_evidence": "Public validation reports contain typed array hashes, not raw output words.",
        "replays": [],
    }

    def checkpoint() -> None:
        _write(destination, result)

    def interrupted(number: int, frame: Any) -> None:
        result.update(status="UNVERIFIED", interrupted_signal=number)
        checkpoint()
        raise KeyboardInterrupt

    prior = {
        number: signal.signal(number, interrupted)
        for number in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        witness = _preflight(args.receipts_dir)
        if witness is None:
            result.update(
                status="NOT_APPLICABLE",
                reason="No single original failed cast witness.",
            )
            checkpoint()
            return 0
        sources_before = _snapshot(witness["source_paths"])
        result.update(
            artifact_sha256=witness["artifact_sha256"],
            original_batch=witness["folder"].name,
            original_event_commit=witness["original_event_commit"],
            original_failure=witness["original_failure"],
            frozen_inputs=witness["inputs"],
            frozen_references=witness["references"],
            evidence_before=witness["evidence_before"],
            original_source_files=witness["original_source_files"],
            replay_source_matches_original=witness["replay_source_matches_original"],
            replay_source_before=sources_before,
            replay_context=_replay_context(),
        )
        checkpoint()
        try:
            _replay(witness, result, checkpoint)
        finally:
            result["evidence_fence_held"] = (
                _snapshot(witness["evidence_paths"]) == witness["evidence_before"]
            )
            result["replay_source_fence_held"] = (
                _snapshot(witness["source_paths"]) == sources_before
            )
            result["artifact_fence_held"] = (
                _sha(_read(witness["model"])) == witness["artifact_sha256"]
            )
            checkpoint()
        complete = len(result["replays"]) == 4 and all(
            entry["state"] == "COMPLETE" for entry in result["replays"]
        )
        result["status"] = (
            "REPLAY_OBSERVED"
            if complete
            and all(
                result[key] is True
                for key in (
                    "evidence_fence_held",
                    "replay_source_fence_held",
                    "artifact_fence_held",
                )
            )
            else "UNVERIFIED"
        )
        checkpoint()
        return 0 if result["status"] == "REPLAY_OBSERVED" else 1
    except Exception as error:
        result.update(
            status="UNVERIFIED",
            exception_type=type(error).__module__ + "." + type(error).__qualname__,
            message=str(error),
        )
        checkpoint()
        return 1
    finally:
        for number, previous in prior.items():
            signal.signal(number, previous)


if __name__ == "__main__":
    raise SystemExit(main())
