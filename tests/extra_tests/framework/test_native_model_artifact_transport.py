# tests/extra_tests/framework/test_native_model_artifact_transport.py

"""Small real files/children; synthetic transport metadata is not model evidence."""

import builtins
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_OWNER = "tests/extra_tests/converter/test_source_model_native_bindings_exports.py"
_FUNCTION = "test_original_full_model_native_bindings_cpu_gate"
_ROLES = ("eager", "jit-first-for-sample", "same-jit-warmed")
_CRITERION = "67cd821a4c40e0464f09a450e7c66afac99a0cef26b8569b17bd842f51c60c51"
_NODES = tuple(
    f"{_OWNER}::{_FUNCTION}[raw-x64-{flag}-{index}]"
    for flag in ("off", "on")
    for index in (0, 1, 3, 4, 20)
) + (
    _OWNER + "::test_frozen_five_owner_output_contract_rejects_corruption",
    _OWNER + "::test_reference_scaled_direction_and_all_native_roles_are_required",
)
_ENV = "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _module() -> Any:
    spec = spec_from_file_location(
        "_native_transport_under_test",
        _ROOT / "scripts/_native_model_artifact_transport.py",
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def transport_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    module = _module()
    # Synthetic scalar packets are deliberately not the real 3.124GB output
    # corpus. This fake disk observation qualifies logic, never local capacity.
    monkeypatch.setattr(module, "_free", lambda path: 1024**4)
    return module


def _npy(
    data: bytes = b"\x00\x00\x80?",
    *,
    dtype: str = "<f4",
    shape: tuple[Any, ...] = (1,),
    version: int = 1,
    fortran: bool = False,
    header_text: str | None = None,
) -> bytes:
    header = (
        repr({"descr": dtype, "fortran_order": fortran, "shape": shape})
        if header_text is None
        else header_text
    )
    width = 2 if version == 1 else 4
    encoded = header.encode("utf-8" if version == 3 else "latin1")
    padding = (-(8 + width + len(encoded) + 1)) % 64
    encoded += b" " * padding + b"\n"
    return (
        b"\x93NUMPY"
        + bytes([version, 0])
        + len(encoded).to_bytes(width, "little")
        + encoded
        + data
    )


def _array(
    data: bytes = b"\x00\x00\x80?", dtype: str = "<f4", shape: list[int] | None = None
) -> dict[str, Any]:
    return {
        "dtype": dtype,
        "shape": [1] if shape is None else shape,
        "weak_type": False,
        "bytes": len(data),
        "sha256": _sha(data),
    }


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize(
    "dtype,data",
    [("<f4", b"\x00\x00\x80?"), ("<i4", b"\x01\x00\x00\x00"), ("|b1", b"\x01")],
    ids=["float", "integer", "bool"],
)
def test_real_npy_file_hash_is_separate_from_exact_typed_data_hash(
    tmp_path: Path,
    transport_module: Any,
    version: int,
    dtype: str,
    data: bytes,
) -> None:
    path = tmp_path / "array.npy"
    raw = _npy(data, dtype=dtype, version=version)
    path.write_bytes(raw)
    proof, retained = transport_module._file(
        path, array=_array(data, dtype), file_sha256=_sha(raw)
    )
    assert retained is None and proof["errors"] == [] and proof["identity_held"] is True
    assert proof["sha256"] == _sha(raw) != proof["array"]["sha256"]
    assert proof["array"]["sha256"] == _sha(data)
    assert proof["array"]["bytes"] == len(data)
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    "fault",
    [
        "magic",
        "version",
        "object",
        "structured",
        "fortran",
        "duplicate-key",
        "bool-dimension",
        "negative-dimension",
        "header-bound",
        "truncated",
        "trailing",
        "raw-hash",
        "file-hash",
        "call-expression",
    ],
)
def test_corrupt_or_unsupported_npy_keeps_full_file_proof_without_loading_arrays(
    tmp_path: Path,
    transport_module: Any,
    fault: str,
) -> None:
    path = tmp_path / "array.npy"
    original = _npy()
    path.write_bytes(original)
    assert transport_module._file(path, array=_array())[0]["errors"] == []
    raw, expected, expected_file = original, _array(), _sha(original)
    if fault == "magic":
        raw = b"INVALID!" + original[8:]
    elif fault == "version":
        raw = original[:6] + b"\x09\x00" + original[8:]
    elif fault == "object":
        raw = _npy(dtype="|O8")
    elif fault == "structured":
        raw = _npy(
            header_text="{'descr': [('field', '<f4')], 'fortran_order': False, 'shape': (1,)}"
        )
    elif fault == "fortran":
        raw = _npy(fortran=True)
    elif fault == "duplicate-key":
        raw = _npy(
            header_text="{'descr': '|O8', 'descr': '<f4', 'fortran_order': False, 'shape': (1,)}"
        )
    elif fault == "bool-dimension":
        raw = _npy(shape=(True,))
    elif fault == "negative-dimension":
        raw = _npy(shape=(-1,))
    elif fault == "header-bound":
        raw = b"\x93NUMPY\x02\x00" + (65537).to_bytes(4, "little") + b"x" * 65537
    elif fault == "truncated":
        raw = original[:-1]
    elif fault == "trailing":
        raw = original + b"x"
    elif fault == "raw-hash":
        expected["sha256"] = "0" * 64
    elif fault == "file-hash":
        expected_file = "0" * 64
    else:
        raw = _npy(
            header_text="{'descr': '<f4', 'fortran_order': False, 'shape': tuple([1])}"
        )
    path.write_bytes(raw)
    proof, _ = transport_module._file(path, array=expected, file_sha256=expected_file)
    assert proof["errors"] and proof["identity_held"] is True
    assert proof["sha256"] == _sha(raw) and proof["bytes"] == len(raw)
    assert path.read_bytes() == raw
    if fault == "fortran":
        assert (
            "array" not in proof
        ), "Fortran bytes must never be labeled the receipt's C-order data"


def test_mutation_during_streaming_invalidates_the_file_identity_fence(
    tmp_path: Path,
    transport_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "array.npy"
    path.write_bytes(_npy())
    original = transport_module._npy_header
    original_time = path.stat().st_mtime_ns

    def change(read: Any) -> Any:
        result = original(read)
        with path.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            stream.write(b"\x00")
            stream.flush()
            os.fsync(stream.fileno())
        os.utime(path, ns=(path.stat().st_atime_ns, original_time + 1))
        return result

    monkeypatch.setattr(transport_module, "_npy_header", change)
    proof, _ = transport_module._file(path, array=_array())
    assert (
        proof["identity_held"] is False and "file-identity-changed" in proof["errors"]
    )
    assert proof["sha256"] and path.exists()


@pytest.mark.parametrize("fault", ["symlink", "hardlink", "oversize"])
def test_unsafe_or_oversized_files_are_not_followed_or_truncated(
    tmp_path: Path,
    transport_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    original = tmp_path / "original"
    original.write_bytes(b"unchanged")
    path = tmp_path / "payload"
    if fault == "symlink":
        path.symlink_to(original)
    elif fault == "hardlink":
        os.link(original, path)
    else:
        path.write_bytes(b"unchanged")
        monkeypatch.setattr(transport_module, "_FILE_LIMIT", 4)
    with pytest.raises(ValueError):
        transport_module._file(path)
    assert original.read_bytes() == b"unchanged" and path.read_bytes() == b"unchanged"


def test_a_symlink_in_a_parent_component_is_not_followed(
    tmp_path: Path, transport_module: Any
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload").write_bytes(b"not owned by the packet")
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        transport_module._file(link / "payload")
    assert (outside / "payload").read_bytes() == b"not owned by the packet"


def test_foreign_file_ownership_is_rejected_before_opening_payload_bytes(
    tmp_path: Path,
    transport_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"unchanged")
    original = transport_module._path_stat
    values = list(path.stat())
    values[4] = os.getuid() + 1
    foreign = os.stat_result(values)
    monkeypatch.setattr(
        transport_module,
        "_path_stat",
        lambda value: foreign if value == path else original(value),
    )
    with pytest.raises(ValueError, match="owned unique regular"):
        transport_module._file(path)
    assert path.read_bytes() == b"unchanged"


def _payload(folder: Path, node: str) -> None:
    index = int(node.rsplit("-", 1)[1][:-1])
    folder.mkdir()
    data = b"\x00\x00\x80?"
    encoded = _npy(data)
    array = _array(data)
    model = b"synthetic transport fixture; not executable ONNX: " + node.encode()
    (folder / "model.onnx").write_bytes(model)
    contract = {
        "fixture_scope": "synthetic transport metadata, not original native output"
    }
    record: dict[str, Any] = {
        "status": "PASS_FINITE_ORIGINAL_FIVE_OWNER_SUBGATE",
        "raw_criteria": None,
        "criterion_sha256": _CRITERION,
        "artifact_sidecars": [],
        "contract": contract,
        "contract_sha256": _sha(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ),
        "function_definitions": [["fixture", "function"]],
        "feeds": {"2": [array], "5": [array]},
        "native": {},
        "cpu": {},
        "events": [],
        "native_input_usage_correspondence": {"artifact_sha256": _sha(model)},
    }
    for batch in (2, 5):
        (folder / f"feed-B{batch}-0.npy").write_bytes(encoded)
    keys = [
        f"B{batch}-{flag}"
        for batch in (2, 5)
        for flag in (("staticFalse",) if index == 20 else ("True", "False"))
    ]

    def event(stage: str, *, returns_array: bool = False, **fields: Any) -> None:
        value = {"stage": stage, "status": "ACTUAL_RETURN", **fields}
        if returns_array:
            name = f"event-{len(record['events']) + 1:03d}-array-0.npy"
            (folder / name).write_bytes(encoded)
            value["arrays"] = [{"path": name, "sha256": _sha(encoded), "actual": array}]
        record["events"].append(value)

    for key in keys:
        record["native"][key] = {"roles": {}}
        for role in _ROLES:
            event("native-" + key + "-" + role, returns_array=True)
            record["native"][key]["roles"][role] = {"actual": array}
    event(
        "export-original-source-callable",
        artifact_path="model.onnx",
        artifact_sha256=_sha(model),
    )
    for mode in ("disabled", "all"):
        event("session-" + mode)
        record["cpu"][mode] = {"artifact_sha256": _sha(model), "runs": {}}
        for key in keys:
            event("cpu-" + mode + "-" + key, returns_array=True)
            feed = {"input": array}
            if index != 20:
                feed["deterministic"] = _array(
                    bytes([int(key.endswith("-True"))]), "|b1", []
                )
            record["cpu"][mode]["runs"][key] = {
                "feed": feed,
                "comparisons": {
                    role: {"criterion_sha256": _CRITERION, "all_components": True}
                    for role in _ROLES
                },
            }
    for name in (
        "original-report.json",
        "original-native-symbolic-trace.json",
        "activated-full-world-symbolic-trace.json",
        "source-function-body-1.json",
    ):
        _write(folder / name, {"synthetic_transport_fixture": True})
    if index != 20:
        _write(
            folder / "activated-inlined-original-symbolic-trace.json",
            {"synthetic_transport_fixture": True},
        )
    _write(
        folder / "source-function-definition-1.json",
        {"domain": "fixture", "name": "function", "overload": ""},
    )
    _write(folder / "receipt.json", record)


def _worker(
    root: Path,
    environment: dict[str, str],
    node: str,
    number: int,
    full: tuple[str, ...],
    *,
    failed: bool = False,
) -> tuple[Path, dict[str, Any]]:
    folder = root / f"batch-{number:04d}"
    folder.mkdir()
    request = {
        "kind": "batch",
        "assigned": [node],
        "full_preselection_ids": list(full),
        "collection_skips": [],
        "pytest_args": ["-v", "--tb=short", "-x"],
    }
    phases = [
        {
            "when": when,
            "outcome": "failed" if failed and when == "call" else "passed",
            "duration": 0.001,
        }
        for when in ("setup", "call", "teardown")
    ]
    receipt = {
        "kind": "batch",
        "assigned_ids": [node],
        "selected_ids": [node],
        "completed": [node],
        "full_preselection_ids": list(full),
        "supplemental": False,
        "native_source_recomputed": False,
        "startup_context": {"environment": {_ENV: environment[_ENV]}},
        "phases": {node: phases},
    }
    _write(folder / "request.json", request)
    _write(folder / "receipt.json", receipt)
    (folder / "events.jsonl").write_text(
        "".join(
            json.dumps({"kind": "phase", "nodeid": node, **phase}) + "\n"
            for phase in phases
        )
    )
    suite = ET.Element("testsuite")
    case = ET.SubElement(
        suite,
        "testcase",
        classname=_OWNER[:-3].replace("/", "."),
        name=node.split("::", 1)[1],
    )
    if failed:
        ET.SubElement(case, "failure", message="original synthetic failure")
    (folder / "junit.xml").write_bytes(ET.tostring(suite))
    (folder / "output.log").write_text(
        "synthetic worker; no model or framework invocation\n"
    )
    return folder, request


def _transport(tmp_path: Path, module: Any, nodes: tuple[str, ...]) -> Any:
    root = tmp_path / "run"
    root.mkdir()
    parent = {"UNRELATED": "retained"}
    transport = module.NativeModelTransport(root, "a" * 40, parent)
    assert parent == {"UNRELATED": "retained"}
    assert transport.environment["UNRELATED"] == "retained"
    transport.collection(list(nodes))
    return transport


def _case_folder(transport: Any, node: str) -> Path:
    flag, index = node.rsplit("[raw-x64-", 1)[1][:-1].split("-")
    return transport.evidence / f"{int(index):02d}-x64-{flag}"


def test_complete_ten_small_synthetic_packets_plus_two_no_payload_controls(
    tmp_path: Path,
    transport_module: Any,
) -> None:
    transport = _transport(tmp_path, transport_module, _NODES)
    for number, node in enumerate(_NODES, 1):
        if number <= 10:
            _payload(_case_folder(transport, node), node)
        folder, request = _worker(
            transport.root, transport.environment, node, number, _NODES
        )
        transport.before_worker(folder, request)
        transport.closed_worker(folder, request, 0)
    transport.finish("PASS")
    result = _load(transport.path)
    assert result["status"] == "COMPLETE_ORIGINAL_OWNER_TRANSPORT_ONLY", result[
        "errors"
    ]
    assert (
        result["whole_original_owner_collected"] is True and len(result["cases"]) == 12
    )
    assert [value["assigned"][0] for value in result["workers"]] == list(_NODES)
    assert (
        sum(
            len(list(path.glob("event-*.npy"))) for path in transport.evidence.iterdir()
        )
        == 180
    )
    assert (
        result["numerical_acceptance"] == "NOT_ASSERTED"
        and result["full_B_to_F_acceptance"] == "UNVERIFIED"
    )
    assert all(
        item["complete_capacity_proof"] is False
        for item in result["resource_observations"]
        if "complete_capacity_proof" in item
    )


def test_subset_scope_does_not_claim_the_whole_ten_model_owner(
    tmp_path: Path, transport_module: Any
) -> None:
    nodes = (_NODES[0],)
    transport = _transport(tmp_path, transport_module, nodes)
    _payload(_case_folder(transport, nodes[0]), nodes[0])
    folder, request = _worker(transport.root, transport.environment, nodes[0], 1, nodes)
    transport.closed_worker(folder, request, 0)
    transport.finish("PASS")
    assert (
        transport.record["status"] == "SUBSET_TRANSPORT_ONLY"
        and transport.record["whole_original_owner_collected"] is False
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing-array",
        "extra-file",
        "corrupt-data",
        "bad-receipt-reference",
        "duplicate-reference",
        "wrong-model-hash",
        "missing-trace",
        "phase-mismatch",
        "original-failure",
        "supplemental",
        "interrupted-no-model-receipt",
    ],
)
def test_partial_or_corrupt_packets_keep_original_outcomes_and_available_bytes(
    tmp_path: Path,
    transport_module: Any,
    fault: str,
) -> None:
    nodes = (_NODES[0],)
    transport = _transport(tmp_path, transport_module, nodes)
    case = _case_folder(transport, nodes[0])
    _payload(case, nodes[0])
    folder, request = _worker(
        transport.root,
        transport.environment,
        nodes[0],
        1,
        nodes,
        failed=fault == "original-failure",
    )
    record = _load(case / "receipt.json")
    if fault == "missing-array":
        (case / "feed-B2-0.npy").unlink()
    elif fault == "extra-file":
        (case / "unexpected.npy").write_bytes(_npy())
    elif fault == "corrupt-data":
        path = case / "feed-B2-0.npy"
        path.write_bytes(path.read_bytes()[:-1] + b"\x00")
    elif fault == "bad-receipt-reference":
        record["events"][0]["arrays"][0]["path"] = "../../foreign.npy"
    elif fault == "duplicate-reference":
        record["events"][1]["arrays"][0]["path"] = record["events"][0]["arrays"][0][
            "path"
        ]
    elif fault == "wrong-model-hash":
        record["events"][12]["artifact_sha256"] = "0" * 64
    elif fault == "missing-trace":
        (case / "source-function-body-1.json").unlink()
    elif fault == "phase-mismatch":
        events = [
            json.loads(line)
            for line in (folder / "events.jsonl").read_text().splitlines()
        ]
        events[1]["outcome"] = "failed"
        (folder / "events.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )
    elif fault == "original-failure":
        record["status"] = "FAIL"
    elif fault == "supplemental":
        receipt = _load(folder / "receipt.json")
        receipt["supplemental"] = True
        _write(folder / "receipt.json", receipt)
    _write(case / "receipt.json", record)
    if fault == "interrupted-no-model-receipt":
        (case / "receipt.json").unlink()
        receipt = _load(folder / "receipt.json")
        receipt["completed"] = []
        receipt["phases"][nodes[0]] = receipt["phases"][nodes[0]][:1]
        _write(folder / "receipt.json", receipt)
        (folder / "events.jsonl").write_text(
            json.dumps(
                {"kind": "phase", "nodeid": nodes[0], **receipt["phases"][nodes[0]][0]}
            )
            + "\n"
        )
        (folder / "junit.xml").unlink()
    before = {path.name: path.read_bytes() for path in case.iterdir()}
    original_worker = (folder / "receipt.json").read_bytes()
    returncode = (
        -15
        if fault == "interrupted-no-model-receipt"
        else (1 if fault == "original-failure" else 0)
    )
    transport.closed_worker(folder, request, returncode)
    transport.finish("FAIL" if returncode else "PASS")
    assert transport.record["status"] == "UNVERIFIED"
    assert {path.name: path.read_bytes() for path in case.iterdir()} == before
    assert (folder / "receipt.json").read_bytes() == original_worker
    assert transport.record["cases"][nodes[0]]["files"]
    assert transport.record["workers"][0]["original_returncode"] == returncode


@pytest.mark.parametrize(
    "fault", ["late-file", "late-directory", "changed-file", "duplicate-worker"]
)
def test_final_membership_stat_fences_and_duplicate_execution_preserve_prior_proofs(
    tmp_path: Path,
    transport_module: Any,
    fault: str,
) -> None:
    nodes = (_NODES[0],)
    transport = _transport(tmp_path, transport_module, nodes)
    case = _case_folder(transport, nodes[0])
    _payload(case, nodes[0])
    folder, request = _worker(transport.root, transport.environment, nodes[0], 1, nodes)
    transport.closed_worker(folder, request, 0)
    before = json.dumps(transport.record["cases"][nodes[0]], sort_keys=True)
    if fault == "late-file":
        (case / "late").write_bytes(b"retained")
    elif fault == "late-directory":
        (transport.evidence / "extra").mkdir()
    elif fault == "changed-file":
        (case / "model.onnx").write_bytes(b"changed after original binding")
    else:
        folder2, request2 = _worker(
            transport.root, transport.environment, nodes[0], 2, nodes
        )
        transport.closed_worker(folder2, request2, 0)
    transport.finish("PASS")
    assert transport.record["status"] == "UNVERIFIED"
    assert json.dumps(transport.record["cases"][nodes[0]], sort_keys=True) == before


@pytest.mark.parametrize(
    "kind", ["collision", "symlink-root", "external", "external-empty"]
)
def test_fresh_owned_paths_and_explicit_external_environment_intent(
    tmp_path: Path,
    transport_module: Any,
    kind: str,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_bytes(b"do not touch")
    if kind in ("collision", "symlink-root"):
        if kind == "collision":
            (root / "native-model-artifacts").mkdir()
        else:
            (root / "native-model-artifacts").symlink_to(
                outside, target_is_directory=True
            )
        with pytest.raises(ValueError):
            transport_module.NativeModelTransport(root, "a" * 40, {})
    else:
        value = "" if kind == "external-empty" else str(outside)
        environment = {_ENV: value, "OTHER": "preserve"}
        before = dict(environment)
        transport = transport_module.NativeModelTransport(root, "a" * 40, environment)
        transport.collection(list(_NODES))
        transport.finish("PASS")
        assert transport.environment == environment == before
        assert (
            transport.record["status"] == "EXTERNAL_UNVERIFIED"
            and not transport.evidence.exists()
        )
    assert (
        list(outside.iterdir()) == [outside / "sentinel"]
        and (outside / "sentinel").read_bytes() == b"do not touch"
    )


def test_capacity_refusal_records_unexecuted_denominator_without_starting_a_worker(
    tmp_path: Path,
    transport_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _transport(tmp_path, transport_module, _NODES)
    monkeypatch.setattr(
        transport_module, "_free", lambda path: 4 * 1024**3 + 128 * 1024**2
    )
    folder = transport.root / "batch-0001"
    with pytest.raises(transport_module.ArtifactResourceCensor):
        transport.before_worker(folder, {"kind": "batch", "assigned": [_NODES[0]]})
    assert not folder.exists() and transport.record["expected_owner_ids"] == list(
        _NODES
    )
    assert transport.record["resource_censored"]["unexecuted_owner_ids"] == list(_NODES)
    assert transport.record["resource_censored"]["complete_capacity_proof"] is False


def test_aggregate_and_manifest_bounds_preserve_partial_proofs_and_files(
    tmp_path: Path,
    transport_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = (_NODES[0],)
    transport = _transport(tmp_path, transport_module, nodes)
    case = _case_folder(transport, nodes[0])
    _payload(case, nodes[0])
    folder, request = _worker(transport.root, transport.environment, nodes[0], 1, nodes)
    before = {path.name: path.read_bytes() for path in case.iterdir()}
    monkeypatch.setattr(transport_module, "_TOTAL_LIMIT", 4096)
    transport.closed_worker(folder, request, 0)
    transport.finish("PASS")
    assert transport.record["status"] == "UNVERIFIED"
    assert transport.record["cases"][nodes[0]]["files"]["receipt.json"][
        "sha256"
    ] == _sha(before["receipt.json"])
    assert {path.name: path.read_bytes() for path in case.iterdir()} == before
    preserved = transport.path.read_bytes()
    monkeypatch.setattr(transport_module, "_MANIFEST_LIMIT", 16)
    with pytest.raises(ValueError, match="manifest exceeds"):
        transport._save()
    assert transport.path.read_bytes() == preserved


def test_directory_entry_bound_and_unknown_owner_never_earn_completion(
    tmp_path: Path,
    transport_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _transport(
        tmp_path, transport_module, (_OWNER + "::test_unmapped_owner_case",)
    )
    transport.finish("PASS")
    assert transport.record["status"] == "UNVERIFIED"
    folder = tmp_path / "too-many"
    folder.mkdir()
    for number in range(4):
        (folder / str(number)).write_bytes(b"retained")
    monkeypatch.setattr(transport_module, "_ENTRY_LIMIT", 2)
    with pytest.raises(ValueError, match="entry bound"):
        transport_module._children(folder)
    assert len(list(folder.iterdir())) == 4


def test_streaming_transport_never_imports_a_framework_or_materializes_a_model(
    tmp_path: Path,
    transport_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__
    attempts = []

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {
            "jax",
            "jaxlib",
            "numpy",
            "onnx",
            "onnxruntime",
            "flax",
        }:
            attempts.append(name)
            raise AssertionError("framework import during passive transport")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    nodes = (_NODES[0],)
    transport = _transport(tmp_path, transport_module, nodes)
    _payload(_case_folder(transport, nodes[0]), nodes[0])
    folder, request = _worker(transport.root, transport.environment, nodes[0], 1, nodes)
    transport.closed_worker(folder, request, 0)
    transport.finish("PASS")
    assert transport.record["status"] == "SUBSET_TRANSPORT_ONLY" and attempts == []


def test_actual_tiny_canonical_workers_receive_one_identical_owned_env_without_parent_mutation(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_tiny.py").write_text(
        "def test_one():\n    assert True\n\ndef test_two():\n    assert True\n"
    )
    root = tmp_path / "receipts"
    environment = dict(os.environ)
    environment.pop(_ENV, None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    before = dict(os.environ)
    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts/run_pytest_batches.py"),
            "--batch-size",
            "1",
            "--output-dir",
            str(root),
            "--test-path",
            str(suite),
            "--",
            "-v",
            "--tb=short",
            "-x",
        ],
        cwd=suite,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert dict(os.environ) == before
    folders = [root / "collection", root / "batch-0001", root / "batch-0002"]
    values = [
        _load(folder / "receipt.json")["startup_context"]["environment"][_ENV]
        for folder in folders
    ]
    assert values == [str(root / "native-model-artifacts")] * 3
    manifest = _load(root / "native-model-artifact-transport.json")
    original_run = _load(root / "run.json")
    assert original_run["status"] == "PASS" and len(original_run["completed"]) == 2
    if original_run["checked_out_commit"] == "unavailable":
        assert (
            manifest["status"] == "UNVERIFIED"
        ), "metadata inability must not rewrite the two original PASS outcomes"
    assert manifest["environment_mode"] == "RUN_OWNED"
    assert all(
        (folder / "native-model-artifact-transport.json").is_file()
        for folder in folders
    )


def test_explicit_diagnostic_child_environment_is_not_overridden(
    tmp_path: Path,
) -> None:
    spec = spec_from_file_location(
        "_runner_environment_under_test", _ROOT / "scripts/run_pytest_batches.py"
    )
    assert spec is not None and spec.loader is not None
    runner = module_from_spec(spec)
    spec.loader.exec_module(runner)
    folder = tmp_path / "batch-0001"
    environment = dict(os.environ)
    environment[_ENV] = "explicit-diagnostic-intent"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    request = {
        "kind": "collection",
        "pytest_args": [str(tmp_path / "test_empty.py"), "--collect-only"],
    }
    (tmp_path / "test_empty.py").write_text("def test_kept():\n    assert True\n")
    original = dict(environment)
    run = {"processes": [], "status": "RUNNING"}
    result = runner._child(
        folder, request, run, tmp_path, environment=environment, timeout_seconds=60
    )
    assert result == 0 and environment == original
    assert (
        _load(folder / "receipt.json")["startup_context"]["environment"][_ENV]
        == "explicit-diagnostic-intent"
    )


def _isolated_runner(
    tmp_path: Path,
    helper_source: str | None,
    *,
    original_pass: bool = True,
    evidence_value: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any], Path, dict[str, Any]]:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / "run_pytest_batches.py"
    runner.write_bytes((_ROOT / "scripts/run_pytest_batches.py").read_bytes())
    if helper_source is not None:
        (scripts / "_native_model_artifact_transport.py").write_text(helper_source)
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_original.py").write_text(
        "from pathlib import Path\n"
        "def test_original():\n"
        "    Path('original-calls').write_text('one original call')\n"
        f"    assert {original_pass!r}, 'original tiny failure'\n"
    )
    # Observe the existing child boundary, then invoke the actual child. This
    # fixture does not replace worker collection, phases, outcomes or _validate.
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "import importlib.util,json,os,sys\n"
        "from pathlib import Path\n"
        f"spec=importlib.util.spec_from_file_location('_isolated_runner', {str(runner)!r})\n"
        "runner=importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(runner)\n"
        "original_environment=dict(os.environ)\n"
        "original_child=runner._child\n"
        "calls=[]\n"
        "def child(*args,**kwargs):\n"
        "    calls.append({'environment_argument_present':'environment' in kwargs,\n"
        "                  'parent_environment_held':dict(os.environ)==original_environment})\n"
        "    return original_child(*args,**kwargs)\n"
        "runner._child=child\n"
        "result=runner._main()\n"
        f"Path({str(tmp_path / 'boundary.json')!r}).write_text(json.dumps(\n"
        "    {'calls':calls,'parent_environment_held':dict(os.environ)==original_environment}))\n"
        "raise SystemExit(result)\n"
    )
    root = tmp_path / "receipts"
    environment = dict(os.environ)
    environment.pop(_ENV, None)
    if evidence_value is not None:
        environment[_ENV] = evidence_value
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["TRANSPORT_FIXTURE_MARKER"] = "original inheritance"
    before = dict(os.environ)
    result = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--output-dir",
            str(root),
            "--test-path",
            str(suite),
            "--",
            "-v",
            "--tb=short",
            "-x",
        ],
        cwd=suite,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert dict(os.environ) == before
    return result, _load(root / "run.json"), root, _load(tmp_path / "boundary.json")


@pytest.mark.parametrize("original_pass", [False, True], ids=["fail", "pass"])
@pytest.mark.parametrize(
    "fault,evidence_value",
    [
        ("missing-helper", None),
        ("missing-helper", "caller-supplied-external-intent"),
        ("import-error", None),
        ("setup-error", "caller-supplied-external-intent"),
        ("missing-original-owner", None),
    ],
    ids=["missing", "missing-explicit-env", "import", "setup", "missing-owner"],
)
def test_unavailable_optional_transport_preserves_real_original_calls_and_inheritance(
    tmp_path: Path,
    original_pass: bool,
    fault: str,
    evidence_value: str | None,
) -> None:
    helper = None
    if fault == "import-error":
        helper = "raise RuntimeError('fixture import failure')\n"
    elif fault == "setup-error":
        helper = (
            "class ArtifactResourceCensor(RuntimeError): pass\n"
            "class NativeModelTransport:\n"
            "    def __init__(self,root,checkout,environment):\n"
            "        self.environment=dict(environment)\n"
            f"        self.environment[{_ENV!r}]='must never reach original workers'\n"
            "        evidence=root/'native-model-artifacts'\n"
            "        evidence.mkdir()\n"
            "        (evidence/'partial.bin').write_bytes(b'preserved partial bytes')\n"
            "        (root/'native-model-artifact-transport.json').write_bytes(b'preserved partial manifest')\n"
            "        raise ValueError('fixture setup unavailable')\n"
        )
    elif fault == "missing-original-owner":
        # The real stdlib helper has no original owner beside this copied runner.
        helper = (_ROOT / "scripts/_native_model_artifact_transport.py").read_text()
    result, run, root, boundary = _isolated_runner(
        tmp_path,
        helper,
        original_pass=original_pass,
        evidence_value=evidence_value,
    )
    assert (result.returncode == 0) is original_pass, result.stdout + result.stderr
    assert run["status"] == ("PASS" if original_pass else "FAIL")
    assert len(run["nodeids"]) == 1 and len(run["processes"]) == 2
    assert run["completed"] == (run["nodeids"] if original_pass else [])
    assert (tmp_path / "suite/original-calls").read_text() == "one original call"
    assert boundary == {
        "calls": [
            {"environment_argument_present": False, "parent_environment_held": True}
        ]
        * 2,
        "parent_environment_held": True,
    }
    manifest = _load(root / run["native_model_artifact_transport"])
    assert manifest["status"] == "UNVERIFIED"
    assert manifest["setup_status"] == "UNAVAILABLE"
    assert manifest["environment_mode"] == "ORIGINAL_INHERITED"
    assert manifest["environment_value"] == evidence_value
    assert manifest["parent_environment_mutated"] is False
    assert manifest["original_run_status"] == run["status"]
    assert manifest["full_collection_ids"] == run["nodeids"]
    assert manifest["completed_ids"] == run["completed"]
    assert manifest["numerical_acceptance"] == "NOT_ASSERTED"
    assert manifest["full_B_to_F_acceptance"] == "UNVERIFIED"
    assert manifest["errors"][0]["stage"] == "setup"
    assert run["native_model_transport_errors"] == [
        {"stage": "setup", "type": manifest["errors"][0]["type"]}
    ]
    for name in ("collection", "batch-0001"):
        receipt = _load(root / name / "receipt.json")
        assert receipt["startup_context"]["environment"].get(_ENV) == evidence_value
    receipt = _load(root / "batch-0001/receipt.json")
    nodeid = run["nodeids"][0]
    expected = [
        {"when": "setup", "outcome": "passed"},
        {"when": "call", "outcome": "passed" if original_pass else "failed"},
        {"when": "teardown", "outcome": "passed"},
    ]
    assert [
        {"when": phase["when"], "outcome": phase["outcome"]}
        for phase in receipt["phases"][nodeid]
    ] == expected
    events = [
        json.loads(line)
        for line in (root / "batch-0001/events.jsonl").read_text().splitlines()
    ]
    assert events == [
        dict(kind="phase", nodeid=nodeid, **phase)
        for phase in receipt["phases"][nodeid]
    ]
    cases = ET.parse(root / "batch-0001/junit.xml").findall(".//testcase")
    assert len(cases) == 1
    assert bool(cases[0].findall("failure")) is (not original_pass)
    if not original_pass:
        assert "original tiny failure" in receipt["phases"][nodeid][1]["longrepr"]
    if fault == "setup-error":
        assert (
            root / "native-model-artifacts/partial.bin"
        ).read_bytes() == b"preserved partial bytes"
        assert (
            root / "native-model-artifact-transport.json"
        ).read_bytes() == b"preserved partial manifest"


@pytest.mark.parametrize(
    "fault,exception_type",
    [
        ("censor", "ArtifactResourceCensor"),
        ("enospc", "OSError"),
        ("edquot", "OSError"),
        ("enomem", "OSError"),
        ("memory", "MemoryError"),
    ],
)
def test_resource_setup_refusal_never_falls_back_or_starts_original_workers(
    tmp_path: Path, fault: str, exception_type: str
) -> None:
    expression = {
        "censor": "ArtifactResourceCensor('fixture resource refusal')",
        "enospc": "OSError(28,'fixture no space')",
        "edquot": "OSError(122,'fixture quota')",
        "enomem": "OSError(12,'fixture no memory')",
        "memory": "MemoryError('fixture memory refusal')",
    }[fault]
    helper = (
        "class ArtifactResourceCensor(RuntimeError): pass\n"
        "class NativeModelTransport:\n"
        "    def __init__(self,*args):\n"
        f"        raise {expression}\n"
    )
    result, run, root, boundary = _isolated_runner(tmp_path, helper)
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["failure"]["type"] == exception_type
    assert run["processes"] == [] and run["completed"] == []
    assert boundary == {"calls": [], "parent_environment_held": True}
    assert not (tmp_path / "suite/original-calls").exists()
    manifest = _load(root / run["native_model_artifact_transport"])
    assert manifest["status"] == "UNVERIFIED"
    assert manifest["setup_status"] == "RESOURCE_CENSORED"
    assert manifest["errors"][0]["type"] == exception_type
    assert manifest["original_run_status"] == "FAIL"


@pytest.mark.parametrize("exception_type", ["KeyboardInterrupt", "SystemExit"])
def test_setup_control_flow_exceptions_are_not_optional_transport_fallback(
    tmp_path: Path, exception_type: str
) -> None:
    result, run, root, boundary = _isolated_runner(
        tmp_path, f"raise {exception_type}('fixture control flow')\n"
    )
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["failure"]["type"] == exception_type
    assert run["processes"] == [] and run["completed"] == []
    assert boundary == {"calls": [], "parent_environment_held": True}
    assert not (tmp_path / "suite/original-calls").exists()
    assert not (root / "native-model-artifact-transport-setup-error.json").exists()


@pytest.mark.parametrize("stage", ["constructor", "before-batch"])
def test_real_managed_transport_disk_floor_is_terminal_without_capacity_bypass(
    tmp_path: Path, stage: str
) -> None:
    path = _ROOT / "scripts/_native_model_artifact_transport.py"
    helper = (
        "from importlib.util import module_from_spec,spec_from_file_location\n"
        f"spec=spec_from_file_location('_real_transport_capacity_fixture',{str(path)!r})\n"
        "real=module_from_spec(spec)\n"
        "spec.loader.exec_module(real)\n"
        "ArtifactResourceCensor=real.ArtifactResourceCensor\n"
    )
    if stage == "constructor":
        helper += (
            "real._free=lambda path:0\n"
            "NativeModelTransport=real.NativeModelTransport\n"
        )
    else:
        helper += (
            "real._free=lambda path:1024**4\n"
            "class NativeModelTransport(real.NativeModelTransport):\n"
            "    def before_worker(self,folder,request):\n"
            "        if request['kind']=='batch':\n"
            "            real._free=lambda path:real._FLOOR+1\n"
            "        return super().before_worker(folder,request)\n"
        )
    result, run, root, boundary = _isolated_runner(tmp_path, helper)
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["failure"]["type"] == "ArtifactResourceCensor"
    assert run["completed"] == []
    assert boundary["parent_environment_held"] is True
    assert not (tmp_path / "suite/original-calls").exists()
    assert not (root / "batch-0001").exists()
    if stage == "constructor":
        assert run["processes"] == [] and boundary["calls"] == []
        manifest = _load(root / run["native_model_artifact_transport"])
        assert manifest["status"] == "UNVERIFIED"
        assert manifest["setup_status"] == "RESOURCE_CENSORED"
    else:
        assert len(run["nodeids"]) == 1 and len(run["processes"]) == 1
        assert boundary["calls"] == [
            {"environment_argument_present": True, "parent_environment_held": True}
        ]


@pytest.mark.parametrize("original_pass", [False, True], ids=["fail", "pass"])
@pytest.mark.parametrize(
    "stage,fault,exception_type",
    [
        ("closed", "MemoryError('closed fixture memory refusal')", "MemoryError"),
        ("closed", "OSError(28,'closed fixture no space')", "OSError"),
        (
            "finish",
            "ArtifactResourceCensor('finish fixture disk floor')",
            "ArtifactResourceCensor",
        ),
    ],
    ids=["closed-memory", "closed-enospc", "finish-censor"],
)
def test_closed_or_finish_resource_failure_keeps_raw_children_and_stops_orchestration(
    tmp_path: Path,
    original_pass: bool,
    stage: str,
    fault: str,
    exception_type: str,
) -> None:
    # A second tiny item uses the existing native-owner isolation rule so the
    # genuine planner creates a later worker. It is not a native/model fixture.
    next_source = (
        "from pathlib import Path\n"
        "def test_next():\n"
        "    Path('next-call').write_text('later original call')\n"
    )
    helper = (
        "import hashlib,json\n"
        "class ArtifactResourceCensor(RuntimeError): pass\n"
        "class NativeModelTransport:\n"
        "    def __init__(self,root,checkout,environment):\n"
        "        self.root=root\n"
        "        self.environment=dict(environment)\n"
        "        self.raw={}\n"
        f"        (root.parent/'suite/test_source_model_native_bindings_exports.py').write_text({next_source!r})\n"
        "        (root/'native-model-artifact-transport.json').write_text(json.dumps(\n"
        "            {'status':'PARTIAL','scope':'synthetic lifecycle fault fixture'}))\n"
        "    def before_worker(self,folder,request): pass\n"
        "    def collection(self,nodeids): pass\n"
        "    def closed_worker(self,folder,request,result):\n"
        "        if request['kind']!='batch': return\n"
        "        self.raw[folder.name]={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()\n"
        "            for name in ('request.json','receipt.json','events.jsonl','junit.xml','output.log')}\n"
        "        (self.root/'raw-before-resource.json').write_text(json.dumps(self.raw))\n"
        + (f"        raise {fault}\n" if stage == "closed" else "")
        + "    def finish(self,status):\n"
        "        (self.root/'original-status-before-finish.txt').write_text(status)\n"
        + (f"        raise {fault}\n" if stage == "finish" else "")
    )
    result, run, root, boundary = _isolated_runner(
        tmp_path, helper, original_pass=original_pass
    )
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert len(run["nodeids"]) == 2 and len(run["batches"]) == 2
    assert (tmp_path / "suite/original-calls").read_text() == "one original call"
    later_ran = stage == "finish" and original_pass
    assert (tmp_path / "suite/next-call").exists() is later_ran
    assert (root / "batch-0002").exists() is later_ran
    assert len(run["processes"]) == (3 if later_ran else 2)
    assert run["processes"][1]["returncode"] == (0 if original_pass else 1)
    assert boundary["parent_environment_held"] is True
    raw_before = _load(root / "raw-before-resource.json")
    assert list(raw_before) == (
        ["batch-0001", "batch-0002"] if later_ran else ["batch-0001"]
    )
    for folder, files in raw_before.items():
        assert {
            name: _sha((root / folder / name).read_bytes()) for name in files
        } == files
    receipt = _load(root / "batch-0001/receipt.json")
    nodeid = run["nodeids"][0]
    assert receipt["assigned_ids"] == receipt["completed"] == [nodeid]
    assert [(p["when"], p["outcome"]) for p in receipt["phases"][nodeid]] == [
        ("setup", "passed"),
        ("call", "passed" if original_pass else "failed"),
        ("teardown", "passed"),
    ]
    events = [
        json.loads(line)
        for line in (root / "batch-0001/events.jsonl").read_text().splitlines()
    ]
    assert events == [
        dict(kind="phase", nodeid=nodeid, **p) for p in receipt["phases"][nodeid]
    ]
    case = ET.parse(root / "batch-0001/junit.xml").findall(".//testcase")
    assert len(case) == 1 and bool(case[0].findall("failure")) is (not original_pass)
    errors = run["native_model_transport_resource_errors"]
    assert len(errors) == 1 and errors[0]["type"] == exception_type
    if stage == "closed":
        assert errors[0]["stage"] == "batch-0001:closed"
        assert errors[0]["original_worker_returncode"] == (0 if original_pass else 1)
        assert errors[0]["original_worker_exception_type"] is None
        assert run["completed"] == []
        assert (root / "original-status-before-finish.txt").read_text() == "FAIL"
    else:
        original_status = "PASS" if original_pass else "FAIL"
        assert errors[0]["stage"] == "finish"
        assert errors[0]["original_pytest_status"] == original_status
        assert errors[0]["original_completed_ids"] == run["completed"]
        assert run["completed"] == (run["nodeids"] if original_pass else [])
        assert (
            root / "original-status-before-finish.txt"
        ).read_text() == original_status
    if original_pass:
        assert run["failure"]["type"] == exception_type
    else:
        assert run["failure"]["type"] == "ValueError"
        assert (
            "batch-0001 process did not finish successfully"
            in run["failure"]["message"]
        )
    assert _load(root / "native-model-artifact-transport.json")["status"] == "PARTIAL"


@pytest.mark.parametrize(
    "stage",
    [
        "file-after-path",
        "case-receipt",
        "case-payload",
        "closed-file",
        "closed-phase",
        "closed-case",
        "checkpoint-floor",
        "finish",
    ],
)
def test_deep_resource_faults_escape_each_diagnostic_layer_with_original_bytes_held(
    tmp_path: Path,
    transport_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    node = _NODES[-1] if stage == "checkpoint-floor" else _NODES[0]
    # Establish that these complete, tiny, synthetic packets qualify before
    # introducing a fault; they never claim actual model/array execution.
    good_root = tmp_path / "good"
    good_root.mkdir()
    good = _transport(good_root, transport_module, (node,))
    if node == _NODES[0]:
        _payload(_case_folder(good, node), node)
    good_folder, good_request = _worker(good.root, good.environment, node, 1, (node,))
    good.closed_worker(good_folder, good_request, 0)
    good.finish("PASS")
    assert good.record["status"] == "SUBSET_TRANSPORT_ONLY"
    fault_root = tmp_path / "fault"
    fault_root.mkdir()
    transport = _transport(fault_root, transport_module, (node,))
    case_folder = _case_folder(transport, node) if node == _NODES[0] else None
    if case_folder is not None:
        _payload(case_folder, node)
    folder, request = _worker(transport.root, transport.environment, node, 1, (node,))
    paths = list(folder.iterdir()) + (
        [] if case_folder is None else list(case_folder.iterdir())
    )
    before = {path: path.read_bytes() for path in paths}
    error: Exception = {
        "file-after-path": OSError(12, "fixture path-stat memory refusal"),
        "case-receipt": MemoryError("fixture receipt memory refusal"),
        "case-payload": OSError(28, "fixture payload no space"),
        "closed-file": MemoryError("fixture worker-file memory refusal"),
        "closed-phase": transport_module.ArtifactResourceCensor(
            "fixture phase resource refusal"
        ),
        "closed-case": OSError(122, "fixture case quota refusal"),
        "checkpoint-floor": transport_module.ArtifactResourceCensor("unused marker"),
        "finish": MemoryError("fixture finish memory refusal"),
    }[stage]

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise error

    operation = transport.closed_worker
    arguments = (folder, request, 0)
    if stage == "file-after-path":
        original_stat = transport_module._path_stat
        visits = []
        target = folder / "receipt.json"

        def after_read(path: Path) -> Any:
            if path == target:
                visits.append(path)
                if len(visits) == 2:
                    raise error
            return original_stat(path)

        monkeypatch.setattr(transport_module, "_path_stat", after_read)
        operation, arguments = transport_module._file, (target,)
    elif stage == "case-receipt":
        monkeypatch.setattr(transport, "_read_json", refuse)
        operation, arguments = transport._case, (node, [], 0)
    elif stage in ("case-payload", "closed-file"):
        original_file = transport_module._file
        target = (
            case_folder / "model.onnx"
            if stage == "case-payload"
            else folder / "events.jsonl"
        )

        def file_fault(path: Path, **kwargs: Any) -> Any:
            if path == target:
                raise error
            return original_file(path, **kwargs)

        monkeypatch.setattr(transport_module, "_file", file_fault)
        if stage == "case-payload":
            operation, arguments = transport._case, (node, [], 0)
    elif stage == "closed-phase":
        monkeypatch.setattr(transport_module.ET, "fromstring", refuse)
    elif stage == "closed-case":
        monkeypatch.setattr(transport, "_case", refuse)
    elif stage == "checkpoint-floor":
        monkeypatch.setattr(
            transport_module, "_free", lambda path: transport_module._FLOOR + 1
        )
    else:
        transport.closed_worker(folder, request, 0)
        monkeypatch.setattr(transport_module, "_file", refuse)
        operation, arguments = transport.finish, ("PASS",)
    with pytest.raises(type(error)) as captured:
        operation(*arguments)
    if stage == "checkpoint-floor":
        assert str(captured.value) == "closed-worker transport manifest disk floor"
    else:
        assert captured.value is error
    assert transport_module._resource_error(captured.value) is True
    assert {path: path.read_bytes() for path in paths} == before
    if stage in ("closed-file", "closed-phase", "closed-case", "checkpoint-floor"):
        assert transport.record["workers"][0]["original_returncode"] == 0
        assert transport.record["workers"][0]["files"]["request.json"][
            "sha256"
        ] == _sha(before[folder / "request.json"])
    if stage in ("case-receipt", "case-payload"):
        assert transport.record["cases"][node]["status"] == "UNVERIFIED"


@pytest.mark.parametrize(
    "exception_type", ["ValueError", "KeyboardInterrupt", "SystemExit"]
)
def test_active_original_exception_has_priority_over_closed_and_finish_resource_faults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: str,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner_path = scripts / "run_pytest_batches.py"
    runner_path.write_bytes((_ROOT / "scripts/run_pytest_batches.py").read_bytes())
    (scripts / "_native_model_artifact_transport.py").write_text(
        "class ArtifactResourceCensor(RuntimeError): pass\n"
        "class NativeModelTransport:\n"
        "    def __init__(self,root,checkout,environment): self.environment=dict(environment)\n"
        "    def before_worker(self,*args): pass\n"
        "    def closed_worker(self,*args): raise MemoryError('fixture closed refusal')\n"
        "    def finish(self,*args): raise ArtifactResourceCensor('fixture finish refusal')\n"
    )
    spec = spec_from_file_location("_original_exception_priority_fixture", runner_path)
    assert spec is not None and spec.loader is not None
    runner = module_from_spec(spec)
    spec.loader.exec_module(runner)
    original = {
        "ValueError": ValueError("original child exception"),
        "KeyboardInterrupt": KeyboardInterrupt("original child interrupt"),
        "SystemExit": SystemExit("original child exit"),
    }[exception_type]
    calls = []

    def original_failure(*args: Any, **kwargs: Any) -> Any:
        calls.append("one original invocation")
        raise original

    monkeypatch.setattr(runner, "_child", original_failure)
    root = tmp_path / "receipts"
    monkeypatch.setattr(sys, "argv", [str(runner_path), "--output-dir", str(root)])
    assert runner._main() == 1
    run = _load(root / "run.json")
    assert calls == ["one original invocation"]
    assert run["status"] == "FAIL" and run["failure"] == {
        "type": exception_type,
        "message": str(original),
    }
    assert run["completed"] == [] and run["processes"] == []
    errors = run["native_model_transport_resource_errors"]
    assert [(e["stage"], e["type"]) for e in errors] == [
        ("collection:closed", "MemoryError"),
        ("finish", "ArtifactResourceCensor"),
    ]
    assert errors[0]["original_worker_exception_type"] == exception_type
    assert errors[1]["original_pytest_status"] == "FAIL"
