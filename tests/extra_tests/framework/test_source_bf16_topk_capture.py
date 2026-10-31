# tests/extra_tests/framework/test_source_bf16_topk_capture.py

"""Bounded controls for the one original BF16 TopK capture lifecycle."""

import ast
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import pytest

from tests.extra_tests.framework import test_pytest_batches as batches

_ROOT = Path(__file__).resolve().parents[3]
_HELPER = _ROOT / "scripts/_source_bf16_topk_capture.py"


def _load():
    spec = importlib.util.spec_from_file_location("_bf16_capture_control", _HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Array:
    def __init__(self, batch, width, dtype, raw=None):
        self.shape = (batch, 8, width)
        self.dtype = SimpleNamespace(name=dtype)
        self.nbytes = batch * 8 * width * (4 if dtype == "int32" else 2)
        self.raw = bytes(self.nbytes) if raw is None else raw

    def tobytes(self):
        return self.raw


def _archive(batch, *, dtype="<V2", duplicate=False, bad_shape=False):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for index, descriptor in enumerate((dtype, "<i4")):
            shape = (batch, 8, 3 if bad_shape else 2)
            header = repr(
                dict(descr=descriptor, fortran_order=False, shape=shape)
            ).encode()
            header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
            raw = bytes(batch * 8 * shape[-1] * (2 if index == 0 else 4))
            name = "leaf_0.npy" if duplicate else f"leaf_{index}.npy"
            archive.writestr(
                name,
                b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + raw,
            )
    return stream.getvalue()


@pytest.fixture
def case(tmp_path, monkeypatch, request):
    module = _load()
    monkeypatch.setattr(module, "NODE_ID", request.node.nodeid)
    root = tmp_path / "source"
    root.mkdir()
    source = root / module.OWNER
    source.parent.mkdir(parents=True)
    compiler = ModuleType("jax._src.compiler")
    compiler.calls = []
    compiler.returned = SimpleNamespace(hlo_modules=lambda: [])
    compiler.raise_original = None
    compiler_source = (
        "def compile_or_get_cached(backend, computation, devices, compile_options, "
        "host_callbacks, executable_devices, pgle_profiler=None):\n"
        "    calls.append((backend, computation, devices, compile_options, host_callbacks, executable_devices, pgle_profiler))\n"
        "    if raise_original is not None:\n"
        "        raise raise_original\n"
        "    return returned\n"
    )
    compiler_path = root / "compiler.py"
    compiler_path.write_text(compiler_source)
    compiler.__file__ = str(compiler_path)
    exec(compile(compiler_source, str(compiler_path), "exec"), vars(compiler))
    pxla, lax = ModuleType("jax._src.interpreters.pxla"), ModuleType("jax._src.lax.lax")
    pxla.compiler = compiler
    for native, name in ((pxla, "pxla"), (lax, "lax")):
        path = root / f"{name}.py"
        text = "def _top_k_lower():\n    return None\n"
        path.write_text(text)
        native.__file__ = str(path)
        exec(compile(text, str(path), "exec"), vars(native))
    owner = ModuleType("original_topk_control")
    owner.compiler = compiler
    owner.outputs = (_Array(2, 2, "bfloat16"), _Array(2, 2, "int32"))
    owner.fail_portable = False
    owner.repeat = 1
    text = (
        "def _retain():\n    return None\n"
        "def _portable_relation(actual, data):\n"
        "    position = (0, 4)\n    row = [1, 127, 32769, 32895]\n"
        "    keys = [32769, 32895, 32766, 32640]\n"
        "    chosen = [0, 1]\n    selected_keys = [32769, 32895]\n"
        "    raise AssertionError('original portable failure')\n"
        "def _bf16_native(data, inputs, directory, arguments, keywords):\n"
        "    compiled = object()\n    operation = compiled\n    label = 'fresh'\n"
        "    references = []\n"
        "    for _ in range(repeat):\n"
        "        result = compiler.compile_or_get_cached(*arguments, **keywords)\n"
        "    actual = outputs\n"
        "    if fail_portable:\n        _portable_relation(actual, data)\n"
        "    references.append(actual)\n    return result\n"
        f"def {module.TEST}(data, directory, arguments, keywords):\n"
        "    scope = 'inline'\n    contains_nan = False\n    size = data.shape[0]\n"
        "    selected = data\n    inputs = (data,)\n    original_words = (data.tobytes(),)\n"
        "    return _bf16_native(data, inputs, directory, arguments, keywords)\n"
    )
    source.write_text(text)
    owner.__file__ = str(source)
    exec(compile(text, str(source), "exec"), vars(owner))
    native_path = root / "native.so"
    native_path.write_bytes(b"bounded original native identity")
    native_module = ModuleType("numpy._core._multiarray_umath")
    native_module.__file__ = str(native_path)
    numpy = ModuleType("numpy")
    numpy.ndarray = _Array
    jax, jaxlib = ModuleType("jax"), ModuleType("jaxlib")
    jaxlib.__file__ = str(root / "jaxlib.py")
    modules = {
        native.__name__: native
        for native in (compiler, pxla, lax, numpy, jax, jaxlib, native_module)
    }
    for name, native in modules.items():
        monkeypatch.setitem(sys.modules, name, native)
    monkeypatch.delitem(sys.modules, "jaxlib._jax", raising=False)
    monkeypatch.delitem(sys.modules, "jaxlib._xla", raising=False)
    monkeypatch.delitem(sys.modules, "jaxlib.xla_extension", raising=False)
    monkeypatch.delitem(sys.modules, "numpy.core._multiarray_umath", raising=False)
    monkeypatch.delitem(sys.modules, "ml_dtypes._ml_dtypes_ext", raising=False)
    monkeypatch.setattr(module, "_ROOT", root)
    monkeypatch.setattr(module, "_SOURCES", (module.OWNER,))
    monkeypatch.setattr(
        module, "_OWNER_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        module,
        "_COMPILER_SCHEMAS",
        {
            hashlib.sha256(compiler_path.read_bytes()).hexdigest(): tuple(
                hashlib.sha256((root / f"{name}.py").read_bytes()).hexdigest()
                for name in ("pxla", "lax")
            ),
        },
    )

    class Item:
        pass

    item = Item()
    item.nodeid, item.module, item.obj = module.NODE_ID, owner, vars(owner)[module.TEST]
    fixture = tmp_path / "pytest-tmp"
    fixture.mkdir()
    item.funcargs = {"tmp_path": fixture}
    pytest_module = ModuleType("pytest")
    pytest_module.Function = Item
    original_module = module._module
    monkeypatch.setattr(
        module,
        "_module",
        lambda name: pytest_module if name == "pytest" else original_module(name),
    )
    folder = tmp_path / "batch-0001"
    folder.mkdir()
    context = {
        "status": "OBSERVED",
        "errors": [],
        "environment": {
            "PYTEST_CURRENT_TEST": module.os.environ.get("PYTEST_CURRENT_TEST")
        },
    }
    capture = module.start_observation(folder, item, context)
    data = _Array(2, 4, "bfloat16")
    backend = SimpleNamespace(platform="cpu")
    computation = SimpleNamespace(
        operation=SimpleNamespace(attributes={"sym_name": "actual_control"})
    )
    arguments = (backend, computation, object(), object(), object(), object())
    result = SimpleNamespace(
        module=module,
        capture=capture,
        context=context,
        compiler=compiler,
        owner=owner,
        item=item,
        root=root,
        folder=folder,
        data=data,
        arguments=arguments,
        native_path=native_path,
        fixture=fixture,
    )
    yield result
    capture.close()


@pytest.mark.parametrize("keyword", (False, True))
def test_exact_original_compiler_arguments_return_and_one_delegation(case, keyword):
    capture, compiler = case.capture, case.compiler
    assert capture.installed and not capture.report["errors"]
    original = capture.original
    args = () if keyword else case.arguments
    kwargs = (
        dict(zip(case.module._FORMALS, (*case.arguments, None), strict=True))
        if keyword
        else {}
    )
    result = case.item.obj(case.data, case.fixture / "native-default/B2", args, kwargs)
    assert result is compiler.returned and len(compiler.calls) == 1
    assert all(
        actual is expected
        for actual, expected in zip(
            compiler.calls[0], (*case.arguments, None), strict=True
        )
    )
    assert capture.events[0]["executable"] is result
    assert capture.events[0]["native_frame_binding"]["role"] == "fresh"
    assert capture.events[0]["native_frame_binding"]["batch"] == 2
    capture.finish()
    assert compiler.compile_or_get_cached is original
    capture.close()
    assert not capture.events and not capture.hosts and capture.original_report is None


def test_original_compiler_exception_identity_propagates_and_restores(case):
    sentinel = RuntimeError("original compiler exception")
    case.compiler.raise_original = sentinel
    with pytest.raises(RuntimeError) as caught:
        case.item.obj(case.data, case.fixture / "native-default/B2", case.arguments, {})
    assert caught.value is sentinel and len(case.compiler.calls) == 1
    assert not case.capture.events
    case.capture.finish()
    assert case.compiler.compile_or_get_cached is case.capture.original


@pytest.mark.parametrize(
    "fault", ("metadata", "overflow", "foreign-hook", "source", "code", "input-words")
)
def test_unknown_observation_never_changes_original_result(case, fault, monkeypatch):
    if fault == "metadata":
        monkeypatch.setattr(
            case.capture,
            "_compiler_frames",
            lambda: (_ for _ in ()).throw(ValueError("metadata")),
        )
    if fault == "overflow":
        case.owner.repeat = 17
    returned = case.item.obj(
        case.data, case.fixture / "native-default/B2", case.arguments, {}
    )
    assert returned is case.compiler.returned
    assert len(case.compiler.calls) == (17 if fault == "overflow" else 1)
    if fault == "foreign-hook":

        def foreign(*args, **kwargs):
            return None

        case.compiler.compile_or_get_cached = foreign
    elif fault == "source":
        (case.root / case.module.OWNER).write_text("changed source")
    elif fault == "code":
        case.owner._retain.__code__ = (lambda: "changed code").__code__
    elif fault == "input-words":
        case.data.raw = b"\x01" * case.data.nbytes
        with pytest.raises(ValueError, match="input words"):
            case.capture._host_bytes()
        return
    case.capture.finish()
    assert (
        case.capture.report["errors"] and case.capture.report["status"] == "UNVERIFIED"
    )
    if fault == "foreign-hook":
        assert case.compiler.compile_or_get_cached is foreign
    else:
        assert case.compiler.compile_or_get_cached is case.capture.original


@pytest.fixture
def unknown_compiler_source_case(case, monkeypatch):
    case.capture.close()
    monkeypatch.setattr(case.module, "_COMPILER_SCHEMAS", {})
    case.restarted = case.module.start_observation(case.folder, case.item, case.context)
    yield case
    case.restarted.close()


def test_unknown_compiler_source_skips_patch_without_source_or_native_call(
    unknown_compiler_source_case,
):
    case = unknown_compiler_source_case
    capture = case.restarted
    assert not capture.installed and capture.report["status"] == "UNVERIFIED"
    assert (
        case.compiler.compile_or_get_cached is case.capture.original
        and not case.compiler.calls
    )
    capture.close()


@pytest.fixture
def missing_compiler_case(case, monkeypatch):
    case.capture.close()
    monkeypatch.delitem(sys.modules, "jax._src.compiler")
    case.restarted = case.module.start_observation(case.folder, case.item, case.context)
    yield case
    case.restarted.close()


def test_missing_loaded_compiler_never_imports_or_changes_original_call(
    missing_compiler_case,
):
    case = missing_compiler_case
    capture = case.restarted
    assert capture.report["errors"] and not capture.installed
    assert case.compiler.compile_or_get_cached is case.capture.original
    assert (
        case.item.obj(case.data, case.fixture / "native-default/B2", case.arguments, {})
        is case.compiler.returned
    )
    assert len(case.compiler.calls) == 1
    capture.close()


def test_uncategorized_original_compilation_has_no_guessed_native_role(case):
    assert (
        case.compiler.compile_or_get_cached(*case.arguments) is case.compiler.returned
    )
    assert len(case.compiler.calls) == 1
    assert case.capture.events[0]["native_frame_binding"] is None
    assert not case.capture.hosts


def test_source_frame_walk_bound_preserves_original_compiler_return(case):
    def nested(depth):
        if depth:
            return nested(depth - 1)
        return case.item.obj(
            case.data, case.fixture / "native-default/B2", case.arguments, {}
        )

    assert nested(65) is case.compiler.returned
    assert len(case.compiler.calls) == 1 and case.capture.report["errors"]
    assert case.capture.events[0]["executable"] is case.compiler.returned


@pytest.fixture
def installed_start_failure_case(case, monkeypatch):
    case.capture.close()
    original_start = case.module._Capture.start
    case.post_install_witness = []

    def fail_after_install(capture):
        original_start(capture)
        case.post_install_witness.append(
            {
                "installed": capture.installed,
                "owned_entry": case.compiler.compile_or_get_cached is capture.wrapper,
                "phase": capture._context()["pytest_current_test"],
            }
        )
        raise RuntimeError("diagnostic setup after install")

    monkeypatch.setattr(case.module._Capture, "start", fail_after_install)
    case.restarted = case.module.start_observation(case.folder, case.item, case.context)
    yield case
    case.restarted.close()


def test_setup_failure_after_install_restores_original_compiler(
    installed_start_failure_case,
):
    case = installed_start_failure_case
    capture = case.restarted
    assert case.post_install_witness == [
        {
            "installed": True,
            "owned_entry": True,
            "phase": f"{case.module.NODE_ID} (setup)",
        }
    ]
    assert case.compiler.compile_or_get_cached is case.capture.original
    assert capture.report["errors"] and not capture.installed
    assert capture.report.get("original_fences_held") is not True
    assert all(
        not entry["completed_call_boundary"]
        for entry in capture.report["pytest_current_test_journal"]
    )
    capture.close()
    assert not capture.events and not capture.hosts


@pytest.fixture
def preexisting_profile_case(case, monkeypatch):
    case.capture.close()
    case.profile_sentinel = object()
    monkeypatch.setattr(case.module.sys, "getprofile", lambda: case.profile_sentinel)
    case.restarted = case.module.start_observation(case.folder, case.item, case.context)
    yield case
    case.restarted.close()


def test_preexisting_profile_is_preserved_without_install(preexisting_profile_case):
    case = preexisting_profile_case
    capture = case.restarted
    assert case.module.sys.getprofile() is case.profile_sentinel
    assert case.compiler.compile_or_get_cached is case.capture.original
    assert not capture.installed and capture.report["errors"]
    capture.close()


def test_failed_original_traceback_binds_whole_actual_arrays_without_reexecution(case):
    from _pytest.reports import TestReport
    from _pytest.runner import CallInfo

    case.owner.fail_portable = True
    call = CallInfo.from_call(
        lambda: case.item.obj(
            case.data, case.fixture / "native-default/B2", case.arguments, {}
        ),
        "call",
    )
    case.capture.finish()
    report = TestReport(
        case.module.NODE_ID,
        (case.module.OWNER, 1, case.module.TEST),
        {},
        "failed",
        "original portable failure",
        "call",
    )
    case.capture.retain_report(call, report)
    assert case.capture.failed["binding"]["role"] == "fresh"
    assert case.capture.failed["binding"]["batch"] == 2
    assert case.capture.failed["actual"] is case.owner.outputs
    assert case.capture.failed["relation"]["selected_keys"] == [32769, 32895]
    assert len(case.compiler.calls) == 1
    case.capture.close()
    assert case.capture.failed is None


def _existing(case, batch, roles):
    directory = case.fixture / "native-default" / f"B{batch}"
    directory.mkdir(parents=True)
    entries = []
    for role in roles:
        payload = _archive(batch)
        (directory / f"{role}.npz").write_bytes(payload)
        (directory / f"{role}.json").write_text(
            json.dumps(dict(status="archived", stage=role, leaves=2))
        )
        entries.append(
            dict(
                native_role=role,
                retained_file=f"{role}.npz",
                retained_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    (directory / "native-word-observations.json").write_text(
        json.dumps(dict(contains_nan=False, native_roles=entries))
    )


def _phase(case, outcome="passed"):
    from _pytest.reports import TestReport

    case.capture.original_report = TestReport(
        case.module.NODE_ID,
        (case.module.OWNER, 1, case.module.TEST),
        {},
        outcome,
        "original portable failure" if outcome == "failed" else None,
        "call",
    )
    phase = dict(
        nodeid=case.module.NODE_ID,
        when="call",
        outcome=outcome,
        durable_phase_recorded=True,
    )
    if outcome == "failed":
        phase["longrepr"] = "original portable failure"
    phase["receipt_target_phases"] = [
        dict(when="setup", outcome="passed"),
        {
            name: phase[name]
            for name in ("when", "outcome", "longrepr")
            if name in phase
        },
    ]
    (case.folder / "receipt.json").write_text(
        json.dumps({"phases": {case.module.NODE_ID: phase["receipt_target_phases"]}})
    )
    (case.folder / "events.jsonl").write_text(
        json.dumps(dict(kind="phase", **phase)) + "\n"
    )
    return phase


@pytest.mark.parametrize("partial", (False, True))
def test_after_report_copies_exact_archives_host_bytes_and_actual_hlo_only_once(
    case, monkeypatch, partial
):
    from _pytest.runner import CallInfo

    capture = case.capture
    hlo_calls = []

    class Executable:
        def hlo_modules(self):
            receipt = json.loads((case.folder / "receipt.json").read_text())
            assert receipt["phases"][case.module.NODE_ID][-1]["when"] == "call"
            assert case.compiler.compile_or_get_cached is capture.original
            hlo_calls.append(self)
            return [SimpleNamespace(to_string=lambda: "actual original HLO control")]

    executable = Executable()
    case.compiler.returned = executable
    for batch in (2,) if partial else (2, 5):
        data = _Array(batch, 4, "bfloat16")
        case.owner.outputs = (_Array(batch, 2, "bfloat16"), _Array(batch, 2, "int32"))
        if partial:
            case.owner.fail_portable = True
            call = CallInfo.from_call(
                lambda: case.item.obj(
                    data, case.fixture / f"native-default/B{batch}", case.arguments, {}
                ),
                "call",
            )
            assert call.excinfo is not None
        else:
            assert (
                case.item.obj(
                    data, case.fixture / f"native-default/B{batch}", case.arguments, {}
                )
                is executable
            )
        host = capture.hosts[batch]
        if not partial:
            host["references"].extend([case.owner.outputs, case.owner.outputs])
        _existing(case, batch, ("fresh",) if partial else case.module._ROLES)
    capture.finish()
    assert not hlo_calls
    monkeypatch.setattr(capture, "_native_builds", lambda: None)
    phase = _phase(case, "failed" if partial else "passed")
    if partial:
        capture.retain_report(call, capture.original_report)
    capture.after_report(phase)
    output = case.folder / case.module.DIRECTORY
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == ("PARTIAL" if partial else "READY")
    assert (
        manifest["native_owners_cleared"] and not capture.events and not capture.hosts
    )
    assert hlo_calls == [executable]  # Shared actual returns are inspected once.
    assert len(case.compiler.calls) == (1 if partial else 2)
    for batch in (2,) if partial else (2, 5):
        roles = ("fresh",) if partial else case.module._ROLES
        for role in roles:
            original = case.fixture / f"native-default/B{batch}/{role}.npz"
            copied = output / f"native-default/B{batch}/{role}.npz"
            assert copied.read_bytes() == original.read_bytes()
            raw = json.loads(
                (output / f"native-default/B{batch}/{role}-raw.json").read_text()
            )
            assert raw["arrays"][0]["actual_dtype_name"] == "bfloat16"
            assert raw["arrays"][0]["storage_dtype"] == "<V2"
    assert all(
        not role["independent_compiler_return_claimed"] for role in manifest["roles"]
    )


def test_no_durable_report_never_inspects_actual_executable_or_copies_inputs(
    case, monkeypatch
):
    calls = []
    case.compiler.returned = SimpleNamespace(
        hlo_modules=lambda: calls.append("forbidden")
    )
    case.item.obj(case.data, case.fixture / "native-default/B2", case.arguments, {})
    case.capture.finish()
    monkeypatch.setattr(
        _Array,
        "tobytes",
        lambda self: (_ for _ in ()).throw(AssertionError("no early host copy")),
    )
    case.capture.after_report(
        dict(nodeid=case.module.NODE_ID, when="call", durable_phase_recorded=False)
    )
    assert not calls and not (case.folder / case.module.DIRECTORY).exists()
    assert case.capture.report["status"] == "UNVERIFIED" and case.capture.closed


@pytest.mark.parametrize("manifest_fault", (False, True))
def test_post_report_io_failure_clears_all_native_owners_and_preserves_report(
    case, monkeypatch, manifest_fault
):
    case.item.obj(case.data, case.fixture / "native-default/B2", case.arguments, {})
    _existing(case, 2, ("fresh",))
    case.capture.finish()
    phase = _phase(case)
    original_write = case.capture._write
    original_report = case.capture.original_report

    def failed_write(relative, payload, **kwargs):
        if (relative == "manifest.json") is manifest_fault:
            raise OSError("post-report write failure")
        return original_write(relative, payload, **kwargs)

    monkeypatch.setattr(case.capture, "_write", failed_write)
    case.capture.after_report(phase)
    assert original_report.outcome == "passed"
    assert case.capture.closed and not case.capture.events and not case.capture.hosts
    assert (
        case.capture.report["status"] == "UNVERIFIED" and case.capture.report["errors"]
    )


@pytest.mark.parametrize(
    "fault",
    (
        "digest",
        "swapped-batch",
        "foreign-role",
        "missing-frame",
        "missing-archive",
        "late-role",
    ),
)
def test_file_role_batch_frame_and_digest_corruption_remains_unverified(case, fault):
    capture = case.capture
    case.item.obj(case.data, case.fixture / "native-default/B2", case.arguments, {})
    output = case.folder / case.module.DIRECTORY
    output.mkdir()
    capture.output = output
    _existing(case, 2, ("fresh",))
    base = case.fixture / "native-default/B2"
    if fault == "digest":
        observation = json.loads((base / "native-word-observations.json").read_text())
        observation["native_roles"][0]["retained_sha256"] = "0" * 64
        (base / "native-word-observations.json").write_text(json.dumps(observation))
    elif fault == "swapped-batch":
        (base / "fresh.npz").write_bytes(_archive(5))
    elif fault == "foreign-role":
        (base / "fresh.json").write_text(
            json.dumps(dict(status="archived", stage="eager", leaves=2))
        )
    elif fault == "missing-archive":
        (base / "fresh.npz").unlink()
    elif fault == "late-role":
        _existing(case, 5, ("fresh",))
    returned = capture._host_bytes()
    if fault == "missing-frame":
        returned.clear()
    capture._existing_files(returned)
    assert capture.report["errors"] and capture.report["status"] == "UNVERIFIED"


@pytest.mark.parametrize("fault", ("none", "changed", "record"))
def test_native_identity_stream_hash_and_installed_record_binding(
    case, monkeypatch, fault
):
    payload = case.native_path.read_bytes()
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
    )
    distributions = {}
    for name in ("jaxlib", "numpy", "ml-dtypes"):
        metadata = case.root / f"{name}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Name: {name}\nVersion: control-1\n")
        (metadata / "RECORD").write_text(
            f"native.so,sha256={digest if fault != 'record' else 'wrong'},{len(payload)}\n"
        )
        distributions[name] = case.module.importlib.metadata.PathDistribution(metadata)
    monkeypatch.setattr(
        case.module.importlib.metadata, "distribution", lambda name: distributions[name]
    )
    if fault == "changed":
        case.native_path.write_bytes(b"changed actual native file")
    if fault == "none":
        case.capture._native_builds()
        assert (
            case.capture.report["native_builds"][0]["sha256"]
            == hashlib.sha256(payload).hexdigest()
        )
    else:
        with pytest.raises(ValueError):
            case.capture._native_builds()


def test_output_and_actual_compiler_event_limits_are_finite(case):
    case.capture.output = case.folder / case.module.DIRECTORY
    case.capture.output.mkdir()
    with pytest.raises(ValueError):
        case.capture._write("oversize", bytes(case.module._FILE_LIMIT + 1))
    case.capture.file_count = 63
    with pytest.raises(ValueError):
        case.capture._write("too-many", b"x")
    case.capture.file_count = 0
    case.capture.used["hlo"] = case.module._HLO_LIMIT
    with pytest.raises(ValueError):
        case.capture._write("too-much-hlo", b"x", hlo=True)
    case.capture.used["hlo"] = 0
    case.capture.used["data"] = case.module._DATA_LIMIT
    with pytest.raises(ValueError):
        case.capture._write("too-much-data", b"x")


@pytest.mark.parametrize(
    "fault", ("object", "shape", "duplicate", "oversize", "header")
)
def test_npz_header_storage_members_and_finite_bounds_reject_corruption(fault):
    module = _load()
    if fault == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            payload = _archive(2, duplicate=True)
    else:
        payload = _archive(
            2, dtype="|O" if fault == "object" else "<V2", bad_shape=fault == "shape"
        )
    if fault == "oversize":
        payload = b"\x93NUMPY\x01\x00\xff\xff" + bytes(module._FILE_LIMIT)
        with pytest.raises(ValueError):
            module._npy(payload, 2, 0)
        return
    if fault == "header":
        payload = b"\x93NUMPY\x01\x00\xff\xff"
        with pytest.raises(ValueError):
            module._npy(payload, 2, 0)
        return
    with pytest.raises(ValueError):
        module._npz(payload, 2)


def test_npz_preserves_entire_void_bf16_storage_and_index_payload():
    module = _load()
    arrays = module._npz(_archive(5), 5)
    assert arrays[0]["storage_dtype"] == "<V2" and arrays[0]["shape"] == (5, 8, 2)
    assert arrays[1]["storage_dtype"] == "<i4" and arrays[1]["shape"] == (5, 8, 2)
    assert len(bytes.fromhex(arrays[0]["raw_hex"])) == 160
    assert len(bytes.fromhex(arrays[1]["raw_hex"])) == 320


@pytest.mark.parametrize(
    "fault", ("symlink", "hardlink", "fifo", "parent-symlink", "size")
)
def test_original_file_transport_rejects_nonregular_replayed_and_unbounded_paths(
    tmp_path, fault
):
    import os

    module = _load()
    original = tmp_path / "original"
    original.write_bytes(b"actual original archive")
    candidate = tmp_path / "candidate"
    if fault == "symlink":
        candidate.symlink_to(original)
    elif fault == "hardlink":
        os.link(original, candidate)
    elif fault == "fifo":
        os.mkfifo(candidate)
    elif fault == "parent-symlink":
        parent = tmp_path / "linked"
        parent.symlink_to(tmp_path, target_is_directory=True)
        candidate = parent / "original"
    else:
        candidate.write_bytes(b"12345")
    with pytest.raises(ValueError):
        module._read(candidate, 4 if fault == "size" else 100)


def test_stable_code_encoding_handles_closure_and_slice_and_rejects_real_mutation():
    module = _load()

    def closure(value):
        def inner():
            return value[1:3:2]

        return inner

    code = closure([1, 2, 3]).__code__
    first = module._fingerprint(code)
    journal = __import__("marshal").dumps(code)
    assert module._fingerprint(code) == first and journal
    assert (
        module._fingerprint(code.replace(co_consts=(*code.co_consts, slice(1, 3, 2))))
        != first
    )
    assert module._encoded(slice(None, 3, -1)) == [
        "slice",
        ["none"],
        ["int", "0x3"],
        ["int", "-0x1"],
    ]
    with pytest.raises(TypeError):
        module._encoded(object())
    with pytest.raises(ValueError):
        module._encoded(tuple(range(4097)))
    deep = None
    for _ in range(34):
        deep = (deep,)
    with pytest.raises(ValueError):
        module._encoded(deep)


@pytest.mark.parametrize("original_pass", (False, True))
@pytest.mark.parametrize(
    "fault", ("none", "start", "finish", "after-report", "durability")
)
def test_copied_runner_preserves_original_phases_and_durable_report(
    tmp_path, monkeypatch, original_pass, fault
):
    copied = tmp_path / "runner"
    copied.mkdir()
    runner = copied / "run_pytest_batches.py"
    runner.write_text(batches._RUNNER.read_text())
    monkeypatch.setattr(batches, "_RUNNER", runner)
    helper = (
        "import json, os\nfrom pathlib import Path\n"
        "class Capture:\n"
        "    def __init__(self, folder, item):\n"
        "        self.folder, self.tmp = folder, item.funcargs['tmp_path']\n"
        "        self.fsync = os.fsync\n"
        "        self.report = {'status': 'READY', 'errors': []}\n"
        "    def finish(self):\n"
        f"        fault = {fault!r}\n"
        "        if fault == 'finish': raise RuntimeError('finish sentinel')\n"
        "        if fault == 'durability':\n"
        "            def fail(fd): raise OSError('durability sentinel')\n"
        "            os.fsync = fail\n"
        "    def finish_context(self, context): pass\n"
        "    def retain_report(self, call, report):\n"
        "        Path('original-report-retained').write_text(report.outcome)\n"
        "    def after_report(self, phase):\n"
        "        receipt = json.loads((self.folder/'receipt.json').read_text())\n"
        "        entries = receipt['phases'][phase['nodeid']]\n"
        "        assert [p['when'] for p in entries] == ['setup', 'call']\n"
        "        assert entries[-1]['outcome'] == phase['outcome']\n"
        "        assert phase['durable_phase_recorded'] is True\n"
        "        assert (self.tmp/'native-default/B2/fresh.npz').read_bytes() == b'original npz bytes'\n"
        "        Path('hlo-after-durable-report').write_text(phase['outcome'])\n"
        f"        if {fault!r} == 'after-report': raise OSError('capture sentinel')\n"
        "    def close(self):\n"
        "        os.fsync = self.fsync\n"
        "        Path('capture-closed').write_text('yes')\n"
        "def start_observation(folder, item, context):\n"
        f"    if {fault!r} == 'start': raise OSError('start sentinel')\n"
        "    return Capture(folder, item)\n"
    )
    (copied / "_source_bf16_topk_capture.py").write_text(helper)
    content = (
        "from pathlib import Path\nimport pytest\n"
        "@pytest.mark.parametrize('scope,contains_nan', [('inline',False)], ids=['finite-inline'])\n"
        f"def {_load().TEST}(scope, contains_nan, tmp_path):\n"
        "    calls = Path('original-calls')\n"
        "    assert not calls.exists()\n    calls.write_text('once')\n"
        "    target = tmp_path/'native-default/B2/fresh.npz'\n"
        "    target.parent.mkdir(parents=True)\n    target.write_bytes(b'original npz bytes')\n"
    )
    if not original_pass:
        content += "    assert False, 'unchanged original BF16 portable failure'\n"
    result, run, folder = batches._run_suite(tmp_path, {_load().OWNER: content})
    suite = tmp_path / "suite"
    assert (suite / "original-calls").read_text() == "once"
    expected = "passed" if original_pass else "failed"
    receipt = json.loads((folder / "batch-0001/receipt.json").read_text())
    phases = next(iter(receipt["phases"].values()))
    assert [p["when"] for p in phases] == ["setup", "call", "teardown"]
    assert [p["outcome"] for p in phases] == ["passed", expected, "passed"]
    assert (result.returncode == 0) is original_pass
    assert run["status"] == ("PASS" if original_pass else "FAIL")
    if not original_pass:
        assert "unchanged original BF16 portable failure" in phases[1]["longrepr"]
    if fault == "start":
        assert not (suite / "capture-closed").exists()
    else:
        assert (suite / "capture-closed").read_text() == "yes"
    assert (suite / "hlo-after-durable-report").exists() is (
        fault not in ("start", "durability")
    )


def test_non_target_collection_and_control_paths_keep_original_cast_ast():
    """The actual repair packet separately pins the full preimage AST projection."""
    helper = _load()
    tree = ast.parse((helper._ROOT / "scripts/run_pytest_batches.py").read_text())
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_worker"
    )
    report = next(
        node
        for node in worker.body
        if isinstance(node, ast.ClassDef) and node.name == "Report"
    )
    call = next(
        node
        for node in report.body
        if isinstance(node, ast.FunctionDef) and node.name == "pytest_runtest_call"
    )
    assert isinstance(call.body[0], ast.If)
    branch = ast.unparse(call.body[0])
    assert (
        'request["kind"] == "batch"' in branch or "request['kind'] == 'batch'" in branch
    )
    assert "control is None" in branch and "_BF16_CAPTURE_NODE" in branch
    assert "_source_bf16_topk_capture.py" in branch
    assert all(
        "_source_bf16_topk_capture" not in ast.unparse(node) for node in call.body[1:]
    )


def test_copied_runner_never_imports_new_capture_for_non_target_or_collection(
    tmp_path, monkeypatch
):
    copied = tmp_path / "runner"
    copied.mkdir()
    runner = copied / "run_pytest_batches.py"
    runner.write_text(batches._RUNNER.read_text())
    monkeypatch.setattr(batches, "_RUNNER", runner)
    marker = copied / "forbidden-import"
    (copied / "_source_bf16_topk_capture.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('forbidden-import').write_text('imported')\n"
        "raise AssertionError('non-target capture import')\n"
    )
    result, run, _ = batches._run_suite(
        tmp_path, {"test_unrelated.py": "def test_original():\n    pass\n"}
    )
    assert result.returncode == 0 and run["status"] == "PASS"
    assert not marker.exists()


def _environment_view(case, monkeypatch, changes):
    # Substitute only this isolated helper's reader; never write process env.
    proxy = ModuleType("os")
    vars(proxy).update(vars(case.module.os))
    proxy.environ = dict(case.module.os.environ)
    for name, value in changes.items():
        if value is None:
            proxy.environ.pop(name, None)
        else:
            proxy.environ[name] = value
    monkeypatch.setattr(case.module, "os", proxy)
    return proxy.environ


def test_pytest_phase_journal_binds_exact_setup_call_without_env_writes(case):
    before = case.capture.report["pytest_current_test_journal"][0]
    assert before["boundary"] == "acquire" and before["accepted"]
    assert before["value"] == f"{case.module.NODE_ID} (setup)"
    assert not before["completed_call_boundary"]
    expected_call = f"{case.module.NODE_ID} (call)"
    assert case.module.os.environ["PYTEST_CURRENT_TEST"] == expected_call
    case.capture.finish()
    after = case.capture.report["pytest_current_test_journal"][-1]
    assert after["boundary"] == "call-finish" and after["accepted"]
    assert after["completed_call_boundary"] and after["value"] == expected_call
    assert before["other_environment_sha256"] == after["other_environment_sha256"]
    assert before["environment_sha256"] != after["environment_sha256"]
    assert case.module.os.environ["PYTEST_CURRENT_TEST"] == expected_call
    context = {**case.context, "environment": {"PYTEST_CURRENT_TEST": expected_call}}
    case.capture.finish_context(context)
    assert (
        case.capture.report["original_fences_held"]
        and not case.capture.report["errors"]
    )


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "foreign-node",
        "malformed",
        "stays-setup",
        "teardown",
        "other-env",
        "other-pytest-env",
    ),
)
def test_phase_or_other_environment_mismatch_fails_closed_and_restores(
    case, monkeypatch, fault
):
    changes = {}
    if fault == "missing":
        changes["PYTEST_CURRENT_TEST"] = None
    elif fault == "foreign-node":
        changes["PYTEST_CURRENT_TEST"] = "foreign/test.py::test_foreign (call)"
    elif fault == "malformed":
        changes["PYTEST_CURRENT_TEST"] = f"{case.module.NODE_ID} (call) extra"
    elif fault == "stays-setup":
        changes["PYTEST_CURRENT_TEST"] = f"{case.module.NODE_ID} (setup)"
    elif fault == "teardown":
        changes["PYTEST_CURRENT_TEST"] = f"{case.module.NODE_ID} (teardown)"
    elif fault == "other-env":
        changes["JAX2ONNX_CAPTURE_PHASE_UNRELATED_CONTROL"] = "changed"
    else:
        changes["PYTEST_ADDOPTS"] = "changed capture control"
    observed = _environment_view(case, monkeypatch, changes)
    original = dict(observed)
    case.capture.finish()
    assert observed == original
    assert case.compiler.compile_or_get_cached is case.capture.original
    assert (
        case.capture.report["status"] == "UNVERIFIED" and case.capture.report["errors"]
    )
    assert not case.capture.report["pytest_current_test_journal"][-1]["accepted"]
    assert case.capture.report.get("original_fences_held") is not True
    case.capture.close()
    assert not case.capture.events and not case.capture.hosts


@pytest.mark.parametrize("boundary", ("retain-report", "after-report"))
def test_original_report_boundaries_reject_target_teardown(case, monkeypatch, boundary):
    case.capture.finish()
    phase = _phase(case)
    original_report = case.capture.original_report
    observed = _environment_view(
        case,
        monkeypatch,
        {
            "PYTEST_CURRENT_TEST": f"{case.module.NODE_ID} (teardown)",
        },
    )
    original = dict(observed)
    if boundary == "retain-report":
        case.capture.retain_report(None, original_report)
    case.capture.after_report(phase)
    assert observed == original and original_report.outcome == "passed"
    output = case.folder / case.module.DIRECTORY
    assert {path.name for path in output.iterdir()} == {"manifest.json"}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "UNVERIFIED"
    assert (
        manifest["runner_observed_original_call"]["validation"]
        == "RUNNER_OBSERVED_NOT_CAPTURE_VALIDATED"
    )
    assert not manifest["pytest_current_test_journal"][-1]["accepted"]
    assert (
        case.capture.report["status"] == "UNVERIFIED" and case.capture.report["errors"]
    )
    assert case.capture.closed and not case.capture.events and not case.capture.hosts


def test_same_node_call_phase_is_not_a_valid_new_acquisition(case):
    case.capture.close()
    capture = case.module.start_observation(case.folder, case.item, case.context)
    assert not capture.installed and not case.compiler.calls
    assert case.compiler.compile_or_get_cached is case.capture.original
    assert capture.report["status"] == "UNVERIFIED" and capture.report["errors"]
    entry = capture.report["pytest_current_test_journal"][0]
    assert entry["value"] == f"{case.module.NODE_ID} (call)" and not entry["accepted"]
    capture.close()


@pytest.mark.parametrize("fault", ("foreign-node", "teardown", "other-env"))
def test_worker_context_allows_only_the_same_exact_pytest_phase_transition(case, fault):
    case.capture.finish()
    context = {
        **case.context,
        "environment": {
            "PYTEST_CURRENT_TEST": f"{case.module.NODE_ID} (call)",
        },
    }
    if fault == "foreign-node":
        context["environment"][
            "PYTEST_CURRENT_TEST"
        ] = "foreign/test.py::test_foreign (call)"
    elif fault == "teardown":
        context["environment"][
            "PYTEST_CURRENT_TEST"
        ] = f"{case.module.NODE_ID} (teardown)"
    else:
        context["environment"]["PYTEST_ADDOPTS"] = "changed capture control"
    case.capture.finish_context(context)
    assert (
        case.capture.report["status"] == "UNVERIFIED" and case.capture.report["errors"]
    )
    assert case.compiler.compile_or_get_cached is case.capture.original
