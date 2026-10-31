# tests/extra_tests/framework/test_source_cast_executable.py

"""Synthetic executable/frame controls and native MLIR parsing, without source calls.

The synthetic classes below deliberately replace the helper's native pin seams.
They do not constitute observations of a JAX executable or numerical execution.
Native integration and additional serialization/linking remain separate checks.
"""

import builtins
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any
import weakref

import pytest


_LOW = "003c003c003c00bcff7b007c"
_HIGH = "003c003c013c00bcff7b007c"


def _words(value: Any, storage: str = _LOW) -> dict[str, Any]:
    return {
        "identity": id(value),
        "dtype": "float16",
        "dtype_str": "<f2",
        "shape": [6],
        "storage_hex": storage,
        "sha256": hashlib.sha256(bytes.fromhex(storage)).hexdigest(),
        "element_storage_hex": [storage[i : i + 4] for i in range(0, 24, 4)],
    }


@pytest.fixture
def executable_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    path = Path(__file__).resolve().parents[3] / "scripts/_source_cast_executable.py"
    name = "_source_cast_executable_independent_controls"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _module_fixture(
    monkeypatch: pytest.MonkeyPatch, folder: Path, name: str, source: str
) -> types.ModuleType:
    module = types.ModuleType(name)
    path = folder / (name + ".py")
    path.write_text(source)
    module.__file__ = str(path)
    module.__dict__["executed"] = []
    exec(compile(source, str(path), "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _Harness:
    def __init__(self, module: Any, monkeypatch: pytest.MonkeyPatch, folder: Path):
        self.module = module
        self.folder = folder
        self.calls: list[dict[str, Any]] = []
        self.forbidden: list[str] = []
        self.owner = _module_fixture(
            monkeypatch,
            folder,
            "_cast_owner_fixture",
            "def test_floating_conversion_rounding_and_strict_special_value_gates():\n"
            "    target = 'float16'\n"
            "    def fn(value):\n"
            "        executed.append('source')\n"
            "        raise AssertionError('synthetic numerical fn must not run: ' + target)\n"
            "    return fn\n",
        )
        self.producer = _module_fixture(
            monkeypatch,
            folder,
            "_cast_producer_fixture",
            "def _artifact(fn, feeds):\n"
            "    executed.append('artifact')\n"
            "    raise AssertionError('synthetic producer must not be called')\n"
            "def _reference(fn, values):\n"
            "    executed.append('reference')\n"
            "    raise AssertionError('synthetic reference must not be called')\n",
        )
        self.compiler_module = _module_fixture(
            monkeypatch,
            folder,
            "_cast_compiler_fixture",
            "def compile_or_get_cached(backend, computation, host_callbacks):\n"
            "    executed.append('compiler')\n"
            "    raise AssertionError('synthetic compiler must not be called')\n"
            "def _pjit_call_impl_python(compiled, sharded_const_args):\n"
            "    executed.append('dispatch')\n"
            "    raise AssertionError('synthetic dispatch must not be called')\n",
        )
        # Construct the real nested Python closure from its pinned code owner.
        # This factory defines/returns fn; it never invokes the numerical fn.
        self.owner.source_function = (
            self.owner.test_floating_conversion_rounding_and_strict_special_value_gates()
        )
        self.owner._artifact = self.producer._artifact
        self.profiler_module = _module_fixture(
            monkeypatch,
            folder,
            "jax._src.profiler",
            "import functools\n"
            "def annotate_function(func, name=None, **decorator_kwargs):\n"
            "    if name is None:\n"
            "        name = func.__qualname__\n"
            "    @functools.wraps(func)\n"
            "    def wrapper(*args, **kwargs):\n"
            "        marker = (decorator_kwargs, func, name)\n"
            "        executed.append(marker)\n"
            "        raise AssertionError('synthetic executor wrapper must not run')\n"
            "    return wrapper\n",
        )
        self.pxla_module = types.ModuleType("jax._src.interpreters.pxla")
        self.pxla_module.profiler = self.profiler_module
        pxla_path = folder / "synthetic_pxla.py"
        pxla_source = (
            "class ExecuteReplicated:\n"
            "    __slots__ = ('xla_executable', 'backend')\n"
            "    def __init__(self, executable, backend):\n"
            "        self.xla_executable = executable\n"
            "        self.backend = backend\n"
            "    @profiler.annotate_function\n"
            "    def __call__(self, *args):\n"
            "        executed.append('executor-call')\n"
            "        raise AssertionError('synthetic executor must not run')\n"
            "class MeshExecutable:\n"
            "    __slots__ = ('xla_executable', '_unsafe_call')\n"
            "    def __init__(self, executable):\n"
            "        self.xla_executable = executable\n"
            "        self._unsafe_call = ExecuteReplicated(executable, executable._client)\n"
            "    @property\n"
            "    def unsafe_call(self):\n"
            "        executed.append('unsafe-call-property')\n"
            "        raise AssertionError('unsafe_call getter must not run')\n"
        )
        pxla_path.write_text(pxla_source)
        self.pxla_module.__file__ = str(pxla_path)
        self.pxla_module.executed = []
        exec(compile(pxla_source, str(pxla_path), "exec"), vars(self.pxla_module))
        monkeypatch.setitem(sys.modules, self.pxla_module.__name__, self.pxla_module)
        calls, forbidden = self.calls, self.forbidden

        class Client:
            def __init__(self) -> None:
                self.behavior = lambda executable: b"EXPLICIT SYNTHETIC EXECUTABLE"
                self._platform: Any = "cpu"

            @property
            def platform(self) -> Any:
                return self._platform

            def serialize_executable(self, executable: Any) -> bytes:
                calls.append({"client": id(self), "executable": id(executable)})
                return self.behavior(executable)

            def compile(self, *args: Any, **kwargs: Any) -> Any:
                forbidden.append("compile")
                raise AssertionError("no replacement compilation")

        class Executable:
            def __init__(self, client: Any) -> None:
                self._client = client

            @property
            def client(self) -> Any:
                return self._client

            def execute(self, *args: Any, **kwargs: Any) -> Any:
                forbidden.append("execute")
                raise AssertionError("no extra executable execution")

            def lower(self, *args: Any, **kwargs: Any) -> Any:
                forbidden.append("lower")
                raise AssertionError("no replacement lowering")

        class Jit:
            def __init__(self, function: Any) -> None:
                self._fun = function

        self.types = {
            "client": Client,
            "executable": Executable,
            "jit": Jit,
            "mesh": self.pxla_module.MeshExecutable,
            "executor": self.pxla_module.ExecuteReplicated,
        }
        self.client: Any = Client()
        self.executables: list[Any] = []
        self.frames: list[Any] = []
        self.meshes: list[Any] = []
        self.input = object()
        self.outputs = [object(), object()]
        self.word_records = {id(value): _words(value) for value in self.outputs}
        self.program_token = object()
        self.program_record = {
            "input": {"dtype": "float64", "shape": [6]},
            "output": {"dtype": "float16", "shape": [6]},
            "operations": ["stablehlo.convert", "func.return"],
            "ir": "EXPLICIT SYNTHETIC PROGRAM; NO NATIVE COMPILATION",
            "source": "Independent synthetic fixture, not an actual native executable",
        }
        self.program_record["ir_sha256"] = hashlib.sha256(
            self.program_record["ir"].encode()
        ).hexdigest()

        def program(value: Any) -> dict[str, Any]:
            if value is not self.program_token:
                raise ValueError("unrecognized fixture compilation")
            return json.loads(json.dumps(self.program_record))

        monkeypatch.setattr(module, "_native_types", lambda: self.types.copy())
        monkeypatch.setattr(
            module,
            "_functions",
            lambda: {
                "compiler": (
                    self.compiler_module,
                    "compile_or_get_cached",
                    self.compiler_module.compile_or_get_cached,
                ),
                "dispatch": (
                    self.compiler_module,
                    "_pjit_call_impl_python",
                    self.compiler_module._pjit_call_impl_python,
                ),
            },
        )
        monkeypatch.setattr(module, "_program", program)
        self.feed_tuple = (self.input,)
        self.feeds = [self.feed_tuple]
        self.test_frame = self.frame(
            self.owner.test_floating_conversion_rounding_and_strict_special_value_gates,
            {
                "source": "float64",
                "target": "float16",
                "group": "ties",
                "fn": self.owner.source_function,
                "value": self.input,
            },
            None,
        )
        self.artifact_frame = self.frame(
            self.producer._artifact,
            {"fn": self.owner.source_function, "feeds": self.feeds},
            self.test_frame,
        )
        self.capture = module.start_capture(
            folder, self.owner, self.producer._artifact, self.array_record
        )
        assert not self.capture.closed, self.capture.report

    def frame(self, function: Any, local: dict[str, Any], parent: Any) -> Any:
        frame = types.SimpleNamespace(
            f_code=function.__code__,
            f_globals=function.__globals__,
            f_locals=local,
            f_back=parent,
        )
        self.frames.append(frame)
        return frame

    def array_record(self, value: Any) -> dict[str, Any]:
        return self.word_records[id(value)].copy()

    def begin(self, index: int) -> Any:
        function = (
            self.owner.source_function
            if index == 0
            else self.types["jit"](self.owner.source_function)
        )
        frame = self.frame(
            self.producer._reference,
            {"fn": function, "values": self.feed_tuple},
            self.artifact_frame,
        )
        self.capture.observe(frame, "call", None)
        return frame

    def executable(self, client: Any = None) -> Any:
        executable = self.types["executable"](self.client if client is None else client)
        self.executables.append(executable)
        return executable

    def compiled(
        self, reference: Any, executable: Any, computation: Any = None
    ) -> None:
        frame = self.frame(
            self.compiler_module.compile_or_get_cached,
            {
                "backend": self.client,
                "computation": (
                    self.program_token if computation is None else computation
                ),
                "host_callbacks": [],
            },
            reference,
        )
        self.capture.observe(frame, "return", executable)

    def dispatched(self, reference: Any, executable: Any) -> None:
        mesh = self.types["mesh"](executable)
        self.meshes.append(mesh)
        constants: tuple[Any, ...] = ()
        frame = self.frame(
            self.compiler_module._pjit_call_impl_python,
            {"compiled": mesh, "sharded_const_args": constants},
            reference,
        )
        self.capture.observe(frame, "return", (None, mesh, None, constants))

    def end(self, reference: Any, index: int) -> None:
        self.capture.observe(reference, "return", [self.outputs[index]])

    def good_role(self, index: int) -> Any:
        reference = self.begin(index)
        executable = self.executable()
        self.compiled(reference, executable)
        self.dispatched(reference, executable)
        self.end(reference, index)
        return executable

    def bind(self) -> dict[str, Any]:
        records = {
            "native_eager": [self.array_record(self.outputs[0])],
            "native_fresh_jit": [self.array_record(self.outputs[1])],
        }
        self.capture.bind_artifact(records)
        return records

    def phase(self, **changed: Any) -> dict[str, Any]:
        return {
            "nodeid": self.module.NODE_ID,
            "when": "call",
            "outcome": "failed",
            "longrepr": "original synthetic test failure remains authoritative",
            "durable_phase_recorded": True,
            **changed,
        }

    def report(self) -> dict[str, Any]:
        return json.loads(
            (self.folder / self.module.DIRECTORY / "report.json").read_text()
        )

    def assert_no_source_calls(self) -> None:
        assert (
            self.owner.executed
            == self.producer.executed
            == self.compiler_module.executed
            == []
        )
        assert self.pxla_module.executed == self.profiler_module.executed == []
        assert self.forbidden == []


@pytest.fixture
def harness(
    executable_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Any:
    fixture = _Harness(executable_module, monkeypatch, tmp_path)
    yield fixture
    fixture.assert_no_source_calls()
    fixture.capture.close()


def _two_bound(harness: _Harness) -> None:
    harness.good_role(0)
    harness.good_role(1)
    harness.bind()


def test_two_original_roles_serialize_only_after_durable_report_and_stopped_profile(
    harness: _Harness,
) -> None:
    _two_bound(harness)
    assert harness.calls == []
    assert not list((harness.folder / harness.module.DIRECTORY).glob("*.bin"))
    harness.capture.stop()
    assert harness.calls == []
    phase = harness.phase()
    phase_before = json.loads(json.dumps(phase))
    words_before = json.loads(json.dumps(harness.word_records))

    def serialize(executable: Any) -> bytes:
        checkpoint = harness.report()
        assert checkpoint["original_call_phase"] == phase_before
        assert checkpoint["serialization_attempts"] == len(harness.calls)
        assert harness.capture.stopped
        assert harness.module.sys.getprofile() is None
        return b"SYNTHETIC ORIGINAL OBJECT " + str(id(executable)).encode()

    harness.client.behavior = serialize
    harness.capture.after_report(phase)
    report = harness.report()
    assert report["status"] == "ORIGINAL_EXECUTABLES_SERIALIZED"
    assert report["serialization_attempts"] == len(harness.calls) == 2
    assert phase == phase_before
    assert json.loads(json.dumps(harness.word_records)) == words_before
    assert [r["role"] for r in report["roles"]] == ["native_eager", "native_fresh_jit"]
    for role in report["roles"]:
        assert role["status"] == "ORIGINAL_EXECUTABLE_SERIALIZED"
        assert role["artifact_reference_held"] is True
        assert role["additional_native_linking_and_allocation"] is True
        assert role["program"]["source"].startswith("Independent synthetic fixture")
        path = harness.folder / harness.module.DIRECTORY / role["binary"]["path"]
        assert len(path.read_bytes()) == role["binary"]["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == role["binary"]["sha256"]
    assert report["native_owners_cleared"] is True
    assert harness.capture.roles == []
    assert harness.capture.bindings == harness.capture.native == {}


@pytest.mark.parametrize(
    "fault",
    (
        "no-compiler",
        "compiler-only",
        "different-executable",
        "bad-client",
        "bad-program",
        "duplicate-compiler",
        "duplicate-dispatch",
        "compiler-globals",
    ),
)
def test_unbound_or_ambiguous_original_role_never_serializes_a_substitute(
    harness: _Harness, fault: str
) -> None:
    reference = harness.begin(0)
    executable = harness.executable()
    if fault != "no-compiler":
        candidate = executable
        if fault == "bad-client":
            candidate = harness.executable(harness.types["client"]())
        if fault == "compiler-globals":
            frame = harness.frame(
                harness.compiler_module.compile_or_get_cached,
                {
                    "backend": harness.client,
                    "computation": harness.program_token,
                    "host_callbacks": [],
                },
                reference,
            )
            frame.f_globals = {}
            harness.capture.observe(frame, "return", candidate)
        else:
            harness.compiled(
                reference, candidate, object() if fault == "bad-program" else None
            )
        if fault == "duplicate-compiler":
            harness.compiled(reference, executable)
    if fault != "compiler-only":
        harness.dispatched(
            reference,
            harness.executable() if fault == "different-executable" else executable,
        )
        if fault == "duplicate-dispatch":
            harness.dispatched(reference, executable)
    harness.end(reference, 0)
    other = harness.good_role(1)
    harness.bind()
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    report = harness.report()
    assert report["status"] == "UNVERIFIED"
    assert report["roles"][0]["status"] == "UNVERIFIED"
    assert report["roles"][0]["errors"]
    assert all(call["executable"] != id(executable) for call in harness.calls)
    assert len(harness.calls) == 1 and harness.calls[0]["executable"] == id(other)
    assert report["roles"][1]["status"] == "ORIGINAL_EXECUTABLE_SERIALIZED"
    assert report["native_owners_cleared"] is True


@pytest.mark.parametrize(
    "fault",
    (
        "wrong-feed-identity",
        "wrong-source-function",
        "wrong-label",
        "fresh-role-not-jit",
        "ambiguous-artifact",
        "missing-test",
        "ancestor-cycle",
    ),
)
def test_wrong_original_reference_provenance_cannot_borrow_a_role(
    harness: _Harness, fault: str
) -> None:
    reference = harness.frame(
        harness.producer._reference,
        {"fn": harness.owner.source_function, "values": harness.feed_tuple},
        harness.artifact_frame,
    )
    if fault == "wrong-feed-identity":
        reference.f_locals["values"] = tuple([harness.input])
    elif fault == "wrong-source-function":
        harness.artifact_frame.f_locals["fn"] = lambda value: value
    elif fault == "wrong-label":
        harness.test_frame.f_locals["source"] = "float32"
    elif fault == "fresh-role-not-jit":
        harness.good_role(0)
    elif fault == "ambiguous-artifact":
        reference.f_back = harness.frame(
            harness.producer._artifact,
            harness.artifact_frame.f_locals,
            harness.artifact_frame,
        )
    elif fault == "missing-test":
        harness.artifact_frame.f_back = None
    else:
        reference.f_back = reference
    harness.capture.observe(reference, "call", None)
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    assert harness.calls == []
    assert harness.report()["status"] == "UNVERIFIED"
    assert harness.report()["errors"]
    assert harness.report()["native_owners_cleared"] is True


def test_later_activated_reference_cannot_replace_or_add_to_the_first_two_roles(
    harness: _Harness,
) -> None:
    _two_bound(harness)
    for _ in range(3):
        later = harness.begin(1)
        executable = harness.executable()
        harness.compiled(later, executable)
        harness.dispatched(later, executable)
        harness.end(later, 1)
    assert len(harness.capture.roles) == 2
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    assert harness.report()["status"] == "ORIGINAL_EXECUTABLES_SERIALIZED"
    assert len(harness.calls) == 2


@pytest.mark.parametrize("fault", ("word", "array-identity"))
def test_artifact_binding_requires_original_returned_words_and_array_identity(
    harness: _Harness, fault: str
) -> None:
    harness.good_role(0)
    harness.good_role(1)
    records = {
        "native_eager": [harness.array_record(harness.outputs[0])],
        "native_fresh_jit": [harness.array_record(harness.outputs[1])],
    }
    if fault == "word":
        records["native_fresh_jit"] = [_words(harness.outputs[1], _HIGH)]
    else:
        records["native_fresh_jit"][0]["identity"] += 1
    harness.capture.bind_artifact(records)
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    assert harness.calls == []
    assert harness.report()["status"] == "UNVERIFIED"
    assert harness.report()["errors"]


@pytest.mark.parametrize(
    "fault",
    (
        "not-stopped",
        "not-durable",
        "setup-phase",
        "wrong-node",
        "sys-profile",
        "thread-profile",
    ),
)
def test_deferred_serialization_requires_the_real_call_boundary_without_clearing_profiles(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    _two_bound(harness)
    if fault != "not-stopped":
        harness.capture.stop()
    phase = harness.phase()
    if fault == "not-durable":
        phase["durable_phase_recorded"] = False
    elif fault == "setup-phase":
        phase["when"] = "setup"
    elif fault == "wrong-node":
        phase["nodeid"] = "different source"
    elif fault == "sys-profile":
        monkeypatch.setattr(harness.module.sys, "getprofile", lambda: object())
    elif fault == "thread-profile":
        monkeypatch.setattr(harness.module.threading, "getprofile", lambda: object())

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("capture attempted to clear or replace a profile")

    monkeypatch.setattr(harness.module.sys, "setprofile", forbidden)
    monkeypatch.setattr(harness.module.threading, "setprofile", forbidden)
    before = json.loads(json.dumps(phase))
    harness.capture.after_report(phase)
    assert phase == before and harness.calls == []
    assert harness.report()["status"] == "UNVERIFIED"
    assert harness.report()["native_owners_cleared"] is True


@pytest.mark.parametrize(
    "fault",
    ("serializer-error", "binary-oversize", "aggregate-oversize", "binary-collision"),
)
def test_serialization_failures_retain_partial_evidence_and_original_outcomes(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    _two_bound(harness)
    first = harness.executables[0]
    monkeypatch.setattr(harness.module, "_MAX_BINARY", 8)
    monkeypatch.setattr(harness.module, "_MAX_TOTAL", 12)

    def serialize(executable: Any) -> bytes:
        if fault == "serializer-error" and executable is first:
            raise RuntimeError("synthetic serialization failure")
        if fault == "binary-oversize" and executable is first:
            return b"x" * 9
        return b"x" * (8 if fault == "aggregate-oversize" else 2)

    harness.client.behavior = serialize
    collision = harness.folder / harness.module.DIRECTORY / "native_eager.bin"
    if fault == "binary-collision":
        collision.write_bytes(b"unchanged existing artifact")
    phase = harness.phase()
    before = json.loads(json.dumps(phase))
    words = json.loads(json.dumps(harness.word_records))
    harness.capture.stop()
    harness.capture.after_report(phase)
    report = harness.report()
    assert report["status"] == "UNVERIFIED"
    assert len(harness.calls) == report["serialization_attempts"] == 2
    assert (
        sum(r["status"] == "ORIGINAL_EXECUTABLE_SERIALIZED" for r in report["roles"])
        == 1
    )
    assert phase == before and json.loads(json.dumps(harness.word_records)) == words
    if fault == "binary-collision":
        assert collision.read_bytes() == b"unchanged existing artifact"
    assert report["native_owners_cleared"] is True


def test_instance_serializer_shadow_is_never_executed(harness: _Harness) -> None:
    _two_bound(harness)
    foreign = []
    harness.client.serialize_executable = lambda executable: foreign.append(executable)
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    assert foreign == []
    assert harness.report()["status"] == "ORIGINAL_EXECUTABLES_SERIALIZED"
    assert len(harness.calls) == 2


@pytest.mark.parametrize("binding", ("artifact", "compiler-module", "serializer"))
def test_mutated_production_bindings_cannot_authorize_deferred_serialization(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    _two_bound(harness)
    if binding == "artifact":
        monkeypatch.setattr(harness.owner, "_artifact", lambda *args: None)
    elif binding == "compiler-module":
        monkeypatch.setitem(
            sys.modules,
            harness.compiler_module.__name__,
            types.ModuleType("replacement"),
        )
    else:
        monkeypatch.setattr(
            harness.types["client"],
            "serialize_executable",
            lambda *args: b"replacement",
        )
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    assert harness.calls == []
    assert harness.report()["status"] == "UNVERIFIED"
    assert harness.report()["native_owners_cleared"] is True


@pytest.mark.parametrize(
    "mutation", ("code", "target-cell", "different-function-between-roles")
)
def test_original_inner_callable_code_closure_and_identity_remain_pinned(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    original = harness.owner.source_function
    assert original.__code__.co_freevars == ("target",)
    assert original.__closure__ is not None and len(original.__closure__) == 1
    assert original.__closure__[0].cell_contents == "float16"
    if mutation == "different-function-between-roles":
        harness.good_role(0)
        replacement = (
            harness.owner.test_floating_conversion_rounding_and_strict_special_value_gates()
        )
        assert replacement is not original and replacement.__code__ is original.__code__
        harness.owner.source_function = replacement
        harness.artifact_frame.f_locals["fn"] = replacement
        harness.test_frame.f_locals["fn"] = replacement
        harness.good_role(1)
    else:
        _two_bound(harness)
        if mutation == "target-cell":
            monkeypatch.setattr(original.__closure__[0], "cell_contents", "float32")
        else:

            def alternative_factory() -> Any:
                target = "float16"

                def fn(value: Any) -> Any:
                    return target

                return fn

            alternative = alternative_factory()
            assert alternative.__code__.co_freevars == ("target",)
            monkeypatch.setattr(original, "__code__", alternative.__code__)
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    assert harness.calls == []
    assert harness.report()["status"] == "UNVERIFIED"
    assert harness.report()["native_owners_cleared"] is True


def test_first_reference_requires_the_actual_nested_code_constant_not_only_fn_identity(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = harness.owner.source_function

    def different_factory() -> Any:
        target = "float16"

        def fn(value: Any) -> Any:
            return target

        return fn

    alternative = different_factory()
    assert (
        alternative.__code__.co_freevars == original.__code__.co_freevars == ("target",)
    )
    assert not any(
        constant is alternative.__code__
        for constant in harness.test_frame.f_code.co_consts
    )
    monkeypatch.setattr(original, "__code__", alternative.__code__)
    assert (
        harness.artifact_frame.f_locals["fn"]
        is harness.test_frame.f_locals["fn"]
        is original
    )
    harness.good_role(0)
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    assert harness.calls == []
    assert harness.report()["roles"] == []
    assert harness.report()["errors"]
    assert harness.report()["status"] == "UNVERIFIED"
    assert harness.report()["native_owners_cleared"] is True


@pytest.mark.parametrize(
    "fault", ("host-callback", "constant-argument", "constant-identity")
)
def test_callbacks_or_constants_cannot_borrow_the_tiny_original_program_scope(
    harness: _Harness, fault: str
) -> None:
    reference = harness.begin(0)
    executable = harness.executable()
    if fault == "host-callback":
        frame = harness.frame(
            harness.compiler_module.compile_or_get_cached,
            {
                "backend": harness.client,
                "computation": harness.program_token,
                "host_callbacks": [object()],
            },
            reference,
        )
        harness.capture.observe(frame, "return", executable)
    else:
        harness.compiled(reference, executable)
    mesh = harness.types["mesh"](executable)
    harness.meshes.append(mesh)
    constants = (object(),) if fault == "constant-argument" else []
    frame = harness.frame(
        harness.compiler_module._pjit_call_impl_python,
        {"compiled": mesh, "sharded_const_args": constants},
        reference,
    )
    returned_constants = () if fault == "constant-identity" else constants
    harness.capture.observe(frame, "return", (None, mesh, None, returned_constants))
    harness.end(reference, 0)
    harness.good_role(1)
    harness.bind()
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    assert harness.report()["status"] == "UNVERIFIED"
    assert all(call["executable"] != id(executable) for call in harness.calls)
    assert len(harness.calls) == 1


@pytest.mark.parametrize(
    "fault",
    (
        "different-executable",
        "different-backend",
        "wrong-executor-type",
        "missing-executor",
    ),
)
def test_cached_actual_executor_must_match_the_compiler_and_mesh_without_invocation(
    harness: _Harness, fault: str
) -> None:
    reference = harness.begin(0)
    executable = harness.executable()
    harness.compiled(reference, executable)
    mesh = harness.types["mesh"](executable)
    harness.meshes.append(mesh)
    if fault == "different-executable":
        mesh._unsafe_call.xla_executable = harness.executable()
    elif fault == "different-backend":
        mesh._unsafe_call.backend = harness.types["client"]()
    elif fault == "wrong-executor-type":
        mesh._unsafe_call = types.SimpleNamespace(
            xla_executable=executable, backend=harness.client
        )
    else:
        mesh._unsafe_call = None
    constants: tuple[Any, ...] = ()
    dispatch = harness.frame(
        harness.compiler_module._pjit_call_impl_python,
        {"compiled": mesh, "sharded_const_args": constants},
        reference,
    )
    harness.capture.observe(dispatch, "return", (None, mesh, None, constants))
    harness.end(reference, 0)
    harness.good_role(1)
    harness.bind()
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    assert harness.report()["status"] == "UNVERIFIED"
    assert harness.report()["roles"][0]["errors"]
    assert len(harness.calls) == 1
    assert all(call["executable"] != id(executable) for call in harness.calls)
    assert harness.pxla_module.executed == harness.profiler_module.executed == []


@pytest.mark.parametrize(
    "binding", ("unsafe-property", "executor-wrapper", "profiler-annotator")
)
def test_changed_dispatch_methods_cannot_authorize_serialization(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    _two_bound(harness)
    if binding == "unsafe-property":
        monkeypatch.setattr(
            harness.types["mesh"], "unsafe_call", property(lambda self: None)
        )
    elif binding == "executor-wrapper":
        monkeypatch.setattr(harness.types["executor"], "__call__", lambda *args: None)
    else:
        monkeypatch.setattr(
            harness.profiler_module, "annotate_function", lambda function: function
        )
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    assert harness.calls == []
    assert harness.report()["status"] == "UNVERIFIED"
    assert harness.report()["native_owners_cleared"] is True


@pytest.mark.parametrize("fault", ("executable-subclass", "client-subclass"))
def test_equal_shaped_foreign_native_subclasses_do_not_borrow_original_identities(
    harness: _Harness, fault: str
) -> None:
    reference = harness.begin(0)
    if fault == "executable-subclass":

        class Subclass(harness.types["executable"]):
            pass

        candidate = Subclass(harness.client)
    else:

        class Subclass(harness.types["client"]):
            pass

        candidate = harness.executable(Subclass())
    harness.compiled(reference, candidate)
    harness.dispatched(reference, candidate)
    harness.end(reference, 0)
    harness.good_role(1)
    harness.bind()
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    assert harness.report()["status"] == "UNVERIFIED"
    assert all(call["executable"] != id(candidate) for call in harness.calls)
    assert len(harness.calls) == 1


@pytest.mark.parametrize(
    "fault", ("non-cpu", "foreign-platform", "changed-after-binding")
)
def test_only_exact_cpu_clients_enter_the_audited_serializer_path(
    harness: _Harness, fault: str
) -> None:
    attempts = []

    class Foreign:
        def __eq__(self, other: Any) -> Any:
            attempts.append("eq")
            raise AssertionError("foreign platform was compared")

        def __str__(self) -> str:
            attempts.append("str")
            raise AssertionError("foreign platform was coerced")

        def __bool__(self) -> bool:
            attempts.append("bool")
            raise AssertionError("foreign platform was tested")

    if fault == "changed-after-binding":
        _two_bound(harness)
        harness.client._platform = "cuda"
    else:
        harness.client._platform = Foreign() if fault == "foreign-platform" else "cuda"
        reference = harness.begin(0)
        executable = harness.executable()
        harness.compiled(reference, executable)
        harness.dispatched(reference, executable)
        harness.end(reference, 0)
        harness.client._platform = "cpu"
        other = harness.good_role(1)
        harness.bind()
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    assert attempts == []
    assert harness.report()["status"] == "UNVERIFIED"
    if fault == "changed-after-binding":
        assert harness.calls == []
    else:
        assert len(harness.calls) == 1 and harness.calls[0]["executable"] == id(other)
    assert harness.report()["native_owners_cleared"] is True


def _assert_final_capture_state_is_literal(value: Any) -> None:
    if (
        value is None
        or type(value) in (str, int, float, bool)
        or isinstance(value, Path)
    ):
        return
    if type(value) in (list, tuple):
        for item in value:
            _assert_final_capture_state_is_literal(item)
        return
    assert type(value) is dict, f"Capture retained a live owner: {type(value)!r}"
    for key, item in value.items():
        assert type(key) is str
        _assert_final_capture_state_is_literal(item)


def test_close_releases_executables_clients_and_callbacks_and_is_idempotent(
    harness: _Harness,
) -> None:
    _two_bound(harness)
    client = weakref.ref(harness.client)
    executable_refs = [weakref.ref(value) for value in harness.executables]
    harness.frames.clear()
    harness.meshes.clear()
    harness.executables.clear()
    harness.client = None
    gc.collect()
    assert client() is not None and all(ref() is not None for ref in executable_refs)
    harness.capture.close()
    harness.capture.close()
    gc.collect()
    assert client() is None and all(ref() is None for ref in executable_refs)
    assert harness.capture.roles == []
    assert harness.capture.bindings == harness.capture.native == {}
    assert harness.capture.array_record is None
    assert (
        harness.capture.owner
        is harness.capture.artifact
        is harness.capture.reference
        is harness.capture.test
        is None
    )
    assert harness.report()["native_owners_cleared"] is True
    assert harness.calls == []
    _assert_final_capture_state_is_literal(vars(harness.capture))


def test_synthetic_capture_never_imports_a_framework_or_calls_source(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__
    attempts = []

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in ("jax", "jaxlib", "numpy", "onnxruntime"):
            attempts.append(name)
            raise AssertionError("framework import attempted during observation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _two_bound(harness)
    harness.capture.stop()
    harness.capture.after_report(harness.phase())
    assert harness.report()["status"] == "ORIGINAL_EXECUTABLES_SERIALIZED"
    assert attempts == []
    harness.assert_no_source_calls()


def _mlir_program(fault: str = "none") -> str:
    source = "tensor<6xf64>"
    target = "tensor<6xf16>"
    if fault == "f32-input":
        source = "tensor<6xf32>"
    elif fault == "scalar":
        source, target = "tensor<f64>", "tensor<f16>"
    elif fault == "f32-output":
        target = "tensor<6xf32>"
    arguments = "%arg0: " + source
    function_inputs = source
    if fault == "extra-input":
        arguments += ", %arg1: " + source
        function_inputs += ", " + source
    extra = ""
    if fault == "constant":
        extra = '%c = "stablehlo.constant"() {value = dense<0.0> : tensor<6xf64>} : () -> tensor<6xf64>\n'
    elif fault == "effect":
        extra = '%effect = "stablehlo.custom_call"(%arg0) {call_target_name = "test_only", has_side_effect = true} : (tensor<6xf64>) -> tensor<6xf64>\n'
    attributes = " {test.unreviewed = 1 : i32}" if fault == "convert-attribute" else ""
    returned = "%arg0" if fault == "returned-input" else "%cast"
    result = source if fault == "returned-input" else target
    function = (
        '"func.func"() ({\n^bb0('
        + arguments
        + "):\n"
        + extra
        + '%cast = "stablehlo.convert"(%arg0)'
        + attributes
        + " : ("
        + source
        + ") -> "
        + target
        + "\n"
        + '"func.return"('
        + returned
        + ") : ("
        + result
        + ") -> ()\n"
        + "}) {function_type = ("
        + function_inputs
        + ") -> "
        + result
        + ', sym_name = "main"} : () -> ()\n'
    )
    if fault == "empty":
        return "module {}"
    if fault == "two-functions":
        function += function.replace('sym_name = "main"', 'sym_name = "other"')
    return "module {\n" + function + "}\n"


def test_actual_native_mlir_parser_accepts_only_the_fixed_single_conversion(
    executable_module: Any,
) -> None:
    from jaxlib.mlir import ir

    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        computation = ir.Module.parse(_mlir_program())
        result = executable_module._program(computation)
    assert result["input"] == {"dtype": "float64", "shape": [6]}
    assert result["output"] == {"dtype": "float16", "shape": [6]}
    assert result["operations"] == ["stablehlo.convert", "func.return"]
    assert hashlib.sha256(result["ir"].encode()).hexdigest() == result["ir_sha256"]


@pytest.mark.parametrize(
    "fault",
    (
        "f32-input",
        "scalar",
        "f32-output",
        "extra-input",
        "constant",
        "effect",
        "convert-attribute",
        "returned-input",
        "empty",
        "two-functions",
    ),
)
def test_actual_native_mlir_wrong_programs_never_qualify_for_serialization(
    executable_module: Any, fault: str
) -> None:
    from jaxlib.mlir import ir

    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        computation = ir.Module.parse(_mlir_program(fault))
        with pytest.raises(ValueError):
            executable_module._program(computation)


def test_mlir_text_bound_and_foreign_computation_do_not_trigger_protocols(
    executable_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jaxlib.mlir import ir

    attempts = []

    class Foreign:
        @property
        def body(self) -> Any:
            attempts.append("body")
            raise AssertionError("foreign MLIR object was inspected")

        def __str__(self) -> str:
            attempts.append("str")
            raise AssertionError("foreign MLIR object was formatted")

    with pytest.raises(ValueError):
        executable_module._program(Foreign())
    assert attempts == []
    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        computation = ir.Module.parse(_mlir_program())
        monkeypatch.setattr(executable_module, "_MAX_TEXT", 8)
        with pytest.raises(ValueError):
            executable_module._program(computation)
