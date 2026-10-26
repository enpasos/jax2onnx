# tests/extra_tests/converter/test_source_original22_phase_observation.py

"""Phase observation controls preserve body execution and original exceptions."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.extra_tests.converter import (
    source_original22_phase_observation as observation,
)


@pytest.fixture(autouse=True)
def _isolated_descriptor(monkeypatch):
    monkeypatch.setattr(observation, "_FILE", None)
    yield
    entry = observation._FILE
    if entry is not None and entry[2] is not None:
        os.close(entry[2])


def _records(directory):
    paths = list(directory.glob("original22-phases-pid-*.jsonl"))
    assert len(paths) == 1
    return [json.loads(line) for line in paths[0].read_text().splitlines()]


def test_disabled_observation_does_no_path_proc_or_file_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled phase observation performed I/O")

    monkeypatch.setattr(observation, "Path", forbidden)
    monkeypatch.setattr(
        observation, "os", SimpleNamespace(environ={}, getpid=forbidden)
    )
    calls = []
    with observation.phase("factory", index=0, caller_x64=False):
        calls.append("original body")
    assert calls == ["original body"]


def test_enabled_and_disabled_body_results_calls_and_exception_identity_match(
    tmp_path, monkeypatch
):
    calls = []
    sentinel = RuntimeError("body payload must not be recorded")

    def original(raise_error):
        calls.append(raise_error)
        if raise_error:
            raise sentinel
        return (3, 7)

    results = []
    for enabled in (False, True):
        if enabled:
            monkeypatch.setenv(observation.OUTPUT_ENV, str(tmp_path))
        else:
            monkeypatch.delenv(observation.OUTPUT_ENV, raising=False)
        with observation.phase("native-references", index=0, caller_x64=False, batch=5):
            results.append(original(False))
        with pytest.raises(RuntimeError) as error:
            with observation.phase(
                "prepared-native", index=0, caller_x64=False, batch=5
            ):
                original(True)
        assert error.value is sentinel
    assert results == [(3, 7), (3, 7)]
    assert calls == [False, True, False, True]
    records = _records(tmp_path)
    assert [record["event"] for record in records] == [
        "enter",
        "returned",
        "enter",
        "raised",
    ]
    assert records[-1]["exception_type"] == "RuntimeError"
    assert "body payload must not be recorded" not in json.dumps(records)
    allowed = {
        "event",
        "phase",
        "pid",
        "monotonic_seconds",
        "index",
        "caller_x64",
        "batch",
        "optimize",
        "replay",
        "position",
        "exception_type",
        "VmRSS_bytes",
        "VmHWM_bytes",
        "RssAnon_bytes",
        "RssFile_bytes",
    }
    for record in records:
        assert set(record) <= allowed
        assert record["pid"] == os.getpid() and record["batch"] == 5
        assert "status" not in record
        if Path("/proc/self/status").exists():
            assert record["VmRSS_bytes"] > 0 and record["VmHWM_bytes"] > 0
    assert observation._FILE is not None
    assert observation._FILE[2] is not None
    assert os.get_inheritable(observation._FILE[2]) is False


def test_exclusive_creation_preserves_a_preexisting_observation_file(
    tmp_path, monkeypatch
):
    path = tmp_path / f"original22-phases-pid-{os.getpid()}.jsonl"
    path.write_text("preexisting evidence\n")
    monkeypatch.setenv(observation.OUTPUT_ENV, str(tmp_path))
    calls = []
    with observation.phase("factory", index=0, caller_x64=False):
        calls.append(True)
    assert calls == [True] and path.read_text() == "preexisting evidence\n"


def test_directory_change_keeps_one_descriptor_and_separate_files(
    tmp_path, monkeypatch
):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        monkeypatch.setenv(observation.OUTPUT_ENV, str(directory))
        with observation.phase("factory", index=0, caller_x64=False):
            pass
    assert [record["event"] for record in _records(first)] == ["enter", "returned"]
    assert [record["event"] for record in _records(second)] == ["enter", "returned"]
    assert observation._FILE is not None and observation._FILE[1] == str(
        second.resolve()
    )


@pytest.mark.parametrize("failure", ["open", "write", "proc"])
def test_observation_io_failure_never_replaces_the_original_body_outcome(
    tmp_path, monkeypatch, failure
):
    def unavailable(*args, **kwargs):
        raise OSError("injected observation failure")

    # Replace only this test-owned helper's aliases; global/native modules and
    # the original model/test callables remain untouched.
    monkeypatch.setenv(observation.OUTPUT_ENV, str(tmp_path))
    if failure == "proc":
        monkeypatch.setattr(observation, "_rss", unavailable)
    else:
        monkeypatch.setattr(
            observation,
            "os",
            SimpleNamespace(
                environ=os.environ,
                getpid=os.getpid,
                close=os.close,
                open=unavailable if failure == "open" else os.open,
                write=unavailable if failure == "write" else os.write,
                O_WRONLY=os.O_WRONLY,
                O_CREAT=os.O_CREAT,
                O_EXCL=os.O_EXCL,
                O_APPEND=os.O_APPEND,
                O_CLOEXEC=getattr(os, "O_CLOEXEC", 0),
            ),
        )
    calls = []
    with observation.phase("cpu-run", index=0, caller_x64=False, position=2):
        calls.append("returned")
    sentinel = ValueError("original body exception")
    with pytest.raises(ValueError) as error:
        with observation.phase("runtime-comparisons", index=0, caller_x64=False):
            calls.append("raised")
            raise sentinel
    assert error.value is sentinel and calls == ["returned", "raised"]


def test_unsupported_metadata_is_never_inspected_or_rendered(tmp_path, monkeypatch):
    class Payload:
        def __repr__(self):
            pytest.fail("A phase observer rendered a payload")

        def __array__(self, *args, **kwargs):
            pytest.fail("A phase observer materialized a payload")

    monkeypatch.setenv(observation.OUTPUT_ENV, str(tmp_path))
    calls = []
    with observation.phase(
        "source-export", index=0, caller_x64=False, position=Payload()
    ):
        calls.append(True)
    assert calls == [True] and list(tmp_path.iterdir()) == []
