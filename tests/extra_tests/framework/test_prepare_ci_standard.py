# tests/extra_tests/framework/test_prepare_ci_standard.py

"""Portable admission is checked against independent, literal cohort data."""

from copy import deepcopy
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[3]


def _preparation():
    spec = spec_from_file_location(
        "_standard_preparation_under_test", _ROOT / "scripts/prepare_ci_standard.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def authority(tmp_path, monkeypatch):
    module = _preparation()
    runner = module._runner()
    root = tmp_path / "checkout"
    root.mkdir()
    sources = {
        ".gitignore": "tests/primitives/\ntests/examples/\n",
        "pyproject.toml": '[project]\nversion = "0.17.1"\n',
        "poetry.lock": '[[package]]\nname = "pytest"\nversion = "9.1.1"\n',
        "tests/test_base.py": "def test_base(): pass\n",
    }
    for name, text in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    generated = {}
    for name in ("tests/examples/__init__.py", "tests/primitives/test_generated.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "" if name.endswith("__init__.py") else "def test_generated(): pass\n"
        )
        generated[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plugin = {
        "name": "example",
        "value": "example.plugin",
        "distribution": "pytest",
        "version": "9.1.1",
    }
    environment = runner._ci_lane_environment(
        {
            "PYTHONPATH": "${CHECKOUT}",
            "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR": "${RUN}/native-model-artifacts",
        }
    )
    jax = {"status": "NOT_LOADED", "values": {}}
    data = {
        "schema": "jax2onnx-ci-standard-cohort-v1",
        "python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "baseline_commit": "a" * 40,
        "baseline": {
            "nodeids": ["tests/test_base.py::test_base"],
            "generated": generated,
        },
        "bindings": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("poetry.lock", "pyproject.toml")
        },
        "packages": {"pip": "25.1.1", "jax2onnx": "0.17.1", "pytest": "9.1.1"},
        "bootstrap": {"pip": "25.1.1"},
        "plugins": [plugin],
        "environment": environment,
        "collection_transition": {
            "startup_jax_config": jax,
            "collected_environment": environment,
            "collected_jax_config": jax,
            "post_pytest_environment": environment,
            "post_pytest_jax_config": jax,
        },
        "allowed_skips": [],
        "provenance": {},
    }
    path = root / "cohort.json"
    _write_json(path, data)
    module._git(root, "init", "-q")
    module._git(root, "add", ".")
    module._git(
        root,
        "-c",
        "user.name=Synthetic fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    event = module._git(root, "rev-parse", "HEAD").decode().strip()
    distributions = [
        SimpleNamespace(
            metadata={"Name": name},
            version=version,
            entry_points=(
                [
                    SimpleNamespace(
                        group="pytest11", name="example", value="example.plugin"
                    )
                ]
                if name == "pytest"
                else []
            ),
        )
        for name, version in data["packages"].items()
    ]
    monkeypatch.setattr(module.metadata, "distributions", lambda: distributions)
    monkeypatch.setattr(module.os, "environ", {"PYTHONPATH": str(root)})
    return module, root, path, event, tmp_path / "future-run", data, distributions


def test_portable_manifest_binds_full_git_generated_and_declared_context(authority):
    module, root, path, event, run, data, _ = authority
    manifest, _ = module._manifest(path, root, event, run)
    assert manifest["nodeids"] == data["baseline"]["nodeids"]
    assert manifest["source_files"] == {
        **module._tracked(root, event),
        **data["baseline"]["generated"],
    }
    assert manifest["environment"]["PYTHONPATH"] == str(root)
    assert manifest["environment"]["JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR"] == str(
        run / "native-model-artifacts"
    )
    assert not run.exists()
    assert "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR" not in module.os.environ
    assert (
        manifest["collection_transition"]["startup_jax_config"]
        == data["collection_transition"]["startup_jax_config"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "head",
        "tracked",
        "untracked",
        "generated-byte",
        "generated-extra",
        "generated-missing",
        "generated-link",
        "run-exists",
        "native-parent",
        "profile",
    ],
)
def test_portable_manifest_rejects_source_output_and_context_drift(authority, fault):
    module, root, path, event, run, _, _ = authority
    if fault == "head":
        event = "0" * 40
    elif fault == "tracked":
        (root / "tests/test_base.py").write_text("changed\n")
    elif fault == "untracked":
        (root / "foreign.py").write_text("unreviewed\n")
    elif fault == "generated-byte":
        (root / "tests/primitives/test_generated.py").write_text("changed\n")
    elif fault == "generated-extra":
        (root / "tests/examples/extra.txt").write_text("extra\n")
    elif fault == "generated-missing":
        (root / "tests/examples/__init__.py").unlink()
    elif fault == "generated-link":
        item = root / "tests/primitives/test_generated.py"
        item.unlink()
        item.symlink_to(root / "tests/test_base.py")
    elif fault == "run-exists":
        run.mkdir()
    elif fault == "native-parent":
        module.os.environ["JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR"] = ""
    else:
        module.os.environ["JAX_ENABLE_X64"] = "True"
    with pytest.raises(ValueError):
        module._manifest(path, root, event, run)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "wrong",
        "extra",
        "duplicate",
        "plugin-missing",
        "plugin-wrong",
        "plugin-extra",
        "unlocked",
        "bootstrap",
    ],
)
def test_portable_installation_rejects_unknown_packages_and_plugin_drift(
    authority, fault
):
    module, root, _, _, _, original, distributions = authority
    data = deepcopy(original)
    if fault == "missing":
        distributions.pop()
    elif fault == "wrong":
        distributions[-1].version = "wrong"
    elif fault == "extra":
        distributions.append(
            SimpleNamespace(metadata={"Name": "unknown"}, version="1", entry_points=[])
        )
    elif fault == "duplicate":
        distributions.append(distributions[-1])
    elif fault == "plugin-missing":
        distributions[-1].entry_points = []
    elif fault == "plugin-wrong":
        distributions[-1].entry_points[0].value = "foreign.plugin"
    elif fault == "plugin-extra":
        distributions[-1].entry_points.append(
            SimpleNamespace(group="pytest11", name="foreign", value="foreign.plugin")
        )
    elif fault == "unlocked":
        data["packages"]["pytest"] = "future"
    else:
        data["bootstrap"]["pip"] = "future"
    with pytest.raises(ValueError):
        module._installation(data, root)


def test_portable_path_slots_do_not_rewrite_context_literals():
    module = _preparation()
    assert module._environment(
        {"JAX_ENABLE_X64": "True", "PYTHONPATH": "${CHECKOUT}"},
        {"${CHECKOUT}": "/fresh"},
    ) == {"JAX_ENABLE_X64": "True", "PYTHONPATH": "/fresh"}
    for template in (
        {"XLA_FLAGS": "${CHECKOUT}"},
        {"PYTHONPATH": "${CHECKOUT}/extra"},
        {"PYTHONPATH": "${UNKNOWN}"},
    ):
        with pytest.raises(ValueError):
            module._environment(template, {"${CHECKOUT}": "/fresh"})


def test_shared_baseline_requires_exact_hash_and_duplicate_keys_are_rejected(authority):
    module, root, path, _, _, data, _ = authority
    shared = root / "shared.json"
    _write_json(shared, data["baseline"])
    data["baseline"] = {
        "path": "shared.json",
        "sha256": hashlib.sha256(shared.read_bytes()).hexdigest(),
    }
    _write_json(path, data)
    _, baseline, _ = module._cohort(path, module._runner()._context_json)
    assert baseline["nodeids"] == ["tests/test_base.py::test_base"]
    shared.write_text(shared.read_text() + " ")
    with pytest.raises(ValueError, match="binding"):
        module._cohort(path, module._runner()._context_json)
    path.write_text('{"schema": 1, "schema": 2}')
    with pytest.raises(ValueError, match="duplicate"):
        module._cohort(path, module._runner()._context_json)


def test_renderer_authenticates_the_same_cohort_bytes_it_interprets(
    authority, monkeypatch
):
    module, root, path, event, run, _, _ = authority
    runner = module._runner()
    original = runner._context_json

    def transient_read(candidate, *args, **kwargs):
        data, raw = original(candidate, *args, **kwargs)
        if candidate == path:
            data["allowed_skips"] = [{"reason": "transient unreviewed data"}]
            raw = (json.dumps(data) + "\n").encode()
        return data, raw

    monkeypatch.setattr(runner, "_context_json", transient_read)
    monkeypatch.setattr(module, "_runner", lambda: runner)
    # The file is already back at its Git bytes when the read returns. A
    # separate read_bytes() authentication would validate the wrong payload.
    with pytest.raises(ValueError, match="read cohort bytes"):
        module._manifest(path, root, event, run)


@pytest.mark.parametrize("destination", ["run", "output"])
@pytest.mark.parametrize("alias", ["parent-symlink", "dotdot", "broken-leaf"])
def test_future_destinations_reject_checkout_aliases_and_existing_symlink_leaves(
    authority, monkeypatch, destination, alias
):
    module, root, path, event, run, _, _ = authority
    if alias == "parent-symlink":
        parent = root.parent / "alias"
        parent.symlink_to(root, target_is_directory=True)
        target = parent / "fresh"
    elif alias == "dotdot":
        parent = root.parent / "outside"
        parent.mkdir()
        target = parent / ".." / root.name / "fresh"
    else:
        target = root.parent / "broken"
        target.symlink_to(root.parent / "absent")
    if destination == "run":
        with pytest.raises(ValueError, match="fresh and external"):
            module._manifest(path, root, event, target)
    else:
        monkeypatch.setattr(
            module, "__file__", str(root / "scripts/prepare_ci_standard.py")
        )
        monkeypatch.setattr(
            module.sys,
            "argv",
            [
                "prepare",
                "--cohort",
                str(path),
                "--event-sha",
                event,
                "--run-dir",
                str(run),
                "--output",
                str(target),
            ],
        )
        with pytest.raises(SystemExit) as error:
            module._main()
        assert error.value.code == 2
    assert not (root / "fresh").exists()
    assert not (root.parent / "absent").exists()


def test_future_run_uses_the_same_canonical_path_as_the_coordinator(authority):
    module, root, path, event, run, _, _ = authority
    alias = root.parent / "external-alias"
    real_parent = root.parent / "external"
    real_parent.mkdir()
    alias.symlink_to(real_parent, target_is_directory=True)
    manifest, _ = module._manifest(path, root, event, alias / run.name)
    assert manifest["environment"]["JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR"] == str(
        (alias / run.name).resolve() / "native-model-artifacts"
    )
