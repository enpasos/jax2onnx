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

from tests.extra_tests.framework.test_ci_lane_selection import _node_files


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
    assert "node_runtime" not in manifest and "support_role" not in manifest
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


def _support_authority(authority, monkeypatch, role="normalization", minor="3.12"):
    module, root, path, _, run, original, distributions = authority
    data = deepcopy(original)
    data.update(
        schema="jax2onnx-ci-support-cohort-v1",
        support_role=role,
        python_minor=minor,
        collection_args=["tests/test_base.py"],
    )
    monkeypatch.setattr(
        module,
        "sys",
        SimpleNamespace(version_info=SimpleNamespace(major=3, minor=int(minor[2:]))),
    )
    if role == "minimum":
        installer = root / "scripts/install_minimum_dependencies.py"
        installer.parent.mkdir()
        installer.write_bytes(
            (_ROOT / "scripts/install_minimum_dependencies.py").read_bytes()
        )
        (root / "pyproject.toml").write_text(
            '[project]\nversion = "0.17.1"\ndependencies = ["pytest>=8.0"]\n'
        )
        data["packages"]["pytest"] = "8.0"
        data["plugins"][0]["version"] = "8.0"
        distributions[-1].version = "8.0"
        for name in module._binding_names(data):
            data["bindings"][name] = module._sha((root / name).read_bytes())
    if role == "node-smoke":
        node, _ = _node_files(root, root.parent / "node-prefix")
        data["node_runtime"] = {
            name: value for name, value in node.items() if name != "prefix"
        }
        for name in module._binding_names(data):
            data["bindings"][name] = module._sha((root / name).read_bytes())
        (root / ".gitignore").write_text(
            (root / ".gitignore").read_text() + "node_modules/\n"
        )
        module.os.environ["PATH"] = module.os.pathsep.join(
            (
                str(root.parent / "node-prefix/bin"),
                module.os.environ.get("PATH", module.os.defpath),
            )
        )
    _write_json(path, data)
    return module, root, path, run, data, distributions


@pytest.mark.parametrize(
    ("role", "minor"),
    [
        ("minimum", "3.12"),
        ("normalization", "3.12"),
        ("normalization", "3.13"),
        ("node-smoke", "3.12"),
    ],
)
def test_support_cohort_admits_only_the_four_original_role_cohorts(
    authority, monkeypatch, role, minor
):
    module, _, path, _, data, _ = _support_authority(
        authority, monkeypatch, role, minor
    )
    observed, _, _ = module._cohort(path, module._runner()._context_json)
    assert observed == data


@pytest.mark.parametrize(
    "fault", ["missing-node", "wrong-role", "reference-byte", "extra-node-key"]
)
def test_node_support_requires_its_source_bound_reference_variant(
    authority, monkeypatch, fault
):
    module, root, path, _, data, _ = _support_authority(
        authority, monkeypatch, "node-smoke"
    )
    if fault == "missing-node":
        del data["node_runtime"]
    elif fault == "wrong-role":
        data["support_role"] = "normalization"
    elif fault == "reference-byte":
        (root / "node-reference.json").write_bytes(b"{}\n")
    else:
        data["node_runtime"]["refresh_from_host"] = True
    _write_json(path, data)
    with pytest.raises(ValueError):
        module._cohort(path, module._runner()._context_json)


@pytest.mark.parametrize("config_name", ["NPM_CONFIG_CACHE", "npm_config_cache"])
def test_node_renderer_requires_explicit_prefix_and_binds_executed_sources(
    authority, monkeypatch, config_name
):
    module, root, path, run, data, _ = _support_authority(
        authority, monkeypatch, "node-smoke"
    )
    data["node_runtime"]["environment"][config_name] = "${NODE_PREFIX_PARENT}/npm-cache"
    module.os.environ[config_name] = str(root.parent / "npm-cache")
    _write_json(path, data)
    module._git(root, "add", ".")
    module._git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "reviewed Node fixture",
    )
    event = module._git(root, "rev-parse", "HEAD").decode().strip()
    with pytest.raises(ValueError, match="node-prefix"):
        module._manifest(path, root, event, run)
    manifest, _ = module._manifest(
        path, root, event, run, node_prefix=root.parent / "node-prefix"
    )
    assert manifest["support_role"] == "node-smoke"
    assert manifest["node_runtime"]["prefix"] == str(root.parent / "node-prefix")
    assert manifest["node_runtime"]["environment"][config_name] == str(
        root.parent / "npm-cache"
    )
    assert set(data["bindings"]) == {
        "poetry.lock",
        "pyproject.toml",
        *module._runner()._CI_NODE_SOURCE_FILES,
    }
    assert (
        manifest["source_files"]["node-reference.json"]
        == data["node_runtime"]["reference"]["sha256"]
    )
    assert not run.exists()


def test_python_renderer_rejects_node_prefix_without_changing_ordinary_shape(authority):
    module, root, path, event, run, _, _ = authority
    with pytest.raises(ValueError, match="only valid"):
        module._manifest(path, root, event, run, node_prefix=root)


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "cohort",
        "empty",
        "duplicate",
        "non-string",
        "toggle",
        "locked-role",
        "locked-selectors",
    ],
)
def test_support_cohort_rejects_role_selector_and_installation_policy_overrides(
    authority, monkeypatch, fault
):
    module, _, path, _, data, _ = _support_authority(authority, monkeypatch)
    if fault == "role":
        data["support_role"] = "arbitrary"
    elif fault == "cohort":
        data["python_minor"] = "3.11"
        module.sys.version_info.minor = 11
    elif fault == "empty":
        data["collection_args"] = []
    elif fault == "duplicate":
        data["collection_args"] *= 2
    elif fault == "non-string":
        data["collection_args"] = [1]
    elif fault == "toggle":
        data["installation_policy"] = "minimum"
    else:
        data["schema"] = "jax2onnx-ci-standard-cohort-v1"
        del data["collection_args" if fault == "locked-role" else "support_role"]
    _write_json(path, data)
    with pytest.raises(ValueError):
        module._cohort(path, module._runner()._context_json)


def test_support_renderer_keeps_ordered_command_and_absent_generated_directories(
    authority, monkeypatch
):
    module, root, path, run, data, _ = _support_authority(authority, monkeypatch)
    (root / "tests/test_other.py").write_text("def test_exact(): pass\n")
    data["collection_args"].append("tests/test_other.py::test_exact")
    data["baseline"]["nodeids"].append(data["collection_args"][-1])
    for name in data["baseline"]["generated"]:
        (root / name).unlink()
    (root / "tests/examples").rmdir()
    (root / "tests/primitives").rmdir()
    data["baseline"]["generated"] = {}
    _write_json(path, data)
    module._git(root, "add", ".")
    module._git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "support",
    )
    event = module._git(root, "rev-parse", "HEAD").decode().strip()
    manifest, _ = module._manifest(path, root, event, run)
    assert manifest["collection_args"] == data["collection_args"]
    assert manifest["nodeids"] == data["baseline"]["nodeids"]


def test_minimum_support_uses_exact_frozen_packages_outside_foundation_lock(
    authority, monkeypatch
):
    module, root, _, _, data, _ = _support_authority(authority, monkeypatch, "minimum")
    # This direct floor deliberately differs from the locked foundation.
    module._installation(data, root)
    assert module._binding_names(data) == {
        "poetry.lock",
        "pyproject.toml",
        "scripts/install_minimum_dependencies.py",
    }


@pytest.mark.parametrize(
    "fault",
    [
        "missing-package",
        "package",
        "extra-package",
        "bootstrap",
        "extra-bootstrap",
        "project",
        "plugin",
    ],
)
def test_minimum_support_rejects_distribution_bootstrap_and_plugin_drift(
    authority, monkeypatch, fault
):
    module, root, _, _, data, distributions = _support_authority(
        authority, monkeypatch, "minimum"
    )
    if fault == "missing-package":
        distributions.pop()
    elif fault == "package":
        distributions[-1].version = "wrong"
    elif fault == "extra-package":
        distributions.append(
            SimpleNamespace(metadata={"Name": "foreign"}, version="1", entry_points=[])
        )
    elif fault == "bootstrap":
        data["bootstrap"]["pip"] = "wrong"
    elif fault == "extra-bootstrap":
        data["bootstrap"]["setuptools"] = "1"
    elif fault == "project":
        (root / "pyproject.toml").write_text(
            '[project]\nversion = "different"\ndependencies = ["pytest>=8.0"]\n'
        )
    else:
        distributions[-1].entry_points[0].value = "foreign.plugin"
    with pytest.raises(ValueError):
        module._installation(data, root)


@pytest.mark.parametrize(
    "binding",
    ["pyproject.toml", "scripts/install_minimum_dependencies.py", "poetry.lock"],
)
def test_minimum_support_rejects_altered_project_installer_and_foundation_bindings(
    authority, monkeypatch, binding
):
    module, root, path, run, data, _ = _support_authority(
        authority, monkeypatch, "minimum"
    )
    data["bindings"][binding] = "0" * 64
    _write_json(path, data)
    module._git(root, "add", ".")
    module._git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "support",
    )
    event = module._git(root, "rev-parse", "HEAD").decode().strip()
    with pytest.raises(ValueError, match="binding"):
        module._manifest(path, root, event, run)


@pytest.mark.parametrize(
    "selectors",
    [
        [],
        ["tests/test_base.py"] * 2,
        ["tests/test_base.py::TestBase"],
        ["tests/test_base.py::test_base[value]"],
        ["tests"],
        ["../test_base.py"],
        ["tests/*.py"],
        ["--pyargs"],
        [1],
        [{}],
    ],
)
def test_main_owner_cohort_rejects_nonliteral_or_partial_module_authority(
    authority, selectors
):
    module, _root, path, _event, _run, original, _distributions = authority
    data = deepcopy(original)
    data.update(schema="jax2onnx-ci-main-owner-cohort-v1", collection_args=selectors)
    _write_json(path, data)
    with pytest.raises(ValueError, match="Main-owner"):
        module._cohort(path, module._runner()._context_json)


def test_main_owner_cohort_keeps_literal_module_order_without_item_inference(authority):
    module, _root, path, _event, _run, original, _distributions = authority
    data = deepcopy(original)
    data.update(
        schema="jax2onnx-ci-main-owner-cohort-v1",
        collection_args=["tests/test_base.py", "tests/primitives/test_generated.py"],
    )
    _write_json(path, data)
    observed, baseline, _bindings = module._cohort(path, module._runner()._context_json)
    assert observed == data and baseline == data["baseline"]
    assert observed["collection_args"] == data["collection_args"]


@pytest.mark.parametrize(
    "schema", ["jax2onnx-ci-standard-cohort-v1", "jax2onnx-ci-support-cohort-v1"]
)
def test_existing_cohort_schemas_do_not_accept_owner_authority_by_extra_keys(
    authority, schema
):
    module, _root, path, _event, _run, original, _distributions = authority
    data = deepcopy(original)
    data.update(
        schema=schema,
        collection_args=["tests/test_base.py"],
        collection_scope="main-owners",
    )
    if schema == "jax2onnx-ci-support-cohort-v1":
        data["support_role"] = "normalization"
    _write_json(path, data)
    with pytest.raises(ValueError, match="unsupported STANDARD cohort"):
        module._cohort(path, module._runner()._context_json)


@pytest.mark.parametrize("scope", [None, "main-owners", "global"])
def test_owner_renderer_binds_only_its_two_declared_scope_materializations(
    authority, scope
):
    module, root, path, _event, run, original, _distributions = authority
    data = deepcopy(original)
    data.update(
        schema="jax2onnx-ci-main-owner-cohort-v1",
        collection_args=["tests/test_base.py"],
    )
    _write_json(path, data)
    module._git(root, "add", ".")
    module._git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "reviewed owner fixture",
    )
    event = module._git(root, "rev-parse", "HEAD").decode().strip()
    manifest, observed = module._manifest(
        path, root, event, run, collection_scope=scope
    )
    assert observed == data
    assert manifest["nodeids"] == data["baseline"]["nodeids"]
    assert (
        manifest["source_files"]["cohort.json"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    if scope == "global":
        assert manifest["collection_args"] == [] and "collection_scope" not in manifest
    else:
        assert manifest["collection_scope"] == "main-owners"
        assert manifest["collection_args"] == data["collection_args"]


@pytest.mark.parametrize("scope", ["main-owners", "global", "unknown"])
def test_existing_global_renderer_does_not_authorize_owner_scope_controls(
    authority, scope
):
    module, root, path, event, run, _data, _distributions = authority
    with pytest.raises(ValueError, match="approved Main-owner cohort"):
        module._manifest(path, root, event, run, collection_scope=scope)


def test_owner_cohort_rejects_omitted_Main_counterpart_module(authority):
    module, _root, path, _event, _run, original, _distributions = authority
    data = deepcopy(original)
    data.update(
        schema="jax2onnx-ci-main-owner-cohort-v1",
        collection_args=["tests/test_other.py"],
    )
    _write_json(path, data)
    with pytest.raises(ValueError, match="omit an original Main counterpart"):
        module._cohort(path, module._runner()._context_json)
