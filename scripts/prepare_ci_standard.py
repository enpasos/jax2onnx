# scripts/prepare_ci_standard.py

"""Render reviewed STANDARD cohort data into an external canonical manifest.

This stdlib-only admission step neither installs packages nor imports pytest/JAX.
Run it after the original generator, before the existing batch coordinator.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import Any


_SUPPORT_COHORTS = {
    "minimum": {"3.12"},
    "normalization": {"3.12", "3.13"},
    "node-smoke": {"3.12"},
}


def _runner() -> Any:
    spec = spec_from_file_location(
        "_standard_authority_runner", Path(__file__).with_name("run_pytest_batches.py")
    )
    if spec is None or spec.loader is None:
        raise ValueError("canonical runner is unavailable")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _relative(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
        or Path(value).as_posix() != value
    ):
        raise ValueError("authority paths must be checkout-relative")
    return value


def _git(root: Path, *arguments: str, input: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        input=input,
        capture_output=True,
        check=True,
    ).stdout


def _tracked(root: Path, event_sha: str) -> dict[str, str]:
    if not re.fullmatch("[0-9a-f]{40}", event_sha) or (
        _git(root, "rev-parse", "HEAD").decode().strip() != event_sha
    ):
        raise ValueError("checkout HEAD differs from the external event SHA")
    entries = []
    for row in _git(root, "ls-tree", "-rz", event_sha).split(b"\0"):
        if row:
            header, path = row.split(b"\t", 1)
            mode, kind, oid = header.split()
            if kind != b"blob" or mode not in (b"100644", b"100755"):
                raise ValueError("unsupported tracked source kind")
            entries.append((_relative(os.fsdecode(path)), oid))
    objects = _git(
        root, "cat-file", "--batch", input=b"\n".join(oid for _, oid in entries) + b"\n"
    )
    sources = {}
    for name, oid in entries:
        header, objects = objects.split(b"\n", 1)
        actual_oid, kind, size = header.split()
        length = int(size)
        raw, objects = objects[:length], objects[length + 1 :]
        path = root / name
        if (
            actual_oid != oid
            or kind != b"blob"
            or path.is_symlink()
            or path.read_bytes() != raw
        ):
            raise ValueError(f"tracked source differs from event Git object: {name}")
        sources[name] = _sha(raw)
    if objects or _git(root, "diff", "--name-only", event_sha, "--"):
        raise ValueError("tracked source modes/index differ from the event")
    if _git(root, "ls-files", "--others", "--exclude-standard"):
        raise ValueError("unreviewed untracked checkout files")
    return sources


def _cohort(
    path: Path, read_json: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[Path, str]]:
    data, raw = read_json(path, 4 * 1024 * 1024, label="STANDARD cohort")
    bindings = {path.resolve(strict=True): _sha(raw)}
    support = (
        type(data) is dict and data.get("schema") == "jax2onnx-ci-support-cohort-v1"
    )
    owner = (
        type(data) is dict and data.get("schema") == "jax2onnx-ci-main-owner-cohort-v1"
    )
    node = support and data.get("support_role") == "node-smoke"
    if (
        type(data) is not dict
        or set(data)
        != {
            "schema",
            "python_minor",
            "baseline_commit",
            "baseline",
            "bindings",
            "packages",
            "bootstrap",
            "plugins",
            "environment",
            "collection_transition",
            "allowed_skips",
            "provenance",
        }
        | ({"support_role", "collection_args"} if support else set())
        | ({"node_runtime"} if node else set())
        | ({"collection_args"} if owner else set())
        | ({"standard_parts"} if owner and "standard_parts" in data else set())
        | ({"focused_extra_ids"} if "focused_extra_ids" in data else set())
        | ({"extra_allowed_skips"} if "extra_allowed_skips" in data else set())
        or data["schema"]
        not in {
            "jax2onnx-ci-standard-cohort-v1",
            "jax2onnx-ci-support-cohort-v1",
            "jax2onnx-ci-main-owner-cohort-v1",
        }
        or data["python_minor"] != f"{sys.version_info.major}.{sys.version_info.minor}"
    ):
        raise ValueError("unsupported STANDARD cohort")
    if "focused_extra_ids" in data and (
        owner
        or support
        or type(data["focused_extra_ids"]) is not list
        or not data["focused_extra_ids"]
        or any(type(node) is not str or not node for node in data["focused_extra_ids"])
        or len(set(data["focused_extra_ids"])) != len(data["focused_extra_ids"])
    ):
        raise ValueError("focused extras need nonempty unique full-global cohort IDs")
    if "extra_allowed_skips" in data and type(data["extra_allowed_skips"]) is not list:
        raise ValueError("EXTRA cohort allowances must be a list")
    if support and (
        type(data["support_role"]) is not str
        or data["support_role"] not in _SUPPORT_COHORTS
        or data["python_minor"] not in _SUPPORT_COHORTS[data["support_role"]]
        or type(data["collection_args"]) is not list
        or not data["collection_args"]
        or any(type(value) is not str or not value for value in data["collection_args"])
        or len(set(data["collection_args"])) != len(data["collection_args"])
    ):
        raise ValueError("unsupported support role/cohort or ordered selectors")
    if owner:
        _runner()._ci_owner_modules(data["collection_args"])
    baseline = data["baseline"]
    if type(baseline) is dict and set(baseline) == {"path", "sha256"}:
        source = path.parent / _relative(baseline["path"])
        baseline, raw = read_json(source, 4 * 1024 * 1024, label="shared baseline")
        if _sha(raw) != data["baseline"]["sha256"]:
            raise ValueError("shared baseline binding differs")
        bindings[source.resolve(strict=True)] = _sha(raw)
    if type(baseline) is not dict or set(baseline) != {"nodeids", "generated"}:
        raise ValueError("STANDARD baseline needs literal IDs and generated bindings")
    if owner and (
        type(baseline["nodeids"]) is not list
        or any(
            type(nodeid) is not str
            or nodeid.split("::", 1)[0] not in data["collection_args"]
            for nodeid in baseline["nodeids"]
        )
    ):
        raise ValueError("Main-owner modules omit an original Main counterpart owner")
    if node:
        authority = data["node_runtime"]
        if (
            type(authority) is not dict
            or set(authority) != {"reference", "environment"}
            or type(authority["environment"]) is not dict
            or type(authority["reference"]) is not dict
            or set(authority["reference"]) != {"path", "sha256"}
        ):
            raise ValueError("Node support needs its reviewed portable runtime binding")
        source = path.parent / _relative(authority["reference"]["path"])
        reference, raw = read_json(
            source, 4 * 1024 * 1024, label="Node runtime reference"
        )
        if _sha(raw) != authority["reference"]["sha256"]:
            raise ValueError("reviewed Node reference binding differs")
        _runner()._ci_node_reference(reference)
        bindings[source.resolve(strict=True)] = _sha(raw)
    return data, baseline, bindings


def _binding_names(data: dict[str, Any]) -> set[str]:
    names = {"poetry.lock", "pyproject.toml"}
    if data.get("support_role") == "minimum":
        names.add("scripts/install_minimum_dependencies.py")
    if data.get("support_role") == "node-smoke":
        names.update(_runner()._CI_NODE_SOURCE_FILES)
    return names


def _installation(data: dict[str, Any], root: Path) -> None:
    lock = tomllib.loads((root / "poetry.lock").read_text())
    locked: dict[str, set[str]] = {}
    for package in lock["package"]:
        locked.setdefault(_name(package["name"]), set()).add(package["version"])
    expected = data["packages"]
    bootstrap = data["bootstrap"]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    # Minimum authority freezes the complete independently reviewed installation;
    # its bound lock records the foundation, rather than resolving floors here.
    if (
        type(expected) is not dict
        or type(bootstrap) is not dict
        or set(bootstrap) != {"pip"}
        or expected.get("pip") != bootstrap["pip"]
        or expected.get("jax2onnx") != project["version"]
        or any(
            type(name) is not str
            or name != _name(name)
            or type(version) is not str
            or not version
            or (
                name not in {"pip", "jax2onnx"}
                and data.get("support_role") != "minimum"
                and version not in locked.get(name, set())
            )
            for name, version in expected.items()
        )
    ):
        raise ValueError("packages differ from lock/project/declared bootstrap pins")
    actual = {}
    plugins = []
    for distribution in metadata.distributions():
        name = _name(distribution.metadata["Name"])
        if name in actual:
            raise ValueError("duplicate installed distribution")
        actual[name] = distribution.version
        plugins.extend(
            {
                "name": entry.name,
                "value": entry.value,
                "distribution": name,
                "version": distribution.version,
            }
            for entry in distribution.entry_points
            if entry.group == "pytest11"
        )
    if actual != expected:
        raise ValueError("installed package set/versions differ from reviewed cohort")
    if (
        sorted(plugins, key=lambda value: (value["name"], value["distribution"]))
        != data["plugins"]
    ):
        raise ValueError("installed pytest plugin set differs from reviewed cohort")


def _environment(template: Any, slots: dict[str, str]) -> dict[str, Any]:
    if type(template) is not dict:
        raise ValueError("environment template must be a mapping")
    allowed = {
        "PYTHONPATH": "${CHECKOUT}",
        "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR": "${RUN}/native-model-artifacts",
        "JAX2ONNX_MAXTEXT_SRC": "${MAXTEXT_SOURCE}",
        "JAX2ONNX_MAXDIFFUSION_SRC": "${MAXDIFFUSION_SOURCE}",
    }
    result = {}
    for name, value in template.items():
        if type(value) is str and "${" in value:
            if allowed.get(name) != value or value not in slots:
                raise ValueError("unreviewed environment path slot")
            value = slots[value]
        result[name] = value
    return result


def _node_binding(
    data: dict[str, Any], path: Path, root: Path, run: Path, node_prefix: Path | None
) -> dict[str, Any] | None:
    if data.get("support_role") != "node-smoke":
        if node_prefix is not None:
            raise ValueError("node-prefix is only valid for reviewed Node support")
        return None
    if node_prefix is None:
        raise ValueError("Node support needs an explicit portable node-prefix")
    prefix = node_prefix.resolve(strict=True)
    expected = data["node_runtime"]
    binding = expected["reference"]
    source = (path.parent / _relative(binding["path"])).resolve(strict=True)
    if not source.is_relative_to(root):
        raise ValueError("reviewed Node reference escapes the checkout")
    environment = dict(expected["environment"])
    suffixes = {
        "NPM_CONFIG_CACHE": ("/npm-cache", "/npm-cache"),
        "NPM_CONFIG_USERCONFIG": ("/node-config/user.npmrc", "/config/user.npmrc"),
        "NPM_CONFIG_GLOBALCONFIG": (
            "/node-config/global.npmrc",
            "/config/global.npmrc",
        ),
    }
    for name, value in environment.items():
        if value is None:
            continue
        suffix = suffixes.get(name.upper())
        allowed = (
            {
                "${RUN}" + suffix[0]: str(run) + suffix[0],
                "${NODE_PREFIX_PARENT}" + suffix[1]: str(prefix.parent) + suffix[1],
            }
            if suffix is not None
            else {}
        )
        if type(value) is not str or value not in allowed:
            raise ValueError(
                "unreviewed Node environment path slot or execution overlay"
            )
        environment[name] = allowed[value]
    return {
        "reference": {
            "path": source.relative_to(root).as_posix(),
            "sha256": binding["sha256"],
        },
        "prefix": str(prefix),
        "environment": environment,
    }


def _manifest(
    cohort_path: Path,
    root: Path,
    event_sha: str,
    run: Path,
    optional_sources: dict[str, Path] | None = None,
    node_prefix: Path | None = None,
    *,
    collection_scope: str | None = None,
    standard_part: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runner = _runner()
    root = root.resolve(strict=True)
    path = cohort_path.resolve(strict=True)
    if os.path.lexists(run):
        raise ValueError("future run directory must be fresh and external")
    run = run.resolve(strict=False)
    if run.is_relative_to(root) or os.path.lexists(run):
        raise ValueError("future run directory must be fresh and external")
    if "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR" in os.environ:
        raise ValueError("parent native evidence key must be genuinely absent")
    sources = _tracked(root, event_sha)
    if path.relative_to(root).as_posix() not in sources:
        raise ValueError("cohort input must be an event-bound tracked source")
    data, baseline, read_bindings = _cohort(path, runner._context_json)
    owner = data["schema"] == "jax2onnx-ci-main-owner-cohort-v1"
    if collection_scope is not None and (
        not owner or collection_scope not in {"main-owners", "global"}
    ):
        raise ValueError("only an approved Main-owner cohort has two collection scopes")
    owner_rendering = owner and collection_scope != "global"
    part_marker = None
    if "standard_parts" in data:
        if (
            not owner
            or data["python_minor"] != "3.12"
            or standard_part not in {"a", "b"}
        ):
            raise ValueError(
                "reviewed split cohort requires exactly one fixed STANDARD part"
            )
        parts = runner._ci_standard_parts(baseline["nodeids"], data["standard_parts"])
        run_id, attempt = os.environ.get("GITHUB_RUN_ID", ""), os.environ.get(
            "GITHUB_RUN_ATTEMPT", ""
        )
        if any(
            re.fullmatch("[1-9][0-9]*", value) is None for value in (run_id, attempt)
        ):
            raise ValueError(
                "STANDARD part needs the actual hosted run/attempt identity"
            )
        part_marker = {
            "id": standard_part,
            "profile": path.relative_to(root).as_posix(),
            "profile_sha256": read_bindings[path],
            "split_after": data["standard_parts"]["split_after"],
            "full_s_sha256": runner._ci_standard_ids_digest(baseline["nodeids"]),
            "assigned_sha256": runner._ci_standard_ids_digest(parts[standard_part]),
            "checkout": event_sha,
            "run_id": run_id,
            "run_attempt": attempt,
        }
    elif standard_part is not None:
        raise ValueError("unsplit cohort cannot accept a STANDARD part")
    if any(
        sources.get(read_path.relative_to(root).as_posix()) != digest
        for read_path, digest in read_bindings.items()
    ):
        raise ValueError("read cohort bytes differ from event-bound source")
    for name, expected in data["bindings"].items():
        if sources.get(_relative(name)) != expected:
            raise ValueError("cohort lock/project binding differs")
    if set(data["bindings"]) != _binding_names(data):
        raise ValueError("cohort must bind lock/project and its minimum installer")
    if set(data["baseline"]) == {"path", "sha256"}:
        shared = (path.parent / data["baseline"]["path"]).relative_to(root).as_posix()
        if sources.get(shared) != data["baseline"]["sha256"]:
            raise ValueError("shared baseline must be event-bound")
    generated = {}
    for directory in ("tests/primitives", "tests/examples"):
        for item in (root / directory).rglob("*"):
            if item.is_symlink():
                raise ValueError("generated output cannot be a symlink")
            if item.is_file() and "__pycache__" not in item.parts:
                generated[item.relative_to(root).as_posix()] = _sha(item.read_bytes())
    if generated != baseline["generated"]:
        raise ValueError("generated output set/bytes differ from reviewed cohort")
    sources.update(generated)
    _installation(data, root)
    slots = {
        "${CHECKOUT}": str(root),
        "${RUN}/native-model-artifacts": str(run / "native-model-artifacts"),
    }
    for name, source in (optional_sources or {}).items():
        if name not in {"MAXTEXT", "MAXDIFFUSION"}:
            raise ValueError("unknown optional source slot")
        resolved = source.resolve(strict=True)
        ref = data["environment"][f"JAX2ONNX_{name}_REF"]
        if _git(resolved, "rev-parse", "HEAD").decode().strip() != ref or _git(
            resolved, "status", "--porcelain"
        ):
            raise ValueError("optional source checkout/ref differs")
        slots[f"${{{name}_SOURCE}}"] = str(resolved)
    environment = _environment(data["environment"], slots)
    actual = {
        **os.environ,
        "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR": slots["${RUN}/native-model-artifacts"],
    }
    if runner._ci_lane_environment(actual) != environment:
        raise ValueError("parent environment differs from the declared worker profile")
    transition = dict(data["collection_transition"])
    for name in ("collected_environment", "post_pytest_environment"):
        transition[name] = _environment(transition[name], slots)
    runner._ci_lane_transition(transition)
    node = _node_binding(data, path, root, run, node_prefix)
    if node is not None:
        reference = runner._ci_node_load(root, node, sources)
        runner._ci_node_check(root, node, reference, full=True)
    return {
        "schema": "jax2onnx-ci-baseline-v1",
        **({"standard_part": part_marker} if part_marker is not None else {}),
        "baseline_commit": data["baseline_commit"],
        "python_minor": data["python_minor"],
        "nodeids": baseline["nodeids"],
        **(
            {"focused_extra_ids": data["focused_extra_ids"]}
            if "focused_extra_ids" in data
            else {}
        ),
        "source_files": dict(sorted(sources.items())),
        "packages": data["packages"],
        "collection_args": (
            data.get("collection_args", []) if not owner or owner_rendering else []
        ),
        **({"collection_scope": "main-owners"} if owner_rendering else {}),
        "environment": environment,
        "collection_transition": transition,
        **(
            {"support_role": "node-smoke", "node_runtime": node}
            if node is not None
            else {}
        ),
    }, data


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--event-sha", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxtext-source", type=Path)
    parser.add_argument("--maxdiffusion-source", type=Path)
    parser.add_argument("--node-prefix", type=Path)
    parser.add_argument("--collection-scope", choices=("main-owners", "global"))
    parser.add_argument("--ci-standard-part", choices=("a", "b"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if os.path.lexists(args.output):
        parser.error("manifest output must be fresh and external")
    output = args.output.resolve(strict=False)
    if output.is_relative_to(root) or os.path.lexists(output):
        parser.error("manifest output must be fresh and external")
    optional = {
        name: source
        for name, source in (
            ("MAXTEXT", args.maxtext_source),
            ("MAXDIFFUSION", args.maxdiffusion_source),
        )
        if source is not None
    }
    manifest, _ = _manifest(
        args.cohort,
        root,
        args.event_sha,
        args.run_dir,
        optional,
        args.node_prefix,
        collection_scope=args.collection_scope,
        standard_part=args.ci_standard_part,
    )
    with output.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
