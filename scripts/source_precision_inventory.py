# scripts/source_precision_inventory.py

"""Inventory existing functionality without treating legacy tests as SOURCE evidence.

The default command compares the raw registry and handwritten regression identities
with the reviewed baseline. ``--refresh`` deliberately updates that baseline; its
diff still needs review. No generated-test precision rewrites or model execution
are used here. Optional examples remain visible when their environment is absent.
"""

from __future__ import annotations

import argparse
import copy
import ast
from collections import Counter
from dataclasses import asdict
from collections.abc import Mapping, Sequence
import json
import re
import hashlib
import functools
import types
import subprocess
import io
import tarfile
from enum import Enum
import os
from numbers import Integral
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = (
    REPOSITORY_ROOT / "internal_docs/precision/source_migration_inventory.json"
)
FEATURES = (
    "input_values",
    "input_dtypes",
    "input_params",
    "inputs_as_nchw",
    "outputs_as_nchw",
    "input_names",
    "output_names",
    "symbolic_shapes",
    "skip_numeric_validation",
    "nondefault_normalization",
    "post_check_onnx_graph",
    "run_only_f32_variant",
    "run_only_f64_variant",
)


CONDITIONAL_REGISTRATIONS = {
    "plugin::one_minus_square": {
        "requires": "jax.lax.one_minus_square_p",
        "source": "jax2onnx/plugins/jax/lax/one_minus_square.py",
        "reason": "JAX 0.10.2 has no dedicated primitive; keep its baseline cell on newer stacks.",
    }
}


# These are the existing optional registration policies, not general exemptions.
OPTIONAL_FAMILIES = {
    "maxtext": {
        "repository": "https://github.com/AI-Hypercomputer/maxtext",
        "config_directory": "src/maxtext/configs/models",
        "factory": "get_maxtext_model_callable",
        "package_attribute": "MAXTEXT_PKG_PATH",
        "configs_attribute": "MODELS_DIR",
    },
    "maxdiffusion": {
        "repository": "https://github.com/AI-Hypercomputer/maxdiffusion",
        "config_directory": "src/maxdiffusion/configs",
        "factory": "get_maxdiffusion_unet",
        "package_attribute": "MAXDIFFUSION_PKG_PATH",
        "configs_attribute": "CONFIGS_DIR",
    },
}


def _optional_condition(family: str, ordinals: Sequence[int]) -> dict[str, Any]:
    return {
        "requires": f"optional:{family}",
        "source": f"jax2onnx/plugins/examples/{family}/{family}_models.py",
        "ordinals": sorted(ordinals),
        "reason": "Retain reviewed optional obligations when unavailable or omitted by the existing model filter.",
    }


def _contract_hash(contract: Any) -> str:
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _contained_file(checkout: Path, path: Path) -> tuple[str, str]:
    resolved = path.resolve(strict=True)
    relative = resolved.relative_to(checkout.resolve(strict=True))
    if not resolved.is_file():
        raise ValueError(f"Not a config file: {path}")
    return relative.as_posix(), hashlib.sha256(resolved.read_bytes()).hexdigest()


def _optional_baseline(
    family: str,
    observations: Mapping[str, Any],
    registrations: Sequence[Sequence[Any]],
    cases: Sequence[Sequence[Any]],
    contracts: Sequence[Any],
) -> dict[str, Any]:
    """Bind observed recipes to actual pinned source/config bytes, without models."""
    policy = OPTIONAL_FAMILIES[family]
    selected = [
        row for row in cases if registrations[row[0]][2] == f"examples.{family}"
    ]
    result: dict[str, Any] = {
        "repository": policy["repository"],
        "ref": None,
        "registration_source_sha256": observations.get("registration_source_sha256"),
        "members": sorted([[registrations[row[0]][0], row[1]] for row in selected]),
        "configs": {},
        "config_tree": [],
        "config_directory": None,
        "registration_exclusions": [],
        "provenance_gaps": [
            "Imported code/config outside the pinned upstream checkout is not established by this inventory."
        ],
        "errors": [],
        "completion_batch": 2,
        "source_export": "AUDIT_PENDING",
        "target_numerics": "UNVERIFIED",
        "owners": (
            ["transformer", "attention", "state"]
            if family == "maxtext"
            else ["UNet", "convolution", "normalization"]
        ),
    }
    checkout_name = observations.get("resolved_checkout")
    config_name = observations.get("resolved_config_directory")
    if not observations.get("registration_available"):
        return result
    if not checkout_name or not config_name:
        result["errors"].append(
            "Available optional registrations lack a verified checkout/config directory"
        )
        return result
    checkout, config_directory = Path(checkout_name), Path(config_name)
    result["checkout"] = str(checkout)
    try:
        result["config_directory"] = (
            config_directory.resolve(strict=True)
            .relative_to(checkout.resolve(strict=True))
            .as_posix()
        )
        patterns = observations.get("registration_filters", {}).get("skip_patterns", [])
        result["registration_exclusions"] = sorted(
            path.name
            for path in config_directory.glob("*.yml")
            if any(pattern in path.name.lower() for pattern in patterns)
        )
        result["ref"] = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip():
            raise ValueError("Upstream checkout is not clean")
        tree_root = config_directory.parent if family == "maxtext" else config_directory
        result["config_tree"] = [
            list(item)
            for item in sorted(
                _contained_file(checkout, path)
                for path in tree_root.rglob("*")
                if path.suffix in {".yml", ".yaml"}
            )
        ]
        for row in selected:
            origin = registrations[row[0]][0]
            identity = f"{origin}::{row[1]}"
            prefix = f"{family}_"
            if not row[2].startswith(prefix):
                raise ValueError(f"Unexpected optional testcase: {identity}")
            label = row[2][len(prefix) :] + ".yml"
            expected_origin = f"example::examples.{family}::{'MaxText' if family == 'maxtext' else 'MaxDiffusion'}_{Path(label).stem.replace('-', '_')}"
            if origin != expected_origin or row[1] != 0:
                raise ValueError(f"Unrecognized optional registration: {identity}")
            contract = contracts[row[5]]
            recipe = contract.get("callable", {})
            factory = recipe.get("factory", {}).get("callable")
            expected_factory = f"jax2onnx.plugins.examples.{family}.{family}_models.{policy['factory']}"
            if factory != expected_factory:
                raise ValueError(f"Unexpected optional factory: {identity}")
            config = config_directory / label
            relative, digest = _contained_file(checkout, config)
            if family == "maxtext":
                raw_path = recipe.get("kwargs", {}).get("config_path")
                if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
                    raise ValueError(
                        f"Missing absolute registered config_path: {identity}"
                    )
                if _contained_file(checkout, Path(raw_path)) != (relative, digest):
                    raise ValueError(
                        f"Registered config does not match its label: {identity}"
                    )
                observed_path = raw_path
            else:
                # Labels bind provenance, not distinct factory arguments/architectures.
                observed_path = str(config)
            result["configs"][identity] = {
                "relative_path": relative,
                "sha256": digest,
                "observed_absolute_path": observed_path,
                "raw_contract_sha256": _contract_hash(contract),
            }
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        result["errors"].append(str(error))
    return result


def raw_registrations() -> (
    tuple[list[tuple[str, str, Mapping[str, Any]]], list[tuple[str, str]]]
):
    """Return raw plugin/example metadata and registry aliases without metadata."""
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from jax2onnx.plugins.plugin_system import (
        EXAMPLE_REGISTRY,
        PLUGIN_REGISTRY,
        import_all_plugins,
    )

    import_all_plugins()
    registrations = []
    without_metadata = []
    for name, plugin in sorted(PLUGIN_REGISTRY.items()):
        module = type(plugin).__module__
        metadata = getattr(plugin, "metadata", None)
        if isinstance(metadata, Mapping):
            registrations.append((f"plugin::{name}", module, metadata))
        else:
            without_metadata.append((name, module))
    for name, metadata in sorted(EXAMPLE_REGISTRY.items()):
        # The example's registration identity is stable even when its factory
        # is represented by a generic _ConstructAndCall object.
        cases = metadata.get("testcases", [])
        target = cases[0].get("callable") if cases else None
        target = getattr(target, "_ctor", target)
        module = getattr(target, "__module__", type(target).__module__)
        registrations.append((f"example::{name}", module, metadata))
    return registrations, without_metadata


def canonical_contract(
    value: Any, *, unresolved: list[str], location: str = "contract", depth: int = 0
) -> Any:
    """Normalize known metadata recipes without executing or reflecting model objects."""
    import numpy as np
    import jax

    def child(item: Any, suffix: str) -> Any:
        return canonical_contract(
            item,
            unresolved=unresolved,
            location=f"{location}.{suffix}",
            depth=depth + 1,
        )

    typename = f"{type(value).__module__}.{type(value).__qualname__}"
    if depth > 12:
        unresolved.append(f"{location}: recipe nesting limit ({typename})")
        return {"unresolved": typename}
    # NumPy scalar classes may subclass Python float/complex/str. Preserve
    # their typed storage before any Python scalar normalization can erase it.
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            unresolved.append(f"{location}: object-bearing NumPy scalar")
            return {"unresolved": typename, "scalar_dtype": value.dtype.str}
        scalar = {
            "scalar_dtype": value.dtype.name,
            "dtype_str": value.dtype.str,
            "bytes_hex": value.tobytes().hex(),
        }
        if value.dtype.fields is not None:
            scalar["dtype_descr"] = child(value.dtype.descr, "dtype_descr")
        return scalar
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else {"float": value.hex()}
    if isinstance(value, complex):
        return {"complex": [child(value.real, "real"), child(value.imag, "imag")]}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, np.dtype):
        return {"dtype": value.name}
    if isinstance(value, type):
        if value.__module__ in {"numpy", "jax.numpy", "ml_dtypes"} or value in {
            bool,
            int,
            float,
            complex,
            str,
            bytes,
        }:
            try:
                return {"dtype": np.dtype(value).name}
            except (TypeError, ValueError):
                pass
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, (np.ndarray, jax.Array)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            payload = json.dumps(
                child(array.tolist(), "object_values"), sort_keys=True
            ).encode()
        else:
            payload = array.tobytes(order="C")
        result = {
            "array_dtype": array.dtype.name,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if isinstance(value, jax.Array):
            result["weak_type"] = bool(value.weak_type)
        return result
    if isinstance(value, jax.ShapeDtypeStruct):
        return {
            "shape": child(value.shape, "shape"),
            "dtype": np.dtype(value.dtype).name,
            "weak_type": bool(value.weak_type),
        }
    if isinstance(value, Enum):
        return {"enum": typename, "name": value.name}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: child(value[key], key) for key in sorted(value)}
        pairs = [
            (child(key, "key"), child(item, str(key))) for key, item in value.items()
        ]
        return {
            "mapping": sorted(
                pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True)
            )
        }
    if isinstance(value, list):
        return [child(item, str(index)) for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return {"tuple": [child(item, str(index)) for index, item in enumerate(value)]}
    if isinstance(value, functools.partial):
        return {
            "partial": child(value.func, "function"),
            "args": child(value.args, "args"),
            "kwargs": child(value.keywords, "kwargs"),
        }
    # These are repository-owned recipe containers, not arbitrary object graphs.
    if typename == "jax2onnx.plugins.plugin_system._ConstructAndCall":
        return {
            "factory": child(value._ctor, "constructor"),
            "dtype": child(value._dtype, "dtype"),
            "args": child(value._args, "args"),
            "kwargs": child(value._kwargs, "kwargs"),
        }
    if typename == "jax2onnx.plugins.plugin_system._FactoryValue":
        return {
            "placeholder": child(value._fn, "function"),
            "metadata": child(value.metadata, "metadata"),
        }
    if typename == "jax2onnx.converter.typing_support.RngTrace":
        return {"rng_kind": value.kind, "seed": value.seed}
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
        from jax2onnx._compat.jax import native_logistic_metadata_contract

        logistic = native_logistic_metadata_contract(value)
        if logistic is not None:
            return logistic
        result = {"callable": f"{value.__module__}.{value.__qualname__}"}
        if isinstance(value, types.FunctionType):
            if (
                value.__module__ == "jax2onnx.plugins._post_check_onnx_graph"
                and value.__qualname__ == "expect_graph.<locals>._run"
            ):
                captures = {
                    name: cell.cell_contents
                    for name, cell in zip(
                        value.__code__.co_freevars, value.__closure__ or (), strict=True
                    )
                }
                defaults = {
                    "mode": "all",
                    "must_absent": None,
                    "symbols": None,
                    "no_unused_inputs": False,
                    "no_unused_function_inputs": False,
                    "search_functions": False,
                    "explain_on_fail": True,
                }
                expectation = {"expect_graph": child(captures["specs"], "specs")}
                for name, default in defaults.items():
                    if name in captures and captures[name] != default:
                        expectation[name] = child(captures[name], name)
                for name in (
                    captures.keys() - defaults.keys() - {"specs", "_parse_spec"}
                ):
                    expectation[name] = child(captures[name], name)
                return expectation
            if value.__defaults__:
                result["defaults"] = child(value.__defaults__, "defaults")
            if value.__kwdefaults__:
                result["kwdefaults"] = child(value.__kwdefaults__, "kwdefaults")
            if value.__closure__:
                captures = {}
                for name, cell in zip(
                    value.__code__.co_freevars, value.__closure__, strict=True
                ):
                    try:
                        captured = cell.cell_contents
                    except ValueError:
                        captured = "<empty-cell>"
                    captures[name] = child(captured, f"capture.{name}")
                result["captures"] = captures
        return result
    # Symbolic dimensions, model instances, callbacks with opaque state, etc.
    # remain explicit audit work; never use their memory-address-bearing repr.
    unresolved.append(f"{location}: unsupported {typename}")
    return {"unresolved": typename}


def _features(case: Mapping[str, Any]) -> int:
    enabled = {name for name in FEATURES if bool(case.get(name))}
    if any(
        not isinstance(dim, Integral)
        for shape in (case.get("input_shapes") or [])
        for dim in shape
    ):
        enabled.add("symbolic_shapes")
    if case.get("normalization_mode", "auto") != "auto":
        enabled.add("nondefault_normalization")
    return sum(1 << index for index, name in enumerate(FEATURES) if name in enabled)


def _test_functions(tree: ast.AST, prefix: str = "") -> list[str]:
    result = []
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, ast.ClassDef):
            result.extend(_test_functions(child, f"{prefix}{child.name}."))
        elif isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and child.name.startswith("test_"):
            result.append(f"{prefix}{child.name}")
    return result


def handwritten_inventory(repository_root: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Keep handwritten boundary/regression identities separate from registry rows."""
    rows = []
    for path in sorted((repository_root / "tests").rglob("test_*.py")):
        relative = path.relative_to(repository_root)
        if relative.parts[1] in {"primitives", "examples"}:
            continue  # Those modules are generated registry entrypoints.
        names = tuple(sorted(_test_functions(ast.parse(path.read_text()))))
        if names:
            rows.append((str(relative), names))
    return rows


def manual_provenance(
    repository_root: Path, manual: Sequence[tuple[str, Sequence[str]]]
) -> dict[str, Any]:
    """Separate the original committed manual scope from new migration tests."""
    anchor = "ccc861b0"
    archive = subprocess.run(
        ["git", "archive", anchor, "tests"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout
    original: set[tuple[str, str]] = set()
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle:
            path = Path(member.name)
            if (
                not member.isfile()
                or path.suffix != ".py"
                or not path.name.startswith("test_")
                or path.parts[1] in {"primitives", "examples"}
            ):
                continue
            stream = bundle.extractfile(member)
            if stream is not None:
                original.update(
                    (str(path), name)
                    for name in _test_functions(ast.parse(stream.read()))
                )
    current = {(path, name) for path, names in manual for name in names}
    return {
        "revision_anchor": anchor,
        "anchor_manual_test_functions": len(original),
        "existing_manual_test_functions": len(original & current),
        "migration_additions": sorted(current - original),
        "removed_since_anchor": sorted(original - current),
    }


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return str(path.relative_to(repository_root))
    except ValueError:
        return str(path)


def _optional_filters(source: Path) -> dict[str, list[str]]:
    if not source.is_file():
        return {}
    result = {}
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name, value = node.targets[0].id, node.value
        else:
            continue
        if (
            name not in {"DEFAULT_MODELS", "_SKIP_PATTERNS", "skip_patterns"}
            or value is None
        ):
            continue
        try:
            strings = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
        if isinstance(strings, (tuple, list)) and all(
            isinstance(item, str) for item in strings
        ):
            result[
                "default_models" if name == "DEFAULT_MODELS" else "skip_patterns"
            ] = list(strings)
    return result


def optional_inventory(repository_root: Path) -> dict[str, Any]:
    """Discover local optional configs without importing or constructing their models."""
    result = {}
    for family, policy in OPTIONAL_FAMILIES.items():
        config_directory = policy["config_directory"]
        source = (
            repository_root / f"jax2onnx/plugins/examples/{family}/{family}_models.py"
        )
        filters = _optional_filters(source)
        excluded_patterns = filters.get("skip_patterns", [])
        env_key = f"JAX2ONNX_{family.upper()}_SRC"
        configured = os.environ.get(env_key, "").strip()
        candidates = [repository_root / "tmp" / family]
        if configured:
            configured_path = Path(configured).expanduser().absolute()
            if configured_path not in candidates:
                candidates.append(configured_path)
        discoveries = []
        for root in sorted(candidates):
            config_path = root / config_directory
            names = sorted(path.name for path in config_path.glob("*.yml"))
            discoveries.append(
                {
                    "checkout": _display_path(root, repository_root),
                    "present": root.is_dir(),
                    "configs": names,
                    "registration_exclusions": [
                        name
                        for name in names
                        if any(pattern in name.lower() for pattern in excluded_patterns)
                    ],
                }
            )
        # raw_registrations imports these modules first. Reuse the actual resolver
        # and guarded import outcome, including MaxText's installed-package path.
        module = sys.modules.get(f"jax2onnx.plugins.examples.{family}.{family}_models")
        if (
            module is not None
            and Path(getattr(module, "__file__", "")).resolve() != source.resolve()
        ):
            module = None
        package = getattr(module, policy["package_attribute"], None)
        configs = getattr(module, policy["configs_attribute"], None)
        resolved_checkout = None
        resolution_error = None
        if package is not None:
            try:
                resolved_checkout = subprocess.run(
                    ["git", "-C", str(package), "rev-parse", "--show-toplevel"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError) as error:
                resolution_error = str(error)
        import_error = getattr(module, f"_{family.upper()}_IMPORT_ERROR", None)
        result[family] = {
            "source_environment_configured": bool(configured),
            "registration_available": bool(
                getattr(module, f"{family.upper()}_AVAILABLE", False)
            ),
            "import_error": str(import_error) if import_error is not None else None,
            "provenance_error": resolution_error,
            "resolved_checkout": resolved_checkout,
            "resolved_config_directory": str(configs) if configs is not None else None,
            "registration_source_sha256": (
                hashlib.sha256(source.read_bytes()).hexdigest()
                if source.is_file()
                else None
            ),
            "model_filter": os.environ.get(
                f"JAX2ONNX_{family.upper()}_MODELS", "<plugin-default>"
            ),
            "registration_source": f"jax2onnx/plugins/examples/{family}/{family}_models.py",
            "registration_filters": filters,
            "discovery": discoveries,
            "source_audit": "not-run",
            "numerical_acceptance": "unverified",
        }
    return result


def build_inventory(
    registrations: Sequence[tuple[str, str, Mapping[str, Any]]],
    without_metadata: Sequence[tuple[str, str]],
    *,
    manual: Sequence[tuple[str, Sequence[str]]],
    optional: Mapping[str, Any],
    manual_origin: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build compact stable identities; preserve collisions and every raw row."""
    registration_rows: list[tuple[str, str, str, str]] = []
    cases: list[tuple[int, int, str, int, int, int]] = []
    contracts: list[Any] = []
    contract_indices: dict[str, int] = {}
    unresolved_recipes: list[tuple[int, list[str]]] = []
    names: dict[tuple[str, str, str], list[int]] = {}
    for origin, module, metadata in sorted(registrations, key=lambda item: item[0]):
        context = str(metadata.get("context", "default"))
        component = str(metadata.get("component", "default"))
        registration_index = len(registration_rows)
        registration_rows.append((origin, module, context, component))
        for ordinal, case in enumerate(metadata.get("testcases", [])):
            testcase = str(case.get("testcase", "unnamed"))
            case_index = len(cases)
            unresolved: list[str] = []
            contract = canonical_contract(
                {
                    key: value
                    for key, value in case.items()
                    if key not in {"testcase", "description"}
                },
                unresolved=unresolved,
            )
            fingerprint = json.dumps(contract, sort_keys=True, separators=(",", ":"))
            if fingerprint not in contract_indices:
                contract_indices[fingerprint] = len(contracts)
                contracts.append(contract)
            contract_index = contract_indices[fingerprint]
            if unresolved:
                unresolved_recipes.append((case_index, sorted(set(unresolved))))
            cases.append(
                (
                    registration_index,
                    ordinal,
                    testcase,
                    _features(case),
                    int(case.get("opset_version", 23)),
                    contract_index,
                )
            )
            names.setdefault((context, component, testcase), []).append(case_index)
    collisions = [
        (key, indices) for key, indices in sorted(names.items()) if len(indices) > 1
    ]
    optional_rows: dict[str, Any] = {}
    conditions: dict[str, Any] = copy.deepcopy(CONDITIONAL_REGISTRATIONS)
    for family, observations in optional.items():
        optional_rows[family] = {
            **observations,
            "registered_case_rows": sum(
                len(metadata.get("testcases", []))
                for _, _, metadata in registrations
                if metadata.get("context") == f"examples.{family}"
            ),
        }
        if family in OPTIONAL_FAMILIES:
            baseline = _optional_baseline(
                family, observations, registration_rows, cases, contracts
            )
            optional_rows[family]["baseline"] = baseline
            for origin, ordinal in baseline["members"]:
                conditions.setdefault(origin, _optional_condition(family, []))[
                    "ordinals"
                ].append(ordinal)
    return {
        "schema_version": 1,
        "scope": "Raw existing metadata and handwritten identities, not a support or acceptance verdict.",
        "source_audit": "not-run",
        "numerical_acceptance": "unverified",
        "conditional_registrations": conditions,
        "metadata_discovery": {
            "random_inputs": "Module-owned isolated fixture recipes; recipe/seed are preserved in raw case contracts.",
        },
        "outcome_categories": [
            "audit-pending",
            "implementation-gap",
            "representation-limit",
            "runtime-gap",
            "numerical-failure",
        ],
        "registration_columns": ["origin", "module", "context", "component"],
        "registrations": registration_rows,
        "registry_without_metadata": sorted(without_metadata),
        "case_columns": [
            "registration",
            "ordinal",
            "testcase",
            "feature_bits",
            "opset",
            "contract",
        ],
        "feature_bits": {name: index for index, name in enumerate(FEATURES)},
        "cases": cases,
        "contracts": contracts,
        "unresolved_recipes": unresolved_recipes,
        "name_collisions": collisions,
        "manual_columns": ["path", "qualified_test_functions"],
        "manual_provenance": dict(manual_origin or {}),
        "manual": sorted(
            (path, tuple(sorted(functions))) for path, functions in manual
        ),
        "optional": optional_rows,
        "counts": {
            "registry_names": sum(
                origin.startswith("plugin::") for origin, _, _ in registrations
            )
            + len(without_metadata),
            "plugin_metadata": sum(
                origin.startswith("plugin::") for origin, _, _ in registrations
            ),
            "example_metadata": sum(
                origin.startswith("example::") for origin, _, _ in registrations
            ),
            "raw_case_rows": len(cases),
            "distinct_case_names": len(names),
            "colliding_names": len(collisions),
            "manual_files": len(manual),
            "manual_test_functions": sum(len(functions) for _, functions in manual),
            "contexts": dict(
                sorted(Counter(row[2] for row in registration_rows).items())
            ),
        },
    }


def collect_inventory(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    registrations, without_metadata = raw_registrations()
    manual = handwritten_inventory(repository_root)
    return build_inventory(
        registrations,
        without_metadata,
        manual=manual,
        optional=optional_inventory(repository_root),
        manual_origin=manual_provenance(repository_root, manual),
    )


def inventory_json(inventory: Mapping[str, Any]) -> str:
    """Put one compact table row per line for reviewable, bounded-size diffs."""
    lines = ["{"]
    for index, (key, value) in enumerate(inventory.items()):
        comma = "," if index < len(inventory) - 1 else ""
        if key in {
            "registrations",
            "registry_without_metadata",
            "cases",
            "contracts",
            "unresolved_recipes",
            "name_collisions",
            "manual",
        }:
            lines.append(f"  {json.dumps(key)}: [")
            for row_index, row in enumerate(value):
                lines.append(
                    "    "
                    + json.dumps(row, separators=(",", ":"))
                    + ("," if row_index < len(value) - 1 else "")
                )
            lines.append(f"  ]{comma}")
        else:
            lines.append(
                f"  {json.dumps(key)}: {json.dumps(value, sort_keys=True, separators=(',', ':'))}{comma}"
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


_OPTIONAL_OBSERVATIONS = {
    "source_environment_configured",
    "model_filter",
    "discovery",
    "registered_case_rows",
    "registration_available",
    "import_error",
    "resolved_checkout",
    "resolved_config_directory",
    "registration_source_sha256",
    "provenance_error",
    "baseline",
}


def _config_descriptor(
    family: str, baseline: Mapping[str, Any], identity: str, case: Sequence[Any]
) -> dict[str, Any]:
    """Validate stored provenance before giving exactly one path portable semantics."""
    policy = OPTIONAL_FAMILIES[family]
    record = baseline["configs"][identity]
    relative = Path(record["relative_path"])
    ref = baseline["ref"]
    if (
        baseline["repository"] != policy["repository"]
        or not isinstance(ref, str)
        or len(ref) != 40
        or any(character not in "0123456789abcdef" for character in ref)
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.parent.as_posix() != baseline.get("config_directory")
        or dict(baseline["config_tree"]).get(relative.as_posix()) != record["sha256"]
    ):
        raise ValueError(f"Invalid pinned config identity: {identity}")
    contract = case[-1]
    recipe = contract.get("callable", {})
    if (
        recipe.get("factory", {}).get("callable")
        != f"jax2onnx.plugins.examples.{family}.{family}_models.{policy['factory']}"
        or _contract_hash(contract) != record["raw_contract_sha256"]
        or not case[0].startswith(family + "_")
        or relative.name != case[0][len(family) + 1 :] + ".yml"
    ):
        raise ValueError(
            f"Config provenance does not bind the original recipe: {identity}"
        )
    observed = record["observed_absolute_path"]
    if (
        not Path(observed).is_absolute()
        or Path(observed).resolve() != (Path(baseline["checkout"]) / relative).resolve()
    ):
        raise ValueError(f"Config outside its recorded checkout: {identity}")
    if family == "maxtext" and recipe.get("kwargs", {}).get("config_path") != observed:
        raise ValueError(f"Config provenance does not bind config_path: {identity}")
    return {
        "family": family,
        "repository": policy["repository"],
        "ref": ref,
        "relative_path": relative.as_posix(),
        "sha256": record["sha256"],
    }


def retained_missing_cells(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> set[str]:
    """Refresh must not erase known cells, including unavailable/filtered rows."""
    conditional = set(CONDITIONAL_REGISTRATIONS)
    for family in OPTIONAL_FAMILIES:
        conditional.update(
            origin
            for origin, _ in expected.get("optional", {})
            .get(family, {})
            .get("baseline", {})
            .get("members", [])
        )
    missing = set(_identity_view(expected)["cases"]) - set(
        _identity_view(actual)["cases"]
    )
    return {
        identity for identity in missing if identity.rsplit("::", 1)[0] in conditional
    }


def _compare_optional(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[list[str], list[str]]:
    differences: list[str] = []
    observations: list[str] = []
    for family in OPTIONAL_FAMILIES:
        before = expected.get("optional", {}).get(family, {})
        after = actual.get("optional", {}).get(family, {})
        baseline = before.get("baseline")
        if not baseline or not baseline.get("members"):
            continue
        observed = after.get("baseline", {})
        members = {f"{origin}::{ordinal}" for origin, ordinal in baseline["members"]}
        expected_family = {
            f"{expected['registrations'][row[0]][0]}::{row[1]}"
            for row in expected["cases"]
            if expected["registrations"][row[0]][2] == f"examples.{family}"
        }
        if (
            members != expected_family
            or set(baseline.get("configs", {})) != members
            or baseline.get("errors")
        ):
            differences.append(
                f"Invalid optional baseline membership/provenance: {family}"
            )
            continue
        valid_origins: dict[str, list[int]] = {}
        for origin, ordinal in baseline["members"]:
            valid_origins.setdefault(origin, []).append(ordinal)
        for origin, ordinals in valid_origins.items():
            condition = _optional_condition(family, ordinals)
            if left["conditional_registrations"].get(origin) != condition:
                differences.append(f"Invalid optional condition: {origin}")
                continue
            # Only frozen origin/ordinal membership may be conditionally absent.
            del left["conditional_registrations"][origin]
            if right["conditional_registrations"].get(origin) == condition:
                del right["conditional_registrations"][origin]
        for view, record in ((left, before), (right, after)):
            view["optional"][family] = {
                key: value
                for key, value in record.items()
                if key not in _OPTIONAL_OBSERVATIONS
            }
        if baseline.get("registration_source_sha256") != after.get(
            "registration_source_sha256"
        ):
            differences.append(f"Changed optional registration source: {family}")
        actual_members = {
            f"{origin}::{ordinal}" for origin, ordinal in observed.get("members", [])
        }
        actual_family = {
            f"{actual['registrations'][row[0]][0]}::{row[1]}"
            for row in actual["cases"]
            if actual["registrations"][row[0]][2] == f"examples.{family}"
        }
        if actual_members != actual_family:
            differences.append(f"Invalid observed optional membership: {family}")
        configured = after.get("source_environment_configured", False)
        available = after.get("registration_available", False)
        failed = (
            after.get("import_error")
            or after.get("provenance_error")
            or observed.get("errors")
        )
        absent = not configured and not available and not actual_family
        if absent and failed:
            observations.append(
                f"Optional {family} unavailable during unconfigured discovery: {failed}"
            )
        if not absent and (failed or not available or not actual_family):
            differences.append(
                f"Optional environment error: {family}: {failed or 'configured/available discovery has no registrations'}"
            )
        if not absent:
            for field in (
                "repository",
                "ref",
                "config_tree",
                "config_directory",
                "registration_exclusions",
                "provenance_gaps",
                "completion_batch",
                "source_export",
                "target_numerics",
                "owners",
            ):
                if baseline.get(field) != observed.get(field):
                    differences.append(f"Changed optional {field}: {family}")
        model_filter = after.get("model_filter", "<plugin-default>").strip()
        defaults = before.get("registration_filters", {}).get("default_models", [])
        allow = (
            None
            if model_filter.lower() == "all"
            else (
                set(defaults)
                if model_filter in {"", "<plugin-default>"}
                else {
                    name if name.endswith(".yml") else name + ".yml"
                    for token in model_filter.split(",")
                    if (name := token.strip())
                }
            )
        )
        known_labels = {
            Path(record["relative_path"]).name
            for record in baseline["configs"].values()
        }
        # Only exact frozen YAML labels excluded by the unchanged policy may
        # occur in an allow-list without a retained registration.
        patterns = before.get("registration_filters", {}).get("skip_patterns", [])
        tree_labels = {
            Path(name).name
            for name, _ in baseline["config_tree"]
            if Path(name).parent.as_posix() == baseline.get("config_directory")
            and Path(name).suffix == ".yml"
        }
        excluded_labels = {
            name
            for name in tree_labels
            if any(pattern in name.lower() for pattern in patterns)
        }
        if excluded_labels != set(baseline.get("registration_exclusions", [])):
            differences.append(f"Invalid optional exclusion provenance: {family}")
        if allow is not None:
            unknown = allow - known_labels - excluded_labels
            if unknown:
                differences.append(
                    f"Unknown optional model filter: {family}: {', '.join(sorted(unknown))}"
                )
        selected = {
            identity
            for identity in members
            if allow is None
            or Path(baseline["configs"][identity]["relative_path"]).name in allow
        }
        if not absent and (actual_family & members) - selected:
            differences.append(
                f"Unexpected filtered optional cases: {family}: {', '.join(sorted((actual_family & members) - selected))}"
            )
        retained = []
        for identity in sorted(members):
            origin = identity.rsplit("::", 1)[0]
            if (
                origin not in valid_origins
                or origin in left["conditional_registrations"]
            ):
                continue
            try:
                descriptor = _config_descriptor(
                    family, baseline, identity, left["cases"][identity]
                )
            except (KeyError, TypeError, ValueError) as error:
                differences.append(
                    f"Invalid optional config provenance: {identity}: {error}"
                )
                continue
            if identity not in right["cases"]:
                label = Path(descriptor["relative_path"]).name
                if absent or (allow is not None and label not in allow):
                    del left["cases"][identity]
                    retained.append(label)
                else:
                    differences.append(f"Missing selected optional case: {identity}")
                continue
            try:
                actual_descriptor = _config_descriptor(
                    family, observed, identity, right["cases"][identity]
                )
            except (KeyError, TypeError, ValueError) as error:
                differences.append(
                    f"Invalid observed config provenance: {identity}: {error}"
                )
                continue
            if descriptor != actual_descriptor:
                differences.append(f"Changed optional config identity: {identity}")
            if family == "maxtext":
                old_record, new_record = (
                    baseline["configs"][identity],
                    observed["configs"][identity],
                )
                for view, replacement in (
                    (left, descriptor),
                    (right, actual_descriptor),
                ):
                    view["cases"][identity][-1]["callable"]["kwargs"][
                        "config_path"
                    ] = replacement
                if (
                    old_record["observed_absolute_path"]
                    != new_record["observed_absolute_path"]
                    and left["cases"][identity] == right["cases"][identity]
                    and baseline.get("config_tree") == observed.get("config_tree")
                    and baseline.get("registration_source_sha256")
                    == observed.get("registration_source_sha256")
                    and not failed
                ):
                    observations.append(
                        f"Verified config relocation: {identity}: {old_record['observed_absolute_path']} -> {new_record['observed_absolute_path']}; raw SHA256 {old_record['raw_contract_sha256']} -> {new_record['raw_contract_sha256']}"
                    )
        for origin in valid_origins:
            if not any(key.startswith(origin + "::") for key in left["cases"]):
                left["registrations"].pop(origin, None)
        if retained:
            reason = (
                "unconfigured optional environment unavailable"
                if absent
                else f"model filter {model_filter}"
            )
            observations.append(
                f"Retained {len(retained)} baseline {family} cases not observed ({reason}): {', '.join(retained)}"
            )
    return differences, observations


def _identity_view(inventory: Mapping[str, Any]) -> dict[str, Any]:
    registrations = inventory["registrations"]
    return {
        "registrations": {row[0]: row[1:] for row in registrations},
        "cases": {
            f"{registrations[row[0]][0]}::{row[1]}": (
                [*row[2:5], inventory["contracts"][row[5]]] if len(row) > 5 else row[2:]
            )
            for row in inventory["cases"]
        },
        "registry_without_metadata": dict(inventory["registry_without_metadata"]),
        "manual": dict(inventory["manual"]),
        "manual_provenance": inventory.get("manual_provenance", {}),
        "metadata_discovery": inventory.get("metadata_discovery", {}),
        "optional": inventory["optional"],
        "schema_version": inventory["schema_version"],
        "feature_bits": inventory["feature_bits"],
        "conditional_registrations": inventory.get("conditional_registrations", {}),
    }


def compare_inventories(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    unavailable: frozenset[str] = frozenset(),
) -> tuple[list[str], list[str]]:
    """Compare stable provenance, explicitly retaining known feature-conditional cells."""
    left = json.loads(json.dumps(_identity_view(expected)))
    right = json.loads(json.dumps(_identity_view(actual)))
    differences, observations = _compare_optional(expected, actual, left, right)
    for origin in sorted(unavailable & CONDITIONAL_REGISTRATIONS.keys()):
        if origin in left["registrations"] and origin not in right["registrations"]:
            observations.append(
                f"Retained baseline entry absent on this JAX stack: {origin} ({CONDITIONAL_REGISTRATIONS[origin]['requires']})"
            )
            del left["registrations"][origin]
            left["cases"] = {
                key: value
                for key, value in left["cases"].items()
                if not key.startswith(origin + "::")
            }
    for section in left:
        if isinstance(left[section], dict) and isinstance(right[section], dict):
            for key in sorted(set(left[section]) | set(right[section])):
                if key not in left[section]:
                    differences.append(f"Added {section}: {key}")
                elif key not in right[section]:
                    differences.append(f"Missing {section}: {key}")
                elif left[section][key] != right[section][key]:
                    differences.append(f"Changed {section}: {key}")
        elif left[section] != right[section]:
            differences.append(f"Changed {section}")
    return differences, observations


def unavailable_registrations() -> frozenset[str]:
    import jax

    return frozenset(
        {"plugin::one_minus_square"}
        if not hasattr(jax.lax, "one_minus_square_p")
        else ()
    )


def source_metadata_assertions(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Original requests plus required source stages, independent of outcomes."""
    requested = {
        name
        for name, value in contract.items()
        if (
            name.startswith(("expected_", "check_", "post_check_"))
            or name
            in {"runtime_input_values", "record_primitive_calls_file", "output_path"}
        )
        and value is not None
        and value is not False
    }
    if contract.get("export_mode", "standard") == "web":
        requested.add("onnxruntime_web")
    requested.update(
        (
            "source_reference",
            "schema",
            "artifact",
            "lowering_audit",
            "structure",
            "runtime",
            "numerics",
            "reviewed_criteria",
            "runtime_binding",
            "runtime_reference",
        )
    )
    if "post_check_onnx_graph" in requested:
        requested.remove("post_check_onnx_graph")
        requested.add("raw_callback")
    return tuple(sorted(requested))


def source_metadata_evidence_statuses(
    evidence: Mapping[str, Any],
    requested: Sequence[str],
) -> dict[str, Any]:
    """Derive summaries only from the existing immutable stage evidence content."""
    statuses = {
        name: {"status": evidence[name], "detail": evidence["detail"]}
        for name in (
            "source_reference",
            "schema",
            "artifact",
            "lowering_audit",
            "structure",
            "runtime",
            "numerics",
        )
    }
    callback = evidence["raw_callback_status"]
    statuses["raw_callback"] = {
        "status": (
            "validated"
            if callback in {"passed", "absent"}
            else "failed" if callback in {"failed", "error"} else "unverified"
        ),
        "detail": callback,
    }
    statuses.update(
        (evidence.get("replay_capture") or {}).get("metadata_assertions", {})
    )
    for name in requested:
        statuses.setdefault(
            name,
            {
                "status": "unverified",
                "detail": "Original requested assertion has no terminal SOURCE evidence",
            },
        )
    return statuses


def source_metadata_pytest_statuses(
    collected: Sequence[Mapping[str, Any]],
    reports: Sequence[Any],
) -> dict[str, dict[str, Any]]:
    """Read actual pytest setup/call/teardown reports without running pytest.

    Body evidence and pytest terminal success are independent observations. A
    setup/teardown failure remains failed even without a completed body; absent
    reports remain incomplete and cannot establish a successful capability.
    """
    from pytest import TestReport

    nodeids = {record["nodeid"] for record in collected}
    by_node: dict[str, dict[str, str]] = {}
    for report in reports:
        if not isinstance(report, TestReport):
            raise ValueError(
                "SOURCE terminal accounting requires actual pytest TestReport objects"
            )
        nodeid, when, outcome = report.nodeid, report.when, report.outcome
        if nodeid not in nodeids:
            raise ValueError("Orphan SOURCE pytest report")
        if when not in {"setup", "call", "teardown"} or outcome not in {
            "passed",
            "failed",
            "skipped",
        }:
            raise ValueError("Unknown SOURCE pytest stage/outcome")
        stages = by_node.setdefault(nodeid, {})
        if when in stages:
            raise ValueError("Duplicate SOURCE pytest stage report")
        stages[when] = outcome
    result = {}
    for nodeid, stages in by_node.items():
        terminal = (
            "setup" in stages
            and "teardown" in stages
            and (stages["setup"] != "passed" or "call" in stages)
        )
        if stages.get("setup") != "passed" and "call" in stages:
            raise ValueError("SOURCE call report contradicts its setup result")
        status = (
            "failed"
            if terminal and "failed" in stages.values()
            else (
                "skipped"
                if terminal and "skipped" in stages.values()
                else (
                    "passed"
                    if terminal
                    and set(stages) == {"setup", "call", "teardown"}
                    and all(value == "passed" for value in stages.values())
                    else "incomplete"
                )
            )
        )
        result[nodeid] = {
            "status": status,
            "terminal": terminal,
            "stages": dict(stages),
        }
    return result


def _canonical_container(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if (
        isinstance(value, Mapping)
        and set(value) == {"tuple"}
        and isinstance(value["tuple"], list)
    ):
        return value["tuple"]
    return None


def source_metadata_selector(identity: Sequence[Any], display: Sequence[str]) -> str:
    """A stable method name preserves aliases and sanitization collisions."""
    label = re.sub(r"[^0-9a-zA-Z_]", "_", "_".join((*display, identity[2])))
    return f"test_source_{label}_{_contract_hash(identity)[:20]}"


def legacy_preparation_obligations(
    inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Describe historical variant recipes without invoking legacy generation.

    This is an out-of-band declarative census of the existing generator's
    preparation rules. It is neither actual pytest collection nor an original
    positive/negative verdict. In particular, raw-x64-on does not discharge an
    explicitly widened factory/value variant. Aliases remain distinct.
    """
    obligations = []
    for entry in inventory["cases"]:
        registration, ordinal, testcase, _, _, contract_index = entry
        origin, module, _, _ = inventory["registrations"][registration]
        contract = inventory["contracts"][contract_index]
        if (
            contract.get("callable") is None
            and contract.get("callable_factory") is None
        ):
            continue
        shapes = _canonical_container(contract.get("input_shapes") or [])
        decoded_shapes = (
            [_canonical_container(shape) for shape in shapes]
            if shapes is not None
            else []
        )
        unresolved_shapes = shapes is None or any(
            shape is None for shape in decoded_shapes
        )
        dynamic = any("B" in shape for shape in decoded_shapes if shape is not None)
        shape_recipes = (
            ["shape-preparation-unresolved"]
            if unresolved_shapes
            else ["dynamic-B"] if dynamic else ["original-shapes"]
        )
        if (
            dynamic
            and not unresolved_shapes
            and not contract.get("run_only_dynamic", False)
        ):
            shape_recipes.append("concrete-B=3")
        for shape_recipe in shape_recipes:
            original_variant_name = str(testcase) + (
                "_dynamic" if shape_recipe == "dynamic-B" else ""
            )
            # This spells out the old generator's branch. Counterpart relations
            # are never inferred from this name or from a raw caller scenario.
            only_f64 = bool(
                contract.get("run_only_f64_variant")
                or contract.get("enable_double_precision")
                or original_variant_name.endswith("_f64")
            )
            dtypes = ["legacy-f64-prepared"] if only_f64 else ["legacy-f32-prepared"]
            if (
                not only_f64
                and not contract.get("disable_float64_test")
                and not contract.get("run_only_f32_variant")
            ):
                dtypes.append("legacy-f64-prepared")
            for dtype_recipe in dtypes:
                recipe = {
                    "raw_identity": [origin, ordinal],
                    "raw_contract_sha256": _contract_hash(contract),
                    "shape_preparation": shape_recipe,
                    "dtype_preparation": dtype_recipe,
                    "factory_preparation": (
                        (
                            "original-factory.with_dtype(float64)"
                            if dtype_recipe == "legacy-f64-prepared"
                            else "original-factory.with_dtype(float32)"
                        )
                        if isinstance(contract.get("callable"), Mapping)
                        and "factory" in contract["callable"]
                        else "original-callable"
                    ),
                    "input_preparation": (
                        "original floating supplied values/input dtypes are explicitly widened; integer/Boolean retain kind"
                        if dtype_recipe == "legacy-f64-prepared"
                        else "original supplied values/input declarations; factory requests float32"
                    ),
                    "caller_x64_enabled": dtype_recipe == "legacy-f64-prepared",
                    "original_variant_name": original_variant_name
                    + (
                        "_f64"
                        if dtype_recipe == "legacy-f64-prepared" and not only_f64
                        else ""
                    ),
                    "legacy_harness_context": "float32 matmul precision plus original named/native settings",
                    "original_contract": contract,
                }
                recipe_sha = _contract_hash(recipe)
                obligations.append(
                    {
                        "obligation_id": f"legacy-preparation:{origin}:{ordinal}:{shape_recipe}:{dtype_recipe}",
                        "kind": "legacy-preparation",
                        "owner": module,
                        "raw_identity": [origin, ordinal],
                        "recipe_sha256": recipe_sha,
                        "recipe": recipe,
                        "baseline_classification": "unclassified",
                        "original_evidence": None,
                        "source_counterpart": None,
                        "status": "unverified",
                        "open_reason": "Original variant outcome and explicitly prepared SOURCE counterpart require review; raw caller coverage does not replace this recipe",
                    }
                )
    return obligations


def reconcile_source_metadata(
    inventory: Mapping[str, Any],
    observed_inventory: Mapping[str, Any],
    plan: Any,
    *,
    baseline_sha256: str,
    epoch: Mapping[str, str],
    planned: Sequence[Mapping[str, Any]],
    collected: Sequence[Mapping[str, Any]] = (),
    outcomes: Sequence[Mapping[str, Any]] = (),
    collection_complete: bool = False,
    outcomes_complete: bool = False,
    unavailable: frozenset[str] = frozenset(),
    original_capabilities: Sequence[Mapping[str, Any]] = (),
    stage_evidence: Sequence[Any] = (),
    pytest_reports: Sequence[Any] = (),
) -> dict[str, Any]:
    """Reconcile the complete frozen denominator with actual bounded receipts.

    Planned selectors are not collection receipts, and passing stage contracts
    are not necessarily accepted capabilities. Inputs contain records returned by
    SourceMetadataCase.record plus actual pytest node IDs/terminal outcomes. A
    completed collection/run fails closed on missing, duplicate or orphan IDs.
    Baseline classifications can only come from explicit original evidence.
    """
    if baseline_sha256 != plan.baseline_sha256:
        raise ValueError("SOURCE accounting baseline epoch drift")
    if set(epoch) != {"source_sha256", "environment_sha256"} or any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in epoch.values()
    ):
        raise ValueError(
            "SOURCE accounting needs exact source/environment epoch hashes"
        )
    differences, observations = compare_inventories(
        inventory,
        observed_inventory,
        unavailable=unavailable,
    )
    retained = retained_missing_cells(inventory, observed_inventory)
    # Added migration tests stay separate. No addition or reclassification can
    # hide an absent original manual anchor from this frozen baseline.
    additions = {
        tuple(item)
        for item in inventory.get("manual_provenance", {}).get(
            "migration_additions", ()
        )
    }
    anchors = {
        (path, function)
        for path, functions in inventory["manual"]
        for function in functions
    } - additions
    actual_manual = {
        (path, function)
        for path, functions in observed_inventory["manual"]
        for function in functions
    }
    if anchors - actual_manual:
        raise ValueError("Original manual anchors disappeared from SOURCE accounting")
    manual_differences = [
        item
        for item in differences
        if item.startswith(
            (
                "Added manual:",
                "Changed manual:",
                "Changed manual_provenance: migration_additions",
            )
        )
    ]
    differences = [item for item in differences if item not in manual_differences]
    if differences:
        raise ValueError("SOURCE inventory drift: " + "; ".join(differences))
    observations.extend(manual_differences)

    registrations = {item[0]: item for item in inventory["registrations"]}
    frozen_rows = {
        (inventory["registrations"][item[0]][0], item[1]): item
        for item in inventory["cases"]
    }
    if len(frozen_rows) != len(inventory["cases"]):
        raise ValueError("Duplicate raw inventory identity")
    rows = {(row.origin, row.ordinal): row for row in plan.rows}
    cells = {cell.identity: cell for cell in plan.cells}
    expected_cells = {
        (origin, ordinal, scenario)
        for origin, ordinal in frozen_rows
        for scenario in ("raw-x64-off", "raw-x64-on")
    }
    if (
        len(rows) != len(plan.rows)
        or set(rows) != set(frozen_rows)
        or len(cells) != len(plan.cells)
        or set(cells) != expected_cells
    ):
        raise ValueError("SOURCE plan lost or duplicated frozen raw cells")
    for identity, row in rows.items():
        contract = inventory["contracts"][frozen_rows[identity][5]]
        if row.contract_sha256 != _contract_hash(contract):
            raise ValueError("SOURCE plan row contract drift")

    def index_records(
        records: Sequence[Mapping[str, Any]], role: str
    ) -> dict[tuple[str, int, str], Mapping[str, Any]]:
        result = {}
        selectors = set()
        nodeids = set()
        for record in records:
            identity = tuple(record["identity"])
            if identity in result or record["selector"] in selectors:
                raise ValueError(f"Duplicate {role} identity/selector")
            if identity not in cells:
                raise ValueError(f"Orphan {role} SOURCE identity")
            if record["baseline_sha256"] != baseline_sha256 or dict(
                record["epoch"]
            ) != dict(epoch):
                raise ValueError(f"Stale {role} source/environment/baseline epoch")
            if record["row_contract_sha256"] != rows[identity[:2]].contract_sha256:
                raise ValueError(f"Stale {role} ordinal/row contract")
            contract = inventory["contracts"][frozen_rows[identity[:2]][5]]
            if tuple(record["requested_assertions"]) != source_metadata_assertions(
                contract
            ):
                raise ValueError(f"Changed {role} original assertion obligations")
            row = rows[identity[:2]]
            if (
                tuple(record["display"]) != row.display
                or record["owner"]
                != inventory["registrations"][frozen_rows[identity[:2]][0]][1]
            ):
                raise ValueError(f"Changed {role} original owner/display provenance")
            if record["selector"] != source_metadata_selector(identity, row.display):
                raise ValueError(f"Changed {role} exact selector")
            frozen_adapter = json.loads(record["adapter_json"])
            expected_scenario = _contract_hash(
                {
                    "identity": identity,
                    "contract": row.contract_sha256,
                    "adapter": frozen_adapter,
                    "criteria": asdict(row.criteria) if row.criteria else None,
                }
            )
            if (
                record["adapter_sha256"] != _contract_hash(frozen_adapter)
                or record["scenario_sha256"] != expected_scenario
            ):
                raise ValueError(f"Changed {role} frozen adapter/sample scenario")
            descriptor = {
                name: record[name]
                for name in (
                    "identity",
                    "display",
                    "owner",
                    "selector",
                    "row_contract_sha256",
                    "baseline_sha256",
                    "epoch",
                    "requested_assertions",
                    "available",
                    "adapter_json",
                    "adapter_sha256",
                    "scenario_sha256",
                )
            }
            if record["descriptor_sha256"] != _contract_hash(descriptor):
                raise ValueError(f"Changed {role} frozen descriptor")
            if role != "planned":
                nodeid = record.get("nodeid")
                if not isinstance(nodeid, str) or not nodeid or nodeid in nodeids:
                    raise ValueError(f"Missing/duplicate actual {role} pytest node ID")
                if not nodeid.endswith("::" + record["selector"]):
                    raise ValueError(
                        f"Actual {role} pytest node ID does not select the declared method"
                    )
                nodeids.add(nodeid)
            result[identity] = record
            selectors.add(record["selector"])
        return result

    selected = index_records(planned, "planned")
    actual = index_records(collected, "collected")
    results = index_records(outcomes, "outcome")
    if set(actual) - set(selected) or set(results) - set(actual):
        raise ValueError("Orphan actual collection or terminal result")
    if collection_complete and set(actual) != set(selected):
        raise ValueError("Completed collection is missing selected SOURCE IDs")
    pytest_statuses = source_metadata_pytest_statuses(collected, pytest_reports)
    terminal_nodeids = {
        nodeid for nodeid, item in pytest_statuses.items() if item["terminal"]
    }
    if outcomes_complete and (
        not collection_complete
        or terminal_nodeids != {record["nodeid"] for record in actual.values()}
    ):
        raise ValueError(
            "Completed run is missing actual terminal SOURCE pytest outcomes"
        )
    for identity, item in actual.items():
        status = pytest_statuses.get(item["nodeid"], {})
        if status.get("status") == "passed" and identity not in results:
            raise ValueError(
                "Passed SOURCE pytest item lacks retained body/stage evidence"
            )
        if identity in results and status.get("stages", {}).get("setup") in {
            "failed",
            "skipped",
        }:
            raise ValueError("SOURCE body evidence contradicts actual setup failure")
    for identity, record in actual.items():
        if record["descriptor_sha256"] != selected[identity]["descriptor_sha256"]:
            raise ValueError("Actual collection differs from planned SOURCE descriptor")
    from scripts.source_precision_probe import ProbeStageEvidence

    retained_evidence = {}
    for evidence in stage_evidence:
        if not isinstance(evidence, ProbeStageEvidence):
            raise ValueError(
                "Terminal metadata receipts require retained ProbeStageEvidence objects"
            )
        digest = evidence.sha256
        if digest in retained_evidence:
            raise ValueError("Duplicate retained SOURCE stage evidence")
        retained_evidence[digest] = json.loads(json.dumps(asdict(evidence)))
    for identity, record in results.items():
        collected_record = actual[identity]
        if (
            record["descriptor_sha256"] != collected_record["descriptor_sha256"]
            or record["nodeid"] != collected_record["nodeid"]
        ):
            raise ValueError("SOURCE terminal result differs from actual collection")
        if (
            record.get("stage_body_complete") is not True
            or "stage_contract_passed" not in record
        ):
            raise ValueError("Incomplete SOURCE body cannot supply stage evidence")
        if record.get("artifact_sha256") != collected_record.get("artifact_sha256"):
            # Collection can bind an expected artifact for a retained epoch. Fresh
            # product collection declares None and the first terminal artifact
            # is then recorded explicitly; it cannot replace an existing hash.
            if collected_record.get("artifact_sha256") is not None:
                raise ValueError("SOURCE result belongs to another artifact epoch")
        if (
            record.get("evidence_sha256") is None
            or len(record["evidence_sha256"]) != 64
        ):
            raise ValueError("SOURCE terminal result lacks exact immutable evidence")
        evidence_epoch = {
            name: record[name]
            for name in (
                "identity",
                "baseline_sha256",
                "epoch",
                "descriptor_sha256",
                "artifact_sha256",
                "scenario_sha256",
                "evidence_sha256",
            )
        }
        if record.get("evidence_epoch_sha256") != _contract_hash(evidence_epoch):
            raise ValueError("SOURCE result artifact/evidence epoch drift")
        actual_evidence = retained_evidence.get(record["evidence_sha256"])
        if actual_evidence is None:
            raise ValueError(
                "SOURCE terminal result lacks retained actual stage evidence"
            )
        if json.loads(json.dumps(record.get("evidence"))) != actual_evidence:
            raise ValueError(
                "SOURCE terminal evidence content differs from retained stage evidence"
            )
        if (
            tuple(actual_evidence["cell_identity"]) != identity
            or actual_evidence["row_contract_sha256"] != record["row_contract_sha256"]
            or actual_evidence["scenario_sha256"] != record["scenario_sha256"]
            or record["scenario_sha256"] != collected_record["scenario_sha256"]
            or actual_evidence["artifact_sha256"] != record["artifact_sha256"]
            or actual_evidence["requested_stage"] != record["requested_stage"]
            or _contract_hash(actual_evidence["environment"])
            != epoch["environment_sha256"]
        ):
            raise ValueError(
                "SOURCE stage evidence provenance/artifact/scenario epoch differs"
            )
        derived = source_metadata_evidence_statuses(
            actual_evidence, record["requested_assertions"]
        )
        if json.loads(json.dumps(record.get("assertions"))) != json.loads(
            json.dumps(derived)
        ):
            raise ValueError(
                "SOURCE assertion summary differs from retained stage evidence"
            )
        assertions = record.get("assertions", {})
        if set(record["requested_assertions"]) - set(assertions):
            raise ValueError("SOURCE outcome omitted original requested assertions")
        if any(
            item.get("status") not in {"validated", "failed", "unverified"}
            for item in assertions.values()
        ):
            raise ValueError("Unknown SOURCE evidence status")
        stage_contract = record.get("stage_contract", {})
        if not stage_contract or set(stage_contract) - set(assertions):
            raise ValueError(
                "SOURCE terminal outcome omitted its explicit stage contract"
            )
        passed = all(
            assertions[name]["status"] == expected
            for name, expected in stage_contract.items()
        )
        if record["stage_contract_passed"] is not passed:
            raise ValueError(
                "SOURCE stage-contract outcome differs from actual assertions"
            )
        requested_statuses = [
            assertions[name]["status"] for name in record["requested_assertions"]
        ]
        expected_capability = (
            "failed"
            if "failed" in requested_statuses
            else (
                "validated"
                if requested_statuses
                and all(value == "validated" for value in requested_statuses)
                else "unverified"
            )
        )
        if record.get("capability_status") != expected_capability:
            raise ValueError("SOURCE capability claim differs from requested evidence")
        if any(value == "validated" for value in requested_statuses) and not record.get(
            "scenario_sha256"
        ):
            raise ValueError("SOURCE outcome lacks its exact source scenario identity")

    obligations = []
    observed_origins = {item[0] for item in observed_inventory["registrations"]}
    opaque = {index for index, _ in inventory.get("unresolved_recipes", ())}
    optional_members = {
        tuple(item)
        for family in inventory.get("optional", {}).values()
        for item in family.get("baseline", {}).get("members", ())
    }
    for identity in sorted(cells):
        row = rows[identity[:2]]
        result = results.get(identity)
        planned_record = selected.get(identity)
        record = {
            "obligation_id": "raw-cell:" + json.dumps(identity, separators=(",", ":")),
            "kind": "raw-cell",
            "identity": identity,
            "owner": registrations[identity[0]][1],
            "row_contract_sha256": row.contract_sha256,
            "selector": planned_record["selector"] if planned_record else None,
            "selector_state": (
                "actual-collected"
                if identity in actual
                else "planned" if planned_record else "not-selected"
            ),
            "nodeid": actual[identity]["nodeid"] if identity in actual else None,
            "availability": (
                "available"
                if identity[0] in observed_origins
                else "unavailable-retained"
            ),
            "optional": identity[:2] in optional_members,
            "unresolved_recipes": row.unresolved_recipes,
            "baseline_classification": "unclassified",
            "original_evidence": None,
            "source_statuses": (
                result.get("assertions", {})
                if result
                else {
                    name: {
                        "status": "unverified",
                        "detail": "No actual terminal metadata-path evidence",
                    }
                    for name in {
                        "source_reference",
                        "schema",
                        "artifact",
                        "lowering_audit",
                        "structure",
                        "runtime",
                        "numerics",
                        "raw_callback",
                        "reviewed_criteria",
                        *(
                            name
                            for name, value in inventory["contracts"][
                                frozen_rows[identity[:2]][5]
                            ].items()
                            if name.startswith(("expected_", "check_", "post_check_"))
                            and value is not None
                            and value is not False
                        ),
                    }
                }
            ),
            "prior_probe_statuses": {
                name: getattr(cells[identity], name).value
                for name in (
                    "source_reference",
                    "schema",
                    "artifact",
                    "lowering_audit",
                    "runtime",
                    "numerics",
                )
            },
            "prior_probe_history_sha256": [
                item.sha256 for item in cells[identity].history
            ],
            "stage_contract_passed": (
                result.get("stage_contract_passed") if result else None
            ),
            "observed_body_capability_status": (
                result.get("capability_status", "unverified")
                if result
                else "unverified"
            ),
            "pytest_outcome": (
                pytest_statuses.get(actual[identity]["nodeid"], {}).get(
                    "status", "incomplete"
                )
                if identity in actual
                else "not-collected"
            ),
            "stage_body_complete": result is not None,
            "capability_status": (
                result.get("capability_status", "unverified")
                if result
                and pytest_statuses.get(actual[identity]["nodeid"], {}).get("status")
                == "passed"
                else (
                    "failed"
                    if result and result.get("capability_status") == "failed"
                    else "unverified"
                )
            ),
            "artifact_sha256": result.get("artifact_sha256") if result else None,
            "evidence_sha256": result.get("evidence_sha256") if result else None,
            "open_reason": (
                "; ".join(
                    name + ": " + result["assertions"][name]["detail"]
                    for name in result["requested_assertions"]
                    if result["assertions"][name]["status"] != "validated"
                )
                if result
                else "No actual terminal SOURCE evidence for this frozen raw cell"
            ),
        }
        obligations.append(record)
    for origin, module in inventory["registry_without_metadata"]:
        obligations.append(
            {
                "obligation_id": "metadata-free:" + origin,
                "kind": "metadata-free",
                "owner": module,
                "origin": origin,
                "selector": None,
                "baseline_classification": "unclassified",
                "original_evidence": None,
                "status": "unverified",
                "open_reason": "Original registration has no raw metadata; separate source/manual counterpart required",
            }
        )
    for path, function in sorted(anchors):
        obligations.append(
            {
                "obligation_id": f"manual:{path}::{function}",
                "kind": "original-manual",
                "owner": path,
                "selector": f"{path}::{function}",
                "selector_state": "original-anchor",
                "baseline_classification": "unclassified",
                "original_evidence": None,
                "status": "unverified",
                "open_reason": "Retained original manual anchor needs its own source/facade/scope counterpart and evidence",
            }
        )
    legacy = legacy_preparation_obligations(inventory)
    obligations.extend(legacy)
    by_id = {item["obligation_id"]: item for item in obligations}
    if len(by_id) != len(obligations):
        raise ValueError("Colliding SOURCE accounting obligation IDs")
    classified = set()
    for capability in original_capabilities:
        identity = capability["obligation_id"]
        if identity not in by_id or identity in classified:
            raise ValueError("Unknown/duplicate original capability evidence")
        classification = capability["classification"]
        if classification not in {"positive", "preexisting-negative", "unclassified"}:
            raise ValueError(
                "New SOURCE rejection cannot classify an original capability"
            )
        if classification != "unclassified" and not capability.get("original_evidence"):
            raise ValueError(
                "Original capability classification requires original evidence"
            )
        by_id[identity]["baseline_classification"] = classification
        by_id[identity]["original_evidence"] = capability.get("original_evidence")
        classified.add(identity)
    return {
        "schema_version": 1,
        "baseline_sha256": baseline_sha256,
        "epoch": dict(epoch),
        "scope": "Complete frozen obligations with bounded actual SOURCE receipts; no public cutover or full capability closure",
        "inventory_observations": observations,
        "retained_missing_cells": sorted(retained),
        "collection_complete": collection_complete,
        "outcomes_complete": outcomes_complete,
        "counts": {
            "raw_rows": len(frozen_rows),
            "raw_caller_cells": len(cells),
            "metadata_free_registrations": len(inventory["registry_without_metadata"]),
            "opaque_raw_rows": len(opaque),
            "optional_raw_rows": len(optional_members),
            "original_manual_anchors": len(anchors),
            "migration_manual_additions": len(additions),
            "legacy_preparation_recipes": len(legacy),
            "actual_legacy_generated_variants": None,
            "actual_legacy_collected_pytest_ids": None,
            "planned_source_selectors": len(selected),
            "actual_source_collected_pytest_ids": len(actual),
            "observed_source_stage_bodies": len(results),
            "actual_source_pytest_terminal_outcomes": len(terminal_nodeids),
            "terminal_source_stage_outcomes": sum(
                actual[identity]["nodeid"] in terminal_nodeids for identity in results
            ),
            "passing_source_stage_contracts": sum(
                bool(item["stage_contract_passed"])
                and pytest_statuses.get(actual[identity]["nodeid"], {}).get("status")
                == "passed"
                for identity, item in results.items()
            ),
            "validated_raw_metadata_capabilities": sum(
                item.get("capability_status") == "validated"
                and pytest_statuses.get(actual[identity]["nodeid"], {}).get("status")
                == "passed"
                for identity, item in results.items()
            ),
            "linked_positive_prepared_counterparts": 0,
        },
        "pytest_outcomes": pytest_statuses,
        "obligations": obligations,
        "open_capability_work": "Original positive/negative variant evidence and explicit prepared FP64/fixed-source/shape/scope/facade counterpart links remain pending; raw caller x64 coverage cannot discharge them",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Intentionally replace the baseline; review the resulting diff.",
    )
    args = parser.parse_args(argv)
    actual = collect_inventory()
    expected = json.loads(args.output.read_text()) if args.output.exists() else None
    unavailable = unavailable_registrations()
    if args.refresh:
        invalid = [
            family
            for family, record in actual.get("optional", {}).items()
            if (
                record.get("source_environment_configured")
                or record.get("registration_available")
                or record.get("registered_case_rows")
            )
            and (
                record.get("import_error")
                or record.get("provenance_error")
                or record.get("baseline", {}).get("errors")
                or not record.get("registration_available")
                or not record.get("registered_case_rows")
            )
        ]
        if invalid:
            print(
                "Cannot refresh an invalid optional environment: " + ", ".join(invalid)
            )
            return 1
        invalid_rows, _ = compare_inventories(actual, actual, unavailable=unavailable)
        if invalid_rows:
            print(
                "Cannot refresh inconsistent optional observations: "
                + "; ".join(invalid_rows)
            )
            return 1
        if expected is not None:
            _, observations = compare_inventories(
                expected, actual, unavailable=unavailable
            )
            if retained_missing_cells(expected, actual):
                print("\n".join(observations))
                print(
                    "Refresh requires every retained baseline cell: prepare the pinned optional environments with both MODELS=all and the JAX stack providing conditional primitives."
                )
                return 1
        args.output.write_text(inventory_json(actual))
        print(
            f"Wrote {args.output}; review registry, collision, manual, and optional changes."
        )
        return 0
    if expected is None:
        print(
            f"Missing baseline: {args.output}; review a newly generated --refresh baseline."
        )
        return 1
    differences, observations = compare_inventories(
        expected, actual, unavailable=unavailable
    )
    print(
        f"Observed raw obligations: {len(actual['cases'])}; retained baseline obligations: {len(expected['cases'])} ({2 * len(expected['cases'])} caller-context obligations)."
    )
    for family in OPTIONAL_FAMILIES:
        print(
            f"{family}: observed {actual.get('optional', {}).get(family, {}).get('registered_case_rows', 0)}; retained baseline {len(expected.get('optional', {}).get(family, {}).get('baseline', {}).get('members', []))}."
        )
    for observation in observations:
        print(observation)
    if not differences:
        print(
            "SOURCE migration inventory identities match; no SOURCE acceptance is implied."
        )
        return 0
    print("\n".join(differences[:120]))
    if len(differences) > 120:
        print(f"... {len(differences) - 120} more differences")
    print(
        "Inventory changed. Review the identities and optional environment before using --refresh."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
