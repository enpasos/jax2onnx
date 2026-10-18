# scripts/source_precision_inventory.py

"""Inventory existing functionality without treating legacy tests as SOURCE evidence.

The default command compares the raw registry and handwritten regression identities
with the reviewed baseline. ``--refresh`` deliberately updates that baseline; its
diff still needs review. No generated-test precision rewrites or model execution
are used here. Optional examples remain visible when their environment is absent.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Mapping, Sequence
import json
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
    specs = {
        "maxtext": "src/maxtext/configs/models",
        "maxdiffusion": "src/maxdiffusion/configs",
    }
    for family, config_directory in specs.items():
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
        result[family] = {
            "source_environment_configured": bool(configured),
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
    return {
        "schema_version": 1,
        "scope": "Raw existing metadata and handwritten identities, not a support or acceptance verdict.",
        "source_audit": "not-run",
        "numerical_acceptance": "unverified",
        "conditional_registrations": CONDITIONAL_REGISTRATIONS,
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
        "optional": {
            family: {
                **observations,
                "registered_case_rows": sum(
                    len(metadata.get("testcases", []))
                    for _, _, metadata in registrations
                    if metadata.get("context") == f"examples.{family}"
                ),
            }
            for family, observations in optional.items()
        },
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
    observations = []
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
    differences = []
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
        if expected is not None:
            _, observations = compare_inventories(
                expected, actual, unavailable=unavailable
            )
            if observations:
                print("\n".join(observations))
                print(
                    "Refresh on a stack with these baseline entries available; this command will not delete their cells."
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
