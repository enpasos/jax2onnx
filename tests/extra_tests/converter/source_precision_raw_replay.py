# tests/extra_tests/converter/source_precision_raw_replay.py

"""Native capture and exact recipe adapters for the full raw migration schedule.

This is replay tooling. It does not change product admission or select criteria.
The old finite corpus replay continues to use its original preparation path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import dis
import functools
import hashlib
import json
from numbers import Integral
from pathlib import Path
import types
import sys
from typing import Any

import jax
from jax import export as jax_export
import numpy as np
import onnx
import onnx_ir as ir

from jax2onnx.precision import CheckStatus, PrecisionCheck, get_precision_report
from jax2onnx._compat.jax import is_selected_prng_key_dtype, selected_prng_key_words
from jax2onnx.converter.source_representation import representation_for_aval
from scripts.source_precision_inventory import canonical_contract
from tests.extra_tests.converter.source_precision_support import (
    PRESERVE_FACTORY_DTYPE,
    PreparedSourceCase,
    SourceCaseResult,
    SourcePreparationError,
    SourceSettings,
    _argument,
    _criteria,
    _export_prepared,
    _properties,
    _require_preserved_dtype,
    _typed_input_spec,
    source_configuration,
    value_evidence,
)


class RawReplayCapture:
    """Write each native result/returned artifact before subsequent checks."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.records: dict[str, Any] = {
            role: {"status": "not-attempted"}
            for role in (
                "eager",
                "fresh-jit",
                "concrete-typed-trace",
                "declared-typed-trace",
            )
        }
        self.records["product_admission"] = "not-attempted"
        self.records["raw_callback"] = {"status": "not-executed"}
        self.model: onnx.ModelProto | None = None
        self.facade_value: Any = None
        self.records["facade_return"] = {"status": "not-attempted"}
        self.native_values: dict[str, Any] = {}
        self.native_errors: dict[str, Exception] = {}
        self.gaps: list[dict[str, str]] = []
        self.target: Any = None
        self.target_state: Any = None
        self.preparation_complete = False
        self.records["native_outcomes"] = {}
        self.records["replay_readiness"] = {"status": "unverified", "gaps": self.gaps}

    def gap(self, role: str, exc: Exception) -> None:
        self.failure(role, exc)
        self.gaps.append({"role": role, "detail": str(exc)})
        self.records["replay_readiness"] = {"status": "unverified", "gaps": self.gaps}

    @property
    def readiness(self) -> CheckStatus:
        status = (
            CheckStatus.VALIDATED
            if self.preparation_complete and not self.gaps
            else CheckStatus.UNVERIFIED
        )
        self.records["replay_readiness"] = {"status": status.value, "gaps": self.gaps}
        return status

    def invoke(self, role: str, call: Any) -> Any:
        # Retain the actual object/exception before any evidence adapter runs.
        try:
            result = call()
        except Exception as exc:
            self.native_errors[role] = exc
            self.records["native_outcomes"][role] = {
                "status": "raised",
                "exception": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "detail": str(exc),
            }
            return None
        self.native_values[role] = result
        self.records["native_outcomes"][role] = {"status": "returned"}
        return result

    def save_arrays(self, role: str, values: Any) -> tuple[np.ndarray, ...] | None:
        try:
            return self.arrays(role, values)
        except Exception as exc:
            self.gap(role, exc)
            return None

    def save_document(self, role: str, document: Any) -> None:
        try:
            self.document(role, document)
        except Exception as exc:
            self.gap(role, exc)

    def bind_target(self, target: Any) -> None:
        self.target = target
        try:
            self.target_state = _capture_target_state(target)
        except Exception as exc:
            self.gap("instantiated-target-state", exc)
            raise SourcePreparationError(
                "Actual instantiated target needs an exact state capture adapter",
                category="replay-lane-gap",
            ) from exc
        self.save_document("instantiated-target-state", self.target_state)
        if (
            self.records.get("instantiated-target-state", {}).get("status")
            != "captured"
        ):
            raise SourcePreparationError(
                "Actual target state capture is unavailable", category="replay-lane-gap"
            )

    def check_target(self, role: str) -> bool:
        try:
            observed = _capture_target_state(self.target)
            if observed != self.target_state:
                self.save_document(role + "-changed-state", observed)
                raise SourcePreparationError(
                    "Actual instantiated target changed weights/state; an exact reset/capture schedule is required",
                    category="replay-lane-gap",
                )
        except Exception as exc:
            self.gap(role + "-state", exc)
            return False
        self.records[role + "-state"] = {"status": "unchanged"}
        return True

    def require_stable_target(self, role: str) -> None:
        if not self.check_target(role):
            raise SourcePreparationError(
                "Actual instantiated target state is not preserved; later source invocations were not attempted",
                category="replay-lane-gap",
            )

    def arrays(self, role: str, values: Any) -> tuple[np.ndarray, ...]:
        payloads = []
        properties = []
        for value in values:
            if is_selected_prng_key_dtype(getattr(value, "dtype", None)):
                array = np.array(selected_prng_key_words(value), copy=True)
                properties.append(
                    {
                        "dtype": value.dtype.name,
                        "shape": list(value.shape),
                        "weak_type": bool(getattr(value, "weak_type", False)),
                        "encoding": "threefry-key-words",
                        "physical_dtype": array.dtype.name,
                        "physical_shape": list(array.shape),
                    }
                )
            else:
                array = np.array(value, copy=True)
                properties.append(
                    {
                        "dtype": array.dtype.name,
                        "shape": list(array.shape),
                        "weak_type": bool(getattr(value, "weak_type", False)),
                    }
                )
            payloads.append(array)
        arrays = tuple(payloads)
        path = self.directory / (role + ".npz")
        with path.open("xb") as stream:
            np.savez(stream, **{f"value_{i}": v for i, v in enumerate(arrays)})
        self.records[role] = {
            "status": "captured",
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "properties": properties,
        }
        for array in arrays:
            array.setflags(write=False)
        return arrays

    def document(self, role: str, document: Any) -> None:
        path = self.directory / (role + ".json")
        payload = json.dumps(document, sort_keys=True, separators=(",", ":"))
        with path.open("x") as stream:
            stream.write(payload + "\n")
        self.records[role] = {
            "status": "captured",
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def failure(self, role: str, exc: Exception) -> None:
        self.records[role] = {
            "status": "failed",
            "exception": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "detail": str(exc),
        }

    def observe_facade_return(self, value: Any) -> None:
        # Successful ordinary return is observed before any normalization/read.
        self.facade_value = value
        kind = (
            "proto"
            if isinstance(value, onnx.ModelProto)
            else (
                "ir"
                if isinstance(value, ir.Model)
                else "file" if isinstance(value, str) else "unknown"
            )
        )
        self.records["facade_return"] = {
            "status": "returned",
            "kind": kind,
            "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
        self.records["product_admission"] = "returned-artifact"
        if isinstance(value, str):
            self.records["facade_return"]["path"] = value
            files = []
            self.records["facade_return"]["files"] = files
            for path in (Path(value), Path(value + ".data")):
                if path != Path(value) and not path.exists():
                    continue
                try:
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    files.append(
                        {
                            "path": str(path),
                            "bytes": path.stat().st_size,
                            "sha256": digest,
                        }
                    )
                except Exception as exc:
                    self.gap("facade-return-file-evidence", exc)

    def artifact(self, model: onnx.ModelProto) -> None:
        # The actual facade return is retained even if its evidence sink fails.
        self.model = model
        try:
            path = self.directory / "returned.onnx"
            with path.open("xb") as stream:
                stream.write(model.SerializeToString(deterministic=True))
            self.records["artifact"] = {
                "status": "captured",
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        except Exception as exc:
            self.gap("artifact", exc)


def structured_source_error(exc: Exception) -> dict[str, Any]:
    """Retain owning SourcePrecisionError fields without parsing its message."""
    return {
        name: getattr(exc, name)
        for name in (
            "phase",
            "category",
            "location",
            "primitive",
            "requested_dtypes",
            "precision_params",
            "opset",
            "requirement",
            "action",
        )
        if hasattr(exc, name)
    }


def _capture_recipe(value: Any, *, location: str = "callable", depth: int = 0) -> Any:
    """Capture only registered recipe containers and bounded dataclass objects.

    No object is rebuilt, copied into a different model or replaced for export.
    Unknown mutable state, identity tokens and unsupported fields stay lane gaps.
    """
    if depth > 48:
        raise SourcePreparationError(
            "Original recipe nesting needs an adapter", category="replay-lane-gap"
        )
    unresolved: list[str] = []
    canonical = canonical_contract(value, unresolved=unresolved)
    if not unresolved:
        return canonical

    def child(item: Any, suffix: str) -> Any:
        return _capture_recipe(item, location=location + "." + suffix, depth=depth + 1)

    typename = f"{type(value).__module__}.{type(value).__qualname__}"
    if (
        is_dataclass(value)
        and not isinstance(value, type)
        and (
            typename.startswith("equinox.nn.")
            or typename
            in {
                "jax2onnx.plugins.examples.eqx.gpt_oss.GPTOSSConfig",
                "jax2onnx.plugins.examples.nnx.gpt_oss_flax.GPTOSSConfig",
            }
        )
    ):
        return {
            "original_type": typename,
            "fields": {
                f.name: child(getattr(value, f.name), f.name) for f in fields(value)
            },
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise SourcePreparationError(
                "Opaque recipe mapping keys need an adapter", category="replay-lane-gap"
            )
        return {key: child(item, key) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return {
            "container": type(value).__name__,
            "values": [child(item, str(i)) for i, item in enumerate(value)],
        }
    if isinstance(value, functools.partial):
        return {
            "partial": child(value.func, "func"),
            "args": child(value.args, "args"),
            "kwargs": child(value.keywords, "kwargs"),
        }
    if typename == "jax2onnx.plugins.plugin_system._ConstructAndCall":
        return {
            "factory": child(value._ctor, "ctor"),
            "dtype": child(value._dtype, "dtype"),
            "args": child(value._args, "args"),
            "kwargs": child(value._kwargs, "kwargs"),
        }
    if isinstance(value, types.FunctionType):
        result = {
            "callable": f"{value.__module__}.{value.__qualname__}",
            "defaults": child(value.__defaults__, "defaults"),
            "kwdefaults": child(value.__kwdefaults__, "kwdefaults"),
        }
        result["captures"] = {
            name: child(cell.cell_contents, "capture." + name)
            for name, cell in zip(
                value.__code__.co_freevars, value.__closure__ or (), strict=True
            )
        }
        return result
    raise SourcePreparationError(
        f"No exact original-object capture adapter for {location}: {typename}",
        category="replay-lane-gap",
    )


_LIBRARY_STATE_NAMESPACES = (
    "jax",
    "jaxlib",
    "numpy",
    "ml_dtypes",
    "flax",
    "equinox",
    "math",
    "operator",
    "functools",
)


def _library_namespace(name: str) -> bool:
    return any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in _LIBRARY_STATE_NAMESPACES
    )


def _imported_executable_binding(value: Any) -> dict[str, str] | None:
    """Recognize loaded module API functions, never models or bound owners.

    A namespace is only a lookup boundary. Exact executable types and identity
    at the defining module's top-level export are both required. Classes and
    owned methods always need actual state capture or an explicit lane gap.
    """
    owner = getattr(value, "__self__", None)
    if owner is not None and not isinstance(owner, types.ModuleType):
        return None
    executable_types = {
        types.FunctionType,
        types.BuiltinFunctionType,
        np.ufunc,
        type(np.sum),
        type(jax.numpy.tanh),
        jax.custom_jvp,
        jax.custom_vjp,
    }
    if type(value) not in executable_types:
        return None
    module_name = getattr(value, "__module__", None) or type(value).__module__
    if not _library_namespace(module_name):
        return None
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if not isinstance(name, str) or "." in name:
        return None  # Class/static methods are not top-level API bindings.
    module = sys.modules.get(module_name)
    if not isinstance(module, types.ModuleType) or vars(module).get(name) is not value:
        return None
    return {
        "library_binding": module_name + "." + name,
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }


_JAX_SCALAR_META = type(jax.numpy.int32)
_JAX_SCALAR_DTYPE_ALIASES = tuple(
    (name, vars(jax.numpy)[name], vars(vars(jax.numpy)[name])["dtype"])
    for name in (
        "bool_",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
    if name in vars(jax.numpy)
)


def _jax_scalar_dtype_binding(value: Any) -> dict[str, str] | None:
    """Capture only original loaded JAX scalar alias objects and exact dtypes.

    JAX's scalar metaclass compares equal to the NumPy scalar type; identity
    checks and a separate alias record prevent that equality from erasing a
    changed binding. Unknown class/static data remains outside this adapter.
    """
    public = sys.modules.get("jax.numpy")
    defining = sys.modules.get("jax._src.numpy.scalar_types")
    if (
        type(value) is not _JAX_SCALAR_META
        or public is not jax.numpy
        or not isinstance(defining, types.ModuleType)
        or vars(defining).get("_ScalarMeta") is not _JAX_SCALAR_META
    ):
        return None
    for name, original, original_dtype in _JAX_SCALAR_DTYPE_ALIASES:
        if value is not original:
            continue
        canonical = np.dtype("bool" if name == "bool_" else name)
        if (
            vars(public).get(name) is not value
            or vars(defining).get(name) is not value
            or value.__module__ != "jax.numpy"
            or value.__name__ != canonical.type.__name__
            or value.__qualname__ != canonical.type.__name__
            or set(vars(value))
            - {
                "dtype",
                "__module__",
                "__doc__",
                "__dict__",
                "__weakref__",
                "__qualname__",
            }
        ):
            return None
        dtype = vars(value).get("dtype")
        if (
            dtype is not original_dtype
            or type(dtype) is not type(canonical)
            or dtype.type is not canonical.type
            or dtype.str != canonical.str
            or dtype.metadata is not None
            or np.dtype(value).type is not canonical.type
        ):
            return None
        return {
            "jax_scalar_alias": "jax.numpy." + name,
            "metaclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "dtype": canonical.name,
            "dtype_str": canonical.str,
            "numpy_scalar_type": f"{canonical.type.__module__}.{canonical.type.__qualname__}",
        }
    return None


def _function_owned_data(value: Any, child: Any, *, depth: int = 0) -> Any:
    if depth > 48:
        raise SourcePreparationError(
            "Referenced function data nesting needs an exact adapter",
            category="replay-lane-gap",
        )
    if isinstance(value, types.ModuleType):
        raise SourcePreparationError(
            "Referenced function data contains module state without an exact adapter",
            category="replay-lane-gap",
        )
    if isinstance(value, type) and value not in {bool, int, float, complex, str, bytes}:
        alias = _jax_scalar_dtype_binding(value)
        if alias is not None:
            return alias
        try:
            scalar_type = np.dtype(value).type
        except (TypeError, ValueError):
            scalar_type = None
        if scalar_type is not value:
            raise SourcePreparationError(
                "Referenced function class/static state needs an exact adapter",
                category="replay-lane-gap",
            )
    if (
        isinstance(value, types.BuiltinFunctionType)
        and getattr(value, "__self__", None) is not None
        and not isinstance(value.__self__, types.ModuleType)
    ):
        raise SourcePreparationError(
            "Owned native method needs an exact owner/state adapter",
            category="replay-lane-gap",
        )
    if isinstance(value, Mapping):
        return {
            "mapping": sorted(
                [
                    [child(key), _function_owned_data(item, child, depth=depth + 1)]
                    for key, item in value.items()
                ],
                key=lambda pair: json.dumps(pair[0], sort_keys=True),
            )
        }
    if isinstance(value, (tuple, list)):
        return {
            "container": type(value).__name__,
            "values": [
                _function_owned_data(item, child, depth=depth + 1) for item in value
            ],
        }
    return child(value)


def _function_global_state(fn: Any, child: Any) -> dict[str, Any]:
    """Bind referenced data; retain explicit boundaries for executable libraries.

    Direct model data and function attributes use the real state adapter. Library
    namespace attributes are resolved only along bytecode-visible access chains;
    arbitrary module escape or custom class/static dependencies remain gaps.
    """
    accesses: dict[str, set[tuple[str, ...]]] = {}

    def inspect_code(code: Any) -> None:
        instructions = list(dis.get_instructions(code))
        for index, instruction in enumerate(instructions):
            if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}:
                continue
            name = instruction.argval
            if name not in fn.__globals__:
                continue  # Native builtin lookup, not mutable module model data.
            path = []
            for following in instructions[index + 1 :]:
                if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                    break
                path.append(following.argval)
            accesses.setdefault(name, set()).add(tuple(path))
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                inspect_code(constant)

    def dependency(value: Any) -> Any:
        binding = _imported_executable_binding(value)
        if binding is not None:
            # Imported implementation is owned by the environment/source pins.
            # Model instances and owned methods cannot enter this branch.
            return binding
        return _function_owned_data(value, child)

    inspect_code(fn.__code__)
    result = {}
    for name, paths in sorted(accesses.items()):
        value = fn.__globals__[name]
        if not isinstance(value, types.ModuleType):
            result[name] = dependency(value)
            continue
        if not _library_namespace(value.__name__):
            raise SourcePreparationError(
                f"Referenced module state needs an exact adapter: {value.__name__}",
                category="replay-lane-gap",
            )
        selected = {}
        for path in sorted(paths):
            if not path:
                raise SourcePreparationError(
                    f"Referenced library module {value.__name__} is passed without a bounded attribute path",
                    category="replay-lane-gap",
                )
            member = value
            for attribute in path:
                if not isinstance(member, types.ModuleType) or attribute not in vars(
                    member
                ):
                    raise SourcePreparationError(
                        f"Referenced attribute path needs an exact state adapter: {name}.{'.'.join(path)}",
                        category="replay-lane-gap",
                    )
                member = vars(member)[attribute]
            selected[".".join(path)] = dependency(member)
        result[name] = {"library_namespace": value.__name__, "attributes": selected}
    return result


def _capture_target_state(value: Any, *, depth: int = 0) -> Any:
    """Snapshot the actual callable, with bounded Eqx/NNX state adapters.

    The snapshot never constructs another model or resets unknown mutable state.
    Unsupported fields remain explicit replay gaps, including new graph layouts.
    """
    if depth > 48:
        raise SourcePreparationError(
            "Target state nesting needs an exact adapter", category="replay-lane-gap"
        )

    def child(item: Any) -> Any:
        return _capture_target_state(item, depth=depth + 1)

    if is_selected_prng_key_dtype(getattr(value, "dtype", None)):
        words = np.asarray(selected_prng_key_words(value))
        return {
            "dtype": value.dtype.name,
            "shape": list(value.shape),
            "weak_type": bool(getattr(value, "weak_type", False)),
            "key_words_sha256": hashlib.sha256(words.tobytes()).hexdigest(),
        }
    if isinstance(value, jax.tree_util.PyTreeDef):
        return {
            "pytree_node": child(value.node_data()),
            "children": [child(tree) for tree in value.children()],
        }
    if isinstance(value, types.MethodType):
        return {"bound_target": child(value.__self__), "method": child(value.__func__)}
    if isinstance(value, types.FunctionType):
        return {
            "callable": f"{value.__module__}.{value.__qualname__}",
            "defaults": _function_owned_data(value.__defaults__, child),
            "kwdefaults": _function_owned_data(value.__kwdefaults__, child),
            "attributes": _function_owned_data(value.__dict__, child),
            "referenced_globals": _function_global_state(value, child),
            "captures": {
                name: _function_owned_data(cell.cell_contents, child)
                for name, cell in zip(
                    value.__code__.co_freevars, value.__closure__ or (), strict=True
                )
            },
        }
    if isinstance(value, functools.partial):
        return {
            "partial": child(value.func),
            "args": child(value.args),
            "kwargs": child(value.keywords),
        }
    if getattr(value, "__jax2onnx_factory__", False):
        raise SourcePreparationError(
            "An actual target cannot be verified through its constructor recipe",
            category="replay-lane-gap",
        )
    # NNX split reads both static graph configuration and all Variable state.
    # It retains the original target object; no merge/reconstruction is used.
    from flax import nnx

    if isinstance(value, nnx.Module):
        graph, state = nnx.split(value)
        return {
            "target_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "graph": child(graph),
            "state": child(state),
        }
    if isinstance(value, nnx.Variable):
        raw = value.get_raw_value() if hasattr(value, "get_raw_value") else value.value
        metadata = value.get_metadata()
        return {
            "variable_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": child(raw),
            "metadata": child(metadata),
        }
    typename = f"{type(value).__module__}.{type(value).__qualname__}"
    if (
        is_dataclass(value)
        and not isinstance(value, type)
        and (
            typename.startswith(
                (
                    "equinox.nn.",
                    "jax2onnx.plugins.examples.eqx.",
                    "flax.nnx.graph.",
                    "flax.nnx.graphlib.",
                )
            )
            or typename in {"jax2onnx.plugins.examples.nnx.gpt_oss_flax.GPTOSSConfig"}
        )
    ):
        return {
            "state_type": typename,
            "fields": {
                field.name: child(getattr(value, field.name)) for field in fields(value)
            },
        }
    graph_fields = {
        "NodeRef": ("index",),
        "NodeDef": ("type", "index", "outer_index", "num_attributes", "metadata"),
        "VariableDef": ("type", "index", "outer_index", "metadata", "array_refdef"),
        "ArrayRefDef": ("index", "outer_index"),
    }
    if type(value).__module__ == "flaxlib" and type(value).__name__ in graph_fields:
        return {
            "state_type": typename,
            "fields": {
                name: child(getattr(value, name))
                for name in graph_fields[type(value).__name__]
            },
        }
    if isinstance(value, Mapping):
        return {
            "mapping": sorted(
                [[child(key), child(item)] for key, item in value.items()],
                key=lambda pair: json.dumps(pair[0], sort_keys=True),
            )
        }
    if isinstance(value, (list, tuple)):
        return {
            "container": type(value).__name__,
            "values": [child(item) for item in value],
        }
    unresolved: list[str] = []
    result = canonical_contract(value, unresolved=unresolved)
    if unresolved:
        raise SourcePreparationError(
            f"No exact instantiated state adapter for {typename}",
            category="replay-lane-gap",
        )
    return result


def replay_adapter(row: Any, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    adapter = manifest.get("rows", {}).get(f"{row.origin}::{row.ordinal}", {})
    if adapter and adapter.get("contract_sha256") != row.contract_sha256:
        raise SourcePreparationError(
            "Replay adapter does not bind this exact raw contract",
            category="replay-lane-gap",
        )
    if row.unresolved_recipes and not adapter.get("capture_original_object"):
        raise SourcePreparationError(
            "Opaque original recipe has no registered capture adapter",
            category="replay-lane-gap",
        )
    return adapter


def _native_specs(specs: Any) -> tuple[Any, ...]:
    # One shared native scope for the exact original string declarations. Existing
    # symbolic objects keep their original scope and are never string-recreated.
    scope = jax_export.SymbolicScope()
    symbols: dict[str, Any] = {}
    normalized = []
    for spec in specs:
        shape = []
        for dim in spec.shape:
            if isinstance(dim, str):
                if dim not in symbols:
                    parsed = jax_export.symbolic_shape(dim, scope=scope)
                    if len(parsed) != 1:
                        raise SourcePreparationError(
                            "Symbolic declaration is not one dimension",
                            category="replay-lane-gap",
                        )
                    symbols[dim] = parsed[0]
                dim = symbols[dim]
            shape.append(dim)
        normalized.append(
            jax.ShapeDtypeStruct(tuple(shape), spec.dtype, weak_type=spec.weak_type)
        )
    return tuple(normalized)


def _concrete_shapes(
    specs: Any, bindings: Mapping[str, Any]
) -> tuple[tuple[int, ...], ...]:
    shapes = []
    for spec in specs:
        concrete = []
        for dim in spec.shape:
            if isinstance(dim, Integral) and not isinstance(dim, bool):
                extent = int(dim)
            elif isinstance(dim, str) and dim in bindings:
                extent = bindings[dim]
            else:
                raise SourcePreparationError(
                    f"Original symbolic dimension {dim!s} needs an explicit exact sample binding",
                    category="replay-lane-gap",
                )
            if not isinstance(extent, int) or isinstance(extent, bool) or extent < 0:
                raise SourcePreparationError(
                    "Sample binding must be a nonnegative integer",
                    category="replay-lane-gap",
                )
            concrete.append(extent)
        shapes.append(tuple(concrete))
    return tuple(shapes)


def _generated_inputs(shapes: Any, specs: Any, scenario: Any) -> tuple[np.ndarray, ...]:
    digest = hashlib.sha256(
        f"{scenario.provenance.origin}\0{scenario.provenance.ordinal}".encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    generated = []
    for shape, spec in zip(shapes, specs, strict=True):
        dtype = np.dtype(spec.dtype)
        if dtype.kind == "f" or dtype.name == "bfloat16":
            data = rng.standard_normal(shape) * 0.25
        elif dtype.kind in "iu":
            data = rng.integers(0, 5, size=shape)
        elif dtype.kind == "b":
            data = rng.random(shape) > 0.5
        else:
            raise SourcePreparationError(
                f"No reviewed raw sample domain for {dtype}; supply original typed input_values",
                category="replay-lane-gap",
            )
        generated.append(np.asarray(data, dtype=dtype))
    return tuple(generated)


def _physical_arrays(values: Any, indices: Any) -> tuple[np.ndarray, ...]:
    selected = set(indices or ())
    return tuple(
        representation_for_aval(
            jax.ShapeDtypeStruct(value.shape, value.dtype),
            boundary_layout="NCHW" if i in selected else None,
        ).encode_array(value)
        for i, value in enumerate(values)
    )


def _save_physical(
    capture: RawReplayCapture, role: str, values: Any, indices: Any
) -> None:
    try:
        capture.arrays(role, _physical_arrays(values, indices))
    except Exception as exc:
        # A replay adapter failure must not erase a successful native attempt or
        # prevent ordinary product admission from checking its own real options.
        capture.gap(role, exc)


def prepare_raw_source_case(
    metadata: Mapping[str, Any],
    scenario: Any,
    *,
    adapter: Mapping[str, Any],
    capture: RawReplayCapture,
    settings_observer: Any = None,
) -> PreparedSourceCase:
    """Attempt native eager/JIT/trace before any SOURCE shape/profile validator."""
    from tests.extra_tests.converter.source_precision_diagnostics import (
        typed_trace_manifest,
    )

    identity = tuple(
        str(metadata.get(key, "default"))
        for key in ("context", "component", "testcase")
    )
    with source_configuration(scenario):
        settings = SourceSettings.observe()
        if settings_observer is not None:
            settings_observer(settings)
        declared = tuple(scenario.input_specs or ())
        params = dict(metadata.get("input_params") or {})
        names = tuple(params)
        named_specs = []
        for value in params.values():
            spec = (
                value
                if isinstance(value, jax.ShapeDtypeStruct)
                else _typed_input_spec(value)
            )
            if spec is None:
                spec = jax.eval_shape(lambda x: x, value)
            named_specs.append(spec)
        all_declarations = (*declared, *named_specs)
        for spec in all_declarations:
            _require_preserved_dtype(spec.dtype)
        bindings = (adapter.get("samples") or [{}])[0]
        concrete = _concrete_shapes(all_declarations, bindings)
        supplied = scenario.input_values
        if supplied is None:
            supplied = _generated_inputs(concrete[: len(declared)], declared, scenario)
        if len(supplied) != len(declared):
            raise SourcePreparationError(
                "Original input value/declaration arity differs"
            )
        arguments = tuple(
            _argument(value, spec)
            for value, spec in zip(supplied, declared, strict=True)
        )
        named = {
            name: _argument(params[name], spec)
            for name, spec in zip(names, named_specs, strict=True)
        }
        all_arguments = (*arguments, *named.values())
        if tuple(tuple(value.shape) for value in all_arguments) != concrete:
            raise SourcePreparationError(
                "Supplied input bytes do not match the explicit original sample binding",
                category="replay-lane-gap",
            )
        capture.save_arrays("inputs", all_arguments)
        _save_physical(
            capture, "physical-feeds", all_arguments, metadata.get("inputs_as_nchw")
        )
        capture.save_document(
            "sample-bindings",
            {
                "primary": bindings,
                "additional_native_samples": adapter.get("samples", [])[1:],
                "declarations": [
                    _properties(spec).__dict__ for spec in all_declarations
                ],
            },
        )
        target = metadata.get("callable")
        recipe = (
            _capture_recipe(target)
            if adapter.get("capture_original_object")
            else canonical_contract(target, unresolved=[])
        )
        capture.save_document("original-recipe", recipe)
        if getattr(target, "__jax2onnx_factory__", False):
            if scenario.model_dtype is not PRESERVE_FACTORY_DTYPE:
                raise SourcePreparationError(
                    "Raw replay must retain original factory dtype",
                    category="replay-lane-gap",
                )
            target = target.instantiate()
        if not callable(target):
            raise SourcePreparationError(
                "Original recipe did not produce a callable", category="replay-lane-gap"
            )
        capture.bind_target(target)

        def original(*values: Any) -> Any:
            return target(
                *values[: len(arguments)],
                **dict(zip(names, values[len(arguments) :], strict=True)),
            )

        outputs_by_role = {}
        trees = {}

        def native_outputs(role: str, values: Any) -> None:
            result = capture.invoke(
                role,
                lambda: jax.device_get(
                    (original if role.endswith("eager") else jax.jit(original))(*values)
                ),
            )
            if role in capture.native_values:
                try:
                    leaves, tree = jax.tree_util.tree_flatten(result)
                    # Native return status is already retained independently of
                    # flattening, NumPy conversion, encoding and file adapters.
                    outputs_by_role[role] = tuple(leaves)
                    trees[role] = str(tree)
                    capture.save_arrays(role, leaves)
                    _save_physical(
                        capture,
                        role + "-physical-reference",
                        leaves,
                        metadata.get("outputs_as_nchw"),
                    )
                except Exception as exc:
                    capture.gap(role + "-capture", exc)
            capture.require_stable_target(role)

        for role in ("eager", "fresh-jit"):
            native_outputs(role, all_arguments)
        for index, sample_bindings in enumerate(adapter.get("samples", [])[1:], 1):
            label = f"native-sample-{index}"
            if scenario.input_values is not None:
                capture.records[label] = {
                    "status": "unavailable",
                    "detail": "Original supplied input bytes fix the sample shape; no alternate bytes are invented.",
                }
                capture.gap(
                    label + "-capture",
                    SourcePreparationError(
                        "Exact alternate supplied-input bytes are unavailable",
                        category="replay-lane-gap",
                    ),
                )
                continue
            sample_shapes = _concrete_shapes(all_declarations, sample_bindings)
            if sample_shapes[len(declared) :] != concrete[len(declared) :]:
                capture.gap(
                    label + "-capture",
                    SourcePreparationError(
                        "Original named input bytes require an exact alternate parameter adapter",
                        category="replay-lane-gap",
                    ),
                )
                continue
            values = _generated_inputs(
                sample_shapes[: len(declared)], declared, scenario
            )
            sample_args = (
                *(
                    _argument(value, spec)
                    for value, spec in zip(values, declared, strict=True)
                ),
                *named.values(),
            )
            capture.save_arrays(label + "-inputs", sample_args)
            for role in ("eager", "fresh-jit"):
                native_outputs(label + "-" + role, sample_args)

        # Each actual tracing attempt is independent from the other attempt and
        # from serialization. A const/trace adapter cannot invent native failure.
        concrete_closed = capture.invoke(
            "concrete-typed-trace", lambda: jax.make_jaxpr(original)(*all_arguments)
        )
        capture.require_stable_target("concrete-typed-trace")
        if "concrete-typed-trace" in capture.native_values:
            try:
                capture.document(
                    "concrete-typed-trace", typed_trace_manifest(concrete_closed)
                )
            except Exception as exc:
                capture.gap("concrete-typed-trace", exc)
            capture.save_arrays("captured-constants", concrete_closed.consts)
        all_specs = None
        try:
            all_specs = _native_specs(all_declarations)
        except Exception as exc:
            capture.gap("declared-input-specs", exc)
        closed = None
        trace = None
        if all_specs is not None:
            closed = capture.invoke(
                "declared-typed-trace", lambda: jax.make_jaxpr(original)(*all_specs)
            )
            capture.require_stable_target("declared-typed-trace")
            if "declared-typed-trace" in capture.native_values:
                try:
                    trace = typed_trace_manifest(closed)
                    capture.document("declared-typed-trace", trace)
                except Exception as exc:
                    capture.gap("declared-typed-trace", exc)
        if capture.native_errors:
            raise SourcePreparationError(
                "Original native invocation/tracing raised; actual outcomes and independent captures retained",
                category="native-source-failure",
            )
        if (
            closed is None
            or trace is None
            or all_specs is None
            or "eager" not in outputs_by_role
            or "fresh-jit" not in outputs_by_role
        ):
            raise SourcePreparationError(
                "Native returns/traces need a supported evidence adapter; successful native observations are retained",
                category="replay-lane-gap",
            )
        if trees["eager"] != trees["fresh-jit"]:
            raise SourcePreparationError(
                "Original eager/fresh-JIT trees differ",
                category="native-source-failure",
            )
        traced_inputs = tuple(_properties(var.aval) for var in closed.jaxpr.invars)
        traced_outputs = tuple(_properties(var.aval) for var in closed.jaxpr.outvars)
        try:
            reference = tuple(
                np.array(value, copy=True) for value in outputs_by_role["eager"]
            )
            input_evidence = tuple(
                value_evidence(value, prop)
                for value, prop in zip(all_arguments, traced_inputs, strict=True)
            )
            reference_evidence = tuple(
                value_evidence(value, prop)
                for value, prop in zip(reference, traced_outputs, strict=True)
            )
            original_trace_json = json.dumps(
                trace, sort_keys=True, separators=(",", ":")
            )
        except Exception as exc:
            capture.gap("prepared-evidence", exc)
            raise SourcePreparationError(
                "Actual native values need a supported prepared-evidence adapter",
                category="replay-lane-gap",
            ) from exc
        if len(reference) != len(traced_outputs) or any(
            value.dtype.name != prop.dtype
            for value, prop in zip(reference, traced_outputs, strict=True)
        ):
            raise SourcePreparationError(
                "Original eager/typed trace dtype or output arity differs",
                category="native-source-failure",
            )
        for value in reference:
            value.setflags(write=False)
        if (
            adapter.get("capture_original_object")
            and _capture_recipe(metadata.get("callable")) != recipe
        ):
            capture.gap(
                "original-recipe-state",
                SourcePreparationError(
                    "Original registered constructor/config recipe changed during native preparation",
                    category="replay-lane-gap",
                ),
            )
            raise SourcePreparationError(
                "Original registered recipe state is not preserved",
                category="replay-lane-gap",
            )
        if SourceSettings.observe() != settings:
            raise SourcePreparationError(
                "Original callable changed caller JAX settings during preparation"
            )
        capture.require_stable_target("native-preparation-complete")
        capture.preparation_complete = True
        capture.readiness
        return PreparedSourceCase(
            identity,
            scenario,
            target,
            dict(metadata),
            all_specs[: len(arguments)],
            arguments,
            traced_inputs,
            traced_outputs,
            reference,
            trees["eager"],
            _criteria(metadata, scenario),
            settings,
            original_trace_json,
            input_evidence,
            reference_evidence,
            closed,
            dict(zip(names, all_specs[len(arguments) :], strict=True)),
            named,
        )


def run_raw_source_case(
    prepared: PreparedSourceCase, *, capture: RawReplayCapture
) -> SourceCaseResult:
    """Ordinary product observations stay separate from replay adapter readiness."""
    model = None
    schema = CheckStatus.UNVERIFIED
    callback_status = "not-executed"
    settings = None
    error = None
    stage = "source-export"
    product_attempted = False
    state_ok = capture.target is prepared.fn and capture.check_target(
        "before-product-export"
    )
    if not state_ok:
        capture.gap(
            "before-product-export",
            SourcePreparationError(
                "Prepared callable identity/state differs from the actual captured target",
                category="replay-lane-gap",
            ),
        )
    else:
        try:
            with source_configuration(prepared.scenario):
                settings = SourceSettings.observe()
                product_attempted = True
                model = _export_prepared(
                    prepared,
                    artifact_observer=capture.artifact,
                    facade_return_observer=capture.observe_facade_return,
                    output_directory=capture.directory,
                )
            schema = CheckStatus.VALIDATED
        except Exception as exc:
            error = exc
            if (
                capture.records["facade_return"]["status"] == "returned"
                and getattr(exc, "category", None) == "replay-lane-gap"
            ):
                capture.gap("facade-return-normalization", exc)
            else:
                capture.failure("source-export", exc)
                capture.records["source_error"] = structured_source_error(exc)
            if isinstance(
                exc, (onnx.checker.ValidationError, onnx.shape_inference.InferenceError)
            ):
                schema = CheckStatus.FAILED
        finally:
            # A changed target does not erase a real returned artifact/checker
            # outcome. The native model is never reset or reconstructed here.
            capture.check_target("after-product-export")
    model = model if model is not None else capture.model
    report = get_precision_report(model) if model is not None else None
    lowering = (
        next(
            (check.status for check in report.checks if check.category == "lowering"),
            CheckStatus.UNVERIFIED,
        )
        if report
        else CheckStatus.UNVERIFIED
    )
    capture.records["product_admission"] = (
        "returned-artifact"
        if capture.records["facade_return"]["status"] == "returned" or model is not None
        else "rejected" if product_attempted else "not-attempted"
    )
    capture.records["schema_observation"] = schema.value
    capture.records["product_audit_status"] = (
        lowering
        if report
        else (
            CheckStatus.UNVERIFIED
            if error is None
            or getattr(error, "category", None)
            in {"profile-exclusion", "replay-lane-gap"}
            else CheckStatus.FAILED
        )
    )
    if model is not None and error is None:
        available = {}
        for role in (
            "physical-feeds",
            "eager-physical-reference",
            "fresh-jit-physical-reference",
        ):
            record = capture.records.get(role, {})
            if record.get("status") == "captured" and Path(record["path"]).is_file():
                available[role] = {"path": record["path"], "sha256": record["sha256"]}
        capture.save_document(
            "physical-port-bindings",
            {
                "inputs": [port.name for port in model.graph.input],
                "outputs": [port.name for port in model.graph.output],
                "captures": available,
                "original_output_path": prepared.metadata.get("output_path"),
                "original_recording_path": prepared.metadata.get(
                    "record_primitive_calls_file"
                ),
                "actual_destinations": [
                    str(path)
                    for path in sorted(capture.directory.glob("facade-return.onnx*"))
                ],
            },
        )
        stage = "raw-callback"
        callback = prepared.metadata.get("post_check_onnx_graph")
        callback_status = "absent" if callback is None else "error"
        try:
            if callback is not None:
                callback_status = "passed" if callback(model) else "failed"
            if callback_status == "failed":
                raise AssertionError("Raw metadata structural callback failed")
        except Exception as exc:
            error = exc
            capture.failure("raw-callback", exc)
    if model is not None:
        capture.check_target("after-raw-callback")
    capture.records["raw_callback"] = {
        "status": callback_status,
        "detail": str(error) if stage == "raw-callback" and error is not None else "",
    }
    readiness = capture.readiness
    category = (
        "replay-lane-gap"
        if capture.gaps
        else (
            "raw-callback-failure"
            if stage == "raw-callback" and error
            else (
                getattr(error, "category", "implementation-gap")
                if error
                else "audited-profile"
            )
        )
    )
    status = (
        CheckStatus.UNVERIFIED
        if capture.gaps
        else (
            CheckStatus.UNVERIFIED
            if getattr(error, "category", None) == "profile-exclusion"
            else CheckStatus.FAILED if error else lowering
        )
    )
    detail = (
        "; ".join(gap["role"] + ": " + gap["detail"] for gap in capture.gaps)
        if capture.gaps
        else (
            str(error)
            if error
            else "Product export/schema checked; raw callback recorded separately. Runtime/numerics unverified."
        )
    )
    return SourceCaseResult(
        (
            "replay-capture-pending"
            if readiness is CheckStatus.UNVERIFIED and capture.gaps
            else getattr(error, "phase", stage) if error else "artifact-checked"
        ),
        category,
        PrecisionCheck("source-audit", status, detail),
        prepared,
        model,
        export_settings=settings,
        schema=schema,
        raw_callback_status=callback_status,
    )
