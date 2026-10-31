# scripts/observe_native_axisdata.py

"""Report finite native AxisData metadata before CI tests; grant no schema support."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import types
from typing import Any


_MISSING = object()
_METHODS = ("__setattr__", "__delattr__")
_SOURCE_FUNCTIONS = (
    "_frozen_set_del_attr",
    "_frozen_get_del_attr",
    "_add_slots",
    "_update_func_cell_for__class__",
    "_dataclass_getstate",
    "_dataclass_setstate",
    "fields",
    "_replace",
)


def _binding(namespace: Any, name: str) -> Any:
    # Callers provide intrinsic namespaces, never a proxy over a foreign mapping.
    if type(namespace) is not dict and type(namespace) is not types.MappingProxyType:
        raise TypeError("observation requires an intrinsic namespace")
    # Identity/type guards precede string equality; no foreign dictionary keys run.
    for key, value in namespace.items():
        if type(key) is str and key == name:
            return value
    return _MISSING


def _module_namespace(module: Any) -> dict[str, Any]:
    if type(module) is not types.ModuleType:
        raise TypeError("observation requires an exact module")
    return vars(types.ModuleType)["__dict__"].__get__(module, types.ModuleType)


def _token(value: Any) -> dict[str, Any]:
    tag = "unknown"
    if value is _MISSING:
        tag = "missing"
    else:
        for native, name in (
            (type(None), "none"),
            (bool, "bool"),
            (int, "int"),
            (str, "str"),
            (tuple, "tuple"),
            (type, "class"),
            (types.FunctionType, "function"),
            (types.CellType, "cell"),
            (types.CodeType, "code"),
            (types.MemberDescriptorType, "member_descriptor"),
            (types.GetSetDescriptorType, "getset_descriptor"),
            (types.MappingProxyType, "mappingproxy"),
            (dict, "dict"),
            (list, "list"),
            (frozenset, "frozenset"),
            (slice, "slice"),
            (float, "float"),
            (bytes, "bytes"),
            (TypeError, "TypeError"),
            (ValueError, "ValueError"),
            (OSError, "OSError"),
            (ImportError, "ImportError"),
        ):
            if type(value) is native:
                tag = name
                break
    return {"id": id(value), "type_id": id(type(value)), "known_type": tag}


def _literal_text(value: Any, limit: int) -> Any:
    if type(value) is not str:
        return {"unresolved": "non-literal text", "value": _token(value)}
    if len(value) > limit:
        return {"unresolved": "text exceeds limit", "characters": len(value)}
    return value


def _stored_token(value: Any) -> dict[str, Any]:
    result = _token(value)
    if type(value) is str:
        result["literal"] = _literal_text(value, 256)
    elif type(value) is bool or value is None:
        result["literal"] = value
    elif type(value) is int:
        if value.bit_length() <= 64:
            result["literal"] = value
        else:
            result["unresolved"] = "integer exceeds 64 bits"
    return result


def _code_constants(code: Any, budget: list[int], depth: int = 0) -> Any:
    if type(code) is not types.CodeType:
        raise TypeError("constant observation requires exact code")

    def describe(value: Any, level: int) -> Any:
        if level > 6:
            raise ValueError("constant observation exceeds finite depth")
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError("constant observation exceeds node budget")
        item = _token(value)
        if type(value) is types.CodeType:
            item["constants"] = _code_constants(value, budget, level + 1)
        elif type(value) is tuple or type(value) is frozenset:
            if len(value) > 256:
                raise ValueError("constant container exceeds finite width")
            # These are exact builtin containers; only fixed type tags are read.
            item["length"] = len(value)
            item["item_types"] = [describe(part, level + 1) for part in value]
        elif type(value) is slice:
            item["components"] = {
                name: _token(vars(slice)[name].__get__(value, slice))
                for name in ("start", "stop", "step")
            }
        return item

    if depth > 6 or len(code.co_consts) > 256:
        raise ValueError("constant observation exceeds finite depth or width")
    return [describe(value, depth) for value in code.co_consts]


def _class_metadata(
    cls: Any, pins: list[Any], fields: Any = _MISSING, params: Any = _MISSING
) -> dict[str, Any]:
    result: dict[str, Any] = {"class": _token(cls)}
    pins.append(cls)
    if type(cls) is not type:
        result["unresolved"] = "non-builtin class"
        return result
    namespace = vars(type)["__dict__"].__get__(cls)
    if len(namespace) > 128 or any(
        type(key) is not str or len(key) > 256 for key in namespace
    ):
        raise ValueError("class namespace exceeds finite literal key schema")
    result["namespace_keys"] = sorted(namespace)
    result["members"] = {name: _token(value) for name, value in namespace.items()}
    pins.extend(namespace)
    pins.extend(value for value in namespace.values())
    slots = _binding(namespace, "__slots__")
    result["slots"] = _token(slots)
    if type(slots) is tuple and len(slots) <= 64:
        result["slots"]["items"] = [_stored_token(slot) for slot in slots]
    elif type(slots) is str:
        result["slots"]["literal"] = _literal_text(slots, 256)
    annotations = _binding(namespace, "__annotations__")
    result["annotations"] = _token(annotations)
    if type(annotations) is dict:
        if len(annotations) > 64 or any(
            type(key) is not str or len(key) > 256 for key in annotations
        ):
            raise ValueError("stored annotations exceed finite literal key schema")
        result["annotations"]["items"] = {
            key: _stored_token(value) for key, value in annotations.items()
        }
        pins.extend(annotations)
        pins.extend(annotations.values())
    result["fields_is_axisdata_fields"] = (
        fields is not _MISSING and _binding(namespace, "__dataclass_fields__") is fields
    )
    result["params_is_axisdata_params"] = (
        params is not _MISSING and _binding(namespace, "__dataclass_params__") is params
    )
    return result


def _carrier_metadata(cls: Any, objects: list[Any], pins: list[Any]) -> dict[str, Any]:
    result = _class_metadata(cls, pins)
    result["instances"] = []
    if type(cls) is not type:
        return result
    namespace = vars(type)["__dict__"].__get__(cls)
    slots = _binding(namespace, "__slots__")
    if not (
        type(slots) is tuple
        and len(slots) <= 32
        and all(type(slot) is str and len(slot) <= 128 for slot in slots)
    ):
        result["unresolved"] = "carrier slots need exact bounded string tuple"
        return result
    descriptors = {}
    for slot in slots:
        descriptor = _binding(namespace, slot)
        item = {"descriptor": _token(descriptor)}
        result.setdefault("descriptors", {})[slot] = item
        if type(descriptor) is types.MemberDescriptorType:
            owner = vars(types.MemberDescriptorType)["__objclass__"].__get__(descriptor)
            name = vars(types.MemberDescriptorType)["__name__"].__get__(descriptor)
            item["owner"] = _token(owner)
            item["owner_is_carrier"] = owner is cls
            item["name"] = _stored_token(name)
            item["name_is_declared_slot"] = type(name) is str and name == slot
            if owner is cls and item["name_is_declared_slot"]:
                descriptors[slot] = descriptor
        pins.append(descriptor)
    for value in objects:
        item = {"value": _token(value), "is_exact_carrier": type(value) is cls}
        result["instances"].append(item)
        pins.append(value)
        if type(value) is cls:
            item["slots"] = {}
            for slot, descriptor in descriptors.items():
                stored = descriptor.__get__(value, cls)
                item["slots"][slot] = _stored_token(stored)
                pins.append(stored)
    return result


def _metadata(axis: Any, dc_namespace: dict[str, Any]) -> tuple[Any, list[Any]]:
    pins: list[Any] = []
    if type(axis) is not type:
        return {"unresolved": "AxisData is not an exact builtin class"}, pins
    namespace = vars(type)["__dict__"].__get__(axis)
    fields = _binding(namespace, "__dataclass_fields__")
    params = _binding(namespace, "__dataclass_params__")
    objects: list[Any] = []
    result = {"axisdata": _class_metadata(axis, pins, fields, params)}
    result["field_mapping"] = _token(fields)
    if type(fields) is dict:
        if len(fields) > 32 or any(
            type(key) is not str or len(key) > 256 for key in fields
        ):
            raise ValueError("field mapping exceeds finite literal key schema")
        result["field_mapping"]["items"] = {
            key: _token(value) for key, value in fields.items()
        }
        pins.extend(fields)
        objects.extend(fields.values())
    result["carriers"] = {
        name: _carrier_metadata(_binding(dc_namespace, name), values, pins)
        for name, values in (("Field", objects), ("_DataclassParams", [params]))
    }
    result["markers"] = {
        name: _token(_binding(dc_namespace, name))
        for name in ("MISSING", "_FIELD", "_FIELD_CLASSVAR", "_FIELD_INITVAR")
    }
    pins.extend(_binding(dc_namespace, name) for name in result["markers"])
    result["selected_stdlib_helpers"] = {}
    for name in _SOURCE_FUNCTIONS:
        function = _binding(dc_namespace, name)
        result["selected_stdlib_helpers"][name] = _token(function)
        pins.append(function)
        if type(function) is types.FunctionType:
            pins.extend(
                vars(types.FunctionType)[field].__get__(function)
                for field in ("__code__", "__globals__", "__closure__")
            )
    return result, pins


def _snapshot(
    batching: Any, dc_namespace: dict[str, Any]
) -> tuple[dict[str, Any], Any]:
    axis = _binding(_module_namespace(batching), "AxisData")
    frozen_error = _binding(dc_namespace, "FrozenInstanceError")
    document: dict[str, Any] = {
        "axisdata": _token(axis),
        "frozen_instance_error": _token(frozen_error),
        "methods": {},
    }
    retained: dict[str, Any] = {"axis": axis, "frozen": frozen_error, "methods": {}}
    document["metadata"], retained["metadata"] = _metadata(axis, dc_namespace)
    if type(axis) is not type:
        document["unresolved"] = "AxisData is not an exact builtin class"
        return document, retained
    namespace = vars(type)["__dict__"].__get__(axis)
    for name in _METHODS:
        function = _binding(namespace, name)
        item: dict[str, Any] = {"function": _token(function)}
        document["methods"][name] = item
        record: dict[str, Any] = {"function": function, "cells": []}
        retained["methods"][name] = record
        if type(function) is not types.FunctionType:
            item["unresolved"] = "non-function"
            continue
        code = vars(types.FunctionType)["__code__"].__get__(function)
        closure = vars(types.FunctionType)["__closure__"].__get__(function)
        globals_ = vars(types.FunctionType)["__globals__"].__get__(function)
        record.update(code=code, closure=closure, globals=globals_)
        item.update(
            code=_token(code),
            globals_id=id(globals_),
            globals_is_dataclasses=globals_ is dc_namespace,
            globals_is_batching=globals_ is _module_namespace(batching),
        )
        freevars = code.co_freevars
        if (
            type(freevars) is tuple
            and len(freevars) <= 32
            and all(type(symbol) is str and len(symbol) <= 128 for symbol in freevars)
        ):
            item["co_freevars"] = list(freevars)
        else:
            item["unresolved_freevars"] = (
                "non-literal, over 32 names or over 128 characters"
            )
        if closure is None:
            cells = ()
        elif type(closure) is tuple:
            cells = closure
        else:
            item["unresolved_closure"] = "non-tuple closure"
            continue
        item["closure_length"] = len(cells)
        item["cells_truncated"] = len(cells) > 32
        item["cells"] = []
        for index, cell in enumerate(cells[:32]):
            cell_record = {"index": index, "cell": _token(cell)}
            item["cells"].append(cell_record)
            if type(cell) is not types.CellType:
                cell_record["unresolved"] = "non-cell"
                record["cells"].append((cell, False, _MISSING))
                continue
            try:
                value = vars(types.CellType)["cell_contents"].__get__(cell)
            except ValueError:
                cell_record["empty"] = True
                record["cells"].append((cell, False, _MISSING))
                continue
            cell_record.update(
                empty=False,
                value=_token(value),
                is_exported_axisdata=value is axis,
                is_frozen_instance_error=value is frozen_error,
            )
            record["cells"].append((cell, True, value))
    relations = []
    left = retained["methods"].get("__setattr__", {}).get("cells", [])
    right = retained["methods"].get("__delattr__", {}).get("cells", [])
    for i, (left_cell, left_present, left_value) in enumerate(left):
        for j, (right_cell, right_present, right_value) in enumerate(right):
            relations.append(
                {
                    "setter_index": i,
                    "deleter_index": j,
                    "same_cell": left_cell is right_cell,
                    "both_initialized": left_present and right_present,
                    "same_value": (
                        left_present and right_present and left_value is right_value
                    ),
                }
            )
    document["setter_deleter_relations"] = relations
    return document, retained


def _source_evidence(dc_namespace: dict[str, Any]) -> dict[str, Any]:
    path = _binding(dc_namespace, "__file__")
    if type(path) is not str or len(path) > 4096 or not path.endswith(".py"):
        return {"unresolved": "dataclasses has no bounded literal Python source path"}
    with Path(path).open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        return {"path": path, "unresolved": "dataclasses source exceeds 1 MiB"}
    text = raw.decode("utf-8")
    tree = ast.parse(text, filename=path)
    definitions = {
        node.name: node for node in tree.body if type(node) is ast.FunctionDef
    }
    result: dict[str, Any] = {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "functions": {},
    }
    for name in _SOURCE_FUNCTIONS:
        function = _binding(dc_namespace, name)
        item: dict[str, Any] = {"binding": _token(function)}
        result["functions"][name] = item
        if function is _MISSING:
            item["absent"] = True
            continue
        if type(function) is not types.FunctionType:
            item["unresolved"] = "non-function binding"
            continue
        code = vars(types.FunctionType)["__code__"].__get__(function)
        globals_ = vars(types.FunctionType)["__globals__"].__get__(function)
        filename = code.co_filename
        item["globals_is_dataclasses"] = globals_ is dc_namespace
        item["co_filename"] = _literal_text(filename, 4096)
        item["filename_matches_module_source"] = (
            type(filename) is str and filename == path
        )
        node = definitions.get(name)
        if node is None:
            item["unresolved"] = "selected top-level source definition absent"
            continue
        first_line = min([node.lineno, *(part.lineno for part in node.decorator_list)])
        item["first_line_matches_source"] = code.co_firstlineno == first_line
        if not (
            item["globals_is_dataclasses"]
            and item["filename_matches_module_source"]
            and item["first_line_matches_source"]
        ):
            item["unresolved"] = "selected function/source ownership does not match"
            continue
        segment = "".join(
            text.splitlines(keepends=True)[first_line - 1 : node.end_lineno]
        )
        encoded = segment.encode("utf-8")
        item.update(
            source_sha256=hashlib.sha256(encoded).hexdigest(),
            source_bytes=len(encoded),
            source_first_line=first_line,
            source_last_line=node.end_lineno,
            source_truncated=len(encoded) > 16 * 1024,
            code_constant_types=_code_constants(code, [1024]),
        )
        if len(encoded) <= 16 * 1024:
            item["source"] = segment
    return result


def _unchanged(before: Any, after: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "axisdata": before["axis"] is after["axis"],
        "frozen_instance_error": before["frozen"] is after["frozen"],
        "metadata": len(before["metadata"]) == len(after["metadata"])
        and all(a is z for a, z in zip(before["metadata"], after["metadata"])),
        "methods": {},
    }
    for name in _METHODS:
        left, right = before["methods"].get(name), after["methods"].get(name)
        if left is None or right is None:
            result["methods"][name] = {"both_absent": left is right}
            continue
        state = {
            field: left.get(field, _MISSING) is right.get(field, _MISSING)
            for field in ("function", "code", "closure", "globals")
        }
        left_cells, right_cells = left["cells"], right["cells"]
        state["cells"] = len(left_cells) == len(right_cells) and all(
            a[0] is z[0] and a[1] is z[1] and a[2] is z[2]
            for a, z in zip(left_cells, right_cells)
        )
        result["methods"][name] = state
    return result


def main() -> int:
    report: dict[str, Any] = {
        "purpose": "finite observation only; no native schema is accepted",
        "python": _literal_text(sys.version, 4096),
    }
    exit_code = 0
    try:
        from jax._src.interpreters import batching

        report["versions"] = {
            name: _literal_text(version(name), 256) for name in ("jax", "jaxlib")
        }
        if type(sys.modules) is not dict:
            raise TypeError("observation requires exact sys.modules dictionary")
        dc_namespace = _module_namespace(dataclasses)
        report["canonical_modules"] = {
            "dataclasses": _binding(sys.modules, "dataclasses") is dataclasses,
            "batching": _binding(sys.modules, "jax._src.interpreters.batching")
            is batching,
        }
        report["before"], before = _snapshot(batching, dc_namespace)
        report["dataclasses_source"] = _source_evidence(dc_namespace)
        report["after"], after = _snapshot(batching, dc_namespace)
        identities = _unchanged(before, after)
        report["selected_identities_unchanged"] = identities
        contradictions = [
            "canonical_module:" + name
            for name, intact in report["canonical_modules"].items()
            if intact is False
        ]
        for name in ("axisdata", "frozen_instance_error", "metadata"):
            if identities[name] is False:
                contradictions.append("selected_identity:" + name)
        for name, state in identities["methods"].items():
            for field, intact in state.items():
                if intact is False:
                    contradictions.append("selected_method:" + name + ":" + field)
        report["contradictions"] = contradictions
        report["operational_status"] = (
            "observation-contradiction" if contradictions else "observed"
        )
        if contradictions:
            exit_code = 1
    except Exception as error:
        report["operational_status"] = "observation-error"
        report["error"] = _token(error)
        exit_code = 1
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if len(serialized.encode("utf-8")) > 512 * 1024:
        serialized = json.dumps(
            {
                "operational_status": "observation-error",
                "error": "report exceeds 512 KiB",
            }
        )
        exit_code = 1
    print("NATIVE_AXISDATA_OBSERVATION_BEGIN")
    print(serialized)
    print("NATIVE_AXISDATA_OBSERVATION_END")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
