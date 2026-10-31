# scripts/_source_cast_postmortem.py

"""Copy bounded evidence from one completed, unprofiled failed source call.

No source, JAX input, or runtime is evaluated here. Traceback/native owners stay
within the callback; its result contains only primitive JSON evidence.
"""

from __future__ import annotations

import hashlib
import json
import marshal
from pathlib import Path
import sys
import threading
import types
from typing import Any


SCHEMA = "source-cast-off-postmortem-v1"
NODE_ID = (
    "tests/extra_tests/converter/test_source_conversion_semantics.py::"
    "test_floating_conversion_rounding_and_strict_special_value_gates"
    "[ties-float16-float64]"
)
_ROOT = Path(__file__).resolve().parents[1]
_TEST = "tests/extra_tests/converter/test_source_conversion_semantics.py"
_ACCEPT = "tests/extra_tests/converter/test_source_pointwise_mul_exp.py"
_SOURCE_FILES = (
    "scripts/_source_cast_postmortem.py",
    "scripts/run_pytest_batches.py",
    "scripts/_source_cast_observer.py",
    _TEST,
    _ACCEPT,
    "jax2onnx/precision.py",
    "tests/extra_tests/converter/test_source_predicate_families.py",
    "jax2onnx/_compat/jax.py",
    "jax2onnx/plugins/jax/lax/convert_element_type.py",
)
_INPUT_HEX = (
    "000000000002f03fffffffffff01f03f010000000002f03f"
    "000000000002f0bf0000000000fcef400000000000feef40"
)
_HIGH = "003c003c013c00bcff7b007c"
_LOW = "003c003c003c00bcff7b007c"
_CRITERIA = {"mode": "bitwise", "rtol": None, "atol": None, "equal_nan": True}
_REPORT_CLASSES = (
    "SampleAcceptanceError",
    "SourceValidationReport",
    "SourceSampleEvidence",
    "ArrayEvidence",
    "SampleCriteria",
    "PrecisionCheck",
    "CpuSampleTarget",
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _literal(value: Any, depth: int = 0, budget: list[int] | None = None) -> Any:
    if budget is None:
        budget = [4096]
    budget[0] -= 1
    _require(depth <= 8 and budget[0] >= 0, "postmortem literal bound exceeded")
    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is str:
        _require(len(value) <= 8192, "postmortem text bound exceeded")
        return value
    if type(value) is tuple or type(value) is list:
        _require(len(value) <= 256, "postmortem sequence bound exceeded")
        return [_literal(item, depth + 1, budget) for item in value]
    if type(value) is dict:
        _require(
            len(value) <= 256
            and all(type(key) is str and len(key) <= 256 for key in value),
            "postmortem mapping bound exceeded",
        )
        return {key: _literal(item, depth + 1, budget) for key, item in value.items()}
    raise ValueError("postmortem requires literal metadata")


def _data(value: Any, cls: Any) -> dict[str, Any]:
    _require(type(value) is cls, "postmortem native data class differs")
    result = object.__getattribute__(value, "__dict__")
    _require(type(result) is dict, "postmortem native data storage differs")
    return result


def _digest(value: Any) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "postmortem SHA256 differs",
    )
    return value


def _source_files() -> dict[str, Any]:
    result = {}
    for relative in _SOURCE_FILES:
        with (_ROOT / relative).open("rb") as stream:
            payload = stream.read(2 * 1024 * 1024 + 1)
        _require(len(payload) <= 2 * 1024 * 1024, "postmortem source exceeds bound")
        result[relative] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return result


def _module(name: str) -> types.ModuleType:
    value = sys.modules.get(name)
    _require(
        type(value) is types.ModuleType, "postmortem module was not already loaded"
    )
    return value


def _method(cls: Any, name: str) -> Any:
    for base in type.__getattribute__(cls, "__mro__"):
        value = vars(base).get(name)
        if value is not None:
            _require(
                type(value) is types.MethodDescriptorType,
                "postmortem requires native copy methods",
            )
            return value
    raise ValueError("postmortem native copy method missing")


def _typed(dtype: str, dtype_str: str, raw: bytes) -> str:
    header = {"dtype": dtype, "dtype_str": dtype_str, "shape": [6]}
    return hashlib.sha256(
        json.dumps(header, sort_keys=True).encode() + b"\0" + raw
    ).hexdigest()


def compare_contexts(before: Any, after: Any) -> dict[str, Any]:
    """Validate pytest's one exact lifecycle change without editing snapshots."""
    first, last = _literal(before), _literal(after)
    _require(
        type(first) is dict and type(last) is dict and set(first) == set(last),
        "postmortem context fields differ",
    )
    initial, final = first["environment"], last["environment"]
    _require(
        type(initial) is dict and type(final) is dict and set(initial) == set(final),
        "postmortem environment name set differs",
    )
    name = "PYTEST_CURRENT_TEST"
    _require(
        initial.get(name) == NODE_ID + " (setup)"
        and final.get(name) == NODE_ID + " (call)",
        "postmortem pytest lifecycle differs",
    )
    _require(
        all(initial[key] == final[key] for key in initial if key != name)
        and all(first[key] == last[key] for key in first if key != "environment"),
        "postmortem context changed outside exact pytest lifecycle",
    )
    return {
        "name": name,
        "before": initial[name],
        "after": final[name],
        "other_environment_and_context_held": True,
    }


class _Pin:
    def __init__(self, item: Any, context_before: dict[str, Any]) -> None:
        self.nodeid = NODE_ID
        self.context_before = _literal(context_before)
        _require(
            self.context_before["status"] == "OBSERVED"
            and self.context_before["errors"] == []
            and self.context_before["sys_profile_is_none"] is True
            and self.context_before["threading_profile_is_none"] is True,
            "postmortem target context is unverified",
        )
        self.bindings: list[tuple[dict[str, Any], str, Any]] = []
        self.functions = {}
        self.codes = {}
        self.classes = {}
        self.provenance = {}
        _require(
            sys.getprofile() is None and threading.getprofile() is None,
            "postmortem requires absent profiles",
        )
        _require(
            type(item) is vars(_module("pytest"))["Function"],
            "postmortem requires the native pytest function item",
        )
        _require(item.nodeid == NODE_ID, "postmortem target differs")
        owner = item.module
        _require(type(owner) is types.ModuleType, "postmortem owner module differs")
        _require(
            Path(vars(owner)["__file__"]).resolve() == (_ROOT / _TEST).resolve(),
            "postmortem owner source differs",
        )
        helper = _module("tests.extra_tests.converter.test_source_pointwise_mul_exp")
        precision, numpy, onnx = (
            _module(name) for name in ("jax2onnx.precision", "numpy", "onnx")
        )
        for role, function, namespace, key, relative in (
            (
                "test",
                item.function,
                vars(owner),
                NODE_ID.split("::")[1].split("[")[0],
                _TEST,
            ),
            ("accept", vars(owner).get("_accept"), vars(helper), "_accept", _ACCEPT),
            (
                "artifact",
                vars(owner).get("_artifact"),
                vars(helper),
                "_artifact",
                _ACCEPT,
            ),
            (
                "reference",
                vars(helper).get("_reference"),
                vars(helper),
                "_reference",
                _ACCEPT,
            ),
        ):
            _require(
                type(function) is types.FunctionType and namespace.get(key) is function,
                "postmortem original function binding differs",
            )
            _require(
                function.__globals__ is namespace
                and Path(function.__code__.co_filename).resolve()
                == (_ROOT / relative).resolve(),
                "postmortem original code source differs",
            )
            payload = marshal.dumps(function.__code__)
            _require(len(payload) <= 2 * 1024 * 1024, "postmortem code exceeds bound")
            self.functions[role], self.codes[role] = function, function.__code__
            self.bindings.append((namespace, key, function))
            self.provenance[role] = {
                "function_identity": id(function),
                "code_identity": id(function.__code__),
                "filename": function.__code__.co_filename,
                "name": function.__code__.co_name,
                "firstlineno": function.__code__.co_firstlineno,
                "marshal_sha256": hashlib.sha256(payload).hexdigest(),
            }
        self.bindings.append((vars(owner), "_accept", self.functions["accept"]))
        self.bindings.append((vars(owner), "_artifact", self.functions["artifact"]))
        for name in _REPORT_CLASSES:
            self.classes[name] = vars(precision)[name]
            self.bindings.append((vars(precision), name, self.classes[name]))
        validator = vars(precision)["validate_source_samples"]
        _require(
            vars(helper).get("validate_source_samples") is validator,
            "postmortem validator alias differs",
        )
        for role, function, namespace, key in (
            ("validate", validator, vars(precision), "validate_source_samples"),
            (
                "runtime",
                vars(precision)["_run_cpu_samples"],
                vars(precision),
                "_run_cpu_samples",
            ),
            ("compare", vars(precision)["_compare"], vars(precision), "_compare"),
            (
                "require",
                vars(self.classes["SourceValidationReport"])[
                    "require_sample_acceptance"
                ],
                vars(self.classes["SourceValidationReport"]),
                "require_sample_acceptance",
            ),
        ):
            _require(
                type(function) is types.FunctionType
                and function.__globals__ is vars(precision)
                and Path(function.__code__.co_filename).resolve()
                == (_ROOT / "jax2onnx/precision.py").resolve(),
                "postmortem validation function binding differs",
            )
            payload = marshal.dumps(function.__code__)
            _require(len(payload) <= 2 * 1024 * 1024, "postmortem code exceeds bound")
            self.functions[role], self.codes[role] = function, function.__code__
            self.bindings.append((namespace, key, function))
            self.provenance[role] = {
                "function_identity": id(function),
                "code_identity": id(function.__code__),
                "filename": function.__code__.co_filename,
                "name": function.__code__.co_name,
                "firstlineno": function.__code__.co_firstlineno,
                "marshal_sha256": hashlib.sha256(payload).hexdigest(),
            }
        self.bindings.append((vars(helper), "validate_source_samples", validator))
        self.statuses = {
            name: vars(vars(precision)["CheckStatus"])[name]
            for name in ("VALIDATED", "FAILED", "UNVERIFIED")
        }
        self.bindings.append(
            (vars(precision), "CheckStatus", vars(precision)["CheckStatus"])
        )
        self.array_type, self.model_type = (
            vars(numpy)["ndarray"],
            vars(onnx)["ModelProto"],
        )
        self.bindings.extend(
            (
                (vars(numpy), "ndarray", self.array_type),
                (vars(onnx), "ModelProto", self.model_type),
            )
        )
        self.array_bytes = _method(self.array_type, "tobytes")
        self.model_size, self.model_bytes = _method(
            self.model_type, "ByteSize"
        ), _method(self.model_type, "SerializeToString")
        self.call_type = vars(_module("_pytest.runner"))["CallInfo"]
        self.excinfo_type = vars(_module("_pytest._code.code"))["ExceptionInfo"]
        self.report_type = vars(_module("_pytest.reports"))["TestReport"]
        self.modules = {
            name: _module(name)
            for name in (
                vars(owner)["__name__"],
                "tests.extra_tests.converter.test_source_pointwise_mul_exp",
                "jax2onnx.precision",
                "numpy",
                "onnx",
                "_pytest.runner",
                "_pytest._code.code",
                "_pytest.reports",
            )
        }
        self.before_files = _source_files()

    def array(self, value: Any) -> dict[str, Any]:
        _require(
            type(value) is self.array_type,
            "postmortem reference is not an exact NumPy ndarray",
        )
        _require(
            value.dtype.str == "<f2"
            and type(value.shape) is tuple
            and value.shape == (6,)
            and value.nbytes == 12,
            "postmortem reference typed shape differs",
        )
        raw = self.array_bytes(value, order="C")
        _require(
            type(raw) is bytes and len(raw) == 12,
            "postmortem reference copy exceeds bound",
        )
        return {
            "dtype": "float16",
            "dtype_str": "<f2",
            "shape": [6],
            "storage_hex": raw.hex(),
            "element_storage_hex": [raw[i : i + 2].hex() for i in range(0, 12, 2)],
            "raw_storage_sha256": hashlib.sha256(raw).hexdigest(),
            "typed_sha256": _typed("float16", "<f2", raw),
        }

    def checks(self, checks: Any) -> list[dict[str, Any]]:
        _require(
            type(checks) is tuple and len(checks) <= 16,
            "postmortem check count differs",
        )
        result = []
        for check in checks:
            data = _data(check, self.classes["PrecisionCheck"])
            status = next(
                (
                    name.lower()
                    for name, member in self.statuses.items()
                    if data["status"] is member
                ),
                None,
            )
            _require(status is not None, "postmortem check status differs")
            result.append(
                {
                    "category": _literal(data["category"]),
                    "detail": _literal(data["detail"]),
                    "status": status,
                }
            )
        return result

    def evidence(self, values: Any, dtype: str) -> dict[str, Any]:
        _require(
            type(values) is tuple and len(values) == 1,
            "postmortem typed array evidence count differs",
        )
        data = _data(values[0], self.classes["ArrayEvidence"])
        _require(
            type(data["dtype"]) is str
            and data["dtype"] == dtype
            and type(data["shape"]) is tuple
            and len(data["shape"]) == 1
            and type(data["shape"][0]) is int
            and data["shape"][0] == 6,
            "postmortem typed array evidence differs",
        )
        return {
            "name": _literal(data["name"]),
            "dtype": dtype,
            "shape": [6],
            "sha256": _digest(data["sha256"]),
        }


def pin_postmortem(item: Any, context_before: dict[str, Any]) -> _Pin:
    return _Pin(item, context_before)


def capture_postmortem(
    pin: _Pin, call: Any, report: Any, context_after: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "UNVERIFIED",
        "nodeid": NODE_ID,
        "scope": "Same supplemental OFF failed call, observed after its original report.",
        "source_called": False,
        "native_references_recomputed_by_postmortem": False,
        "native_source_recomputed_here": False,
        "runtime_called": False,
        "profile_installed": False,
        "raw_target_outputs_captured": False,
        "errors": [],
        "fences": {},
    }
    try:
        _require(type(pin) is _Pin, "postmortem pin state differs")
        result.update(
            code_provenance=pin.provenance, source_files_before=pin.before_files
        )
        result["context_before"] = pin.context_before
        result["context_after"] = _literal(context_after)
        fields = _data(report, pin.report_type)
        _require(
            fields["outcome"] == "failed"
            and fields["when"] == "call"
            and fields["nodeid"] == NODE_ID,
            "postmortem original report is not the failed target call",
        )
        call_fields = _data(call, pin.call_type)
        _require(call_fields["when"] == "call", "postmortem call phase differs")
        excinfo = call_fields["excinfo"]
        triple = _data(excinfo, pin.excinfo_type)["_excinfo"]
        _require(
            type(triple) is tuple
            and len(triple) == 3
            and triple[0] is pin.classes["SampleAcceptanceError"]
            and type(triple[1]) is pin.classes["SampleAcceptanceError"],
            "postmortem exception is not exact SampleAcceptanceError",
        )
        exception = triple[1]
        args = BaseException.__dict__["args"].__get__(exception, type(exception))
        _require(
            type(args) is tuple
            and len(args) == 1
            and type(args[0]) is str
            and len(args[0]) <= 4096
            and "values fail bitwise comparison" in args[0],
            "postmortem exception detail differs",
        )
        result["exception"] = {
            "exact_type": "jax2onnx.precision.SampleAcceptanceError",
            "message": args[0],
        }
        traceback = BaseException.__dict__["__traceback__"].__get__(
            exception, type(exception)
        )
        _require(
            triple[2] is traceback, "postmortem stored and actual traceback differ"
        )
        matches: dict[str, list[Any]] = {"test": [], "accept": []}
        for _ in range(64):
            if traceback is None:
                break
            _require(
                type(traceback) is types.TracebackType
                and type(traceback.tb_frame) is types.FrameType,
                "postmortem traceback type differs",
            )
            for role in matches:
                if traceback.tb_frame.f_code is pin.codes[role]:
                    matches[role].append(traceback.tb_frame)
            traceback = traceback.tb_next
        _require(
            traceback is None and all(len(frames) == 1 for frames in matches.values()),
            "postmortem frame identity is missing, repeated or exceeds bound",
        )
        test, accept = matches["test"][0], matches["accept"][0]
        proto, references = test.f_locals["proto"], test.f_locals["references"]
        _require(
            accept.f_locals["proto"] is proto
            and accept.f_locals["references"] is references,
            "postmortem same-call artifact/reference bindings differ",
        )
        _require(
            type(proto) is pin.model_type
            and type(references) is list
            and len(references) == 1
            and type(references[0]) is tuple
            and len(references[0]) == 2
            and all(
                type(values) is list and len(values) == 1 for values in references[0]
            ),
            "postmortem reference containers differ",
        )
        eager, compiled = (pin.array(values[0]) for values in references[0])
        result.update(
            native_eager=eager,
            native_fresh_jit=compiled,
            native_references_equal=eager["storage_hex"] == compiled["storage_hex"],
            native_references_are_high=eager["storage_hex"]
            == compiled["storage_hex"]
            == _HIGH,
        )
        size = pin.model_size(proto)
        _require(
            type(size) is int and 0 <= size <= 1024 * 1024,
            "postmortem model exceeds serialization bound",
        )
        payload = pin.model_bytes(proto, deterministic=True)
        _require(
            type(payload) is bytes and len(payload) == size,
            "postmortem model serialization differs",
        )
        digest = hashlib.sha256(payload).hexdigest()
        result["artifact"] = {
            "sha256": digest,
            "bytes": size,
            "existing_object_identity": id(proto),
        }
        optimization, reference_index = (
            accept.f_locals["optimization"],
            accept.f_locals["reference_index"],
        )
        _require(
            type(optimization) is str
            and optimization in ("disabled", "all")
            and type(reference_index) is int
            and reference_index in (0, 1),
            "postmortem selected comparison differs",
        )
        result["selection"] = {
            "optimization": optimization,
            "reference_index": reference_index,
        }
        validation = _data(
            accept.f_locals["result"], pin.classes["SourceValidationReport"]
        )
        _require(
            validation["_artifact"] is proto
            and _digest(validation["artifact_sha256"]) == digest
            and type(validation["samples"]) is tuple
            and len(validation["samples"]) == 1,
            "postmortem validation artifact/samples differ",
        )
        sample = _data(validation["samples"][0], pin.classes["SourceSampleEvidence"])
        _require(
            type(sample["index"]) is int and sample["index"] == 0,
            "postmortem sample index differs",
        )
        criteria = _literal(_data(sample["criteria"], pin.classes["SampleCriteria"]))
        _require(criteria == _CRITERIA, "postmortem criterion differs")
        inputs, outputs, native = (
            pin.evidence(sample["inputs"], "float64"),
            pin.evidence(sample["outputs"], "float16"),
            pin.evidence(sample["references"], "float16"),
        )
        _require(
            inputs["sha256"] == _typed("float64", "<f8", bytes.fromhex(_INPUT_HEX))
            and native["sha256"] == (eager, compiled)[reference_index]["typed_sha256"],
            "postmortem fixed-input/selected-reference hash binding differs",
        )
        checks, sample_checks = pin.checks(validation["checks"]), pin.checks(
            sample["checks"]
        )
        _require(
            [(check["category"], check["status"]) for check in checks]
            == [("artifact-load", "validated"), ("onnx-schema", "validated")]
            and [(check["category"], check["status"]) for check in sample_checks]
            == [("runtime-execution", "validated"), ("numerical-acceptance", "failed")]
            and sample_checks[1]["detail"] == "Output 0: values fail bitwise comparison"
            and args[0] == "failed: " + sample_checks[1]["detail"],
            "postmortem exact runtime/numerical evidence differs",
        )
        target = _literal(_data(validation["target"], pin.classes["CpuSampleTarget"]))
        _require(
            type(target["runtime_version"]) is str
            and 0 < len(target["runtime_version"]) <= 128
            and type(target["architecture"]) is str
            and 0 < len(target["architecture"]) <= 128
            and target["optimization"]
            == ("ORT_DISABLE_ALL" if optimization == "disabled" else "ORT_ENABLE_ALL")
            and target["providers"] == ["CPUExecutionProvider"]
            and target["provider_options"] == [["CPUExecutionProvider", []]]
            and type(target["intra_op_threads"]) is int
            and target["intra_op_threads"] == 1
            and type(target["inter_op_threads"]) is int
            and target["inter_op_threads"] == 1
            and target["execution_mode"] == "ORT_SEQUENTIAL"
            and target["session_config"] == []
            and target["deployment_observation"] is None,
            "postmortem original CPU session settings differ",
        )
        result["validation"] = {
            "artifact_sha256": digest,
            "target": target,
            "checks": checks,
            "samples": [
                {
                    "index": 0,
                    "inputs": [inputs],
                    "references": [native],
                    "outputs": [outputs],
                    "criteria": criteria,
                    "checks": sample_checks,
                }
            ],
        }
        result["target_typed_hash_matches_known_low"] = outputs["sha256"] == _typed(
            "float16", "<f2", bytes.fromhex(_LOW)
        )
        result["source_files_end"] = _source_files()
        result["context_transition"] = compare_contexts(
            pin.context_before, result["context_after"]
        )
        result["fences"] = {
            "code_binding_held": all(
                sys.modules.get(name) is module for name, module in pin.modules.items()
            )
            and all(
                namespace.get(name) is value for namespace, name, value in pin.bindings
            )
            and all(
                function.__code__ is pin.codes[role]
                for role, function in pin.functions.items()
            )
            and _method(pin.model_type, "ByteSize") is pin.model_size
            and _method(pin.model_type, "SerializeToString") is pin.model_bytes
            and _method(pin.array_type, "tobytes") is pin.array_bytes,
            "source_held": result["source_files_end"] == pin.before_files,
            "profiles_absent": sys.getprofile() is None
            and threading.getprofile() is None,
            "context_held": result["context_transition"][
                "other_environment_and_context_held"
            ]
            is True,
        }
        _require(
            all(value is True for value in result["fences"].values()),
            "postmortem final binding/source/profile/context fence differs",
        )
        result["status"] = "WORDS_OBSERVED"
    except BaseException as error:
        result["errors"].append(
            {"stage": "post-report-copy", "error_type": type(error).__name__}
        )
    if len(json.dumps(result).encode()) > 256 * 1024:
        return {
            "schema": SCHEMA,
            "status": "UNVERIFIED",
            "nodeid": NODE_ID,
            "errors": [{"stage": "report-bound", "error_type": "ValueError"}],
        }
    return result
