# tests/extra_tests/framework/test_source_cast_postmortem.py

"""Exercise passive copying with synthetic native containers, not cast kernels.

The deliberately compiled fixture bodies never invoke a source producer or an
execution session. Their reports/words qualify extraction and binding only.
"""

import builtins
from copy import deepcopy
from dataclasses import replace
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import platform
import sys
import threading
import types
from typing import Any

import numpy as np
import onnx
import pytest
from _pytest.runner import CallInfo

from jax2onnx import precision


_ROOT = Path(__file__).resolve().parents[3]
_HIGH = "003c003c013c00bcff7b007c"
_LOW = "003c003c003c00bcff7b007c"


def _load(name: str) -> Any:
    path = _ROOT / "scripts" / (name + ".py")
    spec = spec_from_file_location("_postmortem_test_" + name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Unmaterialized:
    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("postmortem materialized an input")

    def __jax_array__(self) -> Any:
        raise AssertionError("postmortem materialized a JAX input")

    def __iter__(self) -> Any:
        raise AssertionError("postmortem traversed an input")


class _Foreign:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError("postmortem ran a foreign attribute protocol")


def _case(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    *,
    words: str = _HIGH,
) -> dict[str, Any]:
    module = _load("_source_cast_postmortem")
    context = {
        "status": "OBSERVED",
        "errors": [],
        "sys_profile_is_none": True,
        "threading_profile_is_none": True,
        "environment": {
            "PYTEST_CURRENT_TEST": module.NODE_ID + " (setup)",
            "JAX_ENABLE_X64": "True",
        },
    }
    context_after = deepcopy(context)
    context_after["environment"]["PYTEST_CURRENT_TEST"] = module.NODE_ID + " (call)"
    proto = onnx.ModelProto(ir_version=10)
    refs = [
        (
            [np.frombuffer(bytes.fromhex(words), dtype="<f2").copy()],
            [np.frombuffer(bytes.fromhex(words), dtype="<f2").copy()],
        )
    ]
    digest = hashlib.sha256(proto.SerializeToString(deterministic=True)).hexdigest()
    input_hash = module._typed("float64", "<f8", bytes.fromhex(module._INPUT_HEX))
    native_hash = module._typed("float16", "<f2", bytes.fromhex(words))
    target_hash = module._typed("float16", "<f2", bytes.fromhex(_LOW))
    criteria = precision.SampleCriteria("bitwise")
    checks = (
        precision.PrecisionCheck(
            "artifact-load", precision.CheckStatus.VALIDATED, "loaded"
        ),
        precision.PrecisionCheck(
            "onnx-schema", precision.CheckStatus.VALIDATED, "schema"
        ),
    )
    sample_checks = (
        precision.PrecisionCheck(
            "runtime-execution", precision.CheckStatus.VALIDATED, "executed"
        ),
        precision.PrecisionCheck(
            "numerical-acceptance",
            precision.CheckStatus.FAILED,
            "Output 0: values fail bitwise comparison",
        ),
    )
    sample = precision.SourceSampleEvidence(
        0,
        (precision.ArrayEvidence("0", "float64", (6,), input_hash),),
        (precision.ArrayEvidence("0", "float16", (6,), native_hash),),
        (precision.ArrayEvidence("0", "float16", (6,), target_hash),),
        criteria,
        sample_checks,
    )
    target = precision.CpuSampleTarget(
        "synthetic-runtime",
        "ORT_DISABLE_ALL",
        platform.machine(),
        ("CPUExecutionProvider",),
        (("CPUExecutionProvider", ()),),
    )
    validation = precision.SourceValidationReport(
        digest, target, checks, (sample,), proto
    )
    helper_name = "tests.extra_tests.converter.test_source_pointwise_mul_exp"
    owner_name = "tests.extra_tests.converter.test_source_conversion_semantics"
    helper, owner = types.ModuleType(helper_name), types.ModuleType(owner_name)
    helper.__file__, owner.__file__ = str(_ROOT / module._ACCEPT), str(
        _ROOT / module._TEST
    )
    counts = {"original_accept": 0, "source": 0, "runtime": 0}
    helper.__dict__.update(
        prepared_report=validation,
        counts=counts,
        validate_source_samples=precision.validate_source_samples,
    )
    exec(
        compile(
            """def _artifact():
    counts['source'] += 1
    raise AssertionError('producer must not run in this synthetic fixture')
def _reference():
    counts['source'] += 1
    raise AssertionError('reference must not run in this synthetic fixture')
def _accept(proto, feeds, references, criteria):
    optimization = 'disabled'
    reference_index = 0
    result = prepared_report
    counts['original_accept'] += 1
    result.require_sample_acceptance()
""",
            helper.__file__,
            "exec",
        ),
        helper.__dict__,
    )
    owner.__dict__.update(
        _accept=helper._accept,
        _artifact=helper._artifact,
        fixture_proto=proto,
        fixture_references=refs,
        criteria=criteria,
        untouched=_Unmaterialized(),
    )
    exec(
        compile(
            """def test_floating_conversion_rounding_and_strict_special_value_gates():
    proto = fixture_proto
    references = fixture_references
    value = untouched
    feeds = untouched
    _accept(proto, feeds, references, criteria)
""",
            owner.__file__,
            "exec",
        ),
        owner.__dict__,
    )
    monkeypatch.setitem(sys.modules, helper_name, helper)
    monkeypatch.setitem(sys.modules, owner_name, owner)
    parent = pytest.Module.from_parent(request.session, path=_ROOT / module._TEST)
    parent._obj = owner
    item = pytest.Function.from_parent(
        parent,
        name=module.NODE_ID.split("::")[1],
        callobj=owner.test_floating_conversion_rounding_and_strict_special_value_gates,
    )
    assert item.nodeid == module.NODE_ID
    return {
        "module": module,
        "context": context,
        "context_after": context_after,
        "proto": proto,
        "references": refs,
        "validation": validation,
        "helper": helper,
        "owner": owner,
        "item": item,
        "counts": counts,
    }


def _failed_call(case: dict[str, Any]) -> tuple[Any, Any]:
    call = CallInfo.from_call(case["item"].function, when="call")
    assert call.excinfo is not None
    report = pytest.TestReport(
        nodeid=case["module"].NODE_ID,
        location=(case["module"]._TEST, 0, "synthetic"),
        keywords={},
        outcome="failed",
        longrepr="original report sentinel",
        when="call",
        sections=[],
        duration=0,
    )
    return call, report


def _capture(case: dict[str, Any]) -> tuple[dict[str, Any], Any, Any]:
    pin = case["module"].pin_postmortem(case["item"], case["context"])
    call, report = _failed_call(case)
    result = case["module"].capture_postmortem(pin, call, report, case["context_after"])
    return result, call, report


@pytest.mark.parametrize("words", (_HIGH, _LOW))
def test_copies_only_existing_same_call_evidence_after_original_report(
    words: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(request, monkeypatch, words=words)
    pin = case["module"].pin_postmortem(case["item"], case["context"])
    call, report = _failed_call(case)
    original_fields = dict(report.__dict__)
    exception = call.excinfo.value
    before_counts = dict(case["counts"])
    before_profiles = (sys.getprofile(), threading.getprofile())
    original_import = builtins.__import__
    import_attempts = []

    def no_framework_import(name: str, *args: Any, **kwargs: Any) -> Any:
        import_attempts.append(name)
        assert name.split(".")[0] not in {
            "jax",
            "jaxlib",
            "numpy",
            "onnx",
            "onnxruntime",
            "jax2onnx",
        }, name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_framework_import)
    result = case["module"].capture_postmortem(pin, call, report, case["context_after"])
    assert result["status"] == "WORDS_OBSERVED" and result["errors"] == [], {
        "result": result,
        "import_attempts": import_attempts,
    }
    assert (
        result["native_eager"]["storage_hex"]
        == result["native_fresh_jit"]["storage_hex"]
        == words
    )
    assert result["native_references_are_high"] is (words == _HIGH)
    assert result["target_typed_hash_matches_known_low"] is True
    assert all(result["fences"].values())
    assert result["context_before"] == case["context"]
    assert result["context_after"] == case["context_after"]
    assert result["context_transition"] == {
        "name": "PYTEST_CURRENT_TEST",
        "before": case["module"].NODE_ID + " (setup)",
        "after": case["module"].NODE_ID + " (call)",
        "other_environment_and_context_held": True,
    }
    assert (
        result["raw_target_outputs_captured"]
        is result["native_source_recomputed_here"]
        is False
    )
    assert report.__dict__ == original_fields and call.excinfo.value is exception
    assert (
        case["counts"]
        == before_counts
        == {"original_accept": 1, "source": 0, "runtime": 0}
    )
    assert (sys.getprofile(), threading.getprofile()) == before_profiles == (None, None)
    # JSON serialization proves that returned evidence retains no native owners.
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize(
    "fault",
    (
        "foreign-array",
        "foreign-report",
        "wrong-input",
        "wrong-reference",
        "wrong-model",
        "wrong-target",
        "missing-check",
        "duplicate-check",
        "conflicting-check",
        "unrelated-failure",
        "missing-runtime",
        "duplicate-numerical",
    ),
)
def test_foreign_data_hash_and_exact_check_failures_stay_unverified(
    fault: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(request, monkeypatch)
    validation = case["validation"]
    sample = validation.samples[0]
    if fault == "foreign-array":
        case["references"][0][0][0] = _Foreign()
    elif fault == "foreign-report":
        case["helper"].prepared_report = _Foreign()
        case["helper"].SampleAcceptanceError = precision.SampleAcceptanceError
        exec(
            compile(
                """def _accept(proto, feeds, references, criteria):
    optimization = 'disabled'
    reference_index = 0
    result = prepared_report
    counts['original_accept'] += 1
    raise SampleAcceptanceError('failed: Output 0: values fail bitwise comparison')
""",
                case["helper"].__file__,
                "exec",
            ),
            case["helper"].__dict__,
        )
        case["owner"]._accept = case["helper"]._accept
    elif fault == "wrong-input":
        sample = replace(sample, inputs=(replace(sample.inputs[0], sha256="0" * 64),))
    elif fault == "wrong-reference":
        sample = replace(
            sample, references=(replace(sample.references[0], sha256="0" * 64),)
        )
    elif fault == "wrong-model":
        validation = replace(validation, _artifact=onnx.ModelProto())
    elif fault == "wrong-target":
        validation = replace(
            validation, target=replace(validation.target, intra_op_threads=2)
        )
    elif fault == "missing-check":
        validation = replace(validation, checks=validation.checks[:1])
    elif fault == "duplicate-check":
        validation = replace(
            validation, checks=validation.checks + validation.checks[:1]
        )
    elif fault == "conflicting-check":
        validation = replace(
            validation,
            checks=validation.checks
            + (replace(validation.checks[0], status=precision.CheckStatus.FAILED),),
        )
    elif fault == "unrelated-failure":
        validation = replace(
            validation,
            checks=validation.checks
            + (
                precision.PrecisionCheck(
                    "unrelated", precision.CheckStatus.FAILED, "other failure"
                ),
            ),
        )
    elif fault == "missing-runtime":
        sample = replace(sample, checks=sample.checks[1:])
    else:
        sample = replace(sample, checks=sample.checks + sample.checks[1:])
    if fault != "foreign-report":
        case["helper"].prepared_report = replace(validation, samples=(sample,))
    result, _, report = _capture(case)
    assert result["status"] == "UNVERIFIED" and result["errors"]
    assert report.outcome == "failed" and report.longrepr == "original report sentinel"
    assert case["counts"]["source"] == case["counts"]["runtime"] == 0
    if fault in ("foreign-array", "foreign-report"):
        assert result["errors"][0]["error_type"] == "ValueError"


@pytest.mark.parametrize(
    "fault",
    (
        "stored-traceback",
        "wrong-phase",
        "missing-test-frame",
        "repeated-accept-frame",
        "traceback-bound",
        "producer-replaced",
        "validator-replaced",
        "runtime-replaced",
        "comparator-replaced",
        "method-replaced",
        "source-fence",
        "context-fence",
        "report-passed",
        "model-bound",
    ),
)
def test_same_call_identity_and_end_fences_are_mandatory(
    fault: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(request, monkeypatch)
    if fault in ("repeated-accept-frame", "traceback-bound"):
        depth = 1 if fault == "repeated-accept-frame" else 70
        namespace = case["helper"].__dict__
        namespace["depth"] = depth
        exec(
            compile(
                """def _accept(proto, feeds, references, criteria):
    global depth
    optimization = 'disabled'
    reference_index = 0
    result = prepared_report
    if depth:
        depth -= 1
        _accept(proto, feeds, references, criteria)
    result.require_sample_acceptance()
""",
                case["helper"].__file__,
                "exec",
            ),
            namespace,
        )
        case["owner"]._accept = namespace["_accept"]
    pin = case["module"].pin_postmortem(case["item"], case["context"])
    call, report = _failed_call(case)
    context = case["context_after"]
    if fault == "stored-traceback":
        triple = call.excinfo._excinfo
        call.excinfo._excinfo = (triple[0], triple[1], triple[2].tb_next)
    elif fault == "wrong-phase":
        call.when = "setup"
    elif fault == "missing-test-frame":
        call = CallInfo.from_call(
            lambda: case["helper"]._accept(
                case["proto"],
                _Unmaterialized(),
                case["references"],
                precision.SampleCriteria("bitwise"),
            ),
            when="call",
        )
    elif fault == "producer-replaced":
        case["owner"]._artifact = lambda: None
    elif fault == "validator-replaced":
        case["helper"].validate_source_samples = lambda: None
    elif fault in ("runtime-replaced", "comparator-replaced"):
        name = "_run_cpu_samples" if fault == "runtime-replaced" else "_compare"
        monkeypatch.setattr(precision, name, lambda *args: None)
    elif fault == "method-replaced":
        monkeypatch.setattr(
            precision.SourceValidationReport,
            "require_sample_acceptance",
            lambda self: None,
        )
    elif fault == "source-fence":
        original = case["module"]._source_files
        monkeypatch.setattr(
            case["module"],
            "_source_files",
            lambda: {**original(), "synthetic-change": {}},
        )
    elif fault == "context-fence":
        context = {**context, "status": "UNVERIFIED"}
    elif fault == "report-passed":
        report.outcome = "passed"
    elif fault == "model-bound":
        case["proto"].producer_name = "x" * (1024 * 1024)
    before = dict(report.__dict__)
    result = case["module"].capture_postmortem(pin, call, report, context)
    assert result["status"] == "UNVERIFIED" and result["errors"]
    assert report.__dict__ == before


def _word_proof(
    case: dict[str, Any], post: dict[str, Any], tmp_path: Path
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    scaffold_path = (
        _ROOT / "tests/extra_tests/framework/test_source_cast_failure_replay.py"
    )
    spec = spec_from_file_location("_postmortem_scaffold", scaffold_path)
    assert spec is not None and spec.loader is not None
    scaffold = module_from_spec(spec)
    spec.loader.exec_module(scaffold)
    helper, _, folder = scaffold._fixture(
        tmp_path, model=case["proto"].SerializeToString(deterministic=True)
    )
    observed = helper._json(folder / helper.OBSERVATION)
    for role in ("artifact", "runtime", "compare"):
        observed["function_code_provenance"][role] = post["code_provenance"][role]
    observed["distributions"] = {"onnxruntime": "synthetic-runtime"}
    (folder / helper.OBSERVATION).write_text(json.dumps(observed))
    witness = {
        "folder": folder,
        "model": folder / helper.MODEL,
        "artifact_sha256": post["artifact"]["sha256"],
        "original_source_files": observed["source_files_before"],
        "references": {
            name: helper._record(observed["artifact_events"][0][name][0], "float16")
            for name in ("native_eager", "native_fresh_jit")
        },
        "inputs": helper._record(
            observed["runtime_events"][0]["inputs"]["in_0"], "float64"
        ),
    }
    receipt = {
        "off_postmortem": post,
        "target_context_before": case["context"],
        "target_context_after": case["context_after"],
    }
    result = {
        "status": "OBSERVER_NOT_NECESSARY",
        "observer_disabled_verified": True,
        "prefix_matched": True,
        "raw_native_words_captured": False,
    }
    return (
        _load("control_source_cast_observer"),
        helper,
        {
            "witness": witness,
            "contexts": {
                "target_context_before": case["context"],
                "target_context_after": case["context_after"],
            },
        },
        {"receipt": receipt, "result": result},
    )


def test_precise_word_finding_binds_both_vectors_model_input_target_and_producer(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = _case(request, monkeypatch)
    post, _, _ = _capture(case)
    assert post["status"] == "WORDS_OBSERVED"
    controller, helper, proof, data = _word_proof(case, post, tmp_path)
    controller._word_comparison(helper, proof, data["receipt"], data["result"])
    assert (
        data["result"]["word_comparison_status"] == "OFF_WORDS_MATCH_ORIGINAL_WITNESS"
    )
    assert data["result"]["raw_native_words_captured"] is True
    assert data["result"]["word_comparison"]["raw_target_outputs_captured"] is False


@pytest.mark.parametrize(
    "fault",
    ("foreign", "oversized-text", "oversized-sequence", "excessive-depth"),
)
def test_literal_bounds_reject_foreign_protocols_without_coercion(fault: str) -> None:
    module = _load("_source_cast_postmortem")
    if fault == "foreign":
        value: Any = _Foreign()
    elif fault == "oversized-text":
        value = "x" * 8193
    elif fault == "oversized-sequence":
        value = [None] * 257
    else:
        value = {"nested": [[[[[[[[[None]]]]]]]]]}
    with pytest.raises(ValueError):
        module._literal(value)


def test_array_and_model_size_bounds_precede_native_copy(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(request, monkeypatch)
    pin = case["module"].pin_postmortem(case["item"], case["context"])
    copies = []

    def forbidden_copy(*args: Any, **kwargs: Any) -> Any:
        copies.append(True)
        raise AssertionError("oversized native value was copied")

    original = pin.array_bytes
    pin.array_bytes = forbidden_copy
    with pytest.raises(ValueError):
        pin.array(np.zeros(7, dtype="<f2"))
    assert copies == []
    pin.array_bytes = original
    call, report = _failed_call(case)
    case["proto"].producer_name = "x" * (1024 * 1024)
    pin.model_bytes = forbidden_copy
    result = case["module"].capture_postmortem(pin, call, report, case["context_after"])
    assert result["status"] == "UNVERIFIED" and copies == []


def test_source_read_has_a_finite_byte_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("_source_cast_postmortem")
    monkeypatch.setattr(module, "_ROOT", tmp_path)
    monkeypatch.setattr(module, "_SOURCE_FILES", ("oversized.py",))
    (tmp_path / "oversized.py").write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    with pytest.raises(ValueError):
        module._source_files()


@pytest.mark.parametrize(
    "fault",
    (
        "none",
        "before-node",
        "after-node",
        "before-phase",
        "after-phase",
        "other-env",
        "env-key",
        "other-context",
    ),
)
def test_only_exact_fixed_node_setup_to_call_lifecycle_is_allowed(fault: str) -> None:
    module = _load("_source_cast_postmortem")
    before = {
        "status": "OBSERVED",
        "source_files": {"source": "unchanged"},
        "environment": {
            "PYTEST_CURRENT_TEST": module.NODE_ID + " (setup)",
            "JAX_ENABLE_X64": "True",
        },
    }
    after = deepcopy(before)
    after["environment"]["PYTEST_CURRENT_TEST"] = module.NODE_ID + " (call)"
    if fault == "before-node":
        before["environment"]["PYTEST_CURRENT_TEST"] = "other-node (setup)"
    elif fault == "after-node":
        after["environment"]["PYTEST_CURRENT_TEST"] = "other-node (call)"
    elif fault == "before-phase":
        before["environment"]["PYTEST_CURRENT_TEST"] = module.NODE_ID + " (call)"
    elif fault == "after-phase":
        after["environment"]["PYTEST_CURRENT_TEST"] = module.NODE_ID + " (teardown)"
    elif fault == "other-env":
        after["environment"]["JAX_ENABLE_X64"] = "False"
    elif fault == "env-key":
        after["environment"]["PYTEST_NEW_CONTEXT"] = "new"
    elif fault == "other-context":
        after["source_files"]["source"] = "changed"
    snapshots = deepcopy((before, after))
    if fault == "none":
        result = module.compare_contexts(before, after)
        assert result["other_environment_and_context_held"] is True
    else:
        with pytest.raises(ValueError):
            module.compare_contexts(before, after)
    assert (before, after) == snapshots


def test_substituted_producer_cannot_supply_an_original_high_word_claim(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = _case(request, monkeypatch)
    original, _, _ = _capture(case)
    controller, helper, proof, data = _word_proof(case, original, tmp_path)
    namespace = case["helper"].__dict__
    namespace.update(synthetic_proto=case["proto"], synthetic_refs=case["references"])
    exec(
        compile(
            """def _artifact():
    return None, synthetic_proto, None, synthetic_refs
""",
            case["helper"].__file__,
            "exec",
        ),
        namespace,
    )
    case["owner"]._artifact = namespace["_artifact"]
    substituted, _, _ = _capture(case)
    assert substituted["status"] == "WORDS_OBSERVED"
    assert substituted["native_references_are_high"] is True
    data["receipt"]["off_postmortem"] = substituted
    controller._word_comparison(helper, proof, data["receipt"], data["result"])
    assert data["result"]["status"] == "OBSERVER_NOT_NECESSARY"
    assert data["result"]["word_comparison_status"] == "UNVERIFIED"


@pytest.mark.parametrize(
    "role,name", (("runtime", "_run_cpu_samples"), ("compare", "_compare"))
)
def test_substituted_runtime_or_comparator_cannot_supply_original_target_evidence(
    role: str,
    name: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _case(request, monkeypatch)
    original, _, _ = _capture(case)
    controller, helper, proof, data = _word_proof(case, original, tmp_path)
    code = compile(
        "def " + name + "(*args, **kwargs):\n    return None\n",
        str(_ROOT / "jax2onnx/precision.py"),
        "exec",
    ).co_consts[0]
    adapter = types.FunctionType(code, vars(precision))
    monkeypatch.setattr(precision, name, adapter)
    substituted, _, _ = _capture(case)
    assert substituted["status"] == "WORDS_OBSERVED"
    assert (
        substituted["code_provenance"][role]["marshal_sha256"]
        != original["code_provenance"][role]["marshal_sha256"]
    )
    data["receipt"]["off_postmortem"] = substituted
    controller._word_comparison(helper, proof, data["receipt"], data["result"])
    assert data["result"]["status"] == "OBSERVER_NOT_NECESSARY"
    assert data["result"]["word_comparison_status"] == "UNVERIFIED"


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "partial",
        "low-eager",
        "low-jit",
        "hash",
        "model",
        "input",
        "target",
        "selected-reference",
        "criterion",
        "duplicate-check",
        "producer",
        "source",
        "profile",
        "generic-proof",
        "captured-context",
        "recorded-transition",
        "receipt-context",
    ),
)
def test_word_mismatch_never_promotes_generic_failure_class(
    fault: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _case(request, monkeypatch)
    post, _, _ = _capture(case)
    controller, helper, proof, data = _word_proof(case, deepcopy(post), tmp_path)
    post = data["receipt"]["off_postmortem"]
    sample = post["validation"]["samples"][0]
    if fault == "missing":
        data["receipt"].clear()
    elif fault == "partial":
        post["status"] = "UNVERIFIED"
    elif fault.startswith("low-"):
        key = "native_eager" if fault == "low-eager" else "native_fresh_jit"
        post[key] = case["module"]._Pin.array(
            case["module"].pin_postmortem(case["item"], case["context"]),
            np.frombuffer(bytes.fromhex(_LOW), dtype="<f2"),
        )
    elif fault == "hash":
        post["native_eager"]["raw_storage_sha256"] = "0" * 64
    elif fault == "model":
        post["artifact"]["sha256"] = "0" * 64
    elif fault in ("input", "target", "selected-reference"):
        key = {
            "input": "inputs",
            "target": "outputs",
            "selected-reference": "references",
        }[fault]
        sample[key][0]["sha256"] = "0" * 64
    elif fault == "criterion":
        sample["criteria"]["mode"] = "exact"
    elif fault == "duplicate-check":
        post["validation"]["checks"].append(post["validation"]["checks"][0])
    elif fault == "producer":
        post["code_provenance"]["artifact"]["marshal_sha256"] = "0" * 64
    elif fault == "source":
        post["source_files_before"][case["module"]._TEST]["sha256"] = "0" * 64
    elif fault == "profile":
        post["fences"]["profiles_absent"] = False
    elif fault == "captured-context":
        post["context_after"]["environment"]["JAX_ENABLE_X64"] = "False"
    elif fault == "recorded-transition":
        post["context_transition"]["after"] = "other-node (call)"
    elif fault == "receipt-context":
        changed = deepcopy(data["receipt"]["target_context_after"])
        changed["environment"]["PYTEST_CURRENT_TEST"] = "other-node (call)"
        data["receipt"]["target_context_after"] = changed
    else:
        data["result"]["prefix_matched"] = False
    controller._word_comparison(helper, proof, data["receipt"], data["result"])
    assert data["result"]["status"] == "OBSERVER_NOT_NECESSARY"
    assert data["result"]["word_comparison_status"] == "UNVERIFIED"
    assert "word_comparison" not in data["result"]
