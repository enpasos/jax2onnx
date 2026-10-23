# tests/extra_tests/converter/test_source_callable_preparation_exports.py

"""Prepared ordinary callables use unmodified audited export and fixed CPU samples."""

from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jax
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx, user_interface as ui
from jax2onnx._source_callable_preparation import prepare_callable
from jax2onnx._source_preparation import SourcePreparationError
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.precision import (
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_callable_preparation import (
    SharedBinding,
    _conditional_root,
    _fresh,
    _global_root,
    _linear_expected,
    _linear_fixture,
    _linear_values,
    _owned_root,
    _owner_snapshot,
    _prepare_rounding,
    _prepared_global,
    _preserve_settings as _preserve_settings,
    _protected_root,
    _real_owners,
    _recipe,
    _root_recipe,
    _rounding,
    _same,
    _selected_branch,
)

_BITWISE = SampleCriteria("bitwise")


_NAMED_ROOT_IMPL = _global_root
_FIRST_VERSION_IMPL = _rounding
_SECOND_VERSION_IMPL = _rounding


@onnx_function(name="PreparedCallableRoot")
def _named_root_caller(x: Any) -> Any:
    return _NAMED_ROOT_IMPL(x)


@onnx_function(name="PreparedVersionFirst")
def _first_version_caller(x: Any) -> Any:
    return _FIRST_VERSION_IMPL(x)


@onnx_function(name="PreparedVersionSecond")
def _second_version_caller(x: Any) -> Any:
    return _SECOND_VERSION_IMPL(x)


@onnx_function(name="NestedPreparedVersionFirst")
def _first_version_nested(x: Any) -> Any:
    return _first_version_caller(x)


@onnx_function(name="NestedPreparedVersionSecond")
def _second_version_nested(x: Any) -> Any:
    return _second_version_caller(x)


def _named_root_entry(x: Any) -> Any:
    return _named_root_caller(x)


def _first_version_entry(x: Any) -> Any:
    return _first_version_nested(x)


def _second_version_entry(x: Any) -> Any:
    return _second_version_nested(x)


def _save(path: Path, values: Any) -> None:
    np.savez(path, **{f"array{i}": np.asarray(value) for i, value in enumerate(values)})


def _port(port: Any, expected: Any) -> None:
    value = np.asarray(expected)
    tensor = port.type.tensor_type
    assert tensor.elem_type == onnx.helper.np_dtype_to_tensor_dtype(value.dtype)
    assert tuple(dim.dim_value for dim in tensor.shape.dim) == value.shape


def _function_edges(model: Any, directory: Path, *, nested: bool = False) -> None:
    definitions = {(f.domain, f.name): f for f in model.functions}
    root_calls = [
        (node.domain, node.op_type)
        for node in model.graph.node
        if (node.domain, node.op_type) in definitions
    ]
    edges = [
        (key, (node.domain, node.op_type))
        for key, function in definitions.items()
        for node in function.node
        if (node.domain, node.op_type) in definitions
    ]
    (directory / "serialized-function-edges.json").write_text(
        json.dumps(
            {
                "definitions": sorted(definitions),
                "root_calls": root_calls,
                "edges": edges,
            },
            indent=2,
        )
    )
    assert root_calls, "named route must serialize a real Function call"
    if nested:
        assert any(
            caller in root_calls and caller != callee for caller, callee in edges
        ), "nested route must serialize a root-reachable Function-to-Function edge"


def _softmax_artifact_path(model: Any, directory: Path) -> None:
    # Uniform samples alone cannot distinguish the original FP32 helper. Inspect
    # the serialized data path as well, without changing its sample criterion.
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = inferred.graph
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*graph.input, *graph.value_info, *graph.output)
    }
    producers = {output: node for node in graph.node for output in node.output}
    layout_ops = {"Identity", "Reshape", "Unsqueeze", "Squeeze", "Expand"}

    def origin(name: str) -> str:
        visited = set()
        while name in producers and producers[name].op_type in layout_ops:
            assert name not in visited
            visited.add(name)
            node = producers[name]
            assert types[node.input[0]] == types[name] == onnx.TensorProto.DOUBLE
            name = node.input[0]
        return name

    (exp,) = [node for node in graph.node if node.op_type == "Exp"]
    (reduction,) = [
        node
        for node in graph.node
        if node.op_type == "ReduceSum" and origin(node.input[0]) == exp.output[0]
    ]
    (division,) = [
        node
        for node in graph.node
        if node.op_type == "Div"
        and origin(node.input[0]) == exp.output[0]
        and origin(node.input[1]) == reduction.output[0]
    ]
    typed_path = []
    for node, data_inputs in ((exp, 1), (reduction, 1), (division, 2)):
        names = [*node.input[:data_inputs], *node.output]
        assert all(types[name] == onnx.TensorProto.DOUBLE for name in names)
        typed_path.append(
            {"op_type": node.op_type, "values": {name: types[name] for name in names}}
        )
    (cast,) = [
        node
        for node in graph.node
        if node.op_type == "Cast"
        and origin(node.input[0]) == division.output[0]
        and any(
            a.name == "to" and a.i == onnx.TensorProto.FLOAT for a in node.attribute
        )
    ]
    assert types[cast.input[0]] == onnx.TensorProto.DOUBLE
    assert types[cast.output[0]] == onnx.TensorProto.FLOAT
    # Public output renaming can insert storage-preserving Identity nodes.
    output = graph.output[0].name
    seen = set()
    while output in producers and producers[output].op_type == "Identity":
        assert output not in seen and types[output] == onnx.TensorProto.FLOAT
        seen.add(output)
        output = producers[output].input[0]
    assert output == cast.output[0]
    (directory / "serialized-softmax-path.json").write_text(
        json.dumps(
            {"double_path": typed_path, "final_float_cast": list(cast.output)}, indent=2
        )
    )


def _observe(
    fn: Any, feeds: Any, intended: Any, directory: Path, records: Any = ()
) -> Any:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "criteria.json").write_text(json.dumps(asdict(_BITWISE)))
    frozen_feeds = [
        tuple(np.asarray(value).copy() for value in values) for values in feeds
    ]
    references = []
    for i, (values, wanted) in enumerate(zip(feeds, intended, strict=True)):
        _save(directory / f"inputs-{i}.npz", values)
        _save(directory / f"intended-{i}.npz", wanted)
        native = []
        native_values = tuple(jax.device_put(value) for value in values)
        for role, call in (("eager", fn), ("fresh-jit", jax.jit(_fresh(fn)))):
            outputs = tuple(
                np.asarray(value).copy()
                for value in jax.tree.leaves(call(*native_values))
            )
            _save(directory / f"{role}-{i}.npz", outputs)
            native.append(outputs)
        references.append(native)
    specs = [
        jax.ShapeDtypeStruct(np.asarray(value).shape, np.asarray(value).dtype)
        for value in feeds[0]
    ]
    native_trace = jax.make_jaxpr(_fresh(fn))(*specs)
    (directory / "native-trace.json").write_text(
        json.dumps(typed_trace_manifest(native_trace), indent=2)
    )
    for i, record in enumerate(records):
        record.assert_current()
        (directory / f"preparation-{i}.json").write_text(record.to_json())
    model = ui._to_onnx(
        fn,
        specs,
        precision_policy=PrecisionPolicy.SOURCE,
        opset=23,
        output_names=[f"result{i}" for i in range(len(intended[0]))],
    )
    payload = model.SerializeToString()
    path = directory / "source.onnx"
    path.write_bytes(payload)
    reloaded = onnx.load_model(path)
    onnx.checker.check_model(reloaded, full_check=True)
    report = get_precision_report(reloaded)
    assert report is not None and report.profile == "source-static-v1"
    (directory / "export-report.json").write_text(precision_report_to_json(report))
    assert {c.category: c.status for c in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    for port, expected in zip(reloaded.graph.output, intended[0], strict=True):
        _port(port, expected)
    observed, actual_payloads = [], []
    constructor = ort.InferenceSession
    session_index = 0
    active_label = "unset"

    def recording_session(*args: Any, **kwargs: Any) -> Any:
        nonlocal session_index
        actual = constructor(*args, **kwargs)
        index = session_index
        session_index += 1
        run = actual.run
        run_index = 0

        def recording_run(*args: Any, **kwargs: Any) -> Any:
            nonlocal run_index
            outputs = run(*args, **kwargs)
            filename = f"actual-{active_label}-session{index}-run{run_index}.npz"
            _save(directory / filename, outputs)
            actual_payloads.append(
                {
                    "label": active_label,
                    "file": filename,
                    "sha256": hashlib.sha256(
                        (directory / filename).read_bytes()
                    ).hexdigest(),
                    "arrays": [
                        {
                            "dtype": np.asarray(value).dtype.name,
                            "shape": np.asarray(value).shape,
                            "sha256": hashlib.sha256(
                                np.asarray(value).tobytes()
                            ).hexdigest(),
                        }
                        for value in outputs
                    ],
                }
            )
            run_index += 1
            return outputs

        actual.run = recording_run
        return actual

    with patch.object(ort, "InferenceSession", recording_session):
        for optimization in ("disabled", "all"):
            for role in (0, 1):
                active_label = f"{optimization}-native{role}"
                samples = [
                    SourceSample(values, native[role], _BITWISE)
                    for values, native in zip(frozen_feeds, references, strict=True)
                ]
                result = validate_source_samples(
                    reloaded, samples=samples, optimization=optimization
                )
                observed.append((optimization, role, result))
    (directory / "actual-runtime-payloads.json").write_text(
        json.dumps(actual_payloads, indent=2)
    )
    (directory / "comparisons-before-guards.json").write_text(
        json.dumps(
            [
                {
                    "optimization": mode,
                    "native_role": role,
                    "target": None if result.target is None else asdict(result.target),
                    "checks": [asdict(c) for c in result.checks],
                    "samples": [asdict(s) for s in result.samples],
                }
                for mode, role, result in observed
            ],
            indent=2,
        )
    )
    for _, _, result in observed:
        result.require_sample_acceptance()
    for native, wanted in zip(references, intended, strict=True):
        for outputs in native:
            for actual, expected in zip(outputs, wanted, strict=True):
                _same(actual, expected)
    for values, frozen in zip(feeds, frozen_feeds, strict=True):
        for actual, expected in zip(values, frozen, strict=True):
            _same(actual, expected)
    for record in records:
        record.assert_current()
    assert model.SerializeToString() == payload
    assert get_precision_report(reloaded) == report
    return reloaded


@pytest.mark.parametrize("route", ["inline", "named", "conditional"])
def test_prepared_complete_root_uses_selected_helper_and_untouched_original_sibling(
    route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.asarray([1e8, 1e8 + 1, -1e8, 0, 16777216], np.float64)
    old = np.asarray([0, 0, 0, 1, 0], np.float32)
    if route == "conditional":
        branch, branch_record, _, helper_record = _prepared_global(_selected_branch)
        fn, record = prepare_callable(
            _conditional_root,
            _root_recipe(
                _conditional_root,
                "_selected_branch",
                _selected_branch,
                branch,
                branch_record,
            ),
        )
        feeds = [(values, np.asarray(True)), (values, np.asarray(False))]
        intended = [(np.ones(5, np.float64),), (old.astype(np.float64),)]
        records = (record, branch_record, helper_record)
    else:
        prepared, record, _, helper_record = _prepared_global()
        if route == "named":
            monkeypatch.setitem(globals(), "_NAMED_ROOT_IMPL", prepared)
        fn = _named_root_entry if route == "named" else prepared
        feeds, intended = [(values,)], [(np.ones(5, np.float64), old)]
        records = (record, helper_record)
    model = _observe(fn, feeds, intended, tmp_path, records=records)
    if route == "named":
        _function_edges(model, tmp_path)
    _same(_rounding(jax.device_put(values)), old)
    _same(_global_root(jax.device_put(values))[0], old)


@pytest.mark.parametrize("owner", ["linear", "softmax"])
def test_actual_prepared_existing_owners_export_finite_exact_controls(
    owner: str, tmp_path: Path
) -> None:
    linear, softmax = _real_owners()
    original = linear if owner == "linear" else softmax
    before = _owner_snapshot(original)
    helper, helper_record = prepare_callable(original, _recipe(original))
    if owner == "linear":
        model = _linear_fixture()
        frozen = tuple(np.asarray(value).copy() for value in (model.weight, model.bias))
        original_root = _owned_root(original, model)
        root, root_record = prepare_callable(
            original_root,
            _root_recipe(
                original_root,
                "helper",
                original,
                helper,
                helper_record,
                scope="closure",
                shared=(SharedBinding("closure", "owner", model),),
            ),
        )
        values = _linear_values()
        intended = [(_linear_expected(values),)]
    else:
        original_root = _owned_root(original)
        root, root_record = prepare_callable(
            original_root,
            _root_recipe(
                original_root,
                "helper",
                original,
                helper,
                helper_record,
                scope="closure",
            ),
        )
        values = np.zeros((2, 4), np.float32)
        intended = [(np.full((2, 4), 0.25, np.float32),)]
    artifact = _observe(
        root, [(values,)], intended, tmp_path, records=(root_record, helper_record)
    )
    assert _owner_snapshot(original) == before
    if owner == "linear":
        for actual, saved in zip((model.weight, model.bias), frozen, strict=True):
            _same(actual, saved)
    else:
        _softmax_artifact_path(artifact, tmp_path)


def test_two_prepared_versions_and_original_keep_separate_nested_export_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, first_record = prepare_callable(
        _rounding, _recipe(_rounding, replacement=np.float32)
    )
    second, second_record = prepare_callable(
        _rounding, _recipe(_rounding, replacement=np.float64)
    )
    assert first.__code__.co_code == second.__code__.co_code
    monkeypatch.setitem(globals(), "_FIRST_VERSION_IMPL", first)
    monkeypatch.setitem(globals(), "_SECOND_VERSION_IMPL", second)
    values = np.asarray([1e8 + 1, 16777216], np.float64)
    versions = (
        ("first", _first_version_entry, first_record, np.zeros(2, np.float32)),
        ("second", _second_version_entry, second_record, np.ones(2, np.float64)),
    )
    for index, (name, nested, record, intended) in enumerate((*versions, versions[0])):
        directory = tmp_path / f"{index}-{name}"
        artifact = _observe(
            nested,
            [(values,)],
            [(intended,)],
            directory,
            records=(record,),
        )
        _function_edges(artifact, directory, nested=True)
        _same(_rounding(jax.device_put(values)), np.zeros(2, np.float32))


def test_protected_owner_barrier_is_preserved_without_validator_replacement() -> None:
    prepared, record, _, _ = _prepared_global(_protected_root)
    values = np.asarray([1e8 + 1, 0], np.float64)
    native = prepared(jax.device_put(values))
    assert np.asarray(native[4]).dtype == np.uint32
    with pytest.raises(SourcePrecisionError) as caught:
        ui._to_onnx(
            prepared,
            [jax.ShapeDtypeStruct(values.shape, values.dtype)],
            precision_policy=PrecisionPolicy.SOURCE,
            opset=23,
        )
    assert caught.value.primitive == "bitcast_convert_type"
    assert caught.value.category == "representation-limit"
    record.assert_current()


@pytest.mark.parametrize("fault", ["dtype", "payload"])
def test_finite_cpu_control_rejects_wrong_runtime_dtype_or_payload(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, record = _prepare_rounding()
    values = np.asarray([1e8 + 1, 16777216], np.float64)
    wanted = np.ones(2, np.float64)
    model = _observe(prepared, [(values,)], [(wanted,)], tmp_path, records=(record,))
    constructor = ort.InferenceSession

    def session(*args: Any, **kwargs: Any) -> Any:
        actual = constructor(*args, **kwargs)
        run = actual.run

        def corrupt(*args: Any, **kwargs: Any) -> Any:
            outputs = run(*args, **kwargs)
            outputs[0] = (
                outputs[0].astype(np.float32)
                if fault == "dtype"
                else np.zeros_like(outputs[0])
            )
            _save(tmp_path / "negative-corrupted-runtime.npz", outputs)
            return outputs

        actual.run = corrupt
        return actual

    monkeypatch.setattr(ort, "InferenceSession", session)
    sample = SourceSample((values,), (wanted,), _BITWISE)
    result = validate_source_samples(model, samples=[sample], optimization="disabled")
    # These are static outputs: the existing generic comparator owns both
    # dtype and value failure, without borrowing symbolic-boundary categories.
    category = "numerical-acceptance"
    assert any(
        c.category == category and c.status is CheckStatus.FAILED
        for c in result.samples[0].checks
    )
    with pytest.raises(SampleAcceptanceError):
        result.require_sample_acceptance()
    assert sample.criteria is _BITWISE
    _same(wanted, np.ones(2, np.float64))


def test_corrupted_selected_helper_binding_invalidates_preparation_before_export() -> (
    None
):
    prepared, record, _, _ = _prepared_global()
    values = np.asarray([1e8 + 1], np.float64)
    _same(prepared(jax.device_put(values))[0], np.ones(1, np.float64))
    prepared.__globals__["_ROUNDING"] = _rounding
    with pytest.raises(SourcePreparationError):
        record.assert_current()
    _same(prepared(jax.device_put(values))[0], np.zeros(1, np.float32))
    assert _global_root.__globals__["_ROUNDING"] is _rounding


def test_preparation_does_not_change_the_ordinary_public_legacy_default() -> None:
    assert "precision_policy" not in inspect.signature(to_onnx).parameters
    assert (
        inspect.signature(ui._to_onnx).parameters["precision_policy"].default
        is PrecisionPolicy.LEGACY
    )
