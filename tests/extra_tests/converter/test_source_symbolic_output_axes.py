# tests/extra_tests/converter/test_source_symbolic_output_axes.py

"""Input-derived atomic B binds to each physical output axis without relaxation."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.converter.source_representation import normalized_shape_dimensions
from jax2onnx.precision import (
    CheckStatus,
    PRECISION_METADATA_KEY,
    SampleAcceptanceError,
    SourceSample,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter import test_source_symbolic_contraction as contraction
from tests.extra_tests.converter.test_source_symbolic_contraction import (
    _BITWISE,
    _ORIENTATION,
    _adapt_zero_output,
    _body,
    _fresh,
    _observe,
    _oracle,
    _payload,
    _port,
    _restore_settings as _restore_settings,
    _same,
    _save,
    _specs,
)

_AXES = [
    (rank, axis, "float32", False) for rank in (2, 3, 4) for axis in range(rank)
] + [(4, 3, "float64", True), (3, 1, "float32", True)]


def _order(rank: int, axis: int) -> tuple[int, ...]:
    if axis == 0 and rank >= 3:
        return (0, 2, 1, *range(3, rank))
    order = list(range(1, rank))
    order.insert(axis, 0)
    return tuple(order)


def _shape(rank: int) -> tuple[Any, ...]:
    return ("B", *((3,) if rank == 2 else (2, 3) if rank == 3 else (2, 3, 4)))


@pytest.mark.parametrize("rank,axis,dtype,x64", _AXES)
def test_real_native_transpose_and_aliases_bind_B_at_every_output_axis(
    rank: int,
    axis: int,
    dtype: str,
    x64: bool,
    tmp_path: Path,
) -> None:
    # Rank-two axis zero is the direct alias companion. Higher-rank axis-zero
    # cases swap unequal static axes and therefore contain an actual transpose.
    with jax.enable_x64(x64):
        shape, order = _shape(rank), _order(rank, axis)
        output_shape = tuple(shape[a] for a in order)

        def fn(x: Any) -> Any:
            moved = jax.lax.transpose(x, order)
            return moved, moved, x

        feeds = [(_payload(dtype, (b, *shape[1:])),) for b in (2, 5)]
        expected = [
            (np.transpose(x, order).copy(), np.transpose(x, order).copy(), x)
            for (x,) in feeds
        ]
        model, trace, _ = _observe(
            fn,
            _specs([shape], dtype),
            feeds,
            expected,
            tmp_path,
            output_shapes=[output_shape, output_shape, shape],
        )
        _port(model.graph.output[0], dtype, output_shape)
        assert [v.name for v in model.graph.output] == [
            v.name for v in get_precision_report(model).exported_outputs
        ]
        body = _body(trace)
        assert body.outvars[0] is body.outvars[1]
        if axis or rank >= 3:
            assert any(
                e.primitive is jax.lax.transpose_p
                and e.params == {"permutation": order}
                for e in body.eqns
            )


@onnx_function(name="NestedNonleadingOutputBoundary")
def _nested_raw(x: Any, weight: Any) -> Any:
    return contraction._named_raw(x, weight)


@pytest.mark.parametrize("compiled,dtype", [(False, "float32"), (True, "float64")])
def test_nested_function_jit_mixed_static_leading_and_nonleading_output_order(
    compiled: bool,
    dtype: str,
    tmp_path: Path,
) -> None:
    with jax.enable_x64(dtype == "float64"):
        weight = _payload(dtype, (64, 128), right=True)
        frozen_weight = weight.copy()

        def fn(x: Any) -> Any:
            result = (
                jax.jit(_nested_raw)(x, weight) if compiled else _nested_raw(x, weight)
            )
            return result, x, weight, result, jax.lax.transpose(result, (1, 0))

        feeds = [(_payload(dtype, (b, 128)),) for b in (2, 5)]
        expected = []
        for (x,) in feeds:
            raw = _oracle(weight, x, _ORIENTATION, dtype)
            expected.append((raw, x, weight, raw, raw.T.copy()))
        shapes = [(64, "B"), ("B", 128), (64, 128), (64, "B"), ("B", 64)]
        model, _, built = _observe(
            fn,
            _specs([("B", 128)], dtype),
            feeds,
            expected,
            tmp_path,
            output_shapes=shapes,
            named=True,
        )
        assert len(built) == len(model.functions) == 2
        definitions = {(f.domain, f.name): f for f in model.functions}
        root_calls = {
            (node.domain, node.op_type)
            for node in model.graph.node
            if (node.domain, node.op_type) in definitions
        }
        (outer,) = root_calls
        inner_calls = {
            (node.domain, node.op_type)
            for node in definitions[outer].node
            if (node.domain, node.op_type) in definitions
        }
        assert inner_calls == set(definitions) - {outer}
        for index, (definition, closed) in enumerate(built):
            (tmp_path / f"function-{index}-native-trace.json").write_text(
                json.dumps(typed_trace_manifest(closed), indent=2)
            )
            body, function = (
                _body(closed),
                definitions[(definition.domain, definition.name)],
            )
            infos = {v.name: v for v in function.value_info}
            for names, variables in (
                (function.input, body.invars),
                (function.output, body.outvars),
            ):
                for name, var in zip(names, variables, strict=True):
                    shape = normalized_shape_dimensions(var.aval.shape)
                    _port(infos[name], var.aval.dtype, shape)
        assert [v.name for v in model.graph.output] == [
            v.name for v in get_precision_report(model).exported_outputs
        ]
        _same(weight, frozen_weight)
        assert all(
            tuple(v.shape) == shape
            for v, shape in zip(
                get_precision_report(model).exported_outputs, shapes, strict=True
            )
        )


@pytest.mark.parametrize(
    "rank,axis,dtype",
    [(r, a, "float32") for r in (2, 3, 4) for a in range(1, r)] + [(4, 3, "float64")],
)
def test_nonleading_empty_results_remain_bound_to_positive_input_B(
    rank: int,
    axis: int,
    dtype: str,
    tmp_path: Path,
) -> None:
    with jax.enable_x64(dtype == "float64"):
        shape, order = _shape(rank), _order(rank, axis)
        weight = np.empty((shape[-1], 0), dtype)
        dimensions = (((rank - 1,), (0,)), ((), ()))
        logical_shape = (*shape[:-1], 0)
        output_shape = tuple(logical_shape[a] for a in order)

        def fn(x: Any) -> Any:
            return jax.lax.transpose(jax.lax.dot_general(x, weight, dimensions), order)

        feeds = [(_payload(dtype, (b, *shape[1:])),) for b in (2, 5)]
        expected = [
            (np.transpose(np.empty((b, *shape[1:-1], 0), dtype), order).copy(),)
            for b in (2, 5)
        ]
        _observe(
            fn,
            _specs([shape], dtype),
            feeds,
            expected,
            tmp_path,
            output_shapes=[output_shape],
        )


def _baseline(*, empty: bool = False) -> tuple[Any, Any]:
    static = np.asarray([-0.5, 0.25], np.float32)
    weight = np.empty((3, 0), np.float32)

    def fn(x: Any) -> Any:
        base = jax.lax.dot_general(x, weight, (((1,), (0,)), ((), ()))) if empty else x
        raw = jax.lax.transpose(base, (1, 0))
        return raw, x, static, raw

    x = _payload(np.float32, (2, 3))
    raw = np.empty((0, 2), np.float32) if empty else x.T.copy()
    wanted = (raw, x, static, raw)
    for call in (fn, jax.jit(_fresh(fn))):
        for actual, expected in zip(jax.tree.leaves(call(x)), wanted, strict=True):
            _same(np.asarray(actual), expected)
    model = ui._to_onnx(
        fn,
        _specs([("B", 3)], np.float32),
        precision_policy=PrecisionPolicy.SOURCE,
        input_names=["x"],
        output_names=["raw", "leading", "static", "again"],
    )
    onnx.checker.check_model(model, full_check=True)
    sample = SourceSample((x,), wanted, _BITWISE)
    frozen = tuple(np.asarray(value).copy() for value in (x, *wanted))
    validate_source_samples(
        model, samples=[sample], optimization="disabled"
    ).require_sample_acceptance()
    for actual, saved in zip((x, *wanted), frozen, strict=True):
        _same(actual, saved)
    assert sample.criteria is _BITWISE
    return model, sample


def _metadata(model: Any) -> dict[str, Any]:
    return json.loads(precision_report_to_json(get_precision_report(model)))


def _attach(model: Any, metadata: Any) -> None:
    for item in model.metadata_props:
        if item.key == PRECISION_METADATA_KEY:
            item.value = json.dumps(metadata)
            return
    raise AssertionError("SOURCE report disappeared")


def _swap(values: Any) -> None:
    copied = type(values[0])()
    copied.CopyFrom(values[0])
    values[0].CopyFrom(values[1])
    values[1].CopyFrom(copied)


def _rename_output(model: Any) -> None:
    old, new = model.graph.output[0].name, "changed_raw"
    for node in model.graph.node:
        for names in (node.input, node.output):
            for i, name in enumerate(names):
                if name == old:
                    names[i] = new
    for value in (*model.graph.value_info, *model.graph.output):
        if value.name == old:
            value.name = new


@pytest.mark.parametrize(
    "fault",
    [
        "count",
        "name",
        "order",
        "dtype",
        "axis",
        "rank",
        "tail",
        "missing-label",
        "repeated-label",
        "derived-label",
        "fresh-label",
        "unknown-extent",
        "bool-tail",
        "negative-tail",
        "rank-five",
        "input-disagreement",
        "input-zero-tail",
        "input-name",
        "no-input-binding",
        "artifact-count",
        "artifact-name",
        "artifact-order",
        "artifact-dtype",
        "artifact-axis",
        "artifact-unknown",
    ],
)
def test_every_output_descriptor_and_input_binding_guard_precedes_runtime(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, sample = _baseline()
    metadata = _metadata(model)
    logical = metadata["traced_outputs"][0]
    if fault == "count":
        metadata["exported_outputs"].pop()
    elif fault == "name":
        metadata["exported_outputs"][0]["name"] = "Other"
    elif fault == "order":
        metadata["exported_outputs"][0], metadata["exported_outputs"][1] = (
            metadata["exported_outputs"][1],
            metadata["exported_outputs"][0],
        )
    elif fault == "dtype":
        logical["dtype"] = "float64"
    elif fault in {
        "axis",
        "rank",
        "tail",
        "missing-label",
        "repeated-label",
        "derived-label",
        "fresh-label",
        "unknown-extent",
        "bool-tail",
        "negative-tail",
        "rank-five",
    }:
        logical["shape"] = {
            "axis": ["B", 3],
            "rank": [3, "B", 1],
            "tail": [4, "B"],
            "missing-label": [3, 2],
            "repeated-label": ["B", "B"],
            "derived-label": [3, "2*B"],
            "fresh-label": [3, "C"],
            "unknown-extent": [None, "B"],
            "bool-tail": [True, "B"],
            "negative-tail": [-1, "B"],
            "rank-five": [3, 1, 1, 1, "B"],
        }[fault]
    elif fault in {"input-disagreement", "input-zero-tail"}:
        metadata["inputs"][0]["normalized"]["shape"][1] = (
            4 if fault == "input-disagreement" else 0
        )
    elif fault == "input-name":
        metadata["inputs"][0]["exported"]["name"] = "Other"
    elif fault == "no-input-binding":
        for key in ("supplied", "normalized", "traced", "exported"):
            metadata["inputs"][0][key]["shape"][0] = 2
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 2
    elif fault == "artifact-count":
        del model.graph.output[-1]
    elif fault == "artifact-name":
        _rename_output(model)
    elif fault == "artifact-order":
        _swap(model.graph.output)
    elif fault == "artifact-dtype":
        _adapt_zero_output(model, "Cast")
    elif fault == "artifact-axis":
        _adapt_zero_output(model, "Transpose")
        dims = model.graph.output[0].type.tensor_type.shape.dim
        dims[0].dim_param, dims[1].dim_value = "B", 3
    else:
        model.graph.output[0].type.tensor_type.shape.dim[1].Clear()
    _attach(model, metadata)
    onnx.checker.check_model(model, full_check=True)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid output boundary created a runtime session")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    result = validate_source_samples(model, samples=[sample], optimization="disabled")
    assert any(
        c.category == "source-shape-contract" and c.status is CheckStatus.FAILED
        for c in result.checks
    )
    with pytest.raises(SampleAcceptanceError):
        result.require_sample_acceptance()


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "B-zero",
        "wrong-reference-B",
        "wrong-reference-static",
        "wrong-input-tail",
        "wrong-input-dtype",
    ],
)
def test_nonleading_reference_shapes_and_input_B_reject_before_session(
    empty: bool,
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, sample = _baseline(empty=empty)
    frozen = tuple(np.asarray(v).copy() for v in sample.expected_outputs)
    inputs, expected = sample.inputs, list(sample.expected_outputs)
    width = 0 if empty else 3
    if fault == "B-zero":
        inputs = (np.empty((0, 3), np.float32),)
        expected[0] = expected[3] = np.empty((width, 0), np.float32)
        expected[1] = np.empty((0, 3), np.float32)
    elif fault == "wrong-reference-B":
        expected[0] = np.empty((width, 5), np.float32)
    elif fault == "wrong-reference-static":
        expected[0] = np.empty((width + 1, 2), np.float32)
    elif fault == "wrong-input-tail":
        inputs = (np.empty((2, 0), np.float32),)
    else:
        inputs = (np.asarray(inputs[0], np.float64),)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid reference/input binding created a runtime session")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    result = validate_source_samples(
        model,
        samples=[SourceSample(inputs, expected, sample.criteria)],
        optimization="all",
    )
    assert any(
        c.category == "sample-input" and c.status is CheckStatus.FAILED
        for c in result.samples[0].checks
    )
    assert any(
        c.category == "numerical-acceptance" and c.status is CheckStatus.UNVERIFIED
        for c in result.samples[0].checks
    )
    assert sample.criteria is _BITWISE
    for actual, saved in zip(sample.expected_outputs, frozen, strict=True):
        _same(actual, saved)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("fault", ["wrong-B", "wrong-static"])
def test_actual_nonleading_runtime_output_is_checked_before_numerical_acceptance(
    empty: bool,
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, sample = _baseline(empty=empty)
    constructor = ort.InferenceSession
    original_criteria = asdict(sample.criteria)

    def session(*args: Any, **kwargs: Any) -> Any:
        actual = constructor(*args, **kwargs)
        run = actual.run

        def changed(*args: Any, **kwargs: Any) -> Any:
            outputs = run(*args, **kwargs)
            width = 0 if empty else 3
            outputs[0] = np.empty(
                (width, 5) if fault == "wrong-B" else (width + 1, 2), np.float32
            )
            return outputs

        actual.run = changed
        return actual

    monkeypatch.setattr(ort, "InferenceSession", session)
    result = validate_source_samples(model, samples=[sample], optimization="disabled")
    assert any(
        c.category == "sample-output" and c.status is CheckStatus.FAILED
        for c in result.samples[0].checks
    )
    assert any(
        c.category == "numerical-acceptance" and c.status is CheckStatus.UNVERIFIED
        for c in result.samples[0].checks
    )
    assert asdict(sample.criteria) == original_criteria
    with pytest.raises(SampleAcceptanceError):
        result.require_sample_acceptance()


def _nchw_boundary(dtype: str, directory: Path | None = None) -> tuple[Any, Any, str]:
    order = (1, 2, 3, 0)

    def fn(x: Any) -> Any:
        return jax.lax.transpose(x, order)

    specs = _specs([("B", 2, 3, 4)], dtype)
    native_trace = jax.make_jaxpr(_fresh(fn))(*specs)
    if directory is not None:
        (directory / "nchw-native-trace.txt").write_text(str(native_trace))
        (directory / "nchw-native-trace.json").write_text(
            json.dumps(typed_trace_manifest(native_trace), indent=2)
        )
        (directory / "nchw-criteria.json").write_text(json.dumps(asdict(_BITWISE)))
    samples = []
    for batch in (2, 5):
        value = _payload(dtype, (batch, 2, 3, 4))
        logical = np.transpose(value, order).copy()
        physical = np.transpose(logical, (0, 3, 1, 2)).copy()
        outputs = []
        for role, call in (("eager", fn), ("fresh-jit", jax.jit(_fresh(fn)))):
            native = np.asarray(call(value)).copy()
            outputs.append(native)
            _same(native, logical)
            if directory is not None:
                _save(directory / f"nchw-{batch}-{role}.npz", (native,))
        if directory is not None:
            _save(directory / f"nchw-{batch}-inputs.npz", (value,))
            _save(directory / f"nchw-{batch}-physical-coordinate.npz", (physical,))
        samples.append(SourceSample((value,), (physical,), _BITWISE))
    model = ui._to_onnx(
        fn,
        _specs([("B", 2, 3, 4)], dtype),
        precision_policy=PrecisionPolicy.SOURCE,
        outputs_as_nchw=[0],
        input_names=["x"],
        output_names=["moved"],
    )
    onnx.checker.check_model(model, full_check=True)
    report = get_precision_report(model)
    assert report is not None
    assert report.traced_outputs[0].shape == (2, 3, 4, "B")
    assert report.exported_outputs[0].shape == (2, "B", 3, 4)
    _port(model.graph.output[0], dtype, (2, "B", 3, 4))
    provenance = (
        "output 0: logical NHWC (2, 3, 4, B) uses external NCHW storage (2, B, 3, 4)"
    )
    assert provenance in report.adaptations
    if directory is not None:
        (directory / "nchw.onnx").write_bytes(model.SerializeToString())
        (directory / "nchw-report.json").write_text(precision_report_to_json(report))
    payload = model.SerializeToString()
    frozen_samples = [
        tuple(
            np.asarray(value).copy()
            for value in (*sample.inputs, *sample.expected_outputs)
        )
        for sample in samples
    ]
    for optimization in ("disabled", "all"):
        result = validate_source_samples(
            model, samples=samples, optimization=optimization
        )
        if directory is not None:
            (directory / f"nchw-{optimization}-comparisons.json").write_text(
                json.dumps(
                    {
                        "target": (
                            None if result.target is None else asdict(result.target)
                        ),
                        "samples": [asdict(s) for s in result.samples],
                    },
                    indent=2,
                )
            )
        result.require_sample_acceptance()
    assert get_precision_report(model) == report
    assert model.SerializeToString() == payload
    for sample, saved in zip(samples, frozen_samples, strict=True):
        for value, original in zip(
            (*sample.inputs, *sample.expected_outputs), saved, strict=True
        ):
            _same(value, original)
        assert sample.criteria is _BITWISE
    return model, samples, provenance


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_exact_NCHW_adapter_moves_nonleading_native_B_to_its_physical_axis(
    dtype: str,
    tmp_path: Path,
) -> None:
    with jax.enable_x64(dtype == "float64"):
        _nchw_boundary(dtype, tmp_path)


@pytest.mark.parametrize(
    "fault",
    ["missing-provenance", "wrong-output-index", "traced-axis", "exported-axis"],
)
def test_nonleading_NCHW_boundary_requires_the_exact_existing_adaptation(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, samples, provenance = _nchw_boundary("float32")
    metadata = _metadata(model)
    if fault == "missing-provenance":
        metadata["adaptations"].remove(provenance)
    elif fault == "wrong-output-index":
        metadata["adaptations"][metadata["adaptations"].index(provenance)] = (
            provenance.replace("output 0:", "output 1:")
        )
    elif fault == "traced-axis":
        metadata["traced_outputs"][0]["shape"] = [2, 3, "B", 4]
    else:
        metadata["exported_outputs"][0]["shape"] = ["B", 2, 3, 4]
    _attach(model, metadata)
    onnx.checker.check_model(model, full_check=True)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("forged NCHW boundary created a runtime session")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    result = validate_source_samples(model, samples=samples, optimization="disabled")
    assert any(
        c.category == "source-shape-contract" and c.status is CheckStatus.FAILED
        for c in result.checks
    )


@pytest.mark.parametrize("shape", [("B", 0), (3, "B")])
def test_output_axis_packet_keeps_public_input_declarations_unchanged(
    shape: Any,
) -> None:
    # Prove an otherwise ordinary identity first, then reject before invocation.
    model, _ = _baseline()
    assert get_precision_report(model) is not None
    called = []

    def fn(x: Any) -> Any:
        called.append(True)
        return x

    with pytest.raises(SourcePrecisionError):
        ui._to_onnx(
            fn,
            [jax.ShapeDtypeStruct(shape, np.float32)],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert not called
