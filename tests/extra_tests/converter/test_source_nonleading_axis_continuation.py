# tests/extra_tests/converter/test_source_nonleading_axis_continuation.py

"""Native axis-one attention layout proof; whole original-model gates stay separate."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import atomic_symbolic_dimension
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    _Site,
    _Tensor,
    _pointwise_shape,
    _reduction_shape,
    validate_source_jaxpr,
)
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_symbolic_contraction import (
    _isolated_graph,
    _rebind,
    _restore_settings as _restore_settings,
)

_CRITERIA = SampleCriteria("bitwise")


def _attention(logits: Any, mask: Any) -> Any:
    shape = (1, *logits.shape)
    expanded = jax.lax.broadcast_in_dim(logits, shape, (1, 2, 3, 4))
    expanded_mask = jax.lax.broadcast_in_dim(mask, shape, (1, 2, 3, 4))
    expanded_mask = jax.lax.bitwise_and(expanded_mask, expanded_mask)
    selected = jnp.where(expanded_mask, expanded, np.float32(-np.inf))
    peak = jax.lax.reduce_max(selected, (4,))
    peak = jax.lax.max(peak, jnp.asarray(np.float32(-np.inf)))
    peak = jax.lax.stop_gradient(peak)
    peak = jax.lax.copy_p.bind(peak)
    centered = jax.lax.sub(
        selected, jax.lax.broadcast_in_dim(peak, shape, (0, 1, 2, 3))
    )
    weights = jax.lax.exp(centered)
    total = jax.lax.reduce_sum(weights, (4,))
    normalized = jax.lax.div(
        weights, jax.lax.broadcast_in_dim(total, shape, (0, 1, 2, 3))
    )
    leading = jax.lax.transpose(normalized, (1, 0, 2, 3, 4))
    return jax.lax.reshape(leading, logits.shape)


@onnx_function
def _named_attention(logits: Any, mask: Any) -> Any:
    return _attention(logits, mask)


def _call_named_attention(logits: Any, mask: Any) -> Any:
    # Resolve the patched global at activation, retaining the actual named call.
    return _named_attention(logits, mask)


def _graph_calls(graph: onnx.GraphProto) -> set[tuple[str, str, str]]:
    calls = {(node.domain, node.op_type, node.overload) for node in graph.node}
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                calls.update(_graph_calls(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    calls.update(_graph_calls(child))
    return calls


def _specs() -> list[Any]:
    (batch,) = export.symbolic_shape("B")
    return [
        jax.ShapeDtypeStruct((batch, 2, 4, 4), dtype)
        for dtype in (np.float32, np.bool_)
    ]


def _values(batch: int) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    # Each row has two selected entries. All finite logits in a row are equal,
    # so the independent oracle is exactly 1/2 on selected entries and +0 else.
    rows = np.arange(batch * 2 * 4, dtype=np.float32).reshape(batch, 2, 4, 1)
    logits = np.broadcast_to(rows, (batch, 2, 4, 4)).copy()
    indices = np.arange(4).reshape(1, 1, 1, 4)
    mask = (indices + rows.astype(np.int32)) % 4 < 2
    expected = mask.astype(np.float32) * np.float32(0.5)
    return (logits, mask), expected


def _native_references(
    function: Any, values: Any, directory: Path
) -> tuple[np.ndarray, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    compiled = jax.jit(function)
    events: list[dict[str, Any]] = []
    references = []
    for label, call in (("fresh", compiled), ("warmed", compiled), ("eager", function)):
        try:
            returned = call(*values)
        except BaseException as exc:
            events.append(
                dict(
                    stage=label,
                    status="raised",
                    exception_type=type(exc).__qualname__,
                    message=str(exc),
                )
            )
            (directory / "native-events.json").write_text(json.dumps(events, indent=2))
            raise
        events.append(dict(stage=label, status="returned"))
        (directory / "native-events.json").write_text(json.dumps(events, indent=2))
        # Retain this actual return before checks or any subsequent native call.
        np.savez(directory / f"native-{label}.npz", actual=jax.device_get(returned))
        actual = np.asarray(returned)
        assert actual.dtype == np.float32 and actual.shape == values[0].shape
        references.append(actual.copy())
    return tuple(references)


@pytest.mark.parametrize("raw_x64", [False, True])
@pytest.mark.parametrize("scope", ["inline", "named", "jit"])
def test_axis_one_attention_keeps_origin_and_exact_cpu_samples(
    raw_x64: bool, scope: str, tmp_path: Path
) -> None:
    with jax.enable_x64(raw_x64):
        function = _attention if scope == "inline" else _call_named_attention
        if scope == "jit":
            function = jax.jit(function)
        specs = _specs()
        native = jax.make_jaxpr(function)(*specs)
        original = typed_trace_manifest(native)
        validate_source_jaxpr(native.jaxpr, 23)
        (tmp_path / "native-trace.json").write_text(json.dumps(original, indent=2))
        (tmp_path / "criteria.json").write_text(json.dumps(asdict(_CRITERIA)))
        samples = []
        for batch in (2, 5):
            directory = tmp_path / f"B{batch}"
            directory.mkdir()
            values, expected = _values(batch)
            np.savez(directory / "independent-oracle.npz", actual=expected)
            np.savez(directory / "inputs.npz", logits=values[0], mask=values[1])
            references = _native_references(function, values, directory)
            for reference in references:
                assert (
                    reference.dtype == expected.dtype
                    and reference.shape == expected.shape
                )
                assert reference.tobytes() == expected.tobytes()
                samples.append(SourceSample(values, [reference], _CRITERIA))
        model = ui._to_onnx(
            function, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        (tmp_path / "source.onnx").write_bytes(model.SerializeToString())
        onnx.checker.check_model(model, full_check=True)
        if scope in {"named", "jit"}:
            functions = [
                function
                for function in model.functions
                if function.name == "_named_attention"
            ]
            assert len(functions) == 1 and functions[0].domain
            definition = functions[0]
            assert (
                definition.domain,
                definition.name,
                definition.overload,
            ) in _graph_calls(model.graph)
        report = get_precision_report(model)
        assert report is not None
        (tmp_path / "source-report.json").write_text(precision_report_to_json(report))
        assert {check.category: check.status for check in report.checks}[
            "numerical-acceptance"
        ] is CheckStatus.UNVERIFIED
        for optimization in ("disabled", "all"):
            checked = validate_source_samples(
                model, samples=samples, optimization=optimization
            )
            (tmp_path / f"runtime-{optimization}.json").write_text(
                json.dumps(
                    dict(
                        target=asdict(checked.target),
                        samples=[asdict(sample) for sample in checked.samples],
                    ),
                    indent=2,
                )
            )
            checked.require_sample_acceptance()
        assert typed_trace_manifest(native) == original


@pytest.mark.parametrize(
    "fault", ["scope", "symbol-axis", "static-B", "tail", "rank", "result-scope"]
)
def test_right_aligned_pointwise_rejects_foreign_or_misaligned_batch(
    fault: str,
) -> None:
    (batch,) = export.symbolic_shape("B")
    (foreign,) = export.symbolic_shape("B")
    symbol = atomic_symbolic_dimension(batch)
    foreign_symbol = atomic_symbolic_dimension(foreign)
    left = _Tensor("float32", (1, "B", 4), symbol, 1)
    right = _Tensor("float32", ("B", 4), symbol, 0)
    result = left
    site = _Site("axis-one-pointwise", 23, "sub")
    _pointwise_shape([left, right], result, site)
    if fault == "scope":
        right = replace(right, symbol=foreign_symbol)
    elif fault == "symbol-axis":
        right = replace(right, shape=("B", 1, 4), symbol_axis=0)
    elif fault == "static-B":
        right = _Tensor("float32", (2, 4))
    elif fault == "tail":
        right = replace(right, shape=("B", 3))
    elif fault == "rank":
        result = replace(result, shape=(1, 1, "B", 4), symbol_axis=2)
    else:
        result = replace(result, symbol=foreign_symbol)
    with pytest.raises(SourcePrecisionError):
        _pointwise_shape([left, right], result, site)


@pytest.mark.parametrize(
    "fault",
    [
        "declared-scope",
        "mapped-axis",
        "shape",
        "dtype",
        "weakness",
        "placement",
        "duplicate-axis",
        "boolean-axis",
    ],
)
def test_native_broadcast_keeps_original_axis_scope_and_metadata(fault: str) -> None:
    native = jax.make_jaxpr(_attention)(*_specs())
    before = typed_trace_manifest(native)
    graph = _isolated_graph(native)
    validate_source_jaxpr(graph, 23)
    eqn = next(e for e in graph.eqns if e.primitive is jax.lax.broadcast_in_dim_p)
    if fault == "declared-scope":
        (foreign,) = export.symbolic_shape("B")
        eqn.params["shape"] = tuple(
            foreign if not isinstance(dim, (int, np.integer)) else dim
            for dim in eqn.params["shape"]
        )
    elif fault == "mapped-axis":
        eqn.params["broadcast_dimensions"] = (0, 2, 3, 4)
    elif fault == "shape":
        shape = list(eqn.params["shape"])
        shape[-1] = 8
        eqn.params["shape"] = tuple(shape)
    elif fault == "dtype":
        _rebind(eqn.outvars[0], dtype=np.float64)
    elif fault == "weakness":
        _rebind(eqn.outvars[0], weak_type=True)
    elif fault == "placement":
        eqn.params["out_sharding"] = object()
    elif fault == "duplicate-axis":
        eqn.params["broadcast_dimensions"] = (1, 2, 3, 3)
    else:
        eqn.params["broadcast_dimensions"] = (True, 2, 3, 4)
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(graph, 23)
    assert typed_trace_manifest(native) == before


@pytest.mark.parametrize("operation", ["sum", "max"])
@pytest.mark.parametrize("axes", [(0,), (3,), (0, 3), (1,)])
def test_reduction_uses_actual_symbolic_axis_after_removed_axes(
    operation: str, axes: tuple[int, ...]
) -> None:
    def function(value: Any) -> Any:
        expanded = jax.lax.broadcast_in_dim(value, (1, *value.shape), (1, 2, 3))
        return (jax.lax.reduce_sum if operation == "sum" else jax.lax.reduce_max)(
            expanded, axes
        )

    (batch,) = export.symbolic_shape("B")
    native = jax.make_jaxpr(function)(jax.ShapeDtypeStruct((batch, 2, 4), np.float32))
    original = typed_trace_manifest(native)
    validate_source_jaxpr(native.jaxpr, 23)
    assert typed_trace_manifest(native) == original
    symbol = atomic_symbolic_dimension(batch)
    source = _Tensor("float32", (1, "B", 2, 4), symbol, 1)
    retained = None if 1 in axes else 1 - sum(axis < 1 for axis in axes)
    target = _Tensor(
        "float32",
        tuple(size for axis, size in enumerate(source.shape) if axis not in axes),
        None if retained is None else symbol,
        retained,
    )
    _reduction_shape(
        source, target, axes, _Site("actual-reduction", 23, "reduce_" + operation)
    )
    with pytest.raises(SourcePrecisionError):
        _reduction_shape(
            source,
            replace(target, symbol_axis=0 if retained is None or retained != 0 else 1),
            axes,
            _Site("corrupt-reduction", 23, "reduce_" + operation),
        )


@pytest.mark.parametrize("operation", ["broadcast", "reduce", "select"])
def test_zero_tail_symbolic_attention_does_not_gain_an_owner(operation: str) -> None:
    def function(value: Any) -> Any:
        expanded = jax.lax.broadcast_in_dim(value, (1, *value.shape), (1, 2, 3))
        if operation == "reduce":
            return jax.lax.reduce_sum(expanded, (3,))
        if operation == "select":
            return jnp.where(True, expanded, expanded)
        return expanded

    (batch,) = export.symbolic_shape("B")
    native = jax.make_jaxpr(function)(jax.ShapeDtypeStruct((batch, 0, 4), np.float32))
    with pytest.raises(SourcePrecisionError, match="no audited owner"):
        validate_source_jaxpr(native.jaxpr, 23)
