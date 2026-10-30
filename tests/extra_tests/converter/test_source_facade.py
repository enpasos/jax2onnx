# tests/extra_tests/converter/test_source_facade.py

"""Private SOURCE and public legacy calls share the actual export facade."""

from collections.abc import Callable, Iterator
import inspect
from pathlib import Path
from typing import Any
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx import user_interface as ui
from jax2onnx.converter import conversion_api
from jax2onnx.converter import source_structure as structure
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from tests.extra_tests.converter.source_capability_counterparts import (
    SourceSchemaObservation,
)
from tests.extra_tests.converter.source_precision_diagnostics import DiagnosticValidator


@pytest.fixture(autouse=True)
def _restore_jax_config() -> Iterator[None]:
    previous_x64 = bool(jax.config.x64_enabled)
    previous_platforms = jax.config.jax_platforms
    previous_precision = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", None)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_platforms", previous_platforms)
        jax.config.update("jax_default_matmul_precision", previous_precision)


def _source(fn: Any, inputs: Any, **kwargs: Any) -> Any:
    return ui._to_onnx(fn, inputs, precision_policy=PrecisionPolicy.SOURCE, **kwargs)


def _proto(artifact: Any) -> onnx.ModelProto:
    if isinstance(artifact, ir.Model):
        return ir.to_proto(artifact)
    if isinstance(artifact, onnx.ModelProto):
        return artifact
    return onnx.load_model(artifact)


@onnx_function
def _cancellation(x: jax.Array) -> jax.Array:
    rounded = x.astype(jnp.float32)
    return (rounded + np.float32(1)) - rounded


@onnx_function
def _formal_aliases(x: jax.Array) -> tuple[jax.Array, ...]:
    return x, x, -x


@onnx_function
def _preferred_matrix(x: jax.Array, w: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        x, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )


@onnx_function
class _Opset16ExcludedFunction:
    """A real module-level Function boundary around the unchanged leaf."""

    def __init__(self, leaf: Callable[[Any], Any]):
        self.leaf = leaf

    def __call__(self, x: Any) -> Any:
        return self.leaf(x)


@onnx_function(name="NamedDiagnosticLeaf")
def _named_bad_leaf(x: jax.Array) -> jax.Array:
    return jax.lax.cond(
        x > np.float32(0), lambda value: jnp.sin(value), lambda value: -value, x
    )


@onnx_function(name="NamedDiagnosticOuter")
def _named_bad_outer(x: jax.Array) -> jax.Array:
    return _named_bad_leaf(x)


@onnx_function(name="NamedUnauditedLeaf")
def _named_unaudited_leaf(x: jax.Array) -> jax.Array:
    return jax.lax.cond(
        x > np.float32(0), lambda value: jax.lax.erf(value), lambda value: -value, x
    )


@onnx_function(name="NamedUnauditedOuter")
def _named_unaudited_outer(x: jax.Array) -> jax.Array:
    return _named_unaudited_leaf(x)


@onnx_function
def _large_fp64_inner(x: jax.Array, *, payload: np.ndarray) -> tuple[Any, ...]:
    return x, payload


@onnx_function
def _large_fp64_outer(x: jax.Array, *, payload: np.ndarray) -> tuple[Any, ...]:
    return _large_fp64_inner(x, payload=payload)


@onnx_function
def _legacy_large_fp64_inner(x: jax.Array, *, payload: np.ndarray) -> np.ndarray:
    return payload


@onnx_function
def _legacy_large_fp64_outer(x: jax.Array, *, payload: np.ndarray) -> np.ndarray:
    return _legacy_large_fp64_inner(x, payload=payload)


def test_source_observes_the_real_trace_once_without_wrapping_the_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    traces = []
    real_trace = conversion_api._trace_to_jaxpr

    def model(x: jax.Array) -> jax.Array:
        calls.append(bool(x.weak_type))
        return -x

    def observe(**kwargs: Any) -> Any:
        traces.append(kwargs["fn"])
        return real_trace(**kwargs)

    monkeypatch.setattr(conversion_api, "_trace_to_jaxpr", observe)
    artifact = _source(model, [()])
    report = get_precision_report(artifact)
    assert calls == [False]
    assert traces == [model]
    assert report is not None
    assert report.inputs[0].supplied.dtype is None
    assert report.inputs[0].supplied.weak_type is None
    assert report.inputs[0].normalized.dtype == "float32"
    assert report.inputs[0].traced.dtype == "float32"
    assert report.inputs[0].exported.dtype == "float32"
    assert report.jax_version == jax.__version__
    assert report.exporter_version == artifact.producer_version


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize(
    "kind", ["weak-array", "weak-spec", "strong-spec", "numpy", "shape"]
)
def test_source_normalizes_once_and_preserves_actual_weak_metadata(
    ambient_x64: bool, kind: str
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)
    weak = jnp.asarray(1.0)
    supplied: Any = {
        "weak-array": weak,
        "weak-spec": jax.ShapeDtypeStruct((), weak.dtype, weak_type=True),
        "strong-spec": jax.ShapeDtypeStruct((), np.float32),
        "numpy": np.asarray(1, np.float32),
        "shape": (),
    }[kind]
    artifact = _source(lambda x: x + jnp.asarray(1, jnp.float16), [supplied])
    report = get_precision_report(artifact)
    assert report is not None
    expected_weak = kind.startswith("weak")
    properties = report.inputs[0]
    assert properties.normalized.weak_type is expected_weak
    assert properties.traced.weak_type is expected_weak
    assert properties.exported.weak_type is None
    assert report.traced_outputs[0].dtype == ("float16" if expected_weak else "float32")
    assert report.exported_outputs[0].dtype == report.traced_outputs[0].dtype
    assert report.caller_x64_enabled is ambient_x64
    assert bool(jax.config.x64_enabled) is ambient_x64


@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16, np.float32, np.float64])
@pytest.mark.parametrize("return_mode", ["ir", "proto", "file"])
def test_source_return_modes_keep_named_direct_and_mixed_outputs_and_metadata(
    dtype: Any, return_mode: str, tmp_path: Path
) -> None:
    jax.config.update("jax_enable_x64", True)
    artifact = _source(
        lambda x: (x, -x, x.astype(jnp.float32)),
        [jax.ShapeDtypeStruct((2,), dtype)],
        return_mode=return_mode,
        output_path=tmp_path / "source.onnx",
        input_names=["features"],
        output_names=["direct", "negative", "single"],
    )
    proto = _proto(artifact)
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(artifact)
    assert report is not None
    assert [value.name for value in proto.graph.input] == ["features"]
    assert [value.name for value in proto.graph.output] == [
        "direct",
        "negative",
        "single",
    ]
    expected = np.dtype(dtype).name
    assert [value.dtype for value in report.exported_outputs] == [
        expected,
        expected,
        "float32",
    ]
    assert [value.shape for value in report.exported_outputs] == [(2,), (2,), (2,)]
    assert report.inputs[0].exported.name == "features"
    assert get_precision_report(proto) == report
    checks = {check.category: check.status for check in report.checks}
    assert checks == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }


def test_source_custom_names_separate_aliases_and_preserve_shared_consumers() -> None:
    def model(x: jax.Array) -> tuple[jax.Array, ...]:
        shared = -x
        return x, shared, shared, shared + shared

    artifact = _source(
        model,
        [(2,)],
        return_mode="ir",
        input_names=["features"],
        output_names=["direct", "first", "second", "sum"],
    )
    nodes = list(artifact.graph)
    assert sum(node.op_type == "Identity" for node in nodes) == 2
    add = next(node for node in nodes if node.op_type == "Add")
    assert add.inputs[0] is add.inputs[1]
    assert add.inputs[0] is artifact.graph.outputs[1]
    assert artifact.graph.outputs[2].producer().inputs[0] is add.inputs[0]
    assert artifact.graph.inputs[0].name == "features"
    assert artifact.graph.outputs[0].producer().inputs[0] is artifact.graph.inputs[0]


def test_source_named_function_formal_aliases_pass_full_serialization() -> None:
    artifact = _source(
        lambda x: _formal_aliases(x),
        [(2,)],
        return_mode="ir",
        input_names=["features"],
        output_names=["first_formal", "second_formal", "negative"],
    )
    assert len(artifact.functions) == 1
    report = get_precision_report(artifact)
    assert report is not None
    assert [value.name for value in report.exported_outputs] == [
        "first_formal",
        "second_formal",
        "negative",
    ]
    assert len(next(iter(artifact.functions.values())).graph.outputs) == 3
    onnx.checker.check_model(ir.to_proto(artifact), full_check=True)


def test_source_output_name_preserves_unnamed_input_alias() -> None:
    artifact = _source(lambda x: x, [(2,)], output_names=["result"])
    assert artifact.graph.input[0].name == "in_0"
    assert artifact.graph.output[0].name == "result"


def test_source_report_preserves_mixed_structured_output_order(
    monkeypatch: pytest.MonkeyPatch, record_property: Callable[[str, object], None]
) -> None:
    def model(x: jax.Array) -> Any:
        return x.astype(jnp.float16), {
            "bits": jax.lax.bitcast_convert_type(x, jnp.uint32),
            "positive": x > np.float32(0),
        }

    observation = SourceSchemaObservation(ui.check_source_schema)
    original_trace = conversion_api._trace_to_jaxpr
    traces = []

    def observe_trace(*args: Any, **kwargs: Any) -> Any:
        returned = original_trace(*args, **kwargs)
        traces.append(returned)
        return returned

    not_returned = object()
    artifact = not_returned
    with monkeypatch.context() as active:
        active.setattr(ui, "check_source_schema", observation)
        active.setattr(conversion_api, "_trace_to_jaxpr", observe_trace)
        try:
            artifact = _source(
                model, [()], opset=26, output_names=["half", "bits", "positive"]
            )
        except onnx.checker.ValidationError as cause:
            diagnostic = observation.missing_schema_failure(
                cause, operator="BitCast", opset=26
            )
            if diagnostic is None:
                raise
            attempt = observation.assert_unchanged()
            assert attempt.cause is cause and artifact is not_returned
            assert len(traces) == 1
            assert [
                np.dtype(value.aval.dtype).name for value in traces[0].jaxpr.outvars
            ] == ["float16", "uint32", "bool"]
            candidate = onnx.load_model_from_string(attempt.blob)
            # The final report is part of the audited/checker candidate now.
            # A rejected candidate still never becomes a returned export.
            candidate_report = get_precision_report(candidate)
            assert candidate_report is not None
            assert [value.dtype for value in candidate_report.exported_outputs] == [
                "float16",
                "uint32",
                "bool",
            ]
            assert [value.name for value in candidate.graph.output] == [
                "half",
                "bits",
                "positive",
            ]
            assert [
                value.type.tensor_type.elem_type for value in candidate.graph.output
            ] == [
                onnx.TensorProto.FLOAT16,
                onnx.TensorProto.UINT32,
                onnx.TensorProto.BOOL,
            ]
            assert all(
                value.type.tensor_type.HasField("shape")
                and not value.type.tensor_type.shape.dim
                for value in candidate.graph.output
            )
            assert len(candidate.graph.input) == 1
            assert (
                candidate.graph.input[0].type.tensor_type.elem_type
                == onnx.TensorProto.FLOAT
            )
            assert (
                candidate.graph.input[0].type.tensor_type.HasField("shape")
                and not candidate.graph.input[0].type.tensor_type.shape.dim
            )
            (bitcast,) = [
                node for node in candidate.graph.node if node.name == diagnostic["node"]
            ]
            assert tuple(bitcast.input) == (candidate.graph.input[0].name,)
            assert tuple(bitcast.output) == ("bits",)
            assert [
                (attribute.name, attribute.type, attribute.i)
                for attribute in bitcast.attribute
            ] == [("to", onnx.AttributeProto.INT, onnx.TensorProto.UINT32)]
            record_property("source_failed_missing_schema", diagnostic)
            return
    attempt = observation.assert_checker_boundary()
    assert attempt.returned and attempt.cause is None
    assert len(traces) == 1

    report = get_precision_report(artifact)
    assert report is not None
    assert [value.dtype for value in report.traced_outputs] == [
        "float16",
        "uint32",
        "bool",
    ]
    assert [value.dtype for value in report.exported_outputs] == [
        "float16",
        "uint32",
        "bool",
    ]
    assert [value.name for value in report.exported_outputs] == [
        "half",
        "bits",
        "positive",
    ]
    observation.assert_report_attachment(artifact, report)


def test_source_saved_external_data_preserves_report_and_boundaries(
    tmp_path: Path,
) -> None:
    captured = np.arange(300_000, dtype=np.float32)
    path = _source(
        lambda x: (captured, -x),
        [(1,)],
        return_mode="file",
        output_path=tmp_path / "external.onnx",
        input_names=["x"],
        output_names=["weights", "negative"],
    )
    proto = onnx.load_model(path, load_external_data=False)
    assert (tmp_path / "external.onnx.data").is_file()
    assert any(initializer.external_data for initializer in proto.graph.initializer)
    report = get_precision_report(path)
    assert report is not None
    assert report.exported_outputs[0].shape == (300_000,)
    assert get_precision_report(onnx.load_model(path)) == report


@pytest.fixture
def _large_fp64_capture() -> Iterator[tuple[Callable[..., Any], np.ndarray]]:
    words = np.asarray(
        [
            0x3FF0000000400000,  # 1 + 2**-30, lost by FP32 rounding.
            0x0000000000000000,
            0x8000000000000000,
            0x7FF8000000000001,
            0x7FF8000000000002,
        ],
        dtype=np.uint64,
    )
    payload = np.resize(words, 150_000).view(np.float64)
    assert payload.nbytes == 1_200_000
    assert payload[0] != np.float64(np.float32(payload[0]))

    def model(x: jax.Array) -> tuple[Any, ...]:
        return payload, *_large_fp64_outer(x, payload=payload)

    with jax.enable_x64(True):
        yield model, payload


def _assert_exact_payload(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


def _large_capture_tensors(
    proto: onnx.ModelProto,
) -> tuple[onnx.TensorProto, onnx.TensorProto]:
    functions = {
        (function.domain, function.name): function for function in proto.functions
    }
    assert len(functions) == 2
    (root_call,) = [
        node for node in proto.graph.node if (node.domain, node.op_type) in functions
    ]
    outer = functions[root_call.domain, root_call.op_type]
    (inner_call,) = [
        node for node in outer.node if (node.domain, node.op_type) in functions
    ]
    inner = functions[inner_call.domain, inner_call.op_type]
    assert inner is not outer
    (constant,) = [node for node in inner.node if inner.output[1] in node.output]
    assert constant.op_type == "Constant"
    (attribute,) = [attr for attr in constant.attribute if attr.name == "value"]
    assert attribute.type == onnx.AttributeProto.TENSOR
    root = next(
        value
        for value in proto.graph.initializer
        if value.name == proto.graph.output[0].name
    )
    for tensor in (root, attribute.t):
        assert tensor.data_type == onnx.TensorProto.DOUBLE
        assert list(tensor.dims) == [150_000]
    return root, attribute.t


def _assert_large_capture_contract(proto: onnx.ModelProto) -> None:
    assert [value.name for value in proto.graph.input] == ["features"]
    assert [value.name for value in proto.graph.output] == [
        "root_weights",
        "direct",
        "nested_weights",
    ]
    assert all(
        value.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
        for value in (*proto.graph.input, *proto.graph.output)
    )
    report = get_precision_report(proto)
    assert report is not None
    assert report.profile == "source-static-v1"
    assert report.caller_x64_enabled
    assert report.inputs[0].normalized.weak_type is False
    assert report.inputs[0].exported.name == "features"
    assert [value.name for value in report.exported_outputs] == [
        "root_weights",
        "direct",
        "nested_weights",
    ]
    assert [value.dtype for value in report.exported_outputs] == ["float64"] * 3
    assert [value.shape for value in report.exported_outputs] == [
        (150_000,),
        (2,),
        (150_000,),
    ]
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }


@pytest.mark.parametrize("return_mode", ["ir", "proto", "file"])
def test_source_large_fp64_root_and_nested_captures_keep_exact_serialized_payload(
    return_mode: str,
    _large_fp64_capture: tuple[Callable[..., Any], np.ndarray],
    tmp_path: Path,
) -> None:
    model, payload = _large_fp64_capture
    path = tmp_path / "large.onnx"
    artifact = _source(
        model,
        [jax.ShapeDtypeStruct((2,), np.float64)],
        return_mode=return_mode,
        output_path=path,
        input_names=["features"],
        output_names=["root_weights", "direct", "nested_weights"],
    )
    report = get_precision_report(artifact)
    assert report is not None
    if return_mode == "file":
        assert Path(artifact) == path
        unresolved = onnx.load_model(path, load_external_data=False)
        root, nested = _large_capture_tensors(unresolved)
        assert root.data_location == onnx.TensorProto.EXTERNAL
        assert not root.raw_data
        assert (
            dict((entry.key, entry.value) for entry in root.external_data)["location"]
            == path.name + ".data"
        )
        assert (tmp_path / (path.name + ".data")).stat().st_size > 0
        assert nested.data_location == onnx.TensorProto.DEFAULT
        assert not nested.external_data
        _assert_exact_payload(onnx.numpy_helper.to_array(nested), payload)
        _assert_large_capture_contract(unresolved)
        assert get_precision_report(unresolved) == report
    elif return_mode == "ir":
        root = artifact.graph.initializers[artifact.graph.outputs[0].name]
        _assert_exact_payload(root.const_value.numpy(), payload)
        inner = next(
            function
            for function in artifact.functions.values()
            if function.name == "_large_fp64_inner"
        )
        constant = inner.graph.outputs[1].producer()
        assert constant.op_type == "Constant"
        _assert_exact_payload(constant.attributes["value"].as_tensor().numpy(), payload)

    proto = _proto(artifact)
    onnx.checker.check_model(proto, full_check=True)
    _assert_large_capture_contract(proto)
    assert get_precision_report(proto) == report
    for tensor in _large_capture_tensors(proto):
        assert tensor.data_location == onnx.TensorProto.DEFAULT
        assert not tensor.external_data
        _assert_exact_payload(onnx.numpy_helper.to_array(tensor), payload)


def test_public_legacy_file_keeps_large_fp64_function_capture_inline(
    _large_fp64_capture: tuple[Callable[..., Any], np.ndarray], tmp_path: Path
) -> None:
    _, payload = _large_fp64_capture

    def model(x: jax.Array) -> tuple[Any, ...]:
        return payload, x, _legacy_large_fp64_outer(x, payload=payload)

    def tensors(proto: onnx.ModelProto) -> tuple[onnx.TensorProto, onnx.TensorProto]:
        functions = {
            (function.domain, function.name): function for function in proto.functions
        }
        assert len(functions) == 2
        (root_call,) = [
            node
            for node in proto.graph.node
            if (node.domain, node.op_type) in functions
        ]
        outer = functions[root_call.domain, root_call.op_type]
        (inner_call,) = [
            node for node in outer.node if (node.domain, node.op_type) in functions
        ]
        inner = functions[inner_call.domain, inner_call.op_type]
        assert inner is not outer
        assert len(inner.output) == len(outer.output) == 1
        (constant,) = [node for node in inner.node if inner.output[0] in node.output]
        assert constant.op_type == "Constant"
        (attribute,) = [attr for attr in constant.attribute if attr.name == "value"]
        assert attribute.type == onnx.AttributeProto.TENSOR
        root = next(
            value
            for value in proto.graph.initializer
            if value.name == proto.graph.output[0].name
        )
        for tensor in (root, attribute.t):
            assert tensor.data_type == onnx.TensorProto.DOUBLE
            assert list(tensor.dims) == [150_000]
        return root, attribute.t

    path = tmp_path / "legacy.onnx"
    artifact = to_onnx(
        model,
        [jax.ShapeDtypeStruct((2,), np.float64)],
        enable_double_precision=True,
        return_mode="file",
        output_path=path,
        input_names=["direct"],
        output_names=["root_weights", "direct", "nested_weights"],
    )
    assert Path(artifact) == path
    unresolved = onnx.load_model(path, load_external_data=False)
    root, nested = tensors(unresolved)
    assert root.data_location == onnx.TensorProto.EXTERNAL
    assert not root.raw_data
    assert (
        dict((entry.key, entry.value) for entry in root.external_data)["location"]
        == path.name + ".data"
    )
    assert (tmp_path / (path.name + ".data")).stat().st_size > 0
    assert nested.data_location == onnx.TensorProto.DEFAULT
    assert not nested.external_data
    _assert_exact_payload(onnx.numpy_helper.to_array(nested), payload)
    assert [value.name for value in unresolved.graph.input] == ["direct"]
    assert [value.name for value in unresolved.graph.output] == [
        "root_weights",
        "direct",
        "nested_weights",
    ]
    assert all(
        value.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
        for value in (*unresolved.graph.input, *unresolved.graph.output)
    )
    loaded = onnx.load_model(path)
    onnx.checker.check_model(loaded, full_check=True)
    for tensor in tensors(loaded):
        assert tensor.data_location == onnx.TensorProto.DEFAULT
        assert not tensor.external_data
        _assert_exact_payload(onnx.numpy_helper.to_array(tensor), payload)


def test_source_web_large_fp64_captures_replace_real_external_sidecar(
    _large_fp64_capture: tuple[Callable[..., Any], np.ndarray], tmp_path: Path
) -> None:
    model, payload = _large_fp64_capture
    path = tmp_path / "large.onnx"
    options = dict(
        return_mode="file",
        output_path=path,
        input_names=["features"],
        output_names=["root_weights", "direct", "nested_weights"],
    )
    standard = _source(model, [jax.ShapeDtypeStruct((2,), np.float64)], **options)
    sidecar = tmp_path / (path.name + ".data")
    assert sidecar.stat().st_size > 0
    root, _ = _large_capture_tensors(
        onnx.load_model(standard, load_external_data=False)
    )
    assert root.data_location == onnx.TensorProto.EXTERNAL
    report = get_precision_report(standard)
    web = _source(
        model, [jax.ShapeDtypeStruct((2,), np.float64)], export_mode="web", **options
    )
    assert Path(web) == path
    assert not sidecar.exists()
    assert list(tmp_path.iterdir()) == [path]
    proto = onnx.load_model(web, load_external_data=False)
    onnx.checker.check_model(proto, full_check=True)
    _assert_large_capture_contract(proto)
    assert get_precision_report(proto) == report
    for tensor in _large_capture_tensors(proto):
        assert tensor.data_location == onnx.TensorProto.DEFAULT
        assert not tensor.external_data
        _assert_exact_payload(onnx.numpy_helper.to_array(tensor), payload)


def test_source_diagnostic_web_keeps_lowering_unverified(tmp_path: Path) -> None:
    observer = DiagnosticValidator()
    path = _source(
        lambda x: -x,
        [jax.ShapeDtypeStruct((2,), np.float32)],
        diagnostic_jaxpr_validator=observer,
        export_mode="web",
        return_mode="file",
        output_path=tmp_path / "diagnostic.onnx",
    )
    proto = onnx.load_model(path, load_external_data=False)
    onnx.checker.check_model(proto, full_check=True)
    assert observer.scopes
    report = get_precision_report(proto)
    assert report is not None
    assert report.profile == "source-diagnostic-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.UNVERIFIED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }


@pytest.mark.parametrize("return_mode", ["ir", "proto"])
def test_source_web_option_is_accepted_for_nonfile_return_modes(
    return_mode: str, tmp_path: Path
) -> None:
    artifact = _source(
        lambda x: -x,
        [jax.ShapeDtypeStruct((2,), np.float32)],
        return_mode=return_mode,
        export_mode="web",
        output_path=tmp_path / "unused.onnx",
    )
    assert isinstance(artifact, ir.Model if return_mode == "ir" else onnx.ModelProto)
    proto = _proto(artifact)
    assert [node.op_type for node in proto.graph.node] == ["Neg"]
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-static-v1"
    assert not list(tmp_path.iterdir())


def test_source_records_real_truncation_warning_and_keeps_it_visible() -> None:
    jax.config.update("jax_enable_x64", False)
    with pytest.warns(UserWarning, match="float64") as observed:
        artifact = _source(lambda x: x + jnp.asarray(1, dtype=jnp.float64), [()])
    report = get_precision_report(artifact)
    assert report is not None
    assert len(observed) == 1
    assert len(report.warnings) == 1
    assert report.warnings[0].category == "UserWarning"
    assert "float64" in report.warnings[0].message
    assert not jax.config.x64_enabled


@pytest.mark.parametrize("ambient_x64", [False, True])
def test_failed_source_trace_replays_warnings_and_restores_contexts(
    ambient_x64: bool,
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)

    def fail(x: jax.Array) -> jax.Array:
        warnings.warn("model trace warning", UserWarning)
        raise RuntimeError("model trace failed")

    with pytest.warns(UserWarning, match="model trace warning") as observed:
        with pytest.raises(RuntimeError, match="model trace failed"):
            _source(fail, [()])
    assert len(observed) == 1
    assert bool(jax.config.x64_enabled) is ambient_x64
    assert active_precision_policy() is PrecisionPolicy.LEGACY


@pytest.mark.parametrize("return_mode", ["proto", "ir", "file"])
def test_private_structure_collector_keeps_exact_facade_parity(
    return_mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def model(x: jax.Array, y: jax.Array) -> jax.Array:
        return jax.lax.sub(x, y)

    actual = ui.validate_source_structure
    results = []
    observed = []
    validations = []

    def validate(final_ir: ir.Model) -> Any:
        result = actual(final_ir)
        validations.append((final_ir, result))
        return result

    def collect(trace: Any, final_ir: ir.Model, result: Any, returned: Any) -> None:
        assert validations[-1][0] is final_ir
        assert validations[-1][1] == result
        assert trace.closed_jaxpr is not None
        assert get_precision_report(final_ir) == get_precision_report(returned)
        assert get_precision_report(returned) is not None
        observed.append(returned)

    monkeypatch.setattr(ui, "validate_source_structure", validate)
    path = tmp_path / "same-destination.onnx"
    for enabled in (False, True):
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            returned = _source(
                model,
                [jax.ShapeDtypeStruct((2,), np.float32)] * 2,
                return_mode=return_mode,
                output_path=path,
                source_structure_collector=collect if enabled else None,
            )
        results.append(
            (
                type(returned),
                _proto(returned).SerializeToString(deterministic=True),
                get_precision_report(returned),
                tuple((entry.category, str(entry.message)) for entry in emitted),
            )
        )
        if enabled:
            assert observed == [returned]
    assert len(validations) == (6 if return_mode == "ir" else 4)
    assert results[0] == results[1]
    assert (
        results[1][0]
        is {
            "proto": onnx.ModelProto,
            "ir": ir.Model,
            "file": str,
        }[return_mode]
    )
    if return_mode == "file":
        assert observed[0] == str(path)


def test_private_structure_collector_preserves_real_warning_and_report() -> None:
    jax.config.update("jax_enable_x64", False)
    results = []
    observed = []
    for enabled in (False, True):
        # Fresh callable identities make both real traces emit the same actual
        # truncation warning, without relying on a prior JAX trace cache.
        def model(x: jax.Array) -> jax.Array:
            return x + jnp.asarray(1, dtype=jnp.float64)

        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            returned = _source(
                model,
                [()],
                source_structure_collector=(
                    (lambda *args: observed.append(args[2])) if enabled else None
                ),
            )
        results.append(
            (
                returned.SerializeToString(deterministic=True),
                get_precision_report(returned),
                tuple((entry.category, str(entry.message)) for entry in emitted),
            )
        )
    assert results[0] == results[1]
    assert len(results[0][2]) == 1 and "float64" in results[0][2][0][1]
    assert len(observed) == 1


@pytest.mark.parametrize("return_mode", ["proto", "ir", "file"])
def test_private_structure_collector_failure_cannot_fail_export(
    return_mode: str, tmp_path: Path
) -> None:
    called = []

    def broken(*args: Any) -> None:
        called.append(True)
        raise OSError("injected private evidence failure")

    artifact = _source(
        lambda x: -x,
        [jax.ShapeDtypeStruct((2,), np.float32)],
        return_mode=return_mode,
        output_path=tmp_path / "returned.onnx",
        source_structure_collector=broken,
    )
    assert called == [True]
    onnx.checker.check_model(_proto(artifact), full_check=True)
    assert get_precision_report(artifact).profile == "source-static-v1"


@pytest.mark.parametrize("fault_at", [1, 2])
@pytest.mark.parametrize("mutation", ["operator", "constant"])
def test_source_serialization_rejects_one_fault_in_either_official_codec_call(
    fault_at: int, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = ir.to_proto
    calls = []
    captured = np.asarray([1, -0.0], np.float32)

    def serialize(model: ir.Model, *args: Any, **kwargs: Any) -> Any:
        proto = actual(model, *args, **kwargs)
        if isinstance(model, ir.Model) and get_precision_report(model) is not None:
            calls.append(model)
            if len(calls) == fault_at:
                if mutation == "operator":
                    node = next(
                        node for node in proto.graph.node if node.op_type == "Sub"
                    )
                    node.op_type = "Add"
                else:
                    tensor = proto.graph.initializer[0]
                    raw = bytearray(tensor.raw_data)
                    raw[0] ^= 1
                    tensor.raw_data = bytes(raw)
        return proto

    monkeypatch.setattr(ir, "to_proto", serialize)
    with pytest.raises(RuntimeError, match="serialization integrity: protobuf differs"):
        _source(lambda x: x - captured, [(2,)])
    assert len(calls) == 2
    np.testing.assert_array_equal(captured.view(np.uint32), [0x3F800000, 0x80000000])


@pytest.mark.parametrize("mutation", ["edge", "constant"])
def test_source_rechecks_frozen_ledger_after_reference_serialization(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = ir.to_proto
    actual_audit = ui.validate_source_structure
    audited = []
    calls = []
    captured = np.asarray([1, 2], np.float32)

    def audit(model: ir.Model) -> Any:
        if not audited:
            audited.append(model)
        return actual_audit(model)

    def serialize(model: ir.Model, *args: Any, **kwargs: Any) -> Any:
        if isinstance(model, ir.Model) and get_precision_report(model) is not None:
            calls.append(model)
            if len(calls) == 1:
                original = audited[0]
                assert original.meta[structure.RESULT_KEY].status == "validated"
                if mutation == "edge":
                    node = next(
                        node for node in original.graph if node.op_type == "Sub"
                    )
                    node.replace_input_with(1, original.graph.inputs[0])
                else:
                    value = next(iter(original.graph.initializers.values()))
                    value.const_value = ir.tensor(np.asarray([3, 4], np.float32))
        return actual(model, *args, **kwargs)

    monkeypatch.setattr(ir, "to_proto", serialize)
    monkeypatch.setattr(ui, "validate_source_structure", audit)
    with pytest.raises(RuntimeError, match="serialization integrity: IR changed"):
        _source(lambda x: x - captured, [(2,)])
    assert len(calls) == 1
    np.testing.assert_array_equal(captured, [1, 2])


@pytest.mark.parametrize("mutation", ["operator", "edge"])
def test_source_audit_capture_binds_a_real_missing_rule_region(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_audit = ui.validate_source_structure
    actual_serialize = ir.to_proto
    audited = []
    outcomes = []
    calls = []

    def audit(model: ir.Model) -> Any:
        if not audited:
            audited.append(model)
        outcome = actual_audit(model)
        outcomes.append(outcome)
        return outcome

    def serialize(model: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(model, ir.Model) and get_precision_report(model) is not None:
            calls.append(model)
            if len(calls) == 1:
                original = audited[0]
                # Empty Add is represented by an Identity, and its missing rule
                # still leaves the semantic observation UNVERIFIED.
                node = next(
                    node for node in original.graph if node.op_type == "Identity"
                )
                if mutation == "operator":
                    node.op_type = "Neg"
                else:
                    node.replace_input_with(0, original.graph.inputs[1])
        return actual_serialize(model, *args, **kwargs)

    monkeypatch.setattr(ui, "validate_source_structure", audit)
    monkeypatch.setattr(ir, "to_proto", serialize)
    with pytest.raises(RuntimeError, match="serialization integrity: protobuf differs"):
        _source(jax.lax.add, [(0, 3), (0, 3)])
    assert len(calls) == 2
    assert outcomes[0] == outcomes[1]
    assert outcomes[0].status == "unverified"
    assert "equation-0:rule-unavailable" in outcomes[0].reasons


@pytest.mark.parametrize("status", ["unverified", "failed"])
@pytest.mark.parametrize(
    "mutation",
    ["none", "operator", "type", "shape", "attribute", "payload", "quantization"],
)
def test_source_audit_capture_binds_shared_state_despite_an_unchanged_audit_result(
    status: str,
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_freeze = ui._freeze_source_ir
    actual_audit = ui.validate_source_structure
    actual_serialize = ir.to_proto
    audited = []
    outcomes = []
    calls = []
    captured = np.asarray([[1, 2], [3, 4]], np.float32)

    def freeze(model: ir.Model) -> Any:
        if mutation == "payload":
            value = next(iter(model.graph.initializers.values()))
            # Own a writable backing array for the fault, without modifying the
            # caller's captured array or depending on JAX array writeability.
            value.const_value = ir.tensor(
                np.array(value.const_value.numpy(), copy=True), name=value.name
            )
        model.graph.inputs[0].meta["quant_parameter_tensor_names"] = {
            "scale": "original"
        }
        return actual_freeze(model)

    def audit(model: ir.Model) -> Any:
        if not audited:
            audited.append(model)
            ledger = model.meta[structure._LEDGER]
            if status == "unverified":
                ledger.gaps.add("injected-existing-gap")
            else:
                ledger.failures.add("injected-existing-failure")
        outcome = actual_audit(model)
        outcomes.append(outcome)
        return outcome

    def serialize(model: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(model, ir.Model) and get_precision_report(model) is not None:
            calls.append(model)
            if len(calls) == 1:
                original = audited[0]
                if mutation == "operator":
                    node = next(
                        node for node in original.graph if node.op_type == "Add"
                    )
                    node.op_type = "Sub"
                elif mutation == "type":
                    original.graph.inputs[0].type.dtype = ir.DataType.DOUBLE
                elif mutation == "shape":
                    original.graph.inputs[0].shape[0] = 3
                elif mutation == "attribute":
                    node = next(
                        node for node in original.graph if node.op_type == "Transpose"
                    )
                    node.attributes["perm"]._value = (0, 1)
                elif mutation == "payload":
                    value = next(iter(original.graph.initializers.values()))
                    tensor = value.const_value
                    tensor.numpy().view(np.uint32).flat[0] ^= np.uint32(1)
                elif mutation == "quantization":
                    annotation = original.graph.inputs[0].meta[
                        "quant_parameter_tensor_names"
                    ]
                    annotation["scale"] = "changed"
        return actual_serialize(model, *args, **kwargs)

    monkeypatch.setattr(ui, "_freeze_source_ir", freeze)
    monkeypatch.setattr(ui, "validate_source_structure", audit)
    monkeypatch.setattr(ir, "to_proto", serialize)

    def export() -> Any:
        return _source(lambda x: jax.lax.transpose(x + captured, (1, 0)), [(2, 2)])

    if mutation == "none":
        assert get_precision_report(export()) is not None
    else:
        with pytest.raises(RuntimeError, match="SOURCE serialization integrity"):
            export()
    assert outcomes[0].status == status
    if len(outcomes) > 1:
        assert outcomes[0] == outcomes[1]
    np.testing.assert_array_equal(captured, [[1, 2], [3, 4]])


@pytest.mark.parametrize("return_mode", ["proto", "ir", "file"])
def test_source_reference_allows_official_shared_initializer_name_assignment(
    return_mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_freeze = ui._freeze_source_ir
    captured = np.asarray([1, -0.0], np.float32)
    shared = []

    def freeze(model: ir.Model) -> Any:
        first = next(iter(model.graph.initializers.values()))
        tensor = first.const_value
        tensor.name = "captured_tensor_name"
        second = ir.Value(
            name="second_weight",
            type=first.type,
            shape=first.shape,
            const_value=tensor,
        )
        model.graph.initializers.add(second)
        node = ir.Node(
            "",
            "Constant",
            [],
            [ir.AttrTensor("value", tensor)],
            num_outputs=1,
        )
        node.outputs[0].name = "shared_attribute"
        model.graph.append(node)
        shared.append(tensor)
        return actual_freeze(model)

    monkeypatch.setattr(ui, "_freeze_source_ir", freeze)
    returned = _source(
        lambda x: x - captured,
        [(2,)],
        return_mode=return_mode,
        output_path=tmp_path / "shared.onnx",
    )
    proto = _proto(returned)
    assert len(proto.graph.initializer) == 2
    assert proto.graph.initializer[1].name == "second_weight"
    node = next(
        node for node in proto.graph.node if tuple(node.output) == ("shared_attribute",)
    )
    assert node.attribute[0].t.name == "second_weight"
    # Actual serialization retains the established official name normalization.
    assert shared[0].name == "second_weight"
    np.testing.assert_array_equal(captured.view(np.uint32), [0x3F800000, 0x80000000])


def _install_source_late_initializer_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> list[ir.TensorProtocol]:
    actual_freeze = ui._freeze_source_ir
    shared = []

    def freeze(model: ir.Model) -> Any:
        tensor = ir.tensor(np.asarray([1, -0.0], np.float32), name="old")
        shared.append(tensor)
        constant = ir.Node(
            "", "Constant", [], [ir.AttrTensor("value", tensor)], num_outputs=1
        )
        constant.outputs[0].name = "alias_root"
        model.graph.append(constant)

        def branch(name: str) -> ir.Graph:
            value = ir.Value(
                name="later",
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape([2]),
                const_value=tensor,
            )
            return ir.Graph([], [value], nodes=[], initializers=[value], name=name)

        condition = ir.Value(
            name="alias_condition",
            type=ir.TensorType(ir.DataType.BOOL),
            shape=ir.Shape([]),
            const_value=ir.tensor(np.asarray(True), name="alias_condition"),
        )
        model.graph.initializers.add(condition)
        selection = ir.Node(
            "",
            "If",
            [condition],
            [
                ir.AttrGraph("then_branch", branch("alias_then")),
                ir.AttrGraph("else_branch", branch("alias_else")),
            ],
            num_outputs=1,
        )
        selection.outputs[0].name = "alias_selected"
        selection.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        selection.outputs[0].shape = ir.Shape([2])
        model.graph.append(selection)
        return actual_freeze(model)

    monkeypatch.setattr(ui, "_freeze_source_ir", freeze)
    return shared


@pytest.mark.parametrize("return_mode", ["proto", "ir", "file"])
def test_source_reference_covers_names_normalized_by_a_later_nested_initializer(
    return_mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _install_source_late_initializer_alias(monkeypatch)
    actual_serialize = ir.to_proto
    calls = []

    def serialize(model: Any, *args: Any, **kwargs: Any) -> Any:
        proto = actual_serialize(model, *args, **kwargs)
        if isinstance(model, ir.Model) and get_precision_report(model) is not None:
            node = next(
                node
                for node in proto.graph.node
                if tuple(node.output) == ("alias_root",)
            )
            calls.append((id(model), node.attribute[0].t.name))
        return proto

    monkeypatch.setattr(ir, "to_proto", serialize)
    returned = _source(
        lambda x: -x,
        [(2,)],
        return_mode=return_mode,
        output_path=tmp_path / "late-alias.onnx",
    )
    if return_mode == "ir":
        assert [name for _, name in calls] == ["old", "later", "old", "later"]
        assert calls[0][0] == calls[1][0]  # Both references use the same clone.
        assert calls[2][0] == calls[3][0] == id(returned)
    else:
        assert [name for _, name in calls] == ["old", "old"]
    assert shared[0].name == "later"
    proto = _proto(returned)
    onnx.checker.check_model(proto, full_check=True)
    node = next(
        node for node in proto.graph.node if tuple(node.output) == ("alias_root",)
    )
    assert node.attribute[0].t.name == ("later" if return_mode == "ir" else "old")


@pytest.mark.parametrize("return_mode", ["proto", "ir", "file"])
@pytest.mark.parametrize("mutation", ["name", "attribute", "payload", "graph"])
def test_source_late_initializer_alias_keeps_ir_mutations_bound_to_the_return(
    return_mode: str,
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _install_source_late_initializer_alias(monkeypatch)
    observed = []

    def mutate(trace: Any, final_ir: ir.Model, result: Any, returned: Any) -> None:
        observed.append(result.status)
        if mutation == "name":
            shared[0].name = "injected_name"
        elif mutation == "attribute":
            node = next(
                node for node in final_ir.graph if node.outputs[0].name == "alias_root"
            )
            node.attributes["value"].doc_string = "injected attribute"
        elif mutation == "payload":
            shared[0].numpy()[0] = np.float32(3)
        else:
            node = next(node for node in final_ir.graph if node.op_type == "Neg")
            node.op_type = "Abs"

    def export() -> Any:
        return _source(
            lambda x: -x,
            [(2,)],
            return_mode=return_mode,
            output_path=tmp_path / "late-alias.onnx",
            source_structure_collector=mutate,
        )

    if return_mode == "ir":
        with pytest.raises(RuntimeError, match="SOURCE serialization integrity"):
            export()
    else:
        returned = export()
        proto = _proto(returned)
        node = next(
            node for node in proto.graph.node if tuple(node.output) == ("alias_root",)
        )
        assert node.attribute[0].t.name == "old"
        assert not node.attribute[0].doc_string
        np.testing.assert_array_equal(
            onnx.numpy_helper.to_array(node.attribute[0].t).view(np.uint32),
            [0x3F800000, 0x80000000],
        )
        assert any(node.op_type == "Neg" for node in proto.graph.node)
    assert observed == ["unverified"]


@pytest.mark.parametrize("mutation", ["name", "payload"])
def test_source_reference_never_restores_attribute_tensor_names_or_payloads(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_freeze = ui._freeze_source_ir
    actual_serialize = ir.to_proto
    tensors = []
    calls = []

    def freeze(model: ir.Model) -> Any:
        tensor = ir.tensor(np.asarray([1, 2], np.float32), name="attribute_only")
        node = ir.Node(
            "", "Constant", [], [ir.AttrTensor("value", tensor)], num_outputs=1
        )
        node.outputs[0].name = "attribute_only_value"
        model.graph.append(node)
        tensors.append(tensor)
        return actual_freeze(model)

    def serialize(model: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(model, ir.Model) and get_precision_report(model) is not None:
            calls.append(model)
            if len(calls) == 1:
                if mutation == "name":
                    tensors[0].name = "injected_name"
                else:
                    tensors[0].numpy()[0] = np.float32(3)
        return actual_serialize(model, *args, **kwargs)

    monkeypatch.setattr(ui, "_freeze_source_ir", freeze)
    monkeypatch.setattr(ir, "to_proto", serialize)
    with pytest.raises(RuntimeError, match="shared tensor changed after audit capture"):
        _source(lambda x: -x, [(2,)])
    assert len(calls) == 1


def test_source_reference_restores_initializer_names_when_the_codec_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_freeze = ui._freeze_source_ir
    actual_serialize = ir.to_proto
    captured = np.asarray([1, 2], np.float32)
    tensors = []

    def freeze(model: ir.Model) -> Any:
        tensor = next(iter(model.graph.initializers.values())).const_value
        tensor.name = "before_reference"
        tensors.append(tensor)
        return actual_freeze(model)

    def serialize(model: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(model, ir.Model) and get_precision_report(model) is not None:
            tensors[0].name = "changed_before_error"
            raise LookupError("injected reference codec failure")
        return actual_serialize(model, *args, **kwargs)

    monkeypatch.setattr(ui, "_freeze_source_ir", freeze)
    monkeypatch.setattr(ir, "to_proto", serialize)
    with pytest.raises(LookupError, match="injected reference codec failure"):
        _source(lambda x: x - captured, [(2,)])
    assert tensors[0].name == "before_reference"


def test_source_reference_name_restoration_failure_cannot_become_export_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(guards: Any) -> None:
        raise OSError("injected reference name restoration failure")

    monkeypatch.setattr(ui, "_restore_source_initializer_names", broken)
    with pytest.raises(OSError, match="injected reference name restoration failure"):
        _source(lambda x: -x, [(2,)])


def test_source_serialization_checkpoint_retains_the_official_codec_trust_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Diagnostic of this bounded checkpoint, not SOURCE numerical acceptance:
    # the same permanent error in both official-codec calls can evade comparison.
    actual = ir.to_proto
    calls = []

    def serialize(model: ir.Model, *args: Any, **kwargs: Any) -> Any:
        proto = actual(model, *args, **kwargs)
        if isinstance(model, ir.Model) and get_precision_report(model) is not None:
            calls.append(model)
            node = next(node for node in proto.graph.node if node.op_type == "Neg")
            node.op_type = "Abs"
        return proto

    monkeypatch.setattr(ir, "to_proto", serialize)
    proto = _source(lambda x: -x, [(2,)])
    assert len(calls) == 2
    assert any(node.op_type == "Abs" for node in proto.graph.node)
    assert get_precision_report(proto).checks[-1].status is CheckStatus.UNVERIFIED


@pytest.mark.parametrize("return_mode", ["proto", "ir", "file"])
def test_source_rejects_collector_mutation_of_the_actual_return(
    return_mode: str,
    tmp_path: Path,
) -> None:
    def mutate(trace: Any, final_ir: ir.Model, result: Any, returned: Any) -> None:
        if isinstance(returned, ir.Model):
            node = next(node for node in returned.graph if node.op_type == "Neg")
            node.op_type = "Abs"
        elif isinstance(returned, onnx.ModelProto):
            node = next(node for node in returned.graph.node if node.op_type == "Neg")
            node.op_type = "Abs"
        else:
            proto = onnx.load_model(returned, load_external_data=False)
            node = next(node for node in proto.graph.node if node.op_type == "Neg")
            node.op_type = "Abs"
            onnx.save_model(proto, returned)
        raise OSError("observer mutation followed by an observational exception")

    with pytest.raises(RuntimeError, match="SOURCE serialization integrity"):
        _source(
            lambda x: -x,
            [(2,)],
            return_mode=return_mode,
            output_path=tmp_path / "mutated.onnx",
            source_structure_collector=mutate,
        )


@pytest.mark.parametrize("return_mode", ["proto", "file"])
def test_source_collector_mutation_of_unreturned_ir_is_only_observational(
    return_mode: str,
    tmp_path: Path,
) -> None:
    def mutate(trace: Any, final_ir: ir.Model, result: Any, returned: Any) -> None:
        next(node for node in final_ir.graph if node.op_type == "Neg").op_type = "Abs"

    returned = _source(
        lambda x: -x,
        [(2,)],
        return_mode=return_mode,
        output_path=tmp_path / "correct.onnx",
        source_structure_collector=mutate,
    )
    assert any(node.op_type == "Neg" for node in _proto(returned).graph.node)


@pytest.mark.parametrize("mutation", ["bytes", "missing", "rewire", "offset"])
def test_source_rechecks_external_payload_and_packaging_after_collector(
    mutation: str,
    tmp_path: Path,
) -> None:
    captured = np.arange(300_000, dtype=np.float32)

    def mutate(trace: Any, final_ir: ir.Model, result: Any, returned: str) -> None:
        proto = onnx.load_model(returned, load_external_data=False)
        tensor = next(
            tensor for tensor in proto.graph.initializer if tensor.external_data
        )
        values = {entry.key: entry.value for entry in tensor.external_data}
        payload = Path(returned).parent / values["location"]
        if mutation == "missing":
            payload.unlink()
        elif mutation == "bytes":
            with payload.open("r+b") as stream:
                stream.write(b"changed!")
        elif mutation == "rewire":
            replacement = payload.with_name("same-bytes.data")
            replacement.write_bytes(payload.read_bytes())
            entry = next(
                entry for entry in tensor.external_data if entry.key == "location"
            )
            entry.value = replacement.name
            onnx.save_model(proto, returned)
        else:
            original = payload.read_bytes()
            payload.write_bytes(b"padding!" + original)
            entry = next(
                entry for entry in tensor.external_data if entry.key == "offset"
            )
            entry.value = str(int(values.get("offset", "0")) + 8)
            onnx.save_model(proto, returned)

    with pytest.raises((RuntimeError, ValueError, OSError)):
        _source(
            lambda x: (captured, -x),
            [(1,)],
            return_mode="file",
            output_path=tmp_path / "external.onnx",
            source_structure_collector=mutate,
        )


def test_source_file_export_accepts_stable_explicit_parent_symlinks(
    tmp_path: Path,
) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    captured = np.arange(300_000, dtype=np.float32)
    returned = _source(
        lambda x: (captured, -x),
        [(1,)],
        return_mode="file",
        output_path=alias / "source.onnx",
    )
    assert returned == str(alias / "source.onnx")
    assert (actual / "source.onnx.data").is_file()
    assert get_precision_report(returned) is not None


def test_source_missing_semantic_rule_stays_unverified_without_a_collector() -> None:
    model = _source(jax.lax.add, [(0, 3), (0, 3)], return_mode="ir")
    observed = model.meta[structure.RESULT_KEY]
    assert observed.status == "unverified"
    assert "equation-0:rule-unavailable" in observed.reasons
    assert get_precision_report(model).checks[-1].status is CheckStatus.UNVERIFIED


def test_structure_collector_is_private_source_success_only() -> None:
    assert "source_structure_collector" not in inspect.signature(to_onnx).parameters
    observed = []

    def collector(*args: Any) -> None:
        observed.append(args)

    ui._to_onnx(
        lambda x: -x,
        [(2,)],
        precision_policy=PrecisionPolicy.LEGACY,
        source_structure_collector=collector,
    )
    _source(
        lambda x: -x,
        [(2,)],
        diagnostic_jaxpr_validator=DiagnosticValidator(),
        source_structure_collector=collector,
    )

    def failing_model(x: jax.Array) -> jax.Array:
        raise ValueError("injected original trace failure")

    with pytest.raises(ValueError, match="original trace failure"):
        _source(failing_model, [(2,)], source_structure_collector=collector)
    assert observed == []


def test_source_fp64_input_disabled_rejects_before_tracing() -> None:
    jax.config.update("jax_enable_x64", False)
    calls = []

    def model(x: jax.Array) -> jax.Array:
        calls.append(True)
        return -x

    with pytest.raises(SourcePrecisionError, match="x64 is disabled"):
        _source(model, [jax.ShapeDtypeStruct((), np.float64)])
    assert not calls
    assert not jax.config.x64_enabled


@pytest.mark.parametrize("dimension", [True, -1, "B"])
def test_source_rejects_invalid_or_pending_raw_shape_before_tracing(
    dimension: Any,
) -> None:
    called = []

    def model(x: jax.Array) -> jax.Array:
        called.append(True)
        return -x

    if dimension == "B":
        # Retain this original parameter ID as the rank-one positive replacement.
        artifact = _source(model, [(dimension,)])
        report = get_precision_report(artifact)
        assert report is not None
        assert (
            report.inputs[0].traced.shape == report.exported_outputs[0].shape == ("B",)
        )
        assert called == [True]
    else:
        with pytest.raises(SourcePrecisionError, match="static nonnegative integer"):
            _source(model, [(dimension,)])
        assert not called


def test_source_metadata_serializes_numpy_integer_shape_dimensions() -> None:
    artifact = _source(lambda x: -x, [(np.int64(2),)])
    report = get_precision_report(artifact)
    assert report is not None
    assert report.inputs[0].supplied.shape == (2,)


@pytest.mark.parametrize("corruption", ["input-shape", "output-shape", "output-dtype"])
def test_source_final_oracle_detects_postprocessing_defects(
    corruption: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    postprocess = ui.postprocess_ir_model

    def corrupt(model: ir.Model, **kwargs: Any) -> None:
        postprocess(model, **kwargs)
        if corruption == "input-shape":
            model.graph.inputs[0].shape = ir.Shape((3,))
        elif corruption == "output-shape":
            model.graph.outputs[0].shape = None
        else:
            model.graph.outputs[0].type = ir.TensorType(ir.DataType.DOUBLE)

    monkeypatch.setattr(ui, "postprocess_ir_model", corrupt)
    with pytest.raises(SourcePrecisionError) as captured:
        _source(lambda x: -x, [(2,)], return_mode="ir")
    assert captured.value.category == "lowering-defect"
    assert captured.value.phase == "finalization"


def test_source_ir_return_runs_full_checker_before_claiming_representation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postprocess = ui.postprocess_ir_model

    def corrupt(model: ir.Model, **kwargs: Any) -> None:
        postprocess(model, **kwargs)
        next(iter(model.graph)).op_type = "UnknownSourceOperator"

    monkeypatch.setattr(ui, "postprocess_ir_model", corrupt)
    with pytest.raises(onnx.checker.ValidationError):
        _source(lambda x: -x, [(2,)], return_mode="ir")


def test_source_checks_serialized_boundary_shape_before_attaching_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Retain the regression identity: the final report is now present already,
    # and corrupt serialized boundaries must still prevent any returned export.
    serialize = ir.to_proto

    def corrupt(model: Any) -> Any:
        proto = serialize(model)
        if isinstance(model, ir.Model):
            proto.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 3
        return proto

    monkeypatch.setattr(ir, "to_proto", corrupt)
    with pytest.raises(SourcePrecisionError, match="boundary dtype or shape differs"):
        _source(lambda x: -x, [(2,)], return_mode="ir")


@pytest.mark.parametrize(
    "names",
    [
        {"output_names": ["duplicate", "duplicate"]},
        {"input_names": ["same"], "output_names": ["same", "second"]},
    ],
)
def test_source_keeps_existing_invalid_name_checks(names: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _source(lambda x: (-x, x + x), [()], **names)


@pytest.mark.parametrize("legacy", [None, False, True])
def test_source_legacy_source_calls_keep_function_cache_and_public_routes_separate(
    legacy: bool | None,
) -> None:
    jax.config.update("jax_enable_x64", True)

    def model(x: jax.Array) -> jax.Array:
        return _cancellation(x)

    inputs = [jax.ShapeDtypeStruct((), np.float32)]
    before = _source(model, inputs)
    options = {} if legacy is None else {"enable_double_precision": legacy}
    ordinary = to_onnx(model, inputs, **options)
    after = _source(model, inputs)
    assert get_precision_report(ordinary) is None
    assert ordinary.graph.output[0].type.tensor_type.elem_type == (
        onnx.TensorProto.DOUBLE if legacy else onnx.TensorProto.FLOAT
    )
    for artifact in (before, after):
        assert (
            artifact.graph.output[0].type.tensor_type.elem_type
            == onnx.TensorProto.FLOAT
        )
        assert get_precision_report(artifact) is not None
        assert len(artifact.functions) == 1
    assert jax.config.x64_enabled


@pytest.mark.parametrize("inside_function", [False, True])
def test_source_reports_actual_matrix_widening_without_runtime_claim(
    inside_function: bool,
) -> None:
    def model(x: jax.Array, w: jax.Array) -> jax.Array:
        if inside_function:
            return _preferred_matrix(x, w)
        return jax.lax.dot_general(
            x, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

    artifact = _source(
        model,
        [
            jax.ShapeDtypeStruct((2, 3), np.float16),
            jax.ShapeDtypeStruct((3, 2), np.float16),
        ],
    )
    report = get_precision_report(artifact)
    assert report is not None
    assert len(report.adaptations) == 1
    assert "FP16 operands widened locally to FP32" in report.adaptations[0]
    assert "cost unmeasured" in report.adaptations[0]
    with jax.default_matmul_precision("highest"):
        unrelated = _source(lambda x: -x, [()])
    assert get_precision_report(unrelated).caller_default_matmul_precision == "highest"


@pytest.mark.parametrize("mode", ["native", "require_native", ""])
def test_source_rejects_invalid_normalization_modes_before_tracing(mode: str) -> None:
    called = []

    def model(x: Any) -> Any:
        called.append(True)
        return -x

    with pytest.raises(ValueError, match="Unsupported normalization_mode"):
        _source(model, [()], normalization_mode=mode)
    assert not called


def test_source_named_keyword_uses_real_python_binding_and_typed_port() -> None:
    called = []

    def unary(x: Any) -> Any:
        called.append(True)
        return -x

    with pytest.raises(TypeError, match="unexpected keyword.*flag"):
        _source(unary, [()], input_params={"flag": True})
    assert not called
    artifact = _source(
        lambda x, *, offset: x + offset,
        [()],
        input_params={"offset": np.float32(2)},
    )
    report = get_precision_report(artifact)
    assert report is not None
    assert [entry.exported.name for entry in report.inputs] == ["in_0", "offset"]
    assert [entry.traced.dtype for entry in report.inputs] == ["float32", "float32"]


@pytest.mark.parametrize("nested", [False, True])
def test_source_function_failure_names_actual_function_and_preserves_branch_details(
    nested: bool,
) -> None:
    def model(x: jax.Array) -> jax.Array:
        return _named_bad_outer(x) if nested else _named_bad_leaf(x)

    artifact = _source(model, [()])
    onnx.checker.check_model(_proto(artifact), full_check=True)
    report = get_precision_report(artifact)
    assert report is not None and report.profile == "source-static-v1"
    assert report.traced_outputs[0].dtype == "float32"

    def unadmitted(x: jax.Array) -> jax.Array:
        return _named_unaudited_outer(x) if nested else _named_unaudited_leaf(x)

    with pytest.raises(SourcePrecisionError) as captured:
        _source(unadmitted, [()])
    error = captured.value
    assert "NamedUnauditedLeaf" in error.location
    assert "_named_unaudited_leaf" in error.location
    if nested:
        assert "NamedUnauditedOuter" in error.location
        assert error.location.index("NamedUnauditedOuter") < error.location.index(
            "NamedUnauditedLeaf"
        )
    assert "/branch " in error.location
    assert error.primitive == "erf"
    assert error.category == "profile-exclusion"
    assert error.phase == "profile-validation"
    assert error.requested_dtypes == ("float32", "float32")
    assert error.precision_params == {}
    assert error.opset == 23
    assert error.requirement and error.action
    assert isinstance(error.__cause__, SourcePrecisionError)
    assert active_precision_policy() is PrecisionPolicy.LEGACY


@pytest.mark.parametrize("option", ["inputs_as_nchw", "outputs_as_nchw"])
@pytest.mark.parametrize("empty", [[], ()])
def test_source_facade_empty_layout_selection_is_the_ordinary_noop(
    option: str, empty: Any
) -> None:
    artifact = _source(lambda x: -x, [(2,)], **{option: empty})
    assert [node.op_type for node in artifact.graph.node] == ["Neg"]
    report = get_precision_report(artifact)
    assert report is not None
    assert report.inputs[0].exported.shape == (2,)


@pytest.mark.parametrize("primitive", ["and", "or", "xor", "not"])
def test_opset16_boolean_bitwise_uses_original_model_and_report_version(
    primitive: str, tmp_path: Path
) -> None:
    operation = {
        "and": jax.lax.bitwise_and,
        "or": jax.lax.bitwise_or,
        "xor": jax.lax.bitwise_xor,
        "not": jax.lax.bitwise_not,
    }[primitive]
    values = (
        np.asarray([[False, True, True], [True, False, False]]),
        np.asarray([[True, False, True], [True, False, True]]),
    )[: 1 if primitive == "not" else 2]
    expected = tuple(np.asarray(v) for v in jax.tree.leaves(operation(*values)))
    compiled = tuple(
        np.asarray(v) for v in jax.tree.leaves(jax.jit(operation)(*values))
    )
    for actual, reference in zip(compiled, expected, strict=True):
        np.testing.assert_array_equal(actual, reference, strict=True)
    artifact = _source(
        operation,
        [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values],
        opset=16,
    )
    path = tmp_path / "boolean-opset16.onnx"
    onnx.save_model(_proto(artifact), path)
    model = onnx.load_model(path)
    onnx.checker.check_model(model, full_check=True)
    report = get_precision_report(model)
    assert report is not None and report.opset == 16
    assert next(item.version for item in model.opset_import if item.domain == "") == 16
    assert [node.op_type for node in model.graph.node] == [
        {"and": "And", "or": "Or", "xor": "Xor", "not": "Not"}[primitive]
    ]
    for optimization in ("disabled", "all"):
        result = validate_source_samples(
            model,
            samples=[SourceSample(values, expected, SampleCriteria("bitwise"))],
            optimization=optimization,
        )
        result.require_sample_acceptance()


@pytest.mark.parametrize("kind", ["integer-bitwise", "uniform", "bernoulli"])
@pytest.mark.parametrize("boundary", ["inline", "jit", "function", "untaken-if"])
def test_opset16_rejects_unreviewed_operations_recursively_before_lowering(
    kind: str, boundary: str
) -> None:
    if kind == "integer-bitwise":

        def leaf(x: Any) -> Any:
            return jax.lax.bitwise_xor(x, np.int32(3))

        spec = jax.ShapeDtypeStruct((2, 3), np.int32)
    else:
        key = jax.random.key(17, impl="threefry2x32")

        def leaf(x: Any) -> Any:
            if kind == "uniform":
                return jax.random.uniform(key, x.shape, dtype=np.float32)
            return jax.random.bernoulli(key, np.float32(0.25), x.shape)

        spec = jax.ShapeDtypeStruct((2, 3), np.float32)

    if boundary == "inline":
        model = leaf
    elif boundary == "jit":
        model = jax.jit(leaf)
    elif boundary == "function":
        model = _Opset16ExcludedFunction(leaf)
    else:

        def model(x: Any) -> Any:
            # The offending branch is traced and audited despite being untaken.
            alternative = (
                (lambda value: value > np.float32(0))
                if kind == "bernoulli"
                else (lambda value: value)
            )
            return jax.lax.cond(np.bool_(False), leaf, alternative, x)

    with pytest.raises(SourcePrecisionError) as caught:
        _source(model, [spec], opset=16)
    error = caught.value
    assert error.opset == 16 and error.category == "profile-exclusion"
    if kind == "integer-bitwise":
        assert error.primitive == "xor"
        assert "integer bitwise" in str(error) and "opset-16" in str(error)
    elif boundary == "untaken-if":
        # Native cond carries the fixed key through both actual branches. The
        # unselected safe branch still has an unused key formal, and its finite
        # version/type boundary is checked before the random branch is visited.
        assert error.primitive is None
        assert error.location == "equation 0/branch 0/input 0"
        assert error.requested_dtypes == (str(key.dtype),)
        assert (
            error.detail
            == "scope dtype is outside the audited opset-16 normalization subset"
        )
    else:
        assert error.primitive in {
            "jit",
            "random_seed",
            "random_wrap",
            "random_unwrap",
            "random_fold_in",
            "random_bits",
        }
        assert "selected native PRNG emission" in str(error)
    if boundary == "untaken-if":
        assert "/branch " in error.location
    if boundary == "function":
        assert "_Opset16ExcludedFunction" in error.location
    assert active_precision_policy() is PrecisionPolicy.LEGACY
