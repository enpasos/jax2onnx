# tests/extra_tests/converter/test_source_facade.py

"""Private SOURCE and public legacy calls share the actual export facade."""

from collections.abc import Callable, Iterator
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
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.precision import CheckStatus, get_precision_report
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


@onnx_function(name="NamedDiagnosticLeaf")
def _named_bad_leaf(x: jax.Array) -> jax.Array:
    return jax.lax.cond(
        x > np.float32(0), lambda value: jnp.sin(value), lambda value: -value, x
    )


@onnx_function(name="NamedDiagnosticOuter")
def _named_bad_outer(x: jax.Array) -> jax.Array:
    return _named_bad_leaf(x)


@onnx_function
def _large_fp64_inner(x: jax.Array, *, payload: np.ndarray) -> tuple[Any, ...]:
    return x, payload


@onnx_function
def _large_fp64_outer(x: jax.Array, *, payload: np.ndarray) -> tuple[Any, ...]:
    return _large_fp64_inner(x, payload=payload)


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


def test_source_report_preserves_mixed_structured_output_order() -> None:
    def model(x: jax.Array) -> Any:
        return x.astype(jnp.float16), {
            "bits": jax.lax.bitcast_convert_type(x, jnp.uint32),
            "positive": x > np.float32(0),
        }

    artifact = _source(model, [()], opset=26, output_names=["half", "bits", "positive"])
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
    serialize = ir.to_proto

    def corrupt(model: ir.Model) -> onnx.ModelProto:
        proto = serialize(model)
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


@pytest.mark.parametrize(
    "options",
    [
        {"input_params": {"flag": True}},
        {"normalization_mode": "prefer_native"},
    ],
)
def test_source_deferred_options_are_explicit_internal_migration_gaps(
    options: dict[str, Any],
) -> None:
    with pytest.raises(
        SourcePrecisionError, match="internal migration gap"
    ) as captured:
        _source(lambda x: -x, [()], **options)
    assert captured.value.category == "profile-exclusion"
    assert "broader existing functionality" in captured.value.action


@pytest.mark.parametrize("nested", [False, True])
def test_source_function_failure_names_actual_function_and_preserves_branch_details(
    nested: bool,
) -> None:
    def model(x: jax.Array) -> jax.Array:
        return _named_bad_outer(x) if nested else _named_bad_leaf(x)

    with pytest.raises(SourcePrecisionError) as captured:
        _source(model, [()])
    error = captured.value
    assert "NamedDiagnosticLeaf" in error.location
    assert "_named_bad_leaf" in error.location
    if nested:
        assert "NamedDiagnosticOuter" in error.location
        assert error.location.index("NamedDiagnosticOuter") < error.location.index(
            "NamedDiagnosticLeaf"
        )
    assert "/branch " in error.location
    assert error.primitive in {"sin", "jax.numpy.sin"}
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
