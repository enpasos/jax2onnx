# tests/extra_tests/converter/test_source_representation.py

"""Declared source values retain exact storage across diagnostic exports.

The finite validator in this file admits passthroughs, real/imaginary extraction,
relu6, negation, Boolean two-branch conditionals and retraced ONNX Functions.
It does not change product admission, and its artifacts retain an unverified
lowering audit.
"""

from collections.abc import Iterator
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function
from jax2onnx import user_interface as ui
from jax2onnx.converter import conversion_api
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.precision import (
    CheckStatus,
    PRECISION_METADATA_KEY,
    PrecisionCheck,
    PrecisionReportError,
    SampleCriteria,
    SampleAcceptanceError,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)


@pytest.fixture(autouse=True)
def _restore_jax_settings() -> Iterator[None]:
    previous = {
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "jax_platforms": jax.config.jax_platforms,
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
    }
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", None)
    try:
        yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)


def _assert_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


def _complex_payload(dtype: Any, scalar: bool = False) -> np.ndarray:
    if np.dtype(dtype) == np.dtype(np.complex64):
        bits = np.asarray(
            [0x80000000, 0, 0x3F800001, 0x80000000, 0x7FC00001, 0x7FC00002],
            dtype=np.uint32,
        )
    else:
        bits = np.asarray(
            [
                0x8000000000000000,
                0,
                0x3FF0000000400000,
                0x8000000000000000,
                0x7FF8000000000001,
                0x7FF8000000000002,
            ],
            dtype=np.uint64,
        )
    values = bits.view(dtype)
    return values[:1].reshape(()) if scalar else values


def _physical_reference(value: np.ndarray) -> np.ndarray:
    """Independent typed-view oracle, separate from the production channel codec."""
    if not np.issubdtype(value.dtype, np.complexfloating):
        return value.copy(order="C")
    base = np.float32 if value.dtype == np.dtype(np.complex64) else np.float64
    return np.ascontiguousarray(value).reshape(-1).view(base).reshape((*value.shape, 2))


def _finite_diagnostic_validator(jaxpr: Any, opset: int) -> None:
    allowed = {
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }

    def require(condition: bool, detail: str, primitive: str | None = None) -> None:
        if not condition:
            raise SourcePrecisionError(
                detail,
                phase="diagnostic-validation",
                category="profile-exclusion",
                location="finite diagnostic scope",
                primitive=primitive,
                opset=opset,
                requirement="typed static passthrough, real/imag, relu6, or recursively checked ONNX Functions",
                action="keep the diagnostic within its explicitly tested operations",
            )

    require(opset in (23, 26) and not jaxpr.effects, "unsupported opset or effects")
    variables = [*jaxpr.invars, *jaxpr.constvars, *jaxpr.outvars]
    for eqn in jaxpr.eqns:
        variables.extend((*eqn.invars, *eqn.outvars))
    for var in variables:
        require(np.dtype(var.aval.dtype).name in allowed, "unsupported logical dtype")
        require(
            all(
                not isinstance(dim, bool)
                and isinstance(dim, (int, np.integer))
                and dim >= 0
                for dim in var.aval.shape
            ),
            "unsupported logical shape",
        )
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        require(not eqn.effects, "equation effects", name)
        if name.startswith("onnx_fn::"):
            # The converter invokes this same callback on the actual body trace.
            continue
        if name == "convert_element_type":
            require(
                len(eqn.invars) == len(eqn.outvars) == 1
                and np.dtype(eqn.invars[0].aval.dtype) == np.dtype(np.bool_)
                and np.dtype(eqn.outvars[0].aval.dtype) == np.dtype(np.int32)
                and tuple(eqn.invars[0].aval.shape)
                == tuple(eqn.outvars[0].aval.shape)
                == ()
                and np.dtype(eqn.params.get("new_dtype")) == np.dtype(np.int32)
                and set(eqn.params)
                <= {"new_dtype", "weak_type", "sharding", "out_sharding"},
                "only a scalar Boolean conditional selector conversion is audited",
                name,
            )
            require(
                any(
                    other.primitive.name == "cond" and other.invars[0] is eqn.outvars[0]
                    for other in jaxpr.eqns
                ),
                "selector conversion must feed a conditional",
                name,
            )
            continue
        if name == "cond":
            require(
                set(eqn.params) == {"branches"}
                and len(eqn.params["branches"]) == 2
                and len(eqn.invars) >= 2
                and np.dtype(eqn.invars[0].aval.dtype) == np.dtype(np.int32)
                and tuple(eqn.invars[0].aval.shape) == (),
                "only a two-branch scalar selector conditional is audited",
                name,
            )
            expected_inputs = [
                (np.dtype(var.aval.dtype), tuple(var.aval.shape))
                for var in eqn.invars[1:]
            ]
            expected_outputs = [
                (np.dtype(var.aval.dtype), tuple(var.aval.shape)) for var in eqn.outvars
            ]
            for branch in eqn.params["branches"]:
                body = getattr(branch, "jaxpr", branch)
                require(
                    [
                        (np.dtype(var.aval.dtype), tuple(var.aval.shape))
                        for var in body.invars
                    ]
                    == expected_inputs
                    and [
                        (np.dtype(var.aval.dtype), tuple(var.aval.shape))
                        for var in body.outvars
                    ]
                    == expected_outputs,
                    "conditional branch logical signatures must agree",
                    name,
                )
                _finite_diagnostic_validator(body, opset)
            continue
        require(
            name in {"real", "imag", "jax.nn.relu6", "neg"},
            "operation outside diagnostic",
            name,
        )
        require(
            not eqn.params and len(eqn.invars) == len(eqn.outvars) == 1,
            "invalid unary contract",
            name,
        )
        source, target = eqn.invars[0].aval, eqn.outvars[0].aval
        require(tuple(source.shape) == tuple(target.shape), "changed unary shape", name)
        if name in {"jax.nn.relu6", "neg"}:
            require(
                np.dtype(source.dtype).name
                in (
                    {
                        "float16",
                        "bfloat16",
                        "float32",
                        "float64",
                        "complex64",
                        "complex128",
                    }
                    if name == "neg"
                    else {"float16", "bfloat16", "float32", "float64"}
                )
                and np.dtype(source.dtype) == np.dtype(target.dtype),
                "invalid dtype for the audited unary operation",
                name,
            )
        else:
            base = {"complex64": "float32", "complex128": "float64"}.get(
                np.dtype(source.dtype).name
            )
            require(
                base is not None and np.dtype(target.dtype).name == base,
                "invalid complex extraction dtype",
                name,
            )


def _diagnostic(fn: Any, inputs: Any, **options: Any) -> Any:
    return ui._to_onnx(
        fn,
        inputs,
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=_finite_diagnostic_validator,
        **options,
    )


def _serialized(artifact: Any, path: Path) -> onnx.ModelProto:
    proto = (
        ir.to_proto(artifact)
        if isinstance(artifact, ir.Model)
        else (
            artifact
            if isinstance(artifact, onnx.ModelProto)
            else onnx.load_model(artifact)
        )
    )
    onnx.save_model(proto, path)
    reloaded = onnx.load_model(path)
    onnx.checker.check_model(reloaded, full_check=True)
    return reloaded


def _source_references(
    fn: Any, arguments: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, ...]:
    source_arguments = tuple(jnp.asarray(value) for value in arguments)
    eager, _ = jax.tree_util.tree_flatten(jax.device_get(fn(*source_arguments)))
    compiled, _ = jax.tree_util.tree_flatten(
        jax.device_get(jax.jit(fn)(*source_arguments))
    )
    assert len(eager) == len(compiled)
    references = tuple(np.asarray(value) for value in eager)
    for original, traced in zip(references, compiled, strict=True):
        _assert_exact(np.asarray(traced), original)
    return references


def _check_cpu(
    proto: onnx.ModelProto,
    arguments: tuple[np.ndarray, ...],
    references: tuple[np.ndarray, ...],
    optimize: str,
) -> None:
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-diagnostic-v1"
    assert {check.category: check.status for check in report.checks}[
        "lowering"
    ] is CheckStatus.UNVERIFIED
    samples = validate_source_samples(
        proto,
        samples=[
            SourceSample(
                tuple(_physical_reference(value) for value in arguments),
                tuple(_physical_reference(value) for value in references),
                SampleCriteria("bitwise"),
            )
        ],
        optimization=optimize,
    )
    samples.require_sample_acceptance()
    assert get_precision_report(proto) == report


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("scalar", [False, True], ids=["vector", "scalar"])
def test_complex_descriptor_roundtrips_exact_zero_nan_payloads_and_dtype(
    dtype: Any, scalar: bool
) -> None:
    original = _complex_payload(dtype, scalar)
    representation = representation_for_aval(
        jax.ShapeDtypeStruct(original.shape, dtype)
    )
    assert representation.encoding == "complex-real-pair"
    packed = representation.encode_array(original)
    _assert_exact(packed, _physical_reference(original))
    _assert_exact(representation.decode_array(packed), original)
    assert not np.shares_memory(packed, original)
    with pytest.raises(FrozenInstanceError):
        representation.encoding = "identity"  # type: ignore[misc]


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        jnp.bfloat16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.uint32,
        np.uint64,
        np.bool_,
    ],
)
def test_ordinary_descriptor_keeps_exact_dtype_shape_and_trailing_two(
    dtype: Any,
) -> None:
    original = np.asarray([[0, 1], [1, 0]], dtype=dtype)
    representation = representation_for_aval(original)
    assert representation.encoding == "identity"
    assert representation.physical_dtype == original.dtype
    assert representation.physical_shape == original.shape
    _assert_exact(representation.encode_array(original), original)
    _assert_exact(representation.decode_array(original), original)


@pytest.mark.parametrize("difference", ["base-dtype", "channel-count", "missing-axis"])
def test_complex_descriptor_rejects_wrong_physical_dtype_and_shape(
    difference: str,
) -> None:
    representation = representation_for_aval(jax.ShapeDtypeStruct((3,), np.complex64))
    bad = {
        "base-dtype": np.zeros((3, 2), np.float64),
        "channel-count": np.zeros((3, 3), np.float32),
        "missing-axis": np.zeros((3,), np.float32),
    }[difference]
    with pytest.raises(ValueError, match="Physical array requires"):
        representation.decode_array(bad)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_context_constant_folder_keeps_logical_payload_separate_from_storage(
    dtype: Any,
) -> None:
    payload = _complex_payload(dtype)
    var = type("LogicalVar", (), {"aval": jax.core.ShapedArray(payload.shape, dtype)})()
    ctx = IRContext(
        opset=23, enable_double_precision=False, precision_policy=PrecisionPolicy.SOURCE
    )
    value = ctx.bind_const_for_var(var, payload)
    _assert_exact(ctx.try_evaluate_const(var), payload)
    _assert_exact(value.const_value.numpy(), _physical_reference(payload))
    assert value.dtype == ir.DataType.from_numpy(_physical_reference(payload).dtype)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("scalar", [False, True], ids=["vector", "scalar"])
@pytest.mark.parametrize("return_mode", ["ir", "proto", "file"])
@pytest.mark.parametrize("optimize", ["disabled", "all"])
def test_diagnostic_complex_passthrough_serializes_matches_original_jax_and_cpu(
    dtype: Any, scalar: bool, return_mode: str, optimize: str, tmp_path: Path
) -> None:
    payload = _complex_payload(dtype, scalar)

    def fn(z: Any) -> Any:
        return z

    references = _source_references(fn, (payload,))
    artifact = _diagnostic(
        fn,
        [jax.ShapeDtypeStruct(payload.shape, payload.dtype)],
        return_mode=return_mode,
        output_path=tmp_path / "complex.onnx",
        input_names=["logical"],
        output_names=["result"],
    )
    proto = _serialized(artifact, tmp_path / "serialized.onnx")
    report = get_precision_report(proto)
    assert report.inputs[0].traced.dtype == np.dtype(dtype).name
    assert report.inputs[0].exported.dtype == references[0].real.dtype.name
    assert report.inputs[0].traced.shape == payload.shape
    assert report.inputs[0].exported.shape == (*payload.shape, 2)
    assert len(report.adaptations) == 2
    _check_cpu(proto, (payload,), references, optimize)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("scalar", [False, True], ids=["vector", "scalar"])
@pytest.mark.parametrize("optimize", ["disabled", "all"])
def test_diagnostic_complex_constants_match_original_jax_and_actual_tensor_payload(
    dtype: Any, scalar: bool, optimize: str, tmp_path: Path
) -> None:
    payload = _complex_payload(dtype, scalar)

    def fn() -> tuple[Any, ...]:
        return payload, jnp.real(payload), jnp.imag(payload)

    references = _source_references(fn, ())
    proto = _serialized(_diagnostic(fn, []), tmp_path / "constants.onnx")
    arrays = [onnx.numpy_helper.to_array(tensor) for tensor in proto.graph.initializer]
    expected = _physical_reference(payload)
    assert any(
        array.dtype == expected.dtype
        and array.shape == expected.shape
        and array.tobytes() == expected.tobytes()
        for array in arrays
    )
    _check_cpu(proto, (), references, optimize)


_CAPTURE64 = _complex_payload(np.complex64)[::-1].copy()
_CAPTURE128 = _complex_payload(np.complex128)[::-1].copy()


@onnx_function
def _mixed_inner(z: Any, real_pair: Any, indices: Any, flags: Any) -> tuple[Any, ...]:
    captured = (
        _CAPTURE64 if np.dtype(z.dtype) == np.dtype(np.complex64) else _CAPTURE128
    )
    return z, captured, jnp.real(z), jnp.imag(captured), real_pair, indices, flags


@onnx_function
def _mixed_outer(z: Any, real_pair: Any, indices: Any, flags: Any) -> tuple[Any, ...]:
    return _mixed_inner(z, real_pair, indices, flags)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("optimize", ["disabled", "all"])
def test_diagnostic_complex_captures_nested_functions_and_mixed_ordinary_outputs(
    dtype: Any, optimize: str, tmp_path: Path
) -> None:
    payload = _complex_payload(dtype)
    captured = _CAPTURE64 if np.dtype(dtype) == np.dtype(np.complex64) else _CAPTURE128

    def model(*args: Any) -> tuple[Any, ...]:
        return _mixed_outer(*args)

    arguments = (
        payload,
        np.asarray([[1, -0.0], [2, 3]], np.float32),
        np.asarray([1, 3], np.int32),
        np.asarray([True, False], np.bool_),
    )
    references = _source_references(model, arguments)
    seen = []

    def observe(jaxpr: Any, opset: int) -> None:
        seen.append(tuple(eqn.primitive.name for eqn in jaxpr.eqns))
        _finite_diagnostic_validator(jaxpr, opset)

    artifact = ui._to_onnx(
        model,
        [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in arguments],
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=observe,
        return_mode="ir",
    )
    assert len(artifact.functions) == 2
    assert ("real", "imag") in seen
    assert any(any(name.endswith("._mixed_inner") for name in scope) for scope in seen)
    assert any(any(name.endswith("._mixed_outer") for name in scope) for scope in seen)
    proto = _serialized(artifact, tmp_path / "nested.onnx")
    report = get_precision_report(proto)
    assert [value.dtype for value in report.exported_outputs][-3:] == [
        "float32",
        "int32",
        "bool",
    ]
    assert report.exported_outputs[-3].shape == (2, 2)
    constant_tensors = [
        attr.t
        for function in proto.functions
        for node in function.node
        for attr in node.attribute
        if attr.type == onnx.AttributeProto.TENSOR
    ]
    expected = _physical_reference(captured)
    assert any(
        tensor.data_type == int(ir.DataType.from_numpy(expected.dtype))
        and onnx.numpy_helper.to_array(tensor).tobytes() == expected.tobytes()
        for tensor in constant_tensors
    )
    _check_cpu(proto, arguments, references, optimize)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("optimize", ["disabled", "all"])
def test_declared_complex_channels_survive_distinct_conditional_branch_scopes(
    dtype: Any, flag: bool, optimize: str, tmp_path: Path
) -> None:
    def model(predicate: Any, value: Any) -> Any:
        return jax.lax.cond(predicate, lambda z: -z, lambda z: z, value)

    arguments = (np.asarray(flag), _complex_payload(dtype))
    references = _source_references(model, arguments)
    artifact = _diagnostic(
        model,
        [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in arguments],
        return_mode="ir",
    )
    assert any(node.op_type == "If" for node in artifact.graph)
    proto = _serialized(artifact, tmp_path / "conditional.onnx")
    _check_cpu(proto, arguments, references, optimize)


def _typed_input(kind: str, dtype: Any, *, weak: bool = False) -> Any:
    if kind == "spec":
        return jax.ShapeDtypeStruct((), dtype, weak_type=weak)
    if kind == "aval":
        return jax.core.ShapedArray((), dtype, weak_type=weak)
    if kind == "numpy-array":
        return np.asarray(1, dtype=dtype)
    if kind == "numpy-scalar":
        return np.dtype(dtype).type(1)
    if kind == "jax-array":
        return jnp.asarray(1, dtype=dtype)
    return SimpleNamespace(shape=(), dtype=np.dtype(dtype), weak_type=weak)


_INPUT_FORMS = ["spec", "aval", "numpy-array", "numpy-scalar", "jax-array", "protocol"]


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint64, np.complex128])
@pytest.mark.parametrize("kind", _INPUT_FORMS)
@pytest.mark.parametrize("diagnostic", [False, True])
def test_generic_canonicalization_guard_precedes_admission_for_all_typed_forms(
    dtype: Any, kind: str, diagnostic: bool
) -> None:
    value = _typed_input(kind, dtype)
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    calls = []

    def fn(x: Any) -> Any:
        calls.append(True)
        return x

    with warnings.catch_warnings(record=True) as emitted:
        with pytest.raises(SourcePrecisionError) as captured:
            ui._to_onnx(
                fn,
                [value],
                precision_policy=PrecisionPolicy.SOURCE,
                diagnostic_jaxpr_validator=(
                    _finite_diagnostic_validator if diagnostic else None
                ),
            )
    assert captured.value.category == "configuration-conflict"
    assert captured.value.phase == "configuration"
    assert captured.value.requested_dtypes == (np.dtype(dtype).name,)
    assert not calls and not emitted
    assert not jax.config.x64_enabled
    assert jax.config.jax_default_matmul_precision == "highest"


@pytest.mark.parametrize("kind", _INPUT_FORMS)
@pytest.mark.parametrize("weak", [False, True])
def test_existing_typed_forms_share_normalization_and_faithful_weak_metadata(
    kind: str, weak: bool
) -> None:
    value = _typed_input(kind, np.float32, weak=weak)
    artifact = ui._to_onnx(
        lambda x: x, [value], precision_policy=PrecisionPolicy.SOURCE
    )
    report = get_precision_report(artifact)
    expected_weak = weak if kind in {"spec", "aval", "protocol"} else False
    assert report.profile == "source-static-v1"
    for properties in (
        report.inputs[0].supplied,
        report.inputs[0].normalized,
        report.inputs[0].traced,
    ):
        assert properties.dtype == "float32" and properties.shape == ()
        assert properties.weak_type is expected_weak


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint64, np.complex128])
def test_core_uses_same_configuration_guard_before_any_trace(dtype: Any) -> None:
    jax.config.update("jax_enable_x64", False)
    called = []
    with pytest.raises(SourcePrecisionError, match="x64 is disabled") as captured:
        conversion_api._convert_to_ir(
            fn=lambda x: called.append(True) or x,
            inputs=[jax.ShapeDtypeStruct((), dtype)],
            input_params=None,
            model_name="configuration_guard",
            opset=23,
            enable_double_precision=False,
            record_primitive_calls_file=None,
            precision_policy=PrecisionPolicy.SOURCE,
            jaxpr_validator=_finite_diagnostic_validator,
        )
    assert captured.value.category == "configuration-conflict" and not called


@onnx_function
def _pending_relu6(x: Any) -> Any:
    return jax.nn.relu6(x)


def test_diagnostic_function_cache_cannot_turn_later_default_export_into_audited_acceptance() -> (
    None
):
    def model(x: Any) -> Any:
        return _pending_relu6(x)

    artifact = _diagnostic(model, [jax.ShapeDtypeStruct((2,), np.float32)])
    assert get_precision_report(artifact).profile == "source-diagnostic-v1"
    with pytest.raises(
        SourcePrecisionError, match="primitive is outside the profile"
    ) as captured:
        ui._to_onnx(
            model,
            [jax.ShapeDtypeStruct((2,), np.float32)],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert "_pending_relu6" in captured.value.location


def test_diagnostic_retains_existing_opset_gap() -> None:
    with pytest.raises(
        SourcePrecisionError, match="internal migration gap"
    ) as captured:
        _diagnostic(lambda x: x, [jax.ShapeDtypeStruct((2,), np.int32)], opset=22)
    assert captured.value.opset == 22
    assert captured.value.phase == "input"
    assert captured.value.category == "profile-exclusion"


def test_diagnostic_prefer_native_keeps_int32_and_unverified_lowering(
    tmp_path: Path,
) -> None:
    artifact = _diagnostic(
        lambda x: x,
        [jax.ShapeDtypeStruct((2,), np.int32)],
        normalization_mode="prefer_native",
    )
    report = get_precision_report(artifact)
    assert report.profile == "source-diagnostic-v1" and report.opset == 23
    assert (
        len(report.inputs)
        == len(report.traced_outputs)
        == len(report.exported_outputs)
        == 1
    )
    for properties in (
        report.inputs[0].supplied,
        report.inputs[0].normalized,
        report.inputs[0].traced,
        report.inputs[0].exported,
        report.traced_outputs[0],
        report.exported_outputs[0],
    ):
        assert properties.dtype == "int32" and properties.shape == (2,)
    checks = {check.category: check for check in report.checks}
    assert checks["representation"].status is CheckStatus.VALIDATED
    assert checks["lowering"].status is CheckStatus.UNVERIFIED
    assert "audit_pending" in checks["lowering"].detail
    assert checks["runtime-execution"].status is CheckStatus.UNVERIFIED
    assert checks["numerical-acceptance"].status is CheckStatus.UNVERIFIED
    normalization = [
        note for note in report.adaptations if note.startswith("normalization:")
    ]
    assert len(normalization) == 1
    assert "requested=prefer_native" in normalization[0]
    assert "effective=UNVERIFIED" in normalization[0]
    assert "source-equivalence=UNVERIFIED" in normalization[0]
    proto = _serialized(artifact, tmp_path / "diagnostic_prefer_native.onnx")
    assert get_precision_report(proto) == report
    assert len(proto.graph.input) == len(proto.graph.output) == 1
    for value in (*proto.graph.input, *proto.graph.output):
        tensor = value.type.tensor_type
        assert tensor.elem_type == onnx.TensorProto.INT32
        assert tuple(dim.dim_value for dim in tensor.shape.dim) == (2,)


def test_diagnostic_callback_rejects_legacy_route_and_cannot_validate_lowering() -> (
    None
):
    with pytest.raises(ValueError, match="requires the private SOURCE"):
        ui._to_onnx(
            lambda x: x, [()], diagnostic_jaxpr_validator=_finite_diagnostic_validator
        )
    artifact = _diagnostic(lambda x: x, [()])
    report = get_precision_report(artifact)
    with pytest.raises(PrecisionReportError, match="cannot claim validated lowering"):
        replace(
            report,
            checks=(PrecisionCheck("lowering", CheckStatus.VALIDATED, "forged"),),
        )
    metadata = next(
        value
        for value in artifact.metadata_props
        if value.key == PRECISION_METADATA_KEY
    )
    payload = json.loads(metadata.value)
    for check in payload["checks"]:
        if check["category"] == "lowering":
            check["status"] = "validated"
    metadata.value = json.dumps(payload)
    with pytest.raises(PrecisionReportError, match="cannot claim validated lowering"):
        get_precision_report(artifact)


def test_false_valued_diagnostic_callable_is_selected_and_reported_explicitly() -> None:
    class Validator:
        calls = 0

        def __bool__(self) -> bool:
            return False

        def __call__(self, jaxpr: Any, opset: int) -> None:
            self.calls += 1
            _finite_diagnostic_validator(jaxpr, opset)

    validator = Validator()
    artifact = ui._to_onnx(
        lambda x: x,
        [jax.ShapeDtypeStruct((), np.complex64)],
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=validator,
    )
    assert validator.calls >= 1
    assert get_precision_report(artifact).profile == "source-diagnostic-v1"


@pytest.mark.parametrize("corruption", ["tensor-dtype", "tensor-shape"])
def test_corrupted_complex_storage_cannot_receive_sample_acceptance(
    corruption: str,
) -> None:
    payload = _complex_payload(np.complex64)
    artifact = _diagnostic(lambda: payload, [])
    physical = _physical_reference(payload)
    changed = (
        physical.astype(np.float64)
        if corruption == "tensor-dtype"
        else physical.reshape(2, 3)
    )
    artifact.graph.initializer[0].CopyFrom(
        onnx.numpy_helper.from_array(changed, name=artifact.graph.initializer[0].name)
    )
    validation = validate_source_samples(
        artifact,
        samples=[SourceSample((), (physical,), SampleCriteria("bitwise"))],
        optimization=None,
    )
    assert validation.status is CheckStatus.FAILED
    with pytest.raises(SampleAcceptanceError):
        validation.require_sample_acceptance()


@pytest.mark.parametrize(
    "kind", ["ordinary-dtype", "complex-base", "complex-channel-count"]
)
def test_core_result_validation_keeps_exact_ordinary_and_encoded_contracts(
    kind: str,
) -> None:
    logical_dtype = np.float32 if kind == "ordinary-dtype" else np.complex64
    var = type("LogicalVar", (), {"aval": jax.core.ShapedArray((3,), logical_dtype)})()
    actual_dtype = (
        ir.DataType.DOUBLE if kind != "complex-channel-count" else ir.DataType.FLOAT
    )
    actual_shape = (
        (3,)
        if kind == "ordinary-dtype"
        else (3, 3) if kind == "complex-channel-count" else (3, 2)
    )
    ctx = IRContext(
        opset=23, enable_double_precision=False, precision_policy=PrecisionPolicy.SOURCE
    )
    value = ir.val("wrong", dtype=actual_dtype, shape=actual_shape)
    ctx.builder.inputs.append(value)
    ctx.bind_value_for_var(var, value)
    with pytest.raises(RuntimeError, match="Source result type"):
        conversion_api._bind_jaxpr_outputs(
            ctx,
            SimpleNamespace(outvars=[var]),
            outputs_as_nchw=[],
            enable_double_precision=False,
        )
