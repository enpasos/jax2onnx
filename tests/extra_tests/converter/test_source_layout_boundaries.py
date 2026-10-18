# tests/extra_tests/converter/test_source_layout_boundaries.py

"""Static SOURCE layout bridges preserve logical values and explicit storage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function
from jax2onnx import user_interface as ui
from jax2onnx.converter import conversion_api
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from tests.extra_tests.converter.source_precision_diagnostics import DiagnosticValidator
from tests.extra_tests.converter.test_source_dropout_parameters import (
    DropoutDiagnosticValidator,
)


def _global_configuration() -> dict[str, Any]:
    return {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }


@contextmanager
def _configuration_scope() -> Iterator[None]:
    previous = _global_configuration()
    try:
        with jax.enable_x64(True):
            yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)


@pytest.fixture(autouse=True)
def _configuration() -> Iterator[None]:
    with _configuration_scope():
        yield


@pytest.mark.parametrize("failure", [False, True])
def test_configuration_fixture_restores_hidden_globals_and_effective_caller_scopes(
    failure: bool,
) -> None:
    original = _global_configuration()
    with (
        jax.enable_x64(False),
        jax.default_matmul_precision("highest"),
        jax.numpy_dtype_promotion("strict"),
        jax.numpy_rank_promotion("raise"),
    ):
        effective = {name: getattr(jax.config, name) for name in original}
        expected_error = (
            pytest.raises(RuntimeError, match="fixture failure")
            if failure
            else nullcontext()
        )
        with expected_error:
            with _configuration_scope():
                assert jax.config.x64_enabled
                assert _global_configuration() == original
                mutations = {
                    "jax_enable_x64": not original["jax_enable_x64"],
                    "jax_default_matmul_precision": (
                        "default"
                        if original["jax_default_matmul_precision"] != "default"
                        else "float32"
                    ),
                    "jax_numpy_dtype_promotion": (
                        "strict"
                        if original["jax_numpy_dtype_promotion"] != "strict"
                        else "standard"
                    ),
                    "jax_numpy_rank_promotion": (
                        "raise"
                        if original["jax_numpy_rank_promotion"] != "raise"
                        else "allow"
                    ),
                    "jax_platforms": (
                        "cpu" if original["jax_platforms"] != "cpu" else None
                    ),
                }
                for name, value in mutations.items():
                    jax.config.update(name, value)
                if failure:
                    raise RuntimeError("fixture failure")
        assert _global_configuration() == original
        assert {name: getattr(jax.config, name) for name in original} == effective
    assert _global_configuration() == original


def _source(fn: Any, inputs: Any, *, diagnostic: bool = False, **options: Any) -> Any:
    if diagnostic:
        options["diagnostic_jaxpr_validator"] = options.pop(
            "validator", DiagnosticValidator()
        )
    return ui._to_onnx(fn, inputs, precision_policy=PrecisionPolicy.SOURCE, **options)


def _proto(artifact: Any) -> onnx.ModelProto:
    if isinstance(artifact, ir.Model):
        return ir.to_proto(artifact)
    return (
        artifact if isinstance(artifact, onnx.ModelProto) else onnx.load_model(artifact)
    )


def _coordinates(
    dtype: Any = np.float32, shape: tuple[int, ...] = (1, 2, 3, 4)
) -> np.ndarray:
    indices = np.arange(np.prod(shape)).reshape(shape)
    if np.dtype(dtype) == np.dtype(np.bool_):
        return indices % 3 == 0
    if np.issubdtype(dtype, np.integer):
        return (indices - 9).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        return ((indices - 9) + 1j * (indices + 3)).astype(dtype)
    result = (indices - 9).astype(dtype) / np.asarray(4, dtype=dtype)
    result.flat[0] = np.asarray(-0.0, dtype=dtype)
    if np.dtype(dtype) == np.dtype(np.float64):
        result.flat[1] += np.float64(2**-40)
    return result


def _physical(value: np.ndarray, nchw: bool) -> np.ndarray:
    """Independent typed view and NumPy axis oracle; never the production codec."""
    value = np.asarray(value)
    if np.issubdtype(value.dtype, np.complexfloating):
        base = np.float32 if value.dtype == np.dtype(np.complex64) else np.float64
        packed = (
            np.ascontiguousarray(value)
            .reshape(-1)
            .view(base)
            .reshape((*value.shape, 2))
        )
        return np.transpose(packed, (0, 3, 1, 2, 4)).copy() if nchw else packed.copy()
    return np.transpose(value, (0, 3, 1, 2)).copy() if nchw else value.copy()


def _native(fn: Any, args: tuple[Any, ...], **params: Any) -> tuple[np.ndarray, ...]:
    # Both references run before the converter installs its trace-time patches.
    eager = jax.tree_util.tree_leaves(jax.device_get(fn(*args, **params)))
    compiled = jax.tree_util.tree_leaves(jax.device_get(jax.jit(fn)(*args, **params)))
    for left, right in zip(eager, compiled, strict=True):
        assert np.asarray(left).dtype == np.asarray(right).dtype
        assert np.asarray(left).tobytes() == np.asarray(right).tobytes()
    return tuple(np.asarray(value) for value in eager)


def _check_samples(artifact: Any, samples: list[SourceSample]) -> None:
    report = get_precision_report(artifact)
    assert report is not None
    for optimize in ("disabled", "all"):
        evidence = validate_source_samples(
            artifact, samples=samples, optimization=optimize
        )
        evidence.require_sample_acceptance()
    assert get_precision_report(artifact) == report
    statuses = {check.category: check.status for check in report.checks}
    assert statuses["representation"] is CheckStatus.VALIDATED
    assert (
        statuses["runtime-execution"]
        is statuses["numerical-acceptance"]
        is CheckStatus.UNVERIFIED
    )
    assert statuses["lowering"] is (
        CheckStatus.UNVERIFIED
        if report.profile == "source-diagnostic-v1"
        else CheckStatus.VALIDATED
    )


@pytest.mark.parametrize(
    ("selected_input", "selected_output"), [(True, False), (False, True), (True, True)]
)
def test_each_layout_direction_preserves_original_coordinate_values(
    selected_input: bool, selected_output: bool
) -> None:
    def model(x: Any) -> Any:
        return -x

    arrays = [_coordinates(), np.flip(_coordinates(), axis=2).copy()]
    expected = [_native(model, (jnp.asarray(value),))[0] for value in arrays]
    artifact = _source(
        model,
        [jax.ShapeDtypeStruct(arrays[0].shape, arrays[0].dtype)],
        inputs_as_nchw=[0] if selected_input else None,
        outputs_as_nchw=[0] if selected_output else None,
    )
    report = get_precision_report(artifact)
    assert (
        report.inputs[0].supplied.shape
        == report.inputs[0].normalized.shape
        == report.inputs[0].traced.shape
        == arrays[0].shape
    )
    assert report.inputs[0].exported.shape == (
        (1, 4, 2, 3) if selected_input else arrays[0].shape
    )
    assert report.exported_outputs[0].shape == (
        (1, 4, 2, 3) if selected_output else arrays[0].shape
    )
    assert (
        sum("external NCHW" in text for text in report.adaptations)
        == selected_input + selected_output
    )
    _check_samples(
        artifact,
        [
            SourceSample(
                (_physical(value, selected_input),),
                (_physical(reference, selected_output),),
                SampleCriteria("bitwise"),
            )
            for value, reference in zip(arrays, expected, strict=True)
        ],
    )


def test_mixed_boundaries_aliases_and_simultaneous_names_use_actual_binding_order() -> (
    None
):
    def model(x: Any, y: Any) -> Any:
        return x, -x, y

    x, y = _coordinates(), np.flip(_coordinates(), axis=3).copy()
    references = _native(model, (jnp.asarray(x), jnp.asarray(y)))
    artifact = _source(
        model,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)] * 2,
        return_mode="ir",
        inputs_as_nchw=[0],
        outputs_as_nchw=[1],
        input_names=["in_1", "in_0"],
        output_names=["canonical_alias", "packed_negative", "other_alias"],
    )
    assert [value.name for value in artifact.graph.inputs] == ["in_1", "in_0"]
    assert [tuple(value.shape) for value in artifact.graph.outputs] == [
        x.shape,
        (1, 4, 2, 3),
        y.shape,
    ]
    report = get_precision_report(artifact)
    assert [item.exported.name for item in report.inputs] == ["in_1", "in_0"]
    _check_samples(
        artifact,
        [
            SourceSample(
                {"in_1": _physical(x, True), "in_0": y},
                tuple(
                    _physical(value, index == 1)
                    for index, value in enumerate(references)
                ),
                SampleCriteria("bitwise"),
            )
        ],
    )


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.bool_, np.int32, np.complex64]
)
def test_layout_identity_preserves_declared_dtype_and_complex_pair_axis(
    dtype: Any,
) -> None:
    value = _coordinates(dtype)
    reference = _native(lambda x: x, (jnp.asarray(value),))[0]
    diagnostic = np.dtype(dtype).kind in "bic"
    artifact = _source(
        lambda x: x,
        [jax.ShapeDtypeStruct(value.shape, value.dtype)],
        diagnostic=diagnostic,
        inputs_as_nchw=[0],
        outputs_as_nchw=[0],
    )
    report = get_precision_report(artifact)
    expected = _physical(reference, True)
    assert report.inputs[0].traced.dtype == np.dtype(dtype).name
    assert report.inputs[0].exported.dtype == expected.dtype.name
    assert (
        report.inputs[0].exported.shape
        == report.exported_outputs[0].shape
        == expected.shape
    )
    assert report.inputs[0].exported.name == "in_0"
    _check_samples(
        artifact,
        [
            SourceSample(
                (_physical(value, True),), (expected,), SampleCriteria("bitwise")
            )
        ],
    )


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_layout_descriptor_composes_codec_without_reinterpreting_a_real_pair_shape(
    dtype: Any,
) -> None:
    value = _coordinates(dtype, shape=(1, 3, 4, 2))
    canonical = representation_for_aval(value)
    boundary = representation_for_aval(value, boundary_layout="NCHW")
    expected = _physical(value, True)
    assert canonical.boundary_layout is None
    assert boundary.logical_shape == value.shape
    assert boundary.physical_shape == expected.shape
    assert boundary.physical_dtype == expected.dtype
    assert boundary.encode_array(value).tobytes() == expected.tobytes()
    assert boundary.decode_array(expected).tobytes() == value.tobytes()
    assert len(boundary.input_permutation) == (5 if dtype is np.complex64 else 4)
    assert canonical.encoding == (
        "complex-real-pair" if dtype is np.complex64 else "identity"
    )
    with pytest.raises(ValueError, match="Physical array"):
        boundary.decode_array(np.zeros(value.shape, dtype=np.float64))


def test_complex_layout_passthrough_preserves_signed_zero_and_nan_component_bits() -> (
    None
):
    bits = np.arange(48, dtype=np.float32).view(np.uint32)
    bits[:4] = [0x80000000, 0, 0x7FC00001, 0xFFC00002]
    value = bits.view(np.complex64).reshape(1, 2, 3, 4)
    reference = _native(lambda x: x, (jnp.asarray(value),))[0]
    artifact = _source(
        lambda x: x,
        [jax.ShapeDtypeStruct(value.shape, value.dtype)],
        diagnostic=True,
        inputs_as_nchw=[0],
        outputs_as_nchw=[0],
    )
    assert any(
        "pair axis unchanged" in text
        for text in get_precision_report(artifact).adaptations
    )
    _check_samples(
        artifact,
        [
            SourceSample(
                (_physical(value, True),),
                (_physical(reference, True),),
                SampleCriteria("bitwise"),
            )
        ],
    )


@pytest.mark.parametrize("kind", ["shape-only", "weak-array"])
def test_logical_input_normalization_and_weak_metadata_are_unchanged(kind: str) -> None:
    value = (
        jnp.broadcast_to(jnp.asarray(1.25), (1, 2, 3, 4))
        if kind == "weak-array"
        else jnp.asarray(_coordinates())
    )
    assert kind != "weak-array" or value.weak_type
    supplied = value if kind == "weak-array" else tuple(value.shape)
    reference = _native(lambda x: -x, (value,))[0]
    artifact = _source(
        lambda x: -x, [supplied], inputs_as_nchw=[0], outputs_as_nchw=[0]
    )
    properties = get_precision_report(artifact).inputs[0]
    assert (
        properties.normalized.weak_type
        is properties.traced.weak_type
        is (kind == "weak-array")
    )
    assert (
        properties.traced.dtype == "float64"
        if kind == "weak-array"
        else properties.traced.dtype == "float32"
    )
    _check_samples(
        artifact,
        [
            SourceSample(
                (_physical(np.asarray(value), True),),
                (_physical(reference, True),),
                SampleCriteria("bitwise"),
            )
        ],
    )


def test_bf16_layout_artifact_checker_and_report_do_not_claim_cpu_execution() -> None:
    artifact = _source(
        lambda x: x,
        [jax.ShapeDtypeStruct((1, 2, 3, 4), jnp.bfloat16)],
        inputs_as_nchw=[0],
        outputs_as_nchw=[0],
    )
    onnx.checker.check_model(artifact, full_check=True)
    report = get_precision_report(artifact)
    assert (
        report.inputs[0].traced.dtype == report.inputs[0].exported.dtype == "bfloat16"
    )
    assert report.inputs[0].exported.shape == (1, 4, 2, 3)
    assert (
        artifact.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
    )
    assert {check.category: check.status for check in report.checks}[
        "runtime-execution"
    ] is CheckStatus.UNVERIFIED


@onnx_function
def _layout_named(x: Any, *, runtime: Any) -> Any:
    return jax.lax.cond(runtime, lambda value: -value, lambda value: value, x)


def test_named_bool_and_function_children_stay_canonical_under_root_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def model(x: Any, *, runtime: Any) -> Any:
        return _layout_named(x, runtime=runtime)

    value = _coordinates()
    references = {
        flag: _native(model, (jnp.asarray(value),), runtime=jnp.asarray(flag))[0]
        for flag in (True, False)
    }
    observed = []
    real = conversion_api._lower_jaxpr_equations

    def inspect_bindings(ctx: Any, jaxpr: Any) -> None:
        observed.append(tuple(ctx.require_value_for_var(jaxpr.invars[0]).shape))
        real(ctx, jaxpr)

    monkeypatch.setattr(conversion_api, "_lower_jaxpr_equations", inspect_bindings)
    artifact = _source(
        model,
        [jax.ShapeDtypeStruct(value.shape, value.dtype)],
        diagnostic=True,
        validator=DropoutDiagnosticValidator(),
        input_params={"runtime": True},
        inputs_as_nchw=[0],
        outputs_as_nchw=[0],
        return_mode="ir",
        input_names=["features"],
    )
    assert observed == [value.shape]
    assert [value.name for value in artifact.graph.inputs] == ["features", "runtime"]
    assert tuple(artifact.graph.inputs[1].shape) == ()
    function = next(iter(artifact.functions.values()))
    assert tuple(function.graph.inputs[0].shape) == value.shape
    assert tuple(function.graph.inputs[1].shape) == ()
    _check_samples(
        artifact,
        [
            SourceSample(
                {"features": _physical(value, True), "runtime": np.asarray(flag)},
                (_physical(references[flag], True),),
                SampleCriteria("bitwise"),
            )
            for flag in (True, False)
        ],
    )


@pytest.mark.parametrize("mode", ["ir", "proto", "file", "web"])
def test_layout_return_modes_reload_complete_names_aliases_and_report(
    mode: str, tmp_path: Path
) -> None:
    value = _coordinates()
    references = _native(lambda x: (x, x, -x), (jnp.asarray(value),))
    path = tmp_path / "layout.onnx"
    sidecar = path.with_name(path.name + ".data")
    if mode == "web":
        sidecar.write_bytes(b"stale")
    artifact = _source(
        lambda x: (x, x, -x),
        [jax.ShapeDtypeStruct(value.shape, value.dtype)],
        inputs_as_nchw=[0],
        outputs_as_nchw=[0, 1, 2],
        input_names=["image"],
        output_names=["first", "second", "negative"],
        return_mode="file" if mode == "web" else mode,
        export_mode="web" if mode == "web" else "standard",
        output_path=path,
    )
    proto = _proto(artifact)
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(artifact)
    assert get_precision_report(proto) == report
    assert [value.name for value in proto.graph.input] == ["image"]
    assert [value.name for value in proto.graph.output] == [
        "first",
        "second",
        "negative",
    ]
    assert report.inputs[0].traced.shape == value.shape
    assert all(
        properties.shape == (1, 4, 2, 3) for properties in report.exported_outputs
    )
    if mode == "web":
        assert not sidecar.exists()
    _check_samples(
        artifact,
        [
            SourceSample(
                (_physical(value, True),),
                tuple(_physical(reference, True) for reference in references),
                SampleCriteria("bitwise"),
            )
        ],
    )


@pytest.mark.parametrize(
    ("option", "selection", "message"),
    [
        ("inputs_as_nchw", [True], "must be integers"),
        ("outputs_as_nchw", [0.0], "must be integers"),
        ("inputs_as_nchw", [-1], "out of range"),
        ("outputs_as_nchw", [1], "out of range"),
        ("inputs_as_nchw", [0, 0], "must be unique"),
        ("outputs_as_nchw", [0, 0], "must be unique"),
    ],
)
def test_invalid_selections_preserve_existing_validation(
    option: str, selection: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _source(lambda x: x, [(1, 2, 3, 4)], **{option: selection})


@pytest.mark.parametrize("option", ["inputs_as_nchw", "outputs_as_nchw"])
def test_layout_requires_logical_rank_four_and_cannot_select_named_bool(
    option: str,
) -> None:
    with pytest.raises(ValueError, match="rank 3, expected 4"):
        _source(lambda x: x, [(1, 2, 3)], **{option: [0]})
    with pytest.raises(ValueError, match="out of range"):
        _source(
            lambda x, *, flag: x,
            [(1, 2, 3, 4)],
            diagnostic=True,
            input_params={"flag": True},
            **{option: [1]},
        )


def test_rank_five_complex_physical_storage_is_not_a_logical_rank_four_input() -> None:
    with pytest.raises(ValueError, match="rank 5, expected 4"):
        _source(
            lambda x: x,
            [jax.ShapeDtypeStruct((1, 2, 3, 4, 2), np.complex64)],
            diagnostic=True,
            inputs_as_nchw=[0],
        )


def test_layout_keeps_x64_conflict_symbolic_normalization_and_primitive_gaps() -> None:
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="x64 is disabled"):
            _source(
                lambda x: x,
                [jax.ShapeDtypeStruct((1, 2, 3, 4), np.float64)],
                inputs_as_nchw=[0],
            )
        assert not jax.config.x64_enabled
    with pytest.raises(SourcePrecisionError, match="static"):
        _source(lambda x: x, [("B", 2, 3, 4)], inputs_as_nchw=[0])
    for mode in ("prefer_native", "force_decomposed"):
        with pytest.raises(SourcePrecisionError, match="normalization_mode"):
            _source(
                lambda x: x, [(1, 2, 3, 4)], inputs_as_nchw=[0], normalization_mode=mode
            )
    with pytest.raises(SourcePrecisionError, match="primitive is outside the profile"):
        _source(lambda x: jnp.clip(x, 0, 1), [(1, 2, 3, 4)], inputs_as_nchw=[0])
    assert active_precision_policy() is PrecisionPolicy.LEGACY


@pytest.mark.parametrize("role", ["input", "output"])
def test_finalized_layout_shape_cannot_be_silently_reset_to_canonical(
    role: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalize = conversion_api._build_and_finalize_ir_model

    def corrupt(*args: Any, **kwargs: Any) -> Any:
        model = finalize(*args, **kwargs)
        boundary = model.graph.inputs if role == "input" else model.graph.outputs
        boundary[0].shape = ir.Shape((1, 2, 3, 4))
        return model

    monkeypatch.setattr(conversion_api, "_build_and_finalize_ir_model", corrupt)
    with pytest.raises(RuntimeError, match="Final.*shape"):
        _source(lambda x: -x, [(1, 2, 3, 4)], inputs_as_nchw=[0], outputs_as_nchw=[0])


def test_reloaded_layout_boundary_shape_remains_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load = onnx.load_model

    def corrupt(*args: Any, **kwargs: Any) -> Any:
        proto = load(*args, **kwargs)
        dims = proto.graph.input[0].type.tensor_type.shape.dim
        for dim, size in zip(dims, (1, 2, 3, 4), strict=True):
            dim.dim_value = size
        return proto

    monkeypatch.setattr(onnx, "load_model", corrupt)
    with pytest.raises(SourcePrecisionError, match="boundary dtype or shape"):
        _source(
            lambda x: -x,
            [(1, 2, 3, 4)],
            inputs_as_nchw=[0],
            outputs_as_nchw=[0],
            return_mode="file",
            output_path=tmp_path / "corrupt.onnx",
        )
