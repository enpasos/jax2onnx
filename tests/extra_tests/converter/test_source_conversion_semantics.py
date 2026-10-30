# tests/extra_tests/converter/test_source_conversion_semantics.py

"""Explicit conversion boundaries with native edge references fixed first."""

import ctypes
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._source_capability import diagnostic_from_detail
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.precision import (
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    validate_source_samples,
)
from tests.extra_tests.converter.test_source_pointwise_mul_exp import _artifact, _accept
from tests.extra_tests.converter.source_precision_support import raw_source_case
from tests.extra_tests.converter.test_source_predicate_families import (
    _INTEGER_DTYPES,
    _FLOAT_DTYPES,
    _restore_context as _restore_context,
)


# Source calls/eager/JIT references are taken by _artifact before activation or
# export. These controls and criteria are fixed without reading target outputs.
_BITWISE = SampleCriteria("bitwise")
_NAN_CLASS = SampleCriteria("exact", equal_nan=True)


_BITCAST_GROUPS = (
    ("int8", "uint8"),
    ("float16", "bfloat16", "int16", "uint16"),
    ("float32", "int32", "uint32"),
    ("float64", "int64", "uint64"),
)


def _bitcast_values(dtype: str) -> np.ndarray:
    """Bitcast-specific payloads; NaN payload equality is part of this operation."""
    native = np.dtype(dtype)
    width = native.itemsize * 8
    sign = 1 << (width - 1)
    if native.kind in "iu":
        return _integer_edges(dtype)
    fraction = {"float16": 10, "bfloat16": 7, "float32": 23, "float64": 52}[dtype]
    exponent = ((1 << (width - fraction - 1)) - 1) << fraction
    # Both zeros, subnormal endpoints, minimum normal, finite neighbors of 1,
    # finite maximum, infinities and multiple signed quiet/signaling NaN words.
    one = {"float16": 15, "bfloat16": 127, "float32": 127, "float64": 1023}[dtype]
    words = [
        0,
        1,
        (1 << fraction) - 1,
        1 << fraction,
        (one << fraction) - 1,
        one << fraction,
        (one << fraction) + 1,
        exponent - 1,
        exponent,
        exponent | 1,
        exponent | (1 << (fraction - 1)) | 1,
        exponent | (1 << (fraction - 1)) | 2,
    ]
    words += [word | sign for word in words]
    return np.asarray(words, dtype=f"uint{width}").view(native)


def _bitcast_artifact(
    fn: Any, specs: list[Any], feeds: Any, path: Path, *, opset: int
) -> tuple[Any, Any]:
    # Keep actual eager, fresh JIT and warmed JIT output trees before export.
    # No source replacement or host numeric conversion supplies a reference.
    traces = 0

    def native(*values: Any) -> Any:
        nonlocal traces
        traces += 1
        return fn(*values)

    # This unique local callable has no previous JIT cache entry, including
    # when fn is an original registration retained by another test.
    compiled = jax.jit(native)
    signatures = set()
    snapshots = []
    for values in feeds:
        signature = (
            jax.tree.structure(values),
            tuple(
                (
                    tuple(value.shape),
                    np.dtype(value.dtype).name,
                    bool(getattr(value, "weak_type", False)),
                )
                for value in jax.tree.leaves(values)
            ),
        )
        eager = jax.tree.map(np.asarray, fn(*values))
        before = traces
        fresh = jax.tree.map(np.asarray, compiled(*values))
        assert traces == before + (signature not in signatures)
        signatures.add(signature)
        after = traces
        warmed = jax.tree.map(np.asarray, compiled(*values))
        assert traces == after
        assert jax.tree.structure(eager) == jax.tree.structure(fresh)
        assert jax.tree.structure(eager) == jax.tree.structure(warmed)
        for reference in (fresh, warmed):
            for actual, expected in zip(
                jax.tree.leaves(reference), jax.tree.leaves(eager), strict=True
            ):
                assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape)
                assert actual.tobytes() == expected.tobytes()
        snapshots.append(eager)
    _, proto, report, references = _artifact(
        fn, specs, feeds=feeds, mode="file", path=path, opset=opset
    )
    assert path.is_file()
    reloaded = onnx.load(path)
    assert reloaded.SerializeToString() == proto.SerializeToString()
    for expected, (eager, fresh) in zip(snapshots, references, strict=True):
        for reference in (eager, fresh):
            for actual, original in zip(
                reference, jax.tree.leaves(expected), strict=True
            ):
                assert (actual.dtype, actual.shape) == (original.dtype, original.shape)
                assert actual.tobytes() == original.tobytes()
    if opset == 27:
        # ONNX schema/structure admission is separate from this observed target
        # load failure. A passing diagnostic leaves numerics UNVERIFIED.
        outcomes = []
        for optimization in ("disabled", "all"):
            evidence = validate_source_samples(
                reloaded,
                samples=[
                    SourceSample(values, expected[0], _BITWISE)
                    for values, expected in zip(feeds, references, strict=True)
                ],
                optimization=optimization,
            )
            assert evidence.status is CheckStatus.FAILED
            assert evidence.artifact_sha256 is not None
            top = {check.category: check for check in evidence.checks}
            assert top["artifact-load"].status is CheckStatus.VALIDATED
            assert top["onnx-schema"].status is CheckStatus.VALIDATED
            assert top["runtime-load"].status is CheckStatus.FAILED
            assert len(evidence.samples) == 1
            checks = {check.category: check for check in evidence.samples[0].checks}
            assert checks["runtime-load"].status is CheckStatus.FAILED
            assert checks["numerical-acceptance"].status is CheckStatus.UNVERIFIED
            assert evidence.samples[0].outputs == ()
            diagnostic = diagnostic_from_detail(
                checks["runtime-load"].detail,
                artifact_sha256=evidence.artifact_sha256,
                providers_requested=("CPUExecutionProvider",),
            )
            assert diagnostic is not None and diagnostic.boundary == "runtime-load"
            assert diagnostic.opset_imports == (("", 27),)
            assert diagnostic.context["requested_opset"] == 27
            assert diagnostic.context["supported_opset"] == 26
            assert diagnostic.context["runtime_version"] == ort.__version__
            with pytest.raises(SampleAcceptanceError):
                evidence.require_sample_acceptance()
            outcomes.append(
                {
                    "optimization": optimization,
                    "runtime-load": "failed",
                    "numerical-acceptance": "unverified",
                    "capability": asdict(diagnostic),
                }
            )
        path.with_suffix(".runtime-load.json").write_text(
            json.dumps(outcomes, indent=2)
        )
        return reloaded, report
    if not any(
        np.asarray(value).dtype.name == "bfloat16"
        for values in feeds
        for value in values
    ) and not any(
        value.dtype.name == "bfloat16"
        for expected in snapshots
        for value in jax.tree.leaves(expected)
    ):
        # This existing acceptance helper pins CPU, disables fallback and runs
        # both ORT optimization settings against independent eager/JIT values.
        _accept(reloaded, feeds, references, _BITWISE)
        return reloaded, report
    # BF16's NumPy runtime binding cannot carry its dtype. Bind the original
    # uint16 storage with the actual ONNX type for inputs and outputs.
    for optimization in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = optimization
        session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        session.disable_fallback()
        assert session.get_providers() == ["CPUExecutionProvider"]
        for values, expected, reference_pair in zip(
            feeds, snapshots, references, strict=True
        ):
            binding = session.io_binding()
            storage = []
            input_values = []
            for value, meta, declared in zip(
                values, session.get_inputs(), reloaded.graph.input, strict=True
            ):
                array = np.asarray(value)
                assert (
                    onnx.helper.tensor_dtype_to_np_dtype(
                        declared.type.tensor_type.elem_type
                    )
                    == array.dtype
                )
                bits = (
                    array.view(np.uint16) if array.dtype.name == "bfloat16" else array
                )
                bits = np.array(bits, copy=True, order="C")
                storage.append(bits)
                input_value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                    bits, declared.type.tensor_type.elem_type
                )
                input_values.append(input_value)
                binding.bind_ortvalue_input(meta.name, input_value)
            output_contracts = {}
            for value, meta, declared in zip(
                jax.tree.leaves(expected),
                session.get_outputs(),
                reloaded.graph.output,
                strict=True,
            ):
                assert (
                    onnx.helper.tensor_dtype_to_np_dtype(
                        declared.type.tensor_type.elem_type
                    )
                    == value.dtype
                )
                contract = (
                    declared.type.tensor_type.elem_type,
                    value.shape,
                    value.nbytes,
                )
                if meta.name in output_contracts:
                    assert output_contracts[meta.name] == contract
                else:
                    output_contracts[meta.name] = contract
                    binding.bind_output(meta.name, "cpu")
            session.run_with_iobinding(binding)
            # Repeated graph output names share a binding. ORT can also return
            # an input alias without writing a requested NumPy output buffer.
            # Retain and inspect the actual returned CPU values in bind order.
            output_values = binding.get_outputs()
            actual_by_name = dict(zip(output_contracts, output_values, strict=True))
            for name, actual in actual_by_name.items():
                dtype, shape, byte_count = output_contracts[name]
                assert actual.device_name() == "cpu"
                assert actual.element_type() == dtype
                assert tuple(actual.shape()) == shape
                assert actual.tensor_size_in_bytes() == byte_count
            for native_outputs in reference_pair:
                for meta, reference in zip(
                    session.get_outputs(), native_outputs, strict=True
                ):
                    actual = actual_by_name[meta.name]
                    # The checked OrtValue owns the CPU allocation throughout
                    # this bounded read; BF16 never crosses a numeric cast.
                    if reference.nbytes:
                        pointer = actual.data_ptr()
                        assert pointer != 0
                        raw = ctypes.string_at(pointer, reference.nbytes)
                    else:
                        raw = b""
                    assert raw == reference.tobytes()
    return reloaded, report


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (source, target)
        for group in _BITCAST_GROUPS
        for source in group
        for target in group
    ],
)
def test_equal_width_bitcast_preserves_adversarial_storage_and_siblings(
    source: str, target: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        original = _bitcast_values(source)
        value = jnp.asarray(original)
        assert np.asarray(value).tobytes() == original.tobytes()

        def model(x: Any) -> Any:
            result = jax.lax.bitcast_convert_type(x, np.dtype(target))
            return result, x, result

        expected = original.view(np.dtype(target))
        eager = model(value)
        assert jax.tree.structure(eager) == jax.tree.structure(
            (expected, original, expected)
        )
        for actual, reference in zip(
            eager, (expected, original, expected), strict=True
        ):
            assert (actual.dtype, actual.shape) == (reference.dtype, reference.shape)
            assert np.asarray(actual).tobytes() == reference.tobytes()
        _, report = _bitcast_artifact(
            model,
            [jax.ShapeDtypeStruct(value.shape, np.dtype(source))],
            ((value,),),
            tmp_path / "bitcast.onnx",
            opset=26,
        )
        assert tuple(v.dtype for v in report.traced_outputs) == (target, source, target)


@pytest.mark.parametrize("ordinal", [0, 1, 2])
def test_original_bitcast_registrations_keep_callable_values_callback_and_opset(
    ordinal: int, tmp_path: Path
) -> None:
    case = raw_source_case("plugin::bitcast_convert_type", ordinal)
    original, callback = case["callable"], case["post_check_onnx_graph"]
    values = tuple(case["input_values"])
    snapshots = tuple(value.tobytes() for value in values)
    opset = case.get("opset_version", 23)
    with jax.enable_x64(True):
        proto, _ = _bitcast_artifact(
            original,
            [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values],
            (values,),
            tmp_path / "original.onnx",
            opset=opset,
        )
    assert callback(proto)
    assert opset == (23 if ordinal == 2 else 26)
    assert snapshots == (
        (
            np.asarray(3.5, np.float32).tobytes()
            if ordinal == 0
            else (
                np.asarray([0x3F800000, 0x40000000], np.uint32).tobytes()
                if ordinal == 1
                else np.asarray([1.5, -2.0], np.float32).tobytes()
            )
        ),
    )
    assert tuple(value.tobytes() for value in values) == snapshots
    reloaded = raw_source_case("plugin::bitcast_convert_type", ordinal)
    assert reloaded["callable"] is original
    assert reloaded["post_check_onnx_graph"] is callback


@pytest.mark.parametrize("opset", [23, 26, 27])
@pytest.mark.parametrize("dtype", [np.float32, np.bool_])
def test_identity_bitcast_saved_artifact_at_original_and_newer_opsets(
    opset: int, dtype: Any, tmp_path: Path
) -> None:
    value = (
        np.asarray([False, True, True, False], np.bool_)
        if dtype is np.bool_
        else _bitcast_values("float32")
    )
    proto, _ = _bitcast_artifact(
        lambda x: jax.lax.bitcast_convert_type(x, dtype),
        [jax.ShapeDtypeStruct(value.shape, dtype)],
        ((value,),),
        tmp_path / "identity.onnx",
        opset=opset,
    )
    assert all(node.op_type == "Identity" for node in proto.graph.node)
    assert proto.opset_import[0].version == opset


@pytest.mark.parametrize("shape", [(), (0, 3), "B, 2, 3"])
def test_opset27_bitcast_scalar_empty_and_native_symbolic_runtime_load_diagnostic(
    shape: Any, tmp_path: Path
) -> None:
    native_shape = export.symbolic_shape(shape) if isinstance(shape, str) else shape
    concrete_shapes = (
        [(batch, 2, 3) for batch in (1, 3)] if isinstance(shape, str) else [shape]
    )
    feeds = tuple(
        (np.resize(_bitcast_values("float32"), concrete),)
        for concrete in concrete_shapes
    )

    def native(x: Any) -> Any:
        if isinstance(shape, str):
            x = jax.lax.transpose(x, (1, 0, 2))
        return jax.lax.bitcast_convert_type(x, np.int32)

    proto, _ = _bitcast_artifact(
        native,
        [jax.ShapeDtypeStruct(native_shape, np.float32)],
        feeds,
        tmp_path / "native-shape.onnx",
        opset=27,
    )
    assert [node.op_type for node in proto.graph.node] == (
        ["Transpose", "BitCast"] if isinstance(shape, str) else ["BitCast"]
    )
    assert onnx.defs.get_schema("BitCast", 27).since_version == 26


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint64])
def test_bitcast_keeps_explicit_x64_input_conflict(dtype: Any) -> None:
    with jax.enable_x64(False):
        with pytest.raises(SourcePrecisionError) as captured:
            ui._to_onnx(
                lambda x: jax.lax.bitcast_convert_type(x, np.uint64),
                [jax.ShapeDtypeStruct((2,), dtype)],
                precision_policy=PrecisionPolicy.SOURCE,
                opset=26,
            )
        assert captured.value.category == "configuration-conflict"


def _integer_edges(dtype: str) -> np.ndarray:
    info = np.iinfo(dtype)
    candidates = [info.min, info.min + 1, info.max - 1, info.max, 0, 1]
    for value in (
        127,
        128,
        255,
        256,
        257,
        2049,
        2051,
        2**24 - 1,
        2**24 + 1,
        2**24 + 3,
        2**53 - 1,
        2**53 + 1,
        2**53 + 3,
        2**63 - 1,
        2**63 + 1,
        2**63 + 3,
    ):
        for signed in (value, -value):
            if info.min <= signed <= info.max:
                candidates.append(signed)
    return np.asarray(candidates, dtype=dtype)


def _floating_integer_edges(source: str, target: str, group: str) -> np.ndarray:
    if group == "fractional":
        values = [-1.75, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.75]
    elif group == "nonfinite":
        values = [-np.inf, np.inf, np.nan]
    else:
        info = np.iinfo(target)
        dtype = np.dtype(source)
        # The fixture records the source's actual representable neighbors.
        # Inf where FP16 cannot represent a bound is an explicit source input.
        with np.errstate(over="ignore", invalid="ignore"):
            bounds = np.asarray([info.min, info.max], dtype=dtype)
            values = [*bounds]
            for bound in bounds:
                values.extend(
                    (
                        np.nextafter(bound, np.asarray(-np.inf, dtype=dtype)),
                        np.nextafter(bound, np.asarray(np.inf, dtype=dtype)),
                    )
                )
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(values, dtype=source)


def _floating_subnormal_edges(source: str) -> np.ndarray:
    """Construct exact bit patterns without host arithmetic on subnormals."""
    width = np.dtype(source).itemsize * 8
    fraction = {16: 10, 32: 23, 64: 52}[width]
    bits = [0, 1, 2, (1 << fraction) - 1, 1 << fraction, (1 << fraction) + 1]
    # Include both destination rounding boundaries in the original wider dtype.
    exponent_bias = {16: 15, 32: 127, 64: 1023}[width]
    for exponent in (-14, -24, -25, -126, -149, -150):
        encoded_exponent = exponent + exponent_bias
        if encoded_exponent > 0:
            boundary = encoded_exponent << fraction
            bits.extend((boundary - 1, boundary, boundary + 1))
    bits += [value | (1 << (width - 1)) for value in bits]
    return np.asarray(bits, dtype=f"uint{width}").view(source)


@pytest.mark.parametrize("source", ["float16", "float32", "float64"])
@pytest.mark.parametrize("target", _INTEGER_DTYPES)
@pytest.mark.parametrize("group", ["fractional", "boundaries", "nonfinite"])
def test_float_to_integer_truncation_saturation_and_nan_zero(
    source: str, target: str, group: str, record_property: Any
) -> None:
    with jax.enable_x64(True):
        value = jnp.asarray(_floating_integer_edges(source, target, group))
        record_property(
            "native-reference-contract",
            "prepared JAX array, unchanged eager/JIT callable, original typed Jaxpr",
        )

        def model(x: Any) -> Any:
            return jax.lax.convert_element_type(x, np.dtype(target))

        value = jnp.asarray(value)
        feeds = [(value,)]
        _, proto, _, references = _artifact(
            model, [jax.ShapeDtypeStruct(value.shape, source)], feeds=feeds
        )
        _accept(proto, feeds, references, _BITWISE)


@pytest.mark.parametrize("source", _INTEGER_DTYPES)
@pytest.mark.parametrize(
    "target", (*_INTEGER_DTYPES, "float16", "float32", "float64", "bool")
)
def test_integer_sign_narrowing_endpoints_and_rounding_ties(
    source: str, target: str
) -> None:
    with jax.enable_x64(True):
        value = jnp.asarray(_integer_edges(source))

        def model(x: Any) -> Any:
            converted = jax.lax.convert_element_type(x, np.dtype(target))
            return converted, x, x

        value = jnp.asarray(value)
        feeds = [(value,)]
        _, proto, _, references = _artifact(
            model, [jax.ShapeDtypeStruct(value.shape, source)], feeds=feeds
        )
        # Integer endpoints are never routed through FP32/FP64 by the test or
        # by a comparison adapter; floating conversions use their actual native
        # rounding. The raw sibling outputs detect canonical-value mutation.
        _accept(proto, feeds, references, _BITWISE)


@pytest.mark.parametrize(
    "source", ["bool", *_INTEGER_DTYPES, "float16", "float32", "float64"]
)
def test_numeric_to_boolean_zero_nonzero_and_nan(source: str) -> None:
    with jax.enable_x64(True):
        if source == "bool":
            value = np.asarray([False, True])
        elif source in _INTEGER_DTYPES:
            value = jnp.asarray(_integer_edges(source))
        else:
            value = np.asarray(
                [-np.inf, -1, -0.0, 0.0, 1, np.inf, np.nan], dtype=source
            )

        def fn(x: Any) -> Any:
            return jax.lax.convert_element_type(x, np.bool_)

        value = jnp.asarray(value)
        feeds = [(value,)]
        _, proto, _, references = _artifact(
            fn, [jax.ShapeDtypeStruct(value.shape, source)], feeds=feeds
        )
        _accept(proto, feeds, references, _BITWISE)


@pytest.mark.parametrize("target", (*_INTEGER_DTYPES, "float16", "float32", "float64"))
def test_boolean_to_numeric_is_explicit_exact_data(target: str) -> None:
    with jax.enable_x64(True):
        value = np.asarray([False, True, True, False])

        def fn(x: Any) -> Any:
            return jax.lax.convert_element_type(x, np.dtype(target))

        value = jnp.asarray(value)
        feeds = [(value,)]
        _, proto, _, references = _artifact(
            fn, [jax.ShapeDtypeStruct(value.shape, bool)], feeds=feeds
        )
        _accept(proto, feeds, references, _BITWISE)


@pytest.mark.parametrize("source", ["float16", "float32", "float64"])
@pytest.mark.parametrize("target", ["float16", "float32", "float64"])
@pytest.mark.parametrize(
    "group", ["signed_zero", "ties", "nan_class", "subnormal_boundaries"]
)
def test_floating_conversion_rounding_and_strict_special_value_gates(
    source: str, target: str, group: str, record_property: Any
) -> None:
    with jax.enable_x64(True):
        if group == "signed_zero":
            value = np.asarray(
                [
                    -0.0,
                    0.0,
                    -np.finfo(source).smallest_subnormal,
                    np.finfo(source).smallest_subnormal,
                ],
                dtype=source,
            )
        elif group == "subnormal_boundaries":
            value = _floating_subnormal_edges(source)
        elif group == "nan_class":
            value = np.asarray([np.nan, np.inf, -np.inf], dtype=source)
        else:
            eps = np.finfo(target).eps
            midpoint = np.float64(1) + np.float64(eps) / 2
            value = np.asarray(
                [
                    midpoint,
                    np.nextafter(midpoint, -np.inf),
                    np.nextafter(midpoint, np.inf),
                    -midpoint,
                    65504,
                    65520,
                ],
                dtype=source,
            )

        def fn(x: Any) -> Any:
            return jax.lax.convert_element_type(x, np.dtype(target))

        value = jnp.asarray(value)
        feeds = [(value,)]
        _, proto, _, references = _artifact(
            fn, [jax.ShapeDtypeStruct(value.shape, source)], feeds=feeds
        )
        if (source, target, group) in {
            ("float32", "float64", "signed_zero"),
            ("float32", "float64", "subnormal_boundaries"),
            ("float64", "float32", "subnormal_boundaries"),
        }:
            # The measured source CPU execution and recorded target differ at
            # FP32 subnormal widening/destination rounding. Keep plain Cast and
            # report this exact runtime-bound mismatch as numerical failure.
            # This is neither portable FTZ policy nor numerical acceptance.
            for optimization in ("disabled", "all"):
                for reference in references[0]:
                    evidence = validate_source_samples(
                        proto,
                        samples=[SourceSample((value,), reference, _BITWISE)],
                        optimization=optimization,
                    )
                    record_property(
                        f"subnormal_cpu_target_{optimization}", repr(evidence.target)
                    )
                    record_property(
                        f"subnormal_cpu_artifact_{optimization}",
                        evidence.artifact_sha256,
                    )
                    assert evidence.status is CheckStatus.FAILED
                    sample = evidence.samples[0]
                    assert sample.outputs[0].dtype == sample.references[0].dtype
                    assert sample.outputs[0].shape == sample.references[0].shape
                    assert sample.outputs[0].sha256 != sample.references[0].sha256
                    assert any(
                        check.category == "runtime-execution"
                        and check.status is CheckStatus.VALIDATED
                        for check in sample.checks
                    )
                    assert any(
                        check.category == "numerical-acceptance"
                        and check.status is CheckStatus.FAILED
                        and "bitwise comparison" in check.detail
                        for check in sample.checks
                    )
                    with pytest.raises(
                        SampleAcceptanceError, match="bitwise comparison"
                    ):
                        evidence.require_sample_acceptance()
            return
        _accept(
            proto, feeds, references, _NAN_CLASS if group == "nan_class" else _BITWISE
        )


@pytest.mark.parametrize("source", _FLOAT_DTYPES)
@pytest.mark.parametrize("target", _FLOAT_DTYPES)
def test_bfloat16_conversion_has_typed_evidence_without_a_cpu_claim(
    source: str, target: str
) -> None:
    with jax.enable_x64(True):

        def fn(x: Any) -> Any:
            return jax.lax.convert_element_type(x, jnp.dtype(target))

        _, _, report, _ = _artifact(fn, [jax.ShapeDtypeStruct((2, 3), source)])
        assert report.traced_outputs[0].dtype == target


def test_deliberate_rounding_boundaries_and_sibling_aliases_survive() -> None:
    with jax.enable_x64(True):
        value = np.asarray([2049, 2051, 1 + 2**-11, -0.0], np.float64)

        def model(x: Any) -> Any:
            rounded = jax.lax.convert_element_type(x, np.float16)
            widened = jax.lax.convert_element_type(rounded, np.float64)
            return x, rounded, widened, rounded, x

        value = jnp.asarray(value)
        feeds = [(value,)]
        _, proto, _, references = _artifact(
            model, [jax.ShapeDtypeStruct(value.shape, np.float64)], feeds=feeds
        )
        _accept(proto, feeds, references, _BITWISE)
        cast_types = [
            node.attribute[0].i for node in proto.graph.node if node.op_type == "Cast"
        ]
        assert 10 in cast_types and 11 in cast_types


@pytest.mark.parametrize("source", [np.float32, np.float64])
def test_saturating_uint64_carrier_preserves_raw_and_rounded_siblings(
    source: Any,
) -> None:
    with jax.enable_x64(True):
        value = jnp.asarray(
            np.asarray([0, 2**63, 2**64 - 1, np.inf, -np.inf, np.nan], source)
        )

        def model(x: Any) -> Any:
            converted = jax.lax.convert_element_type(x, np.uint64)
            sibling = jax.lax.convert_element_type(x, np.float16)
            return converted, x, sibling, converted, x

        feeds = [(value,)]
        _, proto, _, references = _artifact(
            model, [jax.ShapeDtypeStruct(value.shape, source)], feeds=feeds
        )
        _accept(proto, feeds, references, _BITWISE)


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_same_dtype_native_weakness_change_remains_in_the_trace(dtype: Any) -> None:
    with jax.enable_x64(dtype is np.float64):

        def model(x: Any) -> Any:
            return x, jnp.asarray(x, dtype=dtype)

        value = jnp.asarray(1 if dtype is np.int32 else 1.25)
        assert value.dtype == np.dtype(dtype) and value.weak_type
        spec = jax.ShapeDtypeStruct((), dtype, weak_type=True)
        _, proto, report, references = _artifact(model, [spec], feeds=[(value,)])
        assert [output.weak_type for output in report.traced_outputs] == [True, False]
        _accept(proto, [(value,)], references, _BITWISE)


@pytest.mark.parametrize(
    "source,target", [(np.int64, np.uint64), (np.uint64, np.int64)]
)
def test_signed_unsigned_64_bit_roundtrips_preserve_every_carrier_bit(
    source: Any, target: Any
) -> None:
    with jax.enable_x64(True):
        value = jnp.asarray(_integer_edges(np.dtype(source).name))

        def model(x: Any) -> Any:
            converted = jax.lax.convert_element_type(x, target)
            returned = jax.lax.convert_element_type(converted, source)
            return x, converted, returned, converted, x

        feeds = [(value,)]
        _, proto, _, references = _artifact(
            model, [jax.ShapeDtypeStruct(value.shape, source)], feeds=feeds
        )
        assert sum(node.op_type == "Cast" for node in proto.graph.node) >= 2
        _accept(proto, feeds, references, _BITWISE)


@pytest.mark.parametrize(
    "source,target",
    [
        (np.int32, np.int16),
        (np.uint64, np.int64),
        (bool, np.float32),
        (np.float32, np.int32),
    ],
)
def test_symbolic_typed_conversions_keep_one_artifact_for_two_batch_sizes(
    source: Any, target: Any
) -> None:
    with jax.enable_x64(True):
        shape = export.symbolic_shape("B, 3")

        def fn(x: Any) -> Any:
            return jax.lax.convert_element_type(x, target)

        feeds = [
            (jnp.asarray(np.arange(batch * 3).reshape(batch, 3).astype(source)),)
            for batch in (2, 5)
        ]
        _, proto, _, references = _artifact(
            fn, [jax.ShapeDtypeStruct(shape, source)], feeds=feeds
        )
        _accept(proto, feeds, references, _BITWISE)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_weakness",
        "non_bool_weakness",
        "string_dtype",
        "wrong_dtype",
        "placement",
        "unknown",
    ],
)
def test_conversion_native_parameter_closure_is_checked_before_lowering(
    mutation: str,
) -> None:
    native = jax.make_jaxpr(lambda x: jax.lax.convert_element_type(x, np.int16))(
        jax.ShapeDtypeStruct((2,), np.float32)
    ).jaxpr
    equation = native.eqns[0]
    params = dict(equation.params)
    if mutation == "missing_weakness":
        params.pop("weak_type")
    elif mutation == "non_bool_weakness":
        params["weak_type"] = 0
    elif mutation == "string_dtype":
        params["new_dtype"] = "int16"
    elif mutation == "wrong_dtype":
        params["new_dtype"] = np.dtype(np.int32)
    elif mutation == "placement":
        params["sharding"] = object()
    else:
        params["unreviewed"] = None
    corrupted = native.replace(eqns=[equation.replace(params=params)])
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(corrupted, 23)
    assert error.value.category == (
        "profile-exclusion"
        if mutation in {"placement", "unknown"}
        else "lowering-defect"
    )


def test_numpy_eager_overflow_disagreement_retains_its_strict_failure_gate(
    record_property: Any,
) -> None:
    """Host ndarray.astype and the traced JAX conversion are distinct references.

    Retain the exact original first-failure input/reference, without treating a
    target match to the compiled source as a match to the host NumPy fast path.
    The main conversion controls use the JAX-array lane frozen in native123.
    """
    with jax.enable_x64(True):
        value = _floating_integer_edges("float32", "int64", "nonfinite")

        def model(x: Any) -> Any:
            return jax.lax.convert_element_type(x, np.int64)

        _, proto, _, references = _artifact(
            model, [jax.ShapeDtypeStruct(value.shape, np.float32)], feeds=[(value,)]
        )
        eager, compiled = references[0]
        disagreement = eager[0].tobytes() != compiled[0].tobytes()
        record_property("native-numpy-eager-jit-disagreement", disagreement)
        for optimization in ("disabled", "all"):
            validate_source_samples(
                proto,
                samples=[SourceSample((value,), compiled, _BITWISE)],
                optimization=optimization,
            ).require_sample_acceptance()
            host = validate_source_samples(
                proto,
                samples=[SourceSample((value,), eager, _BITWISE)],
                optimization=optimization,
            )
            if disagreement:
                assert host.status is CheckStatus.FAILED
                with pytest.raises(SampleAcceptanceError):
                    host.require_sample_acceptance()
            else:
                host.require_sample_acceptance()
