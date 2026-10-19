# tests/extra_tests/converter/test_source_typed_value_boundaries.py

"""Typed product boundaries/scopes and the small static complex foundation."""

from typing import Any
import json

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.precision import (
    CheckStatus,
    PRECISION_METADATA_KEY,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)
from tests.extra_tests.converter.test_source_pointwise_mul_exp import _artifact, _accept
from tests.extra_tests.converter.test_source_predicate_families import (
    _INTEGER_DTYPES,
    _FLOAT_DTYPES,
    _restore_context as _restore_context,
)


_DTYPES = ("bool", *_INTEGER_DTYPES, *_FLOAT_DTYPES, "complex64", "complex128")
_BITWISE = SampleCriteria("bitwise")


def _accept_declared_storage(proto: Any, feeds: Any, references: Any) -> None:
    """Use the established sample API's physical representation protocol."""
    report = get_precision_report(proto)
    assert report is not None

    def encode(value: Any, declaration: Any) -> np.ndarray:
        array = np.asarray(value)
        assert array.dtype == np.dtype(declaration.dtype)
        if np.issubdtype(array.dtype, np.complexfloating):
            # Complex admission here is static. The declared logical type,
            # rather than a final axis of size two, selects channel storage.
            return representation_for_aval(declaration).encode_array(array)
        return array

    physical_feeds = [
        tuple(
            encode(value, entry.traced)
            for value, entry in zip(values, report.inputs, strict=True)
        )
        for values in feeds
    ]
    physical_references = [
        tuple(
            tuple(
                encode(value, declaration)
                for value, declaration in zip(
                    outputs, report.traced_outputs, strict=True
                )
            )
            for outputs in pair
        )
        for pair in references
    ]
    _accept(proto, physical_feeds, physical_references, _BITWISE)


def _value(dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    if dtype.startswith("complex"):
        base = np.float32 if dtype == "complex64" else np.float64
        value = np.empty(shape, dtype=dtype)
        value.real = np.asarray(-0.0, base)
        value.imag = np.asarray(1.5, base)
        return value
    if dtype in _INTEGER_DTYPES:
        info = np.iinfo(dtype)
        return np.full(shape, info.max - 1, dtype=dtype)
    return np.full(shape, True if dtype == "bool" else -0.0, dtype=dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", [(), (0, 3), (2, 3)])
@pytest.mark.parametrize("opset", [23, 26])
def test_typed_inputs_captures_outputs_and_repeated_aliases(
    dtype: str, shape: tuple[int, ...], opset: int
) -> None:
    with jax.enable_x64(True):
        value = _value(dtype, shape)
        captured = value.copy()
        before = captured.tobytes()

        def model(x: Any) -> Any:
            return x, x, captured, captured

        feeds = [(value,)]
        _, proto, report, references = _artifact(
            model, [jax.ShapeDtypeStruct(shape, dtype)], feeds=feeds, opset=opset
        )
        representation = representation_for_aval(jax.ShapeDtypeStruct(shape, dtype))
        assert proto.graph.input[0].type.tensor_type.elem_type == int(
            representation.ir_dtype
        )
        assert len(proto.graph.input[0].type.tensor_type.shape.dim) == len(
            representation.physical_shape
        )
        assert all(
            output.dtype == dtype and output.shape == shape
            for output in report.traced_outputs
        )
        assert captured.tobytes() == before
        if dtype != "bfloat16":
            _accept_declared_storage(proto, feeds, references)


@onnx_function(name="TypedBoundaryInner")
def _inner(x: Any, capture: Any) -> Any:
    return x, capture, x, capture


@onnx_function(name="TypedBoundaryOuter")
def _outer(x: Any, capture: Any) -> Any:
    return jax.jit(_inner)(x, capture)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("scope", ["named", "jit", "cond"])
def test_actual_typed_named_jit_and_conditional_signatures(
    dtype: str, scope: str
) -> None:
    with jax.enable_x64(True):
        value, capture = _value(dtype, (2, 3)), _value(dtype, (2, 3))
        if dtype in _INTEGER_DTYPES:
            capture = np.zeros_like(value)

        def body(x: Any) -> Any:
            return x, capture, x, capture

        def model(x: Any, flag: Any) -> Any:
            if scope == "named":
                return _outer(x, capture)
            if scope == "jit":
                return jax.jit(body)(x)
            return jax.lax.cond(flag, body, lambda y: (capture, y, capture, y), x)

        specs = [jax.ShapeDtypeStruct((2, 3), dtype), jax.ShapeDtypeStruct((), bool)]
        feeds = [(value, np.asarray(flag)) for flag in (False, True)]
        _, proto, _, references = _artifact(
            model, specs, feeds=feeds, named=scope == "named"
        )
        if dtype != "bfloat16":
            _accept_declared_storage(proto, feeds, references)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.complex128])
def test_preexisting_captured_64_bit_payload_survives_local_x64_off(dtype: Any) -> None:
    with jax.enable_x64(True):
        captured = jnp.asarray(_value(np.dtype(dtype).name, (2, 3)))
    captured_bytes = np.asarray(captured).tobytes()

    def model(x: Any) -> Any:
        return x, captured, captured

    with jax.enable_x64(False):
        feeds = [(np.asarray([1.0, -0.0], np.float32),)]
        _, proto, report, references = _artifact(
            model, [jax.ShapeDtypeStruct((2,), np.float32)], feeds=feeds
        )
        assert report.traced_outputs[1].dtype == np.dtype(dtype).name
        assert references[0][0][1].tobytes() == captured_bytes
        _accept_declared_storage(proto, feeds, references)
    assert np.asarray(captured).tobytes() == captured_bytes


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("mode", ["ir", "proto", "file"])
def test_complex_construction_extraction_and_same_shaped_real_sibling(
    dtype: Any, mode: str, tmp_path: Any
) -> None:
    base = np.float32 if dtype is np.complex64 else np.float64
    with jax.enable_x64(True):
        # Assign channels separately: complex arithmetic would itself change
        # NaN/zero channels before the conversion under test.
        z = np.empty((2,), dtype=dtype)
        z.real = np.asarray([-0.0, np.nan], base)
        z.imag = np.asarray([np.nan, -0.0], base)
        real_pair = np.asarray([[-0.0, 2], [3, -0.0]], base)

        def model(value: Any, ordinary: Any) -> Any:
            real = jax.lax.real(value)
            imag = jax.lax.imag(value)
            constructed = jax.lax.complex(real, imag)
            return value, real, imag, constructed, value, ordinary, ordinary

        specs = [jax.ShapeDtypeStruct((2,), dtype), jax.ShapeDtypeStruct((2, 2), base)]
        feeds = [(z, real_pair)]
        _, proto, report, references = _artifact(
            model,
            specs,
            feeds=feeds,
            mode=mode,
            path=tmp_path / "complex-foundation.onnx" if mode == "file" else None,
        )
        assert [value.dtype for value in report.traced_outputs] == [
            np.dtype(dtype).name,
            np.dtype(base).name,
            np.dtype(base).name,
            np.dtype(dtype).name,
            np.dtype(dtype).name,
            np.dtype(base).name,
            np.dtype(base).name,
        ]
        _accept_declared_storage(proto, feeds, references)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize(
    "real_shape,imag_shape",
    [((2, 1), (1, 3)), ((), (2, 3)), ((2, 3), ()), ((0, 1), (1, 3))],
    ids=["cross-broadcast", "scalar-array", "array-scalar", "empty-broadcast"],
)
@pytest.mark.parametrize("capture_imag", [False, True])
def test_complex_construction_broadcast_keeps_component_and_capture_siblings(
    dtype: Any,
    opset: int,
    real_shape: tuple[int, ...],
    imag_shape: tuple[int, ...],
    capture_imag: bool,
) -> None:
    with jax.enable_x64(True):
        real = np.full(real_shape, -0.0, dtype)
        imag = np.resize(np.asarray([-0.0, -3.0, np.nan], dtype), imag_shape)
        captured = np.empty(
            (2, 2), np.complex64 if dtype is np.float32 else np.complex128
        )
        captured.real = np.asarray([[-0.0, 3.0], [5.0, -0.0]], dtype)
        captured.imag = np.asarray([[np.nan, -0.0], [-2.0, 7.0]], dtype)
        before = real.tobytes(), imag.tobytes(), captured.tobytes()

        def build(left: Any, right: Any) -> Any:
            value = jax.lax.complex(left, right)
            mask = jnp.less(right, np.asarray(0.0, dtype))
            selected = jnp.where(mask, right, np.zeros(imag_shape, dtype))
            return (
                value,
                value,
                jax.lax.real(value),
                jax.lax.imag(value),
                left,
                left,
                right,
                right,
                captured,
                captured,
                jax.lax.real(captured),
                jax.lax.imag(captured),
                mask,
                jnp.bitwise_not(mask),
                selected,
            )

        if capture_imag:

            def model(left: Any) -> Any:
                return build(left, imag)

            feeds = [(real,)]
        else:
            model = build
            feeds = [(real, imag)]
        specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in feeds[0]]
        _, proto, report, references = _artifact(model, specs, feeds=feeds, opset=opset)
        logical_shape = tuple(np.broadcast_shapes(real_shape, imag_shape))
        assert report.traced_outputs[0].shape == logical_shape
        assert report.traced_outputs[0].dtype == np.dtype(captured.dtype).name
        assert sum(node.op_type == "Expand" for node in proto.graph.node) == sum(
            shape != logical_shape for shape in (real_shape, imag_shape)
        )
        _accept_declared_storage(proto, feeds, references)
        assert (real.tobytes(), imag.tobytes(), captured.tobytes()) == before


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_complex_construction_rejects_native_unequal_non_scalar_ranks(
    dtype: Any,
) -> None:
    with jax.enable_x64(True):
        values = (
            jnp.asarray(np.full((3,), -0.0, dtype)),
            jnp.asarray(np.resize(np.asarray([-0.0, -3.0, np.nan], dtype), (2, 1))),
        )
        specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values]
        error = "arrays must have the same number of dimensions"
        # Rank-zero operands have a native scalar broadcast exception. Unequal
        # positive ranks remain a native error before SOURCE admission/lowering.
        for call in (jax.lax.complex, jax.jit(jax.lax.complex)):
            with pytest.raises(TypeError, match=error):
                call(*values)
        with pytest.raises(TypeError, match=error):
            jax.make_jaxpr(jax.lax.complex)(*specs)
        with pytest.raises(TypeError, match=error):
            ui._to_onnx(
                jax.lax.complex,
                specs,
                precision_policy=PrecisionPolicy.SOURCE,
                return_mode="proto",
                opset=23,
            )


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.complex128])
def test_x64_configuration_conflict_precedes_any_typed_trace(dtype: Any) -> None:
    calls = []
    with jax.enable_x64(False), pytest.raises(SourcePrecisionError) as error:
        ui._to_onnx(
            lambda x: calls.append(True) or x,
            [jax.ShapeDtypeStruct((2,), dtype)],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert error.value.category == "configuration-conflict" and not calls


@pytest.mark.parametrize(
    "kind",
    [
        "symbolic_complex",
        "rank_one",
        "integer_switch",
        "integer_add",
        "complex_add",
        "shift",
        "rank_five",
    ],
)
def test_new_typed_boundaries_do_not_admit_unreviewed_semantics(kind: str) -> None:
    (batch,) = export.symbolic_shape("B")
    spec = jax.ShapeDtypeStruct((2,), np.int32)
    if kind == "symbolic_complex":
        spec = jax.ShapeDtypeStruct((batch, 3), np.complex64)
    elif kind == "rank_one":
        spec = jax.ShapeDtypeStruct((batch,), bool)
    elif kind == "rank_five":
        spec = jax.ShapeDtypeStruct((batch, 2, 3, 4, 5), bool)
    elif kind == "integer_switch":
        spec = jax.ShapeDtypeStruct((), np.int32)
    elif kind == "complex_add":
        spec = jax.ShapeDtypeStruct((2,), np.complex64)

    def fn(x: Any) -> Any:
        if kind in {"symbolic_complex", "rank_one", "rank_five"}:
            return x
        if kind == "integer_switch":
            return jax.lax.switch(
                x, (lambda: np.float32(1), lambda: np.float32(2), lambda: np.float32(3))
            )
        if kind == "shift":
            return jax.lax.shift_left(x, np.asarray([1, 1], np.int32))
        return x + x

    if kind == "rank_one":
        # Keep the original BOOL(B,) identity and parameter ID as its positive
        # counterpart, sharing one artifact across both concrete batch sizes.
        feeds = [(np.arange(size) % 3 == 1,) for size in (2, 5)]
        _, proto, report, references = _artifact(fn, [spec], feeds=feeds)
        assert (
            report.inputs[0].traced.shape == report.exported_outputs[0].shape == ("B",)
        )
        assert (
            report.inputs[0].traced.dtype == report.exported_outputs[0].dtype == "bool"
        )
        assert proto.graph.input[0].type.tensor_type.elem_type == 9
        assert proto.graph.output[0].type.tensor_type.elem_type == 9
        _accept_declared_storage(proto, feeds, references)
        return
    with pytest.raises(SourcePrecisionError) as error:
        ui._to_onnx(fn, [spec], precision_policy=PrecisionPolicy.SOURCE)
    assert error.value.category == "profile-exclusion"


def test_boolean_to_int32_is_returned_data_but_not_an_arbitrary_cond_selector() -> None:
    def data(flag: Any) -> Any:
        return jax.lax.convert_element_type(flag, np.int32)

    feeds = [(np.asarray(flag),) for flag in (False, True)]
    _, proto, _, references = _artifact(
        data, [jax.ShapeDtypeStruct((), bool)], feeds=feeds
    )
    _accept(proto, feeds, references, _BITWISE)
    # Native numeric cond first builds a Boolean predicate; that remains valid.
    # A forged integer selector with no audited Boolean origin still fails.
    from tests.extra_tests.converter.test_source_profile import _graph, _spec

    graph = _graph("cond", [_spec((), np.int32)], [_spec()], {"branches": ()})
    with pytest.raises(
        SourcePrecisionError, match="lacks an audited Boolean conversion"
    ):
        validate_source_jaxpr(graph, 23)


def test_ordinary_integer_and_symbolic_boolean_old_exclusion_counterparts() -> None:
    # Exact positive counterparts of the accepted Where packet's now-obsolete
    # integer-port, symbolic-BOOL and BOOL->floating exclusions.
    def model(mask: Any, value: Any) -> Any:
        return (
            jnp.where(mask, value, value),
            mask,
            jax.lax.convert_element_type(mask, np.float32),
        )

    shape = export.symbolic_shape("B, 3")
    specs = [jax.ShapeDtypeStruct(shape, bool), jax.ShapeDtypeStruct(shape, np.int32)]
    feeds = [
        (
            np.arange(batch * 3).reshape(batch, 3) % 2 == 0,
            np.arange(batch * 3, dtype=np.int32).reshape(batch, 3),
        )
        for batch in (2, 5)
    ]
    _, proto, _, references = _artifact(model, specs, feeds=feeds)
    _accept(proto, feeds, references, _BITWISE)


@pytest.mark.parametrize("dtype", [bool, np.int32, np.uint64])
@pytest.mark.parametrize(
    "corruption",
    ["batch", "tail", "dtype", "reference_batch", "artifact_symbol", "report_symbol"],
)
def test_typed_symbolic_samples_reject_malformed_contracts_before_ort(
    dtype: Any, corruption: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with jax.enable_x64(True):
        shape = export.symbolic_shape("B, 3")

        def model(left: Any, right: Any) -> Any:
            return jnp.bitwise_xor(left, right)

        _, proto, report, _ = _artifact(
            model,
            [jax.ShapeDtypeStruct(shape, dtype), jax.ShapeDtypeStruct(shape, dtype)],
        )
        values = [np.ones((2, 3), dtype), np.zeros((2, 3), dtype)]
        expected = np.ones((2, 3), dtype)
        if corruption == "batch":
            values[1] = np.zeros((5, 3), dtype)
        elif corruption == "tail":
            values[1] = np.zeros((2, 4), dtype)
        elif corruption == "dtype":
            values[1] = values[1].astype(
                np.int64 if dtype is not np.int32 else np.uint32
            )
        elif corruption == "reference_batch":
            expected = np.ones((5, 3), dtype)
        elif corruption == "artifact_symbol":
            proto.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "Other"
        else:
            metadata = json.loads(precision_report_to_json(report))
            metadata["inputs"][1]["exported"]["shape"][0] = "Other"
            for entry in proto.metadata_props:
                if entry.key == PRECISION_METADATA_KEY:
                    entry.value = json.dumps(metadata)

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("malformed typed samples must fail before an ORT session")

        monkeypatch.setattr(ort, "InferenceSession", forbidden)
        result = validate_source_samples(
            proto,
            samples=[SourceSample(values, [expected], _BITWISE)],
            optimization="disabled",
        )
        assert result.status is CheckStatus.FAILED and result.target is None
        with pytest.raises(SampleAcceptanceError):
            result.require_sample_acceptance()
