# tests/extra_tests/converter/test_source_complex_conj.py

"""Static SOURCE conjugation flips the declared imaginary component only."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function
from jax2onnx._compat.jax import ShapedArray, Var
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    _Site,
    _tensor,
    validate_source_jaxpr,
)
from jax2onnx.plugins.jax.lax.conj import ConjPlugin
from tests.extra_tests.converter import source_precision_support as source
from tests.extra_tests.converter.test_source_complex_add import _physical
from tests.extra_tests.converter.test_source_complex_broadcast import _export_original


_CONFIGURATIONS = [("complex64", False), ("complex64", True), ("complex128", True)]


def _payload(dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    if dtype == "complex64":
        words = np.asarray(
            [
                0x00000000,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7FC01234,
                0xFFC05678,
                0x00000001,
                0x80000001,
                0x007FFFFF,
                0x807FFFFF,
                0x00800000,
                0x80800000,
                0x3F800001,
                0xBF800001,
                0x7F7FFFFF,
                0xFF7FFFFF,
            ],
            dtype=np.uint32,
        )
        base = np.float32
    else:
        words = np.asarray(
            [
                0x0000000000000000,
                0x8000000000000000,
                0x7FF0000000000000,
                0xFFF0000000000000,
                0x7FF8000000001234,
                0xFFF8000000005678,
                0x0000000000000001,
                0x8000000000000001,
                0x000FFFFFFFFFFFFF,
                0x800FFFFFFFFFFFFF,
                0x0010000000000000,
                0x8010000000000000,
                0x3FF0000000000001,
                0xBFF0000000000001,
                0x7FEFFFFFFFFFFFFF,
                0xFFEFFFFFFFFFFFFF,
            ],
            dtype=np.uint64,
        )
        base = np.float64
    paired = np.stack(
        [np.resize(words, shape), np.resize(np.roll(words, 5), shape)], axis=-1
    )
    return paired.view(base).view(np.dtype(dtype)).reshape(shape).copy()


def _conjugate_bits(value: Any) -> np.ndarray:
    original = np.asarray(value)
    physical = _physical(original).copy()
    unsigned = np.uint32 if physical.dtype == np.dtype(np.float32) else np.uint64
    words = physical.view(unsigned)
    sign = np.asarray(1 << (physical.dtype.itemsize * 8 - 1), dtype=unsigned)
    words[..., 1] ^= sign
    # NaN masks are compared; retained NPZ payloads also expose NaN signs/bits
    # without requiring native/ORT NaN payload or sign preservation.
    return physical.view(original.dtype).reshape(original.shape)


@onnx_function
class _ComplexConjCapture:
    def __init__(self, capture: np.ndarray) -> None:
        self.capture = capture

    def __call__(self, x: Any) -> Any:
        transformed = jax.lax.conj(self.capture)
        return (
            x,
            transformed,
            jax.lax.real(self.capture),
            jax.lax.imag(self.capture),
            transformed,
        )


@pytest.mark.parametrize("dtype,x64", _CONFIGURATIONS)
@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize("api", ["lax", "jnp"])
@pytest.mark.parametrize("kind", ["scalar", "vector", "pair-tail", "empty"])
def test_original_static_complex_conjugation(
    dtype: str, x64: bool, opset: int, api: str, kind: str, tmp_path: Path
) -> None:
    shape = {"scalar": (), "vector": (16,), "pair-tail": (8, 2), "empty": (0, 2)}[kind]
    value = _payload(dtype, shape)
    original = jax.lax.conj if api == "lax" else jnp.conj
    before = bool(jax.config.jax_enable_x64)
    with jax.enable_x64(x64):
        _export_original(original, (value,), (_conjugate_bits(value),), tmp_path, opset)
    assert bool(jax.config.jax_enable_x64) is before


@pytest.mark.parametrize("dtype,x64", _CONFIGURATIONS)
@pytest.mark.parametrize("opset", [23, 26])
@pytest.mark.parametrize("named", [False, True])
def test_original_conjugation_capture_and_real_pair_sibling(
    dtype: str, x64: bool, opset: int, named: bool, tmp_path: Path
) -> None:
    capture = _payload(dtype, (32,))[::2]
    assert not capture.flags.c_contiguous
    frozen, strides = capture.tobytes(), capture.strides

    def original(x: Any) -> Any:
        transformed = jax.lax.conj(capture)
        return x, transformed, jax.lax.real(capture), jax.lax.imag(capture), transformed

    owner = _ComplexConjCapture(capture) if named else None
    fn = owner if owner is not None else original
    if owner is not None:
        assert owner.capture is capture and owner.capture.strides == strides
    ordinary = np.asarray([1.25, -0.0], dtype=np.float32)
    expected = _conjugate_bits(capture)
    with jax.enable_x64(x64):
        proto = _export_original(
            fn,
            (ordinary,),
            (ordinary, expected, capture.real.copy(), capture.imag.copy(), expected),
            tmp_path,
            opset,
        )
    assert bool(proto.functions) is named
    assert capture.tobytes() == frozen and capture.strides == strides
    if owner is not None:
        assert owner.capture is capture and owner.capture.strides == strides


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("weak", [False, True])
def test_actual_native_conjugation_weakness(
    x64: bool, weak: bool, tmp_path: Path
) -> None:
    with jax.enable_x64(x64):
        value = jnp.asarray(1.25 + 2.5j)
        if not weak:
            value = jnp.asarray(np.asarray(value))
        assert jax.typeof(value).weak_type is weak
        _export_original(
            jax.lax.conj, (value,), (_conjugate_bits(value),), tmp_path, 23
        )
    recording = json.loads((tmp_path / "native-recording.json").read_text())
    assert all(
        signature["leaves"][0]["weak_type"] is weak
        for signature in recording["signatures"].values()
    )


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("opset", [23, 26])
def test_original_registry_conjugation_row(
    x64: bool, opset: int, tmp_path: Path
) -> None:
    metadata = source.raw_source_case("plugin::conj", 1)
    original, callback = metadata["callable"], metadata["post_check_onnx_graph"]
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance("plugin::conj", 1)
    )[int(x64)]
    values = tuple(metadata["input_values"])
    frozen = tuple(np.asarray(value).tobytes() for value in values)
    with source.source_configuration(scenario):
        proto = _export_original(
            original, values, (_conjugate_bits(values[0]),), tmp_path, opset
        )
    callback_result = callback(proto)
    (tmp_path / "raw-registry-criteria.json").write_text(
        json.dumps(
            {"callback": bool(callback_result), "criterion": repr(scenario.criteria)}
        )
    )
    assert callback_result
    assert scenario.criteria is None
    assert source.raw_source_case("plugin::conj", 1)["callable"] is original
    assert (
        source.raw_source_case("plugin::conj", 1)["post_check_onnx_graph"] is callback
    )
    assert tuple(np.asarray(value).tobytes() for value in values) == frozen


@pytest.mark.parametrize(
    "defect",
    [
        "identity",
        "effects",
        "unknown",
        "placement",
        "parameter-width",
        "parameter-type",
        "missing",
        "width",
        "shape",
        "weakness",
        "parameter-container",
        "arity",
    ],
)
def test_complex_conjugation_rejects_forged_contract(defect: str) -> None:
    from jax2onnx.converter.source_profile import _validate_complex_conj

    equation = jax.make_jaxpr(jax.lax.conj)(
        jax.ShapeDtypeStruct((2,), np.complex64)
    ).jaxpr.eqns[0]
    changed = SimpleNamespace(
        primitive=equation.primitive,
        effects=equation.effects,
        params=dict(equation.params),
        invars=list(equation.invars),
        outvars=list(equation.outvars),
    )
    if defect == "identity":
        changed.primitive = SimpleNamespace(name="conj")
    elif defect == "effects":
        changed.effects = {"unexpected"}
    elif defect == "unknown":
        changed.params["unknown"] = None
    elif defect == "placement":
        changed.params["sharding"] = object()
    elif defect == "parameter-width":
        changed.params["input_dtype"] = np.dtype(np.complex128)
    elif defect == "parameter-type":
        changed.params["input_dtype"] = "complex64"
    elif defect == "missing":
        del changed.params["input_dtype"]
    elif defect == "width":
        changed.outvars[0] = Var(ShapedArray((2,), np.complex128))
    elif defect == "shape":
        changed.outvars[0] = Var(ShapedArray((3,), np.complex64))
    elif defect == "weakness":
        changed.outvars[0] = Var(ShapedArray((2,), np.complex64, weak_type=True))
    elif defect == "parameter-container":
        changed.params = []
    else:
        changed.outvars = []
    site = _Site("complex conjugation rejection", 23, "conj")
    with pytest.raises(SourcePrecisionError):
        _validate_complex_conj(
            changed,
            [_tensor(var, site) for var in changed.invars],
            [_tensor(var, site) for var in changed.outvars],
            site,
        )


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize(
    "defect", ["width", "missing-pair", "extra-pair", "unknown", "extent"]
)
def test_complex_conjugation_rejects_physical_drift_before_emission(
    dtype: str, defect: str
) -> None:
    with jax.enable_x64(dtype == "complex128"):
        equation = jax.make_jaxpr(jax.lax.conj)(
            jax.ShapeDtypeStruct((2,), np.dtype(dtype))
        ).jaxpr.eqns[0]
    base = ir.DataType.FLOAT if dtype == "complex64" else ir.DataType.DOUBLE
    value = ir.val("source", dtype=base, shape=(2, 2))
    if defect == "width":
        value.type = ir.TensorType(
            ir.DataType.DOUBLE if dtype == "complex64" else ir.DataType.FLOAT
        )
    else:
        value.shape = {
            "missing-pair": ir.Shape((2,)),
            "extra-pair": ir.Shape((2, 2, 2)),
            "unknown": None,
            "extent": ir.Shape((3, 2)),
        }[defect]
    before = value.type, value.shape

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid conjugation pair reached allocation or emission")

    context = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=SimpleNamespace(opset=23),
        require_value_for_var=lambda var: value,
        fresh_name=forbidden,
        allocate_value_for_var=forbidden,
        bind_value_for_var=forbidden,
    )
    with pytest.raises(TypeError, match="complex conjugation physical storage"):
        ConjPlugin().lower(context, equation)
    assert (value.type, value.shape) == before


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("opset", [21, 24])
def test_unselected_conjugation_opsets_stay_explicit(dtype: str, opset: int) -> None:
    with jax.enable_x64(dtype == "complex128"):
        trace = jax.make_jaxpr(jax.lax.conj)(
            jax.ShapeDtypeStruct((2,), np.dtype(dtype))
        )
        with pytest.raises(SourcePrecisionError):
            validate_source_jaxpr(trace, opset)


def test_real_conjugation_keeps_its_separate_source_obligation() -> None:
    trace = jax.make_jaxpr(jax.lax.conj)(jax.ShapeDtypeStruct((2,), np.float32))
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(trace, 23)
