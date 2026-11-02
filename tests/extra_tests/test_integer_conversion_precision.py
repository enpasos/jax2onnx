# tests/extra_tests/test_integer_conversion_precision.py

"""Floating-to-8-bit conversions saturate before truncation and map NaN to zero."""

from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx
from jax2onnx._compat.jax import explicit_x64_dtypes


@pytest.fixture(autouse=True)
def _enable_jax_x64() -> Iterator[None]:
    with jax.enable_x64():
        yield


def _native_reference(
    function: Callable[[jax.Array], jax.Array], values: np.ndarray
) -> np.ndarray:
    argument = jnp.asarray(values)
    eager = np.asarray(function(argument))

    def call(value: jax.Array) -> jax.Array:
        return function(value)

    compiled = jax.jit(call)
    fresh = np.asarray(compiled(argument))
    warm = np.asarray(compiled(argument))
    np.testing.assert_array_equal(fresh, eager)
    np.testing.assert_array_equal(warm, eager)
    assert fresh.dtype == warm.dtype == eager.dtype
    return eager


def _run(model: onnx.ModelProto, values: np.ndarray, optimized: bool) -> np.ndarray:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimized
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    input_type = model.graph.input[0].type.tensor_type.elem_type
    if input_type == onnx.TensorProto.BFLOAT16:
        bits = np.ascontiguousarray(values.view(np.uint16))
        binding = session.io_binding()
        binding.bind_ortvalue_input(
            session.get_inputs()[0].name,
            ort.OrtValue.ortvalue_from_numpy_with_onnx_type(bits, input_type),
        )
        binding.bind_output(session.get_outputs()[0].name, "cpu")
        session.run_with_iobinding(binding)
        (actual,) = binding.copy_outputs_to_cpu()
    else:
        input_dtype = onnx.helper.tensor_dtype_to_np_dtype(input_type)
        assert input_dtype == values.dtype
        (actual,) = session.run(None, {session.get_inputs()[0].name: values})
    return np.asarray(actual)


@pytest.mark.parametrize("source_name", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("target_dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_float_to_8bit_saturates_boundaries_and_exceptional_values(
    source_name: str, target_dtype: type, enable_x64: bool, optimized: bool
) -> None:
    source_dtype = getattr(jnp, source_name)
    largest = float(jnp.finfo(source_dtype).max)
    values = np.asarray(
        jnp.asarray(
            [
                -largest,
                -129.0,
                -128.5,
                -128.0,
                -127.5,
                -1.0,
                -0.0,
                0.0,
                0.5,
                126.999,
                127.5,
                128.0,
                129.0,
                254.999,
                255.5,
                256.0,
                257.0,
                largest,
                -np.inf,
                np.inf,
                np.nan,
            ],
            dtype=source_dtype,
        )
    )

    def convert(x: jax.Array) -> jax.Array:
        return jax.lax.convert_element_type(x, target_dtype)

    with jax.enable_x64(enable_x64), explicit_x64_dtypes("warn"):
        if source_name == "float64" and not enable_x64:
            # The former export option narrowed this same FP64 boundary vector.
            # The source contract now rejects that loss before tracing.
            with pytest.raises(ValueError, match="would be narrowed"):
                to_onnx(convert, [jax.ShapeDtypeStruct(values.shape, source_dtype)])
            return

        expected = _native_reference(convert, values)
        limits = np.iinfo(target_dtype)
        sanitized = np.nan_to_num(
            values.astype(np.float64), nan=0, posinf=limits.max, neginf=limits.min
        )
        oracle = np.trunc(np.clip(sanitized, limits.min, limits.max)).astype(
            target_dtype
        )
        np.testing.assert_array_equal(expected, oracle)

        model = to_onnx(
            convert,
            [jax.ShapeDtypeStruct(values.shape, source_dtype)],
        )
        actual = _run(model, values, optimized)

        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype == target_dtype
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            actual[np.isnan(values)], np.zeros(1, target_dtype)
        )
        assert actual[np.isneginf(values)].item() == limits.min
        assert actual[np.isposinf(values)].item() == limits.max


@pytest.mark.parametrize("target_dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("optimized", [False, True])
def test_float64_conversion_retains_values_next_to_integer_boundaries(
    target_dtype: type, optimized: bool
) -> None:
    boundaries = np.asarray(
        [-128.0, -127.0, 0.0, 1.0, 126.0, 127.0, 128.0, 254.0, 255.0]
    )
    values = np.stack(
        [
            np.nextafter(boundaries, -np.inf),
            boundaries,
            np.nextafter(boundaries, np.inf),
        ]
    ).reshape(-1)

    def convert(x: jax.Array) -> jax.Array:
        return jax.lax.convert_element_type(x, target_dtype)

    expected = _native_reference(convert, values)
    model = to_onnx(convert, [values])
    actual = _run(model, values, optimized)

    assert actual.dtype == expected.dtype == target_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("target_dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("optimized", [False, True])
def test_explicit_float32_rounding_precedes_saturating_conversion(
    target_dtype: type, optimized: bool
) -> None:
    values = np.asarray(
        [
            np.nextafter(np.float64(127), -np.inf),
            np.nextafter(np.float64(-128), np.inf),
            np.nextafter(np.float64(255), -np.inf),
        ]
    )

    def convert(x: jax.Array) -> jax.Array:
        rounded = jax.lax.convert_element_type(x, np.float32)
        return jax.lax.convert_element_type(rounded, target_dtype)

    expected = _native_reference(convert, values)
    model = to_onnx(convert, [values])
    actual = _run(model, values, optimized)

    assert actual.dtype == expected.dtype == target_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("target_dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("optimized", [False, True])
def test_integer_to_8bit_retains_wrapping_conversion(
    target_dtype: type, optimized: bool
) -> None:
    values = np.asarray(
        [-513, -129, -128, -1, 0, 127, 128, 255, 256, 513], dtype=np.int32
    )

    def convert(x: jax.Array) -> jax.Array:
        return jax.lax.convert_element_type(x, target_dtype)

    expected = _native_reference(convert, values)
    model = to_onnx(convert, [values])
    actual = _run(model, values, optimized)

    assert actual.dtype == expected.dtype == target_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("source_dtype", "opset", "message"),
    [
        (np.float32, 10, "requires opset >= 11 for Clip"),
        (jnp.bfloat16, 12, "requires opset >= 13 for bfloat16 Cast"),
    ],
)
def test_float_to_8bit_rejects_opsets_without_required_schemas(
    source_dtype: type, opset: int, message: str
) -> None:
    with pytest.raises(NotImplementedError, match=message):
        to_onnx(
            lambda x: jax.lax.convert_element_type(x, np.int8),
            [jax.ShapeDtypeStruct((3,), source_dtype)],
            opset=opset,
        )


@pytest.mark.parametrize("source_dtype,opset", [(np.float32, 11), (jnp.bfloat16, 13)])
def test_float_to_8bit_supports_the_first_required_opset(
    source_dtype: type, opset: int
) -> None:
    values = np.asarray(
        jnp.asarray([-129.0, 128.0, np.nan, np.inf], dtype=source_dtype)
    )

    def convert(x: jax.Array) -> jax.Array:
        return jax.lax.convert_element_type(x, np.int8)

    expected = _native_reference(convert, values)
    model = to_onnx(convert, [values], opset=opset)
    actual = _run(model, values, optimized=False)

    assert actual.dtype == expected.dtype == np.int8
    np.testing.assert_array_equal(actual, expected)
