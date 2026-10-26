# tests/extra_tests/converter/test_exact_dtype_casts.py

from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx.user_interface import onnx_function, to_onnx


@pytest.fixture(autouse=True)
def _restore_jax_x64() -> Iterator[None]:
    previous = bool(jax.config.read("jax_enable_x64"))
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _run_cpu_model(model: onnx.ModelProto, x: np.ndarray) -> list[np.ndarray]:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return session.run(None, {session.get_inputs()[0].name: x})


def _convert_to_float32(x: jax.Array) -> jax.Array:
    return jax.lax.convert_element_type(x, jnp.float32)


def _bitcast_to_float32(x: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(x, new_dtype=jnp.float32)


def _zeros_float32() -> jax.Array:
    return jnp.zeros((2,), dtype=jnp.float32)


def _ones_float32() -> jax.Array:
    return jnp.ones((2,), dtype=jnp.float32)


def _eye_float32() -> jax.Array:
    return jnp.eye(2, dtype=jnp.float32)


def _full_float32() -> jax.Array:
    return jnp.full((2,), 1.5, dtype=jnp.float32)


def _linspace_float32() -> jax.Array:
    return jnp.linspace(0.0, 1.0, num=3, dtype=jnp.float32)


def _arange_float32() -> jax.Array:
    return jnp.arange(1.0, 2.0, 0.25, dtype=jnp.float32)


def _negate_float32_cast(x: jax.Array) -> jax.Array:
    return -_convert_to_float32(x)


def _add_float32_cast_and_ones(x: jax.Array) -> jax.Array:
    return _convert_to_float32(x) + _ones_float32()


def _float32_cast_and_negation(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    y = _convert_to_float32(x)
    return y, -y


@onnx_function
def _precision_child(x: jax.Array) -> jax.Array:
    return -x


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (lambda x: jnp.sin(_convert_to_float32(x)), np.sin),
        (lambda x: jnp.sqrt(_convert_to_float32(x)), np.sqrt),
        (lambda x: _convert_to_float32(x) - _full_float32(), lambda x: x - 1.5),
        (lambda x: _convert_to_float32(x) * _full_float32(), lambda x: x * 1.5),
        (lambda x: _convert_to_float32(x) / _full_float32(), lambda x: x / 1.5),
        (
            lambda x: jax.lax.reduce_sum(_convert_to_float32(x), axes=(0,)),
            lambda x: np.asarray(np.sum(x)),
        ),
        (
            lambda x: jnp.concatenate((_convert_to_float32(x), _full_float32())),
            lambda x: np.concatenate((x, np.full_like(x, 1.5))),
        ),
    ],
    ids=["numpy_sin", "numpy_sqrt", "sub", "mul", "div", "reduce_sum", "numpy_concat"],
)
def test_native_consumers_preserve_float32_rounding_before_double_computation(
    fn: Callable[[jax.Array], jax.Array],
    expected: Callable[[np.ndarray], np.ndarray],
) -> None:
    x = np.array([0.1, 2.5], dtype=np.float64)
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        enable_double_precision=True,
    )
    (actual,) = _run_cpu_model(model, x)

    rounded = x.astype(np.float32).astype(np.float64)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected(rounded), rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (_negate_float32_cast, np.negative),
        (_add_float32_cast_and_ones, lambda x: x + 1.0),
    ],
    ids=["native_negate_cast", "add_cast_and_ones"],
)
def test_explicit_float32_values_feed_double_precision_consumers(
    fn: Callable[[jax.Array], jax.Array],
    expected: Callable[[np.ndarray], np.ndarray],
) -> None:
    x = np.array([0.1, -2.5], dtype=np.float64)
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        enable_double_precision=True,
    )
    (actual,) = _run_cpu_model(model, x)

    # Preserve the explicit FP32 rounding before the legacy FP64 computation.
    rounded = x.astype(np.float32).astype(np.float64)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected(rounded))


def test_float32_output_retains_rounding_when_also_consumed_in_double() -> None:
    x = np.array([0.1, 2**24 + 1, -0.0, -2.5], dtype=np.float64)
    model = to_onnx(
        _float32_cast_and_negation,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        enable_double_precision=True,
    )
    direct, negated = _run_cpu_model(model, x)

    rounded = x.astype(np.float32)
    assert direct.dtype == np.float32
    assert negated.dtype == np.float64
    np.testing.assert_array_equal(direct, rounded)
    np.testing.assert_array_equal(negated, -rounded.astype(np.float64))
    assert direct[1] != x[1]


def test_function_outputs_preserve_float32_value_shared_with_double_consumer() -> None:
    def fn(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        y = _convert_to_float32(x)
        return y, _precision_child(y)

    x = np.array([0.1, 2**24 + 1], dtype=np.float64)
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        enable_double_precision=True,
    )
    assert model.functions
    direct, negated = _run_cpu_model(model, x)

    rounded = x.astype(np.float32)
    assert direct.dtype == np.float32
    assert negated.dtype == np.float64
    np.testing.assert_array_equal(direct, rounded)
    np.testing.assert_array_equal(negated, -rounded.astype(np.float64))


def test_comparison_and_integer_cast_keep_their_own_result_types() -> None:
    def fn(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        y = _convert_to_float32(x)
        return y == y, y.astype(jnp.int32)

    x = np.array([0.1, 1.1], dtype=np.float64)
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        enable_double_precision=True,
    )
    compared, integers = _run_cpu_model(model, x)

    rounded = x.astype(np.float32)
    assert compared.dtype == np.bool_
    assert integers.dtype == np.int32
    np.testing.assert_array_equal(compared, rounded == rounded)
    np.testing.assert_array_equal(integers, rounded.astype(np.int32))


def test_bitcast_uses_float32_bits_alongside_double_precision_consumer() -> None:
    def fn(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        y = _convert_to_float32(x)
        return y, -y, jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint32)

    x = np.array([0.1, 2**24 + 1, -0.0, -2.5], dtype=np.float64)
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        enable_double_precision=True,
        opset=26,
    )
    direct, negated, bits = _run_cpu_model(model, x)

    rounded = x.astype(np.float32)
    assert direct.dtype == np.float32
    assert negated.dtype == np.float64
    assert bits.dtype == np.uint32
    assert any(node.op_type == "BitCast" for node in model.graph.node)
    np.testing.assert_array_equal(direct, rounded)
    np.testing.assert_array_equal(negated, -rounded.astype(np.float64))
    np.testing.assert_array_equal(bits, rounded.view(np.uint32))


def test_native_unary_consumes_rounded_float32_in_double_precision() -> None:
    def fn(x: jax.Array) -> jax.Array:
        return jax.lax.abs(_convert_to_float32(x))

    x = np.array([0.1, -2.5], dtype=np.float64)
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        enable_double_precision=True,
    )
    (actual,) = _run_cpu_model(model, x)

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(
        actual, np.abs(x.astype(np.float32).astype(np.float64))
    )


def test_lax_concatenate_consumes_rounded_float32_in_double_precision() -> None:
    def fn(x: jax.Array) -> jax.Array:
        y = _convert_to_float32(x)
        return jax.lax.concatenate((y, y), dimension=0)

    x = np.array([0.1, -2.5], dtype=np.float64)
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        enable_double_precision=True,
    )
    (actual,) = _run_cpu_model(model, x)

    rounded = x.astype(np.float32).astype(np.float64)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, np.concatenate((rounded, rounded)))


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_low_precision_negation_retains_dtype_under_double_export(
    dtype_name: str,
) -> None:
    dtype = getattr(jnp, dtype_name)
    model = to_onnx(
        jax.lax.neg,
        [jax.ShapeDtypeStruct((2,), dtype)],
        enable_double_precision=True,
    )
    onnx.checker.check_model(model, full_check=True)

    expected_type = (
        onnx.TensorProto.FLOAT16
        if dtype_name == "float16"
        else onnx.TensorProto.BFLOAT16
    )
    assert model.graph.input[0].type.tensor_type.elem_type == expected_type
    assert model.graph.output[0].type.tensor_type.elem_type == expected_type
    if dtype_name == "float16":
        # BF16 Neg is not implemented by the tested CPU runtime.
        x = np.array([0.1, -2.5], dtype=np.float16)
        (actual,) = _run_cpu_model(model, x)
        assert actual.dtype == np.float16
        np.testing.assert_array_equal(actual, -x)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_low_precision_masked_attention_retains_dtype_under_double_export(
    dtype_name: str,
) -> None:
    dtype = getattr(jnp, dtype_name)
    mask = np.tril(np.ones((1, 1, 4, 4), dtype=bool))
    mask[..., 0, :] = False  # a fully masked query row

    def attention(x: jax.Array) -> jax.Array:
        return jax.nn.dot_product_attention(x, x, x, mask=mask)

    model = to_onnx(
        attention,
        [jax.ShapeDtypeStruct((1, 4, 2, 8), dtype)],
        enable_double_precision=True,
    )
    onnx.checker.check_model(model, full_check=True)

    expected_type = (
        onnx.TensorProto.FLOAT16
        if dtype_name == "float16"
        else onnx.TensorProto.BFLOAT16
    )
    assert model.graph.input[0].type.tensor_type.elem_type == expected_type
    assert model.graph.output[0].type.tensor_type.elem_type == expected_type
    if dtype_name == "float16":
        # BF16 MatMul is not implemented by the tested CPU runtime.
        x = np.random.default_rng(0).standard_normal((1, 4, 2, 8)).astype(np.float16)
        (actual,) = _run_cpu_model(model, x)
        assert actual.dtype == np.float16
        np.testing.assert_allclose(
            actual, np.asarray(attention(jnp.asarray(x))), rtol=2e-3, atol=2e-3
        )


def test_layout_outputs_preserve_float32_value_shared_with_double_consumer() -> None:
    x = (np.arange(24, dtype=np.float64) / 10).reshape((1, 2, 3, 4))
    model = to_onnx(
        _float32_cast_and_negation,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        enable_double_precision=True,
        outputs_as_nchw=[0, 1],
    )
    direct, negated = _run_cpu_model(model, x)

    rounded_nchw = x.astype(np.float32).transpose(0, 3, 1, 2)
    assert direct.dtype == np.float32
    assert negated.dtype == np.float64
    np.testing.assert_array_equal(direct, rounded_nchw)
    np.testing.assert_array_equal(negated, -rounded_nchw.astype(np.float64))


def test_convert_element_type_preserves_explicit_float32_target_under_x64() -> None:
    model = to_onnx(
        _convert_to_float32,
        [jax.ShapeDtypeStruct((2,), jnp.float64)],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_zeros_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _zeros_float32,
        [],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_ones_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _ones_float32,
        [],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_eye_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _eye_float32,
        [],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_full_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _full_float32,
        [],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_linspace_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _linspace_float32,
        [],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_arange_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _arange_float32,
        [],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_bitcast_convert_type_preserves_explicit_float32_target_under_x64() -> None:
    model = to_onnx(
        _bitcast_to_float32,
        [jax.ShapeDtypeStruct((2,), np.uint32)],
        return_mode="ir",
        enable_double_precision=True,
        opset=26,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT
