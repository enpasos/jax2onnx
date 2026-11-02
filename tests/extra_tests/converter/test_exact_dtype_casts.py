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
def _use_jax_x64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


def _run_cpu_model(
    model: onnx.ModelProto,
    x: np.ndarray,
    fn: Callable[[jax.Array], object],
    *,
    rtol: float = 0,
    atol: float = 0,
    outputs_as_nchw: tuple[int, ...] = (),
) -> list[np.ndarray]:
    onnx.checker.check_model(model, full_check=True)
    native_x = jnp.asarray(x, dtype=x.dtype)
    eager = fn(native_x)
    structure = jax.tree.structure(eager)

    def outputs(value: object) -> list[np.ndarray]:
        leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(value)]
        for index in outputs_as_nchw:
            leaves[index] = leaves[index].transpose(0, 3, 1, 2)
        return leaves

    expected = outputs(eager)

    def compare(actual: list[np.ndarray], label: str) -> None:
        assert len(actual) == len(expected), label
        for got, wanted in zip(actual, expected):
            assert got.dtype == wanted.dtype, label
            assert got.shape == wanted.shape, label
            if np.issubdtype(wanted.dtype, np.floating) and (rtol or atol):
                np.testing.assert_allclose(
                    got, wanted, rtol=rtol, atol=atol, err_msg=label
                )
            else:
                np.testing.assert_array_equal(got, wanted, err_msg=label)
            if np.issubdtype(wanted.dtype, np.floating):
                zeros = wanted == 0
                np.testing.assert_array_equal(
                    np.signbit(got[zeros]), np.signbit(wanted[zeros]), err_msg=label
                )

    compiled = jax.jit(lambda value: fn(value))
    for label in ("fresh JIT", "warmed JIT"):
        result = compiled(native_x)
        assert jax.tree.structure(result) == structure, label
        compare(outputs(result), label)

    actuals = []
    for optimization in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = optimization
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        actual = session.run(None, {session.get_inputs()[0].name: x})
        compare(actual, str(optimization))
        actuals.append(actual)
    return actuals[0]


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


def _rounded_float64(x: jax.Array) -> jax.Array:
    return _convert_to_float32(x).astype(jnp.float64)


def _negate_rounded_float64(x: jax.Array) -> jax.Array:
    return -_rounded_float64(x)


def _add_rounded_float64_and_ones(x: jax.Array) -> jax.Array:
    return _rounded_float64(x) + _ones_float32().astype(jnp.float64)


def _float32_cast_and_float64_negation(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    y = _convert_to_float32(x)
    return y, -y.astype(jnp.float64)


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
def test_fixed_float32_consumers_preserve_source_computation(
    fn: Callable[[jax.Array], jax.Array],
    expected: Callable[[np.ndarray], np.ndarray],
) -> None:
    x = np.array([0.1, 2.5], dtype=np.float64)
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
    )
    (actual,) = _run_cpu_model(model, x, fn, rtol=1e-5, atol=1e-5)

    rounded = x.astype(np.float32)
    assert actual.dtype == np.float32
    # These added source cases use STANDARD_F32's established component limits.
    # The configured FP64 counterparts below retain the original 1e-14/0 gate.
    np.testing.assert_allclose(actual, expected(rounded), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (lambda x: jnp.sin(_rounded_float64(x)), np.sin),
        (lambda x: jnp.sqrt(_rounded_float64(x)), np.sqrt),
        (
            lambda x: _rounded_float64(x) - _full_float32().astype(jnp.float64),
            lambda x: x - 1.5,
        ),
        (
            lambda x: _rounded_float64(x) * _full_float32().astype(jnp.float64),
            lambda x: x * 1.5,
        ),
        (
            lambda x: _rounded_float64(x) / _full_float32().astype(jnp.float64),
            lambda x: x / 1.5,
        ),
        (
            lambda x: jax.lax.reduce_sum(_rounded_float64(x), axes=(0,)),
            lambda x: np.asarray(np.sum(x)),
        ),
        (
            lambda x: jnp.concatenate(
                (_rounded_float64(x), _full_float32().astype(jnp.float64))
            ),
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
    model = to_onnx(fn, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
    (actual,) = _run_cpu_model(model, x, fn, rtol=1e-14, atol=0)

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
def test_fixed_float32_values_keep_their_source_dtype(
    fn: Callable[[jax.Array], jax.Array],
    expected: Callable[[np.ndarray], np.ndarray],
) -> None:
    x = np.array([0.1, -2.5], dtype=np.float64)
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
    )
    (actual,) = _run_cpu_model(model, x, fn)

    rounded = x.astype(np.float32)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected(rounded))


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (_negate_rounded_float64, np.negative),
        (_add_rounded_float64_and_ones, lambda x: x + 1.0),
    ],
    ids=["native_negate_cast", "add_cast_and_ones"],
)
def test_explicit_float32_values_feed_double_precision_consumers(
    fn: Callable[[jax.Array], jax.Array],
    expected: Callable[[np.ndarray], np.ndarray],
) -> None:
    x = np.array([0.1, -2.5], dtype=np.float64)
    model = to_onnx(fn, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
    (actual,) = _run_cpu_model(model, x, fn)

    # The model deliberately widens the rounded value before its FP64 consumer.
    rounded = x.astype(np.float32).astype(np.float64)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected(rounded))


def test_float32_output_retains_rounding_when_also_consumed_in_double() -> None:
    x = np.array([0.1, 2**24 + 1, -0.0, -2.5], dtype=np.float64)
    model = to_onnx(
        _float32_cast_and_float64_negation,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
    )
    direct, negated = _run_cpu_model(model, x, _float32_cast_and_float64_negation)

    rounded = x.astype(np.float32)
    assert direct.dtype == np.float32
    assert negated.dtype == np.float64
    np.testing.assert_array_equal(direct, rounded)
    np.testing.assert_array_equal(negated, -rounded.astype(np.float64))
    assert direct[1] != x[1]


def test_fixed_float32_output_and_consumer_retain_source_rounding() -> None:
    x = np.array([0.1, 2**24 + 1, -0.0, -2.5], dtype=np.float64)
    model = to_onnx(
        _float32_cast_and_negation,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
    )
    direct, negated = _run_cpu_model(model, x, _float32_cast_and_negation)

    rounded = x.astype(np.float32)
    assert direct.dtype == negated.dtype == np.float32
    np.testing.assert_array_equal(direct, rounded)
    np.testing.assert_array_equal(negated, -rounded)
    assert direct[1] != x[1]


def test_function_outputs_preserve_fixed_float32_values() -> None:
    def fn(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        y = _convert_to_float32(x)
        return y, _precision_child(y)

    x = np.array([0.1, 2**24 + 1], dtype=np.float64)
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
    )
    assert model.functions
    direct, negated = _run_cpu_model(model, x, fn)

    rounded = x.astype(np.float32)
    assert direct.dtype == negated.dtype == np.float32
    np.testing.assert_array_equal(direct, rounded)
    np.testing.assert_array_equal(negated, -rounded)


def test_function_outputs_preserve_float32_value_shared_with_double_consumer() -> None:
    def configured_fn(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        y = _convert_to_float32(x)
        return y, _precision_child(y.astype(jnp.float64))

    x = np.array([0.1, 2**24 + 1], dtype=np.float64)
    model = to_onnx(configured_fn, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
    assert model.functions
    direct, negated = _run_cpu_model(model, x, configured_fn)

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
    )
    compared, integers = _run_cpu_model(model, x, fn)

    rounded = x.astype(np.float32)
    assert compared.dtype == np.bool_
    assert integers.dtype == np.int32
    np.testing.assert_array_equal(compared, rounded == rounded)
    np.testing.assert_array_equal(integers, rounded.astype(np.int32))


def test_bitcast_uses_float32_bits_alongside_fixed_float32_consumer() -> None:
    def fn(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        y = _convert_to_float32(x)
        return y, -y, jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint32)

    x = np.array([0.1, 2**24 + 1, -0.0, -2.5], dtype=np.float64)
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        opset=26,
    )
    direct, negated, bits = _run_cpu_model(model, x, fn)

    rounded = x.astype(np.float32)
    assert direct.dtype == np.float32
    assert negated.dtype == np.float32
    assert bits.dtype == np.uint32
    assert any(node.op_type == "BitCast" for node in model.graph.node)
    np.testing.assert_array_equal(direct, rounded)
    np.testing.assert_array_equal(negated, -rounded)
    np.testing.assert_array_equal(bits, rounded.view(np.uint32))
    assert bits.tobytes() == rounded.tobytes()


def test_bitcast_uses_float32_bits_alongside_double_precision_consumer() -> None:
    def configured_fn(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        y = _convert_to_float32(x)
        return (
            y,
            -y.astype(jnp.float64),
            jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint32),
        )

    x = np.array([0.1, 2**24 + 1, -0.0, -2.5], dtype=np.float64)
    model = to_onnx(configured_fn, [jax.ShapeDtypeStruct(x.shape, x.dtype)], opset=26)
    direct, negated, bits = _run_cpu_model(model, x, configured_fn)

    rounded = x.astype(np.float32)
    assert direct.dtype == np.float32
    assert negated.dtype == np.float64
    assert bits.dtype == np.uint32
    assert any(node.op_type == "BitCast" for node in model.graph.node)
    np.testing.assert_array_equal(direct, rounded)
    np.testing.assert_array_equal(negated, -rounded.astype(np.float64))
    np.testing.assert_array_equal(bits, rounded.view(np.uint32))
    assert bits.tobytes() == rounded.tobytes()


def test_native_unary_preserves_fixed_float32_computation() -> None:
    def fn(x: jax.Array) -> jax.Array:
        return jax.lax.abs(_convert_to_float32(x))

    x = np.array([0.1, -2.5], dtype=np.float64)
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
    )
    (actual,) = _run_cpu_model(model, x, fn)

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.abs(x.astype(np.float32)))


def test_native_unary_consumes_rounded_float32_in_double_precision() -> None:
    def configured_fn(x: jax.Array) -> jax.Array:
        return jax.lax.abs(_rounded_float64(x))

    x = np.array([0.1, -2.5], dtype=np.float64)
    model = to_onnx(configured_fn, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
    (actual,) = _run_cpu_model(model, x, configured_fn)

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(
        actual, np.abs(x.astype(np.float32).astype(np.float64))
    )


def test_lax_concatenate_preserves_fixed_float32_computation() -> None:
    def fn(x: jax.Array) -> jax.Array:
        y = _convert_to_float32(x)
        return jax.lax.concatenate((y, y), dimension=0)

    x = np.array([0.1, -2.5], dtype=np.float64)
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
    )
    (actual,) = _run_cpu_model(model, x, fn)

    rounded = x.astype(np.float32)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.concatenate((rounded, rounded)))


def test_lax_concatenate_consumes_rounded_float32_in_double_precision() -> None:
    def configured_fn(x: jax.Array) -> jax.Array:
        y = _rounded_float64(x)
        return jax.lax.concatenate((y, y), dimension=0)

    x = np.array([0.1, -2.5], dtype=np.float64)
    model = to_onnx(configured_fn, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
    (actual,) = _run_cpu_model(model, x, configured_fn)

    rounded = x.astype(np.float32).astype(np.float64)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, np.concatenate((rounded, rounded)))


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_low_precision_negation_retains_dtype_under_x64(
    dtype_name: str,
) -> None:
    dtype = getattr(jnp, dtype_name)
    model = to_onnx(
        jax.lax.neg,
        [jax.ShapeDtypeStruct((2,), dtype)],
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
        (actual,) = _run_cpu_model(model, x, jax.lax.neg)
        assert actual.dtype == np.float16
        np.testing.assert_array_equal(actual, -x)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_low_precision_masked_attention_retains_dtype_under_x64(
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
        expected = np.asarray(attention(jnp.asarray(x)))
        # This retains eager/ORT coverage. Compiled-reference qualification is
        # unavailable because fresh CPU JIT compilation fails as diagnosed below.
        for optimization in (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        ):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = optimization
            session = ort.InferenceSession(
                model.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
            (actual,) = session.run(None, {session.get_inputs()[0].name: x})
            assert actual.dtype == np.float16
            assert actual.shape == expected.shape == (1, 4, 2, 8)
            np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_float16_masked_attention_cpu_jit_reports_unsupported_precision() -> None:
    mask = np.tril(np.ones((1, 1, 4, 4), dtype=bool))
    mask[..., 0, :] = False  # a fully masked query row

    def attention(x: jax.Array) -> jax.Array:
        return jax.nn.dot_product_attention(x, x, x, mask=mask)

    x = np.random.default_rng(0).standard_normal((1, 4, 2, 8)).astype(np.float16)
    # A successful diagnostic records unsupported fresh compilation. It is not
    # a numerical JIT reference, and no warmed reference exists after this error.
    with pytest.raises(
        ValueError,
        match=r"\AThe precision 'F16_F16_F32' is not supported by dot_general on CPU(?:\n|\Z)",
    ):
        jax.jit(attention, backend="cpu")(jnp.asarray(x))


def test_layout_outputs_preserve_float32_value_shared_with_double_consumer() -> None:
    x = (np.arange(24, dtype=np.float64) / 10).reshape((1, 2, 3, 4))
    model = to_onnx(
        _float32_cast_and_float64_negation,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        outputs_as_nchw=[0, 1],
    )
    direct, negated = _run_cpu_model(
        model, x, _float32_cast_and_float64_negation, outputs_as_nchw=(0, 1)
    )

    rounded_nchw = x.astype(np.float32).transpose(0, 3, 1, 2)
    assert direct.dtype == np.float32
    assert negated.dtype == np.float64
    np.testing.assert_array_equal(direct, rounded_nchw)
    np.testing.assert_array_equal(negated, -rounded_nchw.astype(np.float64))


def test_layout_outputs_preserve_fixed_float32_values() -> None:
    x = (np.arange(24, dtype=np.float64) / 10).reshape((1, 2, 3, 4))
    model = to_onnx(
        _float32_cast_and_negation,
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        outputs_as_nchw=[0, 1],
    )
    direct, negated = _run_cpu_model(
        model, x, _float32_cast_and_negation, outputs_as_nchw=(0, 1)
    )

    rounded_nchw = x.astype(np.float32).transpose(0, 3, 1, 2)
    assert direct.dtype == negated.dtype == np.float32
    np.testing.assert_array_equal(direct, rounded_nchw)
    np.testing.assert_array_equal(negated, -rounded_nchw)


def test_convert_element_type_preserves_explicit_float32_target_under_x64() -> None:
    model = to_onnx(
        _convert_to_float32,
        [jax.ShapeDtypeStruct((2,), jnp.float64)],
        return_mode="ir",
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_zeros_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _zeros_float32,
        [],
        return_mode="ir",
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_ones_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _ones_float32,
        [],
        return_mode="ir",
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_eye_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _eye_float32,
        [],
        return_mode="ir",
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_full_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _full_float32,
        [],
        return_mode="ir",
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_linspace_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _linspace_float32,
        [],
        return_mode="ir",
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_arange_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _arange_float32,
        [],
        return_mode="ir",
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_bitcast_convert_type_preserves_explicit_float32_target_under_x64() -> None:
    model = to_onnx(
        _bitcast_to_float32,
        [jax.ShapeDtypeStruct((2,), np.uint32)],
        return_mode="ir",
        opset=26,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT
