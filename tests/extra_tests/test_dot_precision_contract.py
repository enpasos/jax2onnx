# tests/extra_tests/test_dot_precision_contract.py

"""Matrix precision parameters own operand rounding, work and result types."""

from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx
from tests.extra_tests.test_precision_contract import check


def _dot(api, lhs, rhs, **kwargs):
    if api == "lax":
        return jax.lax.dot_general(lhs, rhs, (((1,), (0,)), ((), ())), **kwargs)
    return getattr(jnp, api)(lhs, rhs, **kwargs)


@pytest.mark.parametrize("api", ["lax", "dot", "matmul"])
@pytest.mark.parametrize("scoped", [False, True], ids=["argument", "local-default"])
@pytest.mark.parametrize("algorithm", ["F32_F32_F32", "F64_F64_F64"])
def test_explicit_algorithm_preserves_work_type_and_shared_operand(
    api, scoped, algorithm
):
    if algorithm == "F32_F32_F32":
        inputs = [np.asarray([[2**24 + 1]], np.float64), np.ones((1, 1), np.float64)]
        expected = 2**24
        work_type = onnx.TensorProto.FLOAT
        preferred = None
    else:
        inputs = [
            np.asarray([[4097, -4096], [4097, -4096]], np.float32),
            np.asarray([[4097, 4097], [4098, 4098]], np.float32),
        ]
        expected = 1
        work_type = onnx.TensorProto.DOUBLE
        preferred = np.float64

    def source(lhs, rhs):
        context = jax.default_matmul_precision(algorithm) if scoped else nullcontext()
        with context:
            result = _dot(
                api,
                lhs,
                rhs,
                precision=(
                    None if scoped else getattr(jax.lax.DotAlgorithmPreset, algorithm)
                ),
                preferred_element_type=preferred,
            )
        return result, lhs

    with jax.enable_x64():
        model, reference = check(source, inputs)
    assert reference[0].dtype == np.float64
    np.testing.assert_array_equal(reference[0], np.full(reference[0].shape, expected))
    np.testing.assert_array_equal(reference[1], inputs[0])
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*model.graph.value_info, *model.graph.output)
    }
    contractions = [
        node
        for node in model.graph.node
        if node.op_type in {"Gemm", "MatMul", "Einsum"}
    ]
    assert contractions
    assert all(types[node.output[0]] == work_type for node in contractions)


def _original_scalar_f64_case(api, scoped):
    # Preserve the original six scalar-output witnesses and their inputs.
    inputs = [
        np.asarray([[4097, -4096]], np.float32),
        np.asarray([[4097], [4098]], np.float32),
    ]

    def source(lhs, rhs):
        context = (
            jax.default_matmul_precision("F64_F64_F64") if scoped else nullcontext()
        )
        with context:
            result = _dot(
                api,
                lhs,
                rhs,
                precision=None if scoped else jax.lax.DotAlgorithmPreset.F64_F64_F64,
                preferred_element_type=np.float64,
            )
        return result, lhs

    return source, inputs


@pytest.mark.parametrize("api", ["lax", "dot", "matmul"])
@pytest.mark.parametrize("scoped", [False, True], ids=["argument", "local-default"])
def test_scalar_f64_algorithm_cpu_reference_unavailable(api, scoped, record_property):
    source, inputs = _original_scalar_f64_case(api, scoped)
    expected_error = (
        "INVALID_ARGUMENT: Algorithm not supported by the ElementalIrEmitter: "
        "ALG_DOT_F64_F64_F64"
    )
    with jax.enable_x64():
        native_inputs = [jnp.asarray(value) for value in inputs]
        for execution in (source, jax.jit(source)):
            with pytest.raises(RuntimeError) as error:
                execution(*native_inputs)
            assert str(error.value).splitlines()[0] == expected_error
    # This is a compiler diagnostic, not a successful numerical reference.
    # Neither attempt creates an executable that could supply a warmed result.
    record_property("native_execution", "FAILED")
    record_property("native_numerical_qualification", "UNVERIFIED")


@pytest.mark.parametrize("api", ["lax", "dot", "matmul"])
@pytest.mark.parametrize("scoped", [False, True], ids=["argument", "local-default"])
def test_scalar_f64_algorithm_matches_onnx_spec_only(api, scoped, record_property):
    source, inputs = _original_scalar_f64_case(api, scoped)
    with jax.enable_x64():
        model = to_onnx(source, inputs)
    onnx.checker.check_model(model, full_check=True)
    # 4097**2 - 4096*4098 == 1; every integer product is exact in FLOAT64.
    expected = (np.ones((1, 1), np.float64), inputs[0])
    feeds = {meta.name: value for meta, value in zip(model.graph.input, inputs)}
    candidates = [ReferenceEvaluator(model).run(None, feeds)]
    for optimization in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = optimization
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        candidates.append(session.run(None, feeds))
    for outputs in candidates:
        assert len(outputs) == len(expected)
        for actual, wanted in zip(outputs, expected):
            assert actual.dtype == wanted.dtype
            assert actual.shape == wanted.shape
            np.testing.assert_array_equal(actual, wanted)
    record_property("reference", "ONNX_SPECIFICATION_ONLY")
    record_property("native_numerical_qualification", "UNVERIFIED")


@pytest.mark.parametrize("api", ["lax", "dot", "matmul"])
@pytest.mark.parametrize(
    "source_dtype,result_dtype", [(np.float32, np.float16), (np.float64, np.float32)]
)
def test_narrow_preferred_result_keeps_operand_precision(
    api, source_dtype, result_dtype
):
    first = 2049 if source_dtype is np.float32 else 2**24 + 1
    inputs = [
        np.asarray([[first, -(first - 1)]], source_dtype),
        np.ones((2, 1), source_dtype),
    ]

    def source(lhs, rhs):
        return _dot(api, lhs, rhs, preferred_element_type=result_dtype)

    with jax.enable_x64():
        model, reference = check(source, inputs)
    assert reference.dtype == result_dtype
    assert reference.item() == 1
    # ORT's half-kernel adaptation can conceal premature operand casts.
    # The serialized graph must also preserve the original products itself.
    (evaluated,) = ReferenceEvaluator(model).run(
        None, {meta.name: value for meta, value in zip(model.graph.input, inputs)}
    )
    assert evaluated.dtype == result_dtype
    np.testing.assert_array_equal(evaluated, reference)


def test_complex_preferred_accumulation_preserves_fp64_sum_and_shared_input():
    def source(lhs, rhs):
        left = jax.lax.complex(lhs, jnp.zeros_like(lhs))
        right = jax.lax.complex(rhs, jnp.zeros_like(rhs))
        result = jax.lax.dot_general(
            left,
            right,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.complex128,
        )
        return jnp.real(result), jnp.imag(result), lhs

    inputs = [np.asarray([[2**24, 1]], np.float32), np.ones((2, 1), np.float32)]
    with jax.enable_x64():
        _, reference = check(source, inputs)
    assert reference[0].dtype == reference[1].dtype == np.float64
    assert reference[0].item() == 2**24 + 1
    np.testing.assert_array_equal(reference[2], inputs[0])


@pytest.mark.parametrize("use_algorithm", [False, True])
def test_single_group_ragged_dot_preserves_precision_and_original_rhs(use_algorithm):
    dtype = np.float64 if use_algorithm else np.float32
    inputs = [
        np.asarray(
            [[2**24 + 1, -(2**24)], [2, 3]] if use_algorithm else [[2**24, 1], [2, 3]],
            dtype,
        ),
        np.ones((1, 2, 2), dtype),
    ]

    def source(lhs, rhs):
        dimensions = jax.lax.RaggedDotDimensionNumbers(
            (((1,), (1,)), ((), ())), (0,), (0,)
        )
        result = jax.lax.ragged_dot_general(
            lhs,
            rhs,
            jnp.asarray([2], jnp.int32),
            dimensions,
            precision=jax.lax.DotAlgorithmPreset.F32_F32_F32 if use_algorithm else None,
            preferred_element_type=None if use_algorithm else jnp.float64,
        )
        return result, rhs

    with jax.enable_x64():
        _, reference = check(source, inputs)
    first = 0 if use_algorithm else 2**24 + 1
    np.testing.assert_array_equal(
        reference[0], np.asarray([[first, first], [5, 5]], np.float64)
    )
    np.testing.assert_array_equal(reference[1], inputs[1])


@pytest.mark.parametrize("api", ["lax", "dot", "matmul"])
def test_default_algorithm_retains_existing_matrix_path(api):
    def source(lhs, rhs):
        return _dot(api, lhs, rhs, precision=jax.lax.DotAlgorithmPreset.DEFAULT)

    with jax.enable_x64():
        check(source, [np.asarray([[1, 2]], np.float32), np.ones((2, 1), np.float32)])


@pytest.mark.parametrize("api", ["lax", "dot", "matmul"])
@pytest.mark.parametrize("scoped", [False, True], ids=["argument", "local-default"])
def test_unimplemented_dot_algorithm_is_diagnosed_before_artifact(
    api, scoped, tmp_path
):
    algorithm = "TF32_TF32_F32"

    def source(lhs, rhs):
        context = jax.default_matmul_precision(algorithm) if scoped else nullcontext()
        with context:
            return _dot(
                api,
                lhs,
                rhs,
                precision=(
                    None if scoped else getattr(jax.lax.DotAlgorithmPreset, algorithm)
                ),
            )

    path = tmp_path / "unsupported.onnx"
    with (
        jax.enable_x64(),
        pytest.raises(NotImplementedError, match="currently not implemented"),
    ):
        to_onnx(
            source,
            [np.ones((1, 1), np.float32), np.ones((1, 1), np.float32)],
            return_mode="file",
            output_path=path,
        )
    assert not path.exists()
