# tests/extra_tests/converter/test_source_dot_precision.py

"""Bounded source matrix precision; target evidence is ORT CPU only.

The source lowering admits default-precision FP16 matrix operands with a
preferred FP32 result. It makes no claim about hardware-specific algorithms,
arbitrary reduction ordering, or the performance of widening the operands.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx
from tests.extra_tests.converter.precision_test_support import export_source_dtypes


@pytest.fixture(autouse=True)
def _restore_jax_settings() -> Iterator[None]:
    previous_x64 = jax.config.x64_enabled
    previous_precision = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", None)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_default_matmul_precision", previous_precision)


def _dimensions(rank: int) -> tuple[Any, Any]:
    batch = tuple(range(rank - 2))
    return (((rank - 1,), (rank - 2,)), (batch, batch))


def _dot(x: jax.Array, w: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        x,
        w,
        dimension_numbers=_dimensions(x.ndim),
        preferred_element_type=jnp.float32,
    )


def _shared_operands(x: jax.Array, w: jax.Array) -> tuple[jax.Array, ...]:
    return _dot(x, w), x, w


def _session(
    model: onnx.ModelProto, tmp_path: Path, optimize: bool
) -> ort.InferenceSession:
    onnx.checker.check_model(model, full_check=True)
    path = tmp_path / "source_dot.onnx"
    onnx.save_model(model, path)
    reloaded = onnx.load_model(path)
    onnx.checker.check_model(reloaded, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return ort.InferenceSession(
        reloaded.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("optimize", [False, True], ids=["ort-off", "ort-on"])
def test_source_dot_preserves_preferred_fp32_and_shared_fp16_operands(
    rank: int, x64: bool, optimize: bool, tmp_path: Path
) -> None:
    jax.config.update("jax_enable_x64", x64)
    batch = (1,) if rank == 3 else ()
    specs = [
        jax.ShapeDtypeStruct(batch + (1, 2), jnp.float16),
        jax.ShapeDtypeStruct(batch + (2, 1), jnp.float16),
    ]
    model = export_source_dtypes(_shared_operands, specs)
    assert [value.type.tensor_type.elem_type for value in model.graph.input] == [
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.FLOAT16,
    ]
    assert [value.type.tensor_type.elem_type for value in model.graph.output] == [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.FLOAT16,
    ]
    matmul = next(node for node in model.graph.node if node.op_type == "MatMul")
    producers = {output: node for node in model.graph.node for output in node.output}
    for operand, original in zip(matmul.input, model.graph.input, strict=True):
        conversion = producers[operand]
        assert conversion.op_type == "Cast"
        assert list(conversion.input) == [original.name]
        assert next(a.i for a in conversion.attribute if a.name == "to") == (
            onnx.TensorProto.FLOAT
        )
    # FP32 arithmetic precedes the result: an output cast cannot recover 2049.
    assert model.graph.output[0].name in matmul.output

    session = _session(model, tmp_path, optimize)
    cases = [
        ([2048, 1], [1, 1], 2049),
        ([256, 256], [256, 256], 131072),
        ([2**-13, 2**-13], [2**-13, 2**-13], 2**-25),
        ([2**-24, 2**-24], [2**-24, 2**-24], 2**-47),
        ([65504, -65504], [65504, 65504], 0),
    ]
    for left, right, exact in cases:
        x = np.asarray(left, dtype=np.float16).reshape(specs[0].shape)
        w = np.asarray(right, dtype=np.float16).reshape(specs[1].shape)
        eager = np.asarray(_dot(jnp.asarray(x), jnp.asarray(w)))
        compiled = np.asarray(jax.jit(_dot)(jnp.asarray(x), jnp.asarray(w)))
        expected = np.full(batch + (1, 1), exact, dtype=np.float32)
        np.testing.assert_array_equal(eager, expected)
        np.testing.assert_array_equal(compiled, expected)
        result, original_x, original_w = session.run(
            None,
            {
                session.get_inputs()[0].name: x,
                session.get_inputs()[1].name: w,
            },
        )
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, expected)
        assert original_x.dtype == original_w.dtype == np.float16
        np.testing.assert_array_equal(original_x, x)
        np.testing.assert_array_equal(original_w, w)


@pytest.mark.parametrize("optimize", [False, True], ids=["ort-off", "ort-on"])
def test_source_dot_handles_multiple_batches_and_matrix_entries(
    optimize: bool, tmp_path: Path
) -> None:
    x = (np.arange(48).reshape(2, 3, 8) - 24).astype(np.float16) / np.float16(4)
    w = (np.arange(64).reshape(2, 8, 4) - 32).astype(np.float16) / np.float16(8)
    model = export_source_dtypes(
        _dot, [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (x, w)]
    )
    session = _session(model, tmp_path, optimize)
    (result,) = session.run(
        None,
        {session.get_inputs()[0].name: x, session.get_inputs()[1].name: w},
    )
    # These dyadic products and sums are exactly representable in FP32.
    exact = (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(result, exact)
    np.testing.assert_array_equal(_dot(jnp.asarray(x), jnp.asarray(w)), exact)


@pytest.mark.parametrize("optimize", [False, True], ids=["ort-off", "ort-on"])
def test_source_dot_numerical_acceptance_for_selected_longer_reduction(
    optimize: bool, tmp_path: Path
) -> None:
    rng = np.random.default_rng(314)
    x = rng.normal(size=(2, 3, 64)).astype(np.float16)
    w = rng.normal(size=(2, 64, 4)).astype(np.float16)
    model = export_source_dtypes(
        _dot, [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (x, w)]
    )
    session = _session(model, tmp_path, optimize)
    (actual,) = session.run(
        None,
        {session.get_inputs()[0].name: x, session.get_inputs()[1].name: w},
    )
    eager = np.asarray(_dot(jnp.asarray(x), jnp.asarray(w)))
    compiled = np.asarray(jax.jit(_dot)(jnp.asarray(x), jnp.asarray(w)))
    reference = x.astype(np.float64) @ w.astype(np.float64)
    # Fixed acceptance for this sample, not a global bound or reduction order.
    for result in (actual, eager, compiled):
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, reference, rtol=2e-6, atol=1e-5)
    np.testing.assert_allclose(actual, eager, rtol=2e-6, atol=1e-5)


@pytest.mark.parametrize("optimize", [False, True], ids=["ort-off", "ort-on"])
def test_source_dot_selected_special_values(optimize: bool, tmp_path: Path) -> None:
    model = export_source_dtypes(
        _dot,
        [
            jax.ShapeDtypeStruct((1, 2), jnp.float16),
            jax.ShapeDtypeStruct((2, 1), jnp.float16),
        ],
    )
    session = _session(model, tmp_path, optimize)
    for values in (
        [np.nan, 1],
        [np.inf, 1],
        [-np.inf, 1],
        [np.inf, -np.inf],
        [0, -0.0],
    ):
        x = np.asarray([values], dtype=np.float16)
        w = np.ones((2, 1), dtype=np.float16)
        (actual,) = session.run(
            None,
            {session.get_inputs()[0].name: x, session.get_inputs()[1].name: w},
        )
        expected = np.asarray(_dot(jnp.asarray(x), jnp.asarray(w)))
        np.testing.assert_array_equal(actual, expected)
        if not np.isnan(expected).any():
            np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parametrize(
    "precision",
    [
        jax.lax.Precision.DEFAULT,
        (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
    ],
)
def test_source_dot_accepts_explicit_default_precision(precision: Any) -> None:
    def model(x: jax.Array, w: jax.Array) -> jax.Array:
        return jax.lax.dot_general(
            x,
            w,
            _dimensions(2),
            precision=precision,
            preferred_element_type=jnp.float32,
        )

    result = export_source_dtypes(
        model,
        [
            jax.ShapeDtypeStruct((1, 2), jnp.float16),
            jax.ShapeDtypeStruct((2, 1), jnp.float16),
        ],
    )
    assert result.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT


@pytest.mark.parametrize(
    "precision",
    [
        jax.lax.Precision.HIGH,
        jax.lax.Precision.HIGHEST,
        (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST),
        jax.lax.DotAlgorithmPreset.F16_F16_F32,
        jax.lax.DotAlgorithm(np.float16, np.float16, np.float32),
    ],
)
def test_source_dot_rejects_unimplemented_precision_contracts(precision: Any) -> None:
    def model(x: jax.Array, w: jax.Array) -> jax.Array:
        return jax.lax.dot_general(
            x,
            w,
            _dimensions(2),
            precision=precision,
            preferred_element_type=jnp.float32,
        )

    with pytest.raises(NotImplementedError, match="precision .*private subset"):
        export_source_dtypes(
            model,
            [
                jax.ShapeDtypeStruct((1, 2), jnp.float16),
                jax.ShapeDtypeStruct((2, 1), jnp.float16),
            ],
        )


def test_source_dot_respects_traced_default_matmul_precision() -> None:
    with jax.default_matmul_precision("highest"):
        with pytest.raises(NotImplementedError, match="precision .*private subset"):
            export_source_dtypes(
                _dot,
                [
                    jax.ShapeDtypeStruct((1, 2), jnp.float16),
                    jax.ShapeDtypeStruct((2, 1), jnp.float16),
                ],
            )


@pytest.mark.parametrize(
    ("dtype", "preferred"),
    [
        (jnp.float16, None),
        (jnp.float16, jnp.float16),
        (jnp.float16, jnp.float64),
        (jnp.bfloat16, jnp.float32),
    ],
)
def test_source_dot_rejects_unaudited_type_combinations(
    dtype: Any, preferred: Any
) -> None:
    jax.config.update("jax_enable_x64", True)

    def model(x: jax.Array, w: jax.Array) -> jax.Array:
        return jax.lax.dot_general(
            x, w, _dimensions(2), preferred_element_type=preferred
        )

    with pytest.raises(
        NotImplementedError, match="operand/result types .*private subset"
    ):
        export_source_dtypes(
            model,
            [jax.ShapeDtypeStruct((1, 2), dtype), jax.ShapeDtypeStruct((2, 1), dtype)],
        )


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "dimensions"),
    [
        ((0, 2), (2, 1), _dimensions(2)),
        ((1, 0), (0, 1), _dimensions(2)),
        ((1, 2), (2, 0), _dimensions(2)),
        ((1, 1, 1, 2), (1, 1, 2, 1), _dimensions(4)),
        ((2,), (2,), (((0,), (0,)), ((), ()))),
        ((2, 1), (2, 1), (((0,), (0,)), ((), ()))),
    ],
)
def test_source_dot_rejects_unaudited_dimensions(
    lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...], dimensions: Any
) -> None:
    def model(x: jax.Array, w: jax.Array) -> jax.Array:
        return jax.lax.dot_general(x, w, dimensions, preferred_element_type=jnp.float32)

    with pytest.raises(NotImplementedError, match="shapes .*private subset"):
        export_source_dtypes(
            model,
            [
                jax.ShapeDtypeStruct(lhs_shape, jnp.float16),
                jax.ShapeDtypeStruct(rhs_shape, jnp.float16),
            ],
        )


@pytest.mark.parametrize("double", [False, True])
def test_legacy_dot_keeps_its_existing_precision_route(
    double: bool, tmp_path: Path
) -> None:
    x = np.asarray([[1, 2]], dtype=np.float64 if double else np.float32)
    w = np.asarray([[3], [4]], dtype=x.dtype)
    model = to_onnx(
        lambda left, right: jax.lax.dot_general(left, right, _dimensions(2)),
        [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (x, w)],
        enable_double_precision=double,
    )
    session = _session(model, tmp_path, False)
    (actual,) = session.run(
        None,
        {session.get_inputs()[0].name: x, session.get_inputs()[1].name: w},
    )
    assert actual.dtype == x.dtype
    np.testing.assert_array_equal(actual, np.asarray([[11]], dtype=x.dtype))
