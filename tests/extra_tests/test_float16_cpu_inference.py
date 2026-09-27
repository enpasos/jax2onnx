# tests/extra_tests/test_float16_cpu_inference.py

"""FP16 exports must run through native CPU kernels or ORT's FP32 fallback."""

from __future__ import annotations

from collections.abc import Callable

from flax import nnx
import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx


def _float16_callable(kind: str) -> Callable[[jax.Array], jax.Array]:
    if kind == "linear":
        return nnx.Linear(
            16, 8, dtype=jnp.float16, param_dtype=jnp.float16, rngs=nnx.Rngs(0)
        )
    if kind in {"matmul", "dot"}:
        weights = jnp.asarray(
            np.random.default_rng(0).normal(size=(16, 8)) / 4, dtype=jnp.float16
        )

        def matmul(x: jax.Array) -> jax.Array:
            return jnp.matmul(x, weights) if kind == "matmul" else jnp.dot(x, weights)

        return matmul

    bias = jnp.linspace(-1, 1, 16, dtype=jnp.float16)

    def bias_gelu(x: jax.Array) -> jax.Array:
        # This pattern may become BiasGelu at runtime. ORT must handle a fused
        # CPU node even when only a float32 kernel is available for that node.
        return jax.nn.gelu(x + bias, approximate=False)

    return bias_gelu


@pytest.mark.parametrize("kind", ["matmul", "dot", "linear", "bias_gelu"])
@pytest.mark.parametrize("dynamic_batch", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    "optimization",
    [
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ],
    ids=["unoptimized", "optimized"],
)
def test_float16_cpu_inference_preserves_dtype_and_values(
    kind: str,
    dynamic_batch: bool,
    optimization: ort.GraphOptimizationLevel,
) -> None:
    fn = _float16_callable(kind)
    shape = jax_export.symbolic_shape("B, 16") if dynamic_batch else (3, 16)
    model = to_onnx(fn, [jax.ShapeDtypeStruct(shape, jnp.float16)])
    onnx.checker.check_model(model, full_check=True)
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16

    options = ort.SessionOptions()
    options.graph_optimization_level = optimization
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(1)
    for batch in (1, 3, 5) if dynamic_batch else (3,):
        x = rng.normal(size=(batch, 16)).astype(np.float16)
        expected = np.asarray(fn(jnp.asarray(x)))
        (actual,) = session.run(None, {session.get_inputs()[0].name: x})
        assert actual.dtype == np.float16
        assert np.isfinite(actual).all()
        # FP16 intermediate rounding differs when ORT computes a fused node or
        # promotes a CPU operation to FP32. Allow a few half-precision ulps.
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("operation", ["matmul", "dot"])
def test_float16_product_honors_float32_preferred_element_type(
    operation: str,
) -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 16)).astype(np.float16)
    weights = rng.normal(size=(16, 8)).astype(np.float16)

    def fn(a: jax.Array, b: jax.Array) -> jax.Array:
        product = jnp.matmul if operation == "matmul" else jnp.dot
        return product(a, b, preferred_element_type=jnp.float32)

    model = to_onnx(fn, [x, weights])
    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(
        None,
        {
            meta.name: value
            for meta, value in zip(session.get_inputs(), (x, weights), strict=True)
        },
    )
    assert actual.dtype == np.float32
    np.testing.assert_allclose(
        actual, x.astype(np.float32) @ weights.astype(np.float32), rtol=1e-5, atol=1e-6
    )
