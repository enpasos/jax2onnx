# tests/extra_tests/test_float16_cpu_inference.py

"""FP16 exports must run through native CPU kernels or ORT's FP32 fallback."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import subprocess
import sys

from flax import linen, nnx
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


def _float16_layer_norm(framework: str) -> Callable[[jax.Array], jax.Array]:
    if framework == "nnx":
        return nnx.LayerNorm(
            16, dtype=jnp.float16, param_dtype=jnp.float16, rngs=nnx.Rngs(0)
        )
    module = linen.LayerNorm(dtype=jnp.float16, param_dtype=jnp.float16)
    variables = module.init(jax.random.PRNGKey(0), jnp.zeros((3, 16), jnp.float16))
    return lambda x: module.apply(variables, x)


_CREATE_SESSION_WITH_ALL_OPTIMIZATIONS = """
import sys
import onnxruntime as ort

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort.InferenceSession(sys.argv[1], options, providers=["CPUExecutionProvider"])
"""


@pytest.mark.parametrize("framework", ["nnx", "linen"])
def test_float16_parameters_widened_to_float32_load_with_all_optimizations(
    framework: str, tmp_path: Path
) -> None:
    # LayerNorm computes its statistics in float32 and widens the float16 scale
    # and bias for that. ONNX Runtime before 1.30 aborts the process while
    # creating a session at ORT_ENABLE_ALL when a CastLike does the widening:
    # it expands the CastLike to a Cast after constant folding and then looks
    # the Cast up by its node name. It folds a Cast that the model spells out.
    fn = _float16_layer_norm(framework)
    model = to_onnx(fn, [jax.ShapeDtypeStruct((3, 16), jnp.float16)])
    onnx.checker.check_model(model, full_check=True)

    float16_constants = {
        tensor.name
        for tensor in model.graph.initializer
        if tensor.data_type == onnx.TensorProto.FLOAT16
    }
    widenings = [
        node.op_type
        for node in model.graph.node
        if node.input and node.input[0] in float16_constants
    ]
    assert widenings and set(widenings) == {"Cast"}
    assert "CastLike" not in {node.op_type for node in model.graph.node}

    # The defect ends the process with a signal instead of raising. The first
    # session is therefore created in a process of its own, so that a crash
    # fails this test and not the whole test run.
    path = tmp_path / "layer_norm.onnx"
    onnx.save_model(model, path)
    created = subprocess.run(
        [sys.executable, "-c", _CREATE_SESSION_WITH_ALL_OPTIMIZATIONS, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert created.returncode == 0, created.stderr[-2000:]

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    x = np.random.default_rng(1).normal(size=(3, 16)).astype(np.float16)
    expected = np.asarray(fn(jnp.asarray(x)))
    (actual,) = session.run(None, {session.get_inputs()[0].name: x})
    assert actual.dtype == np.float16
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
