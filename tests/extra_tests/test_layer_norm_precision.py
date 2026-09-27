# tests/extra_tests/test_layer_norm_precision.py

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import equinox as eqx
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx

LN_FAMILY = {
    "LayerNormalization",
    "SimplifiedLayerNormalization",
    "SkipLayerNormalization",
    "SkipSimplifiedLayerNormalization",
}


@pytest.fixture(autouse=True)
def _float32_defaults() -> Iterator[None]:
    # Some suite helpers enable x64 at import time; module parameters would then
    # be float64 and add parameter casts to the exported graphs.
    original_x64 = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", original_x64)


def _iter_nodes(model: onnx.ModelProto) -> Iterator[onnx.NodeProto]:
    def visit(nodes: Iterable[onnx.NodeProto]) -> Iterator[onnx.NodeProto]:
        for node in nodes:
            yield node
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    yield from visit(attr.g.node)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for graph in attr.graphs:
                        yield from visit(graph.node)

    yield from visit(model.graph.node)
    for function in model.functions:
        yield from visit(function.node)


def _count(model: onnx.ModelProto, op_type: str) -> int:
    return sum(node.op_type == op_type for node in _iter_nodes(model))


def _run(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(None, {session.get_inputs()[0].name: x})
    result: np.ndarray = np.asarray(actual)
    return result


def _massive_activation_rows(seed: int = 0) -> np.ndarray:
    """Rows with a few huge channels, like DINOv3/Lingbot residual streams."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((257, 384)).astype(np.float32)
    x[:, 7] += 1700.0
    x[:, 123] -= 900.0
    x[:, 300] += np.where(rng.standard_normal(257) > 0, 300.0, -300.0)
    rows: np.ndarray = x.astype(np.float32)
    return rows


def _eqx_ln(features: int) -> Callable[[jax.Array], jax.Array]:
    layer = eqx.nn.LayerNorm(features, eps=1e-5)
    return lambda x: jax.vmap(layer)(x)


def _nnx_ln(features: int, *, fast: bool = True) -> Callable[[jax.Array], jax.Array]:
    layer = nnx.LayerNorm(
        features, epsilon=1e-5, use_fast_variance=fast, rngs=nnx.Rngs(0)
    )
    return lambda x: layer(x)


def _linen_ln(features: int) -> Callable[[jax.Array], jax.Array]:
    module = nn.LayerNorm(epsilon=1e-5)
    params = module.init(jax.random.PRNGKey(0), jnp.zeros((1, features)))
    return lambda x: module.apply(params, x)


_PLUGINS: dict[str, Callable[[int], Callable[[jax.Array], jax.Array]]] = {
    "eqx": _eqx_ln,
    "nnx": _nnx_ln,
    "linen": _linen_ln,
}


@pytest.mark.parametrize("plugin", list(_PLUGINS))
@pytest.mark.parametrize(
    ("opset", "normalization_mode", "expect_native"),
    [
        (16, "auto", False),
        (16, "prefer_native", False),
        (16, "force_decomposed", False),
        (17, "auto", False),
        (17, "prefer_native", True),
        (23, "auto", False),
        (23, "prefer_native", True),
        (23, "force_decomposed", False),
    ],
)
def test_layer_norm_policy_matches_opset_and_mode(
    plugin: str, opset: int, normalization_mode: str, expect_native: bool
) -> None:
    fn = _PLUGINS[plugin](32)
    x = np.asarray(
        jax.random.normal(jax.random.PRNGKey(1), (6, 32)) * 3 + 5, np.float32
    )
    expected = np.asarray(fn(jnp.asarray(x)))

    model = to_onnx(fn, [x], opset=opset, normalization_mode=normalization_mode)

    assert (_count(model, "LayerNormalization") == 1) == expect_native
    np.testing.assert_allclose(_run(model, x), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "make_fn",
    [lambda: _eqx_ln(384), lambda: _nnx_ln(384, fast=False)],
    ids=["eqx", "nnx_slow_variance"],
)
@pytest.mark.parametrize(
    ("opset", "normalization_mode"),
    [(16, "auto"), (18, "force_decomposed"), (23, "force_decomposed")],
)
def test_explicit_layer_norm_keeps_constant_rows_exact(
    make_fn: Callable[[], Callable[[jax.Array], jax.Array]],
    opset: int,
    normalization_mode: str,
) -> None:
    fn = make_fn()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((10, 384)).astype(np.float32)
    constants = [0.09006345, 0.00062171, -0.14996266, -0.04716587, 1700.0, 1 / 3]
    for row, value in enumerate(constants):
        x[row] = np.float32(value)
    expected = np.asarray(fn(jnp.asarray(x)))

    model = to_onnx(fn, [x], opset=opset, normalization_mode=normalization_mode)
    actual = _run(model, x)

    constant_rows = slice(0, len(constants))
    np.testing.assert_array_equal(actual[constant_rows], 0.0)
    np.testing.assert_allclose(
        actual[constant_rows], expected[constant_rows], atol=1e-4
    )
    np.testing.assert_allclose(
        actual[len(constants) :], expected[len(constants) :], rtol=1e-5, atol=1e-5
    )


def test_explicit_layer_norm_propagates_non_finite_constant_rows() -> None:
    fn = _eqx_ln(8)
    x: np.ndarray = np.ones((4, 8), np.float32)
    x[0], x[1], x[2] = np.inf, -np.inf, np.nan
    expected = np.asarray(fn(jnp.asarray(x)))

    actual = _run(to_onnx(fn, [x], opset=23, normalization_mode="force_decomposed"), x)

    assert np.isnan(expected[:3]).all()
    assert np.isnan(actual[:3]).all()
    np.testing.assert_array_equal(actual[3], 0.0)


@pytest.mark.parametrize("normalization_mode", ["auto", "force_decomposed"])
@pytest.mark.parametrize("nan_column", [0, 3, 7], ids=["first", "middle", "last"])
@pytest.mark.parametrize(
    "make_fn",
    [lambda: _eqx_ln(8), lambda: _nnx_ln(8, fast=False)],
    ids=["eqx", "nnx_slow_variance"],
)
def test_explicit_layer_norm_propagates_nan_in_otherwise_constant_rows(
    make_fn: Callable[[], Callable[[jax.Array], jax.Array]],
    nan_column: int,
    normalization_mode: str,
) -> None:
    # ReduceMin/ReduceMax may skip NaN, so min == max must not mark these rows
    # constant.
    fn = make_fn()
    x: np.ndarray = np.ones((3, 8), np.float32)
    x[0, nan_column] = np.nan
    x[1, nan_column] = np.inf
    expected = np.asarray(fn(jnp.asarray(x)))

    model = to_onnx(fn, [x], opset=23, normalization_mode=normalization_mode)
    actual = _run(model, x)

    assert np.isnan(expected[:2]).all()
    np.testing.assert_array_equal(actual, expected)


# Accuracy gate on the massive-activation rows, as maximum absolute errors
# against the float64 two-pass LayerNorm and against JAX. Bounds are the errors
# measured with ONNX Runtime 1.29 CPU (AMD Ryzen 9 9950X3D; JAX 0.10.2 and
# 0.11.1, identical results) rounded up to two significant digits. Change them
# only with a justification and before/after evidence; never derive them from
# the run under test.
_LAYER_NORM_ERROR_BOUNDS: dict[tuple[str, str], tuple[float, float]] = {
    # (variant, normalization_mode): (float64 reference, JAX parity)
    ("eqx", "auto"): (4.1e-6, 3.9e-6),
    ("eqx", "force_decomposed"): (4.1e-6, 3.9e-6),
    ("eqx", "prefer_native"): (1.2e-5, 1.2e-5),
    ("nnx_slow", "auto"): (4.1e-6, 3.9e-6),
    ("nnx_slow", "force_decomposed"): (4.1e-6, 3.9e-6),
    ("nnx_slow", "prefer_native"): (1.2e-5, 1.2e-5),
    ("nnx_fast", "auto"): (4.1e-6, 5.8e-6),
    ("nnx_fast", "force_decomposed"): (4.1e-6, 5.8e-6),
    ("nnx_fast", "prefer_native"): (1.2e-5, 1.4e-5),
    ("linen", "auto"): (4.1e-6, 5.8e-6),
    ("linen", "force_decomposed"): (4.1e-6, 5.8e-6),
    ("linen", "prefer_native"): (1.2e-5, 1.4e-5),
}

_GATE_VARIANTS: dict[str, Callable[[], Callable[[jax.Array], jax.Array]]] = {
    "eqx": lambda: _eqx_ln(384),
    "nnx_slow": lambda: _nnx_ln(384, fast=False),
    "nnx_fast": lambda: _nnx_ln(384),
    "linen": lambda: _linen_ln(384),
}


@pytest.mark.parametrize(
    "optimization_level",
    [
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    ],
    ids=["ort_optimized", "ort_unoptimized"],
)
@pytest.mark.parametrize(
    ("variant", "normalization_mode"), list(_LAYER_NORM_ERROR_BOUNDS)
)
def test_layer_norm_accuracy_on_massive_activations_within_locked_bounds(
    variant: str,
    normalization_mode: str,
    optimization_level: ort.GraphOptimizationLevel,
) -> None:
    fn = _GATE_VARIANTS[variant]()
    x = _massive_activation_rows()
    x64: np.ndarray = x.astype(np.float64)
    centered = x64 - x64.mean(-1, keepdims=True)
    exact = centered / np.sqrt((centered**2).mean(-1, keepdims=True) + 1e-5)
    jax_out = np.asarray(fn(jnp.asarray(x)), np.float64)

    model = to_onnx(fn, [x], opset=23, normalization_mode=normalization_mode)
    options = ort.SessionOptions()
    options.graph_optimization_level = optimization_level
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(None, {session.get_inputs()[0].name: x})
    onnx_out = np.asarray(actual, np.float64)

    reference_error = float(np.abs(onnx_out - exact).max())
    jax_error = float(np.abs(onnx_out - jax_out).max())
    reference_bound, jax_bound = _LAYER_NORM_ERROR_BOUNDS[(variant, normalization_mode)]
    report = (
        f"float64 reference error {reference_error:.3e} (bound {reference_bound:.1e}), "
        f"JAX parity error {jax_error:.3e} (bound {jax_bound:.1e})"
    )
    assert reference_error <= reference_bound, report
    assert jax_error <= jax_bound, report


def _optimized_ln_family_count(model: onnx.ModelProto, path: Path) -> int:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(path)
    ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    optimized = onnx.load(str(path))
    return sum(node.op_type in LN_FAMILY for node in optimized.graph.node)


def _textbook_pow_layer_norm() -> onnx.ModelProto:
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [-1])
    two = helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])
    eps = helper.make_tensor("eps", TensorProto.FLOAT, [], [1e-5])
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [8], [1.0] * 8)
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [8], [0.0] * 8)
    nodes = [
        helper.make_node("ReduceMean", ["x", "axes"], ["mean"]),
        helper.make_node("Sub", ["x", "mean"], ["d"]),
        helper.make_node("Pow", ["d", "two"], ["d2"]),
        helper.make_node("ReduceMean", ["d2", "axes"], ["var"]),
        helper.make_node("Add", ["var", "eps"], ["ve"]),
        helper.make_node("Sqrt", ["ve"], ["std"]),
        helper.make_node("Div", ["d", "std"], ["n"]),
        helper.make_node("Mul", ["n", "scale"], ["s"]),
        helper.make_node("Add", ["s", "bias"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "textbook_layer_norm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])],
        [axes, two, eps, scale, bias],
    )
    return helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", 18)]
    )


@pytest.mark.parametrize(
    "make_fn",
    [
        lambda: _eqx_ln(8),
        lambda: _nnx_ln(8),
        lambda: _nnx_ln(8, fast=False),
        lambda: _linen_ln(8),
    ],
    ids=["eqx", "nnx_fast", "nnx_slow", "linen"],
)
def test_explicit_layer_norm_is_not_refused_by_onnxruntime(
    make_fn: Callable[[], Callable[[jax.Array], jax.Array]], tmp_path: Path
) -> None:
    # Positive control: ORT must fuse a Pow-based textbook LayerNorm, otherwise
    # the negative assertion below would be vacuous.
    if not _optimized_ln_family_count(_textbook_pow_layer_norm(), tmp_path / "c.onnx"):
        pytest.skip("this onnxruntime does not fuse textbook LayerNorm patterns")
    x: np.ndarray = np.zeros((4, 8), np.float32)
    model = to_onnx(make_fn(), [x], opset=23, normalization_mode="force_decomposed")

    assert _optimized_ln_family_count(model, tmp_path / "opt.onnx") == 0


def test_explicit_layer_norm_float16_computes_statistics_in_float32() -> None:
    fn = _eqx_ln(64)
    x = np.asarray(
        jax.random.normal(jax.random.PRNGKey(2), (5, 64)) * 4 + 3, np.float16
    )
    expected = np.asarray(fn(jnp.asarray(x)))

    model = to_onnx(fn, [x], opset=23, normalization_mode="force_decomposed")
    graph_input = model.graph.input[0].name
    graph_output = model.graph.output[0].name
    consumers = [node for node in model.graph.node if graph_input in node.input]
    (producer,) = [node for node in model.graph.node if graph_output in node.output]
    actual = _run(model, x)

    def cast_target(node: onnx.NodeProto) -> tuple[str, int | None]:
        to = next((a for a in node.attribute if a.name == "to"), None)
        return node.op_type, None if to is None else int(to.i)

    assert [cast_target(node) for node in consumers] == [("Cast", TensorProto.FLOAT)]
    assert cast_target(producer) == ("Cast", TensorProto.FLOAT16)
    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_explicit_layer_norm_inside_loop_below_opset_17() -> None:
    layer = eqx.nn.LayerNorm(16, eps=1e-5)

    def fn(x: jax.Array) -> jax.Array:
        return jax.lax.fori_loop(0, 3, lambda _, v: jax.vmap(layer)(v) * 2.0, x)

    x: np.ndarray = np.asarray(
        jax.random.normal(jax.random.PRNGKey(4), (3, 16)), np.float32
    )
    expected = np.asarray(fn(jnp.asarray(x)))

    model = to_onnx(fn, [x], opset=16)

    assert _count(model, "LayerNormalization") == 0
    assert _count(model, "ReduceMin") >= 1
    np.testing.assert_allclose(_run(model, x), expected, rtol=1e-5, atol=1e-5)
