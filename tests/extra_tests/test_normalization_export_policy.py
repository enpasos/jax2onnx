# tests/extra_tests/test_normalization_export_policy.py

from __future__ import annotations

from collections.abc import Iterable, Iterator

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
from onnx.reference import ReferenceEvaluator
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.plugins.jax.lax._control_flow_utils import make_subgraph_context


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


def _run(model: onnx.ModelProto, *inputs: np.ndarray) -> np.ndarray:
    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feed = {
        meta.name: value
        for meta, value in zip(session.get_inputs(), inputs, strict=True)
    }
    (actual,) = session.run(None, feed)
    return actual


@pytest.mark.parametrize(
    ("opset", "normalization_mode", "expect_native"),
    [
        (22, "auto", False),
        (23, "auto", True),
        (22, "prefer_native", False),
        (23, "prefer_native", True),
        (23, "force_decomposed", False),
    ],
)
def test_rms_norm_policy_matches_opset_and_mode(
    opset: int,
    normalization_mode: str,
    expect_native: bool,
) -> None:
    norm = nnx.RMSNorm(
        num_features=6,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    x = jnp.arange(12, dtype=jnp.float32).reshape(2, 6) / 7
    expected = np.asarray(norm(x))

    model = to_onnx(
        norm,
        [x],
        opset=opset,
        normalization_mode=normalization_mode,
    )
    op_types = [node.op_type for node in _iter_nodes(model)]
    assert (op_types.count("RMSNormalization") == 1) is expect_native
    assert (op_types.count("ReduceMean") == 1) is not expect_native

    actual = _run(model, np.asarray(x))
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize(
    ("opset", "normalization_mode"),
    [(22, "auto"), (23, "force_decomposed")],
)
def test_force_decomposed_rms_norm_keeps_float16_constants_type_compatible(
    opset: int,
    normalization_mode: str,
) -> None:
    norm = nnx.RMSNorm(
        num_features=6,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        rngs=nnx.Rngs(0),
    )
    x = jnp.arange(12, dtype=jnp.float16).reshape(2, 6) / jnp.float16(7)
    expected = np.asarray(norm(x))

    model = to_onnx(
        norm,
        [x],
        opset=opset,
        normalization_mode=normalization_mode,
    )
    assert not any(node.op_type == "RMSNormalization" for node in _iter_nodes(model))

    actual = _run(model, np.asarray(x))
    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_native_rms_norm_promotes_float16_statistics_to_float32() -> None:
    norm = nnx.RMSNorm(
        num_features=6,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        rngs=nnx.Rngs(0),
    )
    x = jnp.arange(12, dtype=jnp.float16).reshape(2, 6) / jnp.float16(7)
    expected = np.asarray(norm(x))

    model = to_onnx(norm, [x], opset=23, normalization_mode="prefer_native")
    node = next(
        node for node in _iter_nodes(model) if node.op_type == "RMSNormalization"
    )
    stash_type = next(attr for attr in node.attribute if attr.name == "stash_type")
    assert onnx.helper.get_attribute_value(stash_type) == onnx.TensorProto.FLOAT

    actual = _run(model, np.asarray(x))
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)

    reference = ReferenceEvaluator(model)
    (reference_actual,) = reference.run(
        None,
        {model.graph.input[0].name: np.asarray(x)},
    )
    np.testing.assert_allclose(reference_actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    ("opset", "normalization_mode"),
    [(22, "auto"), (23, "force_decomposed")],
)
def test_linen_rms_norm_without_float32_promotion_has_typed_explicit_constants(
    opset: int,
    normalization_mode: str,
) -> None:
    module = nn.RMSNorm(
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        force_float32_reductions=False,
    )
    x = jnp.arange(12, dtype=jnp.float16).reshape(2, 6) / jnp.float16(7)
    variables = module.init(jax.random.PRNGKey(0), x)

    def fn(value: jax.Array) -> jax.Array:
        return module.apply(variables, value)

    expected = np.asarray(fn(x))
    model = to_onnx(
        fn,
        [x],
        opset=opset,
        normalization_mode=normalization_mode,
    )
    assert not any(node.op_type == "RMSNormalization" for node in _iter_nodes(model))

    actual = _run(model, np.asarray(x))
    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_prefer_native_mode_does_not_expand_to_linen_instance_norm() -> None:
    module = nn.InstanceNorm(dtype=jnp.float32, param_dtype=jnp.float32)
    x = jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 3, 4) / 7
    variables = module.init(jax.random.PRNGKey(0), x)

    def fn(value: jax.Array) -> jax.Array:
        return module.apply(variables, value)

    expected = np.asarray(fn(x))
    model = to_onnx(fn, [x], opset=23, normalization_mode="prefer_native")
    assert not any(node.op_type == "GroupNormalization" for node in _iter_nodes(model))
    assert sum(node.op_type == "ReduceMean" for node in _iter_nodes(model)) == 2

    actual = _run(model, np.asarray(x))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@onnx_function
class _RMSFunction(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.norm = nnx.RMSNorm(6, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.norm(x)


@pytest.mark.parametrize(
    ("normalization_mode", "expect_native"),
    [("prefer_native", True), ("force_decomposed", False)],
)
def test_normalization_mode_reaches_onnx_function_bodies(
    normalization_mode: str,
    expect_native: bool,
) -> None:
    module = _RMSFunction(rngs=nnx.Rngs(0))
    x = jnp.arange(12, dtype=jnp.float32).reshape(2, 6) / 7
    expected = np.asarray(module(x))

    model = to_onnx(
        module,
        [x],
        opset=23,
        normalization_mode=normalization_mode,
    )
    assert model.functions
    native_count = sum(
        node.op_type == "RMSNormalization" for node in _iter_nodes(model)
    )
    assert (native_count == 1) is expect_native

    actual = _run(model, np.asarray(x))
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize(
    "normalization_mode", ["auto", "prefer_native", "force_decomposed"]
)
def test_normalization_mode_reaches_control_flow_contexts(
    normalization_mode: str,
) -> None:
    parent = IRContext(
        opset=23,
        enable_double_precision=False,
        normalization_mode=normalization_mode,
        input_specs=[],
    )
    child = make_subgraph_context(parent, prefix="normalization_policy")
    assert child.normalization_mode == normalization_mode
