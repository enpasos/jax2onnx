# tests/extra_tests/converter/test_onnx_function_unique.py

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from jax2onnx import onnx_function
from jax2onnx.user_interface import to_onnx


@onnx_function(unique=True)
class _UniqueLinearBlock(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(x)


class _ReuseUniqueModel(nnx.Module):
    def __init__(self):
        rng_a = nnx.Rngs(0)
        rng_b = nnx.Rngs(0)
        self.block_a = _UniqueLinearBlock(4, 4, rngs=rng_a)
        self.block_b = _UniqueLinearBlock(4, 4, rngs=rng_b)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.block_b(self.block_a(x))


class _DistinctUniqueModel(nnx.Module):
    def __init__(self):
        rng_a = nnx.Rngs(0)
        rng_b = nnx.Rngs(1)
        self.block_a = _UniqueLinearBlock(4, 4, rngs=rng_a)
        self.block_b = _UniqueLinearBlock(4, 4, rngs=rng_b)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.block_b(self.block_a(x))


def _collect_function_nodes(model, name: str):
    return [node for node in model.graph.node if node.op_type == name]


def _collect_function_defs(model, name: str):
    return [fn for fn in getattr(model, "functions", []) if fn.name == name]


def test_unique_decorator_reuses_matching_blocks_ir():
    model = to_onnx(
        _ReuseUniqueModel(),
        inputs=[(1, 4)],
        model_name="reuse_unique_flag_ir",
    )
    fn_defs = _collect_function_defs(model, "_UniqueLinearBlock")
    assert (
        len(fn_defs) == 1
    ), "Expected a single function definition for identical blocks"
    assert (
        fn_defs[0].domain == "custom._UniqueLinearBlock.unique"
    ), "Unique domain should be stable for identical blocks"

    call_nodes = _collect_function_nodes(model, "_UniqueLinearBlock")
    assert len(call_nodes) == 2, "Both call sites should target the function"
    domains = {node.domain for node in call_nodes}
    assert domains == {
        "custom._UniqueLinearBlock.unique"
    }, "Call sites should share the same domain for reuse"


def test_unique_decorator_distinguishes_different_params_ir():
    model = to_onnx(
        _DistinctUniqueModel(),
        inputs=[(1, 4)],
        model_name="distinct_unique_flag_ir",
    )
    fn_defs = _collect_function_defs(model, "_UniqueLinearBlock")
    assert (
        len(fn_defs) == 2
    ), "Different parameter sets should produce distinct functions"
    domains_def = {fn.domain for fn in fn_defs}
    assert domains_def == {
        "custom._UniqueLinearBlock.unique",
        "custom._UniqueLinearBlock.unique.2",
    }

    call_nodes = _collect_function_nodes(model, "_UniqueLinearBlock")
    assert len(call_nodes) == 2
    domains = {node.domain for node in call_nodes}
    assert domains == {
        "custom._UniqueLinearBlock.unique",
        "custom._UniqueLinearBlock.unique.2",
    }, "Call sites should reference functions with distinct domains"


@onnx_function(unique=True, namespace="my.model")
def _NamespacedSquare(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.square(x)


class _NamespacedModel(nnx.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return _NamespacedSquare(_NamespacedSquare(x))


def test_unique_with_custom_namespace_ir():
    model = to_onnx(
        _NamespacedModel(),
        inputs=[(1, 2)],
        model_name="namespaced_unique_flag_ir",
    )
    fn_defs = _collect_function_defs(model, "_NamespacedSquare")
    assert len(fn_defs) == 1
    assert fn_defs[0].domain == "my.model._NamespacedSquare.unique"
    call_nodes = _collect_function_nodes(model, "_NamespacedSquare")
    assert len(call_nodes) == 2
    assert {node.domain for node in call_nodes} == {"my.model._NamespacedSquare.unique"}
