# tests/extra_tests/converter/test_onnx_function_call_params.py

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    with_rng_seed,
)
from jax2onnx.user_interface import to_onnx


@onnx_function
class FunctionDropout(nnx.Module):
    def __init__(self, rate: float, *, rngs: nnx.Rngs):
        super().__init__()
        self.dropout = nnx.Dropout(rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        return self.dropout(x, deterministic=deterministic)


def test_onnx_function_consumes_call_time_param():
    fn = construct_and_call(
        FunctionDropout,
        rate=0.25,
        rngs=with_rng_seed(0),
    )

    model = to_onnx(
        fn=fn,
        inputs=[jax.ShapeDtypeStruct((2, 3), jnp.float32)],
        input_params={"deterministic": True},
        model_name="function_dropout",
        opset=21,
        enable_double_precision=False,
    )

    check = EG(
        ["Dropout"],
        search_functions=True,
        no_unused_inputs=True,
        no_unused_function_inputs=True,
    )
    assert check(model)


@onnx_function
class _SimpleBlock(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(x)


class _SimpleStack(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.block1 = _SimpleBlock(dim, rngs=rngs)
        self.block2 = _SimpleBlock(dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.block2(self.block1(x))


def test_function_node_names_are_human_friendly():
    model = _SimpleStack(8, rngs=nnx.Rngs(0))
    onnx_model = to_onnx(
        model,
        inputs=[jax.ShapeDtypeStruct((2, 8), jnp.float32)],
        return_mode="proto",
        model_name="simple_stack",
    )

    fn_nodes = [
        node
        for node in onnx_model.graph.node
        if node.domain.startswith("custom") and node.op_type == "_SimpleBlock"
    ]
    assert [node.op_type for node in fn_nodes] == ["_SimpleBlock", "_SimpleBlock"]
    assert [node.name for node in fn_nodes] == ["_SimpleBlock_1", "_SimpleBlock_2"]

    functions = list(getattr(onnx_model, "functions", []))
    assert len(functions) == 2
    assert {fn.name for fn in functions} == {"_SimpleBlock"}
    domains = {fn.domain for fn in functions}
    assert len(domains) == 2
    assert any(domain == "custom" for domain in domains)
    assert any(domain.startswith("custom.") for domain in domains)
