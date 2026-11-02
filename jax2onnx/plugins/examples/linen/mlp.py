# jax2onnx/plugins/examples/linen/mlp.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from flax import linen as nn

from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.flax.nnx.dropout import post_check_onnx_graph
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)


class LinenMLP(nn.Module):
    din: int
    dmid: int
    dout: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:
        x = nn.Dense(self.dmid)(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.gelu(x)
        return nn.Dense(self.dout)(x)


def _mlp_graph_check(batch: str) -> Callable[[Any], bool]:
    """Check this model's FP32 path and deliberately mixed FP64 path."""

    float32_check = expect_graph(
        [
            f"Gemm:{batch}x20 -> BatchNormalization:{batch}x20 -> "
            f"Dropout:{batch}x20 -> Gelu:{batch}x20 -> Gemm:{batch}x10",
        ],
        symbols={"B": None},
        must_absent=["Not"],
        no_unused_inputs=True,
        mode="all",
    )
    float64_check = expect_graph(
        [
            (
                f"Gemm:{batch}x20 -> Sub:{batch}x20 -> Mul:{batch}x20 -> "
                f"Add:{batch}x20 -> Dropout:{batch}x20 -> Gelu:{batch}x20 -> "
                f"Gemm:{batch}x10",
                {
                    "counts": {
                        "Gemm": 2,
                        "Sub": 1,
                        "Add": 2,
                        "Sqrt": 1,
                        "Div": 1,
                        "Mul": 2,
                        "Dropout": 1,
                        "Gelu": 1,
                    }
                },
            ),
            # Existing FP32 statistics and affine parameters form the BN
            # coefficient before its explicit conversion for the FP64 input.
            f"Add:1x20 -> Sqrt:1x20 -> Div:1x20 -> Mul:1x20 -> "
            f"Cast:1x20 -> Mul:{batch}x20",
        ],
        symbols={"B": None},
        must_absent=["BatchNormalization", "Not"],
        no_unused_inputs=True,
        mode="all",
    )

    def check(model: Any) -> bool:
        graph = model.graph
        if len(graph.input) != 1 or len(graph.output) != 1:
            return False
        elem_type = graph.input[0].type.tensor_type.elem_type
        if graph.output[0].type.tensor_type.elem_type != elem_type:
            return False
        if elem_type == 1:
            return float32_check(model)
        if elem_type != 11:
            return False

        value_types = {
            value.name: value.type.tensor_type.elem_type
            for value in (*graph.value_info, *graph.output)
        }
        input_dependent = {graph.input[0].name}
        coefficients: set[str] = set()
        compute_ops = {"Gemm", "Sub", "Add", "Sqrt", "Div", "Mul", "Dropout", "Gelu"}
        for node in graph.node:
            depends_on_input = any(name in input_dependent for name in node.input)
            if depends_on_input:
                input_dependent.update(node.output)
            if node.op_type in compute_ops:
                expected_dtype = 11 if depends_on_input else 1
                if not node.output or value_types.get(node.output[0]) != expected_dtype:
                    return False
                if not depends_on_input:
                    coefficients.add(node.output[0])
        coefficient_casts = [
            node
            for node in graph.node
            if node.op_type == "Cast" and node.input[0] in coefficients
        ]
        if len(coefficient_casts) != 1:
            return False
        if value_types.get(coefficient_casts[0].output[0]) != 11:
            return False
        return float64_check(model)

    return check


register_example(
    component="LinenMLP",
    description="A simple Linen MLP with BatchNorm, Dropout, and GELU activation.",
    source="https://github.com/google/flax/blob/main/README.md",
    since="0.11.0",
    context="examples.linen",
    children=[
        "flax.linen.Dense",
        "flax.linen.BatchNorm",
        "flax.linen.Dropout",
        "flax.linen.gelu",
    ],
    testcases=[
        {
            "testcase": "simple_linen_mlp_static",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=LinenMLP,
                input_shape=(1, 30),
                din=30,
                dmid=20,
                dout=10,
                rngs=with_rng_seed(17),
            ),
            "input_shapes": [(7, 30)],
            "post_check_onnx_graph": _mlp_graph_check("7"),
        },
        {
            "testcase": "simple_linen_mlp",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=LinenMLP,
                input_shape=(1, 30),
                din=30,
                dmid=20,
                dout=10,
                rngs=with_rng_seed(17),
            ),
            "input_shapes": [("B", 30)],
            "run_only_dynamic": True,
            "post_check_onnx_graph": _mlp_graph_check("B"),
        },
        {
            "testcase": "simple_linen_mlp_with_call_params",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=LinenMLP,
                input_shape=(1, 30),
                din=30,
                dmid=20,
                dout=10,
                rngs=with_rng_seed(17),
            ),
            "input_shapes": [("B", 30)],
            "input_params": {"deterministic": True},
            "post_check_onnx_graph": post_check_onnx_graph,
        },
    ],
)
