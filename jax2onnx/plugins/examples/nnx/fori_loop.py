# jax2onnx/plugins/examples/nnx/fori_loop.py

from __future__ import annotations

import jax
from typing import Final

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import register_example


def _model(x: jax.Array) -> jax.Array:
    steps = 5

    def body(index: int, state: tuple[jax.Array, int]) -> tuple[jax.Array, int]:
        value, counter = state
        value = value + 0.1 * value**2
        counter = counter + 1
        return value, counter

    result, _ = jax.lax.fori_loop(0, steps, body, (x, 0))
    return result


def _model_two_inputs(x: jax.Array, y: jax.Array) -> jax.Array:
    steps = 5

    def body(index: int, state: tuple[jax.Array, int]) -> tuple[jax.Array, int]:
        value, counter = state
        value = value + 0.1 * value**2
        counter = counter + 1
        return value, counter

    seed = x + y
    result, _ = jax.lax.fori_loop(0, steps, body, (seed, 0))
    return result


_FORI_LOOP_GRAPH_CHECK: Final = EG(
    [
        {
            "inputs": {
                0: {"const": 5.0},
                1: {"const_bool": True},
                3: {"const": 0.0},
            },
            "path": "Loop:1",
        }
    ],
    no_unused_inputs=True,
)


def _check_named_io_and_structure(model) -> bool:
    input_names = [value.name for value in model.graph.input]
    output_names = [value.name for value in model.graph.output]
    return (
        input_names == ["state_in"]
        and output_names == ["state_out"]
        and _FORI_LOOP_GRAPH_CHECK(model)
    )


def _check_named_io_two_inputs_and_structure(model) -> bool:
    input_names = [value.name for value in model.graph.input]
    output_names = [value.name for value in model.graph.output]
    return (
        input_names == ["state_in_a", "state_in_b"]
        and output_names == ["state_out_sum"]
        and _FORI_LOOP_GRAPH_CHECK(model)
    )


register_example(
    component="ForiLoop",
    description="fori_loop example using nnx-compatible primitives (converter).",
    since="0.5.1",
    context="examples.nnx",
    children=["jax.lax.fori_loop"],
    testcases=[
        {
            "testcase": "fori_loop_counter",
            "callable": _model,
            "input_shapes": [(1,)],
            "expected_output_shapes": [(1,)],
            "post_check_onnx_graph": _FORI_LOOP_GRAPH_CHECK,
        },
        {
            "testcase": "fori_loop_counter_custom_io_names",
            "callable": _model,
            "input_shapes": [(1,)],
            "expected_output_shapes": [(1,)],
            "input_names": ["state_in"],
            "output_names": ["state_out"],
            "post_check_onnx_graph": _check_named_io_and_structure,
        },
        {
            "testcase": "fori_loop_counter_custom_io_names_two_inputs",
            "callable": _model_two_inputs,
            "input_shapes": [(1,), (1,)],
            "expected_output_shapes": [(1,)],
            "input_names": ["state_in_a", "state_in_b"],
            "output_names": ["state_out_sum"],
            "post_check_onnx_graph": _check_named_io_two_inputs_and_structure,
        },
    ],
)
