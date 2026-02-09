# jax2onnx/plugins/examples/nnx/simple_clip.py

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)


class SimpleModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        pass

    def __call__(self, input):
        x = jnp.clip(input, 0.0, 1.0)
        return x


register_example(
    component="SimpleModel",
    description="Minimal NNX model that applies jnp.clip.",
    source="https://github.com/enpasos/jax2onnx/issues/178",
    since="0.12.0",
    context="examples.nnx",
    children=["jax.numpy.clip"],
    testcases=[
        {
            "testcase": "simple_model_clip_nhwc",
            "callable": construct_and_call(
                SimpleModel,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 8, 8, 3)],
            "expected_output_shapes": [(1, 8, 8, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    (
                        "Clip:1x8x8x3",
                        {"counts": {"Clip": 1, "Transpose": 0}},
                    )
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "simple_model_clip_nchw_io",
            "callable": construct_and_call(
                SimpleModel,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 8, 8, 3)],
            "expected_output_shapes": [(1, 3, 8, 8)],
            "inputs_as_nchw": [0],
            "outputs_as_nchw": [0],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    (
                        "Clip:1x3x8x8",
                        {"counts": {"Clip": 1, "Transpose": 0}},
                    )
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
