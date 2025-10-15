# jax2onnx/plugins/equinox/eqx/nn/conv.py

from __future__ import annotations

import equinox as eqx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    register_example,
    with_prng_key,
)


onnx_function(eqx.nn.Conv2d)


register_example(
    component="equinox.nn.Conv2d",
    description="2D convolution layer from Equinox.",
    source="https://docs.kidger.site/equinox/api/nn/conv/#equinox.nn.Conv2d",
    since="v0.9.1",
    context="primitives.eqx",
    children=["equinox.nn.Conv"],
    testcases=[
        {
            "testcase": "eqx_conv2d_nchw",
            "callable": construct_and_call(
                eqx.nn.Conv2d,
                in_channels=3,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
                key=with_prng_key(0),
            ),
            "input_shapes": [(3, 32, 32)],
            "post_check_onnx_graph": EG(
                ["Conv"],
                search_functions=True,
                no_unused_function_inputs=True,
            ),
            "run_only_f32_variant": True,
        }
    ],
)
