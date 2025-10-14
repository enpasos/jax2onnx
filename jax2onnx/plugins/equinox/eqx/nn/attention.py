# jax2onnx/plugins/equinox/eqx/nn/attention.py

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    register_example,
    with_prng_key,
)


onnx_function(eqx.nn.MultiheadAttention)
onnx_function(eqx.nn.RotaryPositionalEmbedding)


register_example(
    component="equinox.nn.MultiheadAttention",
    description="Equinox multi-head attention module.",
    source="https://docs.kidger.site/equinox/api/nn/attention/#attention",
    since="v0.9.1",
    context="primitives.eqx",
    children=["equinox.nn.Linear", "equinox.nn.Dropout"],
    testcases=[
        {
            "testcase": "eqx_multihead_attention",
            "callable": construct_and_call(
                eqx.nn.MultiheadAttention,
                num_heads=4,
                query_size=32,
                key_size=32,
                value_size=32,
                output_size=32,
                key=with_prng_key(0),
            ),
            "input_shapes": [(17, 32), (17, 32), (17, 32)],
            "input_dtypes": [jnp.float32, jnp.float32, jnp.float32],
            "post_check_onnx_graph": EG(
                ["MultiheadAttention_1"],
                search_functions=True,
                no_unused_function_inputs=True,
            ),
            "run_only_f32_variant": True,
        }
    ],
)


register_example(
    component="equinox.nn.RotaryPositionalEmbedding",
    description="Equinox rotary positional embedding module.",
    source="https://docs.kidger.site/equinox/api/nn/attention/#equinox.nn.RotaryPositionalEmbedding",
    since="v0.9.1",
    context="primitives.eqx",
    children=[],
    testcases=[
        {
            "testcase": "eqx_rotary_positional_embedding",
            "callable": construct_and_call(
                eqx.nn.RotaryPositionalEmbedding,
                embedding_size=32,
            ),
            "input_shapes": [(41, 32)],
            "input_dtypes": [jnp.float32],
            "post_check_onnx_graph": EG(
                ["RotaryPositionalEmbedding_1"],
                search_functions=True,
                mode="all",
                no_unused_function_inputs=True,
            ),
            "run_only_f32_variant": True,
        }
    ],
)
