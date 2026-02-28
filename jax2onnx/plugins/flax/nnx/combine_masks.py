# jax2onnx/plugins/flax/nnx/combine_masks.py

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive="nnx.combine_masks",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/attention.html#flax.nnx.combine_masks",
    onnx=[
        {"component": "And", "doc": "https://onnx.ai/onnx/operators/onnx__And.html"},
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
    ],
    since="0.12.2",
    context="primitives.nnx",
    component="combine_masks",
    testcases=[
        {
            "testcase": "combine_masks_two",
            "callable": lambda m1, m2: nnx.combine_masks(m1, m2, dtype=jnp.float32),
            "input_shapes": [(2, 1, 4, 4), (2, 1, 4, 4)],
            "input_dtypes": [np.bool_, np.bool_],
            "expected_output_shapes": [(2, 1, 4, 4)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "combine_masks_with_none",
            "callable": lambda m1, m2: nnx.combine_masks(
                m1, None, m2, dtype=jnp.float32
            ),
            "input_shapes": [(2, 1, 4, 4), (2, 1, 4, 4)],
            "input_dtypes": [np.bool_, np.bool_],
            "expected_output_shapes": [(2, 1, 4, 4)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
        },
    ],
)
class CombineMasksPlugin(PrimitiveLeafPlugin):
    """Metadata plugin for ``flax.nnx.combine_masks`` (inlined to JAX ops)."""

    def lower(self, ctx, eqn):  # type: ignore[override]
        raise NotImplementedError(
            "nnx.combine_masks primitive should not reach lowering; it is inlined."
        )
