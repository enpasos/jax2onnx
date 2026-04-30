# jax2onnx/plugins/flax/nnx/flip_sequences.py

from __future__ import annotations

from typing import Any

import numpy as np
from flax import nnx

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive="nnx.flip_sequences",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/recurrent.html#flax.nnx.nn.recurrent.flip_sequences",
    onnx=[
        {
            "component": "GatherElements",
            "doc": "https://onnx.ai/onnx/operators/onnx__GatherElements.html",
        },
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        },
    ],
    since="0.12.2",
    context="primitives.nnx",
    component="flip_sequences",
    testcases=[
        {
            "testcase": "flip_sequences_batch_major_with_lengths",
            "callable": lambda x, lengths: nnx.nn.recurrent.flip_sequences(
                x,
                lengths,
                num_batch_dims=1,
                time_major=False,
            ),
            "input_shapes": [(2, 5, 3), (2,)],
            "input_dtypes": [np.float32, np.int32],
            "expected_output_shapes": [(2, 5, 3)],
        },
        {
            "testcase": "flip_sequences_batch_major_no_lengths",
            "callable": lambda x: nnx.nn.recurrent.flip_sequences(
                x,
                None,
                num_batch_dims=1,
                time_major=False,
            ),
            "input_shapes": [("B", 5, 3)],
            "expected_output_shapes": [("B", 5, 3)],
        },
        {
            "testcase": "flip_sequences_time_major_with_lengths",
            "callable": lambda x, lengths: nnx.nn.recurrent.flip_sequences(
                x,
                lengths,
                num_batch_dims=1,
                time_major=True,
            ),
            "input_shapes": [(5, 2, 3), (2,)],
            "input_dtypes": [np.float32, np.int32],
            "expected_output_shapes": [(5, 2, 3)],
        },
    ],
)
class FlipSequencesPlugin(PrimitiveLeafPlugin):
    """Metadata plugin for ``flax.nnx.nn.recurrent.flip_sequences`` (inlined)."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        raise NotImplementedError(
            "nnx.flip_sequences primitive should not reach lowering; it is inlined."
        )
