# jax2onnx/plugins/flax/nnx/log_sigmoid.py

from __future__ import annotations

from typing import Any

from flax import nnx

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive="nnx.log_sigmoid",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.log_sigmoid",
    onnx=[
        {
            "component": "Softplus",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softplus.html",
        },
        {"component": "Neg", "doc": "https://onnx.ai/onnx/operators/onnx__Neg.html"},
    ],
    since="0.12.2",
    context="primitives.nnx",
    component="log_sigmoid",
    testcases=[
        {
            "testcase": "log_sigmoid_basic",
            "callable": lambda x: nnx.log_sigmoid(x),
            "input_shapes": [("B", 7)],
            "expected_output_shapes": [("B", 7)],
        },
    ],
)
class LogSigmoidPlugin(PrimitiveLeafPlugin):
    """Metadata plugin for ``flax.nnx.log_sigmoid`` (inlined to JAX ops)."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        raise NotImplementedError(
            "nnx.log_sigmoid primitive should not reach lowering; it is inlined."
        )
