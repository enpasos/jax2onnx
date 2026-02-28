# jax2onnx/plugins/flax/nnx/hard_tanh.py

from __future__ import annotations

from flax import nnx

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive="nnx.hard_tanh",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.hard_tanh",
    onnx=[
        {"component": "Clip", "doc": "https://onnx.ai/onnx/operators/onnx__Clip.html"}
    ],
    since="0.12.2",
    context="primitives.nnx",
    component="hard_tanh",
    testcases=[
        {
            "testcase": "hard_tanh_basic",
            "callable": lambda x: nnx.hard_tanh(x),
            "input_shapes": [("B", 7)],
            "expected_output_shapes": [("B", 7)],
        },
    ],
)
class HardTanhPlugin(PrimitiveLeafPlugin):
    """Metadata plugin for ``flax.nnx.hard_tanh`` (inlined to JAX ops)."""

    def lower(self, ctx, eqn):  # type: ignore[override]
        raise NotImplementedError(
            "nnx.hard_tanh primitive should not reach lowering; it is inlined."
        )
