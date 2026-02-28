# jax2onnx/plugins/flax/nnx/glu.py

from __future__ import annotations

from flax import nnx

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive="nnx.glu",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.glu",
    onnx=[
        {
            "component": "Split",
            "doc": "https://onnx.ai/onnx/operators/onnx__Split.html",
        },
        {
            "component": "Sigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
    ],
    since="0.12.2",
    context="primitives.nnx",
    component="glu",
    testcases=[
        {
            "testcase": "glu_last_axis",
            "callable": lambda x: nnx.glu(x),
            "input_shapes": [(2, 8)],
            "expected_output_shapes": [(2, 4)],
        },
        {
            "testcase": "glu_axis_1",
            "callable": lambda x: nnx.glu(x, axis=1),
            "input_shapes": [("B", 6, 4)],
            "expected_output_shapes": [("B", 3, 4)],
        },
    ],
)
class GLUPlugin(PrimitiveLeafPlugin):
    """Metadata plugin for ``flax.nnx.glu`` (inlined to JAX ops)."""

    def lower(self, ctx, eqn):  # type: ignore[override]
        raise NotImplementedError(
            "nnx.glu primitive should not reach lowering; it is inlined."
        )
