# jax2onnx/plugins/flax/nnx/silu.py

from __future__ import annotations

from typing import Any

from flax import nnx

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive="nnx.silu",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.silu",
    onnx=[
        {
            "component": "Swish",
            "doc": "https://onnx.ai/onnx/operators/onnx__Swish.html",
        },
        {
            "component": "Sigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
    ],
    since="0.13.2",
    context="primitives.nnx",
    component="silu",
    testcases=[
        {
            "testcase": "silu_opset23",
            "callable": lambda x: nnx.silu(x),
            "input_shapes": [(2, 5)],
            "opset_version": 23,
            "post_check_onnx_graph": expect_graph(
                ["Sigmoid:2x5 -> Mul:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "silu_opset24",
            "callable": lambda x: nnx.silu(x),
            "input_shapes": [(2, 5)],
            "opset_version": 24,
            "post_check_onnx_graph": expect_graph(
                ["Swish:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "swish_alias_opset24",
            "callable": lambda x: nnx.swish(x),
            "input_shapes": [("B", 4)],
            "opset_version": 24,
            "post_check_onnx_graph": expect_graph(
                ["Swish:Bx4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class SiluPlugin(PrimitiveLeafPlugin):
    """Metadata for ``flax.nnx.silu``/``swish``.

    The active conversion patch is supplied by ``jax.nn.silu`` so NNX aliases
    share the same primitive, lowering, and autodiff rules.
    """

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        raise NotImplementedError(
            "nnx.silu should lower through the shared jax.nn.silu primitive."
        )
