# jax2onnx/plugins/jax/lax/cummax.py

from __future__ import annotations

import jax
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._cum_extrema import lower_cum_extrema
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive=jax.lax.cummax_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cummax.html",
    onnx=[
        {
            "component": "MaxPool",
            "doc": "https://onnx.ai/onnx/operators/onnx__MaxPool.html",
        },
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="cummax",
    testcases=[
        {
            "testcase": "cummax_axis1",
            "callable": lambda x: jax.lax.cummax(x, axis=1),
            "input_values": [
                np.asarray(
                    [[1.0, -2.0, 3.0, 0.5], [0.0, 5.0, 2.0, -1.0]], dtype=np.float32
                )
            ],
            "post_check_onnx_graph": EG(
                ["MaxPool"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "cummax_reverse_last_axis",
            "callable": lambda x: jax.lax.cummax(x, axis=x.ndim - 1, reverse=True),
            "input_values": [np.asarray([3.0, 1.0, 4.0, 2.0, -1.0], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["MaxPool"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class CumMaxPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.cummax`` via reshape + ``MaxPool`` prefix/suffix windows."""

    def lower(self, ctx, eqn):  # type: ignore[no-untyped-def]
        lower_cum_extrema(ctx, eqn, mode="max")
