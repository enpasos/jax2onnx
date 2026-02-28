# jax2onnx/plugins/jax/lax/cummin.py

from __future__ import annotations

import jax
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._cum_extrema import lower_cum_extrema
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive=jax.lax.cummin_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cummin.html",
    onnx=[
        {"component": "Neg", "doc": "https://onnx.ai/onnx/operators/onnx__Neg.html"},
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
    component="cummin",
    testcases=[
        {
            "testcase": "cummin_axis1",
            "callable": lambda x: jax.lax.cummin(x, axis=1),
            "input_values": [
                np.asarray(
                    [[1.0, -2.0, 3.0, 0.5], [0.0, 5.0, 2.0, -1.0]], dtype=np.float32
                )
            ],
            "post_check_onnx_graph": EG(
                ["Neg -> MaxPool -> Neg"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "cummin_reverse_last_axis",
            "callable": lambda x: jax.lax.cummin(x, axis=x.ndim - 1, reverse=True),
            "input_values": [np.asarray([3.0, 1.0, 4.0, 2.0, -1.0], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Neg -> MaxPool -> Neg"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class CumMinPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.cummin`` via sign-flip + ``MaxPool``."""

    def lower(self, ctx, eqn):  # type: ignore[no-untyped-def]
        lower_cum_extrema(ctx, eqn, mode="min")
