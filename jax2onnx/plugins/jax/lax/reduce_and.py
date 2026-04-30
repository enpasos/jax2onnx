# jax2onnx/plugins/jax/lax/reduce_and.py

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._reduce_utils import lower_boolean_reduction
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_and_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_and.html",
    onnx=[
        {
            "component": "ReduceMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMin.html",
        }
    ],
    since="0.6.1",
    context="primitives.lax",
    component="reduce_and",
    testcases=[
        {
            "testcase": "reduce_and_all_true",
            "callable": lambda x: jax.lax.reduce_and(x, axes=tuple(range(x.ndim))),
            "input_shapes": [(3, 3)],
            "input_dtypes": [jnp.bool_],
            "post_check_onnx_graph": EG(
                ["Cast:3x3 -> ReduceMin -> Cast"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_and_one_false",
            "callable": lambda x: jax.lax.reduce_and(x, axes=tuple(range(x.ndim))),
            "input_values": [jnp.array([[True, True], [True, False]], dtype=jnp.bool_)],
            "post_check_onnx_graph": EG(
                ["Cast:2x2 -> ReduceMin -> Cast"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_and_keepdims",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                jax.lax.reduce_and(x, axes=(1,)),
                shape=(x.shape[0], 1),
                broadcast_dimensions=(0,),
            ),
            "input_shapes": [(3, 4)],
            "input_dtypes": [jnp.bool_],
            "post_check_onnx_graph": EG(
                ["Cast:3x4 -> ReduceMin:3 -> Cast:3 -> Reshape:3x1 -> Expand:3x1"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ReduceAndPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.reduce_and`` via ReduceMin + Cast."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        lower_boolean_reduction(ctx, eqn, mode="reduce_and")
