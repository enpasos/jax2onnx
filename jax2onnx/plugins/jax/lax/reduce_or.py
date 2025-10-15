# jax2onnx/plugins/jax/lax/reduce_or.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._reduce_utils import lower_boolean_reduction
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_or_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_or.html",
    onnx=[
        {
            "component": "ReduceMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMax.html",
        }
    ],
    since="v0.6.1",
    context="primitives.lax",
    component="reduce_or",
    testcases=[
        {
            "testcase": "reduce_or_all_false",
            "callable": lambda x: jnp.any(x, axis=None),
            "input_shapes": [(3, 3)],
            "input_dtypes": [jnp.bool_],
            "post_check_onnx_graph": EG(
                ["Cast:3x3 -> ReduceMax -> Cast"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_or_one_true",
            "callable": lambda x: jnp.any(x, axis=None),
            "input_values": [
                jnp.array([[False, False], [True, False]], dtype=jnp.bool_)
            ],
            "post_check_onnx_graph": EG(
                ["Cast:2x2 -> ReduceMax -> Cast"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_or_keepdims",
            "callable": lambda x: jnp.any(x, axis=(1,), keepdims=True),
            "input_shapes": [(3, 4)],
            "input_dtypes": [jnp.bool_],
            "post_check_onnx_graph": EG(
                ["Cast:3x4 -> ReduceMax:3 -> Cast:3 -> Reshape:3x1 -> Expand:3x1"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ReduceOrPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.reduce_or`` via ReduceMax + Cast."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_boolean_reduction(ctx, eqn, mode="reduce_or")
