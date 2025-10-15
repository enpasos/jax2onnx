# jax2onnx/plugins/jax/lax/reduce_and.py

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
    jaxpr_primitive=jax.lax.reduce_and_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_and.html",
    onnx=[
        {
            "component": "ReduceMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMin.html",
        }
    ],
    since="v0.6.1",
    context="primitives.lax",
    component="reduce_and",
    testcases=[
        {
            "testcase": "reduce_and_all_true",
            "callable": lambda x: jnp.all(x, axis=None),
            "input_shapes": [(3, 3)],
            "input_dtypes": [jnp.bool_],
            "post_check_onnx_graph": EG(
                ["Cast:3x3 -> ReduceMin -> Cast"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_and_one_false",
            "callable": lambda x: jnp.all(x, axis=None),
            "input_values": [jnp.array([[True, True], [True, False]], dtype=jnp.bool_)],
            "post_check_onnx_graph": EG(
                ["Cast:2x2 -> ReduceMin -> Cast"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_and_keepdims",
            "callable": lambda x: jnp.all(x, axis=(1,), keepdims=True),
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

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_boolean_reduction(ctx, eqn, mode="reduce_and")
