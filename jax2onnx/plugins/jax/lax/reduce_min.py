# jax2onnx/plugins/jax/lax/reduce_min.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._reduce_utils import lower_reduction
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_min_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_min.html",
    onnx=[
        {
            "component": "ReduceMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMin.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="reduce_min",
    testcases=[
        {
            "testcase": "reduce_min",
            "callable": lambda x: jnp.min(x, axis=None),
            "input_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                ["ReduceMin"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_min_allaxes",
            "callable": lambda x: jnp.min(x, axis=None),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceMin"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_min_keepdims",
            "callable": lambda x: jnp.min(x, axis=(1,), keepdims=True),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceMin:3 -> Reshape:3x1 -> Expand:3x1"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ReduceMinPlugin(PrimitiveLeafPlugin):
    """IR-only lowering of ``lax.reduce_min`` via ONNX ReduceMin."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_reduction(ctx, eqn, op_type="ReduceMin", allow_dtype_param=False)
