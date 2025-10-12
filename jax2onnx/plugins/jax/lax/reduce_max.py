# jax2onnx/plugins/jax/lax/reduce_max.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._reduce_utils import lower_reduction
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _check_reduce_max_axes_input(model) -> bool:
    """Ensure ReduceMax uses the post-opset-18 signature (axes as input)."""

    for node in model.graph.node:
        if node.op_type != "ReduceMax":
            continue
        if any(attr.name == "axes" for attr in node.attribute):
            return False
        if len(node.input) != 2:
            return False
    return True


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_max_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_max.html",
    onnx=[
        {
            "component": "ReduceMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMax.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="reduce_max",
    testcases=[
        {
            "testcase": "reduce_max",
            "callable": lambda x: jnp.max(x, axis=(0,)),
            "input_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ReduceMax:3",
                        "inputs": {1: {"const": 0.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_max_allaxes",
            "callable": lambda x: jnp.max(x),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceMax"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_max_axes_input",
            "callable": lambda x: jnp.max(x, axis=(1,)),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": _check_reduce_max_axes_input,
        },
        {
            "testcase": "reduce_max_keepdims",
            "callable": lambda x: jnp.max(x, axis=(1,), keepdims=True),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceMax:3 -> Reshape:3x1 -> Expand:3x1"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ReduceMaxPlugin(PrimitiveLeafPlugin):
    """IR-only lowering of ``lax.reduce_max`` via ONNX ReduceMax."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_reduction(ctx, eqn, op_type="ReduceMax", allow_dtype_param=False)
