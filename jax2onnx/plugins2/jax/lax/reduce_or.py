from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jax2onnx.plugins2.jax.lax._reduce_utils import lower_boolean_reduction
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_or_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_or.html",
    onnx=[
        {
            "component": "ReduceMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMax.html",
        }
    ],
    since="v0.1.0",
    context="primitives2.lax",
    component="reduce_or",
    testcases=[
        {
            "testcase": "reduce_or_axis0",
            "callable": lambda x: jnp.any(x, axis=0),
            "input_shapes": [(3, 3)],
            "input_dtypes": [jnp.bool_],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reduce_or_all_axes",
            "callable": lambda x: jnp.any(x),
            "input_shapes": [(2, 3, 4)],
            "input_dtypes": [jnp.bool_],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reduce_or_keepdims",
            "callable": lambda x: jnp.any(x, axis=1, keepdims=True),
            "input_shapes": [(3, 4)],
            "input_dtypes": [jnp.bool_],
            "use_onnx_ir": True,
        },
    ],
)
class ReduceOrPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.reduce_or`` via ReduceMax + Cast."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_boolean_reduction(ctx, eqn, mode="reduce_or")
