from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jax2onnx.plugins2.jax.lax._reduce_utils import lower_boolean_reduction
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_xor_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_xor.html",
    onnx=[
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
        {
            "component": "Mod",
            "doc": "https://onnx.ai/onnx/operators/onnx__Mod.html",
        },
    ],
    since="v0.6.1",
    context="primitives2.lax",
    component="reduce_xor",
    testcases=[
        {
            "testcase": "reduce_xor_axis0",
            "callable": lambda x: jnp.logical_xor.reduce(x, axis=0),
            "input_shapes": [(3, 3)],
            "input_dtypes": [jnp.bool_],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reduce_xor_all_axes",
            "callable": lambda x: jnp.logical_xor.reduce(x),
            "input_shapes": [(2, 3, 4)],
            "input_dtypes": [jnp.bool_],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reduce_xor_keepdims",
            "callable": lambda x: jnp.logical_xor.reduce(x, axis=1, keepdims=True),
            "input_shapes": [(3, 4)],
            "input_dtypes": [jnp.bool_],
            "use_onnx_ir": True,
        },
    ],
)
class ReduceXorPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.reduce_xor`` using parity sum modulo 2."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_boolean_reduction(ctx, eqn, mode="reduce_xor")
