# jax2onnx/plugins/jax/lax/reduce_xor.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._reduce_utils import lower_boolean_reduction
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


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
    context="primitives.lax",
    component="reduce_xor",
    testcases=[
        {
            "testcase": "reduce_xor_all_false",
            "callable": lambda x: jnp.logical_xor.reduce(x),
            "input_shapes": [(3, 4)],
            "input_dtypes": [jnp.bool_],
            "input_values": [np.zeros((3, 4), dtype=np.bool_)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Cast:3x4 -> ReduceSum:4 -> Mod:4 -> Equal:4",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_xor_one_true",
            "callable": lambda x: jnp.logical_xor.reduce(x),
            "input_shapes": [(2, 3)],
            "input_dtypes": [jnp.bool_],
            "input_values": [
                np.array([[True, False, False], [False, False, False]], dtype=np.bool_)
            ],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Cast:2x3 -> ReduceSum:3 -> Mod:3 -> Equal:3",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_xor_two_true",
            "callable": lambda x: jnp.logical_xor.reduce(x),
            "input_shapes": [(2, 3)],
            "input_dtypes": [jnp.bool_],
            "input_values": [
                np.array([[True, False, False], [False, True, False]], dtype=np.bool_)
            ],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Cast:2x3 -> ReduceSum:3 -> Mod:3 -> Equal:3",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_xor_keepdims",
            "callable": lambda x: jnp.logical_xor.reduce(x, axis=1, keepdims=True),
            "input_shapes": [(3, 4)],
            "input_dtypes": [jnp.bool_],
            "input_values": [
                np.array(
                    [
                        [False, True, False, False],
                        [True, False, False, False],
                        [False, False, True, False],
                    ],
                    dtype=np.bool_,
                )
            ],
            "post_check_onnx_graph": EG(
                [
                    "Cast:3x4 -> ReduceSum:3 -> Mod:3 -> Equal:3 -> Reshape:3x1 -> Expand:3x1"
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ReduceXorPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.reduce_xor`` using parity sum modulo 2."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_boolean_reduction(ctx, eqn, mode="reduce_xor")
