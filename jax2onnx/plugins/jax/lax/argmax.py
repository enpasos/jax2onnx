# jax2onnx/plugins/jax/lax/argmax.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._arg_utils import lower_arg_reduction
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.argmax_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmax.html",
    onnx=[
        {
            "component": "ArgMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMax.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="argmax",
    testcases=[
        {
            "testcase": "argmax_float_axis0",
            "callable": lambda x: jax.lax.argmax(x, axis=0, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
            "input_dtypes": [jnp.float32],
            "post_check_onnx_graph": EG(
                ["ArgMax:3 -> Cast:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "argmax_float_axis1",
            "callable": lambda x: jax.lax.argmax(x, axis=1, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
            "input_dtypes": [jnp.float32],
            "post_check_onnx_graph": EG(
                ["ArgMax:3 -> Cast:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "argmax_boolean_input_axis0_specific_values",
            "callable": lambda x: jax.lax.argmax(x, axis=0, index_dtype=jnp.int32),
            "input_values": [
                np.array(
                    [[False, True, False], [True, False, True], [False, False, False]],
                    dtype=np.bool_,
                )
            ],
            "post_check_onnx_graph": EG(
                ["Cast:3x3 -> ArgMax:3 -> Cast:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "argmax_boolean_input_axis1_specific_values",
            "callable": lambda x: jax.lax.argmax(x, axis=1, index_dtype=jnp.int32),
            "input_values": [
                np.array(
                    [[False, True, False], [True, False, True], [False, True, True]],
                    dtype=np.bool_,
                )
            ],
            "post_check_onnx_graph": EG(
                ["Cast:3x3 -> ArgMax:3 -> Cast:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "argmax_boolean_random_input_axis0",
            "callable": lambda x: jax.lax.argmax(x, axis=0, index_dtype=jnp.int32),
            "input_shapes": [(4, 5)],
            "input_dtypes": [np.bool_],
            "post_check_onnx_graph": EG(
                ["Cast:4x5 -> ArgMax:5 -> Cast:5"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ArgMaxPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.argmax`` to ONNX ``ArgMax`` with optional dtype casts."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_arg_reduction(ctx, eqn, op_name="ArgMax", name_prefix="argmax")
