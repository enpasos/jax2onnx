# jax2onnx/plugins/jax/lax/reduce_prod.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._reduce_utils import lower_reduction
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_prod_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_prod.html",
    onnx=[
        {
            "component": "ReduceProd",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceProd.html",
        }
    ],
    since="v0.6.1",
    context="primitives.lax",
    component="reduce_prod",
    testcases=[
        {
            "testcase": "reduce_prod",
            "callable": lambda x: jnp.prod(x, axis=None),
            "input_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                ["ReduceProd"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_prod_allaxes",
            "callable": lambda x: jnp.prod(x, axis=None),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceProd"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_prod_dtype",
            "callable": lambda x: jnp.prod(x, axis=None, dtype=jnp.float32),
            "input_values": [np.ones((2, 3), dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["ReduceProd"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_prod_dtype_f64",
            "callable": lambda x: jnp.prod(x, axis=None, dtype=jnp.float64),
            "input_values": [np.ones((2, 3), dtype=np.float64)],
            "post_check_onnx_graph": EG(
                ["ReduceProd"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_prod_keepdims",
            "callable": lambda x: jnp.prod(x, axis=(1,), keepdims=True),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ReduceProd:3x1",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ReduceProdPlugin(PrimitiveLeafPlugin):
    """IR-only lowering of ``lax.reduce_prod`` via ONNX ReduceProd."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_reduction(ctx, eqn, op_type="ReduceProd", allow_dtype_param=True)
