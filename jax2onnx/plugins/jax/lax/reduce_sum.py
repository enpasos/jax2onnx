# jax2onnx/plugins/jax/lax/reduce_sum.py

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
    jaxpr_primitive=jax.lax.reduce_sum_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_sum.html",
    onnx=[
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="reduce_sum",
    testcases=[
        {
            "testcase": "reduce_sum",
            "callable": lambda x: jnp.sum(x, axis=None),
            "input_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                ["ReduceSum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_sum_allaxes",
            "callable": lambda x: jnp.sum(x, axis=None),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceSum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_sum_dtype",
            "callable": lambda x: jnp.sum(x, axis=None, dtype=jnp.float32),
            "input_values": [np.arange(6, dtype=np.float32).reshape(2, 3)],
            "post_check_onnx_graph": EG(
                ["ReduceSum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_sum_dtype_f64",
            "callable": lambda x: jnp.sum(x, axis=None, dtype=jnp.float64),
            "input_values": [np.arange(6, dtype=np.float64).reshape(2, 3)],
            "post_check_onnx_graph": EG(
                ["ReduceSum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_sum_keepdims",
            "callable": lambda x: jnp.sum(x, axis=(1,), keepdims=True),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceSum:3 -> Reshape:3x1 -> Expand:3x1"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ReduceSumPlugin(PrimitiveLeafPlugin):
    """IR-only lowering of ``lax.reduce_sum`` via ONNX ReduceSum."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_reduction(ctx, eqn, op_type="ReduceSum", allow_dtype_param=True)
