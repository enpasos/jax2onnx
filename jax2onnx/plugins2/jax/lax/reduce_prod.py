from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins2.jax.lax._reduce_utils import lower_reduction
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_prod_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_prod.html",
    onnx=[
        {
            "component": "ReduceProd",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceProd.html",
        }
    ],
    since="v0.1.0",
    context="primitives2.lax",
    component="reduce_prod",
    testcases=[
        {
            "testcase": "reduce_prod_axis0",
            "callable": lambda x: jnp.prod(x, axis=0),
            "input_shapes": [(3, 3)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reduce_prod_all_axes",
            "callable": lambda x: jnp.prod(x),
            "input_shapes": [(2, 3, 4)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reduce_prod_keepdims",
            "callable": lambda x: jnp.prod(x, axis=1, keepdims=True),
            "input_shapes": [(3, 4)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reduce_prod_dtype_promotion",
            "callable": lambda x: jnp.prod(x, axis=(0, 1), dtype=jnp.float32),
            "input_shapes": [(2, 2)],
            "input_dtypes": [np.int32],
            "expected_output_dtypes": [np.float32],
            "use_onnx_ir": True,
        },
    ],
)
class ReduceProdPlugin(PrimitiveLeafPlugin):
    """IR-only lowering of ``lax.reduce_prod`` via ONNX ReduceProd."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_reduction(ctx, eqn, op_type="ReduceProd", allow_dtype_param=True)
