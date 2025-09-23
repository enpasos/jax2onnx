from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jax2onnx.plugins2.jax.lax.scatter_utils import lower_scatter_common
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.scatter_mul_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter_mul.html",
    onnx=[
        {
            "component": "ScatterND(reduction='mul')",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        }
    ],
    since="v0.6.4",
    context="primitives2.lax",
    component="scatter_mul",
    testcases=[
        {
            "testcase": "scatter_mul_simple_1d",
            "callable": lambda operand, indices, updates: jax.lax.scatter_mul(
                operand,
                indices,
                updates,
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (2, 1), (2,)],
            "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
            "use_onnx_ir": True,
        },
        {
            "testcase": "scatter_mul_batch_updates_1d_operand",
            "callable": lambda operand, indices, updates: jax.lax.scatter_mul(
                operand,
                indices,
                updates,
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (2, 2, 1), (2, 2)],
            "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
            "use_onnx_ir": True,
        },
    ],
)
class ScatterMulPlugin(PrimitiveLeafPlugin):
    """IR-first lowering for ``lax.scatter_mul`` (element-wise variant)."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_scatter_common(ctx, eqn, reduction="mul")
