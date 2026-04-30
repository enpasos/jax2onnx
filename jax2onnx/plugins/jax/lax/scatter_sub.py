# jax2onnx/plugins/jax/lax/scatter_sub.py

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.scatter_utils import lower_scatter_common
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive=jax.lax.scatter_sub_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter_sub.html",
    onnx=[
        {"component": "Neg", "doc": "https://onnx.ai/onnx/operators/onnx__Neg.html"},
        {
            "component": "ScatterND(reduction='add')",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="scatter_sub",
    testcases=[
        {
            "testcase": "scatter_sub_simple_1d",
            "callable": lambda operand, indices, updates: jax.lax.scatter_sub(
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
            "post_check_onnx_graph": EG(
                ["Neg:2 -> ScatterND:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_sub_window_2d_operand_1d_indices",
            "callable": lambda operand, indices, updates: jax.lax.scatter_sub(
                operand,
                indices,
                updates,
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1,),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_values": [
                jnp.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=jnp.float32),
                jnp.array([[1]], dtype=jnp.int32),
                jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32),
            ],
            "post_check_onnx_graph": EG(
                ["Neg:1x3 -> ScatterND:2x3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ScatterSubPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.scatter_sub`` via ``Neg`` + ``ScatterND(reduction='add')``."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        updates_var = eqn.invars[2]
        updates_val = ctx.get_value_for_var(
            updates_var,
            name_hint=ctx.fresh_name("scatter_sub_updates"),
        )
        neg_updates = cast(
            ir.Value,
            ctx.builder.Neg(
                updates_val,
                _outputs=[ctx.fresh_name("scatter_sub_neg_updates")],
            ),
        )
        if getattr(updates_val, "type", None) is not None:
            neg_updates.type = updates_val.type
        if getattr(updates_val, "shape", None) is not None:
            neg_updates.shape = updates_val.shape

        lower_scatter_common(
            ctx,
            eqn,
            reduction="add",
            updates_override=neg_updates,
        )
