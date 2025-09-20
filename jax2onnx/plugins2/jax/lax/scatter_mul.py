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
    since="v0.1.0",
    context="primitives2.lax",
    component="scatter_mul",
    testcases=[
        {
            "testcase": "scatter_mul_simple",
            "callable": lambda x: x.at[jnp.array([2], dtype=jnp.int32)].multiply(
                jnp.array([2.0], dtype=x.dtype)
            ),
            "input_shapes": [(5,)],
            "use_onnx_ir": True,
        }
    ],
)
class ScatterMulPlugin(PrimitiveLeafPlugin):
    """IR-first lowering for ``lax.scatter_mul`` (element-wise variant)."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_scatter_common(ctx, eqn, reduction="mul")

