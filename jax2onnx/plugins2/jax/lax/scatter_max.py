from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jax2onnx.plugins2.jax.lax.scatter_utils import lower_scatter_common
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.scatter_max_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter_max.html",
    onnx=[
        {
            "component": "ScatterND(reduction='max')",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        }
    ],
    since="v0.7.5",
    context="primitives2.lax",
    component="scatter_max",
    testcases=[
        {
            "testcase": "scatter_max_simple",
            "callable": lambda x: x.at[jnp.array([1], dtype=jnp.int32)].max(
                jnp.array([3.0], dtype=x.dtype)
            ),
            "input_shapes": [(3,)],
            "use_onnx_ir": True,
        }
    ],
)
class ScatterMaxPlugin(PrimitiveLeafPlugin):
    """IR-first lowering for ``lax.scatter_max`` (element-wise variant)."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_scatter_common(ctx, eqn, reduction="max")
