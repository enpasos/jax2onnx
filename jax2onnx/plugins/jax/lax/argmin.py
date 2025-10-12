# jax2onnx/plugins/jax/lax/argmin.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._arg_utils import lower_arg_reduction
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.argmin_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmin.html",
    onnx=[
        {
            "component": "ArgMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMin.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="argmin",
    testcases=[
        {
            "testcase": "argmin_test1",
            "callable": lambda x: jax.lax.argmin(x, axis=0, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                ["ArgMin:3 -> Cast:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "argmin_test2",
            "callable": lambda x: jax.lax.argmin(x, axis=1, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                ["ArgMin:3 -> Cast:3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ArgMinPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.argmin`` to ONNX ``ArgMin`` with optional index casts."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_arg_reduction(ctx, eqn, op_name="ArgMin", name_prefix="argmin")
