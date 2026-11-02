# jax2onnx/plugins/jax/core/call_hi_primitive.py

from __future__ import annotations

import jax.numpy as jnp

from jax2onnx._compat.jax import JaxprEqn, call_hi_primitive_jaxpr
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.core.jit import JitPlugin
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive="call_hi_primitive",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nonzero.html",
    onnx=[],
    since="0.18.0",
    context="primitives.core",
    component="call_hi_primitive",
    testcases=[
        {
            "testcase": "call_hi_nonzero",
            "callable": lambda x: jnp.nonzero(x, size=3, fill_value=-1),
            "input_shapes": [(4,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(["CumSum:4"], no_unused_inputs=True),
        },
    ],
)
class CallHiPrimitivePlugin(PrimitiveLeafPlugin):
    """Inline JAX's pure high-level array decompositions through normal plugins."""

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        closed = call_hi_primitive_jaxpr(eqn)
        JitPlugin().lower(ctx, eqn.replace(params={"jaxpr": closed}))
