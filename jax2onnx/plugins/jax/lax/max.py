# jax2onnx/plugins/jax/lax/max.py

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


@register_primitive(
    jaxpr_primitive=jax.lax.max_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.max.html",
    onnx=[
        {
            "component": "Max",
            "doc": "https://onnx.ai/onnx/operators/onnx__Max.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="max",
    testcases=[
        {
            "testcase": "max",
            "callable": lambda x1, x2: jax.lax.max(x1, x2),
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class MaxPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("max_lhs"))
        prefer_dtype = np.dtype(getattr(lhs_var.aval, "dtype", np.float32))
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("max_rhs"),
            prefer_np_dtype=prefer_dtype,
        )
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("max_out"))
        if (
            callable(getattr(out_val, "producer", None))
            and out_val.producer() is not None
        ):
            out_val = ir.Value(
                name=ctx.fresh_name("max_out"),
                type=out_val.type,
                shape=out_val.shape,
            )
            ctx.builder._var2val[out_var] = out_val

        node = ir.Node(
            op_type="Max",
            domain="",
            inputs=[lhs_val, rhs_val],
            outputs=[out_val],
            name=ctx.fresh_name("Max"),
        )
        ctx.add_node(node)
