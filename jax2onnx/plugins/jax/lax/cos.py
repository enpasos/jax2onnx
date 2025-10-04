# jax2onnx/plugins/jax/lax/cos.py

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape

if TYPE_CHECKING:
    pass


@register_primitive(
    jaxpr_primitive=jax.lax.cos_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cos.html",
    onnx=[
        {
            "component": "Cos",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cos.html",
        }
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="cos",
    testcases=[
        {
            "testcase": "cos",
            "callable": lambda x: jax.lax.cos(x),
            "input_shapes": [(3,)],
        }
    ],
)
class CosPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("cos_in"))

        x_dtype = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        if x_dtype == np.float64:
            # ONNX runtime lacks a double kernel for Cos; use sin(x + pi/2) instead.
            pi_over_two = ctx.bind_const_for_var(
                object(), np.asarray(np.pi / 2, dtype=np.float64)
            )

            shifted = ir.Value(
                name=ctx.fresh_name("cos_shifted"),
                type=x_val.type,
                shape=x_val.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Add",
                    domain="",
                    inputs=[x_val, pi_over_two],
                    outputs=[shifted],
                    name=ctx.fresh_name("Add"),
                )
            )

            sin_out = ir.Value(
                name=ctx.fresh_name("cos_via_sin"),
                type=x_val.type,
                shape=x_val.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Sin",
                    domain="",
                    inputs=[shifted],
                    outputs=[sin_out],
                    name=ctx.fresh_name("Sin"),
                )
            )

            _stamp_type_and_shape(sin_out, getattr(x_var.aval, "shape", ()))
            ctx.bind_value_for_var(out_var, sin_out)
        else:
            out_val = ctx.get_value_for_var(
                out_var, name_hint=ctx.fresh_name("cos_out")
            )
            node = ir.Node(
                op_type="Cos",
                domain="",
                inputs=[x_val],
                outputs=[out_val],
                name=ctx.fresh_name("Cos"),
            )
            ctx.add_node(node)
