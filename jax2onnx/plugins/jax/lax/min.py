# jax2onnx/plugins/jax/lax/min.py

from typing import TYPE_CHECKING

import jax
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


@register_primitive(
    jaxpr_primitive=jax.lax.min_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.min.html",
    onnx=[
        {
            "component": "Min",
            "doc": "https://onnx.ai/onnx/operators/onnx__Min.html",
        }
    ],
    since="v0.1.0",
    context="primitives.lax",
    component="min",
    testcases=[
        {
            "testcase": "min_test1",
            "callable": lambda x1, x2: jax.lax.min(x1, x2),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Min:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class MinPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("min_lhs"))
        prefer_dtype = np.dtype(getattr(lhs_var.aval, "dtype", np.float32))
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("min_rhs"),
            prefer_np_dtype=prefer_dtype,
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("min_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("min_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("min_out")

        result = ctx.builder.Min(lhs_val, rhs_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
