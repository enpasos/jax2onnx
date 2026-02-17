# jax2onnx/plugins/jax/lax/xor.py

from typing import Any

import jax
import numpy as np
from jax import core

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.xor_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.bitwise_xor.html",
    onnx=[
        {"component": "Xor", "doc": "https://onnx.ai/onnx/operators/onnx__Xor.html"},
        {
            "component": "BitwiseXor",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitwiseXor.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="xor",
    testcases=[
        {
            "testcase": "xor_bool_vec",
            "callable": lambda x, y: jax.lax.bitwise_xor(x, y),
            "input_values": [
                np.array([True, False, True, False], dtype=np.bool_),
                np.array([False, True, True, False], dtype=np.bool_),
            ],
            "post_check_onnx_graph": EG(
                ["Xor:4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "xor_int_vec",
            "callable": lambda x, y: jax.lax.bitwise_xor(x, y),
            "input_values": [
                np.array([1, 2, 3, 4], dtype=np.int32),
                np.array([4, 3, 2, 1], dtype=np.int32),
            ],
            "post_check_onnx_graph": EG(
                ["BitwiseXor:4"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class XorPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.bitwise_xor`` and boolean ``xor``."""

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        prefer_dtype = np.dtype(getattr(lhs_var.aval, "dtype", np.bool_))

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("xor_lhs"))
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("xor_rhs"),
            prefer_np_dtype=prefer_dtype,
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("xor_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("xor_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("xor_out")

        if np.issubdtype(prefer_dtype, np.bool_):
            result = ctx.builder.Xor(lhs_val, rhs_val, _outputs=[desired_name])
        else:
            result = ctx.builder.BitwiseXor(lhs_val, rhs_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
