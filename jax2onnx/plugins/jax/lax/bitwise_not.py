# jax2onnx/plugins/jax/lax/bitwise_not.py

from typing import Any

from jax import core
import jax
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.not_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.bitwise_not.html",
    onnx=[
        {
            "component": "BitwiseNot",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitwiseNot.html",
        },
        {
            "component": "Not",
            "doc": "https://onnx.ai/onnx/operators/onnx__Not.html",
        },
    ],
    since="v0.7.5",
    context="primitives.lax",
    component="bitwise_not",
    testcases=[
        {
            "testcase": "bitwise_not_bool",
            "callable": lambda x: jax.lax.bitwise_not(x),
            "input_values": [np.array(True, dtype=np.bool_)],
            "expected_output_dtypes": [np.bool_],
            "post_check_onnx_graph": EG(
                ["Not"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "bitwise_not_i32",
            "callable": lambda x: jax.lax.bitwise_not(x),
            "input_values": [np.array(7, dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["BitwiseNot"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class BitwiseNotPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(x: core.AbstractValue):
        return x

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("not_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("not_out"))

        x_dtype = np.dtype(getattr(x_var.aval, "dtype", np.bool_))
        op_type = "Not" if x_dtype.kind == "b" else "BitwiseNot"

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("not_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("not_out")

        builder_fn = getattr(ctx.builder, op_type)
        result = builder_fn(x_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
