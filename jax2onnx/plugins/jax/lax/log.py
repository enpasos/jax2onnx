# jax2onnx/plugins/jax/lax/log.py

from typing import Any

from jax import core
import jax

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.log_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.log.html",
    onnx=[
        {
            "component": "Log",
            "doc": "https://onnx.ai/onnx/operators/onnx__Log.html",
        },
        {
            "component": "ReduceLogSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceLogSum.html",
        },
        {
            "component": "ReduceLogSumExp",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceLogSumExp.html",
        },
    ],
    since="0.2.0",
    context="primitives.lax",
    component="log",
    testcases=[
        {
            "testcase": "log",
            "callable": lambda x: jax.lax.log(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Log:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "log_of_reduce_sum_axis1",
            "callable": lambda x: jax.numpy.log(jax.numpy.sum(x, axis=1)),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ReduceLogSum:2",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "log_of_reduce_sum_exp_axis1",
            "callable": lambda x: jax.numpy.log(
                jax.numpy.sum(jax.numpy.exp(x), axis=1)
            ),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ReduceLogSumExp:2",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class LogPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("log_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("log_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("log_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("log_out")

        reduce_getter = getattr(x_val, "producer", lambda: None)
        reduce_node = reduce_getter() if callable(reduce_getter) else None
        if getattr(reduce_node, "op_type", "") == "ReduceSum":
            reduce_inputs = list(getattr(reduce_node, "inputs", ()))
            keepdims_attr = reduce_node.attributes.get("keepdims")
            keepdims = int(
                getattr(keepdims_attr, "value", keepdims_attr)
                if keepdims_attr is not None
                else 1
            )

            if reduce_inputs:
                reduce_data = reduce_inputs[0]
                reduce_axes = reduce_inputs[1:] if len(reduce_inputs) > 1 else []

                exp_getter = getattr(reduce_data, "producer", lambda: None)
                exp_node = exp_getter() if callable(exp_getter) else None
                if getattr(exp_node, "op_type", "") == "Exp" and exp_node.inputs:
                    alias_inputs = [exp_node.inputs[0], *reduce_axes]
                    result = ctx.builder.ReduceLogSumExp(
                        *alias_inputs,
                        keepdims=keepdims,
                        _outputs=[desired_name],
                    )
                else:
                    alias_inputs = [reduce_data, *reduce_axes]
                    result = ctx.builder.ReduceLogSum(
                        *alias_inputs,
                        keepdims=keepdims,
                        _outputs=[desired_name],
                    )
                if getattr(out_spec, "type", None) is not None:
                    result.type = out_spec.type
                if getattr(out_spec, "shape", None) is not None:
                    result.shape = out_spec.shape
                ctx.bind_value_for_var(out_var, result)
                return

        result = ctx.builder.Log(x_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
