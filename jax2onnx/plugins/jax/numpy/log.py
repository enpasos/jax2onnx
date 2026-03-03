# jax2onnx/plugins/jax/numpy/log.py

from __future__ import annotations

from typing import ClassVar, Final

import jax
from jax import core
import jax.numpy as jnp
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.jax.numpy._unary_utils import (
    abstract_eval_via_orig_unary,
    lower_unary_elementwise_with_optional_cast,
    register_unary_elementwise_batch_rule,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_LOG_PRIM: Final = make_jnp_primitive("jax.numpy.log")


@register_primitive(
    jaxpr_primitive=_LOG_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.log.html",
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
    since="0.12.2",
    context="primitives.jnp",
    component="log",
    testcases=[
        {
            "testcase": "jnp_log_basic",
            "callable": lambda x: jnp.log(x),
            "input_values": [np.array([1.0, 2.0, 4.0], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Log:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_log_int_promote",
            "callable": lambda x: jnp.log(x),
            "input_values": [np.array([1, 2, 4], dtype=np.int32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Log:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "log_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.log)(x),
            "input_values": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
        },
        {
            "testcase": "log_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.log(y)))(x),
            "input_values": [
                np.array([[1.0, 1.5, 2.0], [2.5, 3.0, 4.0]], dtype=np.float32)
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLogPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _LOG_PRIM
    _FUNC_NAME: ClassVar[str] = "log"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpLogPlugin._PRIM, JnpLogPlugin._FUNC_NAME, x
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("jnp_log_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_log_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("jnp_log_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("jnp_log_out")

        reduce_getter = getattr(x_val, "producer", lambda: None)
        reduce_node = reduce_getter() if callable(reduce_getter) else None
        if getattr(reduce_node, "op_type", "") == "ReduceSum":
            reduce_inputs = list(getattr(reduce_node, "inputs", ()))
            reduce_attributes = getattr(reduce_node, "attributes", {})
            keepdims_attr = (
                reduce_attributes.get("keepdims")
                if hasattr(reduce_attributes, "get")
                else None
            )
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
                exp_inputs = tuple(getattr(exp_node, "inputs", ()))
                if getattr(exp_node, "op_type", "") == "Exp" and exp_inputs:
                    alias_inputs = [exp_inputs[0], *reduce_axes]
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

        lower_unary_elementwise_with_optional_cast(
            ctx,
            eqn,
            op_name="Log",
            input_hint="jnp_log_in",
            output_hint="jnp_log_out",
            cast_input_to_output_dtype=True,
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpLogPlugin._PRIM.def_impl
def _log_impl(x: object) -> object:
    orig = get_orig_impl(JnpLogPlugin._PRIM, JnpLogPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpLogPlugin._PRIM, _log_impl)
register_unary_elementwise_batch_rule(JnpLogPlugin._PRIM)
