# jax2onnx/plugins/jax/numpy/sqrt.py

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
    cast_input_to_output_dtype,
    register_unary_elementwise_batch_rule,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SQRT_PRIM: Final = make_jnp_primitive("jax.numpy.sqrt")


@register_primitive(
    jaxpr_primitive=_SQRT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sqrt.html",
    onnx=[
        {
            "component": "ReduceL2",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceL2.html",
        },
        {
            "component": "Sqrt",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html",
        },
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="sqrt",
    testcases=[
        {
            "testcase": "jnp_sqrt_basic",
            "callable": lambda x: jnp.sqrt(x),
            "input_values": [np.array([1.0, 4.0, 9.0], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Sqrt:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_sqrt_int_promote",
            "callable": lambda x: jnp.sqrt(x),
            "input_values": [np.array([1, 4, 9], dtype=np.int32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Sqrt:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_sqrt_reduce_sum_square_axis1",
            "callable": lambda x: jnp.sqrt(jnp.sum(jnp.square(x), axis=1)),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ReduceL2:2",
                        "inputs": {1: {"const": 1.0}},
                    },
                    "ReduceSumSquare:2 -> Sqrt:2",
                    "Mul:2x3 -> ReduceSum:2 -> Sqrt:2",
                ],
                mode="any",
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "sqrt_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.sqrt)(x),
            "input_values": [np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32)],
        },
        {
            "testcase": "sqrt_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.sqrt(y)))(x),
            "input_values": [
                np.array([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]], dtype=np.float32)
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpSqrtPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SQRT_PRIM
    _FUNC_NAME: ClassVar[str] = "sqrt"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpSqrtPlugin._PRIM,
            JnpSqrtPlugin._FUNC_NAME,
            x,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("jnp_sqrt_in"))
        out_spec = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("jnp_sqrt_out"),
        )
        op_input = cast_input_to_output_dtype(
            ctx,
            x_var,
            out_var,
            x_val,
            output_hint="jnp_sqrt",
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("jnp_sqrt_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("jnp_sqrt_out")

        input_producer_getter = getattr(op_input, "producer", lambda: None)
        input_producer = (
            input_producer_getter() if callable(input_producer_getter) else None
        )
        if getattr(input_producer, "op_type", "") == "ReduceSumSquare":
            reduce_inputs = list(getattr(input_producer, "inputs", ()))
            if reduce_inputs:
                input_attributes = getattr(input_producer, "attributes", {})
                keepdims_attr = (
                    input_attributes.get("keepdims")
                    if hasattr(input_attributes, "get")
                    else None
                )
                keepdims = int(
                    getattr(keepdims_attr, "value", keepdims_attr)
                    if keepdims_attr is not None
                    else 1
                )
                result = ctx.builder.ReduceL2(
                    *reduce_inputs,
                    keepdims=keepdims,
                    _outputs=[desired_name],
                )
                if getattr(out_spec, "type", None) is not None:
                    result.type = out_spec.type
                if getattr(out_spec, "shape", None) is not None:
                    result.shape = out_spec.shape
                ctx.bind_value_for_var(out_var, result)
                return

        result = ctx.builder.Sqrt(op_input, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpSqrtPlugin._PRIM.def_impl
def _sqrt_impl(x: object) -> object:
    orig = get_orig_impl(JnpSqrtPlugin._PRIM, JnpSqrtPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpSqrtPlugin._PRIM, _sqrt_impl)
register_unary_elementwise_batch_rule(JnpSqrtPlugin._PRIM)
