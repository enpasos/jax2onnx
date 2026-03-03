# jax2onnx/plugins/jax/numpy/cosh.py

from __future__ import annotations

from typing import Any, ClassVar, Final

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


_COSH_PRIM: Final = make_jnp_primitive("jax.numpy.cosh")


@register_primitive(
    jaxpr_primitive=_COSH_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cosh.html",
    onnx=[
        {
            "component": "Cosh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cosh.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="cosh",
    testcases=[
        {
            "testcase": "jnp_cosh_basic",
            "callable": lambda x: jnp.cosh(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Cosh:3", "Neg:3 -> Exp:3 -> Add:3 -> Mul:3"],
                mode="any",
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_cosh_int_promote",
            "callable": lambda x: jnp.cosh(x),
            "input_values": [np.array([0, 1, 2], dtype=np.int32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Cosh:3", "Cast:3 -> Neg:3 -> Exp:3 -> Add:3 -> Mul:3"],
                mode="any",
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "cosh_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.cosh)(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "cosh_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.cosh(y)))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpCoshPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _COSH_PRIM
    _FUNC_NAME: ClassVar[str] = "cosh"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpCoshPlugin._PRIM,
            JnpCoshPlugin._FUNC_NAME,
            x,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("jnp_cosh_in"))
        out_spec = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("jnp_cosh_out"),
        )
        op_input = cast_input_to_output_dtype(
            ctx,
            x_var,
            out_var,
            x_val,
            output_hint="jnp_cosh",
        )

        out_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(out_var, "aval", None), "dtype", np.float32)
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("jnp_cosh_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("jnp_cosh_out")

        if np.issubdtype(out_dtype, np.floating):
            neg_x = ctx.builder.Neg(
                op_input,
                _outputs=[ctx.fresh_name("jnp_cosh_neg")],
            )
            neg_x.type = op_input.type
            neg_x.shape = op_input.shape

            exp_pos = ctx.builder.Exp(
                op_input,
                _outputs=[ctx.fresh_name("jnp_cosh_exp_pos")],
            )
            exp_pos.type = op_input.type
            exp_pos.shape = op_input.shape

            exp_neg = ctx.builder.Exp(
                neg_x,
                _outputs=[ctx.fresh_name("jnp_cosh_exp_neg")],
            )
            exp_neg.type = op_input.type
            exp_neg.shape = op_input.shape

            summed = ctx.builder.Add(
                exp_pos,
                exp_neg,
                _outputs=[ctx.fresh_name("jnp_cosh_sum")],
            )
            summed.type = op_input.type
            summed.shape = op_input.shape

            half = ctx.bind_const_for_var(object(), np.asarray(0.5, dtype=out_dtype))
            result = ctx.builder.Mul(summed, half, _outputs=[desired_name])
        else:
            result = ctx.builder.Cosh(op_input, _outputs=[desired_name])

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


@JnpCoshPlugin._PRIM.def_impl
def _cosh_impl(x: object) -> object:
    orig = get_orig_impl(JnpCoshPlugin._PRIM, JnpCoshPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpCoshPlugin._PRIM, _cosh_impl)
register_unary_elementwise_batch_rule(JnpCoshPlugin._PRIM)
