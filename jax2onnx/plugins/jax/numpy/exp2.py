# jax2onnx/plugins/jax/numpy/exp2.py

from __future__ import annotations

from typing import Any, ClassVar, Final, cast

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


_EXP2_PRIM: Final = make_jnp_primitive("jax.numpy.exp2")


@register_primitive(
    jaxpr_primitive=_EXP2_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.exp2.html",
    onnx=[{"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"}],
    since="0.12.2",
    context="primitives.jnp",
    component="exp2",
    testcases=[
        {
            "testcase": "jnp_exp2_basic",
            "callable": lambda x: jnp.exp2(x),
            "input_values": [np.array([-2.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Pow:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_exp2_int_promote",
            "callable": lambda x: jnp.exp2(x),
            "input_values": [np.array([0, 1, 2], dtype=np.int32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Pow:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "exp2_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.exp2)(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "exp2_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.exp2(y)))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpExp2Plugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _EXP2_PRIM
    _FUNC_NAME: ClassVar[str] = "exp2"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpExp2Plugin._PRIM, JnpExp2Plugin._FUNC_NAME, x
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (y_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("jnp_exp2_in"))
        out_spec = ctx.get_value_for_var(
            y_var, name_hint=ctx.fresh_name("jnp_exp2_out")
        )

        op_input = cast_input_to_output_dtype(
            ctx,
            x_var,
            y_var,
            x_val,
            output_hint="jnp_exp2",
        )
        out_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(y_var, "aval", None), "dtype", np.float32)
        )
        two_scalar = ctx.bind_const_for_var(object(), np.asarray(2.0, dtype=out_dtype))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("jnp_exp2_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("jnp_exp2_out")

        result = ctx.builder.Pow(two_scalar, op_input, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(y_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return cast(
            list[AssignSpec | MonkeyPatchSpec],
            jnp_binding_specs(cls._PRIM, cls._FUNC_NAME),
        )


@JnpExp2Plugin._PRIM.def_impl
def _exp2_impl(x: object) -> object:
    orig = get_orig_impl(JnpExp2Plugin._PRIM, JnpExp2Plugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpExp2Plugin._PRIM, _exp2_impl)
register_unary_elementwise_batch_rule(JnpExp2Plugin._PRIM)
