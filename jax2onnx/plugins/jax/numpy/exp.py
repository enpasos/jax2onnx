# jax2onnx/plugins/jax/numpy/exp.py

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


_EXP_PRIM: Final = make_jnp_primitive("jax.numpy.exp")


@register_primitive(
    jaxpr_primitive=_EXP_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.exp.html",
    onnx=[
        {
            "component": "Exp",
            "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="exp",
    testcases=[
        {
            "testcase": "jnp_exp_basic",
            "callable": lambda x: jnp.exp(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Exp:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_exp_int_promote",
            "callable": lambda x: jnp.exp(x),
            "input_values": [np.array([0, 1, 2], dtype=np.int32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Exp:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "exp_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.exp)(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "exp_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.exp(y)))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpExpPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _EXP_PRIM
    _FUNC_NAME: ClassVar[str] = "exp"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpExpPlugin._PRIM, JnpExpPlugin._FUNC_NAME, x
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_unary_elementwise_with_optional_cast(
            ctx,
            eqn,
            op_name="Exp",
            input_hint="jnp_exp_in",
            output_hint="jnp_exp_out",
            cast_input_to_output_dtype=True,
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return jnp_binding_specs(cls._PRIM, cls._FUNC_NAME)


@JnpExpPlugin._PRIM.def_impl
def _exp_impl(x: object) -> object:
    orig = get_orig_impl(JnpExpPlugin._PRIM, JnpExpPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpExpPlugin._PRIM, _exp_impl)
register_unary_elementwise_batch_rule(JnpExpPlugin._PRIM)
