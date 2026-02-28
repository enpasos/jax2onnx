# jax2onnx/plugins/jax/numpy/sign.py

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


_SIGN_PRIM: Final = make_jnp_primitive("jax.numpy.sign")


@register_primitive(
    jaxpr_primitive=_SIGN_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sign.html",
    onnx=[
        {
            "component": "Sign",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sign.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="sign",
    testcases=[
        {
            "testcase": "jnp_sign_basic",
            "callable": lambda x: jnp.sign(x),
            "input_values": [np.array([-2.0, 0.0, 3.5], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Sign:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_sign_int",
            "callable": lambda x: jnp.sign(x),
            "input_values": [np.array([-2, 0, 3], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Sign:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "sign_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.sign)(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "sign_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.sign(y)))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpSignPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SIGN_PRIM
    _FUNC_NAME: ClassVar[str] = "sign"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpSignPlugin._PRIM,
            JnpSignPlugin._FUNC_NAME,
            x,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_unary_elementwise_with_optional_cast(
            ctx,
            eqn,
            op_name="Sign",
            input_hint="jnp_sign_in",
            output_hint="jnp_sign_out",
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return jnp_binding_specs(cls._PRIM, cls._FUNC_NAME)


@JnpSignPlugin._PRIM.def_impl
def _sign_impl(x: object) -> object:
    orig = get_orig_impl(JnpSignPlugin._PRIM, JnpSignPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpSignPlugin._PRIM, _sign_impl)
register_unary_elementwise_batch_rule(JnpSignPlugin._PRIM)
