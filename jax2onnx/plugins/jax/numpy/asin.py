# jax2onnx/plugins/jax/numpy/asin.py

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


_ASIN_PRIM: Final = make_jnp_primitive("jax.numpy.asin")


@register_primitive(
    jaxpr_primitive=_ASIN_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.asin.html",
    onnx=[
        {
            "component": "Asin",
            "doc": "https://onnx.ai/onnx/operators/onnx__Asin.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="asin",
    testcases=[
        {
            "testcase": "jnp_asin_basic",
            "run_only_f32_variant": True,
            "callable": lambda x: jnp.asin(x),
            "input_values": [np.array([-0.75, 0.0, 0.5], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Asin:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_asin_int_promote",
            "run_only_f32_variant": True,
            "callable": lambda x: jnp.asin(x),
            "input_values": [np.array([0, 1, 0], dtype=np.int32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Asin:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "asin_vmap_batching",
            "run_only_f32_variant": True,
            "callable": lambda x: jax.vmap(jnp.asin)(x),
            "input_values": [
                np.array(
                    [[-0.75, -0.25, 0.0, 0.5], [-0.5, 0.25, 0.5, 0.75]],
                    dtype=np.float32,
                )
            ],
        },
        {
            "testcase": "asin_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.asin(jnp.tanh(y))))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpAsinPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _ASIN_PRIM
    _FUNC_NAME: ClassVar[str] = "asin"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpAsinPlugin._PRIM, JnpAsinPlugin._FUNC_NAME, x
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_unary_elementwise_with_optional_cast(
            ctx,
            eqn,
            op_name="Asin",
            input_hint="jnp_asin_in",
            output_hint="jnp_asin_out",
            cast_input_to_output_dtype=True,
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpAsinPlugin._PRIM.def_impl
def _asin_impl(x: object) -> object:
    orig = get_orig_impl(JnpAsinPlugin._PRIM, JnpAsinPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpAsinPlugin._PRIM, _asin_impl)
register_unary_elementwise_batch_rule(JnpAsinPlugin._PRIM)
