# jax2onnx/plugins/jax/numpy/atanh.py

from __future__ import annotations

from typing import ClassVar, Final, cast

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


_ATANH_PRIM: Final = make_jnp_primitive("jax.numpy.atanh")


@register_primitive(
    jaxpr_primitive=_ATANH_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atanh.html",
    onnx=[
        {
            "component": "Atanh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Atanh.html",
        }
    ],
    since="0.12.7",
    context="primitives.jnp",
    component="atanh",
    testcases=[
        {
            "testcase": "jnp_atanh_basic",
            "run_only_f32_variant": True,
            "callable": lambda x: jnp.atanh(x),
            "input_values": [np.array([-0.75, 0.0, 0.5], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Atanh:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_atanh_int_promote",
            "run_only_f32_variant": True,
            "callable": lambda x: jnp.atanh(x),
            "input_values": [np.array([0, 0, 0], dtype=np.int32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Atanh:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "atanh_vmap_batching",
            "run_only_f32_variant": True,
            "callable": lambda x: jax.vmap(jnp.atanh)(x),
            "input_values": [
                np.array(
                    [[-0.75, -0.25, 0.0, 0.5], [-0.5, 0.25, 0.5, 0.75]],
                    dtype=np.float32,
                )
            ],
        },
        {
            "testcase": "atanh_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(
                lambda y: jnp.sum(jnp.atanh(0.99 * jnp.tanh(y)))
            )(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpAtanhPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _ATANH_PRIM
    _FUNC_NAME: ClassVar[str] = "atanh"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpAtanhPlugin._PRIM, JnpAtanhPlugin._FUNC_NAME, x
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_unary_elementwise_with_optional_cast(
            ctx,
            eqn,
            op_name="Atanh",
            input_hint="jnp_atanh_in",
            output_hint="jnp_atanh_out",
            cast_input_to_output_dtype=True,
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return cast(
            list[AssignSpec | MonkeyPatchSpec],
            jnp_binding_specs(cls._PRIM, cls._FUNC_NAME),
        )


@JnpAtanhPlugin._PRIM.def_impl
def _atanh_impl(x: object) -> object:
    orig = get_orig_impl(JnpAtanhPlugin._PRIM, JnpAtanhPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpAtanhPlugin._PRIM, _atanh_impl)
register_unary_elementwise_batch_rule(JnpAtanhPlugin._PRIM)
