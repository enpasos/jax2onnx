# jax2onnx/plugins/jax/numpy/tanh.py

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


_TANH_PRIM: Final = make_jnp_primitive("jax.numpy.tanh")


@register_primitive(
    jaxpr_primitive=_TANH_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tanh.html",
    onnx=[
        {
            "component": "Tanh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="tanh",
    testcases=[
        {
            "testcase": "jnp_tanh_basic",
            "callable": lambda x: jnp.tanh(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Tanh:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_tanh_int_promote",
            "callable": lambda x: jnp.tanh(x),
            "input_values": [np.array([0, 1, 2], dtype=np.int32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Tanh:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "tanh_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.tanh)(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "tanh_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.tanh(y)))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpTanhPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _TANH_PRIM
    _FUNC_NAME: ClassVar[str] = "tanh"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpTanhPlugin._PRIM,
            JnpTanhPlugin._FUNC_NAME,
            x,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_unary_elementwise_with_optional_cast(
            ctx,
            eqn,
            op_name="Tanh",
            input_hint="jnp_tanh_in",
            output_hint="jnp_tanh_out",
            cast_input_to_output_dtype=True,
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return jnp_binding_specs(cls._PRIM, cls._FUNC_NAME)


@JnpTanhPlugin._PRIM.def_impl
def _tanh_impl(x: object) -> object:
    orig = get_orig_impl(JnpTanhPlugin._PRIM, JnpTanhPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpTanhPlugin._PRIM, _tanh_impl)
register_unary_elementwise_batch_rule(JnpTanhPlugin._PRIM)
