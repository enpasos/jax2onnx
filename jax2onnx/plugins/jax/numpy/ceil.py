# jax2onnx/plugins/jax/numpy/ceil.py

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


_CEIL_PRIM: Final = make_jnp_primitive("jax.numpy.ceil")


@register_primitive(
    jaxpr_primitive=_CEIL_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ceil.html",
    onnx=[
        {
            "component": "Ceil",
            "doc": "https://onnx.ai/onnx/operators/onnx__Ceil.html",
        },
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        },
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="ceil",
    testcases=[
        {
            "testcase": "jnp_ceil_basic",
            "callable": lambda x: jnp.ceil(x),
            "input_values": [np.array([-1.2, -0.1, 0.0, 2.3], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Ceil:4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_ceil_int_identity",
            "callable": lambda x: jnp.ceil(x),
            "input_values": [np.array([-3, 0, 5], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Identity:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "ceil_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.ceil)(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "ceil_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.ceil(y)))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpCeilPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _CEIL_PRIM
    _FUNC_NAME: ClassVar[str] = "ceil"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpCeilPlugin._PRIM,
            JnpCeilPlugin._FUNC_NAME,
            x,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_unary_elementwise_with_optional_cast(
            ctx,
            eqn,
            op_name="Ceil",
            input_hint="jnp_ceil_in",
            output_hint="jnp_ceil_out",
            identity_for_integral_input=True,
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return jnp_binding_specs(cls._PRIM, cls._FUNC_NAME)


@JnpCeilPlugin._PRIM.def_impl
def _ceil_impl(x: object) -> object:
    orig = get_orig_impl(JnpCeilPlugin._PRIM, JnpCeilPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpCeilPlugin._PRIM, _ceil_impl)
register_unary_elementwise_batch_rule(JnpCeilPlugin._PRIM)
