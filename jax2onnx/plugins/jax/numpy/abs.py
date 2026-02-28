# jax2onnx/plugins/jax/numpy/abs.py

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


_ABS_PRIM: Final = make_jnp_primitive("jax.numpy.abs")


@register_primitive(
    jaxpr_primitive=_ABS_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.abs.html",
    onnx=[
        {
            "component": "Abs",
            "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="abs",
    testcases=[
        {
            "testcase": "jnp_abs_basic",
            "callable": lambda x: jnp.abs(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Abs:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_abs_int",
            "callable": lambda x: jnp.abs(x),
            "input_values": [np.array([-3, 0, 4], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Abs:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "abs_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.abs)(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "abs_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.abs(y)))(x),
            "input_values": [np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpAbsPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _ABS_PRIM
    _FUNC_NAME: ClassVar[str] = "abs"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpAbsPlugin._PRIM, JnpAbsPlugin._FUNC_NAME, x
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_unary_elementwise_with_optional_cast(
            ctx,
            eqn,
            op_name="Abs",
            input_hint="jnp_abs_in",
            output_hint="jnp_abs_out",
            cast_input_to_output_dtype=False,
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return jnp_binding_specs(cls._PRIM, cls._FUNC_NAME)


@JnpAbsPlugin._PRIM.def_impl
def _abs_impl(x: object) -> object:
    orig = get_orig_impl(JnpAbsPlugin._PRIM, JnpAbsPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpAbsPlugin._PRIM, _abs_impl)
register_unary_elementwise_batch_rule(JnpAbsPlugin._PRIM)
