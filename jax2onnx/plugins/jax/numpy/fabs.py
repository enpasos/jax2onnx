# jax2onnx/plugins/jax/numpy/fabs.py

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


_FABS_PRIM: Final = make_jnp_primitive("jax.numpy.fabs")


@register_primitive(
    jaxpr_primitive=_FABS_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fabs.html",
    onnx=[
        {
            "component": "Abs",
            "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="fabs",
    testcases=[
        {
            "testcase": "jnp_fabs_basic",
            "callable": lambda x: jnp.fabs(x),
            "input_values": [np.array([-2.0, 0.0, 3.5], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Abs:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_fabs_int_promote",
            "callable": lambda x: jnp.fabs(x),
            "input_values": [np.array([-2, 0, 3], dtype=np.int32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Abs:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "fabs_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.fabs)(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "fabs_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.fabs(y)))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpFabsPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _FABS_PRIM
    _FUNC_NAME: ClassVar[str] = "fabs"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpFabsPlugin._PRIM, JnpFabsPlugin._FUNC_NAME, x
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_unary_elementwise_with_optional_cast(
            ctx,
            eqn,
            op_name="Abs",
            input_hint="jnp_fabs_in",
            output_hint="jnp_fabs_out",
            cast_input_to_output_dtype=True,
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return cast(
            list[AssignSpec | MonkeyPatchSpec],
            jnp_binding_specs(cls._PRIM, cls._FUNC_NAME),
        )


@JnpFabsPlugin._PRIM.def_impl
def _fabs_impl(x: object) -> object:
    orig = get_orig_impl(JnpFabsPlugin._PRIM, JnpFabsPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpFabsPlugin._PRIM, _fabs_impl)
register_unary_elementwise_batch_rule(JnpFabsPlugin._PRIM)
