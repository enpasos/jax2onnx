# jax2onnx/plugins/jax/numpy/acosh.py

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


_ACOSH_PRIM: Final = make_jnp_primitive("jax.numpy.acosh")


@register_primitive(
    jaxpr_primitive=_ACOSH_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.acosh.html",
    onnx=[
        {
            "component": "Acosh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Acosh.html",
        }
    ],
    since="0.12.7",
    context="primitives.jnp",
    component="acosh",
    testcases=[
        {
            "testcase": "jnp_acosh_basic",
            "run_only_f32_variant": True,
            "callable": lambda x: jnp.acosh(x),
            "input_values": [np.array([1.0, 2.0, 10.0], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Acosh:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_acosh_int_promote",
            "run_only_f32_variant": True,
            "callable": lambda x: jnp.acosh(x),
            "input_values": [np.array([1, 2, 3], dtype=np.int32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Acosh:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "acosh_vmap_batching",
            "run_only_f32_variant": True,
            "callable": lambda x: jax.vmap(jnp.acosh)(x),
            "input_values": [
                np.array(
                    [[1.0, 1.5, 2.0, 3.0], [1.2, 2.5, 4.0, 8.0]],
                    dtype=np.float32,
                )
            ],
        },
        {
            "testcase": "acosh_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(
                lambda y: jnp.sum(jnp.acosh(jnp.abs(y) + 1.1))
            )(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpAcoshPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _ACOSH_PRIM
    _FUNC_NAME: ClassVar[str] = "acosh"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpAcoshPlugin._PRIM, JnpAcoshPlugin._FUNC_NAME, x
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_unary_elementwise_with_optional_cast(
            ctx,
            eqn,
            op_name="Acosh",
            input_hint="jnp_acosh_in",
            output_hint="jnp_acosh_out",
            cast_input_to_output_dtype=True,
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return cast(
            list[AssignSpec | MonkeyPatchSpec],
            jnp_binding_specs(cls._PRIM, cls._FUNC_NAME),
        )


@JnpAcoshPlugin._PRIM.def_impl
def _acosh_impl(x: object) -> object:
    orig = get_orig_impl(JnpAcoshPlugin._PRIM, JnpAcoshPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpAcoshPlugin._PRIM, _acosh_impl)
register_unary_elementwise_batch_rule(JnpAcoshPlugin._PRIM)
