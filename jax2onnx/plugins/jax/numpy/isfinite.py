# jax2onnx/plugins/jax/numpy/isfinite.py

from __future__ import annotations

from typing import ClassVar, Final

from jax import core
import jax.numpy as jnp
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.jax.numpy._unary_utils import (
    abstract_eval_via_orig_unary,
    register_unary_elementwise_batch_rule,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_ISFINITE_PRIM: Final = make_jnp_primitive("jax.numpy.isfinite")


@register_primitive(
    jaxpr_primitive=_ISFINITE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isfinite.html",
    onnx=[
        {
            "component": "IsInf",
            "doc": "https://onnx.ai/onnx/operators/onnx__IsInf.html",
        },
        {
            "component": "IsNaN",
            "doc": "https://onnx.ai/onnx/operators/onnx__IsNaN.html",
        },
        {"component": "Or", "doc": "https://onnx.ai/onnx/operators/onnx__Or.html"},
        {"component": "Not", "doc": "https://onnx.ai/onnx/operators/onnx__Not.html"},
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="isfinite",
    testcases=[
        {
            "testcase": "jnp_isfinite_basic",
            "callable": lambda x: jnp.isfinite(x),
            "input_values": [
                np.array([-np.inf, -1.0, 0.0, np.inf, np.nan], dtype=np.float32)
            ],
            "expected_output_dtypes": [np.bool_],
            "post_check_onnx_graph": EG(
                ["IsInf:5 -> Or:5 -> Not:5", "IsNaN:5 -> Or:5 -> Not:5"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class JnpIsFinitePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _ISFINITE_PRIM
    _FUNC_NAME: ClassVar[str] = "isfinite"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpIsFinitePlugin._PRIM,
            JnpIsFinitePlugin._FUNC_NAME,
            x,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(
            x_var,
            name_hint=ctx.fresh_name("jnp_isfinite_in"),
        )
        out_spec = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("jnp_isfinite_out"),
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "jnp_isfinite_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("jnp_isfinite_out")

        is_inf = ctx.builder.IsInf(
            x_val,
            _outputs=[ctx.fresh_name("jnp_isfinite_isinf")],
        )
        is_nan = ctx.builder.IsNaN(
            x_val,
            _outputs=[ctx.fresh_name("jnp_isfinite_isnan")],
        )
        any_non_finite = ctx.builder.Or(
            is_inf,
            is_nan,
            _outputs=[ctx.fresh_name("jnp_isfinite_or")],
        )
        result = ctx.builder.Not(any_non_finite, _outputs=[desired_name])

        if getattr(out_spec, "type", None) is not None:
            is_inf.type = out_spec.type
            is_nan.type = out_spec.type
            any_non_finite.type = out_spec.type
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            is_inf.shape = out_spec.shape
            is_nan.shape = out_spec.shape
            any_non_finite.shape = out_spec.shape
            result.shape = out_spec.shape

        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return jnp_binding_specs(cls._PRIM, cls._FUNC_NAME)


@JnpIsFinitePlugin._PRIM.def_impl
def _isfinite_impl(x: object) -> object:
    orig = get_orig_impl(JnpIsFinitePlugin._PRIM, JnpIsFinitePlugin._FUNC_NAME)
    return orig(x)


register_unary_elementwise_batch_rule(JnpIsFinitePlugin._PRIM)
