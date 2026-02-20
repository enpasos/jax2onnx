# jax2onnx/plugins/jax/nn/softsign.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
from jax.interpreters import ad
import jax.numpy as jnp
from jax.extend.core import Primitive
from numpy.typing import ArrayLike

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import (
    register_unary_elementwise_batch_rule,
)


_SOFTSIGN_PRIM: Final[Primitive] = Primitive("jax.nn.soft_sign")
_SOFTSIGN_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_SOFTSIGN_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.soft_sign.html",
    onnx=[
        {
            "component": "Softsign",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softsign.html",
        }
    ],
    since="0.7.1",
    context="primitives.nn",
    component="soft_sign",
    testcases=[
        {
            "testcase": "jaxnn_soft_sign",
            "callable": lambda x: jax.nn.soft_sign(x),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Softsign:1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_soft_sign_1",
            "callable": lambda x: jax.nn.soft_sign(x),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Softsign:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_softsign_basic",
            "callable": lambda x: jax.nn.soft_sign(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Softsign:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "softsign_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jax.nn.soft_sign(y) ** 2))(
                x
            ),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class SoftsignPlugin(PrimitiveLeafPlugin):
    """IR-only lowering for ``jax.nn.soft_sign``."""

    _PRIM: ClassVar[Primitive] = _SOFTSIGN_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("softsign_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("softsign_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("softsign_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("softsign_out")

        result = ctx.builder.Softsign(x_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = getattr(x_val, "type", None)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            result.shape = getattr(x_val, "shape", None)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jax.nn.soft_sign not found")
            return lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)

        return [
            AssignSpec("jax.nn", "soft_sign_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="soft_sign",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="soft_sign",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="soft_sign",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@SoftsignPlugin._PRIM.def_impl
def _softsign_impl(x: ArrayLike) -> ArrayLike:
    return jax.nn.soft_sign(x)


register_unary_elementwise_batch_rule(SoftsignPlugin._PRIM)


def _softsign_jvp_rule(
    primals: tuple[ArrayLike, ...], tangents: tuple[ArrayLike, ...], **_: object
) -> tuple[ArrayLike, ArrayLike]:
    (x,) = primals
    (x_dot,) = tangents
    x_dot = ad.instantiate_zeros(x_dot)

    one = jnp.asarray(1.0, dtype=x.dtype)
    denom = jax.lax.add(one, jax.lax.abs(x))
    primal_out = jax.lax.div(x, denom)

    denom_sq = jax.lax.mul(denom, denom)
    local_grad = jax.lax.div(one, denom_sq)
    tangent_out = jax.lax.mul(x_dot, local_grad)
    return primal_out, tangent_out


ad.primitive_jvps[SoftsignPlugin._PRIM] = _softsign_jvp_rule
