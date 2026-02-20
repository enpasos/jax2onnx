# jax2onnx/plugins/jax/nn/mish.py

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


_MISH_PRIM: Final[Primitive] = Primitive("jax.nn.mish")
_MISH_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_MISH_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.mish.html",
    onnx=[
        {"component": "Mish", "doc": "https://onnx.ai/onnx/operators/onnx__Mish.html"}
    ],
    since="0.7.1",
    context="primitives.nn",
    component="mish",
    testcases=[
        {
            "testcase": "jaxnn_mish",
            "callable": lambda x: jax.nn.mish(x),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Mish:1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_mish_1",
            "callable": lambda x: jax.nn.mish(x),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Mish:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_mish_basic",
            "callable": lambda x: jax.nn.mish(x),
            "input_shapes": [(2, 3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Mish:2x3x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "mish_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jax.nn.mish(y) ** 2))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class MishPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.mish`` to ONNX ``Mish``."""

    _PRIM: ClassVar[Primitive] = _MISH_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("mish_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("mish_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("mish_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("mish_out")

        result = ctx.builder.Mish(x_val, _outputs=[desired_name])
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
                raise RuntimeError("Original jax.nn.mish not found")
            return lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)

        return [
            AssignSpec("jax.nn", "mish_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="mish",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@MishPlugin._PRIM.def_impl
def _mish_impl(x: ArrayLike) -> ArrayLike:
    return jax.nn.mish(x)


register_unary_elementwise_batch_rule(MishPlugin._PRIM)


def _mish_jvp_rule(
    primals: tuple[ArrayLike, ...], tangents: tuple[ArrayLike, ...], **_: object
) -> tuple[ArrayLike, ArrayLike]:
    (x,) = primals
    (x_dot,) = tangents
    x_dot = ad.instantiate_zeros(x_dot)

    one = jnp.asarray(1.0, dtype=x.dtype)
    softplus_x = jax.lax.log(jax.lax.add(one, jax.lax.exp(x)))
    tanh_sp = jax.lax.tanh(softplus_x)
    primal_out = jax.lax.mul(x, tanh_sp)

    sigmoid_x = jax.lax.div(one, jax.lax.add(one, jax.lax.exp(jax.lax.neg(x))))
    sech2 = jax.lax.sub(one, jax.lax.mul(tanh_sp, tanh_sp))
    deriv = jax.lax.add(tanh_sp, jax.lax.mul(x, jax.lax.mul(sigmoid_x, sech2)))
    tangent_out = jax.lax.mul(x_dot, deriv)
    return primal_out, tangent_out


ad.primitive_jvps[MishPlugin._PRIM] = _mish_jvp_rule
