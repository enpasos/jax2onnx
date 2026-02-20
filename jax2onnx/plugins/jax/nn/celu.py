# jax2onnx/plugins/jax/nn/celu.py

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


_CELU_PRIM: Final[Primitive] = Primitive("jax.nn.celu")
_CELU_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_CELU_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.celu.html",
    onnx=[
        {"component": "Celu", "doc": "https://onnx.ai/onnx/operators/onnx__Celu.html"}
    ],
    since="0.7.1",
    context="primitives.nn",
    component="celu",
    testcases=[
        {
            "testcase": "jaxnn_celu",
            "callable": lambda x: jax.nn.celu(x, alpha=0.1),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Celu:1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_celu_1",
            "callable": lambda x: jax.nn.celu(x, alpha=0.2),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Celu:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_celu_alpha_default",
            "callable": lambda x: jax.nn.celu(x),
            "input_shapes": [(3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Celu:3x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_celu_alpha_custom",
            "callable": lambda x: jax.nn.celu(x, alpha=0.3),
            "input_shapes": [("B", 2, 2)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Celu:Bx2x2"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "celu_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(
                lambda y: jnp.sum(jax.nn.celu(y, alpha=0.3) ** 2)
            )(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class CeluPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.celu`` to ONNX ``Celu`` (IR-only)."""

    _PRIM: ClassVar[Primitive] = _CELU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue, alpha: float = 1.0
    ) -> jax.core.ShapedArray:
        del alpha
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        alpha = float(eqn.params.get("alpha", 1.0))
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("celu_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("celu_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("celu_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("celu_out")

        result = ctx.builder.Celu(
            x_val,
            _outputs=[desired_name],
            alpha=alpha,
        )
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
                raise RuntimeError("Original jax.nn.celu not found")
            return lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)

        return [
            AssignSpec("jax.nn", "celu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="celu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="celu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="celu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@CeluPlugin._PRIM.def_impl
def _celu_impl(x: ArrayLike, alpha: float = 1.0) -> ArrayLike:
    return jax.nn.celu(x, alpha=alpha)


register_unary_elementwise_batch_rule(CeluPlugin._PRIM)


def _celu_jvp_rule(
    primals: tuple[ArrayLike, ...], tangents: tuple[ArrayLike, ...], **params: object
) -> tuple[ArrayLike, ArrayLike]:
    alpha = float(params.get("alpha", 1.0))

    (x,) = primals
    (x_dot,) = tangents
    x_dot = ad.instantiate_zeros(x_dot)

    zero = jnp.asarray(0.0, dtype=x.dtype)
    one = jnp.asarray(1.0, dtype=x.dtype)
    alpha_x = jnp.asarray(alpha, dtype=x.dtype)

    x_div_alpha = jax.lax.div(x, alpha_x)
    neg_branch = jax.lax.mul(alpha_x, jax.lax.sub(jax.lax.exp(x_div_alpha), one))
    primal_out = jax.lax.select(jax.lax.gt(x, zero), x, neg_branch)

    one_like_x = jax.lax.broadcast_in_dim(one, x.shape, ())
    deriv = jax.lax.select(jax.lax.gt(x, zero), one_like_x, jax.lax.exp(x_div_alpha))
    tangent_out = jax.lax.mul(x_dot, deriv)
    return primal_out, tangent_out


ad.primitive_jvps[CeluPlugin._PRIM] = _celu_jvp_rule
