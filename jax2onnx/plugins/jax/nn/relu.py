# jax2onnx/plugins/jax/nn/relu.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
from jax.extend.core import Primitive
from jax.interpreters import ad
import jax.numpy as jnp
from numpy.typing import ArrayLike

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import (
    lower_unary_elementwise,
    register_unary_elementwise_batch_rule,
)


_RELU_PRIM: Final[Primitive] = Primitive("jax.nn.relu")
_RELU_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_RELU_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.relu.html",
    onnx=[
        {"component": "Relu", "doc": "https://onnx.ai/onnx/operators/onnx__Relu.html"}
    ],
    since="0.7.1",
    context="primitives.nn",
    component="relu",
    testcases=[
        {
            "testcase": "jaxnn_relu",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [(1,)],
            "post_check_onnx_graph": EG(
                ["Relu:1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_relu_1",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [(2, 5)],
            "post_check_onnx_graph": EG(
                ["Relu:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_relu_basic",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["Relu:3x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_relu_dynamic",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [("B", 5)],
            "post_check_onnx_graph": EG(
                ["Relu:Bx5"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "relu_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jax.nn.relu(y) ** 2))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class ReluPlugin(PrimitiveLeafPlugin):
    """IR-only lowering for ``jax.nn.relu`` via ONNX ``Relu``."""

    _PRIM: ClassVar[Primitive] = _RELU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Relu",
            input_hint="relu_in",
            output_hint="relu_out",
        )

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
                raise RuntimeError("Original jax.nn.relu not found")
            return lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)

        return [
            AssignSpec("jax.nn", "relu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="relu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="relu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="relu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@ReluPlugin._PRIM.def_impl
def _relu_impl(x: ArrayLike) -> ArrayLike:
    return jax.nn.relu(x)


register_unary_elementwise_batch_rule(ReluPlugin._PRIM)


def _relu_jvp_rule(
    primals: tuple[ArrayLike, ...], tangents: tuple[ArrayLike, ...], **_: object
) -> tuple[ArrayLike, ArrayLike]:
    (x,) = primals
    (x_dot,) = tangents
    x_dot = ad.instantiate_zeros(x_dot)

    zero_x = jnp.zeros((), dtype=x.dtype)
    primal_out = jax.lax.max(x, zero_x)

    zero_dot = jnp.zeros_like(x_dot)
    tangent_out = jax.lax.select(jax.lax.gt(x, zero_x), x_dot, zero_dot)
    return primal_out, tangent_out


ad.primitive_jvps[ReluPlugin._PRIM] = _relu_jvp_rule
