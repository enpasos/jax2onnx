# jax2onnx/plugins/jax/nn/softsign.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
from jax.extend.core import Primitive
from numpy.typing import ArrayLike

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import (
    lower_unary_elementwise,
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
    since="v0.7.1",
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
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Softsign",
            input_hint="softsign_in",
            output_hint="softsign_out",
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
