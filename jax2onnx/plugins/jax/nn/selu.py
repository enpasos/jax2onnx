# jax2onnx/plugins/jax/nn/selu.py

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


_SELU_PRIM: Final[Primitive] = Primitive("jax.nn.selu")
_SELU_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_SELU_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.selu.html",
    onnx=[
        {"component": "Selu", "doc": "https://onnx.ai/onnx/operators/onnx__Selu.html"}
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="selu",
    testcases=[
        {
            "testcase": "jaxnn_selu",
            "callable": lambda x: jax.nn.selu(x),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Selu:1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_selu_1",
            "callable": lambda x: jax.nn.selu(x),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Selu:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_selu_basic",
            "callable": lambda x: jax.nn.selu(x),
            "input_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Selu:Bx8"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class SeluPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.selu`` to ONNX ``Selu``."""

    _PRIM: ClassVar[Primitive] = _SELU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Selu",
            input_hint="selu_in",
            output_hint="selu_out",
            attrs={
                "alpha": 1.6732631921768188,
                "gamma": 1.0507010221481323,
            },
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
                raise RuntimeError("Original jax.nn.selu not found")
            return lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)

        return [
            AssignSpec("jax.nn", "selu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="selu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="selu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="selu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@SeluPlugin._PRIM.def_impl
def _selu_impl(x: ArrayLike) -> ArrayLike:
    return jax.nn.selu(x)


register_unary_elementwise_batch_rule(SeluPlugin._PRIM)
