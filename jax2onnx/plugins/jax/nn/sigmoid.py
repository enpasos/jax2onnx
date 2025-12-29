# jax2onnx/plugins/jax/nn/sigmoid.py

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


_SIGMOID_PRIM: Final[Primitive] = Primitive("jax.nn.sigmoid")
_SIGMOID_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_SIGMOID_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.sigmoid.html",
    onnx=[
        {
            "component": "Sigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
        }
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="sigmoid",
    testcases=[
        {
            "testcase": "jaxnn_sigmoid",
            "callable": lambda x: jax.nn.sigmoid(x),
            "input_shapes": [(1,)],
            "post_check_onnx_graph": EG(
                ["Sigmoid:1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_sigmoid_1",
            "callable": lambda x: jax.nn.sigmoid(x),
            "input_shapes": [(2, 5)],
            "post_check_onnx_graph": EG(
                ["Sigmoid:2x5"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SigmoidPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.sigmoid`` to ONNX ``Sigmoid`` using the IR pipeline."""

    _PRIM: ClassVar[Primitive] = _SIGMOID_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Sigmoid",
            input_hint="sigmoid_in",
            output_hint="sigmoid_out",
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
                raise RuntimeError("Original jax.nn.sigmoid not found")
            return lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)

        return [
            AssignSpec("jax.nn", "sigmoid_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="sigmoid",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="sigmoid",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="sigmoid",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@SigmoidPlugin._PRIM.def_impl
def _sigmoid_impl(x: ArrayLike) -> ArrayLike:
    return jax.nn.sigmoid(x)


register_unary_elementwise_batch_rule(SigmoidPlugin._PRIM)
