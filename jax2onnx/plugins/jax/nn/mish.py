# jax2onnx/plugins/jax/nn/mish.py

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


_MISH_PRIM: Final[Primitive] = Primitive("jax.nn.mish")
_MISH_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_MISH_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.mish.html",
    onnx=[
        {"component": "Mish", "doc": "https://onnx.ai/onnx/operators/onnx__Mish.html"}
    ],
    since="v0.7.1",
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
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Mish",
            input_hint="mish_in",
            output_hint="mish_out",
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
