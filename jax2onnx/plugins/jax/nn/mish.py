# jax2onnx/plugins/jax/nn/mish.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import jax
from jax.extend.core import Primitive

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import lower_unary_elementwise

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


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
    def abstract_eval(x):
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
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
    def binding_specs(cls):
        return [
            AssignSpec("jax.nn", "mish_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="mish",
                make_value=lambda orig: (
                    lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)
                ),
                delete_if_missing=False,
            ),
        ]


@MishPlugin._PRIM.def_impl
def _mish_impl(*args, **kwargs):
    return jax.nn.mish(*args, **kwargs)
