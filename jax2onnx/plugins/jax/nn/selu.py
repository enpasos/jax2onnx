# jax2onnx/plugins/jax/nn/selu.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import jax
from jax.extend.core import Primitive

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import lower_unary_elementwise

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


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
        },
        {
            "testcase": "jaxnn_selu_1",
            "callable": lambda x: jax.nn.selu(x),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "jaxnn_selu_basic",
            "callable": lambda x: jax.nn.selu(x),
            "input_shapes": [("B", 8)],
            "run_only_f32_variant": True,
        },
    ],
)
class SeluPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.selu`` to ONNX ``Selu``."""

    _PRIM: ClassVar[Primitive] = _SELU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x):
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
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
    def binding_specs(cls):
        return [
            AssignSpec("jax.nn", "selu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="selu",
                make_value=lambda orig: (
                    lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)
                ),
                delete_if_missing=False,
            ),
        ]


@SeluPlugin._PRIM.def_impl
def _selu_impl(*args, **kwargs):
    return jax.nn.selu(*args, **kwargs)
