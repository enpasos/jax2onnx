# jax2onnx/plugins/jax/nn/softsign.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import jax
from jax.extend.core import Primitive

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import lower_unary_elementwise

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


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
        },
        {
            "testcase": "jaxnn_soft_sign_1",
            "callable": lambda x: jax.nn.soft_sign(x),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "jaxnn_softsign_basic",
            "callable": lambda x: jax.nn.soft_sign(x),
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
    def abstract_eval(x):
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
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
    def binding_specs(cls):
        return [
            AssignSpec("jax.nn", "soft_sign_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="soft_sign",
                make_value=lambda orig: (
                    lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)
                ),
                delete_if_missing=False,
            ),
        ]


@SoftsignPlugin._PRIM.def_impl
def _softsign_impl(*args, **kwargs):
    return jax.nn.soft_sign(*args, **kwargs)
