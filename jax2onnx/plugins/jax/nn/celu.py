# jax2onnx/plugins/jax/nn/celu.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import jax
from jax.extend.core import Primitive

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import lower_unary_elementwise

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_CELU_PRIM: Final[Primitive] = Primitive("jax.nn.celu")
_CELU_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_CELU_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.celu.html",
    onnx=[
        {"component": "Celu", "doc": "https://onnx.ai/onnx/operators/onnx__Celu.html"}
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="celu",
    testcases=[
        {
            "testcase": "jaxnn_celu",
            "callable": lambda x: jax.nn.celu(x, alpha=0.1),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "jaxnn_celu_1",
            "callable": lambda x: jax.nn.celu(x, alpha=0.2),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "jaxnn_celu_alpha_default",
            "callable": lambda x: jax.nn.celu(x),
            "input_shapes": [(3, 4)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "jaxnn_celu_alpha_custom",
            "callable": lambda x: jax.nn.celu(x, alpha=0.3),
            "input_shapes": [("B", 2, 2)],
            "run_only_f32_variant": True,
        },
    ],
)
class CeluPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.celu`` to ONNX ``Celu`` (IR-only)."""

    _PRIM: ClassVar[Primitive] = _CELU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, alpha: float = 1.0):
        del alpha
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        alpha = float(eqn.params.get("alpha", 1.0))

        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Celu",
            input_hint="celu_in",
            output_hint="celu_out",
            attrs={"alpha": alpha},
        )

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("jax.nn", "celu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="celu",
                make_value=lambda orig: (
                    lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)
                ),
                delete_if_missing=False,
            ),
        ]


@CeluPlugin._PRIM.def_impl
def _celu_impl(*args, **kwargs):
    return jax.nn.celu(*args, **kwargs)
