from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
from jax.extend.core import Primitive
import onnx_ir as ir

from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_MISH_PRIM = Primitive("jax.nn.mish")
_MISH_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_MISH_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.mish.html",
    onnx=[
        {"component": "Mish", "doc": "https://onnx.ai/onnx/operators/onnx__Mish.html"}
    ],
    since="v0.7.1",
    context="primitives2.nn",
    component="mish",
    testcases=[
        {
            "testcase": "jaxnn_mish_basic",
            "callable": lambda x: jax.nn.mish(x),
            "input_shapes": [(2, 3, 4)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        }
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
        x_var = eqn.invars[0]
        y_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("mish_in"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("mish_out"))

        ctx.add_node(
            ir.Node(
                op_type="Mish",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Mish"),
            )
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
