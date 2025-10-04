# jax2onnx/plugins/jax/nn/softplus.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
from jax.extend.core import Primitive
import onnx_ir as ir

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_SOFTPLUS_PRIM = Primitive("jax.nn.softplus")
_SOFTPLUS_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_SOFTPLUS_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html",
    onnx=[
        {
            "component": "Softplus",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softplus.html",
        }
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="softplus",
    testcases=[
        {
            "testcase": "jaxnn_softplus",
            "callable": lambda x: jax.nn.softplus(x),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "jaxnn_softplus_1",
            "callable": lambda x: jax.nn.softplus(x),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "jaxnn_softplus_basic",
            "callable": lambda x: jax.nn.softplus(x),
            "input_shapes": [(4,)],
            "run_only_f32_variant": True,
        },
    ],
)
class SoftplusPlugin(PrimitiveLeafPlugin):
    """IR-only lowering for ``jax.nn.softplus``."""

    _PRIM: ClassVar[Primitive] = _SOFTPLUS_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x):
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        x_var = eqn.invars[0]
        y_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("softplus_in"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("softplus_out"))

        ctx.add_node(
            ir.Node(
                op_type="Softplus",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Softplus"),
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
            AssignSpec("jax.nn", "softplus_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="softplus",
                make_value=lambda orig: (
                    lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)
                ),
                delete_if_missing=False,
            ),
        ]


@SoftplusPlugin._PRIM.def_impl
def _softplus_impl(*args, **kwargs):
    return jax.nn.softplus(*args, **kwargs)
