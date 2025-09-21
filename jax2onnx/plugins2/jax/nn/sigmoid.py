from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
from jax.extend.core import Primitive
import onnx_ir as ir

from jax2onnx.plugins2._ir_shapes import _ensure_value_info as _add_value_info
from jax2onnx.plugins2._ir_shapes import _stamp_type_and_shape
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_SIGMOID_PRIM = Primitive("jax.nn.sigmoid")
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
    context="primitives2.nn",
    component="sigmoid",
    testcases=[
        {
            "testcase": "jaxnn_sigmoid_basic",
            "callable": lambda x: jax.nn.sigmoid(x),
            "input_shapes": [(5,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "jaxnn_sigmoid_dynamic",
            "callable": lambda x: jax.nn.sigmoid(x),
            "input_shapes": [("B", 3, 2)],
            "use_onnx_ir": True,
        },
    ],
)
class SigmoidPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.sigmoid`` to ONNX ``Sigmoid`` using the IR pipeline."""

    _PRIM: ClassVar[Primitive] = _SIGMOID_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x):
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        x_var = eqn.invars[0]
        y_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("sigmoid_in"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("sigmoid_out"))

        ctx.add_node(
            ir.Node(
                op_type="Sigmoid",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Sigmoid"),
            )
        )

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        _stamp_type_and_shape(y_val, x_shape)
        _add_value_info(ctx, y_val)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("jax.nn", "sigmoid_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="sigmoid",
                make_value=lambda orig: (
                    lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)
                ),
                delete_if_missing=False,
            ),
        ]


@SigmoidPlugin._PRIM.def_impl
def _sigmoid_impl(*args, **kwargs):
    return jax.nn.sigmoid(*args, **kwargs)
