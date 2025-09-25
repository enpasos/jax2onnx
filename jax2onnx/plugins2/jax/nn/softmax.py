from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
from jax.extend.core import Primitive
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _ensure_value_info as _add_value_info
from jax2onnx.plugins2._ir_shapes import _stamp_type_and_shape
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_SOFTMAX_PRIM = Primitive("jax.nn.softmax")
_SOFTMAX_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_SOFTMAX_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softmax.html",
    onnx=[
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        }
    ],
    since="v0.7.1",
    context="primitives2.nn",
    component="softmax",
    testcases=[
        {
            "testcase": "softmax",
            "callable": lambda x: jax.nn.softmax(x),
            "input_shapes": [(3,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "softmax_2d",
            "callable": lambda x: jax.nn.softmax(x, axis=1),
            "input_shapes": [(4, 5)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "softmax_3d",
            "callable": lambda x: jax.nn.softmax(x, axis=2),
            "input_shapes": [(2, 3, 4)],
            "use_onnx_ir": True,
        },
    ],
)
class SoftmaxPlugin(PrimitiveLeafPlugin):
    """IR lowering for ``jax.nn.softmax`` using ONNX ``Softmax``."""

    _PRIM: ClassVar[Primitive] = _SOFTMAX_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, axis: int = -1):
        del axis
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        (x_var,) = eqn.invars
        (y_var,) = eqn.outvars

        axis = int(eqn.params.get("axis", -1))
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        norm_axis = (axis % rank) if (axis < 0 and rank) else axis

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("softmax_in"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("softmax_out"))

        attr = IRAttr("axis", IRAttrType.INT, int(norm_axis))
        ctx.add_node(
            ir.Node(
                op_type="Softmax",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Softmax"),
                attributes=[attr],
            )
        )

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
            AssignSpec("jax.nn", "softmax_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="softmax",
                make_value=lambda orig: (
                    lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)
                ),
                delete_if_missing=False,
            ),
        ]


@SoftmaxPlugin._PRIM.def_impl
def _softmax_impl(*args, **kwargs):
    return jax.nn.softmax(*args, **kwargs)
