# jax2onnx/plugins/jax/nn/selu.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
from jax.extend.core import Primitive
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins._ir_shapes import _ensure_value_info as _add_value_info
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_SELU_PRIM = Primitive("jax.nn.selu")
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
        (x_var,) = eqn.invars
        (y_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("selu_in"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("selu_out"))

        attrs = [
            IRAttr("alpha", IRAttrType.FLOAT, 1.6732631921768188),
            IRAttr("gamma", IRAttrType.FLOAT, 1.0507010221481323),
        ]
        ctx.add_node(
            ir.Node(
                op_type="Selu",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Selu"),
                attributes=attrs,
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
