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


_CELU_PRIM = Primitive("jax.nn.celu")
_CELU_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_CELU_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.celu.html",
    onnx=[
        {"component": "Celu", "doc": "https://onnx.ai/onnx/operators/onnx__Celu.html"}
    ],
    since="v0.7.1",
    context="primitives2.nn",
    component="celu",
    testcases=[
        {
            "testcase": "jaxnn_celu",
            "callable": lambda x: jax.nn.celu(x, alpha=0.1),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "jaxnn_celu_1",
            "callable": lambda x: jax.nn.celu(x, alpha=0.2),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "jaxnn_celu_alpha_default",
            "callable": lambda x: jax.nn.celu(x),
            "input_shapes": [(3, 4)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "jaxnn_celu_alpha_custom",
            "callable": lambda x: jax.nn.celu(x, alpha=0.3),
            "input_shapes": [("B", 2, 2)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
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
        (x_var,) = eqn.invars
        (y_var,) = eqn.outvars
        alpha = float(eqn.params.get("alpha", 1.0))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("celu_in"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("celu_out"))

        attr = IRAttr("alpha", IRAttrType.FLOAT, alpha)
        ctx.add_node(
            ir.Node(
                op_type="Celu",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Celu"),
                attributes=[attr],
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
