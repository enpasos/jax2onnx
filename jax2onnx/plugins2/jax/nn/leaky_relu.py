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


_LEAKY_RELU_PRIM = Primitive("jax.nn.leaky_relu")
_LEAKY_RELU_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_LEAKY_RELU_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.leaky_relu.html",
    onnx=[
        {
            "component": "LeakyRelu",
            "doc": "https://onnx.ai/onnx/operators/onnx__LeakyRelu.html",
        }
    ],
    since="v0.7.1",
    context="primitives2.nn",
    component="leaky_relu",
    testcases=[
        {
            "testcase": "jaxnn_leaky_relu",
            "callable": lambda x: jax.nn.leaky_relu(x, negative_slope=0.1),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "jaxnn_leaky_relu_1",
            "callable": lambda x: jax.nn.leaky_relu(x, negative_slope=0.2),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "jaxnn_leaky_relu_default",
            "callable": lambda x: jax.nn.leaky_relu(x),
            "input_shapes": [("B", 3, 4)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "jaxnn_leaky_relu_custom",
            "callable": lambda x: jax.nn.leaky_relu(x, negative_slope=0.3),
            "input_shapes": [(5,)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
    ],
)
class LeakyReluPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.leaky_relu`` to ONNX ``LeakyRelu``."""

    _PRIM: ClassVar[Primitive] = _LEAKY_RELU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, negative_slope: float = 0.01):
        del negative_slope
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        (x_var,) = eqn.invars
        (y_var,) = eqn.outvars
        negative_slope = float(eqn.params.get("negative_slope", 0.01))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("leaky_relu_in"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("leaky_relu_out"))

        attr = IRAttr("alpha", IRAttrType.FLOAT, negative_slope)
        ctx.add_node(
            ir.Node(
                op_type="LeakyRelu",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("LeakyRelu"),
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
            AssignSpec("jax.nn", "leaky_relu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="leaky_relu",
                make_value=lambda orig: (
                    lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)
                ),
                delete_if_missing=False,
            ),
        ]


@LeakyReluPlugin._PRIM.def_impl
def _leaky_relu_impl(*args, **kwargs):
    return jax.nn.leaky_relu(*args, **kwargs)
