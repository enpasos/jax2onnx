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


_RELU_PRIM = Primitive("jax.nn.relu")
_RELU_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_RELU_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.relu.html",
    onnx=[
        {"component": "Relu", "doc": "https://onnx.ai/onnx/operators/onnx__Relu.html"}
    ],
    since="v0.7.1",
    context="primitives2.nn",
    component="relu",
    testcases=[
        {
            "testcase": "jaxnn_relu",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [(1,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "jaxnn_relu_1",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [(2, 5)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "jaxnn_relu_basic",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [(3, 4)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "jaxnn_relu_dynamic",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [("B", 5)],
            "use_onnx_ir": True,
        },
    ],
)
class ReluPlugin(PrimitiveLeafPlugin):
    """IR-only lowering for ``jax.nn.relu`` via ONNX ``Relu``."""

    _PRIM: ClassVar[Primitive] = _RELU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x):
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        x_var = eqn.invars[0]
        y_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("relu_in"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("relu_out"))

        ctx.add_node(
            ir.Node(
                op_type="Relu",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Relu"),
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
            AssignSpec("jax.nn", "relu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="relu",
                make_value=lambda orig: (
                    lambda *args, **kwargs: cls._PRIM.bind(*args, **kwargs)
                ),
                delete_if_missing=False,
            ),
        ]


@ReluPlugin._PRIM.def_impl
def _relu_impl(*args, **kwargs):
    return jax.nn.relu(*args, **kwargs)
