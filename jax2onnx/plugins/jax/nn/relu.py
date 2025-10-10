# jax2onnx/plugins/jax/nn/relu.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import jax
from jax.extend.core import Primitive

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import lower_unary_elementwise

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_RELU_PRIM: Final[Primitive] = Primitive("jax.nn.relu")
_RELU_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_RELU_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.relu.html",
    onnx=[
        {"component": "Relu", "doc": "https://onnx.ai/onnx/operators/onnx__Relu.html"}
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="relu",
    testcases=[
        {
            "testcase": "jaxnn_relu",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_relu_1",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [(2, 5)],
        },
        {
            "testcase": "jaxnn_relu_basic",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "jaxnn_relu_dynamic",
            "callable": lambda x: jax.nn.relu(x),
            "input_shapes": [("B", 5)],
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
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Relu",
            input_hint="relu_in",
            output_hint="relu_out",
        )

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
