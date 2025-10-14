# jax2onnx/plugins/jax/nn/leaky_relu.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import jax
from jax.extend.core import Primitive

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.nn._builder_utils import lower_unary_elementwise

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_LEAKY_RELU_PRIM: Final[Primitive] = Primitive("jax.nn.leaky_relu")
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
    context="primitives.nn",
    component="leaky_relu",
    testcases=[
        {
            "testcase": "jaxnn_leaky_relu",
            "callable": lambda x: jax.nn.leaky_relu(x, negative_slope=0.1),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["LeakyRelu:1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_leaky_relu_1",
            "callable": lambda x: jax.nn.leaky_relu(x, negative_slope=0.2),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["LeakyRelu:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_leaky_relu_default",
            "callable": lambda x: jax.nn.leaky_relu(x),
            "input_shapes": [("B", 3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["LeakyRelu:Bx3x4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_leaky_relu_custom",
            "callable": lambda x: jax.nn.leaky_relu(x, negative_slope=0.3),
            "input_shapes": [(5,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["LeakyRelu:5"],
                no_unused_inputs=True,
            ),
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
        negative_slope = float(eqn.params.get("negative_slope", 0.01))

        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="LeakyRelu",
            input_hint="leaky_relu_in",
            output_hint="leaky_relu_out",
            attrs={"alpha": negative_slope},
        )

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
