# jax2onnx/plugins/flax/nnx/sigmoid.py

from __future__ import annotations
from typing import Callable, ClassVar, cast

import jax
from jax.extend.core import Primitive
from flax import nnx
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.nn._builder_utils import (
    lower_unary_elementwise,
)


@register_primitive(
    jaxpr_primitive="nnx.sigmoid",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.sigmoid",
    onnx=[
        {
            "component": "Sigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
        }
    ],
    since="0.2.0",
    context="primitives.nnx",
    component="sigmoid",
    testcases=[
        {
            "testcase": "sigmoid",
            "callable": lambda x: nnx.sigmoid(x),
            "input_shapes": [("B", 4)],
            "post_check_onnx_graph": expect_graph(
                ["Sigmoid:Bx4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        }
    ],
)
class SigmoidPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.nnx.sigmoid → ONNX Sigmoid."""

    _PRIM: ClassVar[Primitive] = Primitive("nnx.sigmoid")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(x: jax.core.AbstractValue) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Sigmoid",
            input_hint="sigmoid_in",
            output_hint="sigmoid_out",
        )

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(
        orig_fn: Callable[..., ArrayLike] | None,
    ) -> Callable[[ArrayLike], ArrayLike]:
        del orig_fn
        prim = SigmoidPlugin._PRIM

        def patched_sigmoid(x: ArrayLike) -> ArrayLike:
            return cast(ArrayLike, prim.bind(x))

        return patched_sigmoid

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("flax.nnx", "sigmoid_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="sigmoid",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


# ---------- concrete impl for eager execution ----------
@SigmoidPlugin._PRIM.def_impl
def _impl(x: ArrayLike) -> ArrayLike:
    return jax.nn.sigmoid(x)
