# jax2onnx/plugins/flax/nnx/softmax.py

from __future__ import annotations
from typing import Callable, ClassVar, cast

import jax
import jax.numpy as jnp  # noqa: F401  (kept for parity with other plugins)
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
    jaxpr_primitive="nnx.softmax",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softmax.html",
    onnx=[
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        }
    ],
    since="0.1.0",
    context="primitives.nnx",
    component="softmax",
    testcases=[
        {
            "testcase": "softmax",
            "callable": lambda x: nnx.softmax(x),
            "input_shapes": [("B", 2)],
            "post_check_onnx_graph": expect_graph(
                ["Softmax:Bx2"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        }
    ],
)
class SoftmaxPlugin(PrimitiveLeafPlugin):
    """
    IR-only plugin for flax.nnx.softmax → ONNX Softmax.
    """

    _PRIM: ClassVar[Primitive] = Primitive("nnx.softmax")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue, axis: int = -1
    ) -> jax.core.ShapedArray:
        del axis
        # Output has same shape/dtype as input
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        (x_var,) = eqn.invars

        axis = int(eqn.params.get("axis", -1))
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)

        # Normalize negative axes to non-negative for ONNX attr robustness
        axis_attr = (axis % max(rank, 1)) if (axis < 0 and rank) else axis

        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Softmax",
            input_hint="softmax_in",
            output_hint="softmax_out",
            attrs={"axis": int(axis_attr)},
        )

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(
        orig_fn: Callable[..., ArrayLike] | None,
    ) -> Callable[..., ArrayLike]:
        del orig_fn
        prim = SoftmaxPlugin._PRIM

        def patched_softmax(x: ArrayLike, axis: int = -1) -> ArrayLike:
            return cast(ArrayLike, prim.bind(x, axis=axis))

        return patched_softmax

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            # Expose our private primitive as flax.nnx.softmax_p (for parity with other ops)
            AssignSpec("flax.nnx", "softmax_p", cls._PRIM, delete_if_missing=True),
            # Patch the public function nnx.softmax to bind our primitive during tracing
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="softmax",
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
@SoftmaxPlugin._PRIM.def_impl
def _impl(x: ArrayLike, *, axis: int) -> ArrayLike:
    return jax.nn.softmax(x, axis=axis)
