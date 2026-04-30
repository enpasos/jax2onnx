# jax2onnx/plugins/flax/nnx/tanh.py

from __future__ import annotations
from typing import Callable, ClassVar, cast
import jax
import jax.numpy as jnp
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
    jaxpr_primitive="nnx.tanh",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.tanh",
    onnx=[
        {"component": "Tanh", "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html"}
    ],
    since="0.1.0",
    context="primitives.nnx",
    component="tanh",
    testcases=[
        {
            "testcase": "tanh",
            "callable": lambda x: nnx.tanh(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": expect_graph(
                ["Tanh:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class TanhPlugin(PrimitiveLeafPlugin):
    """
    IR-only plugin for flax.nnx.tanh → ONNX Tanh.
    """

    _PRIM: ClassVar[Primitive] = Primitive("nnx.tanh")
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
            op_name="Tanh",
            input_hint="tanh_in",
            output_hint="tanh_out",
        )

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(
        orig_fn: Callable[..., ArrayLike] | None,
    ) -> Callable[[ArrayLike], ArrayLike]:
        del orig_fn
        prim = TanhPlugin._PRIM

        def patched_tanh(x: ArrayLike) -> ArrayLike:
            return cast(ArrayLike, prim.bind(x))

        return patched_tanh

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            # Expose a private primitive handle (created if missing)
            AssignSpec("flax.nnx", "tanh_p", cls._PRIM, delete_if_missing=True),
            # Monkey-patch nnx.tanh while tracing
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="tanh",
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
@TanhPlugin._PRIM.def_impl
def _impl(x: ArrayLike) -> ArrayLike:
    return jnp.tanh(x)
