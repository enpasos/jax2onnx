# jax2onnx/plugins/flax/nnx/softplus.py

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
    jaxpr_primitive="nnx.softplus",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html",
    onnx=[
        {
            "component": "Softplus",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softplus.html",
        }
    ],
    since="0.1.0",
    context="primitives.nnx",
    component="softplus",
    testcases=[
        {
            "testcase": "softplus",
            "callable": lambda x: nnx.softplus(x),
            "input_shapes": [(3,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["Softplus:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class SoftplusPlugin(PrimitiveLeafPlugin):
    """
    IR-only plugin for flax.nnx.softplus → ONNX Softplus.
    """

    _PRIM: ClassVar[Primitive] = Primitive("nnx.softplus")
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
            op_name="Softplus",
            input_hint="softplus_in",
            output_hint="softplus_out",
        )

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(
        orig_fn: Callable[..., ArrayLike] | None,
    ) -> Callable[[ArrayLike], ArrayLike]:
        del orig_fn
        prim = SoftplusPlugin._PRIM

        def patched_softplus(x: ArrayLike) -> ArrayLike:
            return cast(ArrayLike, prim.bind(x))

        return patched_softplus

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("flax.nnx", "softplus_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="softplus",
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
@SoftplusPlugin._PRIM.def_impl
def _impl(x: ArrayLike) -> ArrayLike:
    # Use a stable definition matching jax.nn.softplus
    return cast(ArrayLike, jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0))
