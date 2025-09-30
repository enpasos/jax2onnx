# file: jax2onnx/plugins2/flax/nnx/softplus.py

from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Callable
import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec

if TYPE_CHECKING:
    from jax2onnx.converter2.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


@register_primitive(
    jaxpr_primitive="nnx.softplus",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html",
    onnx=[
        {
            "component": "Softplus",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softplus.html",
        }
    ],
    since="v0.1.0",
    context="primitives2.nnx",
    component="softplus",
    testcases=[
        {
            "testcase": "softplus",
            "callable": lambda x: nnx.softplus(x),
            "input_shapes": [(3,)],
            "run_only_f32_variant": True,
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
    def abstract_eval(x):
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: "IRBuildContext", eqn):
        (x_var,) = eqn.invars
        (y_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))

        ctx.add_node(
            ir.Node(
                op_type="Softplus",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Softplus"),
            )
        )

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(orig_fn: Callable):
        prim = SoftplusPlugin._PRIM

        def patched_softplus(x):
            return prim.bind(x)

        return patched_softplus

    @classmethod
    def binding_specs(cls):
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
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


# ---------- concrete impl for eager execution ----------
@SoftplusPlugin._PRIM.def_impl
def _impl(x):
    # Use a stable definition matching jax.nn.softplus
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)
