# file: jax2onnx/plugins2/flax/nnx/tanh.py

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
    jaxpr_primitive="nnx.tanh",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.tanh",
    onnx=[{"component": "Tanh", "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html"}],
    since="v0.1.0",
    context="primitives2.nnx",
    component="tanh",
    testcases=[
        {
            "testcase": "tanh",
            "callable": lambda x: nnx.tanh(x),
            "input_shapes": [(3,)],
            "use_onnx_ir": True,
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
                op_type="Tanh",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Tanh"),
            )
        )

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(orig_fn: Callable):
        prim = TanhPlugin._PRIM

        def patched_tanh(x):
            return prim.bind(x)

        return patched_tanh

    @classmethod
    def binding_specs(cls):
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
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


# ---------- concrete impl for eager execution ----------
@TanhPlugin._PRIM.def_impl
def _impl(x):
    return jnp.tanh(x)
