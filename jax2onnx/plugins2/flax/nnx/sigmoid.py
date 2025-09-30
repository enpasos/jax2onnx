# file: jax2onnx/plugins2/flax/nnx/sigmoid.py

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ClassVar, List, Union

import numpy as np
import jax
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2._ir_shapes import (
    _stamp_type_and_shape,
    _dim_label_from_value_or_aval,
    _ensure_value_info as _add_value_info,
)

if TYPE_CHECKING:
    from jax2onnx.converter2.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


@register_primitive(
    jaxpr_primitive="nnx.sigmoid",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.sigmoid",
    onnx=[
        {
            "component": "Sigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
        }
    ],
    since="v0.2.0",
    context="primitives2.nnx",
    component="sigmoid",
    testcases=[
        {
            "testcase": "sigmoid",
            "callable": lambda x: nnx.sigmoid(x),
            "input_shapes": [("B", 4)],
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
                op_type="Sigmoid",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Sigmoid"),
            )
        )

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        final_dims: List[Union[int, str]] = []
        for i, d in enumerate(x_shape):
            if isinstance(d, (int, np.integer)):
                final_dims.append(int(d))
            else:
                final_dims.append(_dim_label_from_value_or_aval(x_val, x_shape, i))

        _stamp_type_and_shape(y_val, tuple(final_dims))
        _add_value_info(ctx, y_val)

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(orig_fn: Callable):
        prim = SigmoidPlugin._PRIM

        def patched_sigmoid(x):
            return prim.bind(x)

        return patched_sigmoid

    @classmethod
    def binding_specs(cls):
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
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


# ---------- concrete impl for eager execution ----------
@SigmoidPlugin._PRIM.def_impl
def _impl(x):
    return jax.nn.sigmoid(x)
