# jax2onnx/plugins/flax/nnx/softmax.py

from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Callable, List, Union

import jax
import jax.numpy as jnp  # noqa: F401  (kept for parity with other plugins)
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._ir_shapes import (
    _dim_label_from_value_or_aval,
    _stamp_type_and_shape,
    _ensure_value_metadata,
)

if TYPE_CHECKING:
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


@register_primitive(
    jaxpr_primitive="nnx.softmax",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softmax.html",
    onnx=[
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        }
    ],
    since="v0.1.0",
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
    def abstract_eval(x, axis: int = -1):
        # Output has same shape/dtype as input
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: "IRBuildContext", eqn):
        (x_var,) = eqn.invars
        (y_var,) = eqn.outvars

        axis = int(eqn.params.get("axis", -1))
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)

        # Normalize negative axes to non-negative for ONNX attr robustness
        axis_attr = (axis % max(rank, 1)) if (axis < 0 and rank) else axis

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for Softmax lowering"
            )

        out_name = getattr(y_val, "name", None) or ctx.fresh_name("Softmax")
        result = builder.Softmax(
            x_val,
            _outputs=[out_name],
            axis=int(axis_attr),
        )

        dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        if dtype is not None:
            result.type = ir.TensorType(dtype)

        if x_shape:
            dims: List[Union[int, str]] = [
                _dim_label_from_value_or_aval(x_val, x_shape, i)
                for i in range(len(x_shape))
            ]
            _stamp_type_and_shape(result, tuple(dims))
        _ensure_value_metadata(ctx, result)

        bind_value = getattr(ctx, "bind_value_for_var", None)
        if callable(bind_value):
            bind_value(y_var, result)
        else:
            raise AttributeError("IR build context missing bind_value_for_var")

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(orig_fn: Callable):
        prim = SoftmaxPlugin._PRIM

        def patched_softmax(x, axis: int = -1):
            return prim.bind(x, axis=axis)

        return patched_softmax

    @classmethod
    def binding_specs(cls):
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
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


# ---------- concrete impl for eager execution ----------
@SoftmaxPlugin._PRIM.def_impl
def _impl(x, *, axis: int):
    return jax.nn.softmax(x, axis=axis)
