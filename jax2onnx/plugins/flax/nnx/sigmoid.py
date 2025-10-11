# jax2onnx/plugins/flax/nnx/sigmoid.py

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ClassVar, List, Union

import jax
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
)

if TYPE_CHECKING:
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


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
    def abstract_eval(x):
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: "IRBuildContext", eqn):
        (x_var,) = eqn.invars
        (y_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for Sigmoid lowering"
            )

        out_name = getattr(y_val, "name", None) or ctx.fresh_name("Sigmoid")
        result = builder.Sigmoid(x_val, _outputs=[out_name])

        dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        if dtype is not None:
            result.type = ir.TensorType(dtype)

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
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
