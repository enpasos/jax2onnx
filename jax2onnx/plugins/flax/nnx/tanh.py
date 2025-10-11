# jax2onnx/plugins/flax/nnx/tanh.py

from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Callable, List, Union
import jax
import jax.numpy as jnp
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
    jaxpr_primitive="nnx.tanh",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.tanh",
    onnx=[
        {"component": "Tanh", "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html"}
    ],
    since="v0.1.0",
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
            raise AttributeError("IR build context missing builder for Tanh lowering")

        out_name = getattr(y_val, "name", None) or ctx.fresh_name("Tanh")
        result = builder.Tanh(x_val, _outputs=[out_name])

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
