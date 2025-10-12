# jax2onnx/plugins/flax/nnx/elu.py

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ClassVar, List, Union

import numpy as np
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


def _alpha_attr_equals(model, expected: float) -> bool:
    node = next(
        (n for n in getattr(model.graph, "node", []) if n.op_type == "Elu"), None
    )
    if node is None:
        return False
    for attr in getattr(node, "attribute", []):
        if getattr(attr, "name", "") != "alpha":
            continue
        val = None
        if hasattr(attr, "f"):
            val = attr.f
        elif hasattr(attr, "FLOAT"):
            val = attr.FLOAT
        elif getattr(attr, "floats", None):
            arr = attr.floats
            if len(arr):
                val = arr[0]
        if val is None:
            continue
        return abs(float(val) - float(expected)) < 1e-6
    # Attribute not present => default alpha==1.0
    return abs(float(expected) - 1.0) < 1e-6


def _make_checker(specs, *, alpha: float, **kwargs):
    checker = expect_graph(specs, **kwargs)

    def _run(model):
        return checker(model) and _alpha_attr_equals(model, alpha)

    return _run


@register_primitive(
    jaxpr_primitive="nnx.elu",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.elu.html",
    onnx=[{"component": "Elu", "doc": "https://onnx.ai/onnx/operators/onnx__Elu.html"}],
    since="v0.2.0",
    context="primitives.nnx",
    component="elu",
    testcases=[
        {
            "testcase": "elu",
            "callable": lambda x: nnx.elu(x),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _make_checker(
                ["Elu:1"], alpha=1.0, no_unused_inputs=True
            ),
        },
        {
            "testcase": "elu_default",
            "callable": lambda x: nnx.elu(x),
            "input_shapes": [("B", 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _make_checker(
                ["Elu:Bx3"], symbols={"B": None}, alpha=1.0, no_unused_inputs=True
            ),
        },
        {
            "testcase": "elu_alpha",
            "callable": lambda x: nnx.elu(x, alpha=0.5),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _make_checker(
                ["Elu:2x3"], alpha=0.5, no_unused_inputs=True
            ),
        },
    ],
)
class EluPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.nnx.elu → ONNX Elu."""

    _PRIM: ClassVar[Primitive] = Primitive("nnx.elu")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(x, alpha: float = 1.0):
        del alpha
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: "IRBuildContext", eqn):
        (x_var,) = eqn.invars
        (y_var,) = eqn.outvars
        alpha = float(eqn.params.get("alpha", 1.0))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for Elu lowering")

        out_name = getattr(y_val, "name", None) or ctx.fresh_name("Elu")
        kwargs = {}
        if not np.isclose(alpha, 1.0):
            kwargs["alpha"] = float(alpha)
        result = builder.Elu(x_val, _outputs=[out_name], **kwargs)

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        final_dims: List[Union[int, str]] = []
        for i, d in enumerate(x_shape):
            if isinstance(d, (int, np.integer)):
                final_dims.append(int(d))
            else:
                final_dims.append(_dim_label_from_value_or_aval(x_val, x_shape, i))

        dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        if dtype is not None:
            result.type = ir.TensorType(dtype)

        _stamp_type_and_shape(result, tuple(final_dims))
        _ensure_value_metadata(ctx, result)

        bind_value = getattr(ctx, "bind_value_for_var", None)
        if callable(bind_value):
            bind_value(y_var, result)
        else:
            raise AttributeError("IR build context missing bind_value_for_var")

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(orig_fn: Callable):
        prim = EluPlugin._PRIM

        def patched_elu(x, alpha: float = 1.0):
            return prim.bind(x, alpha=alpha)

        return patched_elu

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("flax.nnx", "elu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="elu",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, *, alpha=1.0: cls.abstract_eval(x, alpha=alpha)
            )
            cls._ABSTRACT_EVAL_BOUND = True


# ---------- concrete impl for eager execution ----------
@EluPlugin._PRIM.def_impl
def _impl(x, *, alpha: float):
    return jax.nn.elu(x, alpha=alpha)
