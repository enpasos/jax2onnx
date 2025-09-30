# file: jax2onnx/plugins2/flax/nnx/leaky_relu.py

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


def _make_float_attr(name: str, value: float):
    Attr = getattr(ir, "Attr", getattr(ir, "Attribute", None))
    if Attr is None:
        return None
    AttrType = getattr(ir, "AttributeType", getattr(ir, "AttrType", None))
    if hasattr(Attr, "f"):
        return Attr.f(name, float(value))
    if AttrType is not None:
        return Attr(name, AttrType.FLOAT, float(value))
    return Attr(name, float(value))


def _alpha_attr_equals(model, expected: float) -> bool:
    node = next(
        (n for n in getattr(model.graph, "node", []) if n.op_type == "LeakyRelu"), None
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
    return abs(float(expected) - 0.01) < 1e-6


@register_primitive(
    jaxpr_primitive="nnx.leaky_relu",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.leaky_relu.html",
    onnx=[
        {
            "component": "LeakyRelu",
            "doc": "https://onnx.ai/onnx/operators/onnx__LeakyRelu.html",
        }
    ],
    since="v0.2.0",
    context="primitives2.nnx",
    component="leaky_relu",
    testcases=[
        {
            "testcase": "leaky_relu",
            "callable": lambda x: nnx.leaky_relu(x),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: _alpha_attr_equals(m, 0.01),
        },
        {
            "testcase": "leaky_relu_default",
            "callable": lambda x: nnx.leaky_relu(x),
            "input_shapes": [("B", 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: _alpha_attr_equals(m, 0.01),
        },
        {
            "testcase": "leaky_relu_custom",
            "callable": lambda x: nnx.leaky_relu(x, negative_slope=0.2),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: _alpha_attr_equals(m, 0.2),
        },
    ],
)
class LeakyReluPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.nnx.leaky_relu → ONNX LeakyRelu."""

    _PRIM: ClassVar[Primitive] = Primitive("nnx.leaky_relu")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(x, negative_slope: float = 0.01):
        del negative_slope
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: "IRBuildContext", eqn):
        (x_var,) = eqn.invars
        (y_var,) = eqn.outvars
        negative_slope = float(eqn.params.get("negative_slope", 0.01))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))

        attrs = []
        attr_obj = _make_float_attr("alpha", negative_slope)
        if attr_obj is not None and not np.isclose(negative_slope, 0.01):
            attrs.append(attr_obj)
        elif attr_obj is None and not np.isclose(negative_slope, 0.01):
            pass

        ctx.add_node(
            ir.Node(
                op_type="LeakyRelu",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("LeakyRelu"),
                attributes=attrs,
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

        if attr_obj is None and not np.isclose(negative_slope, 0.01):
            maybe_set_attrs = getattr(ctx, "set_node_attrs", None)
            if callable(maybe_set_attrs):
                maybe_set_attrs({"alpha": float(negative_slope)})

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(orig_fn: Callable):
        prim = LeakyReluPlugin._PRIM

        def patched_leaky_relu(x, negative_slope: float = 0.01):
            return prim.bind(x, negative_slope=negative_slope)

        return patched_leaky_relu

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("flax.nnx", "leaky_relu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="leaky_relu",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, *, negative_slope=0.01: cls.abstract_eval(
                    x, negative_slope=negative_slope
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


# ---------- concrete impl for eager execution ----------
@LeakyReluPlugin._PRIM.def_impl
def _impl(x, *, negative_slope: float):
    return jax.nn.leaky_relu(x, negative_slope=negative_slope)
