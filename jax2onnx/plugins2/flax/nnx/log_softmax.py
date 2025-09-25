# file: jax2onnx/plugins2/flax/nnx/log_softmax.py

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


def _int_attr(name: str, value: int):
    Attr = getattr(ir, "Attr", getattr(ir, "Attribute", None))
    if Attr is None:
        return None
    AttrType = getattr(ir, "AttributeType", getattr(ir, "AttrType", None))
    if hasattr(Attr, "i"):
        return Attr.i(name, int(value))
    if AttrType is not None:
        return Attr(name, AttrType.INT, int(value))
    return Attr(name, int(value))


def _axis_attr_equals(model, expected: int) -> bool:
    node = next(
        (n for n in getattr(model.graph, "node", []) if n.op_type == "LogSoftmax"),
        None,
    )
    if node is None:
        return False
    for attr in getattr(node, "attribute", []):
        if getattr(attr, "name", "") != "axis":
            continue
        val = None
        if hasattr(attr, "i"):
            val = attr.i
        elif hasattr(attr, "INT"):
            val = attr.INT
        elif getattr(attr, "ints", None):
            arr = attr.ints
            if len(arr):
                val = arr[0]
        if val is None:
            continue
        return int(val) == int(expected)
    # Attribute missing implies ONNX default of -1
    return int(expected) == -1


@register_primitive(
    jaxpr_primitive="nnx.log_softmax",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.log_softmax.html",
    onnx=[
        {
            "component": "LogSoftmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__LogSoftmax.html",
        }
    ],
    since="v0.2.0",
    context="primitives2.nnx",
    component="log_softmax",
    testcases=[
        {
            "testcase": "log_softmax",
            "callable": lambda x: nnx.log_softmax(x),
            "input_shapes": [(1, 4)],
            "use_onnx_ir": True,
            "post_check_onnx_graph": lambda m: _axis_attr_equals(m, 1),
        },
        {
            "testcase": "log_softmax_default_axis",
            "callable": lambda x: nnx.log_softmax(x),
            "input_shapes": [("B", 4)],
            "use_onnx_ir": True,
            "post_check_onnx_graph": lambda m: _axis_attr_equals(m, 1),
        },
        {
            "testcase": "log_softmax_axis0",
            "callable": lambda x: nnx.log_softmax(x, axis=0),
            "input_shapes": [(3, 2)],
            "use_onnx_ir": True,
            "post_check_onnx_graph": lambda m: _axis_attr_equals(m, 0),
        },
    ],
)
class LogSoftmaxPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.nnx.log_softmax → ONNX LogSoftmax."""

    _PRIM: ClassVar[Primitive] = Primitive("nnx.log_softmax")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(x, axis: int = -1):
        del axis
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: "IRBuildContext", eqn):
        (x_var,) = eqn.invars
        (y_var,) = eqn.outvars
        axis = int(eqn.params.get("axis", -1))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        axis_attr = axis
        if rank:
            axis_attr = axis % rank if axis < 0 else axis

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))

        attrs = []
        attr_obj = _int_attr("axis", int(axis_attr))
        if attr_obj is not None:
            attrs.append(attr_obj)

        ctx.add_node(
            ir.Node(
                op_type="LogSoftmax",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("LogSoftmax"),
                attributes=attrs,
            )
        )

        if attr_obj is None:
            maybe_set_attrs = getattr(ctx, "set_node_attrs", None)
            if callable(maybe_set_attrs):
                maybe_set_attrs({"axis": int(axis_attr)})

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
        prim = LogSoftmaxPlugin._PRIM

        def patched_log_softmax(x, axis: int = -1):
            return prim.bind(x, axis=axis)

        return patched_log_softmax

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("flax.nnx", "log_softmax_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="log_softmax",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, *, axis=-1: cls.abstract_eval(x, axis=axis)
            )
            cls._ABSTRACT_EVAL_BOUND = True


# ---------- concrete impl for eager execution ----------
@LogSoftmaxPlugin._PRIM.def_impl
def _impl(x, *, axis: int):
    return jax.nn.log_softmax(x, axis=axis)
