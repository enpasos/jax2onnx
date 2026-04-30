# jax2onnx/plugins/flax/nnx/log_softmax.py

from __future__ import annotations
from typing import Any, Callable, ClassVar, cast

import jax
from jax.extend.core import Primitive
from flax import nnx
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.nn._builder_utils import (
    lower_unary_elementwise,
)


def _axis_attr_equals(model: Any, expected: int) -> bool:
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
    since="0.2.0",
    context="primitives.nnx",
    component="log_softmax",
    testcases=[
        {
            "testcase": "log_softmax",
            "callable": lambda x: nnx.log_softmax(x),
            "input_shapes": [(1, 4)],
            "post_check_onnx_graph": lambda m: _axis_attr_equals(m, 1),
        },
        {
            "testcase": "log_softmax_default_axis",
            "callable": lambda x: nnx.log_softmax(x),
            "input_shapes": [("B", 4)],
            "post_check_onnx_graph": lambda m: _axis_attr_equals(m, 1),
        },
        {
            "testcase": "log_softmax_axis0",
            "callable": lambda x: nnx.log_softmax(x, axis=0),
            "input_shapes": [(3, 2)],
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
    def abstract_eval(
        x: jax.core.AbstractValue, axis: int = -1
    ) -> jax.core.ShapedArray:
        del axis
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        axis = int(eqn.params.get("axis", -1))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        axis_attr = axis
        if rank:
            axis_attr = axis % rank if axis < 0 else axis

        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="LogSoftmax",
            input_hint="log_softmax_in",
            output_hint="log_softmax_out",
            attrs={"axis": int(axis_attr)},
        )

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(
        orig_fn: Callable[..., ArrayLike] | None,
    ) -> Callable[..., ArrayLike]:
        del orig_fn
        prim = LogSoftmaxPlugin._PRIM

        def patched_log_softmax(x: ArrayLike, axis: int = -1) -> ArrayLike:
            return cast(ArrayLike, prim.bind(x, axis=axis))

        return patched_log_softmax

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
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
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, *, axis=-1: cls.abstract_eval(x, axis=axis)
            )
            cls._ABSTRACT_EVAL_BOUND = True


# ---------- concrete impl for eager execution ----------
@LogSoftmaxPlugin._PRIM.def_impl
def _impl(x: ArrayLike, *, axis: int) -> ArrayLike:
    return jax.nn.log_softmax(x, axis=axis)
