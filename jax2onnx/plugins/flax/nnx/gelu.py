# file: jax2onnx/plugins/flax/nnx/gelu.py
from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Callable, Any

import jax
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    _dim_label_from_value_or_aval,
    is_shape_all_unknown,
    _ensure_value_info as _add_value_info,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG

if TYPE_CHECKING:
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


@register_primitive(
    jaxpr_primitive="nnx.gelu",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.gelu",
    onnx=[
        {"component": "Gelu", "doc": "https://onnx.ai/onnx/operators/onnx__Gelu.html"}
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="gelu",
    testcases=[
        {
            "testcase": "gelu",
            "callable": lambda x: nnx.gelu(x, approximate=False),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: EXPECT_GELU(m)
            and _approx_attr_equals(m, "none"),
        },
        {
            "testcase": "gelu_1",
            "callable": lambda x: nnx.gelu(x, approximate=False),
            "input_shapes": [(1, 10)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: EXPECT_GELU(m)
            and _approx_attr_equals(m, "none"),
        },
        {
            "testcase": "gelu_2",
            "callable": lambda x: nnx.gelu(x, approximate=True),
            "input_shapes": [(1,)],
            "post_check_onnx_graph": lambda m: EXPECT_GELU(m)
            and _approx_attr_equals(m, "tanh"),
        },
        {
            "testcase": "gelu_3",
            "callable": lambda x: nnx.gelu(x, approximate=True),
            "input_shapes": [("B", 10)],
            "post_check_onnx_graph": lambda m: EXPECT_GELU(m)
            and _approx_attr_equals(m, "tanh"),
        },
        {
            "testcase": "gelu_4",
            "callable": lambda x: nnx.gelu(x),
            "input_shapes": [(1,)],
            # default path (no flag) must be approximate=True => "tanh"
            "post_check_onnx_graph": lambda m: EXPECT_GELU(m)
            and _approx_attr_equals(m, "tanh"),
        },
        {
            "testcase": "gelu_5",
            "callable": lambda x: nnx.gelu(x),
            "input_shapes": [("B", 10)],
            "post_check_onnx_graph": lambda m: EXPECT_GELU(m)
            and _approx_attr_equals(m, "tanh"),
        },
    ],
)
class GeluPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.nnx.gelu → ONNX Gelu."""

    _PRIM: ClassVar[Primitive] = Primitive("nnx.gelu")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(x, *, approximate=True):
        del approximate
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: "IRBuildContext", eqn):
        (x_var,) = eqn.invars[:1]
        (y_var,) = eqn.outvars
        approximate = bool(eqn.params.get("approximate", True))

        # Values
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))

        # Preserve original input meta shape on graph.input if the binder left it unknown
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if is_shape_all_unknown(getattr(x_val, "shape", None)):
            if any(d is not None for d in x_shape):
                _stamp_type_and_shape(x_val, x_shape)

        # Output value
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))
        y_meta = tuple(
            _dim_label_from_value_or_aval(x_val, x_shape, i)
            for i in range(len(x_shape))
        )
        _stamp_type_and_shape(y_val, y_meta)

        # Build attributes for ONNX Gelu.
        # Set approximate="tanh" when requested; use "none" otherwise to keep tests explicit.
        attrs = []
        Attr = getattr(ir, "Attr", None)
        AttrType = getattr(ir, "AttributeType", getattr(ir, "AttrType", None))
        approx_str = "tanh" if approximate else "none"
        if Attr is not None:
            # Prefer typed classmethods if available; fallback to enum-ctor; final fallback: (name, value)
            if hasattr(Attr, "s"):
                attrs = [Attr.s("approximate", approx_str)]
            elif AttrType is not None:
                attrs = [Attr("approximate", AttrType.STRING, approx_str)]
            else:
                attrs = [Attr("approximate", approx_str)]

        gelu_node = ir.Node(
            op_type="Gelu",
            domain="",
            inputs=[x_val],
            outputs=[y_val],
            name=ctx.fresh_name("Gelu"),
            attributes=attrs,
        )
        ctx.add_node(gelu_node)

        # Re-assert for value_info readability
        _stamp_type_and_shape(y_val, y_meta)
        _add_value_info(ctx, y_val)

    # ---------- runtime impl (eager) ----------
    @staticmethod
    def _gelu(x, *, approximate=True):
        return GeluPlugin._PRIM.bind(x, approximate=bool(approximate))

    @classmethod
    def binding_specs(cls):
        return [
            # Export our private primitive under flax.nnx.gelu_p for compatibility
            AssignSpec("flax.nnx", "gelu_p", cls._PRIM, delete_if_missing=True),
            # Monkey-patch flax.nnx.gelu to bind our primitive while tracing
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="gelu",
                make_value=lambda orig: (
                    lambda x, approximate=True: cls._gelu(x, approximate=approximate)
                ),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, *, approximate=True: cls.abstract_eval(
                    x, approximate=approximate
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


@GeluPlugin._PRIM.def_impl
def _impl(x, *, approximate=True):
    # Eager fallback using JAX's gelu
    return jax.nn.gelu(x, approximate=bool(approximate))


def _approx_attr_equals(model, expected: str) -> bool:
    """Post-check helper: ensure there is a Gelu node with approximate == expected."""
    # Find first Gelu node
    n = next((n for n in model.graph.node if n.op_type == "Gelu"), None)
    if n is None:
        return False
    # Attribute may be encoded differently; try common layouts
    for a in getattr(n, "attribute", []):
        nm = getattr(a, "name", "")
        if nm != "approximate":
            continue
        # ONNX NodeProto: a.s is bytes in many builds
        s = getattr(a, "s", None)
        if s:
            try:
                sval = (
                    s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                )
                return sval == expected
            except Exception:
                pass
        # Some tools put string in .strings[0]
        strs = getattr(a, "strings", None)
        if strs and len(strs) > 0:
            try:
                sval = strs[0]
                sval = (
                    sval.decode("utf-8")
                    if isinstance(sval, (bytes, bytearray))
                    else str(sval)
                )
                return sval == expected
            except Exception:
                pass
    # If attribute list is missing, fail
    return False


# simple presence expectation
EXPECT_GELU: Callable[[Any], bool] = EG([("Gelu", {"counts": {"Gelu": 1}})])
