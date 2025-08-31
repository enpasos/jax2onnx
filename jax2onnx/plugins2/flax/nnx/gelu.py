# file: jax2onnx/plugins2/flax/nnx/gelu.py
from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Callable

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2._ir_shapes import (
    _stamp_type_and_shape,
    _dim_label_from_value_or_aval,
    is_shape_all_unknown,
    _ensure_value_info as _add_value_info,
)

if TYPE_CHECKING:
    from jax2onnx.converter2.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


@register_primitive(
    jaxpr_primitive="nnx.gelu",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.gelu",
    onnx=[{"component": "Gelu", "doc": "https://onnx.ai/onnx/operators/onnx__Gelu.html"}],
    since="v0.1.0",
    context="primitives2.nnx",
    component="gelu",
    testcases=[
        {
            "testcase": "gelu",
            "callable": lambda x: nnx.gelu(x, approximate=False),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "gelu_1",
            "callable": lambda x: nnx.gelu(x, approximate=False),
            "input_shapes": [(1, 10)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "gelu_2",
            "callable": lambda x: nnx.gelu(x, approximate=True),
            "input_shapes": [(1,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "gelu_3",
            "callable": lambda x: nnx.gelu(x, approximate=True),
            "input_shapes": [("B", 10)],
            "use_onnx_ir": True,
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
        y_meta = tuple(_dim_label_from_value_or_aval(x_val, x_shape, i) for i in range(len(x_shape)))
        _stamp_type_and_shape(y_val, y_meta)

        # ONNX Gelu node (attribute: 'approximate' in {'tanh','none'})
        approx_str = "tanh" if approximate else "none"
        ctx.add_node(
            ir.Node(
                op_type="Gelu",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Gelu"),
                attributes=[ir.Attr("approximate", ir.AttributeType.STRING, approx_str)],
            )
        )

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
                make_value=lambda orig: (lambda x, approximate=True: cls._gelu(x, approximate=approximate)),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(lambda x, *, approximate=True: cls.abstract_eval(x, approximate=approximate))
            cls._ABSTRACT_EVAL_BOUND = True


@GeluPlugin._PRIM.def_impl
def _impl(x, *, approximate=True):
    # Eager fallback using JAX's gelu
    return jax.nn.gelu(x, approximate=bool(approximate))
