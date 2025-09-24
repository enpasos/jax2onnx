from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import onnx_ir as ir
from jax import core

from jax2onnx.plugins2._ir_shapes import _stamp_type_and_shape, _ensure_value_info
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_SHAPE_PRIM = make_jnp_primitive("jax.numpy.shape")


def _shape_eval(x):
    orig = getattr(_SHAPE_PRIM, "__orig_impl__shape", jnp.shape)
    result = jax.eval_shape(
        lambda arr: orig(arr), jax.ShapeDtypeStruct(x.shape, x.dtype)
    )
    return result


@register_primitive(
    jaxpr_primitive=_SHAPE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.shape.html",
    onnx=[
        {"component": "Shape", "doc": "https://onnx.ai/onnx/operators/onnx__Shape.html"}
    ],
    since="v0.9.0",
    context="primitives2.jnp",
    component="shape",
    testcases=[
        {
            "testcase": "shape_basic",
            "callable": lambda x: jnp.shape(x),
            "input_shapes": [(2, 3, 4)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "shape_dynamic",
            "callable": lambda x: jnp.shape(x),
            "input_shapes": [("B", 12, "T", "T")],
            "use_onnx_ir": True,
        },
    ],
)
class JnpShapePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SHAPE_PRIM
    _FUNC_NAME: ClassVar[str] = "shape"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x):
        result = _shape_eval(x)
        return core.ShapedArray(result.shape, result.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        (arr_var,) = eqn.invars
        (out_var,) = eqn.outvars

        arr_val = ctx.get_value_for_var(arr_var, name_hint=ctx.fresh_name("shape_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("shape_out"))

        ctx.add_node(
            ir.Node(
                op_type="Shape",
                domain="",
                inputs=[arr_val],
                outputs=[out_val],
                name=ctx.fresh_name("Shape"),
            )
        )

        target_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(out_val, target_shape)
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.shape not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a):
                arr = jnp.asarray(a)
                return cls._PRIM.bind(arr)

            return _patched

        return [
            AssignSpec(
                "jax.numpy", f"{cls._FUNC_NAME}_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@JnpShapePlugin._PRIM.def_impl
def _shape_impl(a):
    orig = get_orig_impl(JnpShapePlugin._PRIM, JnpShapePlugin._FUNC_NAME)
    return orig(a)


JnpShapePlugin._PRIM.def_abstract_eval(JnpShapePlugin.abstract_eval)
