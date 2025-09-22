from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import onnx_ir as ir
from jax import core

from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_MATMUL_PRIM = make_jnp_primitive("jax.numpy.matmul")


def _matmul_shape(a_shape, b_shape, a_dtype):
    spec_a = jax.ShapeDtypeStruct(a_shape, a_dtype)
    # Assume dtype broadcast already handled; use same dtype for b
    spec_b = jax.ShapeDtypeStruct(b_shape, a_dtype)
    orig = getattr(_MATMUL_PRIM, "__orig_impl__matmul", jnp.matmul)
    result = jax.eval_shape(lambda x, y: orig(x, y), spec_a, spec_b)
    return result.shape, result.dtype


@register_primitive(
    jaxpr_primitive=_MATMUL_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matmul.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        }
    ],
    since="v0.9.0",
    context="primitives2.jnp",
    component="matmul",
    testcases=[
        {
            "testcase": "matmul_basic",
            "callable": lambda a, b: jnp.matmul(a, b),
            "input_shapes": [(3, 4), (4, 5)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "matmul_vector_matrix",
            "callable": lambda a, b: jnp.matmul(a, b),
            "input_shapes": [(4,), (4, 5)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "matmul_batch",
            "callable": lambda a, b: jnp.matmul(a, b),
            "input_shapes": [("B", 6, 4), ("B", 4, 3)],
            "use_onnx_ir": True,
        },
    ],
)
class JnpMatmulPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _MATMUL_PRIM
    _FUNC_NAME: ClassVar[str] = "matmul"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(a, b):
        shape, dtype = _matmul_shape(a.shape, b.shape, a.dtype)
        return core.ShapedArray(shape, dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        a_var, b_var = eqn.invars
        out_var = eqn.outvars[0]

        a_val = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("matmul_a"))
        b_val = ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("matmul_b"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("matmul_out"))

        ctx.add_node(
            ir.Node(
                op_type="MatMul",
                domain="",
                inputs=[a_val, b_val],
                outputs=[out_val],
                name=ctx.fresh_name("MatMul"),
            )
        )

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(out_val, out_shape)
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.matmul not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a, b):
                return cls._PRIM.bind(a, b)

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


@JnpMatmulPlugin._PRIM.def_impl
def _matmul_impl(a, b):
    orig = get_orig_impl(JnpMatmulPlugin._PRIM, JnpMatmulPlugin._FUNC_NAME)
    return orig(a, b)


JnpMatmulPlugin._PRIM.def_abstract_eval(JnpMatmulPlugin.abstract_eval)
