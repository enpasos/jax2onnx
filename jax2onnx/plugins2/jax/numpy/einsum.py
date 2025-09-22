from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import onnx_ir as ir
from jax import core
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_EINSUM_PRIM = make_jnp_primitive("jax.numpy.einsum")


def _einsum_shape(avals, equation: str):
    specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in avals]
    orig = getattr(_EINSUM_PRIM, "__orig_impl__einsum", jnp.einsum)
    result = jax.eval_shape(lambda *args: orig(equation, *args), *specs)
    return result.shape, result.dtype


@register_primitive(
    jaxpr_primitive=_EINSUM_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.einsum.html",
    onnx=[
        {
            "component": "Einsum",
            "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
        }
    ],
    since="v0.9.0",
    context="primitives2.jnp",
    component="einsum",
    testcases=[
        {
            "testcase": "einsum_vecdot",
            "callable": lambda x, y: jnp.einsum("i,i->", x, y),
            "input_shapes": [(5,), (5,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "einsum_matmul",
            "callable": lambda x, y: jnp.einsum("ij,jk->ik", x, y),
            "input_shapes": [(3, 5), (5, 2)],
            "use_onnx_ir": True,
        },
    ],
)
class JnpEinsumPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _EINSUM_PRIM
    _FUNC_NAME: ClassVar[str] = "einsum"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(*avals, equation: str):
        shape, dtype = _einsum_shape(avals, equation)
        return core.ShapedArray(shape, dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        params = getattr(eqn, "params", {})
        equation = params["equation"]

        inputs = [
            ctx.get_value_for_var(var, name_hint=ctx.fresh_name("einsum_in"))
            for var in eqn.invars
        ]
        out_var = eqn.outvars[0]
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("einsum_out"))

        ctx.add_node(
            ir.Node(
                op_type="Einsum",
                domain="",
                inputs=inputs,
                outputs=[out_val],
                name=ctx.fresh_name("Einsum"),
                attributes=[IRAttr("equation", IRAttrType.STRING, equation)],
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
                raise RuntimeError("Original jnp.einsum not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(equation, *operands, **kwargs):
                if kwargs:
                    raise NotImplementedError(
                        f"Unsupported kwargs for jnp.einsum: {tuple(kwargs.keys())}"
                    )
                return cls._PRIM.bind(*operands, equation=str(equation))

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


@JnpEinsumPlugin._PRIM.def_impl
def _einsum_impl(equation, *operands):
    orig = get_orig_impl(JnpEinsumPlugin._PRIM, JnpEinsumPlugin._FUNC_NAME)
    return orig(equation, *operands)


JnpEinsumPlugin._PRIM.def_abstract_eval(JnpEinsumPlugin.abstract_eval)
