# jax2onnx/plugins/jax/numpy/matmul.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import jax
import jax.numpy as jnp
import onnx_ir as ir
from jax import core

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_MATMUL_PRIM: Final = make_jnp_primitive("jax.numpy.matmul")


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
    since="v0.1.0",
    context="primitives.jnp",
    component="matmul",
    testcases=[
        {
            "testcase": "matmul_1d",
            "callable": lambda a, b: jnp.matmul(a, b),
            "input_shapes": [(4,), (4,)],
            "post_check_onnx_graph": EG(
                ["MatMul"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "matmul_1d_2d",
            "callable": lambda a, b: jnp.matmul(a, b),
            "input_shapes": [(4,), (4, 5)],
            "post_check_onnx_graph": EG(
                ["MatMul:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "matmul_2d",
            "callable": lambda a, b: jnp.matmul(a, b),
            "input_shapes": [(3, 4), (4, 5)],
            "post_check_onnx_graph": EG(
                ["MatMul:3x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "matmul_2d_1d",
            "callable": lambda a, b: jnp.matmul(a, b),
            "input_shapes": [(3, 4), (4,)],
            "post_check_onnx_graph": EG(
                ["MatMul:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "matmul_3d",
            "callable": lambda a, b: jnp.matmul(a, b),
            "input_shapes": [(2, 3, 4), (2, 4, 5)],
            "post_check_onnx_graph": EG(
                ["MatMul:2x3x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "matmul_dynamic",
            "callable": lambda a, b: jnp.matmul(a, b),
            "input_shapes": [("B", 3, 4), ("B", 4, 5)],
            "post_check_onnx_graph": EG(
                ["MatMul:Bx3x5"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "matmul_dynamic_a",
            "callable": lambda a, b: jnp.matmul(a, b),
            "input_shapes": [("B", 3), (3, 4)],
            "post_check_onnx_graph": EG(
                ["MatMul:Bx4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
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
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("matmul_out")
        )
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for matmul lowering")

        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("MatMul")
        result = builder.MatMul(
            a_val,
            b_val,
            _outputs=[out_name],
        )

        spec_type = getattr(out_spec, "type", None)
        if spec_type is not None:
            result.type = spec_type
        else:
            a_dtype = getattr(getattr(a_val, "type", None), "dtype", None)
            if a_dtype is not None:
                result.type = ir.TensorType(a_dtype)

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(out_var, result)

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
