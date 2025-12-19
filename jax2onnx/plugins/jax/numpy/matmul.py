# jax2onnx/plugins/jax/numpy/matmul.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
import jax.extend.core as jax_core_ext
import jax.numpy as jnp
import onnx_ir as ir
from jax import core
from jax.interpreters import batching
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.plugins._complex_utils import (
    COMPLEX_DTYPES,
    cast_real_tensor,
    coerce_dim_values,
    ensure_packed_real_pair,
    is_packed_complex_tensor,
    pack_real_imag_pair,
    resolve_common_real_dtype,
    split_packed_real_imag,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


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
        {
            "testcase": "matmul_complex64",
            "callable": lambda a, b: jnp.matmul(a, b),
            "input_values": [
                np.array(
                    [[1.0 + 2.0j, -0.5 + 0.25j], [1.5 - 1.0j, 0.25 + 0.75j]],
                    dtype=np.complex64,
                ),
                np.array(
                    [[0.75 - 0.5j, -1.0 + 1.5j], [2.0 + 0.5j, -0.25 - 1.0j]],
                    dtype=np.complex64,
                ),
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [{"path": "MatMul", "counts": {"MatMul": 4}}],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "matmul_vmap_batching",
            "callable": lambda a, b: jax.vmap(jnp.matmul)(a, b),
            "input_shapes": [(3, 2, 4), (3, 4, 5)],
        },
    ],
)
class JnpMatmulPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _MATMUL_PRIM
    _FUNC_NAME: ClassVar[str] = "matmul"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(a: core.AbstractValue, b: core.AbstractValue) -> core.ShapedArray:
        shape, dtype = _matmul_shape(a.shape, b.shape, a.dtype)
        return core.ShapedArray(shape, dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        a_var, b_var = eqn.invars
        out_var = eqn.outvars[0]

        a_val = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("matmul_a"))
        b_val = ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("matmul_b"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("matmul_out")
        )
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for matmul lowering")

        if self._maybe_lower_complex(
            ctx,
            a_var,
            b_var,
            out_var,
            a_val,
            b_val,
            out_spec,
            out_shape,
        ):
            return

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

    def _maybe_lower_complex(
        self,
        ctx: LoweringContextProtocol,
        a_var: jax_core_ext.Var,
        b_var: jax_core_ext.Var,
        out_var: jax_core_ext.Var,
        a_val: ir.Value,
        b_val: ir.Value,
        out_spec: ir.Value,
        out_shape: tuple[int, ...],
    ) -> bool:
        def _is_complex(var) -> bool:
            aval_dtype = getattr(getattr(var, "aval", None), "dtype", None)
            if aval_dtype is None:
                return False
            try:
                return np.issubdtype(np.dtype(aval_dtype), np.complexfloating)
            except TypeError:
                return False

        complex_var_hint = (
            _is_complex(out_var) or _is_complex(a_var) or _is_complex(b_var)
        )
        a_dtype = getattr(a_val, "dtype", None)
        b_dtype = getattr(b_val, "dtype", None)
        complex_dtype_hint = a_dtype in COMPLEX_DTYPES or b_dtype in COMPLEX_DTYPES
        packed_hint = False
        if complex_var_hint or complex_dtype_hint:
            packed_hint = is_packed_complex_tensor(a_val) or is_packed_complex_tensor(
                b_val
            )
        if not (complex_var_hint or complex_dtype_hint or packed_hint):
            return False

        a_packed, a_base = ensure_packed_real_pair(
            ctx, a_val, name_hint="matmul_a_pack"
        )
        b_packed, b_base = ensure_packed_real_pair(
            ctx, b_val, name_hint="matmul_b_pack"
        )
        target_dtype = resolve_common_real_dtype(a_base, b_base)

        a_ready = (
            a_packed
            if a_packed.dtype == target_dtype
            else cast_real_tensor(
                ctx, a_packed, target_dtype, name_hint="matmul_a_cast"
            )
        )
        b_ready = (
            b_packed
            if b_packed.dtype == target_dtype
            else cast_real_tensor(
                ctx, b_packed, target_dtype, name_hint="matmul_b_cast"
            )
        )

        a_real, a_imag = split_packed_real_imag(
            ctx, a_ready, target_dtype, prefix="matmul_a"
        )
        b_real, b_imag = split_packed_real_imag(
            ctx, b_ready, target_dtype, prefix="matmul_b"
        )

        result_dims = coerce_dim_values(out_shape)

        def _matmul(lhs: ir.Value, rhs: ir.Value, name: str) -> ir.Value:
            value = ctx.builder.MatMul(lhs, rhs, _outputs=[ctx.fresh_name(name)])
            value.type = ir.TensorType(target_dtype)
            value.dtype = target_dtype
            _stamp_type_and_shape(value, result_dims)
            _ensure_value_metadata(ctx, value)
            return value

        ar_br = _matmul(a_real, b_real, "matmul_ar_br")
        ai_bi = _matmul(a_imag, b_imag, "matmul_ai_bi")
        ar_bi = _matmul(a_real, b_imag, "matmul_ar_bi")
        ai_br = _matmul(a_imag, b_real, "matmul_ai_br")

        real_part = ctx.builder.Sub(
            ar_br,
            ai_bi,
            _outputs=[ctx.fresh_name("matmul_real_part")],
        )
        real_part.type = ir.TensorType(target_dtype)
        real_part.dtype = target_dtype
        _stamp_type_and_shape(real_part, result_dims)
        _ensure_value_metadata(ctx, real_part)

        imag_part = ctx.builder.Add(
            ar_bi,
            ai_br,
            _outputs=[ctx.fresh_name("matmul_imag_part")],
        )
        imag_part.type = ir.TensorType(target_dtype)
        imag_part.dtype = target_dtype
        _stamp_type_and_shape(imag_part, result_dims)
        _ensure_value_metadata(ctx, imag_part)

        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("MatMul")
        packed = pack_real_imag_pair(
            ctx,
            real_part,
            imag_part,
            target_dtype,
            name_hint="matmul_output",
            output_name=out_name,
        )

        out_spec.type = ir.TensorType(target_dtype)
        out_spec.dtype = target_dtype
        if getattr(packed, "shape", None) is not None:
            out_spec.shape = packed.shape
        _ensure_value_metadata(ctx, packed)
        ctx.bind_value_for_var(out_var, packed)
        return True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jnp.matmul not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a: ArrayLike, b: ArrayLike) -> ArrayLike:
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
def _matmul_impl(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    orig = get_orig_impl(JnpMatmulPlugin._PRIM, JnpMatmulPlugin._FUNC_NAME)
    return orig(a, b)


JnpMatmulPlugin._PRIM.def_abstract_eval(JnpMatmulPlugin.abstract_eval)


def _matmul_batch_rule(args, dims, **params):
    return broadcast_batcher_compat(JnpMatmulPlugin._PRIM, args, dims, **params)


batching.primitive_batchers[JnpMatmulPlugin._PRIM] = _matmul_batch_rule
