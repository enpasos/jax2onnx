# jax2onnx/plugins/jax/nn/scaled_matmul.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, cast

import jax
from jax.extend.core import Primitive
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SCALED_MATMUL_PRIM: Final[Primitive] = Primitive("jax.nn.scaled_matmul")
_SCALED_MATMUL_PRIM.multiple_results = False
_JAX_SCALED_MATMUL_ORIG: Final = jax.nn.scaled_matmul


def _scaled_matmul_reference(
    lhs: ArrayLike,
    rhs: ArrayLike,
    lhs_scales: ArrayLike,
    rhs_scales: ArrayLike,
    *,
    preferred_element_type: np.dtype[Any] | type[Any] = np.float32,
) -> ArrayLike:
    a = jnp.asarray(lhs)
    b = jnp.asarray(rhs)
    a_scales = jnp.asarray(lhs_scales)
    b_scales = jnp.asarray(rhs_scales)

    if not all(x.ndim == 3 for x in (a, b, a_scales, b_scales)):
        raise ValueError("scaled_matmul requires all inputs to be 3-dimensional arrays")

    B_a, M_a, K_a = a.shape
    B_b, N_b, K_b = b.shape
    B_as, M_as, K_as = a_scales.shape
    B_bs, N_bs, K_bs = b_scales.shape

    if K_a != K_b or B_a != B_b:
        raise ValueError(
            "scaled_matmul requires inputs lhs/rhs to share batch and contract dimensions"
        )
    if K_as != K_bs or B_as != B_bs:
        raise ValueError(
            "scaled_matmul requires lhs_scales/rhs_scales to share batch and scale-K dimensions"
        )
    if M_as != M_a or N_bs != N_b:
        raise ValueError(
            "scaled_matmul requires scales to match lhs/rhs non-contracting dimensions"
        )
    if K_as == 0 or K_bs == 0 or K_a % K_as != 0 or K_b % K_bs != 0:
        raise ValueError("scaled_matmul requires K divisible by scale-K dimensions")

    a_block = K_a // K_as
    b_block = K_b // K_bs
    pref_dtype = jnp.dtype(preferred_element_type)
    a = a.astype(pref_dtype)
    b = b.astype(pref_dtype)
    a_scales = a_scales.astype(pref_dtype)
    b_scales = b_scales.astype(pref_dtype)

    a_scaled = a * jnp.repeat(a_scales, a_block, axis=-1)
    b_scaled = b * jnp.repeat(b_scales, b_block, axis=-1)
    return jax.lax.dot_general(
        a_scaled,
        b_scaled,
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
        preferred_element_type=preferred_element_type,
    )


def _shape3(arr: ArrayLike) -> tuple[int, int, int] | None:
    shape = tuple(getattr(arr, "shape", ()))
    if len(shape) != 3:
        return None
    if not all(isinstance(d, (int, np.integer)) for d in shape):
        return None
    return (int(shape[0]), int(shape[1]), int(shape[2]))


@register_primitive(
    jaxpr_primitive=_SCALED_MATMUL_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.scaled_matmul.html",
    onnx=[
        {
            "component": "Unsqueeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Unsqueeze.html",
        },
        {"component": "Tile", "doc": "https://onnx.ai/onnx/operators/onnx__Tile.html"},
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
    ],
    since="0.12.1",
    context="primitives.nn",
    component="scaled_matmul",
    testcases=[
        {
            "testcase": "jaxnn_scaled_matmul_basic",
            "callable": lambda lhs, rhs, lhs_scales, rhs_scales: _SCALED_MATMUL_PRIM.bind(
                lhs,
                rhs,
                lhs_scales,
                rhs_scales,
                preferred_element_type=np.float32,
            ),
            "input_shapes": [(1, 2, 4), (1, 3, 4), (1, 2, 2), (1, 3, 2)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["MatMul:1x2x3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ScaledMatmulPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.scaled_matmul`` as dequantize+batched-MatMul."""

    _PRIM: ClassVar[Primitive] = _SCALED_MATMUL_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        lhs: jax.core.AbstractValue,
        rhs: jax.core.AbstractValue,
        lhs_scales: jax.core.AbstractValue,
        rhs_scales: jax.core.AbstractValue,
        *,
        preferred_element_type: np.dtype[Any] | type[Any],
    ) -> jax.core.ShapedArray:
        lhs_spec = jax.ShapeDtypeStruct(lhs.shape, lhs.dtype)
        rhs_spec = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
        lhs_scales_spec = jax.ShapeDtypeStruct(lhs_scales.shape, lhs_scales.dtype)
        rhs_scales_spec = jax.ShapeDtypeStruct(rhs_scales.shape, rhs_scales.dtype)
        out = jax.eval_shape(
            lambda a, b, as_, bs_: _scaled_matmul_reference(
                a,
                b,
                as_,
                bs_,
                preferred_element_type=preferred_element_type,
            ),
            lhs_spec,
            rhs_spec,
            lhs_scales_spec,
            rhs_scales_spec,
        )
        return jax.core.ShapedArray(out.shape, out.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        lhs_var, rhs_var, lhs_scales_var, rhs_scales_var = eqn.invars
        out_var = eqn.outvars[0]
        preferred_element_type = eqn.params.get("preferred_element_type", np.float32)

        lhs_shape = tuple(getattr(getattr(lhs_var, "aval", None), "shape", ()))
        rhs_shape = tuple(getattr(getattr(rhs_var, "aval", None), "shape", ()))
        lhs_scales_shape = tuple(
            getattr(getattr(lhs_scales_var, "aval", None), "shape", ())
        )
        rhs_scales_shape = tuple(
            getattr(getattr(rhs_scales_var, "aval", None), "shape", ())
        )
        for shape in (lhs_shape, rhs_shape, lhs_scales_shape, rhs_scales_shape):
            if len(shape) != 3 or not all(
                isinstance(d, (int, np.integer)) for d in shape
            ):
                raise NotImplementedError(
                    "scaled_matmul lowering currently requires static 3D shapes"
                )

        B_a, M_a, K_a = (int(x) for x in lhs_shape)
        B_b, N_b, K_b = (int(x) for x in rhs_shape)
        B_as, M_as, K_as = (int(x) for x in lhs_scales_shape)
        B_bs, N_bs, K_bs = (int(x) for x in rhs_scales_shape)

        if K_a != K_b or B_a != B_b:
            raise ValueError(
                "scaled_matmul requires lhs/rhs to share batch and contract dimensions"
            )
        if K_as != K_bs or B_as != B_bs:
            raise ValueError(
                "scaled_matmul requires lhs_scales/rhs_scales to share batch and scale-K dimensions"
            )
        if M_as != M_a or N_bs != N_b:
            raise ValueError(
                "scaled_matmul scales must match lhs/rhs non-contracting dims"
            )
        if K_a % K_as != 0 or K_b % K_bs != 0:
            raise ValueError("scaled_matmul requires K divisible by scale-K dimensions")

        a_block = K_a // K_as
        b_block = K_b // K_bs

        target_dtype = np.dtype(preferred_element_type)
        target_dtype_enum = _dtype_to_ir(
            target_dtype, ctx.builder.enable_double_precision
        )
        if target_dtype_enum is None:
            raise TypeError(
                f"Unsupported preferred_element_type for scaled_matmul: {target_dtype}"
            )

        lhs_val = ctx.get_value_for_var(
            lhs_var, name_hint=ctx.fresh_name("scaled_mm_lhs")
        )
        rhs_val = ctx.get_value_for_var(
            rhs_var, name_hint=ctx.fresh_name("scaled_mm_rhs")
        )
        lhs_scales_val = ctx.get_value_for_var(
            lhs_scales_var, name_hint=ctx.fresh_name("scaled_mm_lhs_scales")
        )
        rhs_scales_val = ctx.get_value_for_var(
            rhs_scales_var, name_hint=ctx.fresh_name("scaled_mm_rhs_scales")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("scaled_mm_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "scaled_mm_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("scaled_mm_out")

        out_type = getattr(out_spec, "type", None) or ir.TensorType(target_dtype_enum)
        out_shape = getattr(out_spec, "shape", None) or tuple(
            getattr(getattr(out_var, "aval", None), "shape", ())
        )

        def _cast_to_target(
            val: ir.Value, name: str, shape: tuple[int, int, int]
        ) -> ir.Value:
            casted = cast(
                ir.Value,
                ctx.builder.Cast(
                    val,
                    _outputs=[ctx.fresh_name(name)],
                    to=int(target_dtype_enum.value),
                ),
            )
            casted.type = ir.TensorType(target_dtype_enum)
            _stamp_type_and_shape(casted, shape)
            return casted

        lhs_cast = _cast_to_target(lhs_val, "scaled_mm_lhs_cast", lhs_shape)
        rhs_cast = _cast_to_target(rhs_val, "scaled_mm_rhs_cast", rhs_shape)
        lhs_scales_cast = _cast_to_target(
            lhs_scales_val, "scaled_mm_lhs_scales_cast", lhs_scales_shape
        )
        rhs_scales_cast = _cast_to_target(
            rhs_scales_val, "scaled_mm_rhs_scales_cast", rhs_scales_shape
        )

        def _repeat_last_axis(
            scale_val: ir.Value,
            repeats: int,
            out_shape3: tuple[int, int, int],
            base: str,
        ) -> ir.Value:
            axes = _const_i64(ctx, np.asarray([3], dtype=np.int64), f"{base}_axes")
            unsqueezed = ctx.builder.Unsqueeze(
                scale_val,
                axes,
                _outputs=[ctx.fresh_name(f"{base}_unsqueeze")],
            )
            unsqueezed.type = ir.TensorType(target_dtype_enum)
            _stamp_type_and_shape(
                unsqueezed, tuple(out_shape3[:-1]) + (out_shape3[-1] // repeats, 1)
            )

            repeats_val = _const_i64(
                ctx,
                np.asarray([1, 1, 1, repeats], dtype=np.int64),
                f"{base}_tile_repeats",
            )
            tiled = ctx.builder.Tile(
                unsqueezed,
                repeats_val,
                _outputs=[ctx.fresh_name(f"{base}_tiled")],
            )
            tiled.type = ir.TensorType(target_dtype_enum)
            _stamp_type_and_shape(
                tiled,
                tuple(out_shape3[:-1]) + (out_shape3[-1] // repeats, repeats),
            )

            reshape_shape = _const_i64(
                ctx,
                np.asarray(list(out_shape3), dtype=np.int64),
                f"{base}_reshape_shape",
            )
            reshaped = cast(
                ir.Value,
                ctx.builder.Reshape(
                    tiled,
                    reshape_shape,
                    _outputs=[ctx.fresh_name(f"{base}_expanded")],
                ),
            )
            reshaped.type = ir.TensorType(target_dtype_enum)
            _stamp_type_and_shape(reshaped, out_shape3)
            return reshaped

        lhs_scales_expanded = _repeat_last_axis(
            lhs_scales_cast,
            repeats=a_block,
            out_shape3=(B_a, M_a, K_a),
            base="scaled_mm_lhs_scales",
        )
        rhs_scales_expanded = _repeat_last_axis(
            rhs_scales_cast,
            repeats=b_block,
            out_shape3=(B_b, N_b, K_b),
            base="scaled_mm_rhs_scales",
        )

        lhs_scaled = ctx.builder.Mul(
            lhs_cast,
            lhs_scales_expanded,
            _outputs=[ctx.fresh_name("scaled_mm_lhs_scaled")],
        )
        lhs_scaled.type = ir.TensorType(target_dtype_enum)
        _stamp_type_and_shape(lhs_scaled, (B_a, M_a, K_a))

        rhs_scaled = ctx.builder.Mul(
            rhs_cast,
            rhs_scales_expanded,
            _outputs=[ctx.fresh_name("scaled_mm_rhs_scaled")],
        )
        rhs_scaled.type = ir.TensorType(target_dtype_enum)
        _stamp_type_and_shape(rhs_scaled, (B_b, N_b, K_b))

        rhs_transposed = ctx.builder.Transpose(
            rhs_scaled,
            _outputs=[ctx.fresh_name("scaled_mm_rhs_t")],
            perm=[0, 2, 1],
        )
        rhs_transposed.type = ir.TensorType(target_dtype_enum)
        _stamp_type_and_shape(rhs_transposed, (B_b, K_b, N_b))

        result = ctx.builder.MatMul(
            lhs_scaled,
            rhs_transposed,
            _outputs=[desired_name],
        )
        result.type = out_type
        if out_shape is not None:
            _stamp_type_and_shape(result, out_shape)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jax.nn.scaled_matmul not found")

            def _patched(
                lhs: ArrayLike,
                rhs: ArrayLike,
                lhs_scales: ArrayLike,
                rhs_scales: ArrayLike,
                preferred_element_type: np.dtype[Any] | type[Any] = np.float32,
            ) -> ArrayLike:
                lhs_arr = jnp.asarray(lhs)
                rhs_arr = jnp.asarray(rhs)
                lhs_scales_arr = jnp.asarray(lhs_scales)
                rhs_scales_arr = jnp.asarray(rhs_scales)

                lhs_s = _shape3(lhs_arr)
                rhs_s = _shape3(rhs_arr)
                lhs_scales_s = _shape3(lhs_scales_arr)
                rhs_scales_s = _shape3(rhs_scales_arr)
                if (
                    lhs_s is None
                    or rhs_s is None
                    or lhs_scales_s is None
                    or rhs_scales_s is None
                ):
                    return orig(
                        lhs_arr,
                        rhs_arr,
                        lhs_scales=lhs_scales_arr,
                        rhs_scales=rhs_scales_arr,
                        preferred_element_type=preferred_element_type,
                    )

                B_a, M_a, K_a = lhs_s
                B_b, N_b, K_b = rhs_s
                B_as, M_as, K_as = lhs_scales_s
                B_bs, N_bs, K_bs = rhs_scales_s
                if K_a != K_b or B_a != B_b:
                    return orig(
                        lhs_arr,
                        rhs_arr,
                        lhs_scales=lhs_scales_arr,
                        rhs_scales=rhs_scales_arr,
                        preferred_element_type=preferred_element_type,
                    )
                if K_as != K_bs or B_as != B_bs:
                    return orig(
                        lhs_arr,
                        rhs_arr,
                        lhs_scales=lhs_scales_arr,
                        rhs_scales=rhs_scales_arr,
                        preferred_element_type=preferred_element_type,
                    )
                if M_as != M_a or N_bs != N_b:
                    return orig(
                        lhs_arr,
                        rhs_arr,
                        lhs_scales=lhs_scales_arr,
                        rhs_scales=rhs_scales_arr,
                        preferred_element_type=preferred_element_type,
                    )
                if K_as == 0 or K_bs == 0 or K_a % K_as != 0 or K_b % K_bs != 0:
                    return orig(
                        lhs_arr,
                        rhs_arr,
                        lhs_scales=lhs_scales_arr,
                        rhs_scales=rhs_scales_arr,
                        preferred_element_type=preferred_element_type,
                    )

                try:
                    np.dtype(preferred_element_type)
                except Exception:
                    return orig(
                        lhs_arr,
                        rhs_arr,
                        lhs_scales=lhs_scales_arr,
                        rhs_scales=rhs_scales_arr,
                        preferred_element_type=preferred_element_type,
                    )

                return cls._PRIM.bind(
                    lhs_arr,
                    rhs_arr,
                    lhs_scales_arr,
                    rhs_scales_arr,
                    preferred_element_type=preferred_element_type,
                )

            return _patched

        return [
            AssignSpec("jax.nn", "scaled_matmul_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="scaled_matmul",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@ScaledMatmulPlugin._PRIM.def_impl
def _scaled_matmul_impl(
    lhs: ArrayLike,
    rhs: ArrayLike,
    lhs_scales: ArrayLike,
    rhs_scales: ArrayLike,
    *,
    preferred_element_type: np.dtype[Any] | type[Any] = np.float32,
) -> ArrayLike:
    return _scaled_matmul_reference(
        lhs,
        rhs,
        lhs_scales,
        rhs_scales,
        preferred_element_type=preferred_element_type,
    )


register_jvp_via_jax_jvp(ScaledMatmulPlugin._PRIM, _scaled_matmul_impl)
