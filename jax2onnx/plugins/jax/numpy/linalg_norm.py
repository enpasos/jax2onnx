# jax2onnx/plugins/jax/numpy/linalg_norm.py

from __future__ import annotations

from collections.abc import Sequence as _Seq
from typing import Callable, ClassVar, Final

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import (
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

_LINALG_NORM_PRIM: Final = make_jnp_primitive("jax.numpy.linalg.norm")


@register_primitive(
    jaxpr_primitive=_LINALG_NORM_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.norm.html",
    onnx=[
        {
            "component": "GlobalLpPool",
            "doc": "https://onnx.ai/onnx/operators/onnx__GlobalLpPool.html",
        },
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_norm",
    testcases=[
        {
            "testcase": "linalg_norm_global_fro",
            "callable": lambda x: jnp.linalg.norm(
                x, ord="fro", axis=(1, 2), keepdims=True
            ),
            "input_shapes": [(2, 8, 8, 3)],
            "expected_output_shapes": [(2, 1, 1, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Transpose:2x3x8x8 -> GlobalLpPool:2x3x1x1 -> Transpose:2x1x1x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linalg_norm_global_default",
            "callable": lambda x: jnp.linalg.norm(x, axis=(1, 2), keepdims=True),
            "input_shapes": [(2, 8, 8, 3)],
            "expected_output_shapes": [(2, 1, 1, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Transpose:2x3x8x8 -> GlobalLpPool:2x3x1x1 -> Transpose:2x1x1x3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpLinalgNormPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _LINALG_NORM_PRIM
    _FUNC_NAME: ClassVar[str] = "norm"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def _normalize_axes(
        axes: tuple[int, ...] | None,
        rank: int,
    ) -> tuple[int, ...]:
        if axes is None:
            return tuple(range(rank))
        normalized: list[int] = []
        for axis in axes:
            axis_int = int(axis)
            if axis_int < 0:
                axis_int += rank
            normalized.append(axis_int)
        return tuple(normalized)

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        axes: tuple[int, ...],
        ord_value: int,
        keepdims: bool,
    ) -> core.ShapedArray:
        storage_slot = f"__orig_impl__{JnpLinalgNormPlugin._FUNC_NAME}"
        orig = getattr(_LINALG_NORM_PRIM, storage_slot, jnp.linalg.norm)
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        axis_arg: int | tuple[int, ...]
        axis_arg = axes[0] if len(axes) == 1 else tuple(axes)
        result = jax.eval_shape(
            lambda arr: orig(arr, ord="fro", axis=axis_arg, keepdims=keepdims),
            spec,
        )
        return core.ShapedArray(result.shape, result.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        params = getattr(eqn, "params", {})
        axes = tuple(int(a) for a in params.get("axes", ()))
        keepdims = bool(params.get("keepdims", False))
        ord_value = int(params.get("ord_value", 2))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        normalized_axes = self._normalize_axes(axes, rank)
        expected_spatial_axes = tuple(range(1, rank - 1))
        if (
            rank < 3
            or not keepdims
            or ord_value != 2
            or normalized_axes != expected_spatial_axes
        ):
            raise NotImplementedError(
                "GlobalLpPool lowering supports rank>=3 with axis=all spatial "
                "dimensions, ord='fro'/default, and keepdims=True."
            )

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("norm_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("norm_out"))
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for linalg.norm")

        perm = [0, rank - 1] + list(range(1, rank - 1))
        inv_perm = [perm.index(i) for i in range(rank)]

        def _label(idx: int) -> object:
            return _dim_label_from_value_or_aval(x_val, x_shape, idx)

        nchw_in_dims = (
            _label(0),
            _label(rank - 1),
            *[_label(i) for i in range(1, rank - 1)],
        )
        n_label = _label(0)
        c_label = _label(rank - 1)
        nchw_out_dims = (n_label, c_label, *([1] * (rank - 2)))

        transposed = builder.Transpose(
            x_val,
            _outputs=[ctx.fresh_name("norm_nchw_in")],
            perm=tuple(perm),
        )
        if getattr(x_val, "type", None) is not None:
            transposed.type = x_val.type
        _stamp_type_and_shape(
            transposed, tuple(_to_ir_dim_for_shape(d) for d in nchw_in_dims)
        )
        _ensure_value_metadata(ctx, transposed)

        pooled = builder.GlobalLpPool(
            transposed,
            p=ord_value,
            _outputs=[ctx.fresh_name("GlobalLpPool")],
        )
        dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        if dtype is not None:
            pooled.type = ir.TensorType(dtype)
        _stamp_type_and_shape(
            pooled, tuple(_to_ir_dim_for_shape(d) for d in nchw_out_dims)
        )
        _ensure_value_metadata(ctx, pooled)

        final = builder.Transpose(
            pooled,
            _outputs=[getattr(out_spec, "name", None) or ctx.fresh_name("norm_out")],
            perm=tuple(inv_perm),
        )
        spec_type = getattr(out_spec, "type", None)
        if spec_type is not None:
            final.type = spec_type
        elif dtype is not None:
            final.type = ir.TensorType(dtype)
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))
        _stamp_type_and_shape(final, out_shape)
        _ensure_value_metadata(ctx, final)
        ctx.bind_value_for_var(out_var, final)

    @classmethod
    def binding_specs(cls) -> list[MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError(
                    "Original jnp.linalg.norm not found for monkey patching"
                )
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                x: ArrayLike,
                ord: int | float | None = None,
                axis: int | _Seq[int] | None = None,
                keepdims: bool = False,
            ) -> jax.Array:
                rank = getattr(x, "ndim", None)
                if not isinstance(rank, int) or rank < 3 or not keepdims:
                    return orig(x, ord=ord, axis=axis, keepdims=keepdims)

                if isinstance(axis, _Seq) and not isinstance(axis, (str, bytes)):
                    axes = tuple(int(a) for a in axis)
                else:
                    return orig(x, ord=ord, axis=axis, keepdims=keepdims)
                axes = tuple(a + rank if a < 0 else a for a in axes)

                if not (
                    ord is None or (isinstance(ord, str) and ord.lower() == "fro")
                ) or axes != tuple(range(1, rank - 1)):
                    return orig(x, ord=ord, axis=axis, keepdims=keepdims)

                return cls._PRIM.bind(
                    jnp.asarray(x),
                    axes=axes,
                    ord_value=2,
                    keepdims=True,
                )

            return _patched

        return [
            MonkeyPatchSpec(
                target="jax.numpy.linalg",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            )
        ]


@JnpLinalgNormPlugin._PRIM.def_impl
def _norm_impl(
    x: ArrayLike,
    *,
    axes: tuple[int, ...],
    ord_value: int,
    keepdims: bool,
) -> jax.Array:
    orig = get_orig_impl(JnpLinalgNormPlugin._PRIM, JnpLinalgNormPlugin._FUNC_NAME)
    axis_arg: int | tuple[int, ...]
    axis_arg = axes[0] if len(axes) == 1 else tuple(axes)
    return orig(jnp.asarray(x), ord="fro", axis=axis_arg, keepdims=keepdims)


BatchDim = int | type(batching.not_mapped)


def _norm_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    axes: tuple[int, ...],
    ord_value: int,
    keepdims: bool,
) -> tuple[jax.Array, BatchDim]:
    (operand,), (bdim,) = batched_args, batch_dims
    if bdim is batching.not_mapped:
        out = JnpLinalgNormPlugin._PRIM.bind(
            operand,
            axes=axes,
            ord_value=ord_value,
            keepdims=keepdims,
        )
        return out, batching.not_mapped

    axis_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, axis_size)
    shifted_axes = tuple(int(a) + 1 for a in axes)
    out = JnpLinalgNormPlugin._PRIM.bind(
        operand,
        axes=shifted_axes,
        ord_value=ord_value,
        keepdims=keepdims,
    )
    return out, 0


batching.primitive_batchers[JnpLinalgNormPlugin._PRIM] = _norm_batch_rule
