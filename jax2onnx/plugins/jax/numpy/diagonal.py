# jax2onnx/plugins/jax/numpy/diagonal.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Final

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.jax.numpy.diag import (
    is_static_int_shape,
    lower_extract_diagonal_2d,
    normalize_k_scalar,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_DIAGONAL_PRIM: Final = make_jnp_primitive("jax.numpy.diagonal")


def _normalize_axis(axis: int, rank: int) -> int:
    axis_n = axis if axis >= 0 else axis + rank
    if axis_n < 0 or axis_n >= rank:
        raise ValueError(f"axis {axis} out of bounds for rank {rank}")
    return axis_n


def _abstract_eval_via_orig(
    prim: Any,
    func_name: str,
    x: core.AbstractValue,
    *,
    offset: int,
    axis1: int,
    axis2: int,
) -> core.ShapedArray:
    x_shape = tuple(getattr(x, "shape", ()))
    x_dtype: np.dtype[Any] = np.dtype(getattr(x, "dtype", np.float32))
    x_spec = jax.ShapeDtypeStruct(x_shape, x_dtype)
    orig = get_orig_impl(prim, func_name)
    out = jax.eval_shape(
        lambda a: orig(a, offset=offset, axis1=axis1, axis2=axis2),
        x_spec,
    )
    out_shape = tuple(getattr(out, "shape", ()))
    out_dtype = np.dtype(getattr(out, "dtype", x_dtype))
    return core.ShapedArray(out_shape, out_dtype)


@register_primitive(
    jaxpr_primitive=_DIAGONAL_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.diagonal.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="diagonal",
    testcases=[
        {
            "testcase": "jnp_diagonal_basic",
            "callable": lambda x: jnp.diagonal(x),
            "input_values": [np.arange(12, dtype=np.float32).reshape(3, 4)],
            "expected_output_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Reshape:12 -> Gather:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_diagonal_offset_neg1",
            "callable": lambda x: jnp.diagonal(x, offset=-1),
            "input_values": [np.arange(12, dtype=np.float32).reshape(4, 3)],
            "expected_output_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Reshape:12 -> Gather:3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpDiagonalPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _DIAGONAL_PRIM
    _FUNC_NAME: ClassVar[str] = "diagonal"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
    ) -> core.ShapedArray:
        return _abstract_eval_via_orig(
            JnpDiagonalPlugin._PRIM,
            JnpDiagonalPlugin._FUNC_NAME,
            x,
            offset=int(offset),
            axis1=int(axis1),
            axis2=int(axis2),
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        params = dict(getattr(eqn, "params", {}) or {})
        offset = normalize_k_scalar(params.get("offset", 0))
        axis1 = int(params.get("axis1", 0))
        axis2 = int(params.get("axis2", 1))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        if rank != 2:
            raise NotImplementedError("jnp.diagonal plugin currently supports rank-2")

        axis1_n = _normalize_axis(axis1, rank)
        axis2_n = _normalize_axis(axis2, rank)
        if axis1_n == axis2_n:
            raise ValueError("jnp.diagonal axis1 and axis2 must be different")
        if (axis1_n, axis2_n) != (0, 1):
            raise NotImplementedError(
                "jnp.diagonal plugin currently supports axis1=0 and axis2=1"
            )

        if not is_static_int_shape(x_shape):
            raise NotImplementedError("jnp.diagonal currently requires static shape")

        rows = int(x_shape[0])
        cols = int(x_shape[1])
        x_val = ctx.get_value_for_var(
            x_var, name_hint=ctx.fresh_name("jnp_diagonal_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_diagonal_out")
        )

        result = lower_extract_diagonal_2d(
            ctx,
            x_val,
            rows=rows,
            cols=cols,
            k=offset,
            out_spec=out_spec,
            output_hint="jnp_diagonal_out",
        )
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.diagonal not found for patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                offset: int = 0,
                axis1: int = 0,
                axis2: int = 1,
            ) -> jax.Array:
                arr = jnp.asarray(a)
                shape = tuple(arr.shape)
                rank = len(shape)

                try:
                    offset_norm = normalize_k_scalar(offset)
                    axis1_n = _normalize_axis(int(axis1), rank)
                    axis2_n = _normalize_axis(int(axis2), rank)
                except Exception:
                    return orig(a, offset=offset, axis1=axis1, axis2=axis2)

                if (
                    rank == 2
                    and is_static_int_shape(shape)
                    and (axis1_n, axis2_n) == (0, 1)
                ):
                    return cls._PRIM.bind(
                        arr,
                        offset=offset_norm,
                        axis1=axis1_n,
                        axis2=axis2_n,
                    )

                return orig(a, offset=offset, axis1=axis1, axis2=axis2)

            return _patched

        return [
            AssignSpec("jax.numpy", "diagonal_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr="diagonal",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@JnpDiagonalPlugin._PRIM.def_impl
def _diagonal_impl(
    a: ArrayLike,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
) -> jax.Array:
    orig = get_orig_impl(JnpDiagonalPlugin._PRIM, JnpDiagonalPlugin._FUNC_NAME)
    return orig(a, offset=offset, axis1=axis1, axis2=axis2)


register_jvp_via_jax_jvp(JnpDiagonalPlugin._PRIM, _diagonal_impl)


def _diagonal_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[Any, ...],
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
) -> tuple[jax.Array, Any]:
    (x,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        out = JnpDiagonalPlugin._PRIM.bind(
            x,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
        )
        return out, batching.not_mapped

    batch_size = x.shape[int(bdim)]
    x_front = batching.bdim_at_front(x, int(bdim), batch_size)
    orig = get_orig_impl(JnpDiagonalPlugin._PRIM, JnpDiagonalPlugin._FUNC_NAME)
    out = jax.vmap(lambda a: orig(a, offset=offset, axis1=axis1, axis2=axis2))(x_front)
    return out, 0


batching.primitive_batchers[JnpDiagonalPlugin._PRIM] = _diagonal_batch_rule
