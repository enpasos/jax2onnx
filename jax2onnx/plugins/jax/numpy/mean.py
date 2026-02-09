# jax2onnx/plugins/jax/numpy/mean.py

from __future__ import annotations

from collections.abc import Sequence as _Seq
from typing import Any, Callable, ClassVar, Final, Sequence

import numpy as np
from numpy.typing import ArrayLike

import jax
from jax import core
import jax.numpy as jnp
from jax.interpreters import batching
from onnx_ir import Attr, AttributeType

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    make_jnp_primitive,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_MEAN_PRIM: Final = make_jnp_primitive("jax.numpy.mean")


@register_primitive(
    jaxpr_primitive=_MEAN_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.mean.html",
    onnx=[
        {
            "component": "ReduceMean",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMean.html",
        }
    ],
    since="0.12.0",
    context="primitives.jnp",
    component="mean",
    testcases=[
        {
            "testcase": "basic_mean",
            "callable": lambda: jnp.mean(np.arange(12, dtype=np.float32).reshape(3, 4)),
            "input_values": [],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["ReduceMean"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "mean_with_axis",
            "callable": lambda: jnp.mean(
                np.arange(60, dtype=np.float32).reshape(3, 4, 5), axis=1
            ),
            "input_values": [],
            "expected_output_shapes": [(3, 5)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}},
                        "path": "ReduceMean:3x5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "mean_with_keepdims",
            "callable": lambda: jnp.mean(
                np.arange(12, dtype=np.float32).reshape(3, 4), axis=0, keepdims=True
            ),
            "input_values": [],
            "expected_output_shapes": [(1, 4)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "ReduceMean:1x4",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_mean_basic",
            "callable": lambda x: jnp.mean(x),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceMean"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_mean_axis",
            "callable": lambda x: jnp.mean(x, axis=1),
            "input_shapes": [(3, 4, 5)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}},
                        "path": "ReduceMean:3x5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpMeanPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _MEAN_PRIM
    _FUNC_NAME: ClassVar[str] = "mean"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        axes: Sequence[int] | None = None,
        dtype: np.dtype[Any] | type | None = None,
        out: None = None,
        keepdims: bool = False,
        where: bool = True,
    ) -> core.ShapedArray:
        if out is not None:
            raise NotImplementedError(
                "jnp.mean with 'out' is not supported in ONNX lowering"
            )
        if where is not True:
            raise NotImplementedError(
                "jnp.mean with 'where' mask is not supported in ONNX lowering"
            )

        print(f"DEBUG: abstract_eval input x: {x}, shape: {x.shape}")

        ndim = len(x.shape)
        # JAX mean default dtype logic might differ slightly but usually follows input or float
        # match jnp behavior: integer inputs -> float
        if dtype is None:
            if jnp.issubdtype(x.dtype, np.integer) or jnp.issubdtype(x.dtype, np.bool_):
                out_dtype = np.dtype(
                    np.float32
                )  # Default typically float32 or 64 depending on jax config
            else:
                out_dtype = np.dtype(x.dtype)
        else:
            out_dtype = np.dtype(dtype)

        if axes is None:
            axes_tuple = tuple(range(ndim))
        else:
            axes_tuple = tuple(
                int(ax) % ndim if ndim and int(ax) < 0 else int(ax) for ax in axes
            )

        if axes is None:
            if keepdims:
                out_shape = tuple(1 for _ in range(ndim))
            else:
                out_shape = ()
        else:
            if keepdims:
                out_shape = tuple(
                    1 if i in axes_tuple else dim for i, dim in enumerate(x.shape)
                )
            else:
                out_shape = tuple(
                    dim for i, dim in enumerate(x.shape) if i not in axes_tuple
                )

        return jax.core.ShapedArray(out_shape, out_dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        # Manual lowering to avoid potential issues in shared utils
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        operand_val = ctx.get_value_for_var(operand_var)

        params = getattr(eqn, "params", {})
        axes = params.get("axes")
        keepdims = bool(params.get("keepdims", False))

        # Resolve axes
        rank = len(getattr(operand_var.aval, "shape", ()))
        # Handle axis normalization locally
        if axes is not None:
            normalized = []
            for ax in axes:
                ax_int = int(ax)
                if ax_int < 0:
                    ax_int += rank
                normalized.append(ax_int)
            axes_attr = tuple(normalized)
        else:
            axes_attr = None  # reduce all

        # Build ReduceMean
        # Note: we skip casting for now as abstract_eval handles weak types usually

        inputs = [operand_val]
        if axes_attr is not None:
            # We use opset 13+ axes input
            axes_const = ctx.builder.const_i64(
                ctx.fresh_name("reducemean_axes"), axes_attr
            )
            inputs.append(axes_const)

        # Create output value
        out_val = ctx.get_value_for_var(out_var)

        # Attributes
        attrs = [Attr("keepdims", AttributeType.INT, 1 if keepdims else 0)]

        ctx.builder.add_node(
            op_type="ReduceMean", inputs=inputs, outputs=[out_val], attributes=attrs
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.mean not found for monkey patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                axis: int | _Seq[int] | None = None,
                dtype: np.dtype[Any] | type | None = None,
                out: None = None,
                keepdims: bool = False,
                *,
                where: bool = True,
            ) -> jax.Array:
                if out is not None:
                    raise NotImplementedError(
                        "jnp.mean with 'out' is not supported for ONNX export"
                    )
                if where is not True:
                    raise NotImplementedError(
                        "jnp.mean with 'where' is not supported for ONNX export"
                    )

                axes_param: tuple[int, ...] | None = None
                # Standardize axis to tuple or int
                if axis is not None:
                    if isinstance(axis, _Seq) and not isinstance(axis, (str, bytes)):
                        axes_param = tuple(int(ax) for ax in axis)
                    else:
                        axes_param = (int(axis),)

                return cls._PRIM.bind(
                    a,
                    axes=axes_param,
                    dtype=dtype,
                    out=out,
                    keepdims=keepdims,
                    where=where,
                )

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


@JnpMeanPlugin._PRIM.def_impl
def _mean_impl(
    a: ArrayLike,
    *,
    axes: Sequence[int] | None = None,
    dtype: np.dtype[Any] | type | None = None,
    out: None = None,
    keepdims: bool = False,
    where: bool = True,
) -> jax.Array:
    if out is not None:
        raise NotImplementedError(
            "jnp.mean with 'out' is not supported for ONNX export"
        )
    if where is not True:
        raise NotImplementedError(
            "jnp.mean with 'where' is not supported for ONNX export"
        )

    try:
        orig = get_orig_impl(JnpMeanPlugin._PRIM, JnpMeanPlugin._FUNC_NAME)
    except RuntimeError:
        orig = jnp.mean

    return orig(a, axis=axes, dtype=dtype, out=out, keepdims=keepdims, where=where)


JnpMeanPlugin._PRIM.def_abstract_eval(JnpMeanPlugin.abstract_eval)


BatchDim = int | type(batching.not_mapped)


def _mean_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    axes: Sequence[int] | None = None,
    dtype: np.dtype[Any] | type | None = None,
    out: None = None,
    keepdims: bool = False,
    where: bool = True,
) -> tuple[jax.Array, BatchDim]:
    (operand,), (bdim,) = batched_args, batch_dims
    axis_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, axis_size)

    slice_rank = operand.ndim - 1

    # Handle axis mapping for batching
    if axes is None:
        # if axis is None, it means reduce over ALL axes.
        # When batched, we reduce over all axes EXCEPT the batch dim (which is now 0)
        # So we reduce over (1, ..., ndim-1)
        axes_full = tuple(range(1, operand.ndim))
    else:
        axes_norm = tuple(int(ax) % slice_rank for ax in axes)
        axes_full = tuple(ax + 1 for ax in axes_norm)

    res = JnpMeanPlugin._PRIM.bind(
        operand, axes=axes_full, dtype=dtype, out=out, keepdims=keepdims, where=where
    )
    return res, 0


batching.primitive_batchers[JnpMeanPlugin._PRIM] = _mean_batch_rule
