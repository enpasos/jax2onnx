# jax2onnx/plugins/jax/numpy/prod.py

from __future__ import annotations

from collections.abc import Sequence as _Seq
from typing import Any, Callable, ClassVar, Final, Sequence

import numpy as np
from numpy.typing import ArrayLike

import jax
from jax import core
import jax.numpy as jnp
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._reduce_utils import lower_reduction
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    make_jnp_primitive,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_PROD_PRIM: Final = make_jnp_primitive("jax.numpy.prod")


@register_primitive(
    jaxpr_primitive=_PROD_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.prod.html",
    onnx=[
        {
            "component": "ReduceProd",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceProd.html",
        }
    ],
    since="v0.8.0",
    context="primitives.jnp",
    component="prod",
    testcases=[
        {
            "testcase": "basic_prod",
            "callable": lambda: jnp.prod(np.arange(12, dtype=np.float32).reshape(3, 4)),
            "input_values": [],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["ReduceProd"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "prod_with_axis",
            "callable": lambda: jnp.prod(
                np.arange(60, dtype=np.float32).reshape(3, 4, 5), axis=1
            ),
            "input_values": [],
            "expected_output_shapes": [(3, 5)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}},
                        "path": "ReduceProd:3x5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "prod_with_keepdims",
            "callable": lambda: jnp.prod(
                np.arange(12, dtype=np.float32).reshape(3, 4), axis=0, keepdims=True
            ),
            "input_values": [],
            "expected_output_shapes": [(1, 4)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "ReduceProd:1x4",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_prod_basic",
            "callable": lambda x: jnp.prod(x),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceProd"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_prod_axis",
            "callable": lambda x: jnp.prod(x, axis=1),
            "input_shapes": [(3, 4, 5)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}},
                        "path": "ReduceProd:3x5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_prod_keepdims",
            "callable": lambda x: jnp.prod(x, axis=0, keepdims=True),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "ReduceProd:1x4",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "prod_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.prod)(x),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpProdPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _PROD_PRIM
    _FUNC_NAME: ClassVar[str] = "prod"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        axes: Sequence[int] | None = None,
        axes_is_tuple: bool = False,  # ignored, kept for signature symmetry
        dtype: np.dtype[Any] | type | None = None,
        keepdims: bool = False,
        initial: ArrayLike | None = None,
        where: bool = True,
    ) -> core.ShapedArray:
        if initial is not None:
            raise NotImplementedError(
                "jnp.prod with 'initial' is not supported in ONNX lowering"
            )
        if where is not True:
            raise NotImplementedError(
                "jnp.prod with 'where' mask is not supported in ONNX lowering"
            )

        ndim = len(x.shape)
        out_dtype = np.dtype(dtype) if dtype is not None else np.dtype(x.dtype)

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
        lower_reduction(ctx, eqn, op_type="ReduceProd", allow_dtype_param=True)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.prod not found for monkey patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                axis: int | _Seq[int] | None = None,
                dtype: np.dtype[Any] | type | None = None,
                keepdims: bool = False,
                *,
                initial: ArrayLike | None = None,
                where: bool = True,
            ) -> jax.Array:
                if initial is not None:
                    raise NotImplementedError(
                        "jnp.prod with 'initial' is not supported for ONNX export"
                    )
                if where is not True:
                    raise NotImplementedError(
                        "jnp.prod with 'where' is not supported for ONNX export"
                    )
                axes_param: tuple[int, ...] | None = None
                axes_is_tuple = False
                if axis is not None:
                    if isinstance(axis, _Seq) and not isinstance(axis, (str, bytes)):
                        axes_param = tuple(int(ax) for ax in axis)
                        axes_is_tuple = True
                    else:
                        axes_param = (int(axis),)
                        axes_is_tuple = False
                return cls._PRIM.bind(
                    jnp.asarray(a),
                    axes=axes_param,
                    axes_is_tuple=axes_is_tuple,
                    dtype=dtype,
                    keepdims=keepdims,
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


@JnpProdPlugin._PRIM.def_impl
def _prod_impl(
    a: ArrayLike,
    *,
    axes: tuple[int, ...] | None = None,
    axes_is_tuple: bool = False,
    dtype: np.dtype[Any] | type | None = None,
    keepdims: bool = False,
    initial: ArrayLike | None = None,
    where: bool = True,
) -> jax.Array:
    if initial is not None:
        raise NotImplementedError(
            "jnp.prod with 'initial' is not supported for ONNX export"
        )
    if where is not True:
        raise NotImplementedError(
            "jnp.prod with 'where' is not supported for ONNX export"
        )

    try:
        orig = get_orig_impl(JnpProdPlugin._PRIM, JnpProdPlugin._FUNC_NAME)
    except RuntimeError:
        orig = jnp.prod

    axis_arg: int | tuple[int, ...] | None = None
    if axes is not None:
        axis_vals = tuple(int(ax) for ax in axes)
        axis_arg = axis_vals if axes_is_tuple else axis_vals[0]

    return orig(jnp.asarray(a), axis=axis_arg, dtype=dtype, keepdims=keepdims)


JnpProdPlugin._PRIM.def_abstract_eval(JnpProdPlugin.abstract_eval)


BatchDim = int | type(batching.not_mapped)


def _prod_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    axes: tuple[int, ...] | None = None,
    axes_is_tuple: bool = False,
    dtype: np.dtype[Any] | type | None = None,
    keepdims: bool = False,
) -> tuple[jax.Array, BatchDim]:
    (operand,), (bdim,) = batched_args, batch_dims
    axis_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, axis_size)

    slice_rank = operand.ndim - 1
    if axes is None:
        axes_full = tuple(range(1, operand.ndim))
        axes_is_tuple = True
    else:
        axes_norm = tuple(int(ax) % slice_rank for ax in axes)
        axes_full = tuple(ax + 1 for ax in axes_norm)
        axes_is_tuple = True

    out = JnpProdPlugin._PRIM.bind(
        operand,
        axes=axes_full,
        axes_is_tuple=axes_is_tuple,
        dtype=dtype,
        keepdims=keepdims,
    )
    return out, 0


batching.primitive_batchers[JnpProdPlugin._PRIM] = _prod_batch_rule
