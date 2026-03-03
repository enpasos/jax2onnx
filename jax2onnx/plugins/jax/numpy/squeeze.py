# jax2onnx/plugins/jax/numpy/squeeze.py

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, ClassVar, Final, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from jax import core
from jax.interpreters import batching

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax._autodiff_utils import (
    register_allowlisted_original_rule_forwarding,
)
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SQUEEZE_PRIM: Final = make_jnp_primitive("jax.numpy.squeeze")


def _normalize_axes(axes: int | Sequence[int] | None, rank: int) -> tuple[int, ...]:
    if axes is None:
        return tuple()
    if isinstance(axes, int):
        axes_iter: Iterable[int] = (axes,)
    else:
        axes_iter = axes
    normalized = []
    for ax in axes_iter:
        ax_int = int(ax)
        if ax_int < 0:
            ax_int += rank
        if ax_int < 0 or ax_int >= rank:
            raise ValueError(f"axis {ax} out of bounds for rank {rank}")
        if ax_int not in normalized:
            normalized.append(ax_int)
    return tuple(normalized)


def _resolve_dimensions(
    *,
    rank: int,
    shape: Sequence[object],
    dimensions: Sequence[int] | None = None,
    axes: int | Sequence[int] | None = None,
    axis: int | Sequence[int] | None = None,
) -> tuple[int, ...]:
    if dimensions is not None:
        dims = _normalize_axes(tuple(dimensions), rank)
    elif axes is not None:
        dims = _normalize_axes(axes, rank)
    elif axis is not None:
        dims = _normalize_axes(axis, rank)
    else:
        dims = tuple(i for i, d in enumerate(shape) if isinstance(d, int) and d == 1)

    if axes is not None and axis is not None:
        if _normalize_axes(axes, rank) != _normalize_axes(axis, rank):
            raise ValueError("Conflicting squeeze params: both 'axes' and 'axis'")
    if dimensions is not None:
        if axes is not None and dims != _normalize_axes(axes, rank):
            raise ValueError("Conflicting squeeze params: both 'dimensions' and 'axes'")
        if axis is not None and dims != _normalize_axes(axis, rank):
            raise ValueError("Conflicting squeeze params: both 'dimensions' and 'axis'")
    return dims


@register_primitive(
    jaxpr_primitive=_SQUEEZE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.squeeze.html",
    onnx=[
        {
            "component": "Squeeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
        }
    ],
    since="0.1.0",
    context="primitives.jnp",
    component="squeeze",
    testcases=[
        {
            "testcase": "squeeze_single_dim",
            "callable": lambda a: jnp.squeeze(a, axis=0),
            "input_shapes": [(1, 49, 10)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "Squeeze:49x10",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "squeeze_multiple_dims",
            "callable": lambda a: jnp.squeeze(a, axis=(0, 2)),
            "input_shapes": [(1, 49, 1, 10)],
            "post_check_onnx_graph": EG(
                ["Squeeze:49x10"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "squeeze_vit_output",
            "callable": lambda a: jnp.squeeze(a, axis=1),
            "input_shapes": [(1, 1, 10)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}},
                        "path": "Squeeze:1x10",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "squeeze_dynamic_batch",
            "callable": lambda a: jnp.squeeze(a, axis=1),
            "input_shapes": [("B", 1, 10)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}},
                        "path": "Squeeze:Bx10",
                    }
                ],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "squeeze_all_dims",
            "callable": lambda a: jnp.squeeze(a),
            "input_shapes": [(1, 1, 1)],
            "post_check_onnx_graph": EG(
                ["Squeeze"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "squeeze_negative_axis",
            "callable": lambda a: jnp.squeeze(a, axis=-1),
            "input_shapes": [(1, 49, 1)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 2.0}},
                        "path": "Squeeze:1x49",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "squeeze_negative_axis_tuple",
            "callable": lambda a: jnp.squeeze(a, axis=(-1, -3)),
            "input_shapes": [(1, 49, 1)],
            "post_check_onnx_graph": EG(
                ["Squeeze:49"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "squeeze_dynamic_and_negative_axis",
            "callable": lambda a: jnp.squeeze(a, axis=(-1, -3)),
            "input_shapes": [(1, "B", 1)],
            "post_check_onnx_graph": EG(
                ["Squeeze:B"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "squeeze_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: jnp.squeeze(y, axis=0))(x),
            "input_shapes": [(3, 1, 4)],
        },
        {
            "testcase": "squeeze_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(
                lambda y: jnp.sum(jnp.squeeze(y, axis=0) ** 2)
            )(x),
            "input_shapes": [(1, 4)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpSqueezePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SQUEEZE_PRIM
    _FUNC_NAME: ClassVar[str] = "squeeze"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        dimensions: Sequence[int] | None = None,
        axes: int | Sequence[int] | None = None,
        axis: int | Sequence[int] | None = None,
    ) -> core.ShapedArray:
        dims = _resolve_dimensions(
            rank=len(x.shape),
            shape=tuple(x.shape),
            dimensions=dimensions,
            axes=axes,
            axis=axis,
        )
        storage_slot = f"__orig_impl__{JnpSqueezePlugin._FUNC_NAME}"
        orig = getattr(JnpSqueezePlugin._PRIM, storage_slot, jnp.squeeze)
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        result = jax.eval_shape(lambda arr: orig(arr, axis=dims), spec)
        return core.ShapedArray(result.shape, result.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        params = getattr(eqn, "params", {})
        dimensions_param = params.get("dimensions")
        axes_param = params.get("axes")
        axis_param = params.get("axis")

        (arr_var,) = eqn.invars
        (out_var,) = eqn.outvars

        arr_shape = tuple(getattr(arr_var.aval, "shape", ()))
        rank = len(arr_shape)

        axes = _resolve_dimensions(
            rank=rank,
            shape=arr_shape,
            dimensions=dimensions_param,
            axes=axes_param,
            axis=axis_param,
        )

        arr_val = ctx.get_value_for_var(arr_var, name_hint=ctx.fresh_name("squeeze_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("squeeze_out")
        )
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for squeeze lowering"
            )

        if not axes:
            # nothing to squeeze
            result = builder.Identity(
                arr_val,
                _outputs=[
                    getattr(out_spec, "name", None) or ctx.fresh_name("Identity")
                ],
            )
            if getattr(arr_val, "type", None) is not None:
                result.type = arr_val.type
            _stamp_type_and_shape(result, tuple(arr_shape))
            _ensure_value_metadata(ctx, result)
            bind_value = getattr(ctx, "bind_value_for_var", None)
            if not callable(bind_value):
                raise AttributeError("IR build context missing bind_value_for_var")
            bind_value(out_var, result)
            return

        axes_vals = _const_i64(ctx, np.asarray(axes, dtype=np.int64), "squeeze_axes")
        result = builder.Squeeze(
            arr_val,
            axes_vals,
            _outputs=[getattr(out_spec, "name", None) or ctx.fresh_name("Squeeze")],
        )
        target_shape = tuple(getattr(out_var.aval, "shape", ()))
        if getattr(arr_val, "type", None) is not None:
            result.type = arr_val.type
        _stamp_type_and_shape(result, target_shape)
        _ensure_value_metadata(ctx, result)
        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.squeeze not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike, axis: int | Sequence[int] | None = None
            ) -> jax.Array:
                arr = jnp.asarray(a)
                dims = _resolve_dimensions(
                    rank=arr.ndim,
                    shape=tuple(arr.shape),
                    axis=axis,
                )
                return cls._PRIM.bind(arr, dimensions=dims)

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


@JnpSqueezePlugin._PRIM.def_impl
def _squeeze_impl(
    a: ArrayLike,
    *,
    dimensions: Sequence[int] | None = None,
    axes: int | Sequence[int] | None = None,
    axis: int | Sequence[int] | None = None,
) -> jax.Array:
    orig = get_orig_impl(JnpSqueezePlugin._PRIM, JnpSqueezePlugin._FUNC_NAME)
    arr = jnp.asarray(a)
    dims = _resolve_dimensions(
        rank=arr.ndim,
        shape=tuple(arr.shape),
        dimensions=dimensions,
        axes=axes,
        axis=axis,
    )
    return orig(arr, axis=dims)


JnpSqueezePlugin._PRIM.def_abstract_eval(JnpSqueezePlugin.abstract_eval)


BatchDim: TypeAlias = int | None


def _squeeze_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    dimensions: Sequence[int] | None = None,
    axes: int | Sequence[int] | None = None,
    axis: int | Sequence[int] | None = None,
) -> tuple[jax.Array, BatchDim]:
    (operand,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        axis_dims = _resolve_dimensions(
            rank=operand.ndim,
            shape=tuple(operand.shape),
            dimensions=dimensions,
            axes=axes,
            axis=axis,
        )
        out = JnpSqueezePlugin._PRIM.bind(operand, dimensions=axis_dims)
        return out, batching.not_mapped

    batch_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, batch_size)

    slice_rank = operand.ndim - 1
    slice_shape = tuple(operand.shape[1:])
    axes_slice = _resolve_dimensions(
        rank=slice_rank,
        shape=slice_shape,
        dimensions=dimensions,
        axes=axes,
        axis=axis,
    )

    axes_full = tuple(int(ax) + 1 for ax in axes_slice)
    out = JnpSqueezePlugin._PRIM.bind(operand, dimensions=axes_full)
    return out, 0


batching.primitive_batchers[JnpSqueezePlugin._PRIM] = _squeeze_batch_rule


register_allowlisted_original_rule_forwarding(
    orig_prim=jax.lax.squeeze_p,
    new_prim=JnpSqueezePlugin._PRIM,
    forward_batching=False,
)
