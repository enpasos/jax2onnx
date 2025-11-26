# jax2onnx/plugins/jax/numpy/squeeze.py

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from jax import core, lax
from jax.interpreters import batching

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
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


@register_primitive(
    jaxpr_primitive=_SQUEEZE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.squeeze.html",
    onnx=[
        {
            "component": "Squeeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
        }
    ],
    since="v0.1.0",
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
    ],
)
class JnpSqueezePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SQUEEZE_PRIM
    _FUNC_NAME: ClassVar[str] = "squeeze"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue, *, axis: int | Sequence[int] | None = None
    ) -> core.ShapedArray:
        storage_slot = f"__orig_impl__{JnpSqueezePlugin._FUNC_NAME}"
        orig = getattr(JnpSqueezePlugin._PRIM, storage_slot, jnp.squeeze)
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        result = jax.eval_shape(lambda arr: orig(arr, axis=axis), spec)
        return core.ShapedArray(result.shape, result.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        params = getattr(eqn, "params", {})
        axis_param = params.get("axes", params.get("axis"))

        (arr_var,) = eqn.invars
        (out_var,) = eqn.outvars

        arr_shape = tuple(getattr(arr_var.aval, "shape", ()))
        rank = len(arr_shape)

        axes = _normalize_axes(axis_param, rank)
        if not axes:
            # Squeeze all singleton dims
            axes = tuple(
                i for i, d in enumerate(arr_shape) if isinstance(d, int) and d == 1
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
                return cls._PRIM.bind(arr, axis=axis)

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
def _squeeze_impl(a: ArrayLike, axis: int | Sequence[int] | None = None) -> jax.Array:
    orig = get_orig_impl(JnpSqueezePlugin._PRIM, JnpSqueezePlugin._FUNC_NAME)
    return orig(a, axis=axis)


JnpSqueezePlugin._PRIM.def_abstract_eval(JnpSqueezePlugin.abstract_eval)


BatchDim = int | type(batching.not_mapped)


def _squeeze_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    axis: int | Sequence[int] | None = None,
) -> tuple[jax.Array, BatchDim]:
    (operand,), (_bdim,) = batched_args, batch_dims
    operand_shape = getattr(operand, "shape", ())
    rank = len(operand_shape)
    axes = _normalize_axes(axis, rank)
    if not axes:
        axes = tuple(
            idx
            for idx, dim in enumerate(operand_shape)
            if isinstance(dim, int) and dim == 1
        )
    return _apply_squeeze_batch_rule(batched_args, batch_dims, dimensions=axes)


batching.primitive_batchers[JnpSqueezePlugin._PRIM] = _squeeze_batch_rule


def _apply_squeeze_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    dimensions: Sequence[int],
) -> tuple[jax.Array, BatchDim]:
    (operand,), (bdim,) = batched_args, batch_dims
    operand, bdim = batching.move_stacked_axis(operand, bdim, 0)
    # account for the leading batch dimension
    dims_with_batch = tuple(int(d) + 1 for d in dimensions)
    squeezed = lax.squeeze(operand, dimensions=dims_with_batch)
    ragged_cls = getattr(batching, "RaggedAxis", None)
    if ragged_cls is not None and isinstance(bdim, ragged_cls):
        out_stack_dim = bdim.stacked_axis
    else:
        out_stack_dim = bdim
    bdim_shape = batching.bdim_as_shape(bdim, operand.shape)
    squeezed_shape = _compute_squeeze_shape(bdim_shape, dims_with_batch)
    bdim_out = batching.shape_as_bdim(out_stack_dim, squeezed_shape)
    return squeezed, bdim_out


def _compute_squeeze_shape(
    shape: Sequence[object], dimensions: Sequence[int]
) -> tuple[object, ...]:
    dims_set = set(dimensions)
    if len(dims_set) != len(dimensions):
        raise ValueError(f"dimensions are not unique: {dimensions}")
    if not all(0 <= d < len(shape) for d in dims_set):
        raise ValueError(f"dimensions outside range [0, ndim): {dimensions}")
    for d in dimensions:
        dim_val = shape[d]
        if isinstance(dim_val, (int, np.integer)) and dim_val != 1:
            raise ValueError(
                f"cannot squeeze axis {d} with size {dim_val}; shape={shape}"
            )
    return tuple(s for i, s in enumerate(shape) if i not in dims_set)
