# jax2onnx/plugins/jax/numpy/split.py

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar, Final, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from jax import core
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax._autodiff_utils import (
    register_allowlisted_original_rule_forwarding,
)
from jax2onnx.plugins._utils import normalize_builder_outputs
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SPLIT_PRIM: Final = make_jnp_primitive("jax.numpy.split")
_SPLIT_PRIM.multiple_results = True

BatchDim: TypeAlias = int | None


def _normalize_axis(axis: int | None, rank: int) -> int:
    ax = 0 if axis is None else int(axis)
    if ax < 0:
        ax += rank
    if ax < 0 or ax >= rank:
        raise ValueError(f"axis {axis} out of bounds for rank {rank}")
    return ax


def _to_int_sequence(values: Sequence[int | np.integer]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _normalize_indices_or_sections(
    indices_or_sections: int | Sequence[int | np.integer] | np.ndarray,
) -> int | tuple[int, ...]:
    if isinstance(indices_or_sections, int):
        return int(indices_or_sections)
    if isinstance(indices_or_sections, np.ndarray):
        if indices_or_sections.ndim != 1:
            raise TypeError("split indices must be 1-D")
        return _to_int_sequence(indices_or_sections.tolist())
    if isinstance(indices_or_sections, Sequence):
        return _to_int_sequence(indices_or_sections)
    raise TypeError("split indices_or_sections must be int or 1-D sequence")


def _split_sizes(
    dim_size: int,
    indices_or_sections: int | Sequence[int | np.integer],
) -> tuple[int, ...]:
    if isinstance(indices_or_sections, int):
        sections = int(indices_or_sections)
        if sections <= 0:
            raise ValueError("number of sections must be positive")
        if dim_size % sections != 0:
            raise ValueError(
                f"Dimension size {dim_size} not divisible by {sections} sections"
            )
        return (dim_size // sections,) * sections
    indices = [0, *(_to_int_sequence(indices_or_sections)), dim_size]
    if sorted(indices) != indices:
        raise ValueError("split indices must be sorted")
    diffs = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
    if any(d <= 0 for d in diffs):
        raise ValueError("split sizes must be positive")
    return tuple(diffs)


def _resolve_split_sizes(
    *,
    dim_size: int,
    sizes: Sequence[int] | None = None,
    indices_or_sections: int | Sequence[int | np.integer] | np.ndarray | None = None,
) -> tuple[int, ...]:
    if sizes is not None:
        sizes_tuple = _to_int_sequence(sizes)
        if any(sz <= 0 for sz in sizes_tuple):
            raise ValueError("split sizes must be positive")
        if sum(sizes_tuple) != int(dim_size):
            raise ValueError(
                f"split sizes {sizes_tuple} do not sum to axis size {dim_size}"
            )
        if indices_or_sections is not None:
            normalized = _normalize_indices_or_sections(indices_or_sections)
            expected = _split_sizes(int(dim_size), normalized)
            if expected != sizes_tuple:
                raise ValueError(
                    "Conflicting split params: 'sizes' and 'indices_or_sections'"
                )
        return sizes_tuple
    if indices_or_sections is None:
        raise KeyError("split params missing 'sizes'/'indices_or_sections'")
    normalized = _normalize_indices_or_sections(indices_or_sections)
    return _split_sizes(int(dim_size), normalized)


@register_primitive(
    jaxpr_primitive=_SPLIT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.split.html",
    onnx=[
        {"component": "Split", "doc": "https://onnx.ai/onnx/operators/onnx__Split.html"}
    ],
    since="0.7.2",
    context="primitives.jnp",
    component="split",
    testcases=[
        {
            "testcase": "split_by_sections",
            "callable": lambda x: jnp.split(x, 3, axis=1),
            "input_shapes": [(1, 9)],
            "post_check_onnx_graph": EG(
                ["Split"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "split_by_indices",
            "callable": lambda x: jnp.split(x, [2, 5], axis=1),
            "input_shapes": [(1, 9)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Split",
                        "counts": {"Split": 1},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "split_by_indices_symbolic",
            "callable": lambda x: jnp.split(x, [3, 7], axis=2),
            "input_shapes": [("B", 4, 10)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Split",
                        "counts": {"Split": 1},
                    }
                ],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "split_sections",
            "callable": lambda x: jnp.split(x, 3, axis=1),
            "input_shapes": [(1, 9)],
            "post_check_onnx_graph": EG(
                ["Split"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "split_indices_numpy",
            "callable": lambda x: jnp.split(x, np.array([2, 5]), axis=1),
            "input_shapes": [(1, 9)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Split",
                        "counts": {"Split": 1},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "split_single_section",
            "callable": lambda x: jnp.split(x, 1, axis=1),
            "input_shapes": [(1, 9)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Split",
                        "counts": {"Split": 1},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "split_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(
                lambda y: jnp.sum(jnp.split(y, 2, axis=0)[0] ** 2)
            )(x),
            "input_shapes": [(4,)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpSplitPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SPLIT_PRIM
    _FUNC_NAME: ClassVar[str] = "split"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.ShapedArray,
        *,
        sizes: Sequence[int] | None = None,
        indices_or_sections: (
            int | Sequence[int | np.integer] | np.ndarray | None
        ) = None,
        axis: int = 0,
    ) -> tuple[core.ShapedArray, ...]:
        rank = len(x.shape)
        axis_norm = _normalize_axis(axis, rank)
        dim = x.shape[axis_norm]

        if isinstance(dim, int):
            sizes_tuple = _resolve_split_sizes(
                dim_size=dim,
                sizes=sizes,
                indices_or_sections=indices_or_sections,
            )
        else:
            raise TypeError(
                "jnp.split requires concrete size along split axis in IR path"
            )

        specs = []
        for sz in sizes_tuple:
            shape = list(x.shape)
            shape[axis_norm] = sz
            specs.append(core.ShapedArray(tuple(shape), x.dtype))
        return tuple(specs)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        params = getattr(eqn, "params", {})
        axis_param = params.get("axis", 0)
        sizes_param = params.get("sizes")
        indices_param = params.get("indices_or_sections")

        (arr_var,) = eqn.invars
        out_vars = list(eqn.outvars)

        arr_shape = tuple(getattr(arr_var.aval, "shape", ()))
        axis = _normalize_axis(axis_param, len(arr_shape))
        dim = arr_shape[axis]
        if not isinstance(dim, int):
            raise TypeError(
                "jnp.split requires concrete size along split axis for ONNX lowering"
            )

        sizes = _resolve_split_sizes(
            dim_size=dim,
            sizes=sizes_param,
            indices_or_sections=indices_param,
        )

        arr_val = ctx.get_value_for_var(arr_var, name_hint=ctx.fresh_name("split_in"))
        split_val = _const_i64(
            ctx, np.asarray(sizes, dtype=np.int64), ctx.fresh_name("split_sizes")
        )
        _stamp_type_and_shape(split_val, (len(sizes),))
        _ensure_value_metadata(ctx, split_val)

        out_specs = [
            ctx.get_value_for_var(v, name_hint=ctx.fresh_name("split_out"))
            for v in out_vars
        ]

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for split lowering")

        output_names = []
        for spec in out_specs:
            name = getattr(spec, "name", None)
            if not name:
                name = ctx.fresh_name("split_out")
            output_names.append(name)

        split_outputs = normalize_builder_outputs(
            builder.Split(
                arr_val,
                split_val,
                axis=int(axis),
                _outputs=output_names,
            )
        )

        for result, spec, sz, out_var in zip(split_outputs, out_specs, sizes, out_vars):
            spec_type = getattr(spec, "type", None)
            if spec_type is not None:
                result.type = spec_type
            elif getattr(arr_val, "type", None) is not None:
                result.type = arr_val.type
            out_shape = list(arr_shape)
            out_shape[axis] = sz
            _stamp_type_and_shape(result, tuple(out_shape))
            _ensure_value_metadata(ctx, result)
            ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., Sequence[jax.Array]] | None,
        ) -> Callable[..., Sequence[jax.Array]]:
            if orig is None:
                raise RuntimeError("Original jnp.split not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                indices_or_sections: int | Sequence[int] | np.ndarray,
                axis: int = 0,
            ) -> Sequence[jax.Array]:
                arr = jnp.asarray(a)
                axis_norm = _normalize_axis(axis, arr.ndim)
                dim = arr.shape[axis_norm]
                if not isinstance(dim, int):
                    raise TypeError(
                        "jnp.split requires concrete size along split axis in IR path"
                    )
                sizes = _resolve_split_sizes(
                    dim_size=dim,
                    indices_or_sections=indices_or_sections,
                )
                return cast(
                    Sequence[jax.Array],
                    cls._PRIM.bind(
                        arr,
                        sizes=sizes,
                        axis=axis_norm,
                    ),
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


@JnpSplitPlugin._PRIM.def_impl
def _split_impl(
    a: ArrayLike,
    *,
    sizes: Sequence[int] | None = None,
    indices_or_sections: int | Sequence[int] | np.ndarray | None = None,
    axis: int = 0,
) -> Sequence[jax.Array]:
    arr = jnp.asarray(a)
    axis_norm = _normalize_axis(axis, arr.ndim)
    dim = arr.shape[axis_norm]
    if not isinstance(dim, int):
        raise TypeError("jnp.split requires concrete size along split axis in IR path")
    sizes_tuple = _resolve_split_sizes(
        dim_size=dim,
        sizes=sizes,
        indices_or_sections=indices_or_sections,
    )
    return tuple(jax.lax.split(arr, sizes_tuple, axis=axis_norm))


JnpSplitPlugin._PRIM.def_abstract_eval(JnpSplitPlugin.abstract_eval)


def _split_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    sizes: Sequence[int] | None = None,
    indices_or_sections: int | Sequence[int] | np.ndarray | None = None,
    axis: int = 0,
) -> tuple[tuple[jax.Array, ...], tuple[BatchDim, ...]]:
    (arr,) = batched_args
    (arr_bdim,) = batch_dims
    axis_int = _normalize_axis(axis, arr.ndim)
    dim = arr.shape[axis_int]
    if not isinstance(dim, int):
        raise TypeError("jnp.split requires concrete size along split axis in IR path")
    sizes_tuple = _resolve_split_sizes(
        dim_size=dim,
        sizes=sizes,
        indices_or_sections=indices_or_sections,
    )

    outputs = tuple(jax.lax.split(arr, sizes_tuple, axis=axis_int))
    if arr_bdim is None:
        return outputs, tuple(None for _ in outputs)
    arr_bdim_int = arr_bdim

    axis_size = None
    arr_shape = getattr(arr, "shape", None)
    if arr_shape is not None and arr_bdim_int < len(arr_shape):
        axis_size = arr_shape[arr_bdim_int]

    arr_front = batching.bdim_at_front(arr, arr_bdim_int, axis_size)

    if arr_bdim_int == axis_int:
        inner_axis = axis_int
    elif arr_bdim_int < axis_int:
        inner_axis = axis_int - 1
    else:
        inner_axis = axis_int

    def _split_single(x: jax.Array) -> tuple[jax.Array, ...]:
        return tuple(jax.lax.split(x, sizes_tuple, axis=inner_axis))

    vmapped = jax.vmap(_split_single)(arr_front)

    results = []
    for part in vmapped:
        if arr_bdim_int != 0:
            part = jnp.moveaxis(part, 0, arr_bdim_int)
        results.append(part)
    return tuple(results), tuple(arr_bdim_int for _ in results)


batching.primitive_batchers[JnpSplitPlugin._PRIM] = _split_batch_rule


register_allowlisted_original_rule_forwarding(
    orig_prim=jax.lax.split_p,
    new_prim=JnpSplitPlugin._PRIM,
    forward_batching=False,
)
