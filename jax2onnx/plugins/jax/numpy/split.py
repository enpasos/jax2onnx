# jax2onnx/plugins/jax/numpy/split.py

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar, Final

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
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SPLIT_PRIM: Final = make_jnp_primitive("jax.numpy.split")
_SPLIT_PRIM.multiple_results = True

BatchDim = int | type(batching.not_mapped)


def _normalize_axis(axis: int | None, rank: int) -> int:
    ax = 0 if axis is None else int(axis)
    if ax < 0:
        ax += rank
    if ax < 0 or ax >= rank:
        raise ValueError(f"axis {axis} out of bounds for rank {rank}")
    return ax


def _to_int_sequence(values: Sequence[int | np.integer]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


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


@register_primitive(
    jaxpr_primitive=_SPLIT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.split.html",
    onnx=[
        {"component": "Split", "doc": "https://onnx.ai/onnx/operators/onnx__Split.html"}
    ],
    since="v0.7.2",
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
        indices_or_sections: int | Sequence[int | np.integer],
        axis: int = 0,
    ) -> tuple[core.ShapedArray, ...]:
        rank = len(x.shape)
        axis_norm = _normalize_axis(axis, rank)
        dim = x.shape[axis_norm]

        if isinstance(dim, int):
            sizes = _split_sizes(dim, indices_or_sections)
        else:
            raise TypeError(
                "jnp.split requires concrete size along split axis in IR path"
            )

        specs = []
        for sz in sizes:
            shape = list(x.shape)
            shape[axis_norm] = sz
            specs.append(core.ShapedArray(tuple(shape), x.dtype))
        return tuple(specs)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        params = getattr(eqn, "params", {})
        axis_param = params.get("axis", 0)
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

        if isinstance(indices_param, int):
            sizes = _split_sizes(dim, indices_param)
        elif isinstance(indices_param, Sequence):
            indices_seq = _to_int_sequence(indices_param)
            sizes = _split_sizes(dim, indices_seq)
        else:
            indices_np = np.asarray(indices_param)
            if indices_np.ndim != 1:
                raise TypeError("split indices must be 1-D")
            indices_seq = _to_int_sequence(indices_np.tolist())
            sizes = _split_sizes(dim, indices_seq)

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

        split_outputs = builder.Split(
            arr_val,
            split_val,
            axis=int(axis),
            _outputs=output_names,
        )
        if not isinstance(split_outputs, (tuple, list)):
            split_outputs = [split_outputs]

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
                if isinstance(indices_or_sections, (list, tuple)):
                    indices_param = tuple(indices_or_sections)
                else:
                    try:
                        indices_param = tuple(indices_or_sections.tolist())
                    except AttributeError:
                        indices_param = indices_or_sections
                return cls._PRIM.bind(
                    arr,
                    indices_or_sections=indices_param,
                    axis=axis_norm,
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
    a: ArrayLike, indices_or_sections: int | Sequence[int], axis: int = 0
) -> Sequence[jax.Array]:
    orig = get_orig_impl(JnpSplitPlugin._PRIM, JnpSplitPlugin._FUNC_NAME)
    return orig(a, indices_or_sections, axis=axis)


JnpSplitPlugin._PRIM.def_abstract_eval(JnpSplitPlugin.abstract_eval)


def _split_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    indices_or_sections: int | Sequence[int],
    axis: int = 0,
) -> tuple[tuple[jax.Array, ...], tuple[BatchDim, ...]]:
    (arr,) = batched_args
    (arr_bdim,) = batch_dims
    axis_int = int(axis)

    try:
        orig = get_orig_impl(JnpSplitPlugin._PRIM, JnpSplitPlugin._FUNC_NAME)
    except RuntimeError:
        orig = jnp.split

    outputs = orig(arr, indices_or_sections, axis=axis_int)
    if isinstance(outputs, list):
        outputs = tuple(outputs)
    if arr_bdim is batching.not_mapped:
        return outputs, tuple(batching.not_mapped for _ in outputs)

    axis_size = None
    arr_shape = getattr(arr, "shape", None)
    if arr_shape is not None and arr_bdim < len(arr_shape):
        axis_size = arr_shape[arr_bdim]

    arr_front = batching.bdim_at_front(arr, arr_bdim, axis_size)

    if arr_bdim == axis_int:
        inner_axis = axis_int
    elif arr_bdim < axis_int:
        inner_axis = axis_int - 1
    else:
        inner_axis = axis_int

    try:
        orig_impl = get_orig_impl(JnpSplitPlugin._PRIM, JnpSplitPlugin._FUNC_NAME)
    except RuntimeError:
        orig_impl = jnp.split

    def _split_single(x):
        parts = orig_impl(x, indices_or_sections, axis=inner_axis)
        if isinstance(parts, list):
            parts = tuple(parts)
        return parts

    vmapped = jax.vmap(_split_single)(arr_front)

    results = []
    for part in vmapped:
        if arr_bdim != 0:
            part = jnp.moveaxis(part, 0, arr_bdim)
        results.append(part)
    return tuple(results), tuple(arr_bdim for _ in results)


batching.primitive_batchers[JnpSplitPlugin._PRIM] = _split_batch_rule
