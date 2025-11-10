# jax2onnx/plugins/jax/numpy/stack.py

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import ClassVar, Final

import jax
from jax import tree_util
from jax import core
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
import onnx_ir as ir
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import make_jnp_primitive, get_orig_impl
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_STACK_PRIM: Final = make_jnp_primitive("jax.numpy.stack")

BatchDim = int | type(batching.not_mapped)


@register_primitive(
    jaxpr_primitive=_STACK_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.stack.html",
    onnx=[
        {
            "component": "Unsqueeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Unsqueeze.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="v0.8.0",
    context="primitives.jnp",
    component="stack",
    testcases=[
        {
            "testcase": "stack_axis_0",
            "callable": lambda: jnp.stack(
                [np.arange(3, dtype=np.float32), np.arange(3, dtype=np.float32)], axis=0
            ),
            "post_check_onnx_graph": EG(
                ["Unsqueeze:1x3 -> Concat:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "stack_axis_1",
            "callable": lambda: jnp.stack(
                [np.ones((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)],
                axis=1,
            ),
            "post_check_onnx_graph": EG(
                ["Unsqueeze:2x1x2 -> Concat:2x2x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "stack_negative_axis",
            "callable": lambda: jnp.stack(
                [
                    np.array([1, 2, 3], dtype=np.float32),
                    np.array([4, 5, 6], dtype=np.float32),
                ],
                axis=-1,
            ),
            "post_check_onnx_graph": EG(
                ["Unsqueeze:3x1 -> Concat:3x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "stack_scalars",
            "callable": lambda: jnp.stack(
                [np.array(1.0, dtype=np.float32), np.array(2.0, dtype=np.float32)]
            ),
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {0: {"const": 1.0}, 1: {"const": 2.0}},
                        "path": "Unsqueeze:1 -> Concat:2",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_stack_axis0",
            "callable": lambda x, y: jnp.stack((x, y), axis=0),
            "input_shapes": [(2,), (2,)],
            "post_check_onnx_graph": EG(
                ["Unsqueeze:1x2 -> Concat:2x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_stack_axis1",
            "callable": lambda x, y: jnp.stack((x, y), axis=1),
            "input_shapes": [(2, 2), (2, 2)],
            "post_check_onnx_graph": EG(
                ["Unsqueeze:2x1x2 -> Concat:2x2x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_stack_negative_axis",
            "callable": lambda x, y: jnp.stack((x, y), axis=-1),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Unsqueeze:3x1 -> Concat:3x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_stack_scalars",
            "callable": lambda x, y: jnp.stack((x, y), axis=0),
            "input_shapes": [(), ()],
            "post_check_onnx_graph": EG(
                ["Unsqueeze:1 -> Concat:2"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpStackPlugin(PrimitiveLeafPlugin):
    """IR-only lowering for ``jax.numpy.stack``."""

    _PRIM: ClassVar = _STACK_PRIM
    _FUNC_NAME: ClassVar[str] = "stack"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        *in_avals: core.AbstractValue, axis: int, **_: object
    ) -> core.ShapedArray:
        if not in_avals:
            raise ValueError("jnp.stack requires at least one array")

        ref_shape = in_avals[0].shape
        ref_dtype = in_avals[0].dtype
        for aval in in_avals[1:]:
            if aval.shape != ref_shape:
                raise ValueError("all input arrays must have the same shape")
            if aval.dtype != ref_dtype:
                raise ValueError("all input arrays must have the same dtype")

        out_rank = len(ref_shape) + 1
        axis_idx = axis if axis >= 0 else axis + out_rank
        if axis_idx < 0 or axis_idx > out_rank:
            raise ValueError(f"stack axis {axis} is out of bounds for rank {out_rank}")

        out_shape = list(ref_shape)
        out_shape.insert(axis_idx, len(in_avals))
        return jax.core.ShapedArray(tuple(out_shape), ref_dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        axis = int(getattr(eqn, "params", {}).get("axis", 0))

        input_vals: list[ir.Value] = []
        input_shapes: list[tuple[int, ...]] = []
        for invar in eqn.invars:
            val = ctx.get_value_for_var(invar, name_hint=ctx.fresh_name("stack_in"))
            input_vals.append(val)
            input_shapes.append(tuple(getattr(invar.aval, "shape", ())))

        if not input_vals:
            raise ValueError("jnp.stack lowering received no inputs")

        rank = len(input_shapes[0])
        unsqueeze_axis = axis if axis >= 0 else axis + rank + 1
        if unsqueeze_axis < 0 or unsqueeze_axis > rank:
            raise ValueError(f"stack axis {axis} is out of range for input rank {rank}")

        axes_const = _const_i64(ctx, [unsqueeze_axis], "stack_unsqueeze_axis")

        unsqueezed_vals: list[ir.Value] = []
        for val, shape in zip(input_vals, input_shapes):
            out_shape = list(shape)
            out_shape.insert(unsqueeze_axis, 1)

            unsqueezed = ctx.builder.Unsqueeze(
                val,
                axes_const,
                _outputs=[ctx.fresh_name("stack_unsqueeze")],
            )
            dtype_enum = getattr(getattr(val, "type", None), "dtype", None)
            if dtype_enum is not None:
                unsqueezed.type = ir.TensorType(dtype_enum)
            _stamp_type_and_shape(unsqueezed, tuple(out_shape))
            _ensure_value_metadata(ctx, unsqueezed)
            unsqueezed_vals.append(unsqueezed)

        out_var = eqn.outvars[0]
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("stack_out"))
        out_shape_tuple = tuple(getattr(out_var.aval, "shape", ()))
        concat_axis = axis if axis >= 0 else axis + len(out_shape_tuple)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Concat")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Concat")

        result = ctx.builder.Concat(
            *unsqueezed_vals,
            axis=int(concat_axis),
            _outputs=[desired_name],
        )
        result_dtype = None
        if unsqueezed_vals:
            result_dtype = getattr(
                getattr(unsqueezed_vals[0], "type", None), "dtype", None
            )
        if result_dtype is None:
            result_dtype = getattr(getattr(out_spec, "type", None), "dtype", None)
        if result_dtype is not None:
            result.type = ir.TensorType(result_dtype)
        else:
            result.type = out_spec.type
        _stamp_type_and_shape(result, tuple(out_shape_tuple))
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.stack not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(arrays: object, axis: int = 0) -> jax.Array:
                flat, _ = tree_util.tree_flatten(arrays)
                if not flat:
                    raise ValueError("jnp.stack expects a non-empty sequence")
                return cls._PRIM.bind(*flat, axis=int(axis))

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


def _stack_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    axis: int = 0,
) -> tuple[jax.Array, BatchDim]:
    mapped = [
        (arg, bd)
        for arg, bd in zip(batched_args, batch_dims)
        if bd is not batching.not_mapped
    ]

    if not mapped:
        out = JnpStackPlugin._PRIM.bind(*batched_args, axis=axis)
        return out, batching.not_mapped

    sample_arg, sample_bd = mapped[0]
    batch_size = sample_arg.shape[sample_bd]

    canonical_args = [
        batching.bdim_at_front(arg, bd, batch_size)
        for arg, bd in zip(batched_args, batch_dims)
    ]

    data_rank = canonical_args[0].ndim - 1
    axis_norm = axis if axis >= 0 else axis + (data_rank + 1)
    stack_axis = axis_norm + 1

    expanded = [jnp.expand_dims(arg, axis=stack_axis) for arg in canonical_args]
    stacked = jnp.concatenate(expanded, axis=stack_axis)
    return stacked, 0


@JnpStackPlugin._PRIM.def_impl
def _stack_impl(arrays: Iterable[ArrayLike], *, axis: int = 0) -> jax.Array:
    orig = get_orig_impl(JnpStackPlugin._PRIM, JnpStackPlugin._FUNC_NAME)
    return orig(arrays, axis=axis)


JnpStackPlugin._PRIM.def_abstract_eval(JnpStackPlugin.abstract_eval)


batching.primitive_batchers[JnpStackPlugin._PRIM] = _stack_batch_rule
