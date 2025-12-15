# jax2onnx/plugins/jax/numpy/unstack.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from jax.interpreters import batching
from numpy.typing import ArrayLike

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_UNSTACK_PRIM: Final = make_jnp_primitive("jax.numpy.unstack")
_UNSTACK_PRIM.multiple_results = True


def _normalize_axis(axis: int, rank: int) -> int:
    ax = int(axis)
    if ax < 0:
        ax += rank
    if ax < 0 or ax >= rank:
        raise ValueError(f"axis {axis} out of bounds for rank {rank}")
    return ax


@register_primitive(
    jaxpr_primitive=_UNSTACK_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unstack.html",
    onnx=[
        {
            "component": "Split",
            "doc": "https://onnx.ai/onnx/operators/onnx__Split.html",
        },
        {
            "component": "Squeeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
        },
    ],
    since="v0.7.1",
    context="primitives.jnp",
    component="unstack",
    testcases=[
        {
            "testcase": "unstack_axis_0",
            "callable": lambda x: jnp.unstack(x, axis=0),
            "input_values": [np.array([[1, 2], [3, 4]], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "Split:1x2 -> Squeeze:2",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "unstack_axis_0_f64",
            "callable": lambda x: jnp.unstack(x, axis=0),
            "input_values": [np.array([[1, 2], [3, 4]], dtype=np.float64)],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "Split:1x2 -> Squeeze:2",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "unstack_axis_1",
            "callable": lambda x: jnp.unstack(x, axis=1),
            "input_values": [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}},
                        "path": "Split:2x1 -> Squeeze:2",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "unstack_axis_1_f64",
            "callable": lambda x: jnp.unstack(x, axis=1),
            "input_values": [
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
            ],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}},
                        "path": "Split:2x1 -> Squeeze:2",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "unstack_negative_axis",
            "callable": lambda x: jnp.unstack(x, axis=-1),
            "input_values": [np.array([[[1, 2], [3, 4]]], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 2.0}},
                        "path": "Split:1x2x1 -> Squeeze:1x2",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "unstack_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: jnp.unstack(y, axis=0))(x),
            "input_shapes": [(3, 2, 4)],
        },
    ],
)
class JnpUnstackPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _UNSTACK_PRIM
    _FUNC_NAME: ClassVar[str] = "unstack"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue, *, axis: int = 0
    ) -> tuple[jax.core.ShapedArray, ...]:
        rank = len(x.shape)
        axis_norm = _normalize_axis(axis, rank)
        size = x.shape[axis_norm]
        if not isinstance(size, (int, np.integer)):
            raise core.InconclusiveDimensionOperation(
                "jnp.unstack requires static axis length"
            )
        out_shape = x.shape[:axis_norm] + x.shape[axis_norm + 1 :]
        return tuple(core.ShapedArray(out_shape, x.dtype) for _ in range(int(size)))

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        params = getattr(eqn, "params", {})
        axis_param = params.get("axis", 0)

        (arr_var,) = eqn.invars
        out_vars = list(eqn.outvars)

        arr_shape = tuple(getattr(arr_var.aval, "shape", ()))
        axis = _normalize_axis(axis_param, len(arr_shape))
        size = arr_shape[axis]
        if not isinstance(size, (int, np.integer)):
            raise TypeError("jnp.unstack requires static axis length for lowering")

        arr_val = ctx.get_value_for_var(arr_var, name_hint=ctx.fresh_name("unstack_in"))
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for unstack lowering"
            )

        split_sizes = _const_i64(
            ctx, np.asarray([1] * int(size), dtype=np.int64), "unstack_split_sizes"
        )

        split_shape = list(arr_shape)
        split_shape[axis] = 1
        split_outputs = [ctx.fresh_name("unstack_split_out") for _ in range(int(size))]
        split_values = builder.Split(
            arr_val,
            split_sizes,
            _outputs=split_outputs,
            axis=int(axis),
        )
        for split_val in split_values:
            if getattr(arr_val, "type", None) is not None:
                split_val.type = arr_val.type
            _stamp_type_and_shape(split_val, tuple(split_shape))
            _ensure_value_metadata(ctx, split_val)

        axes_val = _const_i64(
            ctx, np.asarray([axis], dtype=np.int64), "unstack_squeeze_axes"
        )

        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")

        for split_val, out_var in zip(split_values, out_vars):
            out_spec = ctx.get_value_for_var(
                out_var, name_hint=ctx.fresh_name("unstack_out")
            )
            out_name = getattr(out_spec, "name", None) or ctx.fresh_name("unstack_out")
            squeezed = builder.Squeeze(
                split_val,
                axes_val,
                _outputs=[out_name],
            )
            if getattr(arr_val, "type", None) is not None:
                squeezed.type = arr_val.type
            target_shape = tuple(getattr(out_var.aval, "shape", ()))
            _stamp_type_and_shape(squeezed, target_shape)
            _ensure_value_metadata(ctx, squeezed)
            bind_value(out_var, squeezed)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jnp.unstack not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(x: ArrayLike, axis: int = 0) -> ArrayLike:
                return cls._PRIM.bind(x, axis=axis)

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


@JnpUnstackPlugin._PRIM.def_impl
def _unstack_impl(x: ArrayLike, axis: int = 0):
    orig = get_orig_impl(JnpUnstackPlugin._PRIM, JnpUnstackPlugin._FUNC_NAME)
    return orig(x, axis=axis)


JnpUnstackPlugin._PRIM.def_abstract_eval(JnpUnstackPlugin.abstract_eval)


BatchDim = int | type(batching.not_mapped)


def _unstack_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    axis: int = 0,
) -> tuple[tuple[jax.Array, ...], tuple[BatchDim, ...]]:
    (operand,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        outs = JnpUnstackPlugin._PRIM.bind(operand, axis=axis)
        return outs, tuple(batching.not_mapped for _ in outs)

    batch_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, batch_size)

    slice_rank = operand.ndim - 1
    axis_int = int(axis)
    if slice_rank:
        axis_norm = axis_int % slice_rank
    else:
        axis_norm = 0
    axis_full = axis_norm + 1

    outs = JnpUnstackPlugin._PRIM.bind(operand, axis=axis_full)
    return outs, tuple(0 for _ in outs)


batching.primitive_batchers[JnpUnstackPlugin._PRIM] = _unstack_batch_rule
