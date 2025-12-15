# jax2onnx/plugins/jax/numpy/transpose.py

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, ClassVar, Final

import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import batching
from numpy.typing import ArrayLike

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_TRANSPOSE_PRIM: Final = make_jnp_primitive("jax.numpy.transpose")
AxesArg = Sequence[int] | int | None


def _normalize_axes(axes: AxesArg, rank: int) -> tuple[int, ...]:
    if axes is None:
        return tuple(reversed(range(rank)))
    if isinstance(axes, int):
        canonical = [axes]
        canonical.extend(i for i in range(rank) if i != axes)
        axes_seq: Iterable[int] = canonical
    else:
        axes_seq = axes
    norm: list[int] = []
    seen: set[int] = set()
    for ax in axes_seq:
        ax_int = int(ax)
        if ax_int < 0:
            ax_int += rank
        if ax_int < 0 or ax_int >= rank:
            raise ValueError(f"transpose axis {ax} out of bounds for rank {rank}")
        if ax_int in seen:
            raise ValueError("transpose axes must be a permutation")
        seen.add(ax_int)
        norm.append(ax_int)
    if len(norm) != rank:
        raise ValueError(
            f"transpose axes length {len(norm)} does not match input rank {rank}"
        )
    return tuple(norm)


@register_primitive(
    jaxpr_primitive=_TRANSPOSE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.transpose.html",
    onnx=[
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="transpose",
    testcases=[
        {
            "testcase": "transpose_basic",
            "callable": lambda a: jnp.transpose(a, axes=(1, 0)),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(
                ["Transpose:3x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_reverse_default",
            "callable": lambda a: jnp.transpose(a),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["Transpose:4x3x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_high_dim",
            "callable": lambda a: jnp.transpose(a, axes=(4, 3, 2, 1, 0)),
            "input_shapes": [(2, 3, 4, 5, 6)],
            "post_check_onnx_graph": EG(
                ["Transpose:6x5x4x3x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_3d",
            "callable": lambda a: jnp.transpose(a, axes=(0, 2, 1)),
            "input_shapes": [(3, 4, 5)],
            "post_check_onnx_graph": EG(
                ["Transpose:3x5x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_4d",
            "callable": lambda a: jnp.transpose(a, axes=(0, 2, 3, 1)),
            "input_shapes": [(2, 3, 4, 5)],
            "post_check_onnx_graph": EG(
                ["Transpose:2x4x5x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_no_axes",
            "callable": lambda a: jnp.transpose(a, axes=None),
            "input_shapes": [(4, 5, 6)],
            "post_check_onnx_graph": EG(
                ["Transpose:6x5x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_reverse",
            "callable": lambda a: jnp.transpose(a, axes=(2, 1, 0)),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["Transpose:4x3x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_square_matrix",
            "callable": lambda a: jnp.transpose(a),
            "input_shapes": [(5, 5)],
            "post_check_onnx_graph": EG(
                ["Transpose:5x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: jnp.transpose(y, (1, 0)))(x),
            "input_shapes": [(3, 2, 4)],
        },
    ],
)
class JnpTransposePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _TRANSPOSE_PRIM
    _FUNC_NAME: ClassVar[str] = "transpose"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue, axes: AxesArg = None) -> core.ShapedArray:
        rank = len(x.shape)
        axes_tuple = _normalize_axes(axes, rank)
        out_shape = tuple(x.shape[i] for i in axes_tuple)
        return core.ShapedArray(out_shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (arr_var,) = eqn.invars
        (out_var,) = eqn.outvars
        params = getattr(eqn, "params", {})
        axes_param = params.get("axes")

        arr_shape = tuple(getattr(arr_var.aval, "shape", ()))
        rank = len(arr_shape)
        axes_tuple = _normalize_axes(axes_param, rank)

        arr_val = ctx.get_value_for_var(
            arr_var, name_hint=ctx.fresh_name("transpose_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("transpose_out")
        )
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for transpose")

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "transpose_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("transpose_out")

        result = builder.Transpose(
            arr_val,
            _outputs=[desired_name],
            perm=list(map(int, axes_tuple)),
        )
        spec_type = getattr(out_spec, "type", None)
        if spec_type is not None:
            result.type = spec_type
        elif getattr(arr_val, "type", None) is not None:
            result.type = arr_val.type
        out_shape = tuple(arr_shape[i] for i in axes_tuple)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl___{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.transpose not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a: ArrayLike, axes: AxesArg = None) -> jax.Array:
                arr = jnp.asarray(a)
                axes_tuple = _normalize_axes(axes, arr.ndim)
                return cls._PRIM.bind(arr, axes=axes_tuple)

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


@JnpTransposePlugin._PRIM.def_impl
def _transpose_impl(a: ArrayLike, axes: AxesArg = None) -> jax.Array:
    orig = get_orig_impl(JnpTransposePlugin._PRIM, JnpTransposePlugin._FUNC_NAME)
    return orig(a, axes=axes)


JnpTransposePlugin._PRIM.def_abstract_eval(JnpTransposePlugin.abstract_eval)


BatchDim = int | type(batching.not_mapped)


def _transpose_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    axes: Sequence[int],
) -> tuple[jax.Array, BatchDim]:
    (a,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        out = JnpTransposePlugin._PRIM.bind(a, axes=axes)
        return out, batching.not_mapped

    batch_size = a.shape[bdim]
    a = batching.bdim_at_front(a, bdim, batch_size)
    perm = (0,) + tuple(int(ax) + 1 for ax in axes)
    out = JnpTransposePlugin._PRIM.bind(a, axes=perm)
    return out, 0


batching.primitive_batchers[JnpTransposePlugin._PRIM] = _transpose_batch_rule
