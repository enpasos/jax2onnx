# jax2onnx/plugins/jax/numpy/moveaxis.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, ClassVar, Final

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_fallback_jvp_rule
from jax2onnx.plugins.jax.numpy._common import make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_MOVEAXIS_PRIM: Final = make_jnp_primitive("jax.numpy.moveaxis")


def _to_tuple(axis: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(axis, Sequence) and not isinstance(axis, (str, bytes)):
        return tuple(int(a) for a in axis)
    return (int(axis),)


def _normalize_axis(axis: int, rank: int) -> int:
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"axis {axis} is out of bounds for rank {rank}")
    return axis


def _moveaxis_permutation(
    rank: int,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> tuple[int, ...]:
    source_t = tuple(_normalize_axis(a, rank) for a in _to_tuple(source))
    destination_t = tuple(_normalize_axis(a, rank) for a in _to_tuple(destination))
    if len(source_t) != len(destination_t):
        raise ValueError(
            "source and destination arguments must have the same number of axes"
        )
    if len(set(source_t)) != len(source_t):
        raise ValueError("source axes must be unique")
    if len(set(destination_t)) != len(destination_t):
        raise ValueError("destination axes must be unique")

    order = [axis for axis in range(rank) if axis not in source_t]
    for dest_axis, src_axis in sorted(zip(destination_t, source_t, strict=True)):
        order.insert(dest_axis, src_axis)
    return tuple(order)


@register_primitive(
    jaxpr_primitive=_MOVEAXIS_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.moveaxis.html",
    onnx=[
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="moveaxis",
    testcases=[
        {
            "testcase": "jnp_moveaxis_2d",
            "callable": lambda x: jnp.moveaxis(x, 0, 1),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(["Transpose:3x2"], no_unused_inputs=True),
        },
        {
            "testcase": "jnp_moveaxis_3d_tuple",
            "callable": lambda x: jnp.moveaxis(x, (0, 2), (2, 0)),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(["Transpose:4x3x2"], no_unused_inputs=True),
        },
        {
            "testcase": "jnp_moveaxis_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: jnp.moveaxis(y, 0, 1))(x),
            "input_shapes": [(3, 2, 4)],
            "post_check_onnx_graph": EG(["Transpose:3x4x2"], no_unused_inputs=True),
        },
    ],
)
class JnpMoveaxisPlugin(PrimitiveLeafPlugin):
    """IR-only lowering for ``jax.numpy.moveaxis`` via ``Transpose``."""

    _PRIM: ClassVar = _MOVEAXIS_PRIM
    _FUNC_NAME: ClassVar[str] = "moveaxis"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        a: core.AbstractValue,
        *,
        permutation: tuple[int, ...],
    ) -> core.ShapedArray:
        out_shape = tuple(a.shape[i] for i in permutation)
        return core.ShapedArray(out_shape, a.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (arr_var,) = eqn.invars
        (out_var,) = eqn.outvars
        permutation = tuple(int(p) for p in eqn.params.get("permutation", ()))

        arr_shape = tuple(getattr(getattr(arr_var, "aval", None), "shape", ()))
        arr_val = ctx.get_value_for_var(arr_var, name_hint=ctx.fresh_name("moveaxis_x"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("moveaxis_out")
        )

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for moveaxis")

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("moveaxis_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("moveaxis_out")

        result = builder.Transpose(
            arr_val,
            _outputs=[desired_name],
            perm=list(permutation),
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        elif getattr(arr_val, "type", None) is not None:
            result.type = arr_val.type

        out_shape = tuple(arr_shape[i] for i in permutation)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError(
                    "Original jnp.moveaxis not found for monkey patching"
                )
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                source: int | Sequence[int],
                destination: int | Sequence[int],
            ) -> jax.Array:
                arr = jnp.asarray(a)
                try:
                    permutation = _moveaxis_permutation(arr.ndim, source, destination)
                except Exception:
                    return orig(arr, source, destination)
                return cls._PRIM.bind(arr, permutation=permutation)

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


@JnpMoveaxisPlugin._PRIM.def_impl
def _moveaxis_impl(a: ArrayLike, *, permutation: tuple[int, ...]) -> jax.Array:
    return jnp.transpose(jnp.asarray(a), permutation)


JnpMoveaxisPlugin._PRIM.def_abstract_eval(JnpMoveaxisPlugin.abstract_eval)


def _moveaxis_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[Any, ...],
    *,
    permutation: tuple[int, ...],
) -> tuple[jax.Array, Any]:
    (arr,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        out = JnpMoveaxisPlugin._PRIM.bind(arr, permutation=permutation)
        return out, batching.not_mapped
    if not isinstance(bdim, int):
        raise TypeError(f"Unexpected batch dim for moveaxis: {bdim!r}")

    batch_size = arr.shape[bdim]
    arr = batching.bdim_at_front(arr, bdim, batch_size)
    batched_perm = (0,) + tuple(int(ax) + 1 for ax in permutation)
    out = JnpMoveaxisPlugin._PRIM.bind(arr, permutation=batched_perm)
    return out, 0


batching.primitive_batchers[JnpMoveaxisPlugin._PRIM] = _moveaxis_batch_rule
register_fallback_jvp_rule(JnpMoveaxisPlugin._PRIM, _moveaxis_impl)
