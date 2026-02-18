# jax2onnx/plugins/jax/nn/hardmax.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, ClassVar, Final

import jax
from jax.interpreters import batching
import jax.numpy as jnp
from jax.extend.core import Primitive
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_HARDMAX_PRIM: Final[Primitive] = Primitive("jax.nn.hardmax")
_HARDMAX_PRIM.multiple_results = False


def _normalize_axis(axis: int, rank: int) -> int:
    axis_int = int(axis)
    if axis_int < 0:
        axis_int += rank
    if axis_int < 0 or axis_int >= rank:
        raise ValueError(f"hardmax axis={axis} is out of bounds for rank {rank}")
    return axis_int


def _hardmax_fallback(x: ArrayLike, axis: int = -1) -> ArrayLike:
    arr = jnp.asarray(x)
    axis_norm = _normalize_axis(axis, arr.ndim)
    indices = jnp.argmax(arr, axis=axis_norm)
    depth = arr.shape[axis_norm]
    one_hot = jax.nn.one_hot(indices, depth, axis=axis_norm, dtype=arr.dtype)
    return jnp.asarray(one_hot, dtype=arr.dtype)


def _hardmax_bind(x: ArrayLike, axis: int = -1) -> Any:
    return _HARDMAX_PRIM.bind(x, axis=int(axis))


@register_primitive(
    jaxpr_primitive=_HARDMAX_PRIM.name,
    jax_doc="https://onnx.ai/onnx/operators/onnx__Hardmax.html",
    onnx=[
        {
            "component": "Hardmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Hardmax.html",
        }
    ],
    since="0.12.1",
    context="primitives.nn",
    component="hardmax",
    testcases=[
        {
            "testcase": "jaxnn_hardmax_default_axis",
            "callable": lambda x: _hardmax_bind(x),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Hardmax:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_hardmax_axis0",
            "callable": lambda x: _hardmax_bind(x, axis=0),
            "input_shapes": [(3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Hardmax:3x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_hardmax_dynamic",
            "callable": lambda x: _hardmax_bind(x, axis=-1),
            "input_shapes": [("B", 3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Hardmax:Bx3x4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_hardmax_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: _hardmax_bind(y, axis=-1))(x),
            "input_shapes": [(3, 4)],
            "run_only_f32_variant": True,
        },
    ],
)
class HardmaxPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.hardmax`` to ONNX ``Hardmax``."""

    _PRIM: ClassVar[Primitive] = _HARDMAX_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue,
        axis: int = -1,
    ) -> jax.core.ShapedArray:
        _normalize_axis(int(axis), len(x.shape))
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        axis = int(eqn.params.get("axis", -1))

        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        axis_norm = _normalize_axis(axis, len(x_shape))

        x_val = ctx.get_value_for_var(
            x_var,
            name_hint=ctx.fresh_name("hardmax_in"),
        )
        out_spec = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("hardmax_out"),
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("hardmax_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("hardmax_out")

        result = ctx.builder.Hardmax(
            x_val,
            _outputs=[desired_name],
            axis=axis_norm,
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = getattr(x_val, "type", None)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            result.shape = getattr(x_val, "shape", None)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_value(
            orig: Callable[..., Any] | None,
        ) -> Callable[..., Any]:
            del orig

            def _patched(x: ArrayLike, axis: int = -1) -> Any:
                return cls._PRIM.bind(x, axis=int(axis))

            return _patched

        return [
            AssignSpec("jax.nn", "hardmax_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="hardmax",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="hardmax",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="hardmax",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@HardmaxPlugin._PRIM.def_impl
def _hardmax_impl(x: ArrayLike, axis: int = -1) -> ArrayLike:
    return _hardmax_fallback(x, axis=axis)


def _hardmax_batch_rule(
    batched_args: Sequence[Any],
    batch_dims: Sequence[Any],
    *,
    axis: int = -1,
) -> tuple[Any, Any]:
    (operand,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        out = HardmaxPlugin._PRIM.bind(operand, axis=axis)
        return out, batching.not_mapped

    batch_size = operand.shape[bdim]
    operand_front = batching.bdim_at_front(operand, bdim, batch_size)
    slice_rank = operand_front.ndim - 1
    axis_norm = _normalize_axis(axis, slice_rank)
    axis_with_batch = axis_norm + 1
    out = HardmaxPlugin._PRIM.bind(operand_front, axis=axis_with_batch)
    return out, 0


batching.primitive_batchers[HardmaxPlugin._PRIM] = _hardmax_batch_rule
