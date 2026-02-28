# jax2onnx/plugins/jax/numpy/argmax.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._arg_utils import lower_arg_reduction
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_ARGMAX_PRIM: Final = make_jnp_primitive("jax.numpy.argmax")


@register_primitive(
    jaxpr_primitive=_ARGMAX_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmax.html",
    onnx=[
        {
            "component": "ArgMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMax.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="argmax",
    testcases=[
        {
            "testcase": "jnp_argmax_axis1",
            "callable": lambda x: jnp.argmax(x, axis=1),
            "input_shapes": [(3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["ArgMax:3 -> Cast:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_argmax_axis0_bool",
            "callable": lambda x: jnp.argmax(x, axis=0),
            "input_values": [
                np.array(
                    [[False, True, False], [True, False, True], [False, False, False]],
                    dtype=np.bool_,
                )
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Cast:3x3 -> ArgMax:3 -> Cast:3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpArgmaxPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _ARGMAX_PRIM
    _FUNC_NAME: ClassVar[str] = "argmax"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        a: core.AbstractValue,
        *,
        axes: tuple[int, ...],
        keepdims: bool,
        index_dtype: np.dtype[Any],
        select_last_index: int,
    ) -> core.ShapedArray:
        del index_dtype, select_last_index
        orig = get_orig_impl(JnpArgmaxPlugin._PRIM, JnpArgmaxPlugin._FUNC_NAME)
        axis = int(axes[0])
        shape_dtype = jax.ShapeDtypeStruct(tuple(a.shape), np.dtype(a.dtype))
        out = jax.eval_shape(
            lambda x: orig(x, axis=axis, keepdims=keepdims), shape_dtype
        )
        return core.ShapedArray(tuple(out.shape), np.dtype(out.dtype))

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_arg_reduction(ctx, eqn, op_name="ArgMax", name_prefix="jnp_argmax")

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.argmax not found for monkey patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                axis: int | None = None,
                out: Any = None,
                keepdims: bool | None = False,
            ) -> jax.Array:
                if out is not None:
                    return orig(a, axis=axis, out=out, keepdims=keepdims)
                if axis is None:
                    return orig(a, axis=axis, out=out, keepdims=keepdims)
                if bool(keepdims):
                    return orig(a, axis=axis, out=out, keepdims=keepdims)

                index_dtype: np.dtype[Any] = np.dtype(
                    np.int64 if bool(jax.config.read("jax_enable_x64")) else np.int32
                )
                return cls._PRIM.bind(
                    jnp.asarray(a),
                    axes=(int(axis),),
                    keepdims=False,
                    index_dtype=index_dtype,
                    select_last_index=0,
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


@JnpArgmaxPlugin._PRIM.def_impl
def _argmax_impl(
    a: ArrayLike,
    *,
    axes: tuple[int, ...],
    keepdims: bool,
    index_dtype: np.dtype[Any],
    select_last_index: int,
) -> jax.Array:
    del index_dtype, select_last_index
    orig = get_orig_impl(JnpArgmaxPlugin._PRIM, JnpArgmaxPlugin._FUNC_NAME)
    return orig(a, axis=int(axes[0]), keepdims=keepdims)


JnpArgmaxPlugin._PRIM.def_abstract_eval(JnpArgmaxPlugin.abstract_eval)


def _argmax_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[object, ...],
    *,
    axes: tuple[int, ...],
    keepdims: bool,
    index_dtype: np.dtype[Any],
    select_last_index: int,
) -> tuple[jax.Array, int]:
    (operand,), (bdim,) = batched_args, batch_dims
    if bdim is batching.not_mapped:
        out = JnpArgmaxPlugin._PRIM.bind(
            operand,
            axes=axes,
            keepdims=keepdims,
            index_dtype=index_dtype,
            select_last_index=select_last_index,
        )
        return out, 0
    if not isinstance(bdim, int):
        raise TypeError(f"Unexpected batch dim for argmax: {bdim!r}")
    axis_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, axis_size)

    shifted_axes = tuple(int(ax) + 1 for ax in axes)
    out = JnpArgmaxPlugin._PRIM.bind(
        operand,
        axes=shifted_axes,
        keepdims=keepdims,
        index_dtype=index_dtype,
        select_last_index=select_last_index,
    )
    return out, 0


batching.primitive_batchers[JnpArgmaxPlugin._PRIM] = _argmax_batch_rule
