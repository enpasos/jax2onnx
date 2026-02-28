# jax2onnx/plugins/jax/numpy/argmin.py

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


_ARGMIN_PRIM: Final = make_jnp_primitive("jax.numpy.argmin")


@register_primitive(
    jaxpr_primitive=_ARGMIN_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmin.html",
    onnx=[
        {
            "component": "ArgMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMin.html",
        }
    ],
    since="0.12.4",
    context="primitives.jnp",
    component="argmin",
    testcases=[
        {
            "testcase": "jnp_argmin_axis1",
            "callable": lambda x: jnp.argmin(x, axis=1),
            "input_shapes": [(3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["ArgMin:3 -> Cast:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_argmin_axis0",
            "callable": lambda x: jnp.argmin(x, axis=0),
            "input_shapes": [(3, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["ArgMin:3 -> Cast:3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpArgminPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _ARGMIN_PRIM
    _FUNC_NAME: ClassVar[str] = "argmin"
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
        orig = get_orig_impl(JnpArgminPlugin._PRIM, JnpArgminPlugin._FUNC_NAME)
        axis = int(axes[0])
        shape_dtype = jax.ShapeDtypeStruct(tuple(a.shape), np.dtype(a.dtype))
        out = jax.eval_shape(
            lambda x: orig(x, axis=axis, keepdims=keepdims), shape_dtype
        )
        return core.ShapedArray(tuple(out.shape), np.dtype(out.dtype))

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_arg_reduction(ctx, eqn, op_name="ArgMin", name_prefix="jnp_argmin")

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.argmin not found for monkey patching")
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


@JnpArgminPlugin._PRIM.def_impl
def _argmin_impl(
    a: ArrayLike,
    *,
    axes: tuple[int, ...],
    keepdims: bool,
    index_dtype: np.dtype[Any],
    select_last_index: int,
) -> jax.Array:
    del index_dtype, select_last_index
    orig = get_orig_impl(JnpArgminPlugin._PRIM, JnpArgminPlugin._FUNC_NAME)
    return orig(a, axis=int(axes[0]), keepdims=keepdims)


JnpArgminPlugin._PRIM.def_abstract_eval(JnpArgminPlugin.abstract_eval)


def _argmin_batch_rule(
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
        out = JnpArgminPlugin._PRIM.bind(
            operand,
            axes=axes,
            keepdims=keepdims,
            index_dtype=index_dtype,
            select_last_index=select_last_index,
        )
        return out, 0
    if not isinstance(bdim, int):
        raise TypeError(f"Unexpected batch dim for argmin: {bdim!r}")
    axis_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, axis_size)

    shifted_axes = tuple(int(ax) + 1 for ax in axes)
    out = JnpArgminPlugin._PRIM.bind(
        operand,
        axes=shifted_axes,
        keepdims=keepdims,
        index_dtype=index_dtype,
        select_last_index=select_last_index,
    )
    return out, 0


batching.primitive_batchers[JnpArgminPlugin._PRIM] = _argmin_batch_rule
