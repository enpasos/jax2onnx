# jax2onnx/plugins/jax/numpy/amax.py

from __future__ import annotations

from collections.abc import Sequence as _Seq
from typing import Callable, ClassVar, Final

import jax
from jax import core
import jax.numpy as jnp
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.lax._reduce_utils import lower_reduction
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.jax.numpy._reduction_utils import (
    abstract_eval_via_orig_reduction,
    axis_arg_from_params,
    normalize_axis_to_axes,
    register_reduction_batch_rule,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_AMAX_PRIM: Final = make_jnp_primitive("jax.numpy.amax")


@register_primitive(
    jaxpr_primitive=_AMAX_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.amax.html",
    onnx=[
        {
            "component": "ReduceMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMax.html",
        }
    ],
    since="0.12.7",
    context="primitives.jnp",
    component="amax",
    testcases=[
        {
            "testcase": "jnp_amax_basic",
            "callable": lambda x: jnp.amax(x),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceMax"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_amax_axis",
            "callable": lambda x: jnp.amax(x, axis=1),
            "input_shapes": [(3, 4, 5)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ReduceMax:3x5",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_amax_keepdims",
            "callable": lambda x: jnp.amax(x, axis=0, keepdims=True),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceMax:1x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "amax_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.amax)(x),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpAmaxPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _AMAX_PRIM
    _FUNC_NAME: ClassVar[str] = "amax"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        axes: tuple[int, ...] | None = None,
        axes_is_tuple: bool = False,
        keepdims: bool = False,
    ) -> core.ShapedArray:
        return abstract_eval_via_orig_reduction(
            JnpAmaxPlugin._PRIM,
            JnpAmaxPlugin._FUNC_NAME,
            x,
            axes=axes,
            axes_is_tuple=axes_is_tuple,
            keepdims=keepdims,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_reduction(ctx, eqn, op_type="ReduceMax", allow_dtype_param=False)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.amax not found for monkey patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                axis: int | _Seq[int] | None = None,
                out: None = None,
                keepdims: bool = False,
                initial: ArrayLike | None = None,
                where: ArrayLike | None = None,
            ) -> jax.Array:
                if out is not None:
                    raise NotImplementedError(
                        "jnp.amax with 'out' is not supported for ONNX export"
                    )
                if initial is not None:
                    raise NotImplementedError(
                        "jnp.amax with 'initial' is not supported for ONNX export"
                    )
                if where is not None:
                    raise NotImplementedError(
                        "jnp.amax with 'where' is not supported for ONNX export"
                    )
                axes, axes_is_tuple = normalize_axis_to_axes(axis)
                return cls._PRIM.bind(
                    jnp.asarray(a),
                    axes=axes,
                    axes_is_tuple=axes_is_tuple,
                    keepdims=keepdims,
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


@JnpAmaxPlugin._PRIM.def_impl
def _amax_impl(
    a: ArrayLike,
    *,
    axes: tuple[int, ...] | None = None,
    axes_is_tuple: bool = False,
    keepdims: bool = False,
) -> jax.Array:
    orig = get_orig_impl(JnpAmaxPlugin._PRIM, JnpAmaxPlugin._FUNC_NAME)
    axis = axis_arg_from_params(axes, axes_is_tuple)
    return orig(jnp.asarray(a), axis=axis, keepdims=keepdims)


register_reduction_batch_rule(JnpAmaxPlugin._PRIM, jax.lax.reduce_max_p)
register_jvp_via_jax_jvp(JnpAmaxPlugin._PRIM, _amax_impl)
