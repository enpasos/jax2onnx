# jax2onnx/plugins/jax/numpy/max.py

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


_MAX_PRIM: Final = make_jnp_primitive("jax.numpy.max")


@register_primitive(
    jaxpr_primitive=_MAX_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.max.html",
    onnx=[
        {
            "component": "ReduceMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMax.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="max",
    testcases=[
        {
            "testcase": "jnp_max_basic",
            "callable": lambda x: jnp.max(x),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceMax"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_max_axis",
            "callable": lambda x: jnp.max(x, axis=1),
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
            "testcase": "jnp_max_keepdims",
            "callable": lambda x: jnp.max(x, axis=0, keepdims=True),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceMax:1x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "max_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.max)(x),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpMaxPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _MAX_PRIM
    _FUNC_NAME: ClassVar[str] = "max"
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
            JnpMaxPlugin._PRIM,
            JnpMaxPlugin._FUNC_NAME,
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
                raise RuntimeError("Original jnp.max not found for monkey patching")
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
                        "jnp.max with 'out' is not supported for ONNX export"
                    )
                if initial is not None:
                    raise NotImplementedError(
                        "jnp.max with 'initial' is not supported for ONNX export"
                    )
                if where is not None:
                    raise NotImplementedError(
                        "jnp.max with 'where' is not supported for ONNX export"
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


@JnpMaxPlugin._PRIM.def_impl
def _max_impl(
    a: ArrayLike,
    *,
    axes: tuple[int, ...] | None = None,
    axes_is_tuple: bool = False,
    keepdims: bool = False,
) -> jax.Array:
    orig = get_orig_impl(JnpMaxPlugin._PRIM, JnpMaxPlugin._FUNC_NAME)
    axis = axis_arg_from_params(axes, axes_is_tuple)
    return orig(jnp.asarray(a), axis=axis, keepdims=keepdims)


register_reduction_batch_rule(JnpMaxPlugin._PRIM, jax.lax.reduce_max_p)
register_jvp_via_jax_jvp(JnpMaxPlugin._PRIM, _max_impl)
