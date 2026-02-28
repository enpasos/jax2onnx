# jax2onnx/plugins/jax/numpy/all.py

from __future__ import annotations

from collections.abc import Sequence as _Seq
from typing import Callable, ClassVar, Final

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._reduce_utils import lower_boolean_reduction
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.jax.numpy._reduction_utils import (
    abstract_eval_via_orig_reduction,
    axis_arg_from_params,
    normalize_axis_to_axes,
    register_reduction_batch_rule,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_ALL_PRIM: Final = make_jnp_primitive("jax.numpy.all")


@register_primitive(
    jaxpr_primitive=_ALL_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.all.html",
    onnx=[
        {
            "component": "ReduceMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMin.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="all",
    testcases=[
        {
            "testcase": "jnp_all_basic",
            "callable": lambda x: jnp.all(x),
            "input_values": [np.array([True, True, False], dtype=np.bool_)],
            "expected_output_dtypes": [np.bool_],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> ReduceMin -> Cast"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_all_axis_keepdims",
            "callable": lambda x: jnp.all(x, axis=1, keepdims=True),
            "input_values": [np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int32)],
            "expected_output_dtypes": [np.bool_],
            "post_check_onnx_graph": EG(
                ["ReduceMin:2x1 -> Cast:2x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "all_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.all)(x),
            "input_values": [np.array([[True, False], [True, True]], dtype=np.bool_)],
        },
    ],
)
class JnpAllPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _ALL_PRIM
    _FUNC_NAME: ClassVar[str] = "all"
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
            JnpAllPlugin._PRIM,
            JnpAllPlugin._FUNC_NAME,
            x,
            axes=axes,
            axes_is_tuple=axes_is_tuple,
            keepdims=keepdims,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_boolean_reduction(ctx, eqn, mode="reduce_and")

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.all not found for monkey patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                axis: int | _Seq[int] | None = None,
                out: None = None,
                keepdims: bool = False,
                *,
                where: ArrayLike | None = None,
            ) -> jax.Array:
                if out is not None:
                    raise NotImplementedError(
                        "jnp.all with 'out' is not supported for ONNX export"
                    )
                if where is not None:
                    raise NotImplementedError(
                        "jnp.all with 'where' is not supported for ONNX export"
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


@JnpAllPlugin._PRIM.def_impl
def _all_impl(
    a: ArrayLike,
    *,
    axes: tuple[int, ...] | None = None,
    axes_is_tuple: bool = False,
    keepdims: bool = False,
) -> jax.Array:
    orig = get_orig_impl(JnpAllPlugin._PRIM, JnpAllPlugin._FUNC_NAME)
    axis = axis_arg_from_params(axes, axes_is_tuple)
    return orig(jnp.asarray(a), axis=axis, keepdims=keepdims)


register_reduction_batch_rule(JnpAllPlugin._PRIM, jax.lax.reduce_and_p)
