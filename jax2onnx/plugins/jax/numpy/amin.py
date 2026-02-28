# jax2onnx/plugins/jax/numpy/amin.py

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


_AMIN_PRIM: Final = make_jnp_primitive("jax.numpy.amin")


@register_primitive(
    jaxpr_primitive=_AMIN_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.amin.html",
    onnx=[
        {
            "component": "ReduceMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMin.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="amin",
    testcases=[
        {
            "testcase": "jnp_amin_basic",
            "callable": lambda x: jnp.amin(x),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceMin"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_amin_axis",
            "callable": lambda x: jnp.amin(x, axis=1),
            "input_shapes": [(3, 4, 5)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ReduceMin:3x5",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_amin_keepdims",
            "callable": lambda x: jnp.amin(x, axis=0, keepdims=True),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceMin:1x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "amin_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.amin)(x),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpAminPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _AMIN_PRIM
    _FUNC_NAME: ClassVar[str] = "amin"
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
            JnpAminPlugin._PRIM,
            JnpAminPlugin._FUNC_NAME,
            x,
            axes=axes,
            axes_is_tuple=axes_is_tuple,
            keepdims=keepdims,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_reduction(ctx, eqn, op_type="ReduceMin", allow_dtype_param=False)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.amin not found for monkey patching")
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
                        "jnp.amin with 'out' is not supported for ONNX export"
                    )
                if initial is not None:
                    raise NotImplementedError(
                        "jnp.amin with 'initial' is not supported for ONNX export"
                    )
                if where is not None:
                    raise NotImplementedError(
                        "jnp.amin with 'where' is not supported for ONNX export"
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


@JnpAminPlugin._PRIM.def_impl
def _amin_impl(
    a: ArrayLike,
    *,
    axes: tuple[int, ...] | None = None,
    axes_is_tuple: bool = False,
    keepdims: bool = False,
) -> jax.Array:
    orig = get_orig_impl(JnpAminPlugin._PRIM, JnpAminPlugin._FUNC_NAME)
    axis = axis_arg_from_params(axes, axes_is_tuple)
    return orig(jnp.asarray(a), axis=axis, keepdims=keepdims)


register_reduction_batch_rule(JnpAminPlugin._PRIM, jax.lax.reduce_min_p)
register_jvp_via_jax_jvp(JnpAminPlugin._PRIM, _amin_impl)
