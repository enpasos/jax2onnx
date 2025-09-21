from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Sequence
from collections.abc import Sequence as _Seq

import numpy as np

import jax
import jax.numpy as jnp

from jax2onnx.plugins2.jax.lax._reduce_utils import lower_reduction
from jax2onnx.plugins2.jax.numpy._common import (
    get_orig_impl,
    make_jnp_primitive,
)
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_PROD_PRIM = make_jnp_primitive("jax.numpy.prod")


@register_primitive(
    jaxpr_primitive=_PROD_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.prod.html",
    onnx=[
        {
            "component": "ReduceProd",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceProd.html",
        }
    ],
    since="v0.8.0",
    context="primitives2.jnp",
    component="prod",
    testcases=[
        {
            "testcase": "jnp_prod_basic",
            "callable": lambda x: jnp.prod(x),
            "input_shapes": [(3, 4)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "jnp_prod_axis",
            "callable": lambda x: jnp.prod(x, axis=1),
            "input_shapes": [(3, 4, 5)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "jnp_prod_keepdims",
            "callable": lambda x: jnp.prod(x, axis=0, keepdims=True),
            "input_shapes": [(3, 4)],
            "use_onnx_ir": True,
        },
    ],
)
class JnpProdPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _PROD_PRIM
    _FUNC_NAME: ClassVar[str] = "prod"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x,
        *,
        axes: Sequence[int] | None = None,
        axes_is_tuple: bool = False,  # ignored, kept for signature symmetry
        dtype: np.dtype | None = None,
        keepdims: bool = False,
        initial=None,
        where=True,
    ):
        if initial is not None:
            raise NotImplementedError(
                "jnp.prod with 'initial' is not supported in ONNX lowering"
            )
        if where is not True:
            raise NotImplementedError(
                "jnp.prod with 'where' mask is not supported in ONNX lowering"
            )

        ndim = len(x.shape)
        out_dtype = np.dtype(dtype) if dtype is not None else np.dtype(x.dtype)

        if axes is None:
            axes_tuple = tuple(range(ndim))
        else:
            axes_tuple = tuple(
                int(ax) % ndim if ndim and int(ax) < 0 else int(ax) for ax in axes
            )

        if axes is None:
            if keepdims:
                out_shape = tuple(1 for _ in range(ndim))
            else:
                out_shape = ()
        else:
            if keepdims:
                out_shape = tuple(
                    1 if i in axes_tuple else dim for i, dim in enumerate(x.shape)
                )
            else:
                out_shape = tuple(
                    dim for i, dim in enumerate(x.shape) if i not in axes_tuple
                )

        return jax.core.ShapedArray(out_shape, out_dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_reduction(ctx, eqn, op_type="ReduceProd", allow_dtype_param=True)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.prod not found for monkey patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a, axis=None, dtype=None, keepdims=False, *, initial=None, where=True
            ):
                if initial is not None:
                    raise NotImplementedError(
                        "jnp.prod with 'initial' is not supported for ONNX export"
                    )
                if where is not True:
                    raise NotImplementedError(
                        "jnp.prod with 'where' is not supported for ONNX export"
                    )
                axes_param = None
                axes_is_tuple = False
                if axis is not None:
                    if isinstance(axis, _Seq) and not isinstance(axis, (str, bytes)):
                        axes_param = tuple(int(ax) for ax in axis)
                        axes_is_tuple = True
                    else:
                        axes_param = (int(axis),)
                        axes_is_tuple = False
                return cls._PRIM.bind(
                    a,
                    axes=axes_param,
                    axes_is_tuple=axes_is_tuple,
                    dtype=dtype,
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


@JnpProdPlugin._PRIM.def_impl
def _prod_impl(
    a,
    *,
    axes=None,
    axes_is_tuple: bool = False,
    dtype=None,
    keepdims=False,
    initial=None,
    where=True,
):
    if initial is not None:
        raise NotImplementedError(
            "jnp.prod with 'initial' is not supported for ONNX export"
        )
    if where is not True:
        raise NotImplementedError(
            "jnp.prod with 'where' is not supported for ONNX export"
        )

    try:
        orig = get_orig_impl(JnpProdPlugin._PRIM, JnpProdPlugin._FUNC_NAME)
    except RuntimeError:
        orig = jnp.prod

    axis_arg = None
    if axes is not None:
        axis_vals = tuple(int(ax) for ax in axes)
        axis_arg = axis_vals if axes_is_tuple else axis_vals[0]

    return orig(a, axis=axis_arg, dtype=dtype, keepdims=keepdims)


JnpProdPlugin._PRIM.def_abstract_eval(JnpProdPlugin.abstract_eval)
