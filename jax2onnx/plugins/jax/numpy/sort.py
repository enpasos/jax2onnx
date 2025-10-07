# jax2onnx/plugins/jax/numpy/sort.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import core

from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_SORT_PRIM = make_jnp_primitive("jax.numpy.sort")


def _sort_eval(x, axis=-1):
    orig = getattr(_SORT_PRIM, "__orig_impl__sort", jnp.sort)
    spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
    result = jax.eval_shape(lambda arr: orig(arr, axis=axis), spec)
    return result


@register_primitive(
    jaxpr_primitive=_SORT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sort.html",
    onnx=[
        {"component": "Sort", "doc": "https://onnx.ai/onnx/operators/onnx__Sort.html"}
    ],
    since="v0.9.0",
    context="primitives.jnp",
    component="sort",
    testcases=[
        {
            "testcase": "sort_1d",
            "callable": lambda x: jnp.sort(x),
            "input_shapes": [(5,)],
        },
        {
            "testcase": "sort_2d_axis0",
            "callable": lambda x: jnp.sort(x, axis=0),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "sort_basic",
            "callable": lambda x: jnp.sort(x, axis=1),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpSortPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SORT_PRIM
    _FUNC_NAME: ClassVar[str] = "sort"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *, axis=-1, kind=None, order=None):
        if kind not in (None, "stable", "mergesort"):
            raise NotImplementedError("Only default/stable sorts supported")
        if order is not None:
            raise NotImplementedError("jnp.sort order parameter is not supported")
        result = _sort_eval(x, axis=axis)
        return core.ShapedArray(result.shape, result.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        params = getattr(eqn, "params", {})
        axis = int(params.get("axis", -1))
        kind = params.get("kind", None)
        order = params.get("order", None)
        if order is not None:
            raise NotImplementedError("jnp.sort order parameter is not supported")
        if kind not in (None, "stable", "mergesort"):
            raise NotImplementedError("Only default/stable sorts supported")

        (arr_var,) = eqn.invars
        (out_var,) = eqn.outvars

        arr_shape = tuple(getattr(arr_var.aval, "shape", ()))
        if not arr_shape:
            axis = 0
        else:
            if axis < 0:
                axis += len(arr_shape)
            if axis < 0 or axis >= len(arr_shape):
                raise ValueError("axis out of bounds")

        arr_val = ctx.get_value_for_var(arr_var, name_hint=ctx.fresh_name("sort_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("sort_out"))
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for sort lowering")

        axis_size = arr_shape[axis] if arr_shape else 1
        if not isinstance(axis_size, (int, np.integer)):
            raise TypeError("jnp.sort requires static axis length")
        k_val = _const_i64(ctx, np.asarray([axis_size], dtype=np.int64), "sort_k")
        values, _indices = builder.TopK(
            arr_val,
            k_val,
            _outputs=[
                ctx.fresh_name("sort_values"),
                ctx.fresh_name("sort_indices"),
            ],
            axis=int(axis),
            largest=0,
            sorted=1,
        )
        target_shape = tuple(getattr(out_var.aval, "shape", ()))
        if getattr(arr_val, "type", None) is not None:
            values.type = arr_val.type
        _stamp_type_and_shape(values, target_shape)
        _ensure_value_info(ctx, values)

        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("sort_out")
        result = builder.Identity(
            values,
            _outputs=[out_name],
        )
        if getattr(arr_val, "type", None) is not None:
            result.type = arr_val.type
        _stamp_type_and_shape(result, target_shape)
        _ensure_value_info(ctx, result)
        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(out_var, result)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.sort not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a, axis=-1, kind=None, order=None):
                if order is not None:
                    raise NotImplementedError(
                        "jnp.sort order parameter is not supported"
                    )
                if kind not in (None, "stable", "mergesort"):
                    raise NotImplementedError("Only default/stable sorts supported")
                return cls._PRIM.bind(a, axis=axis, kind=kind, order=order)

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


@JnpSortPlugin._PRIM.def_impl
def _sort_impl(a, axis=-1, kind=None, order=None):
    orig = get_orig_impl(JnpSortPlugin._PRIM, JnpSortPlugin._FUNC_NAME)
    return orig(a, axis=axis, kind=kind, order=order)


JnpSortPlugin._PRIM.def_abstract_eval(JnpSortPlugin.abstract_eval)
