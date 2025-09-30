from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax import core

from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.jax.lax._index_utils import _const_i64
from jax2onnx.plugins2.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_SQUEEZE_PRIM = make_jnp_primitive("jax.numpy.squeeze")


def _normalize_axes(axes: int | Sequence[int] | None, rank: int) -> tuple[int, ...]:
    if axes is None:
        return tuple()
    if isinstance(axes, int):
        axes_iter: Iterable[int] = (axes,)
    else:
        axes_iter = axes
    normalized = []
    for ax in axes_iter:
        ax_int = int(ax)
        if ax_int < 0:
            ax_int += rank
        if ax_int < 0 or ax_int >= rank:
            raise ValueError(f"axis {ax} out of bounds for rank {rank}")
        if ax_int not in normalized:
            normalized.append(ax_int)
    return tuple(normalized)


@register_primitive(
    jaxpr_primitive=_SQUEEZE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.squeeze.html",
    onnx=[
        {
            "component": "Squeeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
        }
    ],
    since="v0.9.0",
    context="primitives2.jnp",
    component="squeeze",
    testcases=[
        {
            "testcase": "squeeze_single_dim",
            "callable": lambda a: jnp.squeeze(a, axis=0),
            "input_shapes": [(1, 49, 10)],
        },
        {
            "testcase": "squeeze_multiple_dims",
            "callable": lambda a: jnp.squeeze(a, axis=(0, 2)),
            "input_shapes": [(1, 49, 1, 10)],
        },
        {
            "testcase": "squeeze_vit_output",
            "callable": lambda a: jnp.squeeze(a, axis=1),
            "input_shapes": [(1, 1, 10)],
        },
        {
            "testcase": "squeeze_dynamic_batch",
            "callable": lambda a: jnp.squeeze(a, axis=1),
            "input_shapes": [("B", 1, 10)],
        },
        {
            "testcase": "squeeze_all_dims",
            "callable": lambda a: jnp.squeeze(a),
            "input_shapes": [(1, 1, 1)],
        },
        {
            "testcase": "squeeze_negative_axis",
            "callable": lambda a: jnp.squeeze(a, axis=-1),
            "input_shapes": [(1, 49, 1)],
        },
        {
            "testcase": "squeeze_negative_axis_tuple",
            "callable": lambda a: jnp.squeeze(a, axis=(-1, -3)),
            "input_shapes": [(1, 49, 1)],
        },
        {
            "testcase": "squeeze_dynamic_and_negative_axis",
            "callable": lambda a: jnp.squeeze(a, axis=(-1, -3)),
            "input_shapes": [(1, "B", 1)],
        },
    ],
)
class JnpSqueezePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SQUEEZE_PRIM
    _FUNC_NAME: ClassVar[str] = "squeeze"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *, axis=None):
        storage_slot = f"__orig_impl__{JnpSqueezePlugin._FUNC_NAME}"
        orig = getattr(JnpSqueezePlugin._PRIM, storage_slot, jnp.squeeze)
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        result = jax.eval_shape(lambda arr: orig(arr, axis=axis), spec)
        return core.ShapedArray(result.shape, result.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        params = getattr(eqn, "params", {})
        axis_param = params.get("axes", params.get("axis"))

        (arr_var,) = eqn.invars
        (out_var,) = eqn.outvars

        arr_shape = tuple(getattr(arr_var.aval, "shape", ()))
        rank = len(arr_shape)

        axes = _normalize_axes(axis_param, rank)
        if not axes:
            # Squeeze all singleton dims
            axes = tuple(
                i for i, d in enumerate(arr_shape) if isinstance(d, int) and d == 1
            )

        arr_val = ctx.get_value_for_var(arr_var, name_hint=ctx.fresh_name("squeeze_in"))
        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("squeeze_out")
        )

        if not axes:
            # nothing to squeeze
            ctx.add_node(
                ir.Node(
                    op_type="Identity",
                    domain="",
                    inputs=[arr_val],
                    outputs=[out_val],
                    name=ctx.fresh_name("Identity"),
                )
            )
            _stamp_type_and_shape(out_val, tuple(arr_shape))
            _ensure_value_info(ctx, out_val)
            return

        axes_vals = _const_i64(ctx, np.asarray(axes, dtype=np.int64), "squeeze_axes")
        ctx.add_node(
            ir.Node(
                op_type="Squeeze",
                domain="",
                inputs=[arr_val, axes_vals],
                outputs=[out_val],
                name=ctx.fresh_name("Squeeze"),
            )
        )
        target_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(out_val, target_shape)
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.squeeze not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a, axis=None):
                arr = jnp.asarray(a)
                return cls._PRIM.bind(arr, axis=axis)

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


@JnpSqueezePlugin._PRIM.def_impl
def _squeeze_impl(a, axis=None):
    orig = get_orig_impl(JnpSqueezePlugin._PRIM, JnpSqueezePlugin._FUNC_NAME)
    return orig(a, axis=axis)


JnpSqueezePlugin._PRIM.def_abstract_eval(JnpSqueezePlugin.abstract_eval)
