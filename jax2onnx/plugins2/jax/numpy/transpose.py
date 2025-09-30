from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Iterable, Sequence

import jax.numpy as jnp
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType
from jax import core

from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_TRANSPOSE_PRIM = make_jnp_primitive("jax.numpy.transpose")


def _normalize_axes(axes: Sequence[int] | int | None, rank: int) -> tuple[int, ...]:
    if axes is None:
        return tuple(reversed(range(rank)))
    if isinstance(axes, int):
        canonical = [axes]
        canonical.extend(i for i in range(rank) if i != axes)
        axes_seq: Iterable[int] = canonical
    else:
        axes_seq = axes
    norm: list[int] = []
    seen: set[int] = set()
    for ax in axes_seq:
        ax_int = int(ax)
        if ax_int < 0:
            ax_int += rank
        if ax_int < 0 or ax_int >= rank:
            raise ValueError(f"transpose axis {ax} out of bounds for rank {rank}")
        if ax_int in seen:
            raise ValueError("transpose axes must be a permutation")
        seen.add(ax_int)
        norm.append(ax_int)
    if len(norm) != rank:
        raise ValueError(
            f"transpose axes length {len(norm)} does not match input rank {rank}"
        )
    return tuple(norm)


@register_primitive(
    jaxpr_primitive=_TRANSPOSE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.transpose.html",
    onnx=[
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        }
    ],
    since="v0.9.0",
    context="primitives2.jnp",
    component="transpose",
    testcases=[
        {
            "testcase": "transpose_basic",
            "callable": lambda a: jnp.transpose(a, axes=(1, 0)),
            "input_shapes": [(2, 3)],
        },
        {
            "testcase": "transpose_reverse_default",
            "callable": lambda a: jnp.transpose(a),
            "input_shapes": [(2, 3, 4)],
        },
        {
            "testcase": "transpose_high_dim",
            "callable": lambda a: jnp.transpose(a, axes=(4, 3, 2, 1, 0)),
            "input_shapes": [(2, 3, 4, 5, 6)],
        },
        {
            "testcase": "transpose_3d",
            "callable": lambda a: jnp.transpose(a, axes=(0, 2, 1)),
            "input_shapes": [(3, 4, 5)],
        },
        {
            "testcase": "transpose_4d",
            "callable": lambda a: jnp.transpose(a, axes=(0, 2, 3, 1)),
            "input_shapes": [(2, 3, 4, 5)],
        },
        {
            "testcase": "transpose_no_axes",
            "callable": lambda a: jnp.transpose(a, axes=None),
            "input_shapes": [(4, 5, 6)],
        },
        {
            "testcase": "transpose_reverse",
            "callable": lambda a: jnp.transpose(a, axes=(2, 1, 0)),
            "input_shapes": [(2, 3, 4)],
        },
        {
            "testcase": "transpose_square_matrix",
            "callable": lambda a: jnp.transpose(a),
            "input_shapes": [(5, 5)],
        },
    ],
)
class JnpTransposePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _TRANSPOSE_PRIM
    _FUNC_NAME: ClassVar[str] = "transpose"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, axes=None):
        rank = len(x.shape)
        axes_tuple = _normalize_axes(axes, rank)
        out_shape = tuple(x.shape[i] for i in axes_tuple)
        return core.ShapedArray(out_shape, x.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        (arr_var,) = eqn.invars
        (out_var,) = eqn.outvars
        params = getattr(eqn, "params", {})
        axes_param = params.get("axes")

        arr_shape = tuple(getattr(arr_var.aval, "shape", ()))
        rank = len(arr_shape)
        axes_tuple = _normalize_axes(axes_param, rank)

        arr_val = ctx.get_value_for_var(
            arr_var, name_hint=ctx.fresh_name("transpose_in")
        )
        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("transpose_out")
        )

        ctx.add_node(
            ir.Node(
                op_type="Transpose",
                domain="",
                inputs=[arr_val],
                outputs=[out_val],
                name=ctx.fresh_name("Transpose"),
                attributes=[
                    IRAttr("perm", IRAttrType.INTS, list(map(int, axes_tuple)))
                ],
            )
        )

        out_shape = tuple(arr_shape[i] for i in axes_tuple)
        _stamp_type_and_shape(out_val, out_shape)
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.transpose not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a, axes=None):
                arr = jnp.asarray(a)
                axes_tuple = _normalize_axes(axes, arr.ndim)
                return cls._PRIM.bind(arr, axes=axes_tuple)

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


@JnpTransposePlugin._PRIM.def_impl
def _transpose_impl(a, axes=None):
    orig = get_orig_impl(JnpTransposePlugin._PRIM, JnpTransposePlugin._FUNC_NAME)
    return orig(a, axes=axes)


JnpTransposePlugin._PRIM.def_abstract_eval(JnpTransposePlugin.abstract_eval)
