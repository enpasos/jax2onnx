from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax import core
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.jax.lax._index_utils import _const_i64
from jax2onnx.plugins2.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_UNSTACK_PRIM = make_jnp_primitive("jax.numpy.unstack")
_UNSTACK_PRIM.multiple_results = True


def _normalize_axis(axis: int, rank: int) -> int:
    ax = int(axis)
    if ax < 0:
        ax += rank
    if ax < 0 or ax >= rank:
        raise ValueError(f"axis {axis} out of bounds for rank {rank}")
    return ax


@register_primitive(
    jaxpr_primitive=_UNSTACK_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unstack.html",
    onnx=[
        {
            "component": "Split",
            "doc": "https://onnx.ai/onnx/operators/onnx__Split.html",
        },
        {
            "component": "Squeeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
        },
    ],
    since="v0.9.0",
    context="primitives2.jnp",
    component="unstack",
    testcases=[
        {
            "testcase": "unstack_axis_0",
            "callable": lambda x: jnp.unstack(x, axis=0),
            "input_values": [np.array([[1, 2], [3, 4]], dtype=np.float32)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "unstack_axis_0_f64",
            "callable": lambda x: jnp.unstack(x, axis=0),
            "input_values": [np.array([[1, 2], [3, 4]], dtype=np.float64)],
            "run_only_f64_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "unstack_axis_1",
            "callable": lambda x: jnp.unstack(x, axis=1),
            "input_values": [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "unstack_axis_1_f64",
            "callable": lambda x: jnp.unstack(x, axis=1),
            "input_values": [
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
            ],
            "run_only_f64_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "unstack_negative_axis",
            "callable": lambda x: jnp.unstack(x, axis=-1),
            "input_values": [np.array([[[1, 2], [3, 4]]], dtype=np.float32)],
            "use_onnx_ir": True,
        },
    ],
)
class JnpUnstackPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _UNSTACK_PRIM
    _FUNC_NAME: ClassVar[str] = "unstack"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *, axis=0):
        rank = len(x.shape)
        axis_norm = _normalize_axis(axis, rank)
        size = x.shape[axis_norm]
        if not isinstance(size, (int, np.integer)):
            raise core.InconclusiveDimensionOperation(
                "jnp.unstack requires static axis length"
            )
        out_shape = x.shape[:axis_norm] + x.shape[axis_norm + 1 :]
        return tuple(core.ShapedArray(out_shape, x.dtype) for _ in range(int(size)))

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        params = getattr(eqn, "params", {})
        axis_param = params.get("axis", 0)

        (arr_var,) = eqn.invars
        out_vars = list(eqn.outvars)

        arr_shape = tuple(getattr(arr_var.aval, "shape", ()))
        axis = _normalize_axis(axis_param, len(arr_shape))
        size = arr_shape[axis]
        if not isinstance(size, (int, np.integer)):
            raise TypeError("jnp.unstack requires static axis length for lowering")

        arr_val = ctx.get_value_for_var(arr_var, name_hint=ctx.fresh_name("unstack_in"))

        split_sizes = _const_i64(
            ctx, np.asarray([1] * int(size), dtype=np.int64), "unstack_split_sizes"
        )

        split_shape = list(arr_shape)
        split_shape[axis] = 1
        split_values = [
            ir.Value(
                name=ctx.fresh_name("unstack_split_out"),
                type=arr_val.type,
                shape=ir.Shape(tuple(split_shape)),
            )
            for _ in range(int(size))
        ]

        ctx.add_node(
            ir.Node(
                op_type="Split",
                domain="",
                inputs=[arr_val, split_sizes],
                outputs=split_values,
                name=ctx.fresh_name("Split"),
                attributes=[IRAttr("axis", IRAttrType.INT, int(axis))],
            )
        )
        for split_val in split_values:
            _stamp_type_and_shape(split_val, tuple(split_shape))
            _ensure_value_info(ctx, split_val)

        axes_val = _const_i64(
            ctx, np.asarray([axis], dtype=np.int64), "unstack_squeeze_axes"
        )

        for split_val, out_var in zip(split_values, out_vars):
            out_val = ctx.get_value_for_var(
                out_var, name_hint=ctx.fresh_name("unstack_out")
            )
            ctx.add_node(
                ir.Node(
                    op_type="Squeeze",
                    domain="",
                    inputs=[split_val, axes_val],
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
                raise RuntimeError("Original jnp.unstack not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(x, axis=0):
                return cls._PRIM.bind(x, axis=axis)

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


@JnpUnstackPlugin._PRIM.def_impl
def _unstack_impl(x, axis=0):
    orig = get_orig_impl(JnpUnstackPlugin._PRIM, JnpUnstackPlugin._FUNC_NAME)
    return orig(x, axis=axis)


JnpUnstackPlugin._PRIM.def_abstract_eval(JnpUnstackPlugin.abstract_eval)
