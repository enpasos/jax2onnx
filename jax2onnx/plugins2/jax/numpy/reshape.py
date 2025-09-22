from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax import core

try:  # pragma: no cover - best effort import for shape polymorphism
    from jax._src.export.shape_poly import _DimExpr as DimExpr
except Exception:  # pragma: no cover
    DimExpr = object  # type: ignore[misc,assignment]

from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.jax.lax._index_utils import _const_i64
from jax2onnx.plugins2.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_RESHAPE_PRIM = make_jnp_primitive("jax.numpy.reshape")


def _iter_newshape(newshape: Sequence[int | object] | int | object) -> Iterable:
    if isinstance(newshape, Sequence):
        return newshape
    return (newshape,)


def _find_axis_for_dim(dim: object, input_shape: Sequence[object]) -> int | None:
    for idx, src in enumerate(input_shape):
        if dim is src:
            return idx
        if isinstance(dim, DimExpr) and isinstance(src, DimExpr):
            if str(dim) == str(src):
                return idx
        if hasattr(dim, "_hashable_content") and hasattr(src, "_hashable_content"):
            if dim._hashable_content() == src._hashable_content():  # type: ignore[attr-defined]
                return idx
        if str(src) == str(dim):
            return idx
    return None


@register_primitive(
    jaxpr_primitive=_RESHAPE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.reshape.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        }
    ],
    since="v0.9.0",
    context="primitives2.jnp",
    component="reshape",
    testcases=[
        {
            "testcase": "reshape_basic",
            "callable": lambda a: jnp.reshape(a, (2, 6)),
            "input_shapes": [(3, 4)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reshape_infer",
            "callable": lambda a: jnp.reshape(a, (-1, 2)),
            "input_shapes": [(3, 4)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reshape_symbolic_flatten",
            "callable": lambda a: jnp.reshape(a, (a.shape[0], -1)),
            "input_shapes": [("B", 8, 4)],
            "use_onnx_ir": True,
        },
    ],
)
class JnpReshapePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _RESHAPE_PRIM
    _FUNC_NAME: ClassVar[str] = "reshape"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *, newshape, order="C"):
        if order not in (None, "C"):
            raise NotImplementedError("Only C-order reshape is supported")
        storage_slot = f"__orig_impl__{JnpReshapePlugin._FUNC_NAME}"
        orig = getattr(JnpReshapePlugin._PRIM, storage_slot, jnp.reshape)
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        result = jax.eval_shape(lambda arr: orig(arr, newshape, order=order), spec)
        return core.ShapedArray(result.shape, result.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        params = getattr(eqn, "params", {})
        newshape_param = params.get("new_sizes", params.get("newshape"))
        order = params.get("order", "C")
        if order not in (None, "C"):
            raise NotImplementedError(
                "jnp.reshape order other than 'C' is not supported"
            )
        if newshape_param is None:
            raise KeyError("reshape parameters missing 'newshape' or 'new_sizes'")

        (arr_var,) = eqn.invars
        (out_var,) = eqn.outvars

        arr_val = ctx.get_value_for_var(arr_var, name_hint=ctx.fresh_name("reshape_in"))
        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("reshape_out")
        )

        input_shape = tuple(getattr(arr_var.aval, "shape", ()))
        target_shape = tuple(getattr(out_var.aval, "shape", ()))

        shape_components: list[ir.Value] = []
        shape_tensor_rank = len(target_shape)

        shape_value: ir.Value | None = None

        def ensure_shape_value() -> ir.Value:
            nonlocal shape_value
            if shape_value is not None:
                return shape_value
            shape_value = ir.Value(
                name=ctx.fresh_name("reshape_input_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((len(input_shape),)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Shape",
                    domain="",
                    inputs=[arr_val],
                    outputs=[shape_value],
                    name=ctx.fresh_name("Shape"),
                )
            )
            _stamp_type_and_shape(shape_value, (len(input_shape),))
            _ensure_value_info(ctx, shape_value)
            return shape_value

        def gather_axis(idx: int) -> ir.Value:
            shape_val = ensure_shape_value()
            axis_const = _const_i64(
                ctx, np.asarray(idx, dtype=np.int64), "reshape_axis"
            )
            gather_val = ir.Value(
                name=ctx.fresh_name("reshape_dim"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(()),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Gather",
                    domain="",
                    inputs=[shape_val, axis_const],
                    outputs=[gather_val],
                    name=ctx.fresh_name("Gather"),
                    attributes=[IRAttr("axis", IRAttrType.INT, 0)],
                )
            )
            _stamp_type_and_shape(gather_val, ())
            _ensure_value_info(ctx, gather_val)
            axes_const = _const_i64(
                ctx, np.asarray([0], dtype=np.int64), "reshape_unsqueeze_axes"
            )
            unsqueezed = ir.Value(
                name=ctx.fresh_name("reshape_dim_unsq"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((1,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Unsqueeze",
                    domain="",
                    inputs=[gather_val, axes_const],
                    outputs=[unsqueezed],
                    name=ctx.fresh_name("Unsqueeze"),
                )
            )
            _stamp_type_and_shape(unsqueezed, (1,))
            _ensure_value_info(ctx, unsqueezed)
            return unsqueezed

        for idx, dim in enumerate(target_shape):
            if isinstance(dim, (int, np.integer)):
                val = _const_i64(
                    ctx,
                    np.asarray([int(dim)], dtype=np.int64),
                    f"reshape_dim_const_{idx}",
                )
                shape_components.append(val)
                continue

            axis_idx = _find_axis_for_dim(dim, input_shape)
            if axis_idx is None:
                raise TypeError(
                    "reshape with symbolic dimensions requires mapping to input axes"
                )
            shape_components.append(gather_axis(axis_idx))

        if shape_tensor_rank == 0:
            shape_tensor = _const_i64(
                ctx, np.asarray([], dtype=np.int64), "reshape_empty_shape"
            )
        elif len(shape_components) == 1:
            shape_tensor = shape_components[0]
        else:
            shape_tensor = ir.Value(
                name=ctx.fresh_name("reshape_target_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((shape_tensor_rank,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Concat",
                    domain="",
                    inputs=shape_components,
                    outputs=[shape_tensor],
                    name=ctx.fresh_name("Concat"),
                    attributes=[IRAttr("axis", IRAttrType.INT, 0)],
                )
            )
            _stamp_type_and_shape(shape_tensor, (shape_tensor_rank,))
            _ensure_value_info(ctx, shape_tensor)

        ctx.add_node(
            ir.Node(
                op_type="Reshape",
                domain="",
                inputs=[arr_val, shape_tensor],
                outputs=[out_val],
                name=ctx.fresh_name("Reshape"),
                attributes=[IRAttr("allowzero", IRAttrType.INT, 0)],
            )
        )
        _stamp_type_and_shape(out_val, target_shape)
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.reshape not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a, newshape, order="C"):
                if order not in (None, "C"):
                    raise NotImplementedError("Only C-order reshape is supported")
                return cls._PRIM.bind(
                    a, newshape=tuple(_iter_newshape(newshape)), order=order
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


@JnpReshapePlugin._PRIM.def_impl
def _reshape_impl(a, newshape, order="C"):
    orig = get_orig_impl(JnpReshapePlugin._PRIM, JnpReshapePlugin._FUNC_NAME)
    return orig(a, newshape, order=order)


JnpReshapePlugin._PRIM.def_abstract_eval(JnpReshapePlugin.abstract_eval)
