from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.converter2.ir_builder import _dtype_to_ir
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_WHERE_PRIM = make_jnp_primitive("jax.numpy.where")


def _maybe_cast_tensor(
    ctx: "IRContext",  # type: ignore[name-defined]
    value: ir.Value,
    source_var,
    target_dtype: np.dtype,
    *,
    tag: str,
) -> ir.Value:
    current_dtype = np.dtype(getattr(getattr(source_var, "aval", None), "dtype", target_dtype))
    if current_dtype == target_dtype:
        return value

    dtype_enum = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)
    cast_val = ir.Value(
        name=ctx.fresh_name(f"where_{tag}_cast"),
        type=ir.TensorType(dtype_enum),
        shape=value.shape,
    )
    ctx.add_node(
        ir.Node(
            op_type="Cast",
            domain="",
            inputs=[value],
            outputs=[cast_val],
            name=ctx.fresh_name("Cast"),
            attributes=[IRAttr("to", IRAttrType.INT, int(dtype_enum.value))],
        )
    )
    _stamp_type_and_shape(cast_val, tuple(getattr(getattr(source_var, "aval", None), "shape", ())))
    _ensure_value_info(ctx, cast_val)
    return cast_val


@register_primitive(
    jaxpr_primitive=_WHERE_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.where.html",
    onnx=[{"component": "Where", "doc": "https://onnx.ai/onnx/operators/onnx__Where.html"}],
    since="v0.8.0",
    context="primitives2.jnp",
    component="where",
    testcases=[
        {
            "testcase": "jnp_where_basic",
            "callable": lambda c, x, y: jnp.where(c, x, y),
            "input_shapes": [(3,), (3,), (3,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "jnp_where_broadcast",
            "callable": lambda c, x, y: jnp.where(c[:, None], x, y),
            "input_shapes": [(4,), (4, 5), (4, 5)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "jnp_where_scalar_else",
            "callable": lambda c, x: jnp.where(c, x, -1e9),
            "input_shapes": [(2, 2), (2, 2)],
            "use_onnx_ir": True,
        },
    ],
)
class JnpWherePlugin(PrimitiveLeafPlugin):
    """IR-only lowering for ``jax.numpy.where`` (three-argument form)."""

    _PRIM: ClassVar = _WHERE_PRIM
    _FUNC_NAME: ClassVar[str] = "where"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(cond, x, y, **_):
        if not all(isinstance(av, jax.core.ShapedArray) for av in (cond, x, y)):
            raise TypeError("jnp.where expects ShapedArray inputs")

        promoted_dtype = np.promote_types(x.dtype, y.dtype)
        out_shape = jnp.broadcast_shapes(cond.shape, x.shape, y.shape)
        return jax.core.ShapedArray(out_shape, promoted_dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        cond_var, x_var, y_var = eqn.invars
        out_var = eqn.outvars[0]

        cond_val = ctx.get_value_for_var(cond_var, name_hint=ctx.fresh_name("where_cond"))
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("where_x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("where_y"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("where_out"))

        cond_dtype = np.dtype(getattr(cond_var.aval, "dtype", np.bool_))
        if cond_dtype != np.bool_:
            cond_cast = ir.Value(
                name=ctx.fresh_name("where_cond_cast"),
                type=ir.TensorType(ir.DataType.BOOL),
                shape=cond_val.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Cast",
                    domain="",
                    inputs=[cond_val],
                    outputs=[cond_cast],
                    name=ctx.fresh_name("Cast"),
                    attributes=[
                        IRAttr("to", IRAttrType.INT, int(ir.DataType.BOOL.value))
                    ],
                )
            )
            _stamp_type_and_shape(
                cond_cast, tuple(getattr(getattr(cond_var, "aval", None), "shape", ()))
            )
            _ensure_value_info(ctx, cond_cast)
            cond_input = cond_cast
        else:
            cond_input = cond_val

        target_dtype = np.promote_types(
            np.dtype(getattr(x_var.aval, "dtype", np.float32)),
            np.dtype(getattr(y_var.aval, "dtype", np.float32)),
        )

        x_input = _maybe_cast_tensor(ctx, x_val, x_var, target_dtype, tag="x")
        y_input = _maybe_cast_tensor(ctx, y_val, y_var, target_dtype, tag="y")

        ctx.add_node(
            ir.Node(
                op_type="Where",
                domain="",
                inputs=[cond_input, x_input, y_input],
                outputs=[out_val],
                name=ctx.fresh_name("Where"),
            )
        )

        out_val.type = ir.TensorType(
            _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)
        )
        _stamp_type_and_shape(
            out_val, tuple(getattr(getattr(out_var, "aval", None), "shape", ()))
        )
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.where not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(condition, x=None, y=None):
                if x is None or y is None:
                    raise NotImplementedError(
                        "jnp.where with fewer than three arguments is not supported"
                    )
                return cls._PRIM.bind(condition, x, y)

            return _patched

        return [
            AssignSpec("jax.numpy", f"{cls._FUNC_NAME}_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@JnpWherePlugin._PRIM.def_impl
def _where_impl(condition, x=None, y=None):
    if x is None or y is None:
        raise NotImplementedError(
            "jnp.where with fewer than three arguments is not supported"
        )
    orig = get_orig_impl(JnpWherePlugin._PRIM, JnpWherePlugin._FUNC_NAME)
    return orig(condition, x, y)


JnpWherePlugin._PRIM.def_abstract_eval(JnpWherePlugin.abstract_eval)
