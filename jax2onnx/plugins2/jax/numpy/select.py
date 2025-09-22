from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax import core
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.converter2.ir_builder import _dtype_to_ir
from jax2onnx.plugins2.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_SELECT_PRIM = make_jnp_primitive("jax.numpy.select")


def _broadcast_shape(*shapes):
    return jnp.broadcast_shapes(*shapes)


def _promote_dtype(*dtypes):
    return jnp.result_type(*dtypes)


@register_primitive(
    jaxpr_primitive=_SELECT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.select.html",
    onnx=[
        {"component": "Where", "doc": "https://onnx.ai/onnx/operators/onnx__Where.html"}
    ],
    since="v0.9.0",
    context="primitives2.jnp",
    component="select",
    testcases=[
        {
            "testcase": "select_basic",
            "callable": lambda: jnp.select(
                [jnp.array([True, False]), jnp.array([False, True])],
                [jnp.array([1, 2]), jnp.array([3, 4])],
                default=jnp.array([0, 0]),
            ),
            "use_onnx_ir": True,
        }
    ],
)
class JnpSelectPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SELECT_PRIM
    _FUNC_NAME: ClassVar[str] = "select"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(*operands, num_conds: int, num_choices: int):
        conds = operands[:num_conds]
        choices = operands[num_conds : num_conds + num_choices]
        default = operands[-1]
        shape = _broadcast_shape(
            *[c.shape for c in conds],
            *[c.shape for c in choices],
            default.shape,
        )
        dtype = _promote_dtype(*(c.dtype for c in choices), default.dtype)
        return core.ShapedArray(shape, dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        params = getattr(eqn, "params", {})
        num_conds = int(params["num_conds"])
        num_choices = int(params["num_choices"])

        cond_vars = eqn.invars[:num_conds]
        choice_vars = eqn.invars[num_conds : num_conds + num_choices]
        default_var = eqn.invars[-1]

        out_var = eqn.outvars[0]

        result_shape = tuple(getattr(out_var.aval, "shape", ()))
        result_dtype = getattr(out_var.aval, "dtype", np.float32)

        else_val = ctx.get_value_for_var(
            default_var, name_hint=ctx.fresh_name("select_default")
        )
        else_val = self._ensure_dtype(ctx, else_val, default_var, result_dtype)

        for cond_var, choice_var in reversed(list(zip(cond_vars, choice_vars))):
            cond_val = ctx.get_value_for_var(
                cond_var, name_hint=ctx.fresh_name("select_cond")
            )
            cond_val = self._ensure_bool(ctx, cond_val, cond_var)

            choice_val = ctx.get_value_for_var(
                choice_var, name_hint=ctx.fresh_name("select_then")
            )
            choice_val = self._ensure_dtype(ctx, choice_val, choice_var, result_dtype)

            out_val = ir.Value(
                name=ctx.fresh_name("select_out"),
                type=ir.TensorType(choice_val.type.dtype),
                shape=choice_val.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Where",
                    domain="",
                    inputs=[cond_val, choice_val, else_val],
                    outputs=[out_val],
                    name=ctx.fresh_name("Where"),
                )
            )
            _stamp_type_and_shape(out_val, result_shape)
            _ensure_value_info(ctx, out_val)
            else_val = out_val

        final_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("select_final")
        )
        ctx.add_node(
            ir.Node(
                op_type="Identity",
                domain="",
                inputs=[else_val],
                outputs=[final_val],
                name=ctx.fresh_name("Identity"),
            )
        )
        _stamp_type_and_shape(final_val, result_shape)
        _ensure_value_info(ctx, final_val)

    @staticmethod
    def _ensure_bool(ctx: "IRContext", val: ir.Value, var) -> ir.Value:
        dtype = getattr(getattr(var, "aval", None), "dtype", np.bool_)
        if dtype == np.bool_:
            return val
        cast = ir.Value(
            name=ctx.fresh_name("select_cond_cast"),
            type=ir.TensorType(ir.DataType.BOOL),
            shape=val.shape,
        )
        ctx.add_node(
            ir.Node(
                op_type="Cast",
                domain="",
                inputs=[val],
                outputs=[cast],
                name=ctx.fresh_name("Cast"),
                attributes=[IRAttr("to", IRAttrType.INT, int(ir.DataType.BOOL.value))],
            )
        )
        _stamp_type_and_shape(cast, tuple(getattr(var.aval, "shape", ())))
        _ensure_value_info(ctx, cast)
        return cast

    @staticmethod
    def _ensure_dtype(ctx: "IRContext", val: ir.Value, var, target_dtype) -> ir.Value:
        dtype = getattr(getattr(var, "aval", None), "dtype", target_dtype)
        if dtype == target_dtype:
            return val
        target_ir_dtype = _dtype_to_ir(
            np.dtype(target_dtype), ctx.builder.enable_double_precision
        )
        cast = ir.Value(
            name=ctx.fresh_name("select_cast"),
            type=ir.TensorType(target_ir_dtype),
            shape=val.shape,
        )
        ctx.add_node(
            ir.Node(
                op_type="Cast",
                domain="",
                inputs=[val],
                outputs=[cast],
                name=ctx.fresh_name("Cast"),
                attributes=[IRAttr("to", IRAttrType.INT, int(target_ir_dtype.value))],
            )
        )
        _stamp_type_and_shape(cast, tuple(getattr(var.aval, "shape", ())))
        _ensure_value_info(ctx, cast)
        return cast

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.select not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(condlist, choicelist, *, default=None):
                return cls._PRIM.bind(
                    *condlist,
                    *choicelist,
                    default,
                    num_conds=len(condlist),
                    num_choices=len(choicelist),
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


@JnpSelectPlugin._PRIM.def_impl
def _select_impl(condlist, choicelist, *, default=None):
    orig = get_orig_impl(JnpSelectPlugin._PRIM, JnpSelectPlugin._FUNC_NAME)
    return orig(condlist, choicelist, default=default)


JnpSelectPlugin._PRIM.def_abstract_eval(JnpSelectPlugin.abstract_eval)
