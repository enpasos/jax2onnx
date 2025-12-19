# jax2onnx/plugins/jax/numpy/select.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, ClassVar, Final

import jax
import jax.extend.core as jax_core_ext
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
import onnx_ir as ir
from jax import core
from jax.interpreters import batching

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SELECT_PRIM: Final = make_jnp_primitive("jax.numpy.select")


def _broadcast_shape(*shapes: Sequence[int | object]) -> tuple[int | object, ...]:
    return jnp.broadcast_shapes(*shapes)


def _promote_dtype(*dtypes: np.dtype[Any] | type) -> np.dtype[Any]:
    return jnp.result_type(*dtypes)


@register_primitive(
    jaxpr_primitive=_SELECT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.select.html",
    onnx=[
        {"component": "Where", "doc": "https://onnx.ai/onnx/operators/onnx__Where.html"}
    ],
    since="v0.7.1",
    context="primitives.jnp",
    component="select",
    testcases=[
        {
            "testcase": "select_simple",
            "callable": lambda: jnp.select(
                [jnp.array([True, False]), jnp.array([False, True])],
                [jnp.array([1, 2]), jnp.array([3, 4])],
                default=jnp.array([0, 0]),
            ),
            "post_check_onnx_graph": EG(
                ["Where:2 -> Where:2 -> Identity:2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "select_broadcast",
            "callable": lambda: jnp.select(
                [jnp.array([True, False]), jnp.array([False, True])],
                [jnp.array([1, 2]), jnp.array([3, 4])],
                default=0,
            ),
            "post_check_onnx_graph": EG(
                ["Where:2 -> Where:2 -> Identity:2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "select_gpt2_attention_mask",
            "callable": lambda scores, mask: jnp.select(
                [mask], [scores], default=jnp.array(-1e9, dtype=jnp.float32)
            ),
            "input_shapes": [("B", 12, "T", "T"), ("B", 1, "T", "T")],
            "input_dtypes": [np.float32, np.bool_],
            "post_check_onnx_graph": EG(
                ["Where:Bx12xTxT -> Identity:Bx12xTxT"],
                symbols={"B": None, "T": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "select_basic",
            "callable": lambda: jnp.select(
                [jnp.array([True, False]), jnp.array([False, True])],
                [jnp.array([1, 2]), jnp.array([3, 4])],
                default=jnp.array([0, 0]),
            ),
            "post_check_onnx_graph": EG(
                ["Where:2 -> Where:2 -> Identity:2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "select_vmap_batching",
            "callable": lambda x: jax.vmap(
                lambda y: jnp.select([y > 0], [y], default=-y)
            )(x),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpSelectPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SELECT_PRIM
    _FUNC_NAME: ClassVar[str] = "select"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        *operands: core.AbstractValue, num_conds: int, num_choices: int
    ) -> core.ShapedArray:
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

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        params = getattr(eqn, "params", {})
        num_conds = int(params["num_conds"])
        num_choices = int(params["num_choices"])

        cond_vars = eqn.invars[:num_conds]
        choice_vars = eqn.invars[num_conds : num_conds + num_choices]
        default_var = eqn.invars[-1]

        out_var = eqn.outvars[0]

        result_shape = tuple(getattr(out_var.aval, "shape", ()))
        result_dtype = getattr(out_var.aval, "dtype", np.float32)

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for select lowering")

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

            out_val = builder.Where(
                cond_val,
                choice_val,
                else_val,
                _outputs=[ctx.fresh_name("select_out")],
            )
            out_dtype = getattr(
                getattr(choice_val, "type", None), "dtype", None
            ) or getattr(getattr(else_val, "type", None), "dtype", None)
            if out_dtype is not None:
                out_val.type = ir.TensorType(out_dtype)
            _stamp_type_and_shape(out_val, result_shape)
            _ensure_value_metadata(ctx, out_val)
            else_val = out_val

        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("select_final")
        )
        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("select_final")
        final_val = builder.Identity(
            else_val,
            _outputs=[out_name],
        )
        final_dtype = getattr(
            getattr(else_val, "type", None), "dtype", None
        ) or _dtype_to_ir(np.dtype(result_dtype), ctx.builder.enable_double_precision)
        final_val.type = ir.TensorType(final_dtype)
        _stamp_type_and_shape(final_val, result_shape)
        _ensure_value_metadata(ctx, final_val)
        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(out_var, final_val)

    @staticmethod
    def _ensure_bool(
        ctx: LoweringContextProtocol, val: ir.Value, var: jax_core_ext.Var
    ) -> ir.Value:
        dtype = getattr(getattr(var, "aval", None), "dtype", np.bool_)
        if dtype == np.bool_:
            return val
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for select lowering")
        cast = builder.Cast(
            val,
            _outputs=[ctx.fresh_name("select_cond_cast")],
            to=int(ir.DataType.BOOL.value),
        )
        cast.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(cast, tuple(getattr(var.aval, "shape", ())))
        _ensure_value_metadata(ctx, cast)
        return cast

    @staticmethod
    def _ensure_dtype(
        ctx: LoweringContextProtocol,
        val: ir.Value,
        var: jax_core_ext.Var,
        target_dtype: np.dtype[Any],
    ) -> ir.Value:
        dtype = getattr(getattr(var, "aval", None), "dtype", target_dtype)
        if dtype == target_dtype:
            return val
        target_ir_dtype = _dtype_to_ir(
            np.dtype(target_dtype), ctx.builder.enable_double_precision
        )
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for select lowering")
        cast = builder.Cast(
            val,
            _outputs=[ctx.fresh_name("select_cast")],
            to=int(target_ir_dtype.value),
        )
        cast.type = ir.TensorType(target_ir_dtype)
        _stamp_type_and_shape(cast, tuple(getattr(var.aval, "shape", ())))
        _ensure_value_metadata(ctx, cast)
        return cast

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.select not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                condlist: Sequence[ArrayLike],
                choicelist: Sequence[ArrayLike],
                *,
                default: ArrayLike | None = None,
            ) -> jax.Array:
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
def _select_impl(
    condlist: Sequence[ArrayLike],
    choicelist: Sequence[ArrayLike],
    *,
    default: ArrayLike | None = None,
) -> jax.Array:
    orig = get_orig_impl(JnpSelectPlugin._PRIM, JnpSelectPlugin._FUNC_NAME)
    return orig(condlist, choicelist, default=default)


JnpSelectPlugin._PRIM.def_abstract_eval(JnpSelectPlugin.abstract_eval)


def _select_batch_rule(args, dims, **params):
    return broadcast_batcher_compat(JnpSelectPlugin._PRIM, args, dims, **params)


batching.primitive_batchers[JnpSelectPlugin._PRIM] = _select_batch_rule
