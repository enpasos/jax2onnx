# jax2onnx/plugins/jax/numpy/sum.py

from __future__ import annotations

from collections.abc import Sequence as _Seq
from types import SimpleNamespace
from typing import Any, Callable, ClassVar, Final

import jax
from jax import core
import jax.numpy as jnp
from jax.interpreters import ad
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.lax._reduce_utils import lower_reduction
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.jax.numpy._reduction_utils import (
    abstract_eval_via_orig_reduction,
    axis_arg_from_params,
    normalize_axis_to_axes,
    register_reduction_batch_rule,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SUM_PRIM: Final = make_jnp_primitive("jax.numpy.sum")


@register_primitive(
    jaxpr_primitive=_SUM_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sum.html",
    onnx=[
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="sum",
    testcases=[
        {
            "testcase": "jnp_sum_basic",
            "callable": lambda x: jnp.sum(x),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceSum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_sum_axis",
            "callable": lambda x: jnp.sum(x, axis=1),
            "input_shapes": [(3, 4, 5)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ReduceSum:3x5",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_sum_keepdims",
            "callable": lambda x: jnp.sum(x, axis=0, keepdims=True),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceSum:1x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_sum_int8_promote",
            "callable": lambda x: jnp.sum(x),
            "input_values": [np.array([1, 2, 3], dtype=np.int8)],
            "expected_output_dtypes": [np.int32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Cast:3 -> ReduceSum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "sum_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.sum)(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "sum_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.sum(y, axis=1)))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpSumPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SUM_PRIM
    _FUNC_NAME: ClassVar[str] = "sum"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        axes: tuple[int, ...] | None = None,
        axes_is_tuple: bool = False,
        dtype: np.dtype[Any] | type | None = None,
        keepdims: bool = False,
        promote_integers: bool = True,
    ) -> core.ShapedArray:
        return abstract_eval_via_orig_reduction(
            JnpSumPlugin._PRIM,
            JnpSumPlugin._FUNC_NAME,
            x,
            axes=axes,
            axes_is_tuple=axes_is_tuple,
            dtype=dtype,
            keepdims=keepdims,
            promote_integers=promote_integers,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        params = dict(getattr(eqn, "params", {}))
        in_dtype = np.dtype(getattr(getattr(eqn.invars[0], "aval", None), "dtype"))
        out_dtype = np.dtype(getattr(getattr(eqn.outvars[0], "aval", None), "dtype"))
        if params.get("dtype") is None and in_dtype != out_dtype:
            params["dtype"] = out_dtype
        params.pop("promote_integers", None)

        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        axes = params.get("axes")
        keepdims = bool(params.get("keepdims", False))
        dtype = params.get("dtype")

        if dtype is None:
            operand_val = ctx.get_value_for_var(
                operand_var, name_hint=ctx.fresh_name("jnp_sum_in")
            )
            out_spec = ctx.get_value_for_var(
                out_var, name_hint=ctx.fresh_name("jnp_sum_out")
            )

            producer_getter = getattr(operand_val, "producer", lambda: None)
            producer = producer_getter() if callable(producer_getter) else None
            producer_op = getattr(producer, "op_type", "")
            producer_inputs = tuple(getattr(producer, "inputs", ()))

            target_base = None
            op_name = None
            if producer_op == "Abs":
                if producer_inputs:
                    target_base = producer_inputs[0]
                    op_name = "ReduceL1"
            elif producer_op == "Mul" and len(producer_inputs) >= 2:
                lhs, rhs = producer_inputs[:2]
                same_input = lhs is rhs or (
                    getattr(lhs, "name", None) is not None
                    and getattr(lhs, "name", None) == getattr(rhs, "name", None)
                )
                if same_input:
                    target_base = lhs
                    op_name = "ReduceSumSquare"
            elif producer_op == "Pow" and len(producer_inputs) >= 2:
                base, exponent = producer_inputs[:2]
                exponent_scalar = _const_scalar(exponent)
                if exponent_scalar is not None and np.allclose(exponent_scalar, 2):
                    target_base = base
                    op_name = "ReduceSumSquare"

            if op_name is not None and target_base is not None:
                rank = len(tuple(getattr(operand_var.aval, "shape", ())))
                axes_norm = None
                if axes is not None:
                    axes_norm = []
                    for ax in axes:
                        ax_i = int(ax)
                        if ax_i < 0:
                            ax_i += rank
                        axes_norm.append(ax_i)

                inputs = [target_base]
                if axes_norm is not None:
                    axes_const = _const_i64(
                        ctx, list(axes_norm), f"{op_name.lower()}_axes"
                    )
                    inputs.append(axes_const)

                desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
                    op_name
                )
                out_producer = getattr(out_spec, "producer", lambda: None)
                if callable(out_producer) and out_producer() is not None:
                    desired_name = ctx.fresh_name(op_name)

                if op_name == "ReduceL1":
                    result = ctx.builder.ReduceL1(
                        *inputs,
                        keepdims=1 if keepdims else 0,
                        _outputs=[desired_name],
                    )
                else:
                    result = ctx.builder.ReduceSumSquare(
                        *inputs,
                        keepdims=1 if keepdims else 0,
                        _outputs=[desired_name],
                    )

                if getattr(out_spec, "type", None) is not None:
                    result.type = out_spec.type
                if getattr(out_spec, "shape", None) is not None:
                    result.shape = out_spec.shape
                ctx.bind_value_for_var(out_var, result)
                return

        proxy_eqn = SimpleNamespace(
            invars=eqn.invars, outvars=eqn.outvars, params=params
        )
        lower_reduction(ctx, proxy_eqn, op_type="ReduceSum", allow_dtype_param=True)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.sum not found for monkey patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                axis: int | _Seq[int] | None = None,
                dtype: np.dtype[Any] | type | None = None,
                out: None = None,
                keepdims: bool = False,
                initial: ArrayLike | None = None,
                where: ArrayLike | None = None,
                promote_integers: bool = True,
            ) -> jax.Array:
                if out is not None:
                    raise NotImplementedError(
                        "jnp.sum with 'out' is not supported for ONNX export"
                    )
                if initial is not None:
                    raise NotImplementedError(
                        "jnp.sum with 'initial' is not supported for ONNX export"
                    )
                if where is not None:
                    raise NotImplementedError(
                        "jnp.sum with 'where' is not supported for ONNX export"
                    )
                axes, axes_is_tuple = normalize_axis_to_axes(axis)
                return cls._PRIM.bind(
                    jnp.asarray(a),
                    axes=axes,
                    axes_is_tuple=axes_is_tuple,
                    dtype=dtype,
                    keepdims=keepdims,
                    promote_integers=bool(promote_integers),
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


@JnpSumPlugin._PRIM.def_impl
def _sum_impl(
    a: ArrayLike,
    *,
    axes: tuple[int, ...] | None = None,
    axes_is_tuple: bool = False,
    dtype: np.dtype[Any] | type | None = None,
    keepdims: bool = False,
    promote_integers: bool = True,
) -> jax.Array:
    orig = get_orig_impl(JnpSumPlugin._PRIM, JnpSumPlugin._FUNC_NAME)
    axis = axis_arg_from_params(axes, axes_is_tuple)
    return orig(
        jnp.asarray(a),
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        promote_integers=promote_integers,
    )


register_jvp_via_jax_jvp(JnpSumPlugin._PRIM, _sum_impl)
register_reduction_batch_rule(JnpSumPlugin._PRIM, jax.lax.reduce_sum_p)


def _const_scalar(value: Any) -> float | int | None:
    const = getattr(value, "const_value", None)
    if const is None:
        return None
    try:
        arr = np.asarray(const)
    except Exception:
        try:
            arr = np.asarray(const.numpy())
        except Exception:
            return None
    if arr.shape != ():
        return None
    scalar = arr.item()
    if isinstance(scalar, (bool, np.bool_)):
        return int(scalar)
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    if isinstance(scalar, (float, np.floating)):
        return float(scalar)
    return None


def _normalize_axes_for_rank(
    axes: tuple[int, ...] | None, rank: int
) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(rank))
    return tuple((int(ax) + rank) % rank for ax in axes)


def _sum_transpose_rule(
    ct: jax.Array | ad.Zero,
    x: Any,
    *,
    axes: tuple[int, ...] | None = None,
    axes_is_tuple: bool = False,
    dtype: np.dtype[Any] | type | None = None,
    keepdims: bool = False,
    promote_integers: bool = True,
) -> tuple[jax.Array | ad.Zero | None,]:
    del axes_is_tuple, dtype, promote_integers
    if not isinstance(x, ad.UndefinedPrimal):
        return (None,)
    if isinstance(ct, ad.Zero):
        return (ad.Zero(x.aval),)

    rank = len(x.aval.shape)
    axes_norm = _normalize_axes_for_rank(axes, rank)
    ct_val = ct
    if not keepdims:
        ct_val = jax.lax.expand_dims(ct_val, axes_norm)
    tangent = jax.lax.broadcast_in_dim(
        ct_val,
        shape=x.aval.shape,
        broadcast_dimensions=tuple(range(rank)),
    )
    input_dtype = getattr(x.aval, "dtype", None)
    if input_dtype is not None and getattr(tangent, "dtype", None) != input_dtype:
        tangent = jax.lax.convert_element_type(tangent, input_dtype)
    return (tangent,)


ad.primitive_transposes[JnpSumPlugin._PRIM] = _sum_transpose_rule
