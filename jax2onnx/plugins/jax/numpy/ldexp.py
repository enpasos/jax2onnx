# jax2onnx/plugins/jax/numpy/ldexp.py

from __future__ import annotations

from typing import Any, ClassVar, Final, cast

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import ir_dtype_to_numpy
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_LDEXP_PRIM: Final = make_jnp_primitive("jax.numpy.ldexp")


def _abstract_eval_via_orig_binary(
    prim: Any,
    func_name: str,
    x: core.AbstractValue,
    exp: core.AbstractValue,
) -> core.ShapedArray:
    x_shape = tuple(getattr(x, "shape", ()))
    exp_shape = tuple(getattr(exp, "shape", ()))
    x_dtype: np.dtype[Any] = np.dtype(getattr(x, "dtype", np.float32))
    exp_dtype: np.dtype[Any] = np.dtype(getattr(exp, "dtype", np.int32))
    x_spec = jax.ShapeDtypeStruct(x_shape, x_dtype)
    exp_spec = jax.ShapeDtypeStruct(exp_shape, exp_dtype)
    orig = get_orig_impl(prim, func_name)
    out = jax.eval_shape(lambda a, b: orig(a, b), x_spec, exp_spec)
    out_shape = tuple(getattr(out, "shape", ()))
    out_dtype = np.dtype(getattr(out, "dtype", x_dtype))
    return core.ShapedArray(out_shape, out_dtype)


def _cast_to_dtype(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    np_dtype: np.dtype[Any],
    name_hint: str,
) -> ir.Value:
    dtype_enum = _dtype_to_ir(np_dtype, ctx.builder.enable_double_precision)
    if getattr(getattr(val, "type", None), "dtype", None) == dtype_enum:
        return val
    cast_val = ctx.builder.Cast(
        val,
        to=int(dtype_enum.value),
        _outputs=[ctx.fresh_name(name_hint)],
    )
    cast_val.type = ir.TensorType(dtype_enum)
    cast_val.shape = getattr(val, "shape", None)
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


@register_primitive(
    jaxpr_primitive=_LDEXP_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ldexp.html",
    onnx=[
        {"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
    ],
    since="0.13.0",
    context="primitives.jnp",
    component="ldexp",
    testcases=[
        {
            "testcase": "jnp_ldexp_vector",
            "callable": lambda x, exp: jnp.ldexp(x, exp),
            "input_values": [
                np.array([0.5, -1.25, 3.0], dtype=np.float32),
                np.array([1, 2, -1], dtype=np.int32),
            ],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Pow:3 -> Mul:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_ldexp_broadcast",
            "callable": lambda x, exp: jnp.ldexp(x, exp),
            "input_shapes": [(2, 1), (3,)],
            "input_dtypes": [np.float32, np.int32],
            "expected_output_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Pow:3 -> Mul:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "ldexp_vmap_batching",
            "callable": lambda x, exp: jax.vmap(jnp.ldexp)(x, exp),
            "input_shapes": [(3, 4), (3, 4)],
            "input_dtypes": [np.float32, np.int32],
        },
    ],
)
class JnpLdexpPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _LDEXP_PRIM
    _FUNC_NAME: ClassVar[str] = "ldexp"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        exp: core.AbstractValue,
    ) -> core.ShapedArray:
        return _abstract_eval_via_orig_binary(
            JnpLdexpPlugin._PRIM,
            JnpLdexpPlugin._FUNC_NAME,
            x,
            exp,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        x_var, exp_var = eqn.invars
        (out_var,) = eqn.outvars

        out_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(out_var, "aval", None), "dtype", np.float32)
        )
        target_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("jnp_ldexp_x"))
        exp_val = ctx.get_value_for_var(
            exp_var, name_hint=ctx.fresh_name("jnp_ldexp_exp")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_ldexp_out")
        )

        x_ready = _cast_to_dtype(
            ctx,
            x_val,
            np_dtype=out_dtype,
            name_hint="jnp_ldexp_x_cast",
        )
        exp_ready = _cast_to_dtype(
            ctx,
            exp_val,
            np_dtype=out_dtype,
            name_hint="jnp_ldexp_exp_cast",
        )

        base_dtype = ir_dtype_to_numpy(target_enum, default=out_dtype)
        if base_dtype is None:
            base_dtype = out_dtype
        two_val = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("jnp_ldexp_two"),
            array=np.asarray(2.0, dtype=base_dtype),
        )
        two_val.type = ir.TensorType(target_enum)
        _stamp_type_and_shape(two_val, ())
        _ensure_value_metadata(ctx, two_val)

        scale_val = ctx.builder.Pow(
            two_val,
            exp_ready,
            _outputs=[ctx.fresh_name("jnp_ldexp_scale")],
        )
        scale_val.type = ir.TensorType(target_enum)
        _stamp_type_and_shape(scale_val, tuple(getattr(exp_var.aval, "shape", ())))
        _ensure_value_metadata(ctx, scale_val)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "jnp_ldexp_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("jnp_ldexp_out")

        result = ctx.builder.Mul(x_ready, scale_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = ir.TensorType(target_enum)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpLdexpPlugin._PRIM.def_impl
def _ldexp_impl(*args: object, **kwargs: object) -> object:
    orig = get_orig_impl(JnpLdexpPlugin._PRIM, JnpLdexpPlugin._FUNC_NAME)
    return orig(*args, **kwargs)


def _ldexp_batch_rule(
    args: tuple[Any, ...], dims: tuple[Any, ...], **params: Any
) -> tuple[Any, Any]:
    return cast(
        tuple[Any, Any],
        broadcast_batcher_compat(JnpLdexpPlugin._PRIM, args, dims, **params),
    )


batching.primitive_batchers[JnpLdexpPlugin._PRIM] = _ldexp_batch_rule
