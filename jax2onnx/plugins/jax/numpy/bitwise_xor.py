# jax2onnx/plugins/jax/numpy/bitwise_xor.py

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
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_BITWISE_XOR_PRIM: Final = make_jnp_primitive("jax.numpy.bitwise_xor")


def _abstract_eval_via_orig_binary(
    prim: Any,
    func_name: str,
    x: core.AbstractValue,
    y: core.AbstractValue,
) -> core.ShapedArray:
    x_shape = tuple(getattr(x, "shape", ()))
    y_shape = tuple(getattr(y, "shape", ()))
    x_dtype: np.dtype[Any] = np.dtype(getattr(x, "dtype", np.int32))
    y_dtype: np.dtype[Any] = np.dtype(getattr(y, "dtype", np.int32))
    x_spec = jax.ShapeDtypeStruct(x_shape, x_dtype)
    y_spec = jax.ShapeDtypeStruct(y_shape, y_dtype)
    orig = get_orig_impl(prim, func_name)
    out = jax.eval_shape(lambda a, b: orig(a, b), x_spec, y_spec)
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
    jaxpr_primitive=_BITWISE_XOR_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bitwise_xor.html",
    onnx=[
        {
            "component": "Xor",
            "doc": "https://onnx.ai/onnx/operators/onnx__Xor.html",
        },
        {
            "component": "BitwiseXor",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitwiseXor.html",
        },
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="bitwise_xor",
    testcases=[
        {
            "testcase": "jnp_bitwise_xor_int",
            "callable": lambda x, y: jnp.bitwise_xor(x, y),
            "input_values": [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([3, 1, 2], dtype=np.int32),
            ],
            "post_check_onnx_graph": EG(
                ["BitwiseXor:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_bitwise_xor_bool",
            "callable": lambda x, y: jnp.bitwise_xor(x, y),
            "input_values": [
                np.array([True, True, False], dtype=np.bool_),
                np.array([True, False, True], dtype=np.bool_),
            ],
            "expected_output_dtypes": [np.bool_],
            "post_check_onnx_graph": EG(
                ["Xor:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_bitwise_xor_mixed_dtype",
            "callable": lambda x, y: jnp.bitwise_xor(x, y),
            "input_values": [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([True, False, True], dtype=np.bool_),
            ],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> BitwiseXor:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_bitwise_xor_broadcast",
            "callable": lambda x, y: jnp.bitwise_xor(x, y),
            "input_shapes": [(2, 3), (1, 3)],
            "input_dtypes": [np.int32, np.int32],
            "post_check_onnx_graph": EG(
                ["BitwiseXor:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "bitwise_xor_vmap_batching",
            "callable": lambda x, y: jax.vmap(jnp.bitwise_xor)(x, y),
            "input_shapes": [(3, 4), (3, 4)],
            "input_dtypes": [np.int32, np.int32],
        },
    ],
)
class JnpBitwiseXorPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _BITWISE_XOR_PRIM
    _FUNC_NAME: ClassVar[str] = "bitwise_xor"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        y: core.AbstractValue,
    ) -> core.ShapedArray:
        return _abstract_eval_via_orig_binary(
            JnpBitwiseXorPlugin._PRIM,
            JnpBitwiseXorPlugin._FUNC_NAME,
            x,
            y,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        x_var, y_var = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(
            x_var, name_hint=ctx.fresh_name("jnp_bitwise_xor_x")
        )
        prefer_dt: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.int32)
        )
        y_val = ctx.get_value_for_var(
            y_var,
            name_hint=ctx.fresh_name("jnp_bitwise_xor_y"),
            prefer_np_dtype=prefer_dt,
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_bitwise_xor_out")
        )

        out_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(out_var, "aval", None), "dtype", np.int32)
        )
        x_ready = _cast_to_dtype(
            ctx,
            x_val,
            np_dtype=out_dtype,
            name_hint="jnp_bitwise_xor_x_cast",
        )
        y_ready = _cast_to_dtype(
            ctx,
            y_val,
            np_dtype=out_dtype,
            name_hint="jnp_bitwise_xor_y_cast",
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "jnp_bitwise_xor_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("jnp_bitwise_xor_out")

        if np.issubdtype(out_dtype, np.bool_):
            result = ctx.builder.Xor(x_ready, y_ready, _outputs=[desired_name])
        else:
            result = ctx.builder.BitwiseXor(x_ready, y_ready, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return cast(
            list[AssignSpec | MonkeyPatchSpec],
            jnp_binding_specs(cls._PRIM, cls._FUNC_NAME),
        )


@JnpBitwiseXorPlugin._PRIM.def_impl
def _bitwise_xor_impl(*args: object, **kwargs: object) -> object:
    orig = get_orig_impl(JnpBitwiseXorPlugin._PRIM, JnpBitwiseXorPlugin._FUNC_NAME)
    return orig(*args, **kwargs)


def _bitwise_xor_batch_rule(
    args: tuple[Any, ...], dims: tuple[Any, ...], **params: Any
) -> tuple[Any, Any]:
    return cast(
        tuple[Any, Any],
        broadcast_batcher_compat(JnpBitwiseXorPlugin._PRIM, args, dims, **params),
    )


batching.primitive_batchers[JnpBitwiseXorPlugin._PRIM] = _bitwise_xor_batch_rule
