# jax2onnx/plugins/flax/linen/multi_head_dot_product_attention.py

from __future__ import annotations

from typing import Callable, ClassVar

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Primitive
from flax import linen as nn

from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)


def _get_param_group(params: dict, name: str) -> dict | None:
    group = params.get(name)
    if isinstance(group, dict):
        return group
    return None


def _dense_general_apply(x, kernel, bias, *, precision=None):
    y = jax.lax.dot_general(
        x,
        kernel,
        dimension_numbers=(((x.ndim - 1,), (0,)), ((), ())),
        precision=precision,
    )
    if bias is not None:
        y = y + bias
    return y


def _project_output(x, kernel, bias, *, precision=None):
    if kernel.ndim == 3:
        y = jax.lax.dot_general(
            x,
            kernel,
            dimension_numbers=(((x.ndim - 2, x.ndim - 1), (0, 1)), ((), ())),
            precision=precision,
        )
    elif kernel.ndim == 2:
        x_flat = jnp.reshape(x, x.shape[:-2] + (-1,))
        y = jax.lax.dot_general(
            x_flat,
            kernel,
            dimension_numbers=(((x_flat.ndim - 1,), (0,)), ((), ())),
            precision=precision,
        )
    else:
        raise ValueError("MultiHeadAttention output kernel rank must be 2 or 3.")
    if bias is not None:
        y = y + bias
    return y


def _normalize_axes(axes, ndim: int) -> tuple[int, ...] | None:
    if axes is None:
        return None
    try:
        if isinstance(axes, int):
            axes_tuple = (int(axes),)
        else:
            axes_tuple = tuple(int(a) for a in axes)
    except Exception:
        return None
    normalized = []
    for a in axes_tuple:
        if a < 0:
            a += ndim
        normalized.append(a)
    return tuple(normalized)


def _make_mha_patch(orig_fn: Callable):
    def patched(self, inputs_q, inputs_kv=None, *args, **kwargs):
        def _fallback():
            if args:
                return orig_fn(self, inputs_q, inputs_kv, *args, **kwargs)
            return orig_fn(self, inputs_q, inputs_kv=inputs_kv, **kwargs)

        if args:
            return _fallback()

        mask = kwargs.get("mask")
        bias = kwargs.get("bias")
        attention_bias = kwargs.get("attention_bias")
        deterministic = kwargs.get("deterministic", None)
        decode_kw = kwargs.get("decode", None)
        sow_attention_weights = kwargs.get("sow_attention_weights", False)
        dropout_rng = kwargs.get("dropout_rng", None)
        allowed_keys = {
            "mask",
            "bias",
            "attention_bias",
            "deterministic",
            "decode",
            "sow_attention_weights",
            "dropout_rng",
        }
        if kwargs.keys() - allowed_keys:
            return _fallback()
        if bias is not None and attention_bias is not None:
            return _fallback()
        if bias is None:
            bias = attention_bias
        if bias is not None:
            return _fallback()

        kv_inputs = inputs_q if inputs_kv is None else inputs_kv

        if inputs_q.ndim != 3 or kv_inputs.ndim != 3:
            return _fallback()

        if bool(getattr(self, "decode", False)) or decode_kw:
            return _fallback()

        if sow_attention_weights:
            return _fallback()

        attention_axes = _normalize_axes(
            getattr(self, "attention_axes", None), inputs_q.ndim
        )
        if attention_axes is not None and attention_axes != (inputs_q.ndim - 2,):
            return _fallback()

        dropout_rate = float(getattr(self, "dropout_rate", 0.0))
        if dropout_rate != 0.0:
            det = deterministic
            if det is None:
                det = getattr(self, "deterministic", None)
            if det is not True:
                return _fallback()
            if dropout_rng is not None:
                return _fallback()

        attention_fn = getattr(self, "attention_fn", None)
        if (
            attention_fn is not None
            and attention_fn is not jax.nn.dot_product_attention
        ):
            fn_name = getattr(attention_fn, "__name__", "")
            if fn_name != "dot_product_attention":
                return _fallback()

        scope = getattr(self, "scope", None)
        if scope is None or not hasattr(scope, "variables"):
            return _fallback()
        variables = scope.variables()
        params = variables.get("params", {})

        q_params = _get_param_group(params, "query")
        k_params = _get_param_group(params, "key")
        v_params = _get_param_group(params, "value")
        o_params = _get_param_group(params, "out")
        if q_params is None or k_params is None or v_params is None or o_params is None:
            return _fallback()

        q_kernel = q_params.get("kernel")
        k_kernel = k_params.get("kernel")
        v_kernel = v_params.get("kernel")
        o_kernel = o_params.get("kernel")
        if q_kernel is None or k_kernel is None or v_kernel is None or o_kernel is None:
            return _fallback()

        use_bias = bool(getattr(self, "use_bias", True))
        q_bias = q_params.get("bias") if use_bias else None
        k_bias = k_params.get("bias") if use_bias else None
        v_bias = v_params.get("bias") if use_bias else None
        o_bias = o_params.get("bias") if use_bias else None

        if any(kernel.ndim != 3 for kernel in (q_kernel, k_kernel, v_kernel)):
            return _fallback()
        if o_kernel.ndim not in (2, 3):
            return _fallback()

        num_heads_attr = getattr(self, "num_heads", None)
        try:
            num_heads = (
                int(num_heads_attr)
                if num_heads_attr is not None
                else int(q_kernel.shape[1])
            )
        except Exception:
            num_heads = int(q_kernel.shape[1])
        if q_kernel.shape[1] != num_heads:
            return _fallback()
        if k_kernel.shape[1] != num_heads or v_kernel.shape[1] != num_heads:
            return _fallback()

        precision = getattr(self, "precision", None)
        dtype = getattr(self, "dtype", None)
        if not hasattr(self, "promote_dtype"):
            return _fallback()
        (
            inputs_q,
            kv_inputs,
            q_kernel,
            k_kernel,
            v_kernel,
            o_kernel,
            q_bias,
            k_bias,
            v_bias,
            o_bias,
        ) = self.promote_dtype(
            inputs_q,
            kv_inputs,
            q_kernel,
            k_kernel,
            v_kernel,
            o_kernel,
            q_bias,
            k_bias,
            v_bias,
            o_bias,
            dtype=dtype,
        )

        q = _dense_general_apply(inputs_q, q_kernel, q_bias, precision=precision)
        k = _dense_general_apply(kv_inputs, k_kernel, k_bias, precision=precision)
        v = _dense_general_apply(kv_inputs, v_kernel, v_bias, precision=precision)

        attn_out = jax.nn.dot_product_attention(q, k, v, bias=bias, mask=mask)
        out = _project_output(attn_out, o_kernel, o_bias, precision=precision)
        return out

    return patched


@register_primitive(
    jaxpr_primitive="linen.multi_head_dot_product_attention",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.MultiHeadDotProductAttention",
    onnx=[
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        },
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="multi_head_dot_product_attention",
    testcases=[
        {
            "testcase": "multi_head_dot_product_attention_basic",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.MultiHeadDotProductAttention,
                input_shape=(1, 4, 8),
                num_heads=2,
                dropout_rate=0.0,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 8)],
            "expected_output_shapes": [("B", 4, 8)],
        },
        {
            "testcase": "multi_head_dot_product_attention_no_bias",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.MultiHeadDotProductAttention,
                input_shape=(1, 3, 6),
                num_heads=3,
                use_bias=False,
                dropout_rate=0.0,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 3, 6)],
            "expected_output_shapes": [("B", 3, 6)],
        },
    ],
)
class MultiHeadDotProductAttentionPlugin(PrimitiveLeafPlugin):
    """IR-only support for flax.linen.MultiHeadDotProductAttention via dot_product_attention."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.multi_head_dot_product_attention")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "MultiHeadDotProductAttention primitive should not reach lowering; it is inlined."
        )

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.MultiHeadDotProductAttention",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            )
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable):
        MultiHeadDotProductAttentionPlugin._ORIGINAL_CALL = orig_fn
        return _make_mha_patch(orig_fn)
