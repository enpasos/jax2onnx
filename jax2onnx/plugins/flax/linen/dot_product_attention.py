# jax2onnx/plugins/flax/linen/dot_product_attention.py

from __future__ import annotations

from typing import Callable, ClassVar

import numpy as np
import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Primitive
from flax import linen as nn

from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_primitive,
)


def _normalize_attention_mask(mask):
    if mask is None:
        return None
    if mask.ndim == 3:
        return mask[:, None, :, :]
    if mask.ndim == 4:
        return mask
    return None


def _masked_softmax(weights, mask_bool):
    if mask_bool is None:
        return weights
    mask_float = mask_bool.astype(weights.dtype)
    masked_weights = weights * mask_float
    sum_weights = masked_weights.sum(axis=-1, keepdims=True)
    denom_nonzero = sum_weights > 0
    safe_denom = jnp.where(denom_nonzero, sum_weights, jnp.ones_like(sum_weights))
    normalized = masked_weights / safe_denom
    is_f64 = np.dtype(weights.dtype) == np.dtype(np.float64)
    nan_value = jnp.asarray(np.nan if is_f64 else 0.0, dtype=weights.dtype)
    return jnp.where(denom_nonzero, normalized, nan_value)


@register_primitive(
    jaxpr_primitive="linen.dot_product_attention_weights",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.dot_product_attention_weights",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        },
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="dot_product_attention_weights",
    testcases=[
        {
            "testcase": "dot_product_attention_weights_basic",
            "callable": lambda q, k: nn.dot_product_attention_weights(q, k),
            "input_shapes": [(2, 4, 2, 8), (2, 4, 2, 8)],
            "expected_output_shapes": [(2, 2, 4, 4)],
        },
    ],
)
class DotProductAttentionWeightsPlugin(PrimitiveLeafPlugin):
    """Support flax.linen.dot_product_attention_weights via JAX ops."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.dot_product_attention_weights")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "dot_product_attention_weights primitive should not reach lowering; it is inlined."
        )

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.attention",
                attr="dot_product_attention_weights",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="dot_product_attention_weights",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable):
        DotProductAttentionWeightsPlugin._ORIGINAL_CALL = orig_fn

        def patched(query, key, *args, **kwargs):
            bias = None
            mask = None
            if len(args) >= 1:
                bias = args[0]
            if len(args) >= 2:
                mask = args[1]
            if len(args) > 2:
                return orig_fn(query, key, *args, **kwargs)

            kwargs_local = dict(kwargs)
            bias = kwargs_local.pop("bias", bias)
            mask = kwargs_local.pop("mask", mask)
            dropout_rate = kwargs_local.pop("dropout_rate", 0.0)
            deterministic = kwargs_local.pop("deterministic", None)
            dropout_rng = kwargs_local.pop("dropout_rng", None)
            broadcast_dropout = kwargs_local.pop("broadcast_dropout", None)
            dtype = kwargs_local.pop("dtype", None)
            precision = kwargs_local.pop("precision", None)

            if kwargs_local:
                return orig_fn(query, key, *args, **kwargs)

            if dropout_rate not in (0, 0.0) or dropout_rng is not None:
                return orig_fn(query, key, *args, **kwargs)
            if deterministic is not None and deterministic is not True:
                return orig_fn(query, key, *args, **kwargs)
            if broadcast_dropout not in (None, False):
                return orig_fn(query, key, *args, **kwargs)

            if query.ndim != 4 or key.ndim != 4:
                return orig_fn(query, key, *args, **kwargs)

            head_dim = query.shape[-1]
            if not isinstance(head_dim, (int, np.integer)):
                return orig_fn(query, key, *args, **kwargs)

            if dtype is not None and dtype != query.dtype:
                query = query.astype(dtype)
                key = key.astype(dtype)
                if bias is not None:
                    bias = bias.astype(dtype)

            q_dtype = np.dtype(query.dtype)
            if not np.issubdtype(q_dtype, np.floating):
                return orig_fn(query, key, *args, **kwargs)

            logits = jnp.einsum(
                "BTNH,BSNH->BNTS",
                query,
                key,
                precision=precision,
            )
            scale = jnp.asarray(1.0 / np.sqrt(float(head_dim)), dtype=logits.dtype)
            logits = logits * scale

            if bias is not None:
                logits = logits + bias

            mask_bool = None
            if mask is not None:
                mask = _normalize_attention_mask(mask)
                if mask is None:
                    return orig_fn(query, key, *args, **kwargs)
                mask_bool = mask if mask.dtype == jnp.bool_ else mask.astype(jnp.bool_)
                neg_inf = jnp.asarray(np.finfo(q_dtype).min, dtype=logits.dtype)
                logits = jnp.where(mask_bool, logits, neg_inf)

            weights = jax.nn.softmax(logits, axis=-1)
            weights = _masked_softmax(weights, mask_bool)
            return weights

        return patched


@register_primitive(
    jaxpr_primitive="linen.dot_product_attention",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.dot_product_attention",
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
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        },
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="dot_product_attention",
    testcases=[
        {
            "testcase": "dot_product_attention_basic",
            "callable": lambda q, k, v: nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 2, 8), (2, 4, 2, 8), (2, 4, 2, 8)],
            "expected_output_shapes": [(2, 4, 2, 8)],
        },
    ],
)
class DotProductAttentionPlugin(PrimitiveLeafPlugin):
    """Support flax.linen.dot_product_attention by routing to jax.nn.dot_product_attention."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.dot_product_attention")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "dot_product_attention primitive should not reach lowering; it is inlined."
        )

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.attention",
                attr="dot_product_attention",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="dot_product_attention",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable):
        DotProductAttentionPlugin._ORIGINAL_CALL = orig_fn

        def patched(query, key, value, *args, **kwargs):
            bias = None
            mask = None
            if len(args) >= 1:
                bias = args[0]
            if len(args) >= 2:
                mask = args[1]
            if len(args) > 2:
                return orig_fn(query, key, value, *args, **kwargs)

            kwargs_local = dict(kwargs)
            bias = kwargs_local.pop("bias", bias)
            mask = kwargs_local.pop("mask", mask)
            dropout_rate = kwargs_local.pop("dropout_rate", 0.0)
            deterministic = kwargs_local.pop("deterministic", None)
            dropout_rng = kwargs_local.pop("dropout_rng", None)
            broadcast_dropout = kwargs_local.pop("broadcast_dropout", None)
            dtype = kwargs_local.pop("dtype", None)
            precision = kwargs_local.pop("precision", None)

            if kwargs_local:
                return orig_fn(query, key, value, *args, **kwargs)

            if bias is not None:
                return orig_fn(query, key, value, *args, **kwargs)
            if dropout_rate not in (0, 0.0) or dropout_rng is not None:
                return orig_fn(query, key, value, *args, **kwargs)
            if deterministic is not None and deterministic is not True:
                return orig_fn(query, key, value, *args, **kwargs)
            if broadcast_dropout not in (None, False):
                return orig_fn(query, key, value, *args, **kwargs)
            if precision is not None:
                return orig_fn(query, key, value, *args, **kwargs)

            if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
                return orig_fn(query, key, value, *args, **kwargs)

            if dtype is not None and dtype != query.dtype:
                query = query.astype(dtype)
                key = key.astype(dtype)
                value = value.astype(dtype)

            norm_mask = _normalize_attention_mask(mask)
            if mask is not None and norm_mask is None:
                return orig_fn(query, key, value, *args, **kwargs)
            mask = norm_mask

            return jax.nn.dot_product_attention(query, key, value, mask=mask)

        return patched


@register_primitive(
    jaxpr_primitive="linen.make_attention_mask",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.make_attention_mask",
    onnx=[
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="make_attention_mask",
    testcases=[
        {
            "testcase": "make_attention_mask_basic",
            "callable": lambda q, k: nn.make_attention_mask(q, k),
            "input_shapes": [(2, 4), (2, 5)],
            "expected_output_shapes": [(2, 1, 4, 5)],
        },
    ],
)
class MakeAttentionMaskPlugin(PrimitiveLeafPlugin):
    """Support flax.linen.make_attention_mask via JAX ops."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.make_attention_mask")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "make_attention_mask primitive should not reach lowering; it is inlined."
        )

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.attention",
                attr="make_attention_mask",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="make_attention_mask",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable):
        MakeAttentionMaskPlugin._ORIGINAL_CALL = orig_fn

        def patched(query_input, key_input, *args, **kwargs):
            pairwise_fn = None
            extra_batch_dims = 0
            dtype = None
            if len(args) == 1:
                pairwise_fn = args[0]
            elif len(args) == 2:
                pairwise_fn, extra_batch_dims = args
            elif len(args) == 3:
                pairwise_fn, extra_batch_dims, dtype = args
            elif len(args) > 2:
                return orig_fn(query_input, key_input, *args, **kwargs)

            kwargs_local = dict(kwargs)
            pairwise_fn = kwargs_local.pop("pairwise_fn", pairwise_fn)
            extra_batch_dims = kwargs_local.pop("extra_batch_dims", extra_batch_dims)
            dtype = kwargs_local.pop("dtype", dtype)
            if kwargs_local:
                return orig_fn(query_input, key_input, *args, **kwargs)

            if pairwise_fn is None:
                pairwise_fn = jnp.multiply
            if not callable(pairwise_fn):
                return orig_fn(query_input, key_input, *args, **kwargs)
            try:
                extra_batch_dims = int(extra_batch_dims or 0)
            except Exception:
                return orig_fn(query_input, key_input, *args, **kwargs)
            if extra_batch_dims < 0:
                return orig_fn(query_input, key_input, *args, **kwargs)
            if dtype is None:
                dtype = jnp.float32

            mask = pairwise_fn(
                jnp.expand_dims(query_input, axis=-1),
                jnp.expand_dims(key_input, axis=-2),
            )
            mask = jnp.expand_dims(mask, axis=-3)
            mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
            return mask.astype(dtype)

        return patched


@register_primitive(
    jaxpr_primitive="linen.make_causal_mask",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.make_causal_mask",
    onnx=[
        {"component": "Less", "doc": "https://onnx.ai/onnx/operators/onnx__Less.html"},
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="make_causal_mask",
    testcases=[
        {
            "testcase": "make_causal_mask_basic",
            "callable": lambda x: nn.make_causal_mask(x),
            "input_shapes": [(2, 4)],
            "expected_output_shapes": [(2, 1, 4, 4)],
        },
    ],
)
class MakeCausalMaskPlugin(PrimitiveLeafPlugin):
    """Support flax.linen.make_causal_mask via JAX ops."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.make_causal_mask")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "make_causal_mask primitive should not reach lowering; it is inlined."
        )

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.attention",
                attr="make_causal_mask",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="make_causal_mask",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable):
        MakeCausalMaskPlugin._ORIGINAL_CALL = orig_fn

        def patched(x, *args, **kwargs):
            extra_batch_dims = 0
            dtype = None
            if len(args) == 1:
                extra_batch_dims = args[0]
            elif len(args) == 2:
                extra_batch_dims, dtype = args
            elif len(args) > 2:
                return orig_fn(x, *args, **kwargs)

            kwargs_local = dict(kwargs)
            extra_batch_dims = kwargs_local.pop("extra_batch_dims", extra_batch_dims)
            dtype = kwargs_local.pop("dtype", dtype)
            if kwargs_local:
                return orig_fn(x, *args, **kwargs)

            try:
                extra_batch_dims = int(extra_batch_dims or 0)
            except Exception:
                return orig_fn(x, *args, **kwargs)
            if extra_batch_dims < 0:
                return orig_fn(x, *args, **kwargs)
            if dtype is None:
                dtype = jnp.float32

            idxs = jnp.cumsum(jnp.ones_like(x, dtype=jnp.int32), axis=-1) - 1
            return nn.make_attention_mask(
                idxs,
                idxs,
                jnp.greater_equal,
                extra_batch_dims=extra_batch_dims,
                dtype=dtype,
            )

        return patched
