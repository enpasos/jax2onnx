# jax2onnx/plugins/equinox/eqx/nn/multihead_attention.py

# TODO when widely supported, we can use https://onnx.ai/onnx/operators/onnx__Attention.html

from __future__ import annotations

import math
from typing import Any, Callable, ClassVar

import equinox as eqx
import jax
from jax import lax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._utils import cast_param_like, inline_reshape_initializer
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.equinox.eqx.nn.rotary_positional_embedding import (
    RotaryProcessHeads,
    compute_rope_caches,
    lower_rotary_application,
    _value_dims,
)
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_prng_key,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph


HeadsTuple = tuple[jax.Array, jax.Array, jax.Array]
ProcessHeadsFn = Callable[[jax.Array, jax.Array, jax.Array], HeadsTuple]


def _normalized_dim(dim: object) -> object:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    if isinstance(dim, str):
        return dim
    return dim


def _static_dims(dims: tuple[object, ...]) -> tuple[int, ...] | None:
    static_dims: list[int] = []
    for dim in dims:
        try:
            dim_val = int(dim)
        except Exception:  # symbolic or dynamic
            return None
        if dim_val < 0:
            return None
        static_dims.append(dim_val)
    return tuple(static_dims)


def _first_defined(*values: int | None) -> int | None:
    for value in values:
        if value is not None:
            return value
    return None


def _apply_process_heads_python(
    process_heads: ProcessHeadsFn,
    q_heads: jax.Array,
    k_heads: jax.Array,
    v_heads: jax.Array,
) -> HeadsTuple:
    if q_heads.ndim <= 3:
        return process_heads(q_heads, k_heads, v_heads)
    leading = q_heads.shape[:-3]
    if not leading:
        return process_heads(q_heads, k_heads, v_heads)
    seq_len, num_heads, q_width = (
        q_heads.shape[-3],
        q_heads.shape[-2],
        q_heads.shape[-1],
    )
    v_width = v_heads.shape[-1]
    flat = int(np.prod(leading))
    q_flat = q_heads.reshape((flat, seq_len, num_heads, q_width))
    k_flat = k_heads.reshape((flat, seq_len, num_heads, q_width))
    v_flat = v_heads.reshape((flat, seq_len, num_heads, v_width))
    mapped = jax.vmap(process_heads, in_axes=(0, 0, 0), out_axes=0)
    q_proc, k_proc, v_proc = mapped(q_flat, k_flat, v_flat)
    new_shape_q = leading + (seq_len, num_heads, q_width)
    new_shape_v = leading + (seq_len, num_heads, v_width)
    return (
        q_proc.reshape(new_shape_q),
        k_proc.reshape(new_shape_q),
        v_proc.reshape(new_shape_v),
    )


def _apply_rotary_process_heads_lowering(
    ctx: LoweringContextProtocol,
    builder: Any,
    process_heads: object,
    q_heads: ir.Value,
    k_heads: ir.Value,
    *,
    q_seq_static: int | None,
    k_seq_static: int | None,
    prefix: str,
) -> tuple[ir.Value, ir.Value]:
    if q_seq_static is None or k_seq_static is None:
        raise NotImplementedError(
            "Rotary process_heads requires a static sequence length."
        )
    if q_seq_static != k_seq_static:
        raise NotImplementedError(
            "Query and key sequence lengths must match for rotary process_heads."
        )

    def _pad_with_identity(
        cos_tail: np.ndarray,
        sin_tail: np.ndarray,
        prefix_tokens: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        seq_len = prefix_tokens + cos_tail.shape[0]
        dtype = np.result_type(cos_tail.dtype, sin_tail.dtype)
        full_cos = np.ones((seq_len, cos_tail.shape[1]), dtype=dtype)
        full_sin = np.zeros((seq_len, sin_tail.shape[1]), dtype=dtype)
        if prefix_tokens:
            full_cos[prefix_tokens:, :] = cos_tail.astype(dtype)
            full_sin[prefix_tokens:, :] = sin_tail.astype(dtype)
        else:
            full_cos[:, :] = cos_tail.astype(dtype)
            full_sin[:, :] = sin_tail.astype(dtype)
        return full_cos, full_sin

    seq_len = int(q_seq_static)
    if isinstance(process_heads, RotaryProcessHeads):
        rope_module = getattr(process_heads, "rope", None)
        if rope_module is None and not hasattr(process_heads, "sin"):
            raise NotImplementedError(
                "RotaryProcessHeads instance missing rope attribute."
            )
        if rope_module is not None and not hasattr(process_heads, "sin"):
            embedding_size = int(rope_module.embedding_size)
            cos_np, sin_np = compute_rope_caches(rope_module, seq_len)
        elif hasattr(process_heads, "sin") and hasattr(process_heads, "cos"):
            sin_arr = np.asarray(process_heads.sin)
            cos_arr = np.asarray(process_heads.cos)
            if sin_arr.shape != cos_arr.shape:
                raise ValueError("sin and cos caches must have matching shapes.")
            tail_len, embedding_size = sin_arr.shape
            prefix_tokens = int(getattr(process_heads, "prefix_tokens", 0))
            if prefix_tokens + tail_len != seq_len:
                raise NotImplementedError(
                    "DinoRoPE process_heads length does not match attention sequence."
                )
            cos_np, sin_np = _pad_with_identity(cos_arr, sin_arr, prefix_tokens)
        else:
            raise NotImplementedError(
                "Unsupported RotaryProcessHeads subclass without sin/cos caches."
            )
    elif hasattr(process_heads, "sin") and hasattr(process_heads, "cos"):
        sin_arr = np.asarray(process_heads.sin)
        cos_arr = np.asarray(process_heads.cos)
        if sin_arr.shape != cos_arr.shape:
            raise ValueError("sin and cos caches must have matching shapes.")
        tail_len, embedding_size = sin_arr.shape
        prefix_tokens = int(getattr(process_heads, "prefix_tokens", 0))
        if prefix_tokens + tail_len != seq_len:
            raise NotImplementedError(
                "DinoRoPE process_heads length does not match attention sequence."
            )
        cos_np, sin_np = _pad_with_identity(cos_arr, sin_arr, prefix_tokens)
    else:
        raise NotImplementedError(
            "Only RotaryProcessHeads-style callbacks are supported for process_heads."
        )

    cos_val_q = ctx.bind_const_for_var(object(), cos_np)
    sin_val_q = ctx.bind_const_for_var(object(), sin_np)
    _stamp_type_and_shape(cos_val_q, (seq_len, embedding_size))
    _stamp_type_and_shape(sin_val_q, (seq_len, embedding_size))
    _ensure_value_metadata(ctx, cos_val_q)
    _ensure_value_metadata(ctx, sin_val_q)

    def _reshape_cache(
        cache_val: ir.Value, dims: tuple[object, ...], tag: str
    ) -> ir.Value:
        rank = len(dims)
        if rank <= 2:
            return cache_val
        reshape_vals = np.ones(rank, dtype=np.int64)
        seq_axis = rank - 3
        if seq_axis < 0:
            seq_axis = 0
        reshape_vals[seq_axis] = seq_len
        reshape_vals[-1] = embedding_size
        target_shape = tuple(int(v) for v in reshape_vals)
        const_payload = getattr(cache_val, "const_value", None)
        if const_payload is not None:
            original = cache_val
            cache_val = inline_reshape_initializer(
                ctx,
                cache_val,
                target_shape,
                name_hint=f"{prefix}_{tag}_reshape_inline",
            )
            try:
                ctx.builder.initializers.remove(original)
            except (ValueError, AttributeError):
                try:
                    ctx._initializers.remove(original)
                except Exception:
                    pass
        else:
            reshape_shape = _const_i64(
                ctx,
                reshape_vals,
                f"{prefix}_{tag}_reshape_shape",
            )
            cache_val = builder.Reshape(
                cache_val,
                reshape_shape,
                _outputs=[ctx.fresh_name(f"{prefix}_{tag}_reshape")],
            )
        out_meta = []
        for idx in range(rank):
            if idx == seq_axis:
                out_meta.append(_normalized_dim(seq_len))
            elif idx == rank - 1:
                out_meta.append(_normalized_dim(embedding_size))
            else:
                out_meta.append(_normalized_dim(1))
        _stamp_type_and_shape(cache_val, tuple(out_meta))
        _ensure_value_metadata(ctx, cache_val)
        return cache_val

    dims_q = _value_dims(q_heads)
    cos_val_q = _reshape_cache(cos_val_q, dims_q, "q_cos")
    sin_val_q = _reshape_cache(sin_val_q, dims_q, "q_sin")

    cos_val_q = cast_param_like(
        ctx, cos_val_q, q_heads, name_hint=f"{prefix}_q_cos_cast"
    )
    sin_val_q = cast_param_like(
        ctx, sin_val_q, q_heads, name_hint=f"{prefix}_q_sin_cast"
    )
    _ensure_value_metadata(ctx, cos_val_q)
    _ensure_value_metadata(ctx, sin_val_q)
    q_heads = lower_rotary_application(
        ctx,
        builder,
        q_heads,
        cos_val_q,
        sin_val_q,
        embedding_size=embedding_size,
        prefix=f"{prefix}_q",
    )

    cos_val_k = ctx.bind_const_for_var(object(), cos_np)
    sin_val_k = ctx.bind_const_for_var(object(), sin_np)
    _stamp_type_and_shape(cos_val_k, (seq_len, embedding_size))
    _stamp_type_and_shape(sin_val_k, (seq_len, embedding_size))
    _ensure_value_metadata(ctx, cos_val_k)
    _ensure_value_metadata(ctx, sin_val_k)
    dims_k = _value_dims(k_heads)
    cos_val_k = _reshape_cache(cos_val_k, dims_k, "k_cos")
    sin_val_k = _reshape_cache(sin_val_k, dims_k, "k_sin")

    cos_val_k = cast_param_like(
        ctx, cos_val_k, k_heads, name_hint=f"{prefix}_k_cos_cast"
    )
    sin_val_k = cast_param_like(
        ctx, sin_val_k, k_heads, name_hint=f"{prefix}_k_sin_cast"
    )
    _ensure_value_metadata(ctx, cos_val_k)
    _ensure_value_metadata(ctx, sin_val_k)
    k_heads = lower_rotary_application(
        ctx,
        builder,
        k_heads,
        cos_val_k,
        sin_val_k,
        embedding_size=embedding_size,
        prefix=f"{prefix}_k",
    )

    return q_heads, k_heads


def _linear_forward(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array,
    *,
    use_bias: bool,
) -> jax.Array:
    orig_shape = x.shape[:-1]
    x_flat = x.reshape((-1, x.shape[-1]))
    out_flat = jnp.dot(x_flat, jnp.swapaxes(weight, -1, -2))
    if use_bias:
        out_flat = out_flat + bias
    return out_flat.reshape(orig_shape + (weight.shape[0],))


def _mha_forward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    q_weight: jax.Array,
    q_bias: jax.Array,
    k_weight: jax.Array,
    k_bias: jax.Array,
    v_weight: jax.Array,
    v_bias: jax.Array,
    o_weight: jax.Array,
    o_bias: jax.Array,
    *,
    num_heads: int,
    qk_size: int,
    vo_size: int,
    use_query_bias: bool,
    use_key_bias: bool,
    use_value_bias: bool,
    use_output_bias: bool,
    process_heads: ProcessHeadsFn | None = None,
) -> jax.Array:
    query_proj = _linear_forward(query, q_weight, q_bias, use_bias=use_query_bias)
    key_proj = _linear_forward(key, k_weight, k_bias, use_bias=use_key_bias)
    value_proj = _linear_forward(value, v_weight, v_bias, use_bias=use_value_bias)

    q_heads = query_proj.reshape(query.shape[0], num_heads, qk_size)
    k_heads = key_proj.reshape(key.shape[0], num_heads, qk_size)
    v_heads = value_proj.reshape(value.shape[0], num_heads, vo_size)

    if process_heads is not None:
        q_heads, k_heads, v_heads = _apply_process_heads_python(
            process_heads, q_heads, k_heads, v_heads
        )

    q_heads = lax.transpose(q_heads, (1, 0, 2))  # (num_heads, Q, d_k)
    k_heads = lax.transpose(k_heads, (1, 2, 0))  # (num_heads, d_k, K)
    v_heads = lax.transpose(v_heads, (1, 0, 2))  # (num_heads, K, d_v)

    scale = 1.0 / math.sqrt(float(qk_size))

    def _matmul_heads(lhs, rhs):
        return lax.dot_general(
            lhs,
            rhs,
            dimension_numbers=(((1,), (0,)), ((), ())),
        )

    scores = jax.vmap(_matmul_heads)(q_heads, k_heads) * scale
    attn = jax.nn.softmax(scores, axis=-1)
    context = jax.vmap(_matmul_heads)(attn, v_heads)
    context = lax.transpose(context, (1, 0, 2)).reshape(
        query.shape[0], num_heads * vo_size
    )

    output = _linear_forward(context, o_weight, o_bias, use_bias=use_output_bias)
    return output


def _attention_core(
    dim: int, num_heads: int, *, key: jax.Array
) -> Callable[[jax.Array], jax.Array]:
    """Mirror the AttentionCore example (batch of sequences)."""

    attn = eqx.nn.MultiheadAttention(
        num_heads=num_heads,
        query_size=dim,
        key_size=dim,
        value_size=dim,
        output_size=dim,
        use_query_bias=True,
        use_key_bias=True,
        use_value_bias=True,
        use_output_bias=True,
        key=key,
    )

    def _call(tokens: jax.Array) -> jax.Array:
        def _attend(single_tokens: jax.Array) -> jax.Array:
            return attn(single_tokens, single_tokens, single_tokens, inference=True)

        return eqx.filter_vmap(_attend, in_axes=0, out_axes=0)(tokens)

    return _call


@register_primitive(
    jaxpr_primitive="eqx.nn.multihead_attention",
    jax_doc="https://docs.kidger.site/equinox/api/nn/attention/",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        },
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
    ],
    since="v0.10.0",
    context="primitives.eqx",
    component="multihead_attention",
    testcases=[
        {
            "testcase": "eqx_multihead_attention",
            "callable": construct_and_call(
                eqx.nn.MultiheadAttention,
                num_heads=4,
                query_size=32,
                key_size=32,
                value_size=32,
                output_size=32,
                key=with_prng_key(0),
            ),
            "input_shapes": [(17, 32), (17, 32), (17, 32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                [
                    {"path": "Gemm", "counts": {"Gemm": 4}},
                    {"path": "MatMul", "counts": {"MatMul": 2}},
                    {"path": "Softmax", "counts": {"Softmax": 1}},
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_multihead_attention_core",
            "callable": construct_and_call(
                _attention_core, dim=384, num_heads=6, key=with_prng_key(0)
            ),
            "input_shapes": [("B", 257, 384)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                [
                    "Reshape:?x384 -> Gemm -> Reshape:Bx257x6x64 -> Reshape:?x?x6x64 -> "
                    "Transpose:?x6x?x64 -> MatMul:?x6x?x? -> Mul:?x6x?x? -> "
                    "Softmax:?x6x?x? -> MatMul:?x6x?x64 -> Transpose:?x?x6x64 -> "
                    "Reshape:?x384 -> Gemm -> Reshape:Bx257x384"
                ],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
            "run_only_dynamic": True,
        },
        {
            "testcase": "eqx_multihead_attention_core_static",
            "callable": construct_and_call(
                _attention_core, dim=384, num_heads=6, key=with_prng_key(0)
            ),
            "input_shapes": [(3, 257, 384)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                [
                    "Reshape:771x384 -> Gemm:771x384 -> Reshape:3x257x6x64",
                    "Transpose:3x6x257x64 -> MatMul:3x6x257x257 -> Mul:3x6x257x257 -> "
                    "Softmax:3x6x257x257 -> MatMul:3x6x257x64 -> Transpose:3x257x6x64",
                    "Reshape:771x384 -> Gemm:771x384 -> Reshape:3x257x384",
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class MultiheadAttentionPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.multihead_attention")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        query: jax_core.AbstractValue,
        key: jax_core.AbstractValue,
        value: jax_core.AbstractValue,
        q_weight: jax_core.AbstractValue,
        q_bias: jax_core.AbstractValue,
        k_weight: jax_core.AbstractValue,
        k_bias: jax_core.AbstractValue,
        v_weight: jax_core.AbstractValue,
        v_bias: jax_core.AbstractValue,
        o_weight: jax_core.AbstractValue,
        o_bias: jax_core.AbstractValue,
        **params: Any,
    ) -> jax_core.ShapedArray:
        query_shape = tuple(getattr(query, "shape", ()))
        query_dtype = getattr(query, "dtype", None)
        o_weight_shape = tuple(getattr(o_weight, "shape", ()))

        if o_weight_shape:
            output_dim = o_weight_shape[0]
        else:
            num_heads = params.get("num_heads", 1)
            vo_size = params.get("vo_size", query_shape[-1] if query_shape else 1)
            output_dim = num_heads * vo_size

        output_shape = query_shape[:-1] + (output_dim,)
        if query_dtype is None:
            query_dtype = getattr(value, "dtype", getattr(key, "dtype", jnp.float32))
        return jax_core.ShapedArray(output_shape, query_dtype)

    def lower(
        self,
        ctx: LoweringContextProtocol,
        eqn: jax_core.JaxprEqn,
    ) -> None:
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for Eqx MultiheadAttention lowering"
            )

        (
            query_var,
            key_var,
            value_var,
            q_weight_var,
            q_bias_var,
            k_weight_var,
            k_bias_var,
            v_weight_var,
            v_bias_var,
            o_weight_var,
            o_bias_var,
        ) = eqn.invars
        out_var = eqn.outvars[0]

        params = dict(eqn.params)
        num_heads = int(params["num_heads"])
        qk_size = int(params["qk_size"])
        vo_size = int(params["vo_size"])
        use_query_bias = bool(params["use_query_bias"])
        use_key_bias = bool(params["use_key_bias"])
        use_value_bias = bool(params["use_value_bias"])
        use_output_bias = bool(params["use_output_bias"])
        process_heads_param = params.get("process_heads")

        query_val = ctx.get_value_for_var(
            query_var, name_hint=ctx.fresh_name("mha_query")
        )
        key_val = ctx.get_value_for_var(key_var, name_hint=ctx.fresh_name("mha_key"))
        value_val = ctx.get_value_for_var(
            value_var, name_hint=ctx.fresh_name("mha_value")
        )

        q_weight = cast_param_like(
            ctx,
            ctx.get_value_for_var(q_weight_var, name_hint=ctx.fresh_name("mha_wq")),
            query_val,
            name_hint="mha_wq_cast",
        )
        q_bias = cast_param_like(
            ctx,
            ctx.get_value_for_var(q_bias_var, name_hint=ctx.fresh_name("mha_bq")),
            query_val,
            name_hint="mha_bq_cast",
        )
        k_weight = cast_param_like(
            ctx,
            ctx.get_value_for_var(k_weight_var, name_hint=ctx.fresh_name("mha_wk")),
            key_val,
            name_hint="mha_wk_cast",
        )
        k_bias = cast_param_like(
            ctx,
            ctx.get_value_for_var(k_bias_var, name_hint=ctx.fresh_name("mha_bk")),
            key_val,
            name_hint="mha_bk_cast",
        )
        v_weight = cast_param_like(
            ctx,
            ctx.get_value_for_var(v_weight_var, name_hint=ctx.fresh_name("mha_wv")),
            value_val,
            name_hint="mha_wv_cast",
        )
        v_bias = cast_param_like(
            ctx,
            ctx.get_value_for_var(v_bias_var, name_hint=ctx.fresh_name("mha_bv")),
            value_val,
            name_hint="mha_bv_cast",
        )
        o_weight = cast_param_like(
            ctx,
            ctx.get_value_for_var(o_weight_var, name_hint=ctx.fresh_name("mha_wo")),
            query_val,
            name_hint="mha_wo_cast",
        )
        o_bias = cast_param_like(
            ctx,
            ctx.get_value_for_var(o_bias_var, name_hint=ctx.fresh_name("mha_bo")),
            query_val,
            name_hint="mha_bo_cast",
        )

        def _prepare_input(val, var, prefix):
            aval_shape = tuple(getattr(getattr(var, "aval", None), "shape", ()))
            rank = len(aval_shape)
            axis = max(rank - 1, 0)

            leading_dims = None
            sequence_dim_val = None
            leading_static_dims: tuple[int, ...] | None = None
            seq_len_static: int | None = None
            feature_static: int | None = None
            seq_len_hint: int | None = None
            if rank > 0:
                leading_static_dims = _static_dims(aval_shape[:-1])
                try:
                    feature_candidate = int(aval_shape[-1]) if aval_shape else None
                except Exception:
                    feature_candidate = None
                if feature_candidate is not None and feature_candidate >= 0:
                    feature_static = feature_candidate
                if leading_static_dims is not None and leading_static_dims:
                    seq_len_static = leading_static_dims[-1]
                    seq_len_hint = seq_len_static
                elif rank >= 2:
                    try:
                        seq_candidate = int(aval_shape[-2])
                    except Exception:
                        seq_candidate = None
                    if seq_candidate is not None and seq_candidate >= 0:
                        seq_len_static = seq_candidate
                        seq_len_hint = seq_candidate

            if seq_len_static is not None and sequence_dim_val is None:
                sequence_dim_val = _const_i64(
                    ctx,
                    np.asarray([seq_len_static], dtype=np.int64),
                    f"{prefix}_seq_len_hint",
                )
                _ensure_value_metadata(ctx, sequence_dim_val)

            flatten_shape: ir.Value | None = None
            flatten_first_static: int | None = None
            if rank > 1:
                if leading_static_dims is not None and feature_static is not None:
                    if leading_static_dims:
                        leading_vals = np.asarray(leading_static_dims, dtype=np.int64)
                        leading_dims = _const_i64(
                            ctx,
                            leading_vals,
                            f"{prefix}_leading_static",
                        )
                        _ensure_value_metadata(ctx, leading_dims)
                        if seq_len_static is not None:
                            sequence_dim_val = _const_i64(
                                ctx,
                                np.asarray([seq_len_static], dtype=np.int64),
                                f"{prefix}_seq_len_static",
                            )
                            _ensure_value_metadata(ctx, sequence_dim_val)
                    try:
                        flatten_first_static = int(
                            np.prod(tuple(int(dim) for dim in leading_static_dims))
                        )
                    except Exception:
                        flatten_first_static = None
                    if flatten_first_static is not None:
                        flatten_vals = np.asarray(
                            [flatten_first_static, feature_static], dtype=np.int64
                        )
                    else:
                        flatten_vals = np.asarray([-1, feature_static], dtype=np.int64)
                    flatten_shape = _const_i64(
                        ctx,
                        flatten_vals,
                        f"{prefix}_flatten_shape_static",
                    )
                    _ensure_value_metadata(ctx, flatten_shape)
                else:
                    shape_vec = builder.Shape(
                        val,
                        _outputs=[ctx.fresh_name(f"{prefix}_shape")],
                    )
                    shape_vec.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(shape_vec, (rank,))
                    _ensure_value_metadata(ctx, shape_vec)

                    start = _const_i64(
                        ctx,
                        np.asarray([0], dtype=np.int64),
                        f"{prefix}_leading_start",
                    )
                    end = _const_i64(
                        ctx,
                        np.asarray([axis], dtype=np.int64),
                        f"{prefix}_leading_end",
                    )
                    axes = _const_i64(
                        ctx,
                        np.asarray([0], dtype=np.int64),
                        f"{prefix}_leading_axes",
                    )
                    steps = _const_i64(
                        ctx,
                        np.asarray([1], dtype=np.int64),
                        f"{prefix}_leading_steps",
                    )
                    leading_dims = builder.Slice(
                        shape_vec,
                        start,
                        end,
                        axes,
                        steps,
                        _outputs=[ctx.fresh_name(f"{prefix}_leading")],
                    )
                    leading_dims.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(leading_dims, (axis,))
                    _ensure_value_metadata(ctx, leading_dims)

                    if axis > 0:
                        seq_index = _const_i64(
                            ctx,
                            np.asarray([axis - 1], dtype=np.int64),
                            f"{prefix}_seq_index",
                        )
                        seq_scalar = builder.Gather(
                            shape_vec,
                            seq_index,
                            axis=0,
                            _outputs=[ctx.fresh_name(f"{prefix}_seq_scalar")],
                        )
                        seq_scalar.type = ir.TensorType(ir.DataType.INT64)
                        _stamp_type_and_shape(seq_scalar, (1,))
                        _ensure_value_metadata(ctx, seq_scalar)
                        seq_shape = _const_i64(
                            ctx,
                            np.asarray([1], dtype=np.int64),
                            f"{prefix}_seq_shape",
                        )
                        sequence_dim_val = builder.Reshape(
                            seq_scalar,
                            seq_shape,
                            _outputs=[ctx.fresh_name(f"{prefix}_seq_dim")],
                        )
                        sequence_dim_val.type = ir.TensorType(ir.DataType.INT64)
                        _stamp_type_and_shape(sequence_dim_val, (1,))
                        _ensure_value_metadata(ctx, sequence_dim_val)
                    else:
                        sequence_dim_val = _const_i64(
                            ctx,
                            np.asarray([1], dtype=np.int64),
                            f"{prefix}_seq_default",
                        )
                        _ensure_value_metadata(ctx, sequence_dim_val)

                    feature_index = _const_i64(
                        ctx,
                        np.asarray([axis], dtype=np.int64),
                        f"{prefix}_feature_index",
                    )
                    feature_scalar = builder.Gather(
                        shape_vec,
                        feature_index,
                        axis=0,
                        _outputs=[ctx.fresh_name(f"{prefix}_feature_scalar")],
                    )
                    feature_scalar.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(feature_scalar, (1,))
                    _ensure_value_metadata(ctx, feature_scalar)
                    feature_shape = _const_i64(
                        ctx,
                        np.asarray([1], dtype=np.int64),
                        f"{prefix}_feature_shape",
                    )
                    feature_dim = builder.Reshape(
                        feature_scalar,
                        feature_shape,
                        _outputs=[ctx.fresh_name(f"{prefix}_feature_dim")],
                    )
                    feature_dim.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(feature_dim, (1,))
                    _ensure_value_metadata(ctx, feature_dim)
                    minus_one = _const_i64(
                        ctx,
                        np.asarray([-1], dtype=np.int64),
                        f"{prefix}_minus_one",
                    )
                    flatten_shape = builder.Concat(
                        minus_one,
                        feature_dim,
                        axis=0,
                        _outputs=[ctx.fresh_name(f"{prefix}_flatten_shape")],
                    )
                    flatten_shape.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(flatten_shape, (2,))
                    _ensure_value_metadata(ctx, flatten_shape)

                flat = builder.Reshape(
                    val,
                    flatten_shape,
                    _outputs=[ctx.fresh_name(f"{prefix}_flat")],
                )
                val_dtype = getattr(getattr(val, "type", None), "dtype", None)
                if val_dtype is not None:
                    flat.type = ir.TensorType(val_dtype)
                flat_first_meta = None
                if flatten_first_static is not None:
                    flat_first_meta = _normalized_dim(flatten_first_static)
                elif leading_static_dims is not None:
                    try:
                        flat_first_meta = _normalized_dim(
                            int(np.prod(tuple(int(dim) for dim in leading_static_dims)))
                        )
                    except Exception:
                        flat_first_meta = None
                flat_second_meta = None
                if feature_static is not None:
                    flat_second_meta = _normalized_dim(feature_static)
                elif aval_shape:
                    flat_second_meta = _normalized_dim(aval_shape[-1])
                flat_meta = (flat_first_meta, flat_second_meta)
                _stamp_type_and_shape(flat, flat_meta)
                _ensure_value_metadata(ctx, flat)
            else:
                flat = val

            if sequence_dim_val is None:
                sequence_dim_val = _const_i64(
                    ctx,
                    np.asarray([1], dtype=np.int64),
                    f"{prefix}_seq_fallback",
                )
                _ensure_value_metadata(ctx, sequence_dim_val)

            return (
                flat,
                leading_dims,
                aval_shape,
                sequence_dim_val,
                leading_static_dims,
                flatten_first_static,
                flat_meta,
                seq_len_hint,
            )

        def _project(
            x_val,
            weight_val,
            bias_val,
            tail_dims: tuple[int, int],
            prefix,
            *,
            use_bias,
            leading_dims,
            aval_shape,
            leading_static_dims: tuple[int, ...] | None,
            flat_meta,
        ):
            gemm_inputs = [x_val, weight_val]
            beta = 1.0 if use_bias else 0.0
            if use_bias:
                gemm_inputs.append(bias_val)
            proj = builder.Gemm(
                *gemm_inputs,
                alpha=1.0,
                beta=beta,
                transB=1,
                _outputs=[ctx.fresh_name(f"{prefix}_proj")],
            )
            proj_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
            if proj_dtype is not None:
                proj.type = ir.TensorType(proj_dtype)
            if isinstance(flat_meta, tuple):
                _stamp_type_and_shape(proj, flat_meta)
                _ensure_value_metadata(ctx, proj)
            reshape_tail = _const_i64(
                ctx,
                np.asarray(tail_dims, dtype=np.int64),
                f"{prefix}_tail",
            )
            if leading_static_dims:
                reshape_vals = np.asarray(
                    tuple(int(dim) for dim in leading_static_dims) + tail_dims,
                    dtype=np.int64,
                )
                reshape_shape = _const_i64(
                    ctx,
                    reshape_vals,
                    f"{prefix}_shape_static",
                )
                _stamp_type_and_shape(reshape_shape, (len(reshape_vals),))
                _ensure_value_metadata(ctx, reshape_shape)
            elif leading_dims is not None:
                reshape_shape = builder.Concat(
                    leading_dims,
                    reshape_tail,
                    axis=0,
                    _outputs=[ctx.fresh_name(f"{prefix}_shape")],
                )
                concat_len = len(aval_shape[:-1]) + len(tail_dims)
                reshape_shape.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(reshape_shape, (concat_len,))
                _ensure_value_metadata(ctx, reshape_shape)
            else:
                reshape_shape = reshape_tail
            reshaped = builder.Reshape(
                proj,
                reshape_shape,
                _outputs=[ctx.fresh_name(f"{prefix}_heads")],
            )
            if proj_dtype is not None:
                reshaped.type = ir.TensorType(proj_dtype)
            if aval_shape:
                meta_shape = tuple(
                    _normalized_dim(dim) for dim in aval_shape[:-1]
                ) + tuple(_normalized_dim(dim) for dim in tail_dims)
            else:
                meta_shape = (None,) + tuple(_normalized_dim(dim) for dim in tail_dims)
            _stamp_type_and_shape(reshaped, meta_shape)
            _ensure_value_metadata(ctx, reshaped)
            return reshaped

        def _reshape_heads(
            val,
            prefix,
            tail_dims: tuple[int, int],
            seq_dim_val: ir.Value | None,
            leading_static_dims: tuple[int, ...] | None,
            static_seq_hint: int | None,
        ):
            batch_static: int | None = None
            seq_static: int | None = static_seq_hint
            if leading_static_dims is not None:
                seq_static = leading_static_dims[-1] if leading_static_dims else 1
                prefix_dims = leading_static_dims[:-1] if leading_static_dims else ()
                batch_static = int(np.prod(prefix_dims)) if prefix_dims else 1
                full_vals = np.asarray(
                    (batch_static, seq_static) + tail_dims,
                    dtype=np.int64,
                )
                full_shape = _const_i64(
                    ctx,
                    full_vals,
                    f"{prefix}_reshape4_shape_static",
                )
                _stamp_type_and_shape(full_shape, (len(full_vals),))
                _ensure_value_metadata(ctx, full_shape)
            else:
                if seq_dim_val is None:
                    seq_dim_val = _const_i64(
                        ctx,
                        np.asarray([1], dtype=np.int64),
                        f"{prefix}_seq_auto",
                    )
                    _ensure_value_metadata(ctx, seq_dim_val)
                minus_one = _const_i64(
                    ctx,
                    np.asarray([-1], dtype=np.int64),
                    f"{prefix}_batch_flat",
                )
                prefix_shape = builder.Concat(
                    minus_one,
                    seq_dim_val,
                    axis=0,
                    _outputs=[ctx.fresh_name(f"{prefix}_batch_seq_shape")],
                )
                prefix_shape.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(prefix_shape, (2,))
                _ensure_value_metadata(ctx, prefix_shape)
                tail_vals = _const_i64(
                    ctx,
                    np.asarray(tail_dims, dtype=np.int64),
                    f"{prefix}_tail_shape",
                )
                full_shape = builder.Concat(
                    prefix_shape,
                    tail_vals,
                    axis=0,
                    _outputs=[ctx.fresh_name(f"{prefix}_reshape4_shape")],
                )
                full_shape.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(full_shape, (2 + len(tail_dims),))
                _ensure_value_metadata(ctx, full_shape)
            reshaped = builder.Reshape(
                val,
                full_shape,
                _outputs=[ctx.fresh_name(f"{prefix}_reshape4")],
            )
            val_dtype = getattr(getattr(val, "type", None), "dtype", None)
            if val_dtype is not None:
                reshaped.type = ir.TensorType(val_dtype)
            if leading_static_dims is not None:
                seq_static = leading_static_dims[-1] if leading_static_dims else 1
                prefix_dims = leading_static_dims[:-1] if leading_static_dims else ()
                batch_static = int(np.prod(prefix_dims)) if prefix_dims else 1
                reshape_meta = (
                    _normalized_dim(batch_static),
                    _normalized_dim(seq_static),
                ) + tuple(_normalized_dim(d) for d in tail_dims)
            else:
                reshape_meta = (None, None) + tuple(
                    _normalized_dim(d) for d in tail_dims
                )
            _stamp_type_and_shape(reshaped, reshape_meta)
            _ensure_value_metadata(ctx, reshaped)
            return reshaped, batch_static, seq_static

        (
            query_flat,
            query_leading,
            query_aval_shape,
            query_seq_dim,
            query_leading_static,
            _query_flatten_static,
            query_flat_meta,
            query_seq_static_hint,
        ) = _prepare_input(query_val, query_var, "mha_query")
        (
            key_flat,
            key_leading,
            key_aval_shape,
            key_seq_dim,
            key_leading_static,
            _key_flatten_static,
            key_flat_meta,
            key_seq_static_hint,
        ) = _prepare_input(key_val, key_var, "mha_key")
        (
            value_flat,
            value_leading,
            value_aval_shape,
            value_seq_dim,
            value_leading_static,
            _value_flatten_static,
            value_flat_meta,
            value_seq_static_hint,
        ) = _prepare_input(value_val, value_var, "mha_value")

        q_heads = _project(
            query_flat,
            q_weight,
            q_bias,
            tail_dims=(num_heads, qk_size),
            prefix="mha_q",
            use_bias=use_query_bias,
            leading_dims=query_leading,
            aval_shape=query_aval_shape,
            leading_static_dims=query_leading_static,
            flat_meta=query_flat_meta,
        )
        k_heads = _project(
            key_flat,
            k_weight,
            k_bias,
            tail_dims=(num_heads, qk_size),
            prefix="mha_k",
            use_bias=use_key_bias,
            leading_dims=key_leading,
            aval_shape=key_aval_shape,
            leading_static_dims=key_leading_static,
            flat_meta=key_flat_meta,
        )
        v_heads = _project(
            value_flat,
            v_weight,
            v_bias,
            tail_dims=(num_heads, vo_size),
            prefix="mha_v",
            use_bias=use_value_bias,
            leading_dims=value_leading,
            aval_shape=value_aval_shape,
            leading_static_dims=value_leading_static,
            flat_meta=value_flat_meta,
        )

        q_heads, q_batch_static, q_seq_static = _reshape_heads(
            q_heads,
            "mha_q",
            (num_heads, qk_size),
            query_seq_dim,
            query_leading_static,
            query_seq_static_hint,
        )
        k_heads, k_batch_static, k_seq_static = _reshape_heads(
            k_heads,
            "mha_k",
            (num_heads, qk_size),
            key_seq_dim,
            key_leading_static,
            key_seq_static_hint,
        )
        v_heads, v_batch_static, v_seq_static = _reshape_heads(
            v_heads,
            "mha_v",
            (num_heads, vo_size),
            value_seq_dim,
            value_leading_static,
            value_seq_static_hint,
        )

        def _is_rotary_like(candidate) -> bool:
            if candidate is None:
                return False
            if isinstance(candidate, RotaryProcessHeads):
                return True
            return hasattr(candidate, "sin") and hasattr(candidate, "cos")

        if _is_rotary_like(process_heads_param):
            q_heads, k_heads = _apply_rotary_process_heads_lowering(
                ctx,
                builder,
                process_heads_param,
                q_heads,
                k_heads,
                q_seq_static=q_seq_static,
                k_seq_static=k_seq_static,
                prefix=ctx.fresh_name("mha_rotary_process"),
            )

        q_trans = builder.Transpose(
            q_heads,
            perm=(0, 2, 1, 3),
            _outputs=[ctx.fresh_name("mha_q_trans")],
        )
        q_dtype = getattr(getattr(q_heads, "type", None), "dtype", None)
        if q_dtype is not None:
            q_trans.type = ir.TensorType(q_dtype)
        q_trans_meta = (
            _normalized_dim(q_batch_static),
            _normalized_dim(num_heads),
            _normalized_dim(q_seq_static),
            _normalized_dim(qk_size),
        )
        _stamp_type_and_shape(q_trans, q_trans_meta)
        _ensure_value_metadata(ctx, q_trans)

        k_trans = builder.Transpose(
            k_heads,
            perm=(0, 2, 3, 1),
            _outputs=[ctx.fresh_name("mha_k_trans")],
        )
        k_dtype = getattr(getattr(k_heads, "type", None), "dtype", None)
        if k_dtype is not None:
            k_trans.type = ir.TensorType(k_dtype)
        k_trans_meta = (
            _normalized_dim(_first_defined(k_batch_static, q_batch_static)),
            _normalized_dim(num_heads),
            _normalized_dim(qk_size),
            _normalized_dim(k_seq_static),
        )
        _stamp_type_and_shape(k_trans, k_trans_meta)
        _ensure_value_metadata(ctx, k_trans)

        v_trans = builder.Transpose(
            v_heads,
            perm=(0, 2, 1, 3),
            _outputs=[ctx.fresh_name("mha_v_trans")],
        )
        v_dtype = getattr(getattr(v_heads, "type", None), "dtype", None)
        if v_dtype is not None:
            v_trans.type = ir.TensorType(v_dtype)
        v_trans_meta = (
            _normalized_dim(_first_defined(v_batch_static, q_batch_static)),
            _normalized_dim(num_heads),
            _normalized_dim(v_seq_static),
            _normalized_dim(vo_size),
        )
        _stamp_type_and_shape(v_trans, v_trans_meta)
        _ensure_value_metadata(ctx, v_trans)

        scores = builder.MatMul(
            q_trans,
            k_trans,
            _outputs=[ctx.fresh_name("mha_scores")],
        )
        scores_dtype = q_dtype
        if scores_dtype is not None:
            scores.type = ir.TensorType(scores_dtype)
        scores_meta = (
            _normalized_dim(_first_defined(q_batch_static, k_batch_static)),
            _normalized_dim(num_heads),
            _normalized_dim(q_seq_static),
            _normalized_dim(k_seq_static),
        )
        _stamp_type_and_shape(scores, scores_meta)
        _ensure_value_metadata(ctx, scores)

        scale_value = ctx.bind_const_for_var(
            object(), np.asarray(1.0 / math.sqrt(float(qk_size)), dtype=np.float32)
        )
        scale_cast = cast_param_like(
            ctx, scale_value, scores, name_hint="mha_scale_cast"
        )
        _stamp_type_and_shape(scale_cast, ())
        _ensure_value_metadata(ctx, scale_cast)
        scaled_scores = builder.Mul(
            scores,
            scale_cast,
            _outputs=[ctx.fresh_name("mha_scaled_scores")],
        )
        if scores_dtype is not None:
            scaled_scores.type = ir.TensorType(scores_dtype)
        _stamp_type_and_shape(scaled_scores, scores_meta)
        _ensure_value_metadata(ctx, scaled_scores)

        attn = builder.Softmax(
            scaled_scores,
            axis=-1,
            _outputs=[ctx.fresh_name("mha_attn")],
        )
        if scores_dtype is not None:
            attn.type = ir.TensorType(scores_dtype)
        _stamp_type_and_shape(attn, scores_meta)
        _ensure_value_metadata(ctx, attn)

        context = builder.MatMul(
            attn,
            v_trans,
            _outputs=[ctx.fresh_name("mha_context")],
        )
        if scores_dtype is not None:
            context.type = ir.TensorType(scores_dtype)
        context_meta = (
            _normalized_dim(_first_defined(q_batch_static, v_batch_static)),
            _normalized_dim(num_heads),
            _normalized_dim(q_seq_static),
            _normalized_dim(vo_size),
        )
        _stamp_type_and_shape(context, context_meta)
        _ensure_value_metadata(ctx, context)

        context_trans = builder.Transpose(
            context,
            perm=(0, 2, 1, 3),
            _outputs=[ctx.fresh_name("mha_context_trans")],
        )
        if scores_dtype is not None:
            context_trans.type = ir.TensorType(scores_dtype)
        context_trans_meta = (
            _normalized_dim(_first_defined(q_batch_static, v_batch_static)),
            _normalized_dim(q_seq_static),
            _normalized_dim(num_heads),
            _normalized_dim(vo_size),
        )
        _stamp_type_and_shape(context_trans, context_trans_meta)
        _ensure_value_metadata(ctx, context_trans)

        if q_batch_static is not None and q_seq_static is not None:
            context_flat_vals = np.asarray(
                [q_batch_static * q_seq_static, num_heads * vo_size],
                dtype=np.int64,
            )
        else:
            context_flat_vals = np.asarray(
                [-1, num_heads * vo_size],
                dtype=np.int64,
            )
        context_flat_shape = _const_i64(
            ctx,
            context_flat_vals,
            "mha_context_flat_shape",
        )
        _stamp_type_and_shape(context_flat_shape, (2,))
        _ensure_value_metadata(ctx, context_flat_shape)
        context_flat = builder.Reshape(
            context_trans,
            context_flat_shape,
            _outputs=[ctx.fresh_name("mha_context_flat")],
        )
        if scores_dtype is not None:
            context_flat.type = ir.TensorType(scores_dtype)
        flat_first = None
        if q_batch_static is not None and q_seq_static is not None:
            flat_first = _normalized_dim(q_batch_static * q_seq_static)
        _stamp_type_and_shape(
            context_flat,
            (
                flat_first,
                _normalized_dim(num_heads * vo_size),
            ),
        )
        _ensure_value_metadata(ctx, context_flat)

        gemm_inputs = [context_flat, o_weight]
        beta = 1.0 if use_output_bias else 0.0
        if use_output_bias:
            gemm_inputs.append(o_bias)
        output = builder.Gemm(
            *gemm_inputs,
            alpha=1.0,
            beta=beta,
            transB=1,
            _outputs=[ctx.fresh_name("mha_output")],
        )
        out_dtype = getattr(getattr(query_val, "type", None), "dtype", None)
        if out_dtype is not None:
            output.type = ir.TensorType(out_dtype)
        context_flat_shape = getattr(getattr(context_flat, "shape", None), "dims", None)
        if isinstance(context_flat_shape, tuple):
            output_meta = tuple(_normalized_dim(dim) for dim in context_flat_shape)
            _stamp_type_and_shape(output, output_meta)
            _ensure_value_metadata(ctx, output)

        o_weight_shape = getattr(getattr(o_weight_var, "aval", None), "shape", None)
        if o_weight_shape is None and getattr(o_weight, "shape", None) is not None:
            o_weight_shape = tuple(getattr(o_weight, "shape"))
        if not o_weight_shape:
            o_weight_shape = (vo_size * num_heads,)
        output_dim = int(o_weight_shape[0])
        output_tail = _const_i64(
            ctx,
            np.asarray([output_dim], dtype=np.int64),
            "mha_output_tail",
        )
        if query_leading_static is not None:
            output_vals = np.asarray(
                tuple(int(dim) for dim in query_leading_static) + (output_dim,),
                dtype=np.int64,
            )
            output_shape_val = _const_i64(
                ctx,
                output_vals,
                "mha_output_shape_static",
            )
            _stamp_type_and_shape(output_shape_val, (len(output_vals),))
            _ensure_value_metadata(ctx, output_shape_val)
        elif query_leading is not None:
            output_shape_val = builder.Concat(
                query_leading,
                output_tail,
                axis=0,
                _outputs=[ctx.fresh_name("mha_output_shape")],
            )
            out_len = len(query_aval_shape[:-1]) + 1
            output_shape_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(output_shape_val, (out_len,))
            _ensure_value_metadata(ctx, output_shape_val)
        else:
            output_shape_val = output_tail

        final_output = builder.Reshape(
            output,
            output_shape_val,
            _outputs=[ctx.fresh_name("mha_output_final")],
        )
        if out_dtype is not None:
            final_output.type = ir.TensorType(out_dtype)
        output_meta_shape = tuple(
            _normalized_dim(dim) for dim in query_aval_shape[:-1]
        ) + (_normalized_dim(output_dim),)
        _stamp_type_and_shape(final_output, output_meta_shape)
        _ensure_value_metadata(ctx, final_output)
        ctx.bind_value_for_var(out_var, final_output)

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def _normalize_process_heads(cls, process_heads: object | None) -> object | None:
        if process_heads is None:
            return None
        if isinstance(process_heads, RotaryProcessHeads):
            return process_heads
        if hasattr(process_heads, "sin") and hasattr(process_heads, "cos"):
            return process_heads
        raise NotImplementedError(
            "process_heads callable is not supported by the MultiheadAttention lowering."
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_patch(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            def wrapped(
                self: eqx.nn.MultiheadAttention,
                query: jax.Array,
                key_: jax.Array,
                value: jax.Array,
                mask: jax.Array | None = None,
                *,
                key: jax.Array | None = None,
                inference: bool | None = None,
                deterministic: bool | None = None,
                process_heads: ProcessHeadsFn | RotaryProcessHeads | None = None,
            ) -> jax.Array:
                if mask is not None:
                    raise NotImplementedError(
                        "eqx.nn.MultiheadAttention mask handling is not supported"
                    )
                if key is not None:
                    raise NotImplementedError(
                        "Dropout RNG for MultiheadAttention is not supported"
                    )
                if deterministic is not None and inference is None:
                    inference = deterministic
                if inference not in (None, True):
                    raise NotImplementedError(
                        "Inference overrides for Dropout are not supported"
                    )
                process_param = cls._normalize_process_heads(process_heads)
                dropout = getattr(self, "dropout", None)
                if dropout is not None and getattr(dropout, "p", 0.0):
                    raise NotImplementedError(
                        "Dropout in MultiheadAttention is not supported"
                    )

                q_weight = jnp.asarray(self.query_proj.weight, dtype=query.dtype)
                k_weight = jnp.asarray(self.key_proj.weight, dtype=key_.dtype)
                v_weight = jnp.asarray(self.value_proj.weight, dtype=value.dtype)
                o_weight = jnp.asarray(self.output_proj.weight, dtype=query.dtype)

                def _bias(
                    weight: jax.Array,
                    bias_attr: jax.Array | None,
                    use_bias: bool,
                    dtype: jnp.dtype,
                ) -> jax.Array:
                    if not use_bias or bias_attr is None:
                        return jnp.zeros((weight.shape[0],), dtype=dtype)
                    return jnp.asarray(bias_attr, dtype=dtype)

                q_bias = _bias(
                    q_weight, self.query_proj.bias, self.use_query_bias, query.dtype
                )
                k_bias = _bias(
                    k_weight, self.key_proj.bias, self.use_key_bias, key_.dtype
                )
                v_bias = _bias(
                    v_weight, self.value_proj.bias, self.use_value_bias, value.dtype
                )
                o_bias = _bias(
                    o_weight, self.output_proj.bias, self.use_output_bias, query.dtype
                )

                return cls._PRIM.bind(
                    query,
                    key_,
                    value,
                    q_weight,
                    q_bias,
                    k_weight,
                    k_bias,
                    v_weight,
                    v_bias,
                    o_weight,
                    o_bias,
                    num_heads=self.num_heads,
                    qk_size=self.qk_size,
                    vo_size=self.vo_size,
                    use_query_bias=self.use_query_bias,
                    use_key_bias=self.use_key_bias,
                    use_value_bias=self.use_value_bias,
                    use_output_bias=self.use_output_bias,
                    process_heads=process_param,
                )

            return wrapped

        return [
            AssignSpec(
                "equinox.nn", "multihead_attention_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target=eqx.nn.MultiheadAttention,
                attr="__call__",
                make_value=_make_patch,
                delete_if_missing=False,
            ),
        ]


@MultiheadAttentionPlugin._PRIM.def_impl
def _multihead_attention_impl(*args: Any, **params: Any) -> jax.Array:
    return _mha_forward(*args, **params)


def _mha_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | None, ...],
    **params: Any,
) -> tuple[jax.Array, int | None]:
    (
        query,
        key,
        value,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        o_weight,
        o_bias,
    ) = batched_args
    (
        q_bdim,
        k_bdim,
        v_bdim,
        qw_bdim,
        qb_bdim,
        kw_bdim,
        kb_bdim,
        vw_bdim,
        vb_bdim,
        ow_bdim,
        ob_bdim,
    ) = batch_dims

    if any(
        dim is not None
        for dim in (
            qw_bdim,
            qb_bdim,
            kw_bdim,
            kb_bdim,
            vw_bdim,
            vb_bdim,
            ow_bdim,
            ob_bdim,
        )
    ):
        raise NotImplementedError(
            "Batching over MultiheadAttention parameters is not supported."
        )

    if q_bdim != k_bdim or q_bdim != v_bdim:
        raise NotImplementedError("Inconsistent batching of MHA inputs.")

    if q_bdim is None:
        out = MultiheadAttentionPlugin._PRIM.bind(
            query,
            key,
            value,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            o_weight,
            o_bias,
            **params,
        )
        return out, None

    if q_bdim != 0:
        query = jnp.moveaxis(query, q_bdim, 0)
    if k_bdim != 0:
        key = jnp.moveaxis(key, k_bdim, 0)
    if v_bdim != 0:
        value = jnp.moveaxis(value, v_bdim, 0)

    out = MultiheadAttentionPlugin._PRIM.bind(
        query,
        key,
        value,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        o_weight,
        o_bias,
        **params,
    )
    if q_bdim != 0:
        out = jnp.moveaxis(out, 0, q_bdim)
    return out, q_bdim


batching.primitive_batchers[MultiheadAttentionPlugin._PRIM] = _mha_batch_rule
