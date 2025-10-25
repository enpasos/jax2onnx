# jax2onnx/plugins/equinox/eqx/nn/rotary_positional_embedding.py

# status 2025-10-15
# TODO: use ONNX https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html
# support is still developing therefore we implement it via basic ops, see
# https://github.com/microsoft/onnxruntime/issues/26070
# opset23 rotary embedding fails on Win CUDA #26070


from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._utils import cast_param_like, inline_reshape_initializer
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_primitive,
    construct_and_call,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _rope_forward(x, cos_cache, sin_cache, *, embedding_size: int):
    rotated = _rotate_half(x)
    return (x * cos_cache) + (rotated * sin_cache)


def _rope_heads(embedding_size: int):
    """Apply RotaryPositionalEmbedding across attention heads as in the DINO example."""

    rope = eqx.nn.RotaryPositionalEmbedding(embedding_size=embedding_size)
    rotate_heads = eqx.filter_vmap(rope, in_axes=1, out_axes=1)

    def _call(heads):
        return rotate_heads(heads)

    return _call


class RotaryProcessHeads(eqx.Module):
    """process_heads adapter that applies RoPE to Q/K heads and forwards V."""

    rope: eqx.nn.RotaryPositionalEmbedding

    def __init__(self, rope: eqx.nn.RotaryPositionalEmbedding):
        self.rope = rope

    def __call__(
        self,
        query_heads: jax.Array,
        key_heads: jax.Array,
        value_heads: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        rotate = eqx.filter_vmap(self.rope, in_axes=1, out_axes=1)
        rotated_query = rotate(query_heads)
        rotated_key = rotate(key_heads)
        return rotated_query, rotated_key, value_heads


def compute_rope_caches(
    rope: eqx.nn.RotaryPositionalEmbedding, seq_length: int
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.float32(rope.theta)
    embedding_size = int(rope.embedding_size)
    base = np.arange(0.0, embedding_size, 2, dtype=np.float32)
    freq_exponent = base / np.float32(embedding_size)
    freqs = np.power(theta, freq_exponent).astype(np.float32)
    freqs = np.float32(1.0) / freqs
    positions = np.arange(float(seq_length), dtype=np.float32)
    freqs_outer = (positions[:, None] * freqs[None, :]).astype(np.float32)
    rope_dtype = np.dtype(jnp.dtype(rope.dtype))
    cos = np.cos(freqs_outer).astype(rope_dtype)
    sin = np.sin(freqs_outer).astype(rope_dtype)
    cos = np.tile(cos, (1, 2)).astype(rope_dtype)
    sin = np.tile(sin, (1, 2)).astype(rope_dtype)
    return cos, sin


def _value_dims(value: ir.Value) -> tuple[object, ...]:
    shape = getattr(value, "shape", None)
    if isinstance(shape, ir.Shape):
        return tuple(shape.dims)
    return ()


def _normalized_dim(dim: object) -> object:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    if isinstance(dim, str):
        return dim
    return dim


def lower_rotary_application(
    ctx,
    builder,
    x_val: ir.Value,
    cos_val: ir.Value,
    sin_val: ir.Value,
    *,
    embedding_size: int,
    prefix: str,
) -> ir.Value:
    embedding_half = embedding_size // 2
    x_dims = _value_dims(x_val)
    opset = getattr(builder, "opset", None)
    if opset is None:
        opset = getattr(builder, "opset_version", None)
    if opset is None:
        imports = getattr(builder, "opset_imports", {})
        if isinstance(imports, dict):
            opset = imports.get("", None)
    opset_int = int(opset) if opset is not None else 0
    use_fused = (
        opset_int >= 23
        and embedding_size % 2 == 0
        and embedding_half > 0
        and len(x_dims) in {2, 3}
        and hasattr(builder, "RotaryEmbedding")
    )

    if use_fused:
        original_dims = tuple(x_dims)
        attr_kwargs: dict[str, object] = {}
        restore_fn = None
        x_for_op = x_val

        x_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        if len(x_dims) == 2:
            axes_const = _const_i64(
                ctx,
                np.asarray([0], dtype=np.int64),
                f"{prefix}_unsqueeze_axes",
            )
            x_for_op = builder.Unsqueeze(
                x_val,
                axes_const,
                _outputs=[ctx.fresh_name(f"{prefix}_unsqueezed")],
            )
            if x_dtype is not None:
                x_for_op.type = ir.TensorType(x_dtype)
            unsqueezed_dims = (1,) + original_dims
            _stamp_type_and_shape(
                x_for_op, tuple(_normalized_dim(dim) for dim in unsqueezed_dims)
            )
            _ensure_value_metadata(ctx, x_for_op)

            def _restore(val: ir.Value) -> ir.Value:
                squeezed = builder.Squeeze(
                    val,
                    axes_const,
                    _outputs=[ctx.fresh_name(f"{prefix}_squeezed")],
                )
                if x_dtype is not None:
                    squeezed.type = ir.TensorType(x_dtype)
                _stamp_type_and_shape(
                    squeezed,
                    tuple(_normalized_dim(dim) for dim in original_dims),
                )
                _ensure_value_metadata(ctx, squeezed)
                return squeezed

            restore_fn = _restore
            attr_kwargs["num_heads"] = 1
        else:  # len(x_dims) == 3
            attr_kwargs["num_heads"] = 1

        def _slice_half(value: ir.Value, tag: str) -> ir.Value | None:
            dims = _value_dims(value)
            if not dims:
                return None
            axis_index = len(dims) - 1
            starts = _const_i64(
                ctx,
                np.asarray([0], dtype=np.int64),
                f"{tag}_start",
            )
            ends = _const_i64(
                ctx,
                np.asarray([embedding_half], dtype=np.int64),
                f"{tag}_end",
            )
            axes = _const_i64(
                ctx,
                np.asarray([axis_index], dtype=np.int64),
                f"{tag}_axes",
            )
            steps = _const_i64(
                ctx,
                np.asarray([1], dtype=np.int64),
                f"{tag}_steps",
            )
            sliced = builder.Slice(
                value,
                starts,
                ends,
                axes,
                steps,
                _outputs=[ctx.fresh_name(f"{tag}_half")],
            )
            val_dtype = getattr(getattr(value, "type", None), "dtype", None)
            if val_dtype is not None:
                sliced.type = ir.TensorType(val_dtype)
            new_dims = list(dims)
            new_dims[-1] = embedding_half
            _stamp_type_and_shape(
                sliced,
                tuple(_normalized_dim(dim) for dim in new_dims),
            )
            _ensure_value_metadata(ctx, sliced)
            return sliced

        cos_half = _slice_half(cos_val, f"{prefix}_cos")
        sin_half = _slice_half(sin_val, f"{prefix}_sin")

        def _align_cache(cache_val: ir.Value, tag: str) -> ir.Value:
            cache_dims = _value_dims(cache_val)
            cache_dtype = getattr(getattr(cache_val, "type", None), "dtype", None)
            if len(cache_dims) == 2:
                const_payload = getattr(cache_val, "const_value", None)
                if const_payload is not None:
                    try:
                        seq_len = int(cache_dims[0])
                        embed_half = int(cache_dims[1])
                    except Exception as exc:  # pragma: no cover - defensive
                        raise ValueError(
                            "RotaryEmbedding requires static rotary cache dimensions."
                        ) from exc
                    cache_val = inline_reshape_initializer(
                        ctx,
                        cache_val,
                        (1, seq_len, embed_half),
                        f"{prefix}_{tag}_reshape_const",
                    )
                    cache_dims = (1, seq_len, embed_half)
                    if cache_dtype is not None:
                        cache_val.type = ir.TensorType(cache_dtype)
                    _stamp_type_and_shape(
                        cache_val,
                        tuple(_normalized_dim(dim) for dim in cache_dims),
                    )
                    _ensure_value_metadata(ctx, cache_val)
                else:
                    axes_const = _const_i64(
                        ctx,
                        np.asarray([0], dtype=np.int64),
                        f"{tag}_unsqueeze_axes",
                    )
                    cache_val = builder.Unsqueeze(
                        cache_val,
                        axes_const,
                        _outputs=[ctx.fresh_name(f"{tag}_unsqueezed")],
                    )
                    if cache_dtype is not None:
                        cache_val.type = ir.TensorType(cache_dtype)
                    new_dims = (1,) + cache_dims
                    _stamp_type_and_shape(
                        cache_val,
                        tuple(_normalized_dim(dim) for dim in new_dims),
                    )
                    _ensure_value_metadata(ctx, cache_val)
                    cache_dims = new_dims

            target_dims = _value_dims(x_for_op)
            target_rank = len(target_dims)
            if len(cache_dims) == 3 and target_rank in (3, 4):
                seq_axis = target_rank - 2  # 3D -> 1, 4D -> 2
                match_batch = cache_dims[0] == target_dims[0]
                match_seq = cache_dims[1] == target_dims[seq_axis]
                match_heads = True
                if target_rank == 4:
                    match_heads = cache_dims[0] == target_dims[0] and (
                        cache_dims[0] == 1 or cache_dims[0] == target_dims[0]
                    )
                if not (match_batch and match_seq and match_heads):
                    shape_val = builder.Shape(
                        x_for_op,
                        _outputs=[ctx.fresh_name(f"{prefix}_{tag}_xshape")],
                    )
                    shape_val.type = ir.TensorType(ir.DataType.INT64)
                    _ensure_value_metadata(ctx, shape_val)

                    def _gather_dim(index: int, name: str) -> ir.Value:
                        idx = _const_i64(
                            ctx,
                            np.asarray([index], dtype=np.int64),
                            f"{prefix}_{tag}_{name}_idx",
                        )
                        dim_val = builder.Gather(
                            shape_val,
                            idx,
                            axis=0,
                            _outputs=[ctx.fresh_name(f"{prefix}_{tag}_{name}")],
                        )
                        dim_val.type = ir.TensorType(ir.DataType.INT64)
                        _stamp_type_and_shape(dim_val, (1,))
                        _ensure_value_metadata(ctx, dim_val)
                        return dim_val

                    batch_dim = _gather_dim(0, "batch")
                    embed_const = _const_i64(
                        ctx,
                        np.asarray([embedding_half], dtype=np.int64),
                        f"{prefix}_{tag}_embed",
                    )
                    if target_rank == 4:
                        head_dim = _gather_dim(1, "head")
                        seq_dim = _gather_dim(2, "seq")
                        target_shape = builder.Concat(
                            batch_dim,
                            head_dim,
                            seq_dim,
                            embed_const,
                            axis=0,
                            _outputs=[ctx.fresh_name(f"{prefix}_{tag}_expand_shape")],
                        )
                        shape_tuple = (4,)
                        aligned_dims = (
                            _normalized_dim(target_dims[0]),
                            _normalized_dim(target_dims[1]),
                            _normalized_dim(target_dims[2]),
                            _normalized_dim(embedding_half),
                        )
                    else:  # target_rank == 3
                        seq_dim = _gather_dim(1, "seq")
                        target_shape = builder.Concat(
                            batch_dim,
                            seq_dim,
                            embed_const,
                            axis=0,
                            _outputs=[ctx.fresh_name(f"{prefix}_{tag}_expand_shape")],
                        )
                        shape_tuple = (3,)
                        aligned_dims = (
                            _normalized_dim(target_dims[0]),
                            _normalized_dim(target_dims[1]),
                            _normalized_dim(embedding_half),
                        )
                    target_shape.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(target_shape, shape_tuple)
                    _ensure_value_metadata(ctx, target_shape)
                    cache_val = builder.Expand(
                        cache_val,
                        target_shape,
                        _outputs=[ctx.fresh_name(f"{prefix}_{tag}_expanded")],
                    )
                    if cache_dtype is not None:
                        cache_val.type = ir.TensorType(cache_dtype)
                    _stamp_type_and_shape(cache_val, aligned_dims)
                    _ensure_value_metadata(ctx, cache_val)
                else:
                    aligned_dims = tuple(_normalized_dim(dim) for dim in cache_dims)
                    _stamp_type_and_shape(cache_val, aligned_dims)
                    _ensure_value_metadata(ctx, cache_val)
            return cache_val

        if cos_half is not None and sin_half is not None:
            cos_half = _align_cache(cos_half, f"{prefix}_cos")
            sin_half = _align_cache(sin_half, f"{prefix}_sin")
            cos_half = cast_param_like(
                ctx,
                cos_half,
                x_for_op,
                name_hint=f"{prefix}_cos_half_cast",
            )
            sin_half = cast_param_like(
                ctx,
                sin_half,
                x_for_op,
                name_hint=f"{prefix}_sin_half_cast",
            )
            rotary = builder.RotaryEmbedding(
                x_for_op,
                cos_half,
                sin_half,
                _outputs=[ctx.fresh_name(f"{prefix}_rotary_embedding")],
                **attr_kwargs,
            )
            if x_dtype is not None:
                rotary.type = ir.TensorType(x_dtype)
            rotary_dims = _value_dims(rotary) or _value_dims(x_for_op)
            if rotary_dims:
                _stamp_type_and_shape(
                    rotary,
                    tuple(_normalized_dim(dim) for dim in rotary_dims),
                )
            _ensure_value_metadata(ctx, rotary)
            if callable(restore_fn):
                return restore_fn(rotary)
            return rotary

    half = embedding_size // 2
    dims = _value_dims(x_val)
    axis = len(dims) - 1 if dims else 0
    prefix_dims = dims[:-1]
    first_half_shape = prefix_dims + (half,)
    rotated_shape = prefix_dims + (embedding_size,)

    split_sizes = _const_i64(
        ctx,
        np.asarray([half, half], dtype=np.int64),
        ctx.fresh_name(f"{prefix}_split_sizes"),
    )
    split_outputs = [
        ctx.fresh_name(f"{prefix}_first_half"),
        ctx.fresh_name(f"{prefix}_second_half"),
    ]
    first_half, second_half = builder.Split(
        x_val,
        split_sizes,
        axis=axis,
        _outputs=split_outputs,
    )
    x_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
    if x_dtype is not None:
        first_half.type = ir.TensorType(x_dtype)
        second_half.type = ir.TensorType(x_dtype)
    _stamp_type_and_shape(first_half, first_half_shape)
    _stamp_type_and_shape(second_half, first_half_shape)
    _ensure_value_metadata(ctx, first_half)
    _ensure_value_metadata(ctx, second_half)

    neg_second = builder.Neg(
        second_half,
        _outputs=[ctx.fresh_name(f"{prefix}_neg_second")],
    )
    if x_dtype is not None:
        neg_second.type = ir.TensorType(x_dtype)
    _stamp_type_and_shape(neg_second, first_half_shape)
    _ensure_value_metadata(ctx, neg_second)

    rotated = builder.Concat(
        neg_second,
        first_half,
        axis=axis,
        _outputs=[ctx.fresh_name(f"{prefix}_rotated")],
    )
    if x_dtype is not None:
        rotated.type = ir.TensorType(x_dtype)
    _stamp_type_and_shape(rotated, rotated_shape)
    _ensure_value_metadata(ctx, rotated)

    x_mul_cos = builder.Mul(
        x_val,
        cos_val,
        _outputs=[ctx.fresh_name(f"{prefix}_mul_cos")],
    )
    if x_dtype is not None:
        x_mul_cos.type = ir.TensorType(x_dtype)
    _stamp_type_and_shape(x_mul_cos, rotated_shape)
    _ensure_value_metadata(ctx, x_mul_cos)

    rotated_mul_sin = builder.Mul(
        rotated,
        sin_val,
        _outputs=[ctx.fresh_name(f"{prefix}_mul_sin")],
    )
    if x_dtype is not None:
        rotated_mul_sin.type = ir.TensorType(x_dtype)
    _stamp_type_and_shape(rotated_mul_sin, rotated_shape)
    _ensure_value_metadata(ctx, rotated_mul_sin)

    output = builder.Add(
        x_mul_cos,
        rotated_mul_sin,
        _outputs=[ctx.fresh_name(f"{prefix}_output")],
    )
    if x_dtype is not None:
        output.type = ir.TensorType(x_dtype)
    _stamp_type_and_shape(output, rotated_shape)
    _ensure_value_metadata(ctx, output)
    return output


@register_primitive(
    jaxpr_primitive="eqx.nn.rotary_positional_embedding",
    jax_doc="https://docs.kidger.site/equinox/api/nn/embedding/#rotary-positional-embedding",
    onnx=[
        {
            "component": "Multiply",
            "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html",
        },
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
        {
            "component": "RotaryEmbedding",
            "doc": "https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html",
        },
    ],
    since="v0.10.0",
    context="primitives.eqx",
    component="rotary_positional_embedding",
    testcases=[
        {
            "testcase": "eqx_rotary_positional_embedding",
            "callable": construct_and_call(
                eqx.nn.RotaryPositionalEmbedding,
                embedding_size=32,
            ),
            "input_shapes": [(41, 32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["Split", "Mul", "Mul", "Add"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_rotary_positional_embedding_opset23",
            "callable": construct_and_call(
                eqx.nn.RotaryPositionalEmbedding,
                embedding_size=32,
            ),
            "input_shapes": [(41, 32)],
            "opset_version": 23,
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["RotaryEmbedding"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_rotary_positional_embedding_heads",
            "callable": construct_and_call(
                _rope_heads,
                embedding_size=64,
            ),
            "input_shapes": [(257, 6, 64)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                [
                    {"path": "Concat", "counts": {"Concat": 1}},
                    {"path": "Mul", "counts": {"Mul": 2}},
                    "Add",
                ],
                search_functions=True,
            ),
        },
        {
            "testcase": "eqx_rotary_positional_embedding_heads_opset23",
            "callable": construct_and_call(
                _rope_heads,
                embedding_size=64,
            ),
            "input_shapes": [(257, 6, 64)],
            "opset_version": 23,
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["RotaryEmbedding"],
                search_functions=True,
            ),
        },
    ],
)
class RotaryPositionalEmbeddingPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.rotary_positional_embedding")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, cos_cache, sin_cache, *, embedding_size: int):
        spec_x = jax.ShapeDtypeStruct(x.shape, x.dtype)
        spec_cos = jax.ShapeDtypeStruct(cos_cache.shape, cos_cache.dtype)
        spec_sin = jax.ShapeDtypeStruct(sin_cache.shape, sin_cache.dtype)
        out = jax.eval_shape(
            lambda *args: _rope_forward(*args, embedding_size=embedding_size),
            spec_x,
            spec_cos,
            spec_sin,
        )
        return jax.core.ShapedArray(out.shape, out.dtype)

    def lower(self, ctx, eqn):
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for RotaryPositionalEmbedding lowering"
            )

        x_var, cos_var, sin_var = eqn.invars
        out_var = eqn.outvars[0]
        embedding_size = int(eqn.params["embedding_size"])
        half = embedding_size // 2

        aval = getattr(x_var, "aval", None)
        aval_shape = tuple(getattr(aval, "shape", ()))
        split_axis = len(aval_shape) - 1 if aval_shape else 1
        if split_axis < 0:
            split_axis = 0
        prefix_dims = tuple(aval_shape[:-1]) if aval_shape else ()
        prefix_dims + (half,)
        prefix_dims + (embedding_size,)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("rope_input"))
        cos_val = cast_param_like(
            ctx,
            ctx.get_value_for_var(cos_var, name_hint=ctx.fresh_name("rope_cos")),
            x_val,
            name_hint="rope_cos_cast",
        )
        sin_val = cast_param_like(
            ctx,
            ctx.get_value_for_var(sin_var, name_hint=ctx.fresh_name("rope_sin")),
            x_val,
            name_hint="rope_sin_cast",
        )
        _stamp_type_and_shape(cos_val, (None, embedding_size))
        _stamp_type_and_shape(sin_val, (None, embedding_size))
        _ensure_value_metadata(ctx, cos_val)
        _ensure_value_metadata(ctx, sin_val)

        output = lower_rotary_application(
            ctx,
            builder,
            x_val,
            cos_val,
            sin_val,
            embedding_size=embedding_size,
            prefix="rope",
        )
        ctx.bind_value_for_var(out_var, output)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls):
        def _make_patch(orig):
            def wrapped(self, x, *, key=None):
                if key is not None:
                    raise NotImplementedError(
                        "RotaryPositionalEmbedding does not use RNG keys."
                    )
                seq_len_raw, embed = x.shape
                if embed != self.embedding_size:
                    raise ValueError(
                        f"x.shape[-1] must equal embedding_size ({embed} != {self.embedding_size})."
                    )
                seq_len = jax_core.concrete_or_error(
                    int,
                    seq_len_raw,
                    "RotaryPositionalEmbedding requires a static sequence length.",
                )
                rope_dtype = np.dtype(jnp.dtype(self.dtype))
                theta = np.float32(self.theta)
                emb_size = np.float32(self.embedding_size)
                base = np.arange(0.0, self.embedding_size, 2, dtype=np.float32)
                freq_exponent = base / emb_size
                freqs = np.power(theta, freq_exponent).astype(np.float32)
                freqs = np.float32(1.0) / freqs
                positions = np.arange(float(seq_len), dtype=np.float32)
                freqs_outer = (positions[:, None] * freqs[None, :]).astype(np.float32)
                cos = np.cos(freqs_outer).astype(rope_dtype)
                sin = np.sin(freqs_outer).astype(rope_dtype)
                cos = np.tile(cos, (1, 2)).astype(np.dtype(x.dtype))
                sin = np.tile(sin, (1, 2)).astype(np.dtype(x.dtype))
                cos = jnp.asarray(cos)
                sin = jnp.asarray(sin)
                return cls._PRIM.bind(
                    x,
                    cos,
                    sin,
                    embedding_size=self.embedding_size,
                )

            return wrapped

        return [
            AssignSpec(
                "equinox.nn",
                "rotary_positional_embedding_p",
                cls._PRIM,
                delete_if_missing=True,
            ),
            MonkeyPatchSpec(
                target=eqx.nn.RotaryPositionalEmbedding,
                attr="__call__",
                make_value=_make_patch,
                delete_if_missing=False,
            ),
        ]


@RotaryPositionalEmbeddingPlugin._PRIM.def_impl
def _rope_impl(*args, **params):
    return _rope_forward(*args, **params)


def _rope_batch_rule(batched_args, batch_dims, **params):
    x, cos_cache, sin_cache = batched_args
    x_bdim, cos_bdim, sin_bdim = batch_dims
    if cos_bdim is not None or sin_bdim is not None:
        raise NotImplementedError("Batching over rotary caches is not supported.")
    if x_bdim is None:
        return (
            RotaryPositionalEmbeddingPlugin._PRIM.bind(
                x, cos_cache, sin_cache, **params
            ),
            None,
        )
    if x_bdim != 0:
        x = jnp.moveaxis(x, x_bdim, 0)
    out = RotaryPositionalEmbeddingPlugin._PRIM.bind(x, cos_cache, sin_cache, **params)
    if x_bdim != 0:
        out = jnp.moveaxis(out, 0, x_bdim)
    return out, x_bdim


batching.primitive_batchers[RotaryPositionalEmbeddingPlugin._PRIM] = _rope_batch_rule
