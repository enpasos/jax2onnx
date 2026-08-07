# jax2onnx/plugins/flax/nnx/group_norm.py

from __future__ import annotations
from typing import Any, Callable, ClassVar, Final, Sequence, cast

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import dtypes as nnx_dtypes
import numpy as np

import onnx_ir as ir
from jax2onnx.converter.typing_support import (
    IRBuilderProtocol,
    LoweringContextProtocol,
)
from jax2onnx._compat.jax import JaxprEqn, Primitive, ShapedArray
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.lax._opset_utils import builder_reduce_with_axes

GROUP_NORM_PRIM: Final[Primitive] = Primitive("nnx.group_norm")
GROUP_NORM_PRIM.multiple_results = False


EXPECT_GROUP_NORM_FALLBACK: Final = EG(
    [
        (
            "Reshape -> ReduceMean",
            {
                "counts": {
                    "GroupNormalization": 0,
                    "ReduceMean": 2,
                }
            },
        )
    ]
)

GROUP_NORM_ONNX_COMPONENTS: Final = [
    {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    {
        "component": "CastLike",
        "doc": "https://onnx.ai/onnx/operators/onnx__CastLike.html",
    },
    {
        "component": "Concat",
        "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
    },
    {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    {"component": "Max", "doc": "https://onnx.ai/onnx/operators/onnx__Max.html"},
    {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
    {
        "component": "ReduceMean",
        "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMean.html",
    },
    {
        "component": "ReduceProd",
        "doc": "https://onnx.ai/onnx/operators/onnx__ReduceProd.html",
    },
    {
        "component": "Reshape",
        "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
    },
    {
        "component": "Shape",
        "doc": "https://onnx.ai/onnx/operators/onnx__Shape.html",
    },
    {
        "component": "Slice",
        "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
    },
    {"component": "Sqrt", "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html"},
    {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
]


def _require_builder(ctx: LoweringContextProtocol) -> IRBuilderProtocol:
    return ctx.builder


@register_primitive(
    jaxpr_primitive=GROUP_NORM_PRIM.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.GroupNorm",
    onnx=GROUP_NORM_ONNX_COMPONENTS,
    since="0.2.0",
    context="primitives.nnx",
    component="group_norm",
    testcases=[
        {
            "testcase": "group_norm",
            "callable": construct_and_call(
                nnx.GroupNorm,
                num_features=64,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(11, 2, 2, 64)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
        {
            "testcase": "group_norm_rank2",
            "callable": construct_and_call(
                nnx.GroupNorm,
                num_features=8,
                num_groups=4,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
        {
            "testcase": "group_norm_rank4",
            "callable": construct_and_call(
                nnx.GroupNorm,
                num_features=64,
                num_groups=8,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 7, 7, 64)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
        {
            "testcase": "group_norm_no_bias",
            "callable": construct_and_call(
                nnx.GroupNorm,
                num_features=32,
                num_groups=8,
                use_bias=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 5, 5, 32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
        {
            "testcase": "group_norm_no_bias_no_scale",
            "callable": construct_and_call(
                nnx.GroupNorm,
                num_features=32,
                num_groups=8,
                use_bias=False,
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 16, 16, 32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
        {
            "testcase": "group_norm_bias_no_scale",
            "callable": construct_and_call(
                nnx.GroupNorm,
                num_features=32,
                num_groups=8,
                use_bias=True,
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 16, 16, 32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
        {
            "testcase": "group_norm_no_scale",
            "callable": construct_and_call(
                nnx.GroupNorm,
                num_features=32,
                num_groups=8,
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 5, 5, 32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
        {
            "testcase": "group_norm_no_bias_scale",
            "callable": construct_and_call(
                nnx.GroupNorm,
                num_features=32,
                num_groups=8,
                use_bias=False,
                use_scale=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 16, 16, 32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
        {
            "testcase": "group_norm_bias_scale",
            "callable": construct_and_call(
                nnx.GroupNorm,
                num_features=32,
                num_groups=8,
                use_bias=True,
                use_scale=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 16, 16, 32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
    ],
)
class GroupNormPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for framework-faithful flax.nnx.GroupNorm decomposition."""

    _PRIM: ClassVar[Primitive] = GROUP_NORM_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------------- abstract eval ----------------
    @staticmethod
    def abstract_eval(
        x: Any,
        scale: Any,
        bias: Any,
        *,
        epsilon: float,
        num_groups: int,
        channel_axis: int,
        use_fast_variance: bool,
        clamp_negative_variance: bool,
        batch_rank: int,
    ) -> ShapedArray:
        del (
            scale,
            bias,
            epsilon,
            num_groups,
            channel_axis,
            use_fast_variance,
            clamp_negative_variance,
            batch_rank,
        )
        return ShapedArray(x.shape, x.dtype)

    # ---------------- lowering ----------------
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var, scale_var, bias_var = eqn.invars[:3]
        y_var = eqn.outvars[0]

        params = dict(getattr(eqn, "params", {}) or {})
        epsilon = float(params.get("epsilon", 1e-5))
        num_groups = int(params.get("num_groups", 1))
        channel_axis = int(params.get("channel_axis", -1))
        use_fast_variance = bool(params.get("use_fast_variance", False))
        clamp_negative_variance = bool(
            params.get("clamp_negative_variance", use_fast_variance)
        )
        batch_rank = int(params.get("batch_rank", 1))

        builder = _require_builder(ctx)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        scale_val = ctx.get_value_for_var(scale_var, name_hint=ctx.fresh_name("scale"))
        bias_val = ctx.get_value_for_var(bias_var, name_hint=ctx.fresh_name("bias"))

        scale_val = cast_param_like(ctx, scale_val, x_val, name_hint="gn_scale_cast")
        bias_val = cast_param_like(ctx, bias_val, x_val, name_hint="gn_bias_cast")

        original_x_val = x_val
        original_x_shape: tuple[Any, ...] = tuple(
            getattr(getattr(x_var, "aval", None), "shape", ())
        )
        original_rank = len(original_x_shape)
        if original_rank == 0:
            raise ValueError("GroupNorm requires tensor inputs")
        if channel_axis < 0:
            channel_axis += original_rank
        if channel_axis < 0 or channel_axis >= original_rank:
            raise ValueError("channel_axis out of range for GroupNorm")
        if batch_rank < 1 or batch_rank > channel_axis:
            raise ValueError("GroupNorm batch axes must be a leading prefix")

        def _original_label(idx: int) -> str | int | None:
            label: str | int | None = _dim_label_from_value_or_aval(
                original_x_val, original_x_shape, idx
            )
            return label

        original_dims = tuple(
            (
                _original_label(idx)
                if _original_label(idx) is not None
                else original_x_shape[idx]
            )
            for idx in range(original_rank)
        )

        restore_shape_val: ir.Value | None = None
        x_shape = original_x_shape
        rank = original_rank
        if batch_rank > 1:
            restore_shape_val = cast(
                ir.Value,
                builder.Shape(
                    original_x_val,
                    _outputs=[ctx.fresh_name("gn_original_shape")],
                ),
            )
            restore_shape_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(restore_shape_val, (original_rank,))
            _ensure_value_metadata(ctx, restore_shape_val)

            shape_axes = _const_i64(ctx, [0], name_hint="gn_batch_shape_axes")
            shape_steps = _const_i64(ctx, [1], name_hint="gn_batch_shape_steps")
            batch_shape = cast(
                ir.Value,
                builder.Slice(
                    restore_shape_val,
                    _const_i64(ctx, [0], name_hint="gn_batch_shape_start"),
                    _const_i64(ctx, [batch_rank], name_hint="gn_batch_shape_end"),
                    shape_axes,
                    shape_steps,
                    _outputs=[ctx.fresh_name("gn_batch_shape")],
                ),
            )
            batch_shape.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(batch_shape, (batch_rank,))
            _ensure_value_metadata(ctx, batch_shape)

            flattened_batch = builder_reduce_with_axes(
                ctx,
                batch_shape,
                op_type="ReduceProd",
                axes=(0,),
                keepdims=1,
                name_hint="gn_flattened_batch",
            )
            flattened_batch.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(flattened_batch, (1,))
            _ensure_value_metadata(ctx, flattened_batch)

            trailing_shape = cast(
                ir.Value,
                builder.Slice(
                    restore_shape_val,
                    _const_i64(ctx, [batch_rank], name_hint="gn_trailing_shape_start"),
                    _const_i64(ctx, [original_rank], name_hint="gn_trailing_shape_end"),
                    shape_axes,
                    shape_steps,
                    _outputs=[ctx.fresh_name("gn_trailing_shape")],
                ),
            )
            trailing_shape.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(trailing_shape, (original_rank - batch_rank,))
            _ensure_value_metadata(ctx, trailing_shape)

            collapsed_shape_val = cast(
                ir.Value,
                builder.Concat(
                    flattened_batch,
                    trailing_shape,
                    axis=0,
                    _outputs=[ctx.fresh_name("gn_collapsed_shape")],
                ),
            )
            collapsed_shape_val.type = ir.TensorType(ir.DataType.INT64)
            collapsed_rank = original_rank - batch_rank + 1
            _stamp_type_and_shape(collapsed_shape_val, (collapsed_rank,))
            _ensure_value_metadata(ctx, collapsed_shape_val)

            static_batch_dims = original_x_shape[:batch_rank]
            collapsed_batch_dim: int | None = None
            static_batch_values = [
                int(dim)
                for dim in static_batch_dims
                if isinstance(dim, (int, np.integer))
            ]
            if len(static_batch_values) == batch_rank:
                collapsed_batch_dim = int(np.prod(tuple(static_batch_values)))
            x_shape = (
                collapsed_batch_dim,
                *original_x_shape[batch_rank:],
            )
            x_val = cast(
                ir.Value,
                builder.Reshape(
                    original_x_val,
                    collapsed_shape_val,
                    allowzero=1,
                    _outputs=[ctx.fresh_name("gn_collapsed_batch")],
                ),
            )
            x_dtype = getattr(getattr(original_x_val, "type", None), "dtype", None)
            if x_dtype is not None:
                x_val.type = ir.TensorType(x_dtype)
            _stamp_type_and_shape(x_val, x_shape)
            _ensure_value_metadata(ctx, x_val)

            rank = collapsed_rank
            channel_axis -= batch_rank - 1

        channels = x_shape[channel_axis]
        if not isinstance(channels, (int, np.integer)):
            raise TypeError("GroupNorm requires a static channel dimension")
        channels_int = int(channels)
        if channels_int % num_groups != 0:
            raise ValueError("num_groups must divide the channel dimension")

        def _label(idx: int) -> str | int | None:
            label: str | int | None = _dim_label_from_value_or_aval(x_val, x_shape, idx)
            return label

        def _dims_for(indices: Sequence[int]) -> tuple[Any, ...]:
            dims: list[Any] = []
            for idx in indices:
                label = _label(idx)
                if label is not None:
                    dims.append(label)
                elif 0 <= idx < len(x_shape):
                    dims.append(x_shape[idx])
                else:
                    dims.append(None)
            return tuple(dims)

        layout_dims = _dims_for(range(rank))

        x_ir_dtype = getattr(getattr(x_val, "type", None), "dtype", None)

        def _stamp_x(value: ir.Value, dims: Sequence[Any]) -> ir.Value:
            if x_ir_dtype is not None:
                value.type = ir.TensorType(x_ir_dtype)
            _stamp_type_and_shape(value, tuple(dims))
            _ensure_value_metadata(ctx, value)
            return value

        def _lower_explicit_group_norm() -> ir.Value:
            shape_val = cast(
                ir.Value,
                builder.Shape(
                    x_val,
                    _outputs=[ctx.fresh_name("gn_input_shape")],
                ),
            )
            shape_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(shape_val, (rank,))
            _ensure_value_metadata(ctx, shape_val)

            slice_axes = _const_i64(ctx, [0], name_hint="gn_shape_axes")
            slice_steps = _const_i64(ctx, [1], name_hint="gn_shape_steps")
            prefix_dims = cast(
                ir.Value,
                builder.Slice(
                    shape_val,
                    _const_i64(ctx, [0], name_hint="gn_prefix_start"),
                    _const_i64(ctx, [channel_axis], name_hint="gn_prefix_end"),
                    slice_axes,
                    slice_steps,
                    _outputs=[ctx.fresh_name("gn_prefix_dims")],
                ),
            )
            prefix_dims.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(prefix_dims, (channel_axis,))
            _ensure_value_metadata(ctx, prefix_dims)

            group_dims = _const_i64(
                ctx,
                [num_groups, channels_int // num_groups],
                name_hint="gn_group_dims",
            )
            grouped_shape_parts = [prefix_dims, group_dims]
            if channel_axis + 1 < rank:
                suffix_dims = cast(
                    ir.Value,
                    builder.Slice(
                        shape_val,
                        _const_i64(
                            ctx,
                            [channel_axis + 1],
                            name_hint="gn_suffix_start",
                        ),
                        _const_i64(ctx, [rank], name_hint="gn_suffix_end"),
                        slice_axes,
                        slice_steps,
                        _outputs=[ctx.fresh_name("gn_suffix_dims")],
                    ),
                )
                suffix_dims.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(suffix_dims, (rank - channel_axis - 1,))
                _ensure_value_metadata(ctx, suffix_dims)
                grouped_shape_parts.append(suffix_dims)

            grouped_shape = cast(
                ir.Value,
                builder.Concat(
                    *grouped_shape_parts,
                    axis=0,
                    _outputs=[ctx.fresh_name("gn_grouped_shape")],
                ),
            )
            grouped_shape.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(grouped_shape, (rank + 1,))
            _ensure_value_metadata(ctx, grouped_shape)

            grouped_dims = (
                *layout_dims[:channel_axis],
                num_groups,
                channels_int // num_groups,
                *layout_dims[channel_axis + 1 :],
            )
            grouped = cast(
                ir.Value,
                builder.Reshape(
                    x_val,
                    grouped_shape,
                    _outputs=[ctx.fresh_name("gn_grouped")],
                ),
            )
            _stamp_x(grouped, grouped_dims)

            group_axis = channel_axis
            reduce_axes = tuple(
                idx for idx in range(rank + 1) if idx not in (0, group_axis)
            )
            reduced_dims = tuple(
                dim if idx in (0, group_axis) else 1
                for idx, dim in enumerate(grouped_dims)
            )
            mean = builder_reduce_with_axes(
                ctx,
                grouped,
                op_type="ReduceMean",
                axes=reduce_axes,
                keepdims=1,
                name_hint="gn_mean",
            )
            _stamp_x(mean, reduced_dims)

            centered = cast(
                ir.Value,
                builder.Sub(
                    grouped,
                    mean,
                    _outputs=[ctx.fresh_name("gn_centered")],
                ),
            )
            _stamp_x(centered, grouped_dims)

            variance_source = grouped if use_fast_variance else centered
            squared = cast(
                ir.Value,
                builder.Mul(
                    variance_source,
                    variance_source,
                    _outputs=[ctx.fresh_name("gn_squared")],
                ),
            )
            _stamp_x(squared, grouped_dims)

            variance = builder_reduce_with_axes(
                ctx,
                squared,
                op_type="ReduceMean",
                axes=reduce_axes,
                keepdims=1,
                name_hint=("gn_second_moment" if use_fast_variance else "gn_variance"),
            )
            _stamp_x(variance, reduced_dims)

            if use_fast_variance:
                mean_squared = cast(
                    ir.Value,
                    builder.Mul(
                        mean,
                        mean,
                        _outputs=[ctx.fresh_name("gn_mean_squared")],
                    ),
                )
                _stamp_x(mean_squared, reduced_dims)
                variance = cast(
                    ir.Value,
                    builder.Sub(
                        variance,
                        mean_squared,
                        _outputs=[ctx.fresh_name("gn_variance")],
                    ),
                )
                _stamp_x(variance, reduced_dims)

            x_np_dtype = np.dtype(
                getattr(getattr(x_var, "aval", None), "dtype", np.float32)
            )
            if clamp_negative_variance:
                zero_val = ctx.bind_const_for_var(
                    object(), np.asarray(0.0, dtype=x_np_dtype)
                )
                variance = cast(
                    ir.Value,
                    builder.Max(
                        variance,
                        zero_val,
                        _outputs=[ctx.fresh_name("gn_nonnegative_variance")],
                    ),
                )
                _stamp_x(variance, reduced_dims)

            eps_val = ctx.bind_const_for_var(
                object(), np.asarray(epsilon, dtype=x_np_dtype)
            )
            variance_eps = cast(
                ir.Value,
                builder.Add(
                    variance,
                    eps_val,
                    _outputs=[ctx.fresh_name("gn_variance_eps")],
                ),
            )
            _stamp_x(variance_eps, reduced_dims)

            stddev = cast(
                ir.Value,
                builder.Sqrt(
                    variance_eps,
                    _outputs=[ctx.fresh_name("gn_stddev")],
                ),
            )
            _stamp_x(stddev, reduced_dims)

            normalized_grouped = cast(
                ir.Value,
                builder.Div(
                    centered,
                    stddev,
                    _outputs=[ctx.fresh_name("gn_normalized_grouped")],
                ),
            )
            _stamp_x(normalized_grouped, grouped_dims)

            normalized = cast(
                ir.Value,
                builder.Reshape(
                    normalized_grouped,
                    shape_val,
                    _outputs=[ctx.fresh_name("gn_normalized")],
                ),
            )
            _stamp_x(normalized, layout_dims)

            affine_shape_dims = [1] * rank
            affine_shape_dims[channel_axis] = channels_int
            affine_shape = _const_i64(
                ctx,
                affine_shape_dims,
                name_hint="gn_affine_shape",
            )
            affine_dims = tuple(affine_shape_dims)
            scale_broadcast = cast(
                ir.Value,
                builder.Reshape(
                    scale_val,
                    affine_shape,
                    _outputs=[ctx.fresh_name("gn_scale")],
                ),
            )
            _stamp_x(scale_broadcast, affine_dims)
            bias_broadcast = cast(
                ir.Value,
                builder.Reshape(
                    bias_val,
                    affine_shape,
                    _outputs=[ctx.fresh_name("gn_bias")],
                ),
            )
            _stamp_x(bias_broadcast, affine_dims)

            scaled = cast(
                ir.Value,
                builder.Mul(
                    normalized,
                    scale_broadcast,
                    _outputs=[ctx.fresh_name("gn_scaled")],
                ),
            )
            _stamp_x(scaled, layout_dims)
            gn_out = cast(
                ir.Value,
                builder.Add(
                    scaled,
                    bias_broadcast,
                    _outputs=[ctx.fresh_name("GroupNorm")],
                ),
            )
            _stamp_x(gn_out, layout_dims)
            return gn_out

        final_val = _lower_explicit_group_norm()
        _stamp_type_and_shape(final_val, layout_dims)
        _ensure_value_metadata(ctx, final_val)

        if restore_shape_val is not None:
            restored_val = cast(
                ir.Value,
                builder.Reshape(
                    final_val,
                    restore_shape_val,
                    allowzero=1,
                    _outputs=[ctx.fresh_name("gn_restored_batch_shape")],
                ),
            )
            if x_ir_dtype is not None:
                restored_val.type = ir.TensorType(x_ir_dtype)
            _stamp_type_and_shape(restored_val, original_dims)
            _ensure_value_metadata(ctx, restored_val)
            final_val = restored_val

        ctx.bind_value_for_var(y_var, final_val)

    # ---------------- monkey patch & binding ----------------
    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("flax.nnx", "group_norm_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(nnx.GroupNorm, "__call__", cls._patch_call),
        ]

    @staticmethod
    def _prepare_param(
        vec: jax.Array | None, size: int, dtype: Any, *, default: float
    ) -> jax.Array:
        if vec is None:
            return jnp.full((size,), default, dtype=dtype)
        arr = jnp.asarray(vec, dtype=dtype)
        if arr.size != size:
            return jnp.full((size,), default, dtype=dtype)
        return jnp.reshape(arr, (size,))

    @classmethod
    def _patch_call(cls, orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(self: nnx.GroupNorm, x: Any, *, mask: Any | None = None) -> Any:
            if mask is not None:
                # Fall back to original implementation when masks are involved.
                return orig(self, x, mask=mask)
            if (
                getattr(self, "reduction_axes", None) is not None
                or getattr(self, "axis_name", None) is not None
            ):
                # Custom statistics axes and collectives need the framework's
                # original implementation; this primitive models the default
                # per-example GroupNorm contract.
                return orig(self, x, mask=mask)

            x_arr = jnp.asarray(x)

            feature_axis = getattr(self, "feature_axis", -1)
            if isinstance(feature_axis, Sequence):
                feature_axis = feature_axis[0]
            feature_axis = int(feature_axis)

            channels = x_arr.shape[feature_axis]
            if channels is None:
                raise ValueError("GroupNorm requires a known channel dimension")

            scale_param = (
                self.scale.value if getattr(self, "use_scale", False) else None
            )
            bias_param = self.bias.value if getattr(self, "use_bias", False) else None
            promote_dtype = getattr(self, "promote_dtype", None)
            if promote_dtype is not nnx_dtypes.promote_dtype:
                # Custom promotion hooks can change values as well as dtypes;
                # preserve their exact framework behavior via the original path.
                return orig(self, x, mask=mask)
            x_promoted, scale_promoted, bias_promoted = promote_dtype(
                (x_arr, scale_param, bias_param),
                dtype=getattr(self, "dtype", None),
            )
            result_dtype = x_promoted.dtype
            stats_dtype = jnp.promote_types(result_dtype, jnp.float32)
            x_stats = jnp.asarray(x_promoted, dtype=stats_dtype)

            scale_val = cls._prepare_param(
                scale_promoted,
                channels,
                stats_dtype,
                default=1.0,
            )
            bias_val = cls._prepare_param(
                bias_promoted,
                channels,
                stats_dtype,
                default=0.0,
            )

            use_fast_variance = bool(getattr(self, "use_fast_variance", True))
            out = cls._PRIM.bind(
                x_stats,
                scale_val,
                bias_val,
                epsilon=float(getattr(self, "epsilon", 1e-5)),
                num_groups=int(getattr(self, "num_groups", 1)),
                channel_axis=feature_axis,
                use_fast_variance=use_fast_variance,
                clamp_negative_variance=use_fast_variance,
                batch_rank=1,
            )
            return jnp.asarray(out, dtype=result_dtype)

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


@GroupNormPlugin._PRIM.def_impl
def _impl_group_norm(
    x: Any,
    scale: Any,
    bias: Any,
    *,
    epsilon: float,
    num_groups: int,
    channel_axis: int,
    use_fast_variance: bool,
    clamp_negative_variance: bool,
    batch_rank: int,
) -> Any:
    axis = int(channel_axis)
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        raise ValueError("channel_axis out of range for GroupNorm")

    channels = x.shape[axis]
    if channels is None:
        raise ValueError("GroupNorm requires statically known channel dimension")
    if channels % num_groups != 0:
        raise ValueError("num_groups must divide the channel dimension")

    x_last = jnp.moveaxis(x, axis, -1)
    group_size = channels // num_groups
    group_shape = x_last.shape[:-1] + (num_groups, group_size)
    x_grouped = jnp.reshape(x_last, group_shape)

    if batch_rank < 1 or batch_rank >= x.ndim:
        raise ValueError("GroupNorm batch_rank must precede the channel axis")
    reduce_axes = [
        i
        for i in range(x_grouped.ndim)
        if i not in (*range(batch_rank), x_grouped.ndim - 2)
    ]
    mean = jnp.mean(x_grouped, axis=reduce_axes, keepdims=True)
    if use_fast_variance:
        second_moment = jnp.mean(jnp.square(x_grouped), axis=reduce_axes, keepdims=True)
        var = second_moment - jnp.square(mean)
    else:
        var = jnp.mean(jnp.square(x_grouped - mean), axis=reduce_axes, keepdims=True)
    if clamp_negative_variance:
        var = jnp.maximum(jnp.asarray(0.0, dtype=var.dtype), var)

    normed = (x_grouped - mean) / jnp.sqrt(var + epsilon)
    normed = jnp.reshape(normed, x_last.shape)

    scale = jnp.asarray(scale, dtype=normed.dtype)
    bias = jnp.asarray(bias, dtype=normed.dtype)
    bshape = [1] * normed.ndim
    bshape[-1] = scale.shape[0]
    scale = jnp.reshape(scale, bshape)
    bias = jnp.reshape(bias, bshape)

    out = normed * scale + bias
    return jnp.moveaxis(out, -1, axis)
