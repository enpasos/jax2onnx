# jax2onnx/plugins/flax/nnx/group_norm.py

from __future__ import annotations
from typing import Any, Callable, ClassVar, Final, Sequence, cast

import jax
import jax.numpy as jnp
from flax import nnx
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

GROUP_NORM_PRIM: Final[Primitive] = Primitive("nnx.group_norm")
GROUP_NORM_PRIM.multiple_results = False


EXPECT_GROUP_NORM_PLAIN: Final = EG(
    [
        (
            "GroupNormalization",
            {
                "counts": {
                    "GroupNormalization": 1,
                    "Transpose": 0,
                }
            },
        )
    ]
)


EXPECT_GROUP_NORM_TRANSPOSED: Final = EG(
    [
        (
            "Transpose -> GroupNormalization -> Transpose",
            {
                "counts": {
                    "GroupNormalization": 1,
                    "Transpose": 2,
                }
            },
        )
    ]
)


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


def _require_builder(ctx: LoweringContextProtocol) -> IRBuilderProtocol:
    return ctx.builder


@register_primitive(
    jaxpr_primitive=GROUP_NORM_PRIM.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.GroupNorm",
    onnx=[
        {
            "component": "GroupNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__GroupNormalization.html",
        }
    ],
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
            "post_check_onnx_graph": EXPECT_GROUP_NORM_TRANSPOSED,
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
            "post_check_onnx_graph": EXPECT_GROUP_NORM_PLAIN,
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
            "post_check_onnx_graph": EXPECT_GROUP_NORM_TRANSPOSED,
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
            "post_check_onnx_graph": EXPECT_GROUP_NORM_TRANSPOSED,
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
            "post_check_onnx_graph": EXPECT_GROUP_NORM_TRANSPOSED,
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
            "post_check_onnx_graph": EXPECT_GROUP_NORM_TRANSPOSED,
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
            "post_check_onnx_graph": EXPECT_GROUP_NORM_TRANSPOSED,
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
            "post_check_onnx_graph": EXPECT_GROUP_NORM_TRANSPOSED,
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
            "post_check_onnx_graph": EXPECT_GROUP_NORM_TRANSPOSED,
        },
    ],
)
class GroupNormPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.nnx.GroupNorm → ONNX GroupNormalization."""

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
    ) -> ShapedArray:
        del scale, bias, epsilon, num_groups, channel_axis
        return ShapedArray(x.shape, x.dtype)

    # ---------------- lowering ----------------
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var, scale_var, bias_var = eqn.invars[:3]
        y_var = eqn.outvars[0]

        params = dict(getattr(eqn, "params", {}) or {})
        epsilon = float(params.get("epsilon", 1e-5))
        num_groups = int(params.get("num_groups", 1))
        channel_axis = int(params.get("channel_axis", -1))

        builder = _require_builder(ctx)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        scale_val = ctx.get_value_for_var(scale_var, name_hint=ctx.fresh_name("scale"))
        bias_val = ctx.get_value_for_var(bias_var, name_hint=ctx.fresh_name("bias"))

        scale_val = cast_param_like(ctx, scale_val, x_val, name_hint="gn_scale_cast")
        bias_val = cast_param_like(ctx, bias_val, x_val, name_hint="gn_bias_cast")

        x_shape: tuple[object, ...] = tuple(
            getattr(getattr(x_var, "aval", None), "shape", ())
        )
        rank = len(x_shape)
        if rank == 0:
            raise ValueError("GroupNorm requires tensor inputs")
        if channel_axis < 0:
            channel_axis += rank
        if channel_axis < 0 or channel_axis >= rank:
            raise ValueError("channel_axis out of range for GroupNorm")

        channels = x_shape[channel_axis]
        if not isinstance(channels, (int, np.integer)):
            raise TypeError("GroupNorm requires a static channel dimension")
        channels_int = int(channels)
        if channels_int % num_groups != 0:
            raise ValueError("num_groups must divide the channel dimension")

        # Prepare permutation to make channel axis = 1 (NCHW-like) when needed
        need_layout_convert = rank > 2 and channel_axis != 1
        if need_layout_convert:
            perm = [0]
            if channel_axis != 0:
                perm.append(channel_axis)
            perm.extend(i for i in range(1, rank) if i != channel_axis)
            if len(perm) != rank:
                raise ValueError(f"Invalid permutation derived for GroupNorm: {perm}")
            inv_perm = [perm.index(i) for i in range(rank)]
        else:
            perm = list(range(rank))
            inv_perm = perm

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

        nchw_dims = _dims_for(perm)
        nhwc_dims = _dims_for(range(rank))

        x_ir_dtype = getattr(getattr(x_val, "type", None), "dtype", None)

        gn_input = x_val
        if need_layout_convert:
            gn_input = cast(
                ir.Value,
                builder.Transpose(
                    x_val,
                    perm=tuple(int(p) for p in perm),
                    _outputs=[ctx.fresh_name("gn_nchw_in")],
                ),
            )
            if x_ir_dtype is not None:
                gn_input.type = ir.TensorType(x_ir_dtype)
            _stamp_type_and_shape(gn_input, nchw_dims)
            _ensure_value_metadata(ctx, gn_input)

        normalized_dims = nchw_dims if need_layout_convert else nhwc_dims

        def _stamp_x(value: ir.Value, dims: Sequence[Any]) -> ir.Value:
            if x_ir_dtype is not None:
                value.type = ir.TensorType(x_ir_dtype)
            _stamp_type_and_shape(value, tuple(dims))
            _ensure_value_metadata(ctx, value)
            return value

        opset = int(getattr(builder, "opset", 21))
        if opset >= 21:
            gn_out = cast(
                ir.Value,
                builder.GroupNormalization(
                    gn_input,
                    scale_val,
                    bias_val,
                    epsilon=float(epsilon),
                    num_groups=int(num_groups),
                    _outputs=[ctx.fresh_name("GroupNorm")],
                ),
            )
            _stamp_x(gn_out, normalized_dims)
        else:
            shape_val = cast(
                ir.Value,
                builder.Shape(
                    gn_input,
                    _outputs=[ctx.fresh_name("gn_input_shape")],
                ),
            )
            shape_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(shape_val, (rank,))
            _ensure_value_metadata(ctx, shape_val)

            slice_start = _const_i64(ctx, [0], name_hint="gn_shape_start")
            slice_batch_end = _const_i64(ctx, [1], name_hint="gn_batch_end")
            slice_axes = _const_i64(ctx, [0], name_hint="gn_shape_axes")
            slice_steps = _const_i64(ctx, [1], name_hint="gn_shape_steps")
            batch_dim = cast(
                ir.Value,
                builder.Slice(
                    shape_val,
                    slice_start,
                    slice_batch_end,
                    slice_axes,
                    slice_steps,
                    _outputs=[ctx.fresh_name("gn_batch_dim")],
                ),
            )
            batch_dim.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(batch_dim, (1,))
            _ensure_value_metadata(ctx, batch_dim)

            group_dims = _const_i64(
                ctx,
                [num_groups, channels_int // num_groups],
                name_hint="gn_group_dims",
            )
            grouped_shape_parts = [batch_dim, group_dims]
            if rank > 2:
                spatial_start = _const_i64(ctx, [2], name_hint="gn_spatial_start")
                spatial_end = _const_i64(ctx, [rank], name_hint="gn_spatial_end")
                spatial_dims = cast(
                    ir.Value,
                    builder.Slice(
                        shape_val,
                        spatial_start,
                        spatial_end,
                        slice_axes,
                        slice_steps,
                        _outputs=[ctx.fresh_name("gn_spatial_dims")],
                    ),
                )
                spatial_dims.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(spatial_dims, (rank - 2,))
                _ensure_value_metadata(ctx, spatial_dims)
                grouped_shape_parts.append(spatial_dims)

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
                normalized_dims[0],
                num_groups,
                channels_int // num_groups,
                *normalized_dims[2:],
            )
            grouped = cast(
                ir.Value,
                builder.Reshape(
                    gn_input,
                    grouped_shape,
                    _outputs=[ctx.fresh_name("gn_grouped")],
                ),
            )
            _stamp_x(grouped, grouped_dims)

            reduce_axes = _const_i64(
                ctx,
                tuple(range(2, rank + 1)),
                name_hint="gn_reduce_axes",
            )
            reduced_dims = (*grouped_dims[:2], *((1,) * (rank - 1)))
            mean = cast(
                ir.Value,
                builder.ReduceMean(
                    grouped,
                    reduce_axes,
                    keepdims=1,
                    _outputs=[ctx.fresh_name("gn_mean")],
                ),
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

            squared = cast(
                ir.Value,
                builder.Mul(
                    centered,
                    centered,
                    _outputs=[ctx.fresh_name("gn_squared")],
                ),
            )
            _stamp_x(squared, grouped_dims)

            variance = cast(
                ir.Value,
                builder.ReduceMean(
                    squared,
                    reduce_axes,
                    keepdims=1,
                    _outputs=[ctx.fresh_name("gn_variance")],
                ),
            )
            _stamp_x(variance, reduced_dims)

            x_np_dtype = np.dtype(
                getattr(getattr(x_var, "aval", None), "dtype", np.float32)
            )
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
            _stamp_x(normalized, normalized_dims)

            affine_shape = _const_i64(
                ctx,
                [1, channels_int, *([1] * (rank - 2))],
                name_hint="gn_affine_shape",
            )
            affine_dims = (1, channels_int, *([1] * (rank - 2)))
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
            _stamp_x(scaled, normalized_dims)
            gn_out = cast(
                ir.Value,
                builder.Add(
                    scaled,
                    bias_broadcast,
                    _outputs=[ctx.fresh_name("GroupNorm")],
                ),
            )
            _stamp_x(gn_out, normalized_dims)

        if need_layout_convert:
            final_val = cast(
                ir.Value,
                builder.Transpose(
                    gn_out,
                    perm=tuple(int(p) for p in inv_perm),
                    _outputs=[ctx.fresh_name("gn_out")],
                ),
            )
            if x_ir_dtype is not None:
                final_val.type = ir.TensorType(x_ir_dtype)
            _stamp_type_and_shape(final_val, nhwc_dims)
            _ensure_value_metadata(ctx, final_val)
        else:
            final_val = gn_out
            _stamp_type_and_shape(final_val, nhwc_dims)
            _ensure_value_metadata(ctx, final_val)

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

            param_dtype = getattr(self, "param_dtype", None) or x.dtype
            if x.dtype != param_dtype:
                x = x.astype(param_dtype)

            feature_axis = getattr(self, "feature_axis", -1)
            if isinstance(feature_axis, Sequence):
                feature_axis = feature_axis[0]
            feature_axis = int(feature_axis)

            channels = x.shape[feature_axis]
            if channels is None:
                raise ValueError("GroupNorm requires a known channel dimension")

            scale_val = cls._prepare_param(
                self.scale.value if getattr(self, "use_scale", False) else None,
                channels,
                param_dtype,
                default=1.0,
            )
            bias_val = cls._prepare_param(
                self.bias.value if getattr(self, "use_bias", False) else None,
                channels,
                param_dtype,
                default=0.0,
            )

            return cls._PRIM.bind(
                x,
                scale_val,
                bias_val,
                epsilon=float(getattr(self, "epsilon", 1e-5)),
                num_groups=int(getattr(self, "num_groups", 1)),
                channel_axis=feature_axis,
            )

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


@GroupNormPlugin._PRIM.def_impl
def _impl_group_norm(
    x: Any, scale: Any, bias: Any, *, epsilon: float, num_groups: int, channel_axis: int
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

    reduce_axes = [i for i in range(x_grouped.ndim) if i not in (0, x_grouped.ndim - 2)]
    mean = jnp.mean(x_grouped, axis=reduce_axes, keepdims=True)
    var = jnp.var(x_grouped, axis=reduce_axes, keepdims=True)

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
