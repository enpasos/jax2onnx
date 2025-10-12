# jax2onnx/plugins/flax/nnx/group_norm.py

from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar, Final, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax.extend.core import Primitive

import onnx_ir as ir
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

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore

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


def _require_builder(ctx: Any):
    builder = getattr(ctx, "builder", None)
    if builder is None:
        raise AttributeError("IR build context missing builder")
    return builder


@register_primitive(
    jaxpr_primitive=GROUP_NORM_PRIM.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.GroupNorm",
    onnx=[
        {
            "component": "GroupNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__GroupNormalization.html",
        }
    ],
    since="v0.2.0",
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
    def abstract_eval(x, scale, bias, *, epsilon, num_groups, channel_axis):
        del scale, bias, epsilon, num_groups, channel_axis
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------------- lowering ----------------
    def lower(self, ctx: "IRBuildContext", eqn):
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

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        if rank == 0:
            raise ValueError("GroupNorm requires tensor inputs")
        if channel_axis < 0:
            channel_axis += rank

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

        def _label(idx: int):
            return _dim_label_from_value_or_aval(x_val, x_shape, idx)

        def _dims_for(indices: Sequence[int]):
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
            gn_input = builder.Transpose(
                x_val,
                perm=tuple(int(p) for p in perm),
                _outputs=[ctx.fresh_name("gn_nchw_in")],
            )
            if x_ir_dtype is not None:
                gn_input.type = ir.TensorType(x_ir_dtype)
            _stamp_type_and_shape(gn_input, nchw_dims)
            _ensure_value_metadata(ctx, gn_input)

        gn_out = builder.GroupNormalization(
            gn_input,
            scale_val,
            bias_val,
            epsilon=float(epsilon),
            num_groups=int(num_groups),
            _outputs=[ctx.fresh_name("GroupNorm")],
        )
        if x_ir_dtype is not None:
            gn_out.type = ir.TensorType(x_ir_dtype)
        _stamp_type_and_shape(gn_out, nchw_dims if need_layout_convert else nhwc_dims)
        _ensure_value_metadata(ctx, gn_out)

        if need_layout_convert:
            final_val = builder.Transpose(
                gn_out,
                perm=tuple(int(p) for p in inv_perm),
                _outputs=[ctx.fresh_name("gn_out")],
            )
            if x_ir_dtype is not None:
                final_val.type = ir.TensorType(x_ir_dtype)
            _stamp_type_and_shape(final_val, nhwc_dims)
            _ensure_value_metadata(ctx, final_val)
        else:
            final_val = gn_out
            _stamp_type_and_shape(final_val, nhwc_dims)
            _ensure_value_metadata(ctx, final_val)

        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(y_var, final_val)

    # ---------------- monkey patch & binding ----------------
    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("flax.nnx", "group_norm_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(nnx.GroupNorm, "__call__", cls._patch_call),
        ]

    @staticmethod
    def _prepare_param(vec: jax.Array | None, size: int, dtype, *, default: float):
        if vec is None:
            return jnp.full((size,), default, dtype=dtype)
        arr = jnp.asarray(vec, dtype=dtype)
        if arr.size != size:
            return jnp.full((size,), default, dtype=dtype)
        return jnp.reshape(arr, (size,))

    @classmethod
    def _patch_call(cls, orig):
        def wrapped(self: nnx.GroupNorm, x, *, mask=None):
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
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


@GroupNormPlugin._PRIM.def_impl
def _impl_group_norm(
    x, scale, bias, *, epsilon: float, num_groups: int, channel_axis: int
):
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
