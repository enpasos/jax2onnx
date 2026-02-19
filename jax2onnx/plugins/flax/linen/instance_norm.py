# jax2onnx/plugins/flax/linen/instance_norm.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, Sequence

import jax.numpy as jnp
import onnx_ir as ir
from flax import linen as nn
from jax.extend.core import Primitive

from jax2onnx.plugins._ir_shapes import (
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
    _stamp_type_and_shape,
)
from jax2onnx.plugins.flax.nnx import group_norm as nnx_group_norm
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins._utils import cast_param_like

EXPECT_INSTANCE_NORM_PLAIN: Final = nnx_group_norm.EG(
    [
        (
            "InstanceNormalization",
            {
                "counts": {
                    "InstanceNormalization": 1,
                    "Transpose": 0,
                }
            },
        )
    ]
)


EXPECT_INSTANCE_NORM_TRANSPOSED: Final = nnx_group_norm.EG(
    [
        (
            "Transpose -> InstanceNormalization -> Transpose",
            {
                "counts": {
                    "InstanceNormalization": 1,
                    "Transpose": 2,
                }
            },
        )
    ]
)


def _canonicalize_axes(ndim: int, axes: Sequence[int] | int) -> tuple[int, ...]:
    if isinstance(axes, int):
        axes = (axes,)
    out = []
    for axis in axes:
        axis = int(axis)
        if axis < 0:
            axis += ndim
        out.append(axis)
    return tuple(out)


@register_primitive(
    jaxpr_primitive="linen.instance_norm",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.InstanceNorm",
    onnx=[
        {
            "component": "InstanceNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html",
        },
        {
            "component": "GroupNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__GroupNormalization.html",
        },
    ],
    since="0.11.0",
    context="primitives.linen",
    component="instance_norm",
    testcases=[
        {
            "testcase": "instance_norm_rank4",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.InstanceNorm,
                input_shape=(1, 4, 4, 3),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 4, 3)],
            "expected_output_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_INSTANCE_NORM_TRANSPOSED,
        },
        {
            "testcase": "instance_norm_rank2",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.InstanceNorm,
                input_shape=(1, 8),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": nnx_group_norm.EXPECT_GROUP_NORM_PLAIN,
        },
    ],
)
class InstanceNormPlugin(nnx_group_norm.GroupNormPlugin):
    """IR-only plugin for flax.linen.InstanceNorm via GroupNormalization."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.instance_norm")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None

    def lower(self, ctx, eqn):
        x_var, scale_var, bias_var = eqn.invars[:3]
        y_var = eqn.outvars[0]

        params = dict(getattr(eqn, "params", {}) or {})
        epsilon = float(params.get("epsilon", 1e-5))
        num_groups = int(params.get("num_groups", 1))
        channel_axis = int(params.get("channel_axis", -1))

        builder = nnx_group_norm._require_builder(ctx)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        scale_val = ctx.get_value_for_var(scale_var, name_hint=ctx.fresh_name("scale"))
        bias_val = ctx.get_value_for_var(bias_var, name_hint=ctx.fresh_name("bias"))

        scale_val = cast_param_like(ctx, scale_val, x_val, name_hint="in_scale_cast")
        bias_val = cast_param_like(ctx, bias_val, x_val, name_hint="in_bias_cast")

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        if rank == 0:
            raise ValueError("InstanceNorm requires tensor inputs")
        if rank < 3:
            # ORT requires InstanceNormalization input rank >= 3.
            super().lower(ctx, eqn)
            return
        if channel_axis < 0:
            channel_axis += rank
        channels = x_shape[channel_axis] if 0 <= channel_axis < rank else None
        if not isinstance(channels, int) or num_groups != channels:
            # Fallback for cases that are not true instance-normalization.
            super().lower(ctx, eqn)
            return

        need_layout_convert = rank > 2 and channel_axis != 1
        if need_layout_convert:
            perm = [0]
            if channel_axis != 0:
                perm.append(channel_axis)
            perm.extend(i for i in range(1, rank) if i != channel_axis)
            if len(perm) != rank:
                raise ValueError(
                    f"Invalid permutation derived for InstanceNorm: {perm}"
                )
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

        inst_input = x_val
        if need_layout_convert:
            inst_input = builder.Transpose(
                x_val,
                perm=tuple(int(p) for p in perm),
                _outputs=[ctx.fresh_name("in_nchw_in")],
            )
            if x_ir_dtype is not None:
                inst_input.type = ir.TensorType(x_ir_dtype)
            _stamp_type_and_shape(inst_input, nchw_dims)
            _ensure_value_metadata(ctx, inst_input)

        inst_out = builder.InstanceNormalization(
            inst_input,
            scale_val,
            bias_val,
            epsilon=float(epsilon),
            _outputs=[ctx.fresh_name("InstanceNorm")],
        )
        if x_ir_dtype is not None:
            inst_out.type = ir.TensorType(x_ir_dtype)
        _stamp_type_and_shape(
            inst_out,
            nchw_dims if need_layout_convert else nhwc_dims,
        )
        _ensure_value_metadata(ctx, inst_out)

        if need_layout_convert:
            final_val = builder.Transpose(
                inst_out,
                perm=tuple(int(p) for p in inv_perm),
                _outputs=[ctx.fresh_name("in_out")],
            )
            if x_ir_dtype is not None:
                final_val.type = ir.TensorType(x_ir_dtype)
            _stamp_type_and_shape(final_val, nhwc_dims)
            _ensure_value_metadata(ctx, final_val)
        else:
            final_val = inst_out
            _stamp_type_and_shape(final_val, nhwc_dims)
            _ensure_value_metadata(ctx, final_val)

        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(y_var, final_val)

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.InstanceNorm",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable):
        InstanceNormPlugin._ORIGINAL_CALL = orig_fn
        prim = InstanceNormPlugin._PRIM

        def patched(self, x, *, mask=None):
            if mask is not None:
                return orig_fn(self, x, mask=mask)
            if getattr(self, "axis_name", None) is not None:
                return orig_fn(self, x, mask=mask)
            if getattr(self, "axis_index_groups", None) is not None:
                return orig_fn(self, x, mask=mask)
            if not getattr(self, "use_fast_variance", True):
                return orig_fn(self, x, mask=mask)

            scope = getattr(self, "scope", None)
            if scope is None or not hasattr(scope, "variables"):
                return orig_fn(self, x, mask=mask)
            variables = scope.variables()
            params = variables.get("params", {})

            feature_axes = getattr(self, "feature_axes", -1)
            feat_axes = _canonicalize_axes(x.ndim, feature_axes)
            if len(feat_axes) != 1:
                return orig_fn(self, x, mask=mask)
            feature_axis = feat_axes[0]
            if feature_axis == 0:
                return orig_fn(self, x, mask=mask)

            channels = x.shape[feature_axis]
            if channels is None:
                return orig_fn(self, x, mask=mask)

            num_groups = int(channels)
            if num_groups <= 0:
                return orig_fn(self, x, mask=mask)

            param_dtype = getattr(self, "param_dtype", None) or x.dtype
            if x.dtype != param_dtype:
                x = x.astype(param_dtype)

            if getattr(self, "use_scale", True):
                scale_param = params.get("scale")
                if scale_param is None:
                    return orig_fn(self, x, mask=mask)
                scale = jnp.asarray(scale_param, dtype=param_dtype)
            else:
                scale = jnp.ones((channels,), dtype=param_dtype)

            if getattr(self, "use_bias", True):
                bias_param = params.get("bias")
                if bias_param is None:
                    return orig_fn(self, x, mask=mask)
                bias = jnp.asarray(bias_param, dtype=param_dtype)
            else:
                bias = jnp.zeros((channels,), dtype=param_dtype)

            if tuple(scale.shape) != (channels,):
                scale = jnp.reshape(scale, (channels,))
            if tuple(bias.shape) != (channels,):
                bias = jnp.reshape(bias, (channels,))

            return prim.bind(
                x,
                scale,
                bias,
                epsilon=float(getattr(self, "epsilon", 1e-5)),
                num_groups=num_groups,
                channel_axis=feature_axis,
            )

        return patched
