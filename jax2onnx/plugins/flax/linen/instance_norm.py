# jax2onnx/plugins/flax/linen/instance_norm.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Sequence

import jax.numpy as jnp
from flax import linen as nn

from jax2onnx._compat.jax import Primitive
from jax2onnx.plugins.flax.linen.group_norm import (
    LINEN_NORM_ONNX_COMPONENTS,
    _can_use_original_linen_slow_path,
    _stage_linen_norm_operands,
)
from jax2onnx.plugins.flax.nnx import group_norm as nnx_group_norm
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec


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
    onnx=LINEN_NORM_ONNX_COMPONENTS,
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
            "post_check_onnx_graph": nnx_group_norm.EXPECT_GROUP_NORM_FALLBACK,
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
            "post_check_onnx_graph": nnx_group_norm.EXPECT_GROUP_NORM_FALLBACK,
        },
    ],
)
class InstanceNormPlugin(nnx_group_norm.GroupNormPlugin):
    """IR-only framework-faithful decomposition for flax.linen.InstanceNorm."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.instance_norm")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False
    _ORIGINAL_CALL: ClassVar[Callable[..., Any] | None] = None

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            MonkeyPatchSpec(
                target="flax.linen.InstanceNorm",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable[..., Any] | None) -> Callable[..., Any]:
        InstanceNormPlugin._ORIGINAL_CALL = orig_fn
        prim = InstanceNormPlugin._PRIM

        def call_orig(self: Any, x: Any, *, mask: Any | None = None) -> Any:
            if orig_fn is None:
                raise RuntimeError("flax.linen.InstanceNorm.__call__ is not available.")
            return orig_fn(self, x, mask=mask)

        def patched(self: Any, x: Any, *, mask: Any | None = None) -> Any:
            if mask is not None:
                return call_orig(self, x, mask=mask)
            if getattr(self, "axis_name", None) is not None:
                return call_orig(self, x, mask=mask)
            if getattr(self, "axis_index_groups", None) is not None:
                return call_orig(self, x, mask=mask)
            use_fast_variance = bool(getattr(self, "use_fast_variance", True))
            force_float32_reductions = bool(
                getattr(self, "force_float32_reductions", True)
            )
            scope = getattr(self, "scope", None)
            if scope is None or not hasattr(scope, "variables"):
                return call_orig(self, x, mask=mask)
            variables = scope.variables()
            params = variables.get("params", {})

            feature_axes = getattr(self, "feature_axes", -1)
            feat_axes = _canonicalize_axes(x.ndim, feature_axes)
            if len(feat_axes) != 1:
                return call_orig(self, x, mask=mask)
            feature_axis = feat_axes[0]
            if feature_axis == 0:
                return call_orig(self, x, mask=mask)

            channels = x.shape[feature_axis]
            if channels is None:
                return call_orig(self, x, mask=mask)

            num_groups = int(channels)
            if num_groups <= 0:
                return call_orig(self, x, mask=mask)

            use_scale = bool(getattr(self, "use_scale", True))
            use_bias = bool(getattr(self, "use_bias", True))
            scale_param = params.get("scale") if use_scale else None
            bias_param = params.get("bias") if use_bias else None
            if _can_use_original_linen_slow_path(
                x,
                scale_param,
                bias_param,
                use_scale=use_scale,
                use_bias=use_bias,
                use_fast_variance=use_fast_variance,
                force_float32_reductions=force_float32_reductions,
            ):
                # Preserve Linen's centered-variance implementation whenever
                # its separate statistics/affine dtypes form a valid graph.
                return call_orig(self, x, mask=mask)

            staged = _stage_linen_norm_operands(
                x,
                scale_param,
                bias_param,
                channels=channels,
                dtype=getattr(self, "dtype", None),
                use_scale=use_scale,
                use_bias=use_bias,
                force_float32_reductions=force_float32_reductions,
            )
            if staged is None:
                raise NotImplementedError(
                    "Linen InstanceNorm export cannot preserve this configuration's "
                    "separate statistics, input, affine, and result dtype staging."
                )
            x_stats, scale, bias, result_dtype = staged

            if tuple(scale.shape) != (channels,):
                scale = jnp.reshape(scale, (channels,))
            if tuple(bias.shape) != (channels,):
                bias = jnp.reshape(bias, (channels,))

            out = prim.bind(
                x_stats,
                scale,
                bias,
                epsilon=float(getattr(self, "epsilon", 1e-5)),
                num_groups=num_groups,
                channel_axis=feature_axis,
                use_fast_variance=use_fast_variance,
                clamp_negative_variance=use_fast_variance,
                batch_rank=1,
            )
            return jnp.asarray(out, dtype=result_dtype)

        return patched
