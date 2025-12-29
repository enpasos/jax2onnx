# jax2onnx/plugins/flax/linen/instance_norm.py

from __future__ import annotations

from typing import Callable, ClassVar, Final, Sequence

import jax.numpy as jnp
from flax import linen as nn
from jax.extend.core import Primitive

from jax2onnx.plugins.flax.nnx import group_norm as nnx_group_norm
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins._patching import MonkeyPatchSpec

EXPECT_GROUP_NORM_PLAIN: Final = nnx_group_norm.EXPECT_GROUP_NORM_PLAIN
EXPECT_GROUP_NORM_TRANSPOSED: Final = nnx_group_norm.EXPECT_GROUP_NORM_TRANSPOSED


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
            "component": "GroupNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__GroupNormalization.html",
        }
    ],
    since="v0.11.0",
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
            "post_check_onnx_graph": EXPECT_GROUP_NORM_TRANSPOSED,
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
            "post_check_onnx_graph": EXPECT_GROUP_NORM_PLAIN,
        },
    ],
)
class InstanceNormPlugin(nnx_group_norm.GroupNormPlugin):
    """IR-only plugin for flax.linen.InstanceNorm via GroupNormalization."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.instance_norm")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None

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
