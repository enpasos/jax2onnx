# jax2onnx/plugins/flax/linen/rms_norm.py

from __future__ import annotations

from typing import Callable, ClassVar, Final, Sequence

import jax.numpy as jnp
from flax import linen as nn
from jax.extend.core import Primitive

from jax2onnx.plugins.flax.nnx import rms_norm as nnx_rms_norm
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins._patching import MonkeyPatchSpec

EXPECT_RMS_NORM_GRAPH: Final = nnx_rms_norm.EXPECT_RMS_NORM_GRAPH


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
    jaxpr_primitive="linen.rms_norm",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.RMSNorm",
    onnx=[
        {
            "component": "RMSNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__RMSNormalization.html",
        }
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="rms_norm",
    testcases=[
        {
            "testcase": "rms_norm_basic",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.RMSNorm,
                input_shape=(1, 6),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 6)],
            "expected_output_shapes": [(2, 6)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_RMS_NORM_GRAPH,
        },
        {
            "testcase": "rms_norm_use_scale_false",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.RMSNorm,
                input_shape=(1, 6),
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 6)],
            "expected_output_shapes": [(2, 6)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_RMS_NORM_GRAPH,
        },
        {
            "testcase": "rms_norm_4d_dynamic",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.RMSNorm,
                input_shape=(1, 4, 4, 3),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 4, 3)],
            "expected_output_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_RMS_NORM_GRAPH,
        },
    ],
)
class RMSNormPlugin(nnx_rms_norm.RMSNormPlugin):
    """IR-only plugin for flax.linen.RMSNorm → ONNX RMSNormalization."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.rms_norm")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.RMSNorm",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable):
        RMSNormPlugin._ORIGINAL_CALL = orig_fn
        prim = RMSNormPlugin._PRIM

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

            reduction_axes = getattr(self, "reduction_axes", -1)
            feature_axes = getattr(self, "feature_axes", -1)
            red_axes = _canonicalize_axes(x.ndim, reduction_axes)
            feat_axes = _canonicalize_axes(x.ndim, feature_axes)
            if tuple(sorted(red_axes)) != tuple(sorted(feat_axes)):
                return orig_fn(self, x, mask=mask)

            axis0 = min(red_axes)
            expected_axes = tuple(range(axis0, x.ndim))
            if tuple(sorted(red_axes)) != expected_axes:
                return orig_fn(self, x, mask=mask)

            tail_shape = tuple(x.shape[a] for a in red_axes)
            param_dtype = getattr(self, "param_dtype", None) or x.dtype
            if x.dtype != param_dtype:
                x = x.astype(param_dtype)

            if getattr(self, "use_scale", True):
                scale_param = params.get("scale")
                if scale_param is None:
                    return orig_fn(self, x, mask=mask)
                base_scale = (
                    jnp.reshape(scale_param, tail_shape)
                    if tuple(scale_param.shape) != tail_shape
                    else scale_param
                )
                base_scale = jnp.asarray(base_scale, dtype=param_dtype)
            else:
                base_scale = jnp.ones(tail_shape, dtype=param_dtype)

            eps = float(getattr(self, "epsilon", 1e-5))
            needs_flatten = (axis0 != x.ndim - 1) or (len(red_axes) > 1)
            if needs_flatten:
                orig_shape = x.shape
                x_flat = jnp.reshape(x, (*orig_shape[:axis0], -1))
                scale_vec = jnp.reshape(base_scale, (-1,))
                y_flat = prim.bind(
                    x_flat,
                    scale_vec,
                    epsilon=eps,
                    axis=x_flat.ndim - 1,
                )
                return jnp.reshape(y_flat, orig_shape)

            return prim.bind(
                x,
                base_scale,
                epsilon=eps,
                axis=axis0,
            )

        return patched


@RMSNormPlugin._PRIM.def_impl
def _impl_rms_norm(x, scale, *, epsilon: float, axis: int):
    return nnx_rms_norm._impl_rms_norm(
        x,
        scale,
        epsilon=epsilon,
        axis=axis,
    )
