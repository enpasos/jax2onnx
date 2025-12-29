# jax2onnx/plugins/flax/linen/batch_norm.py

from __future__ import annotations

from typing import Callable, ClassVar, Final
import logging
import numpy as np
from flax import linen as nn
from jax.extend.core import Primitive

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.flax.nnx import batch_norm as nnx_batch_norm
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)

EXPECT_BN_ONLY: Final = nnx_batch_norm.EXPECT_BN_ONLY
EXPECT_T_BN_T: Final = nnx_batch_norm.EXPECT_T_BN_T


@register_primitive(
    jaxpr_primitive="linen.batch_norm",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.BatchNorm",
    onnx=[
        {
            "component": "BatchNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__BatchNormalization.html",
        }
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="batch_norm",
    testcases=[
        {
            "testcase": "batch_norm_no_bias_no_scale",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.BatchNorm,
                input_shape=(1, 8),
                use_running_average=True,
                use_bias=False,
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_BN_ONLY,
        },
        {
            "testcase": "batch_norm_bias_no_scale",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.BatchNorm,
                input_shape=(1, 8),
                use_running_average=True,
                use_bias=True,
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_BN_ONLY,
        },
        {
            "testcase": "batch_norm_no_bias_scale",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.BatchNorm,
                input_shape=(1, 8),
                use_running_average=True,
                use_bias=False,
                use_scale=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_BN_ONLY,
        },
        {
            "testcase": "batch_norm_bias_scale",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.BatchNorm,
                input_shape=(1, 8),
                use_running_average=True,
                use_bias=True,
                use_scale=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_BN_ONLY,
        },
        {
            "testcase": "batch_norm_3d",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.BatchNorm,
                input_shape=(1, 4, 3),
                use_running_average=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 3)],
            "expected_output_shapes": [("B", 4, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_T_BN_T,
        },
        {
            "testcase": "batch_norm_4d",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.BatchNorm,
                input_shape=(1, 4, 4, 3),
                use_running_average=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 4, 3)],
            "expected_output_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_T_BN_T,
        },
        {
            "testcase": "batch_norm_4d_no_bias_no_scale",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.BatchNorm,
                input_shape=(1, 4, 4, 3),
                use_running_average=True,
                use_bias=False,
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 4, 3)],
            "expected_output_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_T_BN_T,
        },
    ],
)
class BatchNormPlugin(nnx_batch_norm.BatchNormPlugin):
    """IR-only plugin for flax.linen.BatchNorm (inference behavior)."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.batch_norm")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def _batch_norm(
        x,
        scale,
        bias,
        mean,
        var,
        *,
        epsilon,
        momentum,
        scale_is_default=False,
        bias_is_default=False,
    ):
        return BatchNormPlugin._PRIM.bind(
            x,
            scale,
            bias,
            mean,
            var,
            epsilon=epsilon,
            momentum=momentum,
            scale_is_default=scale_is_default,
            bias_is_default=bias_is_default,
        )

    @staticmethod
    def _make_patch(orig_fn: Callable):
        BatchNormPlugin._ORIGINAL_CALL = orig_fn
        prim = BatchNormPlugin._PRIM

        def patched(self, x, use_running_average=None, *, mask=None):
            try:
                use_running_average = nn.merge_param(
                    "use_running_average",
                    getattr(self, "use_running_average", None),
                    use_running_average,
                )
            except ValueError:
                return orig_fn(
                    self, x, use_running_average=use_running_average, mask=mask
                )

            if not use_running_average:
                logging.warning(
                    "BatchNorm exported with use_running_average=False; converting to inference mode."
                )

            axis = getattr(self, "axis", -1)
            if not isinstance(axis, int):
                return orig_fn(
                    self, x, use_running_average=use_running_average, mask=mask
                )
            if axis < 0:
                axis += x.ndim
            if axis != x.ndim - 1:
                return orig_fn(
                    self, x, use_running_average=use_running_average, mask=mask
                )

            scope = getattr(self, "scope", None)
            if scope is None or not hasattr(scope, "variables"):
                return orig_fn(
                    self, x, use_running_average=use_running_average, mask=mask
                )

            variables = scope.variables()
            params = variables.get("params", {})
            batch_stats = variables.get("batch_stats", {})

            mean = batch_stats.get("mean")
            var = batch_stats.get("var")
            if mean is None or var is None:
                return orig_fn(
                    self, x, use_running_average=use_running_average, mask=mask
                )

            scale = params.get("scale") if getattr(self, "use_scale", True) else None
            bias = params.get("bias") if getattr(self, "use_bias", True) else None

            num_features = None
            for arr in (mean, var, scale, bias):
                if arr is None:
                    continue
                shape = getattr(arr, "shape", ())
                if len(shape) == 1 and isinstance(shape[0], (int, np.integer)):
                    num_features = int(shape[0])
                    break
            if num_features is None:
                maybe_dim = x.shape[axis]
                if isinstance(maybe_dim, (int, np.integer)):
                    num_features = int(maybe_dim)
            if num_features is None:
                num_features = 1

            param_dtype = self.param_dtype if self.param_dtype is not None else x.dtype
            np_dtype = np.dtype(param_dtype)

            if getattr(self, "use_scale", True):
                if scale is None:
                    return orig_fn(
                        self, x, use_running_average=use_running_average, mask=mask
                    )
                scale_is_default = False
            else:
                scale = np.ones((num_features,), dtype=np_dtype)
                scale_is_default = True

            if getattr(self, "use_bias", True):
                if bias is None:
                    return orig_fn(
                        self, x, use_running_average=use_running_average, mask=mask
                    )
                bias_is_default = False
            else:
                bias = np.zeros((num_features,), dtype=np_dtype)
                bias_is_default = True

            return prim.bind(
                x,
                scale,
                bias,
                mean,
                var,
                epsilon=self.epsilon,
                momentum=self.momentum,
                scale_is_default=scale_is_default,
                bias_is_default=bias_is_default,
            )

        return patched

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("flax.linen", "batch_norm_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.linen.BatchNorm",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]


@BatchNormPlugin._PRIM.def_impl
def _impl(
    x,
    scale,
    bias,
    mean,
    var,
    *,
    epsilon,
    momentum,
    scale_is_default=False,
    bias_is_default=False,
):
    return nnx_batch_norm._impl(
        x,
        scale,
        bias,
        mean,
        var,
        epsilon=epsilon,
        momentum=momentum,
        scale_is_default=scale_is_default,
        bias_is_default=bias_is_default,
    )
