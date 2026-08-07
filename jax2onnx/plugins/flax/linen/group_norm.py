# jax2onnx/plugins/flax/linen/group_norm.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, Sequence

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import dtypes as linen_dtypes

from jax2onnx._compat.jax import Primitive
from jax2onnx.plugins.flax.nnx import group_norm as nnx_group_norm
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec

EXPECT_GROUP_NORM_FALLBACK: Final = nnx_group_norm.EXPECT_GROUP_NORM_FALLBACK
LINEN_NORM_ONNX_COMPONENTS: Final = [
    *nnx_group_norm.GROUP_NORM_ONNX_COMPONENTS,
    {"component": "Expand", "doc": "https://onnx.ai/onnx/operators/onnx__Expand.html"},
    {
        "component": "ReduceSum",
        "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
    },
    {
        "component": "ReduceSumSquare",
        "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html",
    },
]


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


def _stage_linen_norm_operands(
    x: Any,
    scale_param: Any | None,
    bias_param: Any | None,
    *,
    channels: int,
    dtype: Any | None,
    use_scale: bool,
    use_bias: bool,
    force_float32_reductions: bool,
) -> tuple[Any, Any, Any, Any] | None:
    """Match Linen's split statistics, affine, and result dtype staging."""

    x_arr = jnp.asarray(x)
    stats_base_dtype = dtype if dtype is not None else jnp.result_type(x_arr)
    stats_dtype = (
        jnp.promote_types(stats_base_dtype, jnp.float32)
        if force_float32_reductions
        else jnp.dtype(stats_base_dtype)
    )
    if not jnp.issubdtype(stats_dtype, jnp.floating):
        return None

    result_args = [x_arr]
    affine_params: list[Any] = []
    if use_scale:
        if scale_param is None:
            return None
        raw_scale = jnp.asarray(scale_param)
        result_args.append(raw_scale)
        affine_params.append(raw_scale)
    else:
        raw_scale = None
    if use_bias:
        if bias_param is None:
            return None
        raw_bias = jnp.asarray(bias_param)
        result_args.append(raw_bias)
        affine_params.append(raw_bias)
    else:
        raw_bias = None

    result_dtype = linen_dtypes.canonicalize_dtype(*result_args, dtype=dtype)
    raw_x_dtype = jnp.result_type(x_arr)
    raw_x_fits_stats = (
        jnp.issubdtype(raw_x_dtype, jnp.floating)
        and jnp.promote_types(stats_dtype, raw_x_dtype) == stats_dtype
    ) or jnp.issubdtype(raw_x_dtype, jnp.integer)
    if not raw_x_fits_stats:
        # Linen explicitly casts integer inputs before computing statistics.
        # Preserve that contract; for floating inputs, only use the shared
        # compute operand when the statistics dtype does not narrow x.
        return None

    operands_fit_stats = all(
        jnp.promote_types(stats_dtype, param.dtype) == stats_dtype
        for param in affine_params
    )
    result_fits_stats = jnp.promote_types(stats_dtype, result_dtype) == stats_dtype
    if not operands_fit_stats or not result_fits_stats:
        # The fused primitive uses one compute dtype. Preserve Linen's separate
        # lower-precision statistics / wider affine contract by rejecting
        # configurations whose stages cannot be represented losslessly.
        return None

    x_stats = jnp.asarray(x_arr, dtype=stats_dtype)
    scale = (
        jnp.asarray(raw_scale, dtype=stats_dtype)
        if raw_scale is not None
        else jnp.ones((channels,), dtype=stats_dtype)
    )
    bias = (
        jnp.asarray(raw_bias, dtype=stats_dtype)
        if raw_bias is not None
        else jnp.zeros((channels,), dtype=stats_dtype)
    )
    return x_stats, scale, bias, result_dtype


def _can_use_original_linen_slow_path(
    x: Any,
    scale_param: Any | None,
    bias_param: Any | None,
    *,
    use_scale: bool,
    use_bias: bool,
    use_fast_variance: bool,
    force_float32_reductions: bool,
) -> bool:
    """Whether Linen's original slow path has a checker-valid dtype contract."""

    if use_fast_variance or not force_float32_reductions:
        return False
    if jnp.result_type(jnp.asarray(x)) != jnp.dtype(jnp.float32):
        return False

    active_params: list[Any] = []
    if use_scale:
        if scale_param is None:
            return False
        active_params.append(scale_param)
    if use_bias:
        if bias_param is None:
            return False
        active_params.append(bias_param)

    supported_dtypes = {jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)}
    return all(jnp.asarray(param).dtype in supported_dtypes for param in active_params)


@register_primitive(
    jaxpr_primitive="linen.group_norm",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.GroupNorm",
    onnx=LINEN_NORM_ONNX_COMPONENTS,
    since="0.11.0",
    context="primitives.linen",
    component="group_norm",
    testcases=[
        {
            "testcase": "group_norm_rank4",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.GroupNorm,
                input_shape=(1, 7, 7, 64),
                num_groups=8,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 7, 7, 64)],
            "expected_output_shapes": [(3, 7, 7, 64)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
        {
            "testcase": "group_norm_rank2",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.GroupNorm,
                input_shape=(1, 8),
                num_groups=4,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
        {
            "testcase": "group_norm_no_bias_no_scale",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.GroupNorm,
                input_shape=(1, 16, 16, 32),
                num_groups=8,
                use_bias=False,
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 16, 16, 32)],
            "expected_output_shapes": [("B", 16, 16, 32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
    ],
)
class GroupNormPlugin(nnx_group_norm.GroupNormPlugin):
    """IR-only framework-faithful decomposition for flax.linen.GroupNorm."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.group_norm")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False
    _ORIGINAL_CALL: ClassVar[Callable[..., Any] | None] = None

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            MonkeyPatchSpec(
                target="flax.linen.GroupNorm",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable[..., Any] | None) -> Callable[..., Any]:
        GroupNormPlugin._ORIGINAL_CALL = orig_fn
        prim = GroupNormPlugin._PRIM

        def call_orig(self: Any, x: Any, *, mask: Any | None = None) -> Any:
            if orig_fn is None:
                raise RuntimeError("flax.linen.GroupNorm.__call__ is not available.")
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

            reduction_axes = getattr(self, "reduction_axes", None)
            if reduction_axes is None:
                reduction_axes = list(range(1, x.ndim - 1)) + [-1]
            reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
            expected_axes = tuple(range(1, x.ndim - 1)) + (x.ndim - 1,)
            if tuple(reduction_axes) != expected_axes:
                return call_orig(self, x, mask=mask)

            channels = x.shape[-1]
            if channels is None:
                return call_orig(self, x, mask=mask)

            num_groups = None
            group_size = getattr(self, "group_size", None)
            configured_groups = getattr(self, "num_groups", None)
            if (group_size is None and configured_groups is None) or (
                group_size is not None and configured_groups is not None
            ):
                return call_orig(self, x, mask=mask)
            if group_size is not None:
                if channels % group_size != 0:
                    return call_orig(self, x, mask=mask)
                num_groups = channels // group_size
            else:
                if configured_groups is None:
                    return call_orig(self, x, mask=mask)
                try:
                    num_groups = int(configured_groups)
                except Exception:
                    return call_orig(self, x, mask=mask)
                if num_groups <= 0 or channels % num_groups != 0:
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
                    "Linen GroupNorm export cannot preserve this configuration's "
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
                num_groups=int(num_groups),
                channel_axis=-1,
                use_fast_variance=use_fast_variance,
                clamp_negative_variance=use_fast_variance,
                batch_rank=1,
            )
            return jnp.asarray(out, dtype=result_dtype)

        return patched


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
    return nnx_group_norm._impl_group_norm(
        x,
        scale,
        bias,
        epsilon=epsilon,
        num_groups=num_groups,
        channel_axis=channel_axis,
        use_fast_variance=use_fast_variance,
        clamp_negative_variance=clamp_negative_variance,
        batch_rank=batch_rank,
    )
