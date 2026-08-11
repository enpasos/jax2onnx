# jax2onnx/plugins/equinox/eqx/nn/group_norm.py

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, Callable, ClassVar, Final

import equinox as eqx
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.flax.nnx import group_norm as nnx_group_norm
from jax2onnx.plugins.plugin_system import register_primitive


_GROUP_NORM_STATE_SENTINEL: Final = (
    inspect.signature(eqx.nn.GroupNorm.__call__).parameters["state"].default
)

_EQX_GROUP_NORM: Final[eqx.nn.GroupNorm] = eqx.nn.GroupNorm(
    groups=4,
    channels=8,
    eps=1e-5,
    channelwise_affine=True,
)
_EQX_GROUP_NORM_NO_AFFINE: Final[eqx.nn.GroupNorm] = eqx.nn.GroupNorm(
    groups=4,
    channels=None,
    eps=1e-5,
    channelwise_affine=False,
)
EXPECT_GROUP_NORM_FALLBACK: Final = nnx_group_norm.EXPECT_GROUP_NORM_EXPLICIT_SLOW


@register_primitive(
    jaxpr_primitive="eqx.nn.group_norm",
    jax_doc="https://docs.kidger.site/equinox/api/nn/normalisation/#equinox.nn.GroupNorm",
    onnx=nnx_group_norm.GROUP_NORM_EXPLICIT_COMPONENTS,
    since="0.12.2",
    context="primitives.eqx",
    component="group_norm",
    testcases=[
        {
            "testcase": "eqx_group_norm_rank3",
            "callable": _EQX_GROUP_NORM,
            "input_shapes": [(8, 4, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
        {
            "testcase": "eqx_group_norm_no_affine",
            "callable": _EQX_GROUP_NORM_NO_AFFINE,
            "input_shapes": [(8, 4, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
        {
            "testcase": "eqx_group_norm_opset18",
            "callable": _EQX_GROUP_NORM,
            "input_shapes": [(8, 4, 4)],
            "run_only_f32_variant": True,
            "opset_version": 18,
            "post_check_onnx_graph": EXPECT_GROUP_NORM_FALLBACK,
        },
    ],
)
class GroupNormPlugin(nnx_group_norm.GroupNormPlugin):
    """IR-only framework-faithful decomposition for ``equinox.nn.GroupNorm``."""

    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.group_norm")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("equinox.nn", "group_norm_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="equinox.nn.GroupNorm",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def _patch_call(cls, orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(
            self: eqx.nn.GroupNorm,
            x: Any,
            state: Any = _GROUP_NORM_STATE_SENTINEL,
            *,
            key: Any = None,
        ) -> Any:
            if state is not _GROUP_NORM_STATE_SENTINEL:
                return orig(self, x, state=state, key=key)

            x_arr = jnp.asarray(x)
            if x_arr.ndim < 1:
                return orig(self, x, state=state, key=key)

            channels = x_arr.shape[0]
            if channels is None or channels % int(self.groups) != 0:
                return orig(self, x, state=state, key=key)

            in_dtype = x_arr.dtype
            stats_dtype = jnp.promote_types(in_dtype, jnp.float32)
            x_stats = x_arr.astype(stats_dtype)
            if bool(self.channelwise_affine):
                weight = jnp.asarray(self.weight, dtype=stats_dtype)
                bias = jnp.asarray(self.bias, dtype=stats_dtype)
            else:
                weight = jnp.ones((channels,), dtype=stats_dtype)
                bias = jnp.zeros((channels,), dtype=stats_dtype)

            if tuple(weight.shape) != (channels,):
                weight = jnp.reshape(weight, (channels,))
            if tuple(bias.shape) != (channels,):
                bias = jnp.reshape(bias, (channels,))

            x_nchw = jnp.expand_dims(x_stats, axis=0)
            y_nchw = cls._PRIM.bind(
                x_nchw,
                weight,
                bias,
                epsilon=float(self.eps),
                num_groups=int(self.groups),
                channel_axis=1,
                use_fast_variance=False,
                clamp_negative_variance=True,
                batch_rank=1,
            )
            return jnp.squeeze(y_nchw, axis=0).astype(in_dtype)

        return wrapped


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


def _group_norm_batch_rule(
    args: Sequence[Any],
    dims: Sequence[Any],
    **params: Any,
) -> Any:
    x, scale, bias = args
    x_bdim, scale_bdim, bias_bdim = dims
    if scale_bdim is not None or bias_bdim is not None:
        raise NotImplementedError(
            "vmapped Equinox GroupNorm requires unmapped scale and bias"
        )

    if x_bdim is None:
        return GroupNormPlugin._PRIM.bind(x, scale, bias, **params), None

    logical_rank = x.ndim - 1
    channel_axis = int(params["channel_axis"])
    if channel_axis < 0:
        channel_axis += logical_rank
    if channel_axis < 0 or channel_axis >= logical_rank:
        raise ValueError("channel_axis out of range for vmapped GroupNorm")
    physical_channel_axis = channel_axis + int(x_bdim <= channel_axis)
    x_front = jnp.moveaxis(x, x_bdim, 0)
    moved_channel_axis = physical_channel_axis
    if physical_channel_axis < x_bdim:
        moved_channel_axis += 1
    if moved_channel_axis <= 1:
        raise ValueError("vmapped GroupNorm requires a distinct channel axis")

    bound_params = dict(params)
    bound_params["channel_axis"] = moved_channel_axis
    bound_params["batch_rank"] = int(params.get("batch_rank", 1)) + 1
    out = GroupNormPlugin._PRIM.bind(x_front, scale, bias, **bound_params)
    return out, 0


batching.primitive_batchers[GroupNormPlugin._PRIM] = _group_norm_batch_rule
