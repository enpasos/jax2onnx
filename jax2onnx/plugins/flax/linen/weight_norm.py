# jax2onnx/plugins/flax/linen/weight_norm.py

from __future__ import annotations

from typing import Callable, ClassVar, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import linear as linen_linear
from jax.core import ShapedArray
from jax.extend.core import Primitive

from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins._patching import MonkeyPatchSpec


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


def _l2_normalize(x: jax.Array, *, axis: Sequence[int], eps: float) -> jax.Array:
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


def _maybe_broadcast(
    x: int | list[int] | tuple[int, ...] | None, rank: int
) -> tuple[int, ...]:
    if x is None:
        x = 1
    if isinstance(x, int):
        return (int(x),) * rank
    return tuple(int(v) for v in x)


def _call_dense(layer: nn.Dense, x, kernel, bias):
    from jax2onnx.plugins.flax.linen import dense as linen_dense

    use_bias = bool(getattr(layer, "use_bias", True))
    features = int(getattr(layer, "features", kernel.shape[-1]))
    if bias is None:
        bias = jnp.zeros((features,), dtype=x.dtype)
    if kernel.dtype != x.dtype:
        kernel = kernel.astype(x.dtype)
    dn = (((x.ndim - 1,), (0,)), ((), ()))
    return linen_dense.DensePlugin._PRIM.bind(
        x,
        kernel,
        bias,
        use_bias=use_bias,
        dimension_numbers=dn,
    )


def _call_conv(layer: nn.Conv, x, kernel, bias):
    from jax2onnx.plugins.flax.linen import conv as linen_conv

    kernel_size = (
        (layer.kernel_size,)
        if isinstance(layer.kernel_size, int)
        else tuple(layer.kernel_size)
    )
    if x.ndim < len(kernel_size) + 2:
        return None
    if not getattr(layer, "shared_weights", True):
        return None

    strides = _maybe_broadcast(getattr(layer, "strides", 1), len(kernel_size))
    input_dilation = _maybe_broadcast(
        getattr(layer, "input_dilation", 1), len(kernel_size)
    )
    kernel_dilation = _maybe_broadcast(
        getattr(layer, "kernel_dilation", 1), len(kernel_size)
    )
    if any(d != 1 for d in input_dilation):
        return None

    padding_lax = linen_linear.canonicalize_padding(
        getattr(layer, "padding", "SAME"),
        len(kernel_size),
    )
    if padding_lax in ("CIRCULAR", "REFLECT"):
        kernel_size_dilated = [
            (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
        ]
        zero_pad = [(0, 0)]
        pads = (
            zero_pad + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] + [(0, 0)]
        )
        padding_mode = {
            "CIRCULAR": "wrap",
            "REFLECT": "reflect",
        }[padding_lax]
        x = jnp.pad(x, pads, mode=padding_mode)
        padding_lax = "VALID"
    elif padding_lax == "CAUSAL":
        if len(kernel_size) != 1:
            return None
        left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
        x = jnp.pad(x, [(0, 0), (left_pad, 0), (0, 0)])
        padding_lax = "VALID"

    if getattr(layer, "mask", None) is not None:
        if layer.mask.shape != kernel.shape:
            raise ValueError(
                "Mask needs to have the same shape as weights. "
                f"Shapes are: {layer.mask.shape}, {kernel.shape}"
            )
        kernel = kernel * layer.mask

    x, kernel, bias = layer.promote_dtype(
        x,
        kernel,
        bias,
        dtype=getattr(layer, "dtype", None),
    )
    if bool(getattr(layer, "use_bias", True)):
        if bias is None:
            bias = jnp.zeros((int(layer.features),), dtype=x.dtype)
    else:
        bias = jnp.asarray(0, dtype=x.dtype)

    dimension_numbers = linen_linear._conv_dimension_numbers(x.shape)
    return linen_conv.ConvPlugin._PRIM.bind(
        x,
        kernel,
        bias,
        use_bias=bool(getattr(layer, "use_bias", True)),
        strides=strides,
        padding=padding_lax,
        dilations=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=int(getattr(layer, "feature_group_count", 1)),
    )


class _WeightNormDense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        return nn.WeightNorm(nn.Dense(self.features))(x)


@register_primitive(
    jaxpr_primitive="linen.weight_norm",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.WeightNorm",
    onnx=[
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="weight_norm",
    testcases=[
        {
            "testcase": "weight_norm_dense",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=_WeightNormDense,
                input_shape=(1, 8),
                features=4,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 4)],
            "run_only_f32_variant": True,
        },
    ],
)
class WeightNormPlugin(PrimitiveLeafPlugin):
    """Support flax.linen.WeightNorm by normalizing weights before dispatch."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.weight_norm")
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "WeightNorm primitive should not reach lowering; it is inlined."
        )

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.WeightNorm",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable):
        WeightNormPlugin._ORIGINAL_CALL = orig_fn

        def patched(self, *args, **kwargs):
            if kwargs:
                return orig_fn(self, *args, **kwargs)
            if len(args) != 1:
                return orig_fn(self, *args, **kwargs)

            layer = getattr(self, "layer_instance", None)
            if layer is None:
                return orig_fn(self, *args, **kwargs)

            inner_scope = getattr(layer, "scope", None)
            if inner_scope is None or not hasattr(inner_scope, "variables"):
                return orig_fn(self, *args, **kwargs)
            inner_vars = inner_scope.variables()
            inner_params = inner_vars.get("params", {})

            kernel = inner_params.get("kernel")
            bias = inner_params.get("bias")
            if kernel is None:
                return orig_fn(self, *args, **kwargs)

            variable_filter = getattr(self, "variable_filter", None)
            layer_name = getattr(layer, "name", None)
            if not layer_name:
                return orig_fn(self, *args, **kwargs)
            param_path = f"{layer_name}/kernel"
            if variable_filter is not None:
                try:
                    if not any(str(v) in param_path for v in variable_filter):
                        return orig_fn(self, *args, **kwargs)
                except Exception:
                    return orig_fn(self, *args, **kwargs)

            feature_axes = getattr(self, "feature_axes", -1)
            if feature_axes is None:
                feat_axes = ()
                reduction_axes = tuple(range(kernel.ndim))
            else:
                feat_axes = _canonicalize_axes(kernel.ndim, feature_axes)
                reduction_axes = tuple(
                    i for i in range(kernel.ndim) if i not in feat_axes
                )

            feature_shape = [1] * kernel.ndim
            reduced_feature_shape = []
            for ax in feat_axes:
                feature_shape[ax] = kernel.shape[ax]
                reduced_feature_shape.append(kernel.shape[ax])

            value_bar = _l2_normalize(
                kernel,
                axis=reduction_axes,
                eps=float(getattr(self, "epsilon", 1e-12)),
            )

            if bool(getattr(self, "use_scale", True)):
                scope = getattr(self, "scope", None)
                if scope is None or not hasattr(scope, "variables"):
                    return orig_fn(self, *args, **kwargs)
                wrapper_vars = scope.variables()
                wrapper_params = wrapper_vars.get("params", {})
                scale_name = f"{param_path}/scale"
                scale_param = wrapper_params.get(scale_name)
                if scale_param is None:
                    return orig_fn(self, *args, **kwargs)
                param_dtype = getattr(self, "param_dtype", None) or kernel.dtype
                scale = jnp.asarray(scale_param, dtype=param_dtype)
                if reduced_feature_shape:
                    if tuple(scale.shape) != tuple(reduced_feature_shape):
                        return orig_fn(self, *args, **kwargs)
                    scale = jnp.reshape(scale, tuple(reduced_feature_shape))
                else:
                    scale = jnp.reshape(scale, ())
                scale = jnp.reshape(scale, tuple(feature_shape))
                value_bar = value_bar * scale

            x = args[0]
            if isinstance(layer, nn.Dense):
                return _call_dense(layer, x, value_bar, bias)
            if isinstance(layer, nn.Conv):
                out = _call_conv(layer, x, value_bar, bias)
                if out is not None:
                    return out

            return orig_fn(self, *args, **kwargs)

        return patched
