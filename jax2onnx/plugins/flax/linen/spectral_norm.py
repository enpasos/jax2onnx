# jax2onnx/plugins/flax/linen/spectral_norm.py

from __future__ import annotations

from typing import Callable, ClassVar

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


def _l2_normalize(x, *, eps: float) -> jax.Array:
    return x * jax.lax.rsqrt((x * x).sum(axis=None, keepdims=True) + eps)


def _spectral_normalize(
    value: jax.Array,
    u_init: jax.Array,
    *,
    n_steps: int,
    eps: float,
    error_on_non_matrix: bool,
) -> jax.Array:
    value = jnp.asarray(value)
    value_shape = value.shape
    if value.ndim <= 1 or n_steps < 1:
        return value
    if value.ndim > 2:
        if error_on_non_matrix:
            raise ValueError(
                f"SpectralNorm received {value.ndim}D weight with error_on_non_matrix=True"
            )
        value = jnp.reshape(value, (-1, value.shape[-1]))

    u0 = jnp.asarray(u_init, dtype=value.dtype)
    if u0.ndim != 2 or u0.shape[-1] != value.shape[-1]:
        raise ValueError("SpectralNorm u-vector shape does not match weight")

    for _ in range(int(n_steps)):
        v0 = _l2_normalize(jnp.matmul(u0, value.transpose([1, 0])), eps=eps)
        u0 = _l2_normalize(jnp.matmul(v0, value), eps=eps)

    v0 = jax.lax.stop_gradient(v0)
    u0 = jax.lax.stop_gradient(u0)
    sigma = jnp.matmul(jnp.matmul(v0, value), jnp.transpose(u0))[0, 0]
    value = value / jnp.where(sigma != 0, sigma, 1)
    return value.reshape(value_shape)


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


class _SpectralNormDense(nn.Module):
    features: int
    n_steps: int = 1

    @nn.compact
    def __call__(self, x, *, update_stats: bool = False):
        return nn.SpectralNorm(
            nn.Dense(self.features),
            n_steps=self.n_steps,
        )(x, update_stats=update_stats)


@register_primitive(
    jaxpr_primitive="linen.spectral_norm",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.SpectralNorm",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="spectral_norm",
    testcases=[
        {
            "testcase": "spectral_norm_dense",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=_SpectralNormDense,
                input_shape=(1, 8),
                features=4,
                n_steps=1,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "input_params": {"update_stats": False},
            "expected_output_shapes": [("B", 4)],
            "run_only_f32_variant": True,
        },
    ],
)
class SpectralNormPlugin(PrimitiveLeafPlugin):
    """Support flax.linen.SpectralNorm by normalizing weights before dispatch."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.spectral_norm")
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "SpectralNorm primitive should not reach lowering; it is inlined."
        )

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.SpectralNorm",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _make_patch(orig_fn: Callable):
        SpectralNormPlugin._ORIGINAL_CALL = orig_fn

        def patched(self, *args, update_stats: bool, **kwargs):
            if update_stats:
                raise NotImplementedError(
                    "SpectralNorm export only supports update_stats=False."
                )
            if kwargs:
                return orig_fn(self, *args, update_stats=update_stats, **kwargs)
            if len(args) != 1:
                return orig_fn(self, *args, update_stats=update_stats, **kwargs)

            scope = getattr(self, "scope", None)
            if scope is None or not hasattr(scope, "variables"):
                return orig_fn(self, *args, update_stats=update_stats, **kwargs)
            variables = scope.variables()

            stats = variables.get(getattr(self, "collection_name", "batch_stats"), {})

            layer = getattr(self, "layer_instance", None)
            if layer is None:
                return orig_fn(self, *args, update_stats=update_stats, **kwargs)

            inner_scope = getattr(layer, "scope", None)
            if inner_scope is None or not hasattr(inner_scope, "variables"):
                return orig_fn(self, *args, update_stats=update_stats, **kwargs)
            inner_vars = inner_scope.variables()
            params = inner_vars.get("params", {})

            kernel = params.get("kernel")
            bias = params.get("bias")
            if kernel is None:
                return orig_fn(self, *args, update_stats=update_stats, **kwargs)

            layer_name = getattr(layer, "name", None)
            if not layer_name:
                return orig_fn(self, *args, update_stats=update_stats, **kwargs)

            u_name = f"{layer_name}/kernel/u"
            u_val = stats.get(u_name)
            if u_val is None:
                return orig_fn(self, *args, update_stats=update_stats, **kwargs)

            try:
                kernel = _spectral_normalize(
                    kernel,
                    u_val,
                    n_steps=int(getattr(self, "n_steps", 1)),
                    eps=float(getattr(self, "epsilon", 1e-12)),
                    error_on_non_matrix=bool(
                        getattr(self, "error_on_non_matrix", False)
                    ),
                )
            except Exception:
                return orig_fn(self, *args, update_stats=update_stats, **kwargs)

            x = args[0]

            if isinstance(layer, nn.Dense):
                return _call_dense(layer, x, kernel, bias)
            if isinstance(layer, nn.Conv):
                out = _call_conv(layer, x, kernel, bias)
                if out is not None:
                    return out

            return orig_fn(self, *args, update_stats=update_stats, **kwargs)

        return patched
