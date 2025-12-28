# jax2onnx/plugins/flax/linen/conv_local.py

from __future__ import annotations

from typing import Callable, ClassVar, Sequence
import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from flax import linen as nn
from flax.linen import linear as linen_linear

from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)


def _maybe_broadcast(x: int | Sequence[int] | None, rank: int) -> tuple[int, ...]:
    if x is None:
        x = 1
    if isinstance(x, int):
        return (int(x),) * rank
    return tuple(int(v) for v in x)


@register_primitive(
    jaxpr_primitive="linen.conv_local",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.ConvLocal",
    onnx=[
        {"component": "Conv", "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html"},
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="conv_local",
    testcases=[
        {
            "testcase": "conv_local_valid",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.ConvLocal,
                input_shape=(1, 8, 8, 3),
                features=4,
                kernel_size=(3, 3),
                padding="VALID",
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 8, 8, 3)],
            "expected_output_shapes": [(1, 6, 6, 4)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "conv_local_same",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.ConvLocal,
                input_shape=(1, 8, 8, 3),
                features=4,
                kernel_size=(3, 3),
                padding="SAME",
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 8, 8, 3)],
            "expected_output_shapes": [(1, 8, 8, 4)],
            "run_only_f32_variant": True,
        },
    ],
)
class ConvLocalPlugin(PrimitiveLeafPlugin):
    """Patch flax.linen.ConvLocal to use conv_general_dilated_local during tracing."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.conv_local")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def _make_patch(orig_fn: Callable):
        ConvLocalPlugin._ORIGINAL_CALL = orig_fn

        def patched(self, inputs):
            kernel_size = (
                (self.kernel_size,)
                if isinstance(self.kernel_size, int)
                else tuple(self.kernel_size)
            )
            if inputs.ndim < len(kernel_size) + 2:
                return orig_fn(self, inputs)

            num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
            if num_batch_dimensions != 1:
                input_batch_shape = inputs.shape[:num_batch_dimensions]
                flat_input_shape = (-1,) + inputs.shape[num_batch_dimensions:]
                inputs = jnp.reshape(inputs, flat_input_shape)
            else:
                input_batch_shape = None

            if getattr(self, "feature_group_count", 1) != 1:
                raise NotImplementedError(
                    "linen.ConvLocal does not support feature_group_count != 1."
                )

            strides = _maybe_broadcast(getattr(self, "strides", 1), len(kernel_size))
            input_dilation = _maybe_broadcast(
                getattr(self, "input_dilation", 1), len(kernel_size)
            )
            kernel_dilation = _maybe_broadcast(
                getattr(self, "kernel_dilation", 1), len(kernel_size)
            )
            padding_lax = linen_linear.canonicalize_padding(
                getattr(self, "padding", "SAME"),
                len(kernel_size),
            )
            if padding_lax in ("CIRCULAR", "REFLECT"):
                kernel_size_dilated = [
                    (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
                ]
                zero_pad = [(0, 0)]
                pads = (
                    zero_pad
                    + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
                    + [(0, 0)]
                )
                padding_mode = {
                    "CIRCULAR": "wrap",
                    "REFLECT": "reflect",
                }[padding_lax]
                inputs = jnp.pad(inputs, pads, mode=padding_mode)
                padding_lax = "VALID"
            elif padding_lax == "CAUSAL":
                if len(kernel_size) != 1:
                    raise ValueError(
                        "Causal padding is only implemented for 1D convolutions."
                    )
                left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
                inputs = jnp.pad(inputs, [(0, 0), (left_pad, 0), (0, 0)])
                padding_lax = "VALID"

            scope = getattr(self, "scope", None)
            if scope is None or not hasattr(scope, "variables"):
                return orig_fn(self, inputs)

            variables = scope.variables()
            params = variables.get("params", {})
            kernel = params.get("kernel")
            bias = params.get("bias") if self.use_bias else None
            if kernel is None:
                return orig_fn(self, inputs)

            if self.mask is not None:
                if self.mask.shape != kernel.shape:
                    raise ValueError(
                        "Mask needs to have the same shape as weights. "
                        f"Shapes are: {self.mask.shape}, {kernel.shape}"
                    )
                kernel = kernel * self.mask

            inputs, kernel, bias = self.promote_dtype(
                inputs,
                kernel,
                bias,
                dtype=getattr(self, "dtype", None),
            )

            dimension_numbers = linen_linear._conv_dimension_numbers(inputs.shape)
            y = jax.lax.conv_general_dilated_local(
                lhs=inputs,
                rhs=kernel,
                window_strides=strides,
                padding=padding_lax,
                filter_shape=kernel_size,
                lhs_dilation=input_dilation,
                rhs_dilation=kernel_dilation,
                dimension_numbers=dimension_numbers,
                precision=getattr(self, "precision", None),
            )

            if self.use_bias:
                if bias is None:
                    return orig_fn(self, inputs)
                bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
                y = y + bias

            if input_batch_shape is not None:
                y = jnp.reshape(y, input_batch_shape + y.shape[1:])
            return y

        return patched

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.ConvLocal",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            )
        ]
