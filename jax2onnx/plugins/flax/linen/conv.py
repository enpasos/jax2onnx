# jax2onnx/plugins/flax/linen/conv.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, Sequence

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import linear as linen_linear
from jax.extend.core import Primitive

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.flax.nnx import conv as nnx_conv
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)

EXPECT_TCT: Final = nnx_conv.EXPECT_TCT


@register_primitive(
    jaxpr_primitive="linen.conv",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.Conv",
    onnx=[
        {"component": "Conv", "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html"},
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "CastLike",
            "doc": "https://onnx.ai/onnx/operators/onnx__CastLike.html",
        },
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="conv",
    testcases=[
        {
            "testcase": "conv_basic",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.Conv,
                input_shape=(1, 28, 28, 3),
                features=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "run_only_f32_variant": True,
            "input_shapes": [("B", 28, 28, 3)],
            "expected_output_shapes": [("B", 28, 28, 16)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_no_bias",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.Conv,
                input_shape=(1, 10, 10, 3),
                features=8,
                kernel_size=(3, 3),
                padding="VALID",
                use_bias=False,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "run_only_f32_variant": True,
            "input_shapes": [(1, 10, 10, 3)],
            "expected_output_shapes": [(1, 8, 8, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_stride",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.Conv,
                input_shape=(1, 9, 9, 3),
                features=4,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "run_only_f32_variant": True,
            "input_shapes": [(1, 9, 9, 3)],
            "expected_output_shapes": [(1, 5, 5, 4)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
    ],
)
class ConvPlugin(nnx_conv.ConvPlugin):
    """IR-only plugin for flax.linen.Conv → ONNX Conv."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.conv")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def _make_patch(orig_fn: Callable):
        real_orig = getattr(orig_fn, "__j2o_linen_conv_original__", orig_fn)
        ConvPlugin._ORIGINAL_CALL = real_orig
        prim = ConvPlugin._PRIM

        def _maybe_broadcast(
            x: int | Sequence[int] | None, rank: int
        ) -> tuple[int, ...]:
            if x is None:
                x = 1
            if isinstance(x, int):
                return (int(x),) * rank
            return tuple(int(v) for v in x)

        def patched(self, inputs):
            kernel_size = (
                (self.kernel_size,)
                if isinstance(self.kernel_size, int)
                else tuple(self.kernel_size)
            )
            if inputs.ndim < len(kernel_size) + 2:
                return real_orig(self, inputs)
            if not getattr(self, "shared_weights", True):
                raise NotImplementedError(
                    "linen.Conv with shared_weights=False is not supported."
                )

            strides = _maybe_broadcast(getattr(self, "strides", 1), len(kernel_size))
            input_dilation = _maybe_broadcast(
                getattr(self, "input_dilation", 1), len(kernel_size)
            )
            kernel_dilation = _maybe_broadcast(
                getattr(self, "kernel_dilation", 1), len(kernel_size)
            )
            if any(d != 1 for d in input_dilation):
                return real_orig(self, inputs)

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
            print(f"DEBUG: ConvPlugin patched called. self={self}")
            if scope is not None and hasattr(scope, "variables"):
                variables = scope.variables()
                print(f"DEBUG: Scope found. Variables keys: {variables.keys()}")
                params = variables.get("params", {})
                kernel = params.get("kernel")
                bias = params.get("bias") if self.use_bias else None
                print(
                    f"DEBUG: kernel={kernel is not None}, bias={bias is not None}, use_bias={self.use_bias}"
                )

                if kernel is not None:
                    feature_group_count = int(getattr(self, "feature_group_count", 1))
                    if getattr(self, "mask", None) is not None:
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
                    if bool(getattr(self, "use_bias", True)):
                        if bias is None:
                            bias = jnp.zeros(
                                (int(self.features),),
                                dtype=inputs.dtype,
                            )
                    else:
                        bias = jnp.asarray(0, dtype=inputs.dtype)

                    dimension_numbers = linen_linear._conv_dimension_numbers(
                        inputs.shape
                    )
                    return prim.bind(
                        inputs,
                        kernel,
                        bias,
                        use_bias=bool(getattr(self, "use_bias", True)),
                        strides=strides,
                        padding=padding_lax,
                        dilations=kernel_dilation,
                        dimension_numbers=dimension_numbers,
                        feature_group_count=feature_group_count,
                    )

            return real_orig(self, inputs)

        patched_any: Any = patched
        setattr(patched_any, "__j2o_linen_conv_shim__", True)
        setattr(patched_any, "__j2o_linen_conv_original__", real_orig)
        return patched

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("flax.linen", "conv_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.linen.Conv",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def patch_info():
        def _wrapper(orig):
            return ConvPlugin._make_patch(orig)

        return {
            "patch_targets": [nn.Conv],
            "patch_function": _wrapper,
            "target_attribute": "__call__",
            "extra_assignments": [("flax.linen", "conv_p", ConvPlugin._PRIM)],
        }


@ConvPlugin._PRIM.def_impl
def _impl(
    x,
    kernel,
    bias,
    *,
    use_bias,
    strides,
    padding,
    dilations,
    dimension_numbers,
    feature_group_count,
):
    return nnx_conv._impl(
        x,
        kernel,
        bias,
        use_bias=use_bias,
        strides=strides,
        padding=padding,
        dilations=dilations,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
    )
