# jax2onnx/plugins/flax/linen/conv_transpose.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Sequence
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from flax import linen as nn
from flax.linen import linear as linen_linear

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)


def _conv_transpose_padding(k: int, s: int, padding: str) -> tuple[int, int]:
    if padding == "SAME":
        pad_len = k + s - 2
        if s > k - 1:
            pad_a = k - 1
        else:
            pad_a = int(np.ceil(pad_len / 2))
    elif padding == "VALID":
        pad_len = k + s - 2 + max(k - s, 0)
        pad_a = k - 1
    else:
        raise ValueError("Padding mode must be 'SAME' or 'VALID'.")
    pad_b = pad_len - pad_a
    return int(pad_a), int(pad_b)


def _maybe_broadcast(x: int | Sequence[int] | None, rank: int) -> tuple[int, ...]:
    if x is None:
        x = 1
    if isinstance(x, int):
        return (int(x),) * rank
    return tuple(int(v) for v in x)


def _flatten_padding(pads: Sequence[Sequence[int]]) -> list[int]:
    befores = [int(before) for before, _ in pads]
    afters = [int(after) for _, after in pads]
    return befores + afters


def _normalize_padding(
    padding: str | Sequence[Sequence[int]],
    kernel_spatial: Sequence[int],
    strides: Sequence[int],
    dilations: Sequence[int],
) -> str | tuple[tuple[int, int], ...]:
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in {"SAME", "VALID"}:
            raise NotImplementedError(
                f"ConvTranspose padding '{padding}' is not supported."
            )
        pairs = []
        for k, s, d in zip(kernel_spatial, strides, dilations):
            k_eff = (int(k) - 1) * int(d) + 1
            pairs.append(_conv_transpose_padding(k_eff, s, padding))
        return tuple(pairs)
    return tuple((int(lo), int(hi)) for lo, hi in padding)


def _transpose_value(ctx, value, perm, shape, name_hint):
    builder = getattr(ctx, "builder", None)
    if builder is None:
        raise AttributeError("IR build context missing builder")
    out = builder.Transpose(
        value,
        _outputs=[ctx.fresh_name(name_hint)],
        perm=list(perm),
    )
    if getattr(value, "type", None) is not None:
        out.type = value.type
    _stamp_type_and_shape(out, shape)
    _ensure_value_metadata(ctx, out)
    return out


def _flip_spatial_dims(ctx, value, shape, spatial_axes, name_hint):
    builder = getattr(ctx, "builder", None)
    if builder is None:
        raise AttributeError("IR build context missing builder")
    if not spatial_axes:
        return value
    starts = [int(shape[i]) - 1 for i in spatial_axes]
    ends = [-(2**63)] * len(spatial_axes)
    steps = [-1] * len(spatial_axes)
    starts_val = _const_i64(ctx, starts, name_hint=f"{name_hint}_starts")
    ends_val = _const_i64(ctx, ends, name_hint=f"{name_hint}_ends")
    axes_val = _const_i64(ctx, spatial_axes, name_hint=f"{name_hint}_axes")
    steps_val = _const_i64(ctx, steps, name_hint=f"{name_hint}_steps")
    flipped = builder.Slice(
        value,
        starts_val,
        ends_val,
        axes_val,
        steps_val,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    if getattr(value, "type", None) is not None:
        flipped.type = value.type
    _stamp_type_and_shape(flipped, shape)
    _ensure_value_metadata(ctx, flipped)
    return flipped


@register_primitive(
    jaxpr_primitive="linen.conv_transpose",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.ConvTranspose",
    onnx=[
        {
            "component": "ConvTranspose",
            "doc": "https://onnx.ai/onnx/operators/onnx__ConvTranspose.html",
        }
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="conv_transpose",
    testcases=[
        {
            "testcase": "conv_transpose_basic",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.ConvTranspose,
                input_shape=(1, 8, 8, 3),
                features=4,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "run_only_f32_variant": True,
            "input_shapes": [("B", 8, 8, 3)],
            "expected_output_shapes": [("B", 8, 8, 4)],
        },
        {
            "testcase": "conv_transpose_valid_stride",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.ConvTranspose,
                input_shape=(1, 5, 5, 2),
                features=3,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "run_only_f32_variant": True,
            "input_shapes": [(1, 5, 5, 2)],
            "expected_output_shapes": [(1, 11, 11, 3)],
        },
    ],
)
class ConvTransposePlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.linen.ConvTranspose → ONNX ConvTranspose."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.conv_transpose")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x,
        kernel,
        bias,
        *,
        use_bias: bool,
        strides: Sequence[int],
        padding: str | Sequence[Sequence[int]],
        kernel_dilation: Sequence[int],
        transpose_kernel: bool,
        precision=None,
        preferred_element_type=None,
    ):
        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        k_spec = jax.ShapeDtypeStruct(kernel.shape, kernel.dtype)
        b_spec = jax.ShapeDtypeStruct(bias.shape, bias.dtype)

        def _shape_fn(xv, kv, bv):
            y = jax.lax.conv_transpose(
                xv,
                kv,
                strides,
                padding,
                rhs_dilation=kernel_dilation,
                transpose_kernel=transpose_kernel,
                precision=precision,
                preferred_element_type=preferred_element_type,
            )
            if use_bias:
                y = y + bv
            return y

        out = jax.eval_shape(_shape_fn, x_spec, k_spec, b_spec)
        return jax.core.ShapedArray(out.shape, out.dtype)

    def lower(self, ctx: Any, eqn):
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for Linen ConvTranspose lowering"
            )

        params = dict(eqn.params)
        use_bias = bool(params.get("use_bias", True))
        strides = params.get("strides", (1, 1))
        padding = params.get("padding", "SAME")
        kernel_dilation = params.get("kernel_dilation", (1, 1))
        transpose_kernel = bool(params.get("transpose_kernel", False))

        if transpose_kernel:
            raise NotImplementedError(
                "linen.ConvTranspose with transpose_kernel=True is not supported yet."
            )

        x_var, k_var, b_var = eqn.invars[:3]
        out_var = eqn.outvars[0]

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        k_shape = tuple(getattr(getattr(k_var, "aval", None), "shape", ()))
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))

        spatial_dims = max(len(k_shape) - 2, 1)
        if len(x_shape) != spatial_dims + 2:
            raise NotImplementedError(
                "linen.ConvTranspose only supports a single batch dimension."
            )

        strides = _maybe_broadcast(strides, spatial_dims)
        kernel_dilation = _maybe_broadcast(kernel_dilation, spatial_dims)
        kernel_spatial = tuple(int(k) for k in k_shape[:spatial_dims])
        padding_pairs = _normalize_padding(
            padding, kernel_spatial, strides, kernel_dilation
        )
        pads_jax = [int(v) for v in _flatten_padding(padding_pairs)]
        pads_jax_starts = pads_jax[:spatial_dims]
        pads_jax_ends = pads_jax[spatial_dims:]
        kernel_effective = [
            (k - 1) * d + 1 for k, d in zip(kernel_spatial, kernel_dilation)
        ]
        pads_onnx_starts = [
            k - 1 - p for k, p in zip(kernel_effective, pads_jax_starts)
        ]
        pads_onnx_ends = [k - 1 - p for k, p in zip(kernel_effective, pads_jax_ends)]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("conv_x"))
        k_val = ctx.get_value_for_var(k_var, name_hint=ctx.fresh_name("conv_kernel"))
        b_val = (
            ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("bias"))
            if use_bias
            else None
        )

        k_val = cast_param_like(ctx, k_val, x_val, "kernel_cast")
        if use_bias and b_val is not None:
            b_val = cast_param_like(ctx, b_val, x_val, "bias_cast")

        x_nchw_shape = (x_shape[0], x_shape[-1], *x_shape[1:-1])
        x_val = _transpose_value(
            ctx,
            x_val,
            perm=(0, spatial_dims + 1, *range(1, spatial_dims + 1)),
            shape=x_nchw_shape,
            name_hint="conv_x_nchw",
        )

        k_iohw_shape = (
            k_shape[spatial_dims],
            k_shape[spatial_dims + 1],
            *k_shape[:spatial_dims],
        )
        k_val = _transpose_value(
            ctx,
            k_val,
            perm=(spatial_dims, spatial_dims + 1, *range(spatial_dims)),
            shape=k_iohw_shape,
            name_hint="conv_kernel_iohw",
        )

        spatial_axes = list(range(2, 2 + spatial_dims))
        k_val = _flip_spatial_dims(
            ctx,
            k_val,
            k_iohw_shape,
            spatial_axes,
            name_hint="conv_kernel_flipped",
        )

        conv_kwargs = {
            "strides": list(strides),
            "pads": pads_onnx_starts + pads_onnx_ends,
        }
        if any(d != 1 for d in kernel_dilation):
            conv_kwargs["dilations"] = list(kernel_dilation)

        conv_inputs = [x_val, k_val] + (
            [b_val] if use_bias and b_val is not None else []
        )
        conv_out_name = ctx.fresh_name("conv_out_nchw")
        conv_out = builder.ConvTranspose(
            *conv_inputs,
            _outputs=[conv_out_name],
            **conv_kwargs,
        )
        if getattr(x_val, "type", None) is not None:
            conv_out.type = x_val.type
        out_nchw_shape = (out_shape[0], out_shape[-1], *out_shape[1:-1])
        _stamp_type_and_shape(conv_out, out_nchw_shape)
        _ensure_value_metadata(ctx, conv_out)

        final_val = _transpose_value(
            ctx,
            conv_out,
            perm=(0, *range(2, 2 + spatial_dims), 1),
            shape=out_shape,
            name_hint="conv_out",
        )
        ctx.bind_value_for_var(out_var, final_val)

    @staticmethod
    def _make_patch(orig_fn: Callable):
        ConvTransposePlugin._ORIGINAL_CALL = orig_fn
        prim = ConvTransposePlugin._PRIM

        def patched(self, inputs):
            kernel_size = (
                (self.kernel_size,)
                if isinstance(self.kernel_size, int)
                else tuple(self.kernel_size)
            )
            if inputs.ndim != len(kernel_size) + 2:
                raise NotImplementedError(
                    "linen.ConvTranspose only supports a single batch dimension."
                )
            if getattr(self, "transpose_kernel", False):
                raise NotImplementedError(
                    "linen.ConvTranspose with transpose_kernel=True is not supported yet."
                )

            strides = _maybe_broadcast(getattr(self, "strides", 1), len(kernel_size))
            kernel_dilation = _maybe_broadcast(
                getattr(self, "kernel_dilation", 1), len(kernel_size)
            )
            padding_lax = linen_linear.canonicalize_padding(
                getattr(self, "padding", "SAME"),
                len(kernel_size),
            )
            if padding_lax == "CIRCULAR":
                raise NotImplementedError(
                    "linen.ConvTranspose with CIRCULAR padding is not supported yet."
                )

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
            if bool(getattr(self, "use_bias", True)):
                if bias is None:
                    bias = jnp.zeros((int(self.features),), dtype=inputs.dtype)
            else:
                bias = jnp.asarray(0, dtype=inputs.dtype)

            return prim.bind(
                inputs,
                kernel,
                bias,
                use_bias=bool(getattr(self, "use_bias", True)),
                strides=strides,
                padding=padding_lax,
                kernel_dilation=kernel_dilation,
                transpose_kernel=bool(getattr(self, "transpose_kernel", False)),
                precision=getattr(self, "precision", None),
                preferred_element_type=getattr(self, "preferred_element_type", None),
            )

        return patched

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec(
                "flax.linen",
                "conv_transpose_p",
                cls._PRIM,
                delete_if_missing=True,
            ),
            MonkeyPatchSpec(
                target="flax.linen.ConvTranspose",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


@ConvTransposePlugin._PRIM.def_impl
def _impl(
    x,
    kernel,
    bias,
    *,
    use_bias,
    strides,
    padding,
    kernel_dilation,
    transpose_kernel,
    precision=None,
    preferred_element_type=None,
):
    y = jax.lax.conv_transpose(
        x,
        kernel,
        strides,
        padding,
        rhs_dilation=kernel_dilation,
        transpose_kernel=transpose_kernel,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    if use_bias:
        y = y + bias
    return y
