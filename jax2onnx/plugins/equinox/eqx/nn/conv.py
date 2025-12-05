# jax2onnx/plugins/equinox/eqx/nn/conv.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax import ShapeDtypeStruct
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import (
    _ensure_value_metadata,
    _is_static_int,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins._utils import inline_reshape_initializer
from jax2onnx.plugins._complex_utils import cast_real_tensor
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _normalized_shape(
    shape: Tuple[int | str | None, ...],
) -> Tuple[int | str | None, ...]:
    dims: list[int | str | None] = []
    for dim in shape:
        if _is_static_int(dim):
            dims.append(int(dim))  # type: ignore[arg-type]
        else:
            dims.append(dim)
    return tuple(dims)


def _conv_forward(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array,
    *,
    use_bias: bool,
    strides: Sequence[int],
    padding: Any,
    dilations: Sequence[int],
    groups: int,
    num_spatial_dims: int,
) -> jnp.ndarray:
    arr = jnp.asarray(x)
    w = jnp.asarray(weight)

    if arr.ndim == num_spatial_dims + 1:
        arr = jnp.expand_dims(arr, axis=0)
        squeeze_axis = True
    elif arr.ndim == num_spatial_dims + 2:
        squeeze_axis = False
    else:
        raise ValueError(
            f"eqx Conv expected {num_spatial_dims + 1} dims (without batch) or "
            f"{num_spatial_dims + 2} dims (with batch); got rank {arr.ndim}."
        )

    rhs_dilation = tuple(int(d) for d in dilations)
    window_strides = tuple(int(s) for s in strides)
    conv_out = jax.lax.conv_general_dilated(
        lhs=arr,
        rhs=w,
        window_strides=window_strides,
        padding=padding,
        rhs_dilation=rhs_dilation,
        feature_group_count=int(groups),
    )

    if squeeze_axis:
        conv_out = jnp.squeeze(conv_out, axis=0)

    if use_bias:
        conv_out = conv_out + jnp.asarray(bias, dtype=conv_out.dtype)

    return conv_out


def _conv_shape(
    x_aval: ShapedArray,
    weight_aval: ShapedArray,
    bias_aval: ShapedArray,
    *,
    use_bias: bool,
    strides: Sequence[int],
    padding: Any,
    dilations: Sequence[int],
    groups: int,
    num_spatial_dims: int,
) -> ShapedArray:
    x_spec = ShapeDtypeStruct(x_aval.shape, x_aval.dtype)
    w_spec = ShapeDtypeStruct(weight_aval.shape, weight_aval.dtype)
    b_spec = ShapeDtypeStruct(bias_aval.shape, bias_aval.dtype)

    def _shape_fn(x, w, b):
        return _conv_forward(
            x,
            w,
            b,
            use_bias=use_bias,
            strides=strides,
            padding=padding,
            dilations=dilations,
            groups=groups,
            num_spatial_dims=num_spatial_dims,
        )

    out_spec = jax.eval_shape(_shape_fn, x_spec, w_spec, b_spec)
    return ShapedArray(out_spec.shape, out_spec.dtype)


_EQX_CONV_EXAMPLE: Final[eqx.nn.Conv2d] = eqx.nn.Conv2d(
    in_channels=3,
    out_channels=8,
    kernel_size=3,
    stride=2,
    padding=1,
    key=jax.random.PRNGKey(0),
)


@register_primitive(
    jaxpr_primitive="eqx.nn.conv",
    jax_doc="https://docs.kidger.site/equinox/api/nn/conv/",
    onnx=[
        {
            "component": "Conv",
            "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html",
        }
    ],
    since="v0.10.0",
    context="primitives.eqx",
    component="conv",
    testcases=[
        {
            "testcase": "eqx_conv2d_nchw",
            "callable": _EQX_CONV_EXAMPLE,
            "input_shapes": [(3, 32, 32)],
            "post_check_onnx_graph": expect_graph(
                ["Unsqueeze:1x3x32x32 -> Conv:1x8x16x16 -> Squeeze:8x16x16"],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "eqx_conv2d_batched_nchw",
            "callable": jax.vmap(_EQX_CONV_EXAMPLE),
            "input_shapes": [(5, 3, 32, 32)],
            "post_check_onnx_graph": expect_graph(
                ["Conv:5x8x16x16"],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
    ],
)
class ConvPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.conv")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax_core.AbstractValue,
        weight: jax_core.AbstractValue,
        bias: jax_core.AbstractValue,
        **params: Any,
    ) -> ShapedArray:
        params = dict(params)
        params.pop("padding_mode", None)
        return _conv_shape(
            x,
            weight,
            bias,
            **params,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: jax_core.JaxprEqn) -> None:
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for Eqx Conv lowering"
            )

        params = dict(eqn.params)
        num_spatial = int(params.get("num_spatial_dims", 2))
        use_bias = bool(params.get("use_bias", True))
        strides = tuple(int(s) for s in params.get("strides", (1,) * num_spatial))
        dilations = tuple(int(d) for d in params.get("dilations", (1,) * num_spatial))
        groups = int(params.get("groups", 1))
        padding = params.get("padding", "VALID")
        padding_mode = params.get("padding_mode", "ZEROS")

        if padding_mode != "ZEROS":
            raise NotImplementedError(
                "Eqx Conv with non-zero padding modes is not supported in ONNX lowering."
            )

        x_var, weight_var, bias_var = eqn.invars
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("eqx_conv_x"))
        w_val = ctx.get_value_for_var(
            weight_var, name_hint=ctx.fresh_name("eqx_conv_w")
        )
        b_val = ctx.get_value_for_var(bias_var, name_hint=ctx.fresh_name("eqx_conv_b"))

        b_val = ctx.get_value_for_var(bias_var, name_hint=ctx.fresh_name("eqx_conv_b"))

        target_dtype = _dtype_to_ir(
            np.dtype(getattr(out_var.aval, "dtype", np.float32)),
            builder.enable_double_precision,
        )

        x_val = cast_real_tensor(ctx, x_val, target_dtype, name_hint="eqx_conv_x_cast")
        w_val = cast_real_tensor(ctx, w_val, target_dtype, name_hint="eqx_conv_w_cast")
        b_val = cast_real_tensor(ctx, b_val, target_dtype, name_hint="eqx_conv_b_cast")

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        has_batch = len(x_shape) == num_spatial + 2

        input_val = x_val
        if not has_batch:
            unsq_axes = _const_i64(ctx, [0], name_hint="eqx_conv_unsq_axes")
            input_val = builder.Unsqueeze(
                x_val,
                unsq_axes,
                _outputs=[ctx.fresh_name("eqx_conv_unsqueezed")],
            )
            input_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
            if input_dtype is not None:
                input_val.type = ir.TensorType(input_dtype)
            unsq_dims = (1,) + _normalized_shape(x_shape)
            _stamp_type_and_shape(input_val, unsq_dims)
            _ensure_value_metadata(ctx, input_val)
        else:
            input_dtype = getattr(getattr(input_val, "type", None), "dtype", None)

        bias_val: Optional[ir.Value] = None
        if use_bias:
            bias_shape = tuple(getattr(getattr(bias_var, "aval", None), "shape", ()))
            bias_val = b_val
            if len(bias_shape) > 1:
                const_payload = getattr(bias_val, "const_value", None)
                if const_payload is not None:
                    original_bias_val = bias_val
                    bias_val = inline_reshape_initializer(
                        ctx,
                        bias_val,
                        (bias_shape[0],),
                        name_hint="eqx_conv_bias_inline",
                    )
                    try:
                        ctx.builder.initializers.remove(original_bias_val)
                    except ValueError:
                        try:
                            ctx._initializers.remove(original_bias_val)
                        except Exception:
                            pass
                    if input_dtype is not None:
                        bias_val.type = ir.TensorType(input_dtype)
                    _stamp_type_and_shape(bias_val, (bias_shape[0],))
                    _ensure_value_metadata(ctx, bias_val)
                else:
                    squeeze_axes = [
                        idx
                        for idx, dim in enumerate(bias_shape)
                        if idx != 0 and dim == 1
                    ]
                    if squeeze_axes:
                        squeeze_axes_val = _const_i64(
                            ctx, squeeze_axes, name_hint="eqx_conv_bias_axes"
                        )
                        bias_val = builder.Squeeze(
                            bias_val,
                            squeeze_axes_val,
                            _outputs=[ctx.fresh_name("eqx_conv_bias")],
                        )
                        if input_dtype is not None:
                            bias_val.type = ir.TensorType(input_dtype)
                        _stamp_type_and_shape(bias_val, (bias_shape[0],))
                        _ensure_value_metadata(ctx, bias_val)

        conv_kwargs: dict[str, object] = {
            "strides": [int(s) for s in strides],
        }
        if any(d != 1 for d in dilations):
            conv_kwargs["dilations"] = [int(d) for d in dilations]

        if groups != 1:
            conv_kwargs["group"] = groups

        if isinstance(padding, str):
            mode = padding.upper()
            if mode == "VALID":
                conv_kwargs["pads"] = [0] * (2 * num_spatial)
            elif mode in {"SAME", "SAME_UPPER"}:
                conv_kwargs["auto_pad"] = "SAME_UPPER"
            elif mode == "SAME_LOWER":
                conv_kwargs["auto_pad"] = "SAME_LOWER"
            else:
                raise NotImplementedError(f"Unsupported padding mode {padding!r}")
        else:
            pad_pairs = tuple(
                (int(lo), int(hi)) for lo, hi in tuple(padding)  # type: ignore[arg-type]
            )
            conv_kwargs["pads"] = [p[0] for p in pad_pairs] + [p[1] for p in pad_pairs]

        conv_inputs = [input_val, w_val]
        if use_bias and bias_val is not None:
            conv_inputs.append(bias_val)

        conv_out_name = ctx.fresh_name("eqx_conv_out_nchw")
        conv_result = builder.Conv(
            *conv_inputs,
            _outputs=[conv_out_name],
            **conv_kwargs,
        )

        if input_dtype is not None:
            conv_result.type = ir.TensorType(input_dtype)

        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))
        if not has_batch:
            conv_dims = (1,) + _normalized_shape(out_shape)
        else:
            conv_dims = _normalized_shape(out_shape)
        _stamp_type_and_shape(conv_result, conv_dims)
        _ensure_value_metadata(ctx, conv_result)

        if not has_batch:
            squeeze_axes = _const_i64(ctx, [0], name_hint="eqx_conv_squeeze_axes")
            final_val = builder.Squeeze(
                conv_result,
                squeeze_axes,
                _outputs=[ctx.fresh_name("eqx_conv_out")],
            )
            if input_dtype is not None:
                final_val.type = ir.TensorType(input_dtype)
            _stamp_type_and_shape(final_val, _normalized_shape(out_shape))
            _ensure_value_metadata(ctx, final_val)
            ctx.bind_value_for_var(out_var, final_val)
        else:
            ctx.bind_value_for_var(out_var, conv_result)

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("equinox.nn", "conv_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="equinox.nn._conv.Conv",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _patch_call(
        orig: Callable[..., jax.Array] | None,
    ) -> Callable[[eqx.nn._conv.Conv, jax.Array], jax.Array]:
        del orig

        def wrapped(
            self: eqx.nn._conv.Conv, x: jax.Array, *, key: jax.Array | None = None
        ) -> jax.Array:
            del key

            arr = jnp.asarray(x)
            expected_rank = int(self.num_spatial_dims) + 1
            if arr.ndim not in {expected_rank, expected_rank + 1}:
                raise ValueError(
                    f"Input to eqx Conv needs rank {expected_rank} (no batch) or "
                    f"{expected_rank + 1} (batched); received shape {arr.shape}."
                )

            if self.padding_mode != "ZEROS":
                raise NotImplementedError(
                    "Eqx Conv padding_mode other than 'ZEROS' is not supported."
                )

            if self.use_bias:
                bias = jnp.asarray(self.bias, dtype=arr.dtype)
            else:
                bias = jnp.zeros(
                    (self.out_channels,) + (1,) * self.num_spatial_dims,
                    dtype=arr.dtype,
                )

            return ConvPlugin._PRIM.bind(
                arr,
                jnp.asarray(self.weight, dtype=arr.dtype),
                bias,
                use_bias=bool(self.use_bias),
                strides=tuple(self.stride),
                padding=self.padding,
                dilations=tuple(self.dilation),
                groups=int(self.groups),
                num_spatial_dims=int(self.num_spatial_dims),
                padding_mode=str(self.padding_mode),
            )

        return wrapped


@ConvPlugin._PRIM.def_impl
def _conv_impl(
    x: jax.Array, weight: jax.Array, bias: jax.Array, **params: Any
) -> jax.Array:
    params = dict(params)
    params.pop("padding_mode", None)
    return _conv_forward(x, weight, bias, **params)


def _conv_batch_rule(
    batched_args: tuple[jax.Array, jax.Array, jax.Array],
    batch_dims: tuple[int | None, int | None, int | None],
    **params: Any,
) -> tuple[jax.Array, int | None]:
    x, weight, bias = batched_args
    x_bdim, w_bdim, b_bdim = batch_dims

    if w_bdim is not None or b_bdim is not None:
        raise NotImplementedError("Batching over Eqx Conv parameters is not supported.")

    if x_bdim is not None and x_bdim != 0:
        x = jnp.moveaxis(x, x_bdim, 0)

    out = ConvPlugin._PRIM.bind(x, weight, bias, **params)

    if x_bdim is not None and x_bdim != 0:
        out = jnp.moveaxis(out, 0, x_bdim)

    return out, x_bdim


batching.primitive_batchers[ConvPlugin._PRIM] = _conv_batch_rule
