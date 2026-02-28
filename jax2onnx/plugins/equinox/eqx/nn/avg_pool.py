# jax2onnx/plugins/equinox/eqx/nn/avg_pool.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, Sequence

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.lax.reduce_window_sum import reduce_window_sum
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

_EQX_AVG_POOL_2D: Final[eqx.nn.AvgPool2d] = eqx.nn.AvgPool2d(
    kernel_size=(2, 2), stride=(2, 2), padding=0
)


def _normalize_padding(padding: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    return tuple((int(pair[0]), int(pair[1])) for pair in padding)


def _flatten_padding(pads: Sequence[tuple[int, int]]) -> list[int]:
    befores = [int(lo) for lo, _ in pads]
    afters = [int(hi) for _, hi in pads]
    return befores + afters


def _validate_padding(
    pads: Sequence[tuple[int, int]], kernel_size: Sequence[int]
) -> None:
    for (left, right), kernel in zip(pads, kernel_size, strict=False):
        if max(int(left), int(right)) > int(kernel):
            raise RuntimeError(
                "Paddings should be less than the kernel size. "
                f"Got padding {(left, right)} for kernel {kernel}."
            )


def _update_padding_for_ceil(
    input_shape: Sequence[int],
    padding: Sequence[tuple[int, int]],
    kernel_size: Sequence[int],
    strides: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    new_padding: list[tuple[int, int]] = []
    for input_size, (left, right), kernel, stride in zip(
        input_shape, padding, kernel_size, strides, strict=False
    ):
        if (int(input_size) + int(left) + int(right) - int(kernel)) % int(stride) == 0:
            new_padding.append((int(left), int(right)))
        else:
            new_padding.append((int(left), int(right) + int(stride)))
    return tuple(new_padding)


def _avg_pool_forward(
    x: jax.Array,
    *,
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
) -> jax.Array:
    arr = jnp.asarray(x)
    num_spatial = len(kernel_size)

    if arr.ndim == num_spatial + 1:
        arr = jnp.expand_dims(arr, axis=0)
        squeeze_batch = True
    elif arr.ndim == num_spatial + 2:
        squeeze_batch = False
    else:
        raise ValueError(
            f"eqx AvgPool expects rank {num_spatial + 1} (no batch) or "
            f"{num_spatial + 2} (with batch); got {arr.ndim}."
        )

    window_dims = (1, 1, *kernel_size)
    window_strides = (1, 1, *strides)
    pad_cfg = ((0, 0), (0, 0), *padding)
    summed = reduce_window_sum(
        arr,
        window_dimensions=window_dims,
        window_strides=window_strides,
        padding=pad_cfg,
        base_dilation=(1,) * len(window_dims),
        window_dilation=(1,) * len(window_dims),
    )
    divisor = jnp.asarray(float(np.prod(kernel_size)), dtype=summed.dtype)
    out = summed / divisor

    if squeeze_batch:
        out = jnp.squeeze(out, axis=0)
    return out


@register_primitive(
    jaxpr_primitive="eqx.nn.avg_pool",
    jax_doc="https://docs.kidger.site/equinox/api/nn/pooling/#equinox.nn.AvgPool2d",
    onnx=[
        {
            "component": "AveragePool",
            "doc": "https://onnx.ai/onnx/operators/onnx__AveragePool.html",
        }
    ],
    since="0.12.2",
    context="primitives.eqx",
    component="avg_pool",
    testcases=[
        {
            "testcase": "eqx_avg_pool2d_basic",
            "callable": _EQX_AVG_POOL_2D,
            "input_shapes": [(3, 8, 8)],
            "expected_output_shapes": [(3, 4, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["Unsqueeze:1x3x8x8 -> AveragePool:1x3x4x4 -> Squeeze:3x4x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_avg_pool2d_batched",
            "callable": lambda x, _mod=_EQX_AVG_POOL_2D: jax.vmap(_mod)(x),
            "input_shapes": [("B", 3, 8, 8)],
            "expected_output_shapes": [("B", 3, 4, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["AveragePool:Bx3x4x4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class AvgPoolPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.avg_pool")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax_core.AbstractValue,
        *,
        kernel_size: tuple[int, ...],
        strides: tuple[int, ...],
        padding: tuple[tuple[int, int], ...],
    ) -> ShapedArray:
        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        out_spec = jax.eval_shape(
            lambda v: _avg_pool_forward(
                v,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
            ),
            x_spec,
        )
        return ShapedArray(out_spec.shape, out_spec.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax_core.JaxprEqn) -> None:
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for Eqx AvgPool lowering"
            )

        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars
        params = dict(eqn.params)
        kernel_size = tuple(int(v) for v in params["kernel_size"])
        strides = tuple(int(v) for v in params["strides"])
        padding = tuple((int(lo), int(hi)) for lo, hi in params["padding"])

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("avgpool_x"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("avgpool_out")
        )

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))
        num_spatial = len(kernel_size)
        rank = len(x_shape)

        if rank not in {num_spatial + 1, num_spatial + 2}:
            raise ValueError(
                f"Eqx AvgPool lowering expects rank {num_spatial + 1} or "
                f"{num_spatial + 2}; received {rank}."
            )

        dtype = getattr(getattr(x_val, "type", None), "dtype", None)

        pooled_input = x_val
        if rank == num_spatial + 1:
            unsq_axes = _const_i64(ctx, [0], name_hint="eqx_avgpool_unsq_axes")
            pooled_input = builder.Unsqueeze(
                x_val,
                unsq_axes,
                _outputs=[ctx.fresh_name("eqx_avgpool_unsqueeze")],
            )
            if dtype is not None:
                pooled_input.type = ir.TensorType(dtype)
            _stamp_type_and_shape(pooled_input, (1, *x_shape))
            _ensure_value_metadata(ctx, pooled_input)

        pads = _flatten_padding(padding)
        pool_kwargs: dict[str, Any] = {
            "kernel_shape": kernel_size,
            "strides": strides,
            "count_include_pad": 1,
        }
        if any(int(p) != 0 for p in pads):
            pool_kwargs["pads"] = pads

        pooled_name = (
            ctx.fresh_name("eqx_avgpool_pooled")
            if rank == num_spatial + 1
            else (getattr(out_spec, "name", None) or ctx.fresh_name("eqx_avgpool"))
        )
        pooled = builder.AveragePool(
            pooled_input,
            _outputs=[pooled_name],
            **pool_kwargs,
        )
        if dtype is not None:
            pooled.type = ir.TensorType(dtype)
        pooled_shape = (1, *out_shape) if rank == num_spatial + 1 else out_shape
        _stamp_type_and_shape(pooled, pooled_shape)
        _ensure_value_metadata(ctx, pooled)

        if rank == num_spatial + 1:
            squeeze_axes = _const_i64(ctx, [0], name_hint="eqx_avgpool_sq_axes")
            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
                "eqx_avgpool_out"
            )
            result = builder.Squeeze(
                pooled,
                squeeze_axes,
                _outputs=[desired_name],
            )
        else:
            result = pooled

        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        elif dtype is not None:
            result.type = ir.TensorType(dtype)

        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("equinox.nn", "avg_pool_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="equinox.nn.AvgPool1d",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="equinox.nn.AvgPool2d",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="equinox.nn.AvgPool3d",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _patch_call(
        orig: Callable[..., jax.Array] | None,
    ) -> Callable[..., jax.Array]:
        del orig

        def wrapped(
            self: eqx.nn.Pool, x: jax.Array, *, key: jax.Array | None = None
        ) -> jax.Array:
            del key
            kernel_size = tuple(int(v) for v in self.kernel_size)
            strides = tuple(int(v) for v in self.stride)
            padding = _normalize_padding(self.padding)

            if bool(getattr(self, "use_ceil", False)):
                spatial_shape = tuple(int(v) for v in x.shape[-len(kernel_size) :])
                padding = _update_padding_for_ceil(
                    spatial_shape,
                    padding,
                    kernel_size,
                    strides,
                )
            _validate_padding(padding, kernel_size)

            return AvgPoolPlugin._PRIM.bind(
                x,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
            )

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, *, kernel_size, strides, padding: cls.abstract_eval(
                    x,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


@AvgPoolPlugin._PRIM.def_impl
def _avg_pool_impl(
    x: jax.Array,
    *,
    kernel_size: Sequence[int],
    strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> jax.Array:
    return _avg_pool_forward(
        x,
        kernel_size=tuple(int(v) for v in kernel_size),
        strides=tuple(int(v) for v in strides),
        padding=tuple((int(lo), int(hi)) for lo, hi in padding),
    )


def _avg_pool_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | None, ...],
    *,
    kernel_size: Sequence[int],
    strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> tuple[jax.Array, int | None]:
    (x,) = batched_args
    (x_bdim,) = batch_dims

    if x_bdim is None:
        out = AvgPoolPlugin._PRIM.bind(
            x,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
        )
        return out, None

    if x_bdim != 0:
        x = jnp.moveaxis(x, x_bdim, 0)
    out = AvgPoolPlugin._PRIM.bind(
        x,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
    )
    return out, 0


batching.primitive_batchers[AvgPoolPlugin._PRIM] = _avg_pool_batch_rule
