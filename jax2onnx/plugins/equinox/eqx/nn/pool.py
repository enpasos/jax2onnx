# jax2onnx/plugins/equinox/eqx/nn/pool.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, ClassVar, Final

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
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


POOL_PRIM: Final[Primitive] = Primitive("eqx.nn.pool")
POOL_PRIM.multiple_results = False

_EQX_POOL_MAX_2D: Final[eqx.nn.Pool] = eqx.nn.Pool(
    init=-jnp.inf,
    operation=jax.lax.max,
    num_spatial_dims=2,
    kernel_size=(2, 2),
    stride=(2, 2),
    padding=0,
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


def _pool_max_forward(
    x: jax.Array,
    *,
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
) -> jax.Array:
    arr = jnp.asarray(x)
    num_spatial = len(kernel_size)
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        raise TypeError("Eqx Pool(max) plugin currently supports floating dtypes only.")

    if arr.ndim == num_spatial + 1:
        arr = jnp.expand_dims(arr, axis=0)
        squeeze_batch = True
    elif arr.ndim == num_spatial + 2:
        squeeze_batch = False
    else:
        raise ValueError(
            f"eqx Pool(max) expects rank {num_spatial + 1} (no batch) or "
            f"{num_spatial + 2} (with batch); got {arr.ndim}."
        )

    init = jnp.asarray(-jnp.inf, dtype=arr.dtype)
    window_dims = (1, 1, *kernel_size)
    window_strides = (1, 1, *strides)
    pad_cfg = ((0, 0), (0, 0), *padding)
    out = jax.lax.reduce_window(
        arr,
        init,
        jax.lax.max,
        window_dimensions=window_dims,
        window_strides=window_strides,
        padding=pad_cfg,
    )

    if squeeze_batch:
        out = jnp.squeeze(out, axis=0)
    return out


def _resolve_pool_op(operation: Any, init: Any) -> str | None:
    init_arr = np.asarray(init)
    if operation is jax.lax.max and bool(np.isneginf(init_arr).all()):
        return "max"
    name = getattr(operation, "__name__", "")
    if name == "max" and bool(np.isneginf(init_arr).all()):
        return "max"
    return None


@register_primitive(
    jaxpr_primitive=POOL_PRIM.name,
    jax_doc="https://docs.kidger.site/equinox/api/nn/pooling/#equinox.nn.Pool",
    onnx=[
        {
            "component": "MaxPool",
            "doc": "https://onnx.ai/onnx/operators/onnx__MaxPool.html",
        }
    ],
    since="0.12.2",
    context="primitives.eqx",
    component="pool",
    testcases=[
        {
            "testcase": "eqx_pool_max2d_basic",
            "callable": _EQX_POOL_MAX_2D,
            "input_shapes": [(3, 8, 8)],
            "expected_output_shapes": [(3, 4, 4)],
            "post_check_onnx_graph": EG(
                ["Unsqueeze:1x3x8x8 -> MaxPool:1x3x4x4 -> Squeeze:3x4x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_pool_max2d_batched",
            "callable": lambda x, _mod=_EQX_POOL_MAX_2D: jax.vmap(_mod)(x),
            "input_shapes": [("B", 3, 8, 8)],
            "expected_output_shapes": [("B", 3, 4, 4)],
            "post_check_onnx_graph": EG(
                ["MaxPool:Bx3x4x4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class PoolPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for ``equinox.nn.Pool`` (currently max-reducer subset)."""

    _PRIM: ClassVar[Primitive] = POOL_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax_core.AbstractValue,
        *,
        op: str,
        kernel_size: tuple[int, ...],
        strides: tuple[int, ...],
        padding: tuple[tuple[int, int], ...],
    ) -> ShapedArray:
        if op != "max":
            raise NotImplementedError(f"Unsupported eqx.nn.Pool op '{op}'.")
        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        out_spec = jax.eval_shape(
            lambda v: _pool_max_forward(
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
                "IR build context missing builder for Eqx Pool lowering"
            )

        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars
        params = dict(eqn.params)
        op = str(params["op"])
        if op != "max":
            raise NotImplementedError(f"Unsupported eqx.nn.Pool op '{op}'.")

        kernel_size = tuple(int(v) for v in params["kernel_size"])
        strides = tuple(int(v) for v in params["strides"])
        padding = tuple((int(lo), int(hi)) for lo, hi in params["padding"])

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("pool_x"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("pool_out"))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))
        num_spatial = len(kernel_size)
        rank = len(x_shape)

        if rank not in {num_spatial + 1, num_spatial + 2}:
            raise ValueError(
                f"Eqx Pool lowering expects rank {num_spatial + 1} or "
                f"{num_spatial + 2}; received {rank}."
            )

        dtype = getattr(getattr(x_val, "type", None), "dtype", None)

        pooled_input = x_val
        if rank == num_spatial + 1:
            unsq_axes = _const_i64(ctx, [0], name_hint="eqx_pool_unsq_axes")
            pooled_input = builder.Unsqueeze(
                x_val,
                unsq_axes,
                _outputs=[ctx.fresh_name("eqx_pool_unsqueeze")],
            )
            if dtype is not None:
                pooled_input.type = ir.TensorType(dtype)
            _stamp_type_and_shape(pooled_input, (1, *x_shape))
            _ensure_value_metadata(ctx, pooled_input)

        pads = _flatten_padding(padding)
        pool_kwargs: dict[str, Any] = {
            "kernel_shape": kernel_size,
            "strides": strides,
        }
        if any(int(p) != 0 for p in pads):
            pool_kwargs["pads"] = pads

        pooled_name = (
            ctx.fresh_name("eqx_pool_pooled")
            if rank == num_spatial + 1
            else (getattr(out_spec, "name", None) or ctx.fresh_name("eqx_pool"))
        )
        pooled = builder.MaxPool(
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
            squeeze_axes = _const_i64(ctx, [0], name_hint="eqx_pool_sq_axes")
            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
                "eqx_pool_out"
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
            AssignSpec("equinox.nn", "pool_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="equinox.nn.Pool",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def _patch_call(cls, orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(
            self: eqx.nn.Pool,
            x: jax.Array,
            *,
            key: jax.Array | None = None,
        ) -> jax.Array:
            op = _resolve_pool_op(self.operation, self.init)
            if op != "max":
                return orig(self, x, key=key)

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

            return cls._PRIM.bind(
                x,
                op=op,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
            )

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, *, op, kernel_size, strides, padding: cls.abstract_eval(
                    x,
                    op=op,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


@PoolPlugin._PRIM.def_impl
def _pool_impl(
    x: jax.Array,
    *,
    op: str,
    kernel_size: Sequence[int],
    strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> jax.Array:
    if op != "max":
        raise NotImplementedError(f"Unsupported eqx.nn.Pool op '{op}'.")
    return _pool_max_forward(
        x,
        kernel_size=tuple(int(v) for v in kernel_size),
        strides=tuple(int(v) for v in strides),
        padding=tuple((int(lo), int(hi)) for lo, hi in padding),
    )


def _pool_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | None, ...],
    *,
    op: str,
    kernel_size: Sequence[int],
    strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> tuple[jax.Array, int | None]:
    (x,) = batched_args
    (x_bdim,) = batch_dims

    if x_bdim is None:
        out = PoolPlugin._PRIM.bind(
            x,
            op=op,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
        )
        return out, None

    if x_bdim != 0:
        x = jnp.moveaxis(x, x_bdim, 0)
    out = PoolPlugin._PRIM.bind(
        x,
        op=op,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
    )
    return out, 0


batching.primitive_batchers[PoolPlugin._PRIM] = _pool_batch_rule
