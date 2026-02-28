# jax2onnx/plugins/equinox/eqx/nn/adaptive_pool.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, Sequence

import equinox as eqx
from equinox.nn import _pool as eqx_pool
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
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

_SUPPORTED_OPS: Final[dict[str, Callable[..., jax.Array]]] = {
    "avg": jnp.mean,
    "max": jnp.max,
}

_ONNX_OPS: Final[dict[str, str]] = {
    "avg": "AveragePool",
    "max": "MaxPool",
}

_EQX_ADAPTIVE_AVG_2D: Final[eqx.nn.AdaptiveAvgPool2d] = eqx.nn.AdaptiveAvgPool2d((4, 4))
_EQX_ADAPTIVE_MAX_2D: Final[eqx.nn.AdaptiveMaxPool2d] = eqx.nn.AdaptiveMaxPool2d((4, 4))
_EQX_ADAPTIVE_GENERIC_AVG: Final[eqx.nn.AdaptivePool] = eqx.nn.AdaptivePool(
    (4, 4), num_spatial_dims=2, operation=jnp.mean
)


def _adaptive_pool_unbatched(
    x: jax.Array,
    *,
    target_shape: tuple[int, ...],
    operation: Callable[..., jax.Array],
) -> jax.Array:
    arr = jnp.asarray(x)
    if arr.ndim - 1 != len(target_shape):
        raise ValueError(
            f"AdaptivePool expected {len(target_shape)} spatial dimensions, "
            f"got rank {arr.ndim}."
        )

    out = arr
    for i in range(1, out.ndim):
        op = jax.vmap(eqx_pool._adaptive_pool1d, (0, None, None), 0)
        for j in range(1, out.ndim):
            if i == j:
                continue
            op = jax.vmap(op, in_axes=(j, None, None), out_axes=j)
        out = op(out, int(target_shape[i - 1]), operation)
    return out


def _adaptive_pool_forward(
    x: jax.Array,
    *,
    target_shape: tuple[int, ...],
    op: str,
) -> jax.Array:
    operation = _SUPPORTED_OPS.get(op)
    if operation is None:
        raise NotImplementedError(f"Unsupported adaptive pool op '{op}'.")

    arr = jnp.asarray(x)
    spatial_rank = len(target_shape)

    if arr.ndim == spatial_rank + 1:
        return _adaptive_pool_unbatched(
            arr,
            target_shape=target_shape,
            operation=operation,
        )

    if arr.ndim == spatial_rank + 2:
        return jax.vmap(
            lambda v: _adaptive_pool_unbatched(
                v,
                target_shape=target_shape,
                operation=operation,
            )
        )(arr)

    raise ValueError(
        f"AdaptivePool expects rank {spatial_rank + 1} (no batch) or "
        f"{spatial_rank + 2} (batched); got {arr.ndim}."
    )


@register_primitive(
    jaxpr_primitive="eqx.nn.adaptive_pool",
    jax_doc="https://docs.kidger.site/equinox/api/nn/pooling/#equinox.nn.AdaptivePool",
    onnx=[
        {
            "component": "AveragePool",
            "doc": "https://onnx.ai/onnx/operators/onnx__AveragePool.html",
        },
        {
            "component": "MaxPool",
            "doc": "https://onnx.ai/onnx/operators/onnx__MaxPool.html",
        },
    ],
    since="0.12.2",
    context="primitives.eqx",
    component="adaptive_pool",
    testcases=[
        {
            "testcase": "eqx_adaptive_avg_pool2d_divisible",
            "callable": _EQX_ADAPTIVE_AVG_2D,
            "input_shapes": [(3, 8, 8)],
            "expected_output_shapes": [(3, 4, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["Unsqueeze:1x3x8x8 -> AveragePool:1x3x4x4 -> Squeeze:3x4x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_adaptive_max_pool2d_divisible",
            "callable": _EQX_ADAPTIVE_MAX_2D,
            "input_shapes": [(3, 8, 8)],
            "expected_output_shapes": [(3, 4, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["Unsqueeze:1x3x8x8 -> MaxPool:1x3x4x4 -> Squeeze:3x4x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_adaptive_pool_generic_avg_vmap",
            "callable": lambda x, _mod=_EQX_ADAPTIVE_GENERIC_AVG: jax.vmap(_mod)(x),
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
class AdaptivePoolPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.adaptive_pool")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax_core.AbstractValue,
        *,
        target_shape: tuple[int, ...],
        op: str,
    ) -> ShapedArray:
        if op not in _SUPPORTED_OPS:
            raise NotImplementedError(f"Unsupported adaptive pool op '{op}'.")

        x_shape = tuple(x.shape)
        rank = len(x_shape)
        spatial_rank = len(target_shape)

        if rank not in {spatial_rank + 1, spatial_rank + 2}:
            raise ValueError(
                f"AdaptivePool expected rank {spatial_rank + 1} or "
                f"{spatial_rank + 2}; got {rank}."
            )

        if rank == spatial_rank + 1:
            out_shape = (x_shape[0], *tuple(int(v) for v in target_shape))
        else:
            out_shape = (x_shape[0], x_shape[1], *tuple(int(v) for v in target_shape))

        return ShapedArray(out_shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax_core.JaxprEqn) -> None:
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for Eqx AdaptivePool lowering"
            )

        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars
        params = dict(eqn.params)

        target_shape = tuple(int(v) for v in params["target_shape"])
        op = str(params["op"])
        onnx_op = _ONNX_OPS.get(op)
        if onnx_op is None:
            raise NotImplementedError(f"Unsupported adaptive pool op '{op}'.")

        x_val = ctx.get_value_for_var(
            x_var, name_hint=ctx.fresh_name("adaptive_pool_x")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("adaptive_pool_out")
        )

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))

        spatial_rank = len(target_shape)
        rank = len(x_shape)
        if rank not in {spatial_rank + 1, spatial_rank + 2}:
            raise ValueError(
                f"AdaptivePool lowering expects rank {spatial_rank + 1} or "
                f"{spatial_rank + 2}; got {rank}."
            )

        dtype = getattr(getattr(x_val, "type", None), "dtype", None)

        pooled_input = x_val
        if rank == spatial_rank + 1:
            unsq_axes = _const_i64(ctx, [0], name_hint="eqx_adaptive_pool_unsq_axes")
            pooled_input = builder.Unsqueeze(
                x_val,
                unsq_axes,
                _outputs=[ctx.fresh_name("eqx_adaptive_pool_unsqueeze")],
            )
            if dtype is not None:
                pooled_input.type = ir.TensorType(dtype)
            _stamp_type_and_shape(pooled_input, (1, *x_shape))
            _ensure_value_metadata(ctx, pooled_input)

        spatial_dims = x_shape[-spatial_rank:]
        kernel_shape: list[int] = []
        for in_dim, out_dim in zip(spatial_dims, target_shape, strict=False):
            if not isinstance(in_dim, (int, np.integer)):
                raise NotImplementedError(
                    "AdaptivePool lowering currently requires static spatial input dims."
                )
            in_dim_i = int(in_dim)
            if in_dim_i <= 0 or int(out_dim) <= 0:
                raise ValueError(
                    f"AdaptivePool requires positive dims, got input={in_dim_i}, target={out_dim}."
                )
            if in_dim_i % int(out_dim) != 0:
                raise NotImplementedError(
                    "AdaptivePool lowering currently supports divisible targets only. "
                    f"Got input dim {in_dim_i} and target {out_dim}."
                )
            kernel_shape.append(in_dim_i // int(out_dim))

        pool_kwargs: dict[str, Any] = {
            "kernel_shape": tuple(kernel_shape),
            "strides": tuple(kernel_shape),
        }
        if op == "avg":
            pool_kwargs["count_include_pad"] = 1

        pooled_name = (
            ctx.fresh_name("eqx_adaptive_pool_pooled")
            if rank == spatial_rank + 1
            else (
                getattr(out_spec, "name", None)
                or ctx.fresh_name("eqx_adaptive_pool_result")
            )
        )

        method = getattr(builder, onnx_op, None)
        if not callable(method):
            raise AttributeError(f"IR builder missing {onnx_op} for AdaptivePool")

        pooled = method(
            pooled_input,
            _outputs=[pooled_name],
            **pool_kwargs,
        )
        if dtype is not None:
            pooled.type = ir.TensorType(dtype)
        pooled_shape = (1, *out_shape) if rank == spatial_rank + 1 else out_shape
        _stamp_type_and_shape(pooled, pooled_shape)
        _ensure_value_metadata(ctx, pooled)

        if rank == spatial_rank + 1:
            squeeze_axes = _const_i64(ctx, [0], name_hint="eqx_adaptive_pool_sq_axes")
            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
                "eqx_adaptive_pool_out"
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

    @staticmethod
    def _resolve_op(operation: Any) -> str | None:
        if operation is jnp.mean:
            return "avg"
        if operation is jnp.max:
            return "max"

        name = getattr(operation, "__name__", "")
        if name == "mean":
            return "avg"
        if name in {"max", "amax"}:
            return "max"
        return None

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec(
                "equinox.nn",
                "adaptive_pool_p",
                cls._PRIM,
                delete_if_missing=True,
            ),
            MonkeyPatchSpec(
                target="equinox.nn.AdaptivePool",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _patch_call(
        orig: Callable[..., jax.Array] | None,
    ) -> Callable[..., jax.Array]:
        def wrapped(
            self: eqx.nn.AdaptivePool,
            x: jax.Array,
            *,
            key: jax.Array | None = None,
        ) -> jax.Array:
            op = AdaptivePoolPlugin._resolve_op(self.operation)
            if op is None:
                if orig is not None:
                    return orig(self, x, key=key)
                raise NotImplementedError(
                    "Unsupported eqx.nn.AdaptivePool operation; expected mean or max."
                )

            del key
            target_shape = tuple(int(v) for v in self.target_shape)
            return AdaptivePoolPlugin._PRIM.bind(
                x,
                target_shape=target_shape,
                op=op,
            )

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, *, target_shape, op: cls.abstract_eval(
                    x,
                    target_shape=target_shape,
                    op=op,
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


@AdaptivePoolPlugin._PRIM.def_impl
def _adaptive_pool_impl(
    x: jax.Array,
    *,
    target_shape: Sequence[int],
    op: str,
) -> jax.Array:
    return _adaptive_pool_forward(
        x,
        target_shape=tuple(int(v) for v in target_shape),
        op=op,
    )


def _adaptive_pool_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | None, ...],
    *,
    target_shape: Sequence[int],
    op: str,
) -> tuple[jax.Array, int | None]:
    (x,) = batched_args
    (x_bdim,) = batch_dims

    if x_bdim is None:
        out = AdaptivePoolPlugin._PRIM.bind(
            x,
            target_shape=tuple(int(v) for v in target_shape),
            op=op,
        )
        return out, None

    if x_bdim != 0:
        x = jnp.moveaxis(x, x_bdim, 0)
    out = AdaptivePoolPlugin._PRIM.bind(
        x,
        target_shape=tuple(int(v) for v in target_shape),
        op=op,
    )
    return out, 0


batching.primitive_batchers[AdaptivePoolPlugin._PRIM] = _adaptive_pool_batch_rule
