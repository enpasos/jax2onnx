# jax2onnx/plugins/jax/nn/logmeanexp.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_LOGMEANEXP_PRIM: Final[Primitive] = Primitive("jax.nn.logmeanexp")
_LOGMEANEXP_PRIM.multiple_results = False
_JAX_LOGMEANEXP_ORIG: Final = jax.nn.logmeanexp


def _axis_arg(axes: tuple[int, ...] | None) -> int | tuple[int, ...] | None:
    if axes is None:
        return None
    if len(axes) == 1:
        return int(axes[0])
    return tuple(int(ax) for ax in axes)


def _normalize_axes(axes: tuple[int, ...] | None, rank: int) -> tuple[int, ...] | None:
    if axes is None:
        return None
    out: list[int] = []
    for ax in axes:
        axis_i = int(ax)
        if axis_i < 0:
            axis_i += rank
        if axis_i < 0 or axis_i >= rank:
            raise ValueError(f"axis {ax} out of bounds for rank {rank}")
        out.append(axis_i)
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate axes are not supported: {axes}")
    return tuple(out)


@register_primitive(
    jaxpr_primitive=_LOGMEANEXP_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.logmeanexp.html",
    onnx=[
        {
            "component": "ReduceLogSumExp",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceLogSumExp.html",
        },
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
    ],
    since="0.12.1",
    context="primitives.nn",
    component="logmeanexp",
    testcases=[
        {
            "testcase": "jaxnn_logmeanexp_axis1",
            "callable": lambda x: jax.nn.logmeanexp(x, axis=1),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(
                ["ReduceLogSumExp:2 -> Sub:2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_logmeanexp_axis12_keepdims",
            "callable": lambda x: jax.nn.logmeanexp(x, axis=(1, 2), keepdims=True),
            "input_shapes": [(2, 3, 4)],
            "expected_output_shapes": [(2, 1, 1)],
            "post_check_onnx_graph": EG(
                ["ReduceLogSumExp:2x1x1 -> Sub:2x1x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_logmeanexp_dynamic_static_reduction_axis",
            "callable": lambda x: jax.nn.logmeanexp(x, axis=1),
            "input_shapes": [("B", 5)],
            "expected_output_shapes": [("B",)],
            "post_check_onnx_graph": EG(
                ["ReduceLogSumExp:B -> Sub:B"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class LogMeanExpPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.logmeanexp`` as ``ReduceLogSumExp - log(count)``."""

    _PRIM: ClassVar[Primitive] = _LOGMEANEXP_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue,
        *,
        axes: tuple[int, ...] | None,
        keepdims: bool,
    ) -> jax.core.ShapedArray:
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        axis_arg = _axis_arg(axes)
        out = jax.eval_shape(
            lambda arr: _JAX_LOGMEANEXP_ORIG(
                arr,
                axis=axis_arg,
                where=None,
                keepdims=keepdims,
            ),
            spec,
        )
        return jax.core.ShapedArray(out.shape, out.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        axes = eqn.params.get("axes")
        keepdims = bool(eqn.params.get("keepdims", False))

        axes_norm = _normalize_axes(axes, rank) if axes is not None else None
        reduce_axes = tuple(range(rank)) if axes_norm is None else axes_norm

        reduce_count = 1
        for ax in reduce_axes:
            dim = x_shape[ax]
            if not isinstance(dim, (int, np.integer)):
                raise NotImplementedError(
                    "jax.nn.logmeanexp lowering requires static reduced dimensions"
                )
            reduce_count *= int(dim)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("logmeanexp_in"))
        out_spec = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("logmeanexp_out"),
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "logmeanexp_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("logmeanexp_out")

        out_type = getattr(out_spec, "type", None) or getattr(x_val, "type", None)
        out_shape = getattr(out_spec, "shape", None) or getattr(
            getattr(out_var, "aval", None), "shape", None
        )
        x_dtype = np.dtype(getattr(getattr(x_var, "aval", None), "dtype", np.float32))

        reduce_inputs = [x_val]
        if axes_norm is not None:
            axes_const = _const_i64(
                ctx, np.asarray(axes_norm, dtype=np.int64), "logmeanexp_axes"
            )
            reduce_inputs.append(axes_const)

        reduced = ctx.builder.ReduceLogSumExp(
            *reduce_inputs,
            keepdims=1 if keepdims else 0,
            _outputs=[ctx.fresh_name("logmeanexp_reduced")],
        )
        if out_type is not None:
            reduced.type = out_type
        if out_shape is not None:
            reduced.shape = out_shape

        log_count_const = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("logmeanexp_log_count"),
            array=np.asarray(np.log(float(reduce_count)), dtype=x_dtype),
        )

        result = ctx.builder.Sub(
            reduced,
            log_count_const,
            _outputs=[desired_name],
        )
        if out_type is not None:
            result.type = out_type
        if out_shape is not None:
            result.shape = out_shape
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jax.nn.logmeanexp not found")

            def _patched(
                x: ArrayLike,
                axis=None,
                where: ArrayLike | None = None,
                keepdims: bool = False,
            ) -> ArrayLike:
                if where is not None:
                    return orig(x, axis=axis, where=where, keepdims=keepdims)

                x_arr = jnp.asarray(x)
                if not jnp.issubdtype(x_arr.dtype, jnp.floating):
                    return orig(x, axis=axis, where=where, keepdims=keepdims)

                axes: tuple[int, ...] | None
                if axis is None:
                    axes = None
                elif isinstance(axis, (tuple, list)):
                    if len(axis) == 0:
                        return orig(x, axis=axis, where=where, keepdims=keepdims)
                    axes = tuple(int(ax) for ax in axis)
                else:
                    axes = (int(axis),)

                rank = x_arr.ndim
                try:
                    axes_norm = (
                        _normalize_axes(axes, rank) if axes is not None else None
                    )
                except ValueError:
                    return orig(x, axis=axis, where=where, keepdims=keepdims)

                reduced_axes = tuple(range(rank)) if axes_norm is None else axes_norm
                for ax in reduced_axes:
                    dim = x_arr.shape[ax]
                    if not isinstance(dim, (int, np.integer)):
                        return orig(x, axis=axis, where=where, keepdims=keepdims)

                return cls._PRIM.bind(
                    x_arr,
                    axes=axes_norm,
                    keepdims=bool(keepdims),
                )

            return _patched

        return [
            AssignSpec("jax.nn", "logmeanexp_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="logmeanexp",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@LogMeanExpPlugin._PRIM.def_impl
def _logmeanexp_impl(
    x: ArrayLike,
    *,
    axes: tuple[int, ...] | None,
    keepdims: bool,
) -> ArrayLike:
    return _JAX_LOGMEANEXP_ORIG(
        x,
        axis=_axis_arg(axes),
        where=None,
        keepdims=keepdims,
    )


def _logmeanexp_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | type(batching.not_mapped), ...],
    *,
    axes: tuple[int, ...] | None,
    keepdims: bool,
):
    (x,), (bdim,) = batched_args, batch_dims
    if bdim is batching.not_mapped:
        out = LogMeanExpPlugin._PRIM.bind(x, axes=axes, keepdims=keepdims)
        return out, batching.not_mapped

    x_front = batching.bdim_at_front(x, bdim, x.shape[bdim])
    slice_rank = x_front.ndim - 1
    if axes is None:
        new_axes = tuple(range(1, x_front.ndim))
    else:
        axes_norm = _normalize_axes(axes, slice_rank)
        new_axes = tuple(int(ax) + 1 for ax in axes_norm)

    out = LogMeanExpPlugin._PRIM.bind(x_front, axes=new_axes, keepdims=keepdims)
    return out, 0


batching.primitive_batchers[LogMeanExpPlugin._PRIM] = _logmeanexp_batch_rule
register_jvp_via_jax_jvp(LogMeanExpPlugin._PRIM, _logmeanexp_impl)
