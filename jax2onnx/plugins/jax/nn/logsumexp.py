# jax2onnx/plugins/jax/nn/logsumexp.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
import jax.numpy as jnp
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_LOGSUMEXP_PRIM: Final[Primitive] = Primitive("jax.nn.logsumexp")
_LOGSUMEXP_PRIM.multiple_results = False
_JAX_LOGSUMEXP_ORIG: Final = jax.nn.logsumexp


@register_primitive(
    jaxpr_primitive=_LOGSUMEXP_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.logsumexp.html",
    onnx=[
        {
            "component": "ReduceLogSumExp",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceLogSumExp.html",
        }
    ],
    since="0.12.1",
    context="primitives.nn",
    component="logsumexp",
    testcases=[
        {
            "testcase": "jaxnn_logsumexp_axis1",
            "callable": lambda x: jax.nn.logsumexp(x, axis=1),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ReduceLogSumExp:2",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_logsumexp_axis12_keepdims",
            "callable": lambda x: jax.nn.logsumexp(x, axis=(1, 2), keepdims=True),
            "input_shapes": [(2, 3, 4)],
            "expected_output_shapes": [(2, 1, 1)],
            "post_check_onnx_graph": EG(
                ["ReduceLogSumExp:2x1x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxscipy_logsumexp_axis_last",
            "callable": lambda x: jax.scipy.special.logsumexp(x, axis=-1),
            "input_shapes": [("B", 5)],
            "expected_output_shapes": [("B",)],
            "post_check_onnx_graph": EG(
                ["ReduceLogSumExp:B"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class LogSumExpPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.logsumexp`` / ``jax.scipy.special.logsumexp``."""

    _PRIM: ClassVar[Primitive] = _LOGSUMEXP_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def _axis_arg(axes: tuple[int, ...] | None) -> int | tuple[int, ...] | None:
        if axes is None:
            return None
        if len(axes) == 1:
            return int(axes[0])
        return tuple(int(ax) for ax in axes)

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue,
        *,
        axes: tuple[int, ...] | None,
        keepdims: bool,
    ) -> jax.core.ShapedArray:
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        axis_arg = LogSumExpPlugin._axis_arg(axes)
        out = jax.eval_shape(
            lambda arr: _JAX_LOGSUMEXP_ORIG(
                arr,
                axis=axis_arg,
                b=None,
                keepdims=keepdims,
                return_sign=False,
                where=None,
            ),
            spec,
        )
        return jax.core.ShapedArray(out.shape, out.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        axes = eqn.params.get("axes")
        keepdims = bool(eqn.params.get("keepdims", False))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        axes_norm: tuple[int, ...] | None
        if axes is None:
            axes_norm = None
        else:
            normalized: list[int] = []
            for ax in axes:
                axis_i = int(ax)
                if axis_i < 0:
                    axis_i += rank
                normalized.append(axis_i)
            axes_norm = tuple(normalized)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("logsumexp_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("logsumexp_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("logsumexp")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("logsumexp")

        inputs = [x_val]
        if axes_norm is not None:
            axes_const = _const_i64(ctx, list(axes_norm), "logsumexp_axes")
            inputs.append(axes_const)

        result = ctx.builder.ReduceLogSumExp(
            *inputs,
            keepdims=1 if keepdims else 0,
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[MonkeyPatchSpec]:
        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original logsumexp not found")

            def _patched(
                a: ArrayLike,
                axis=None,
                b=None,
                keepdims: bool = False,
                return_sign: bool = False,
                where=None,
            ):
                if b is not None or where is not None or bool(return_sign):
                    return orig(
                        a,
                        axis=axis,
                        b=b,
                        keepdims=keepdims,
                        return_sign=return_sign,
                        where=where,
                    )

                axes: tuple[int, ...] | None
                if axis is None:
                    axes = None
                elif isinstance(axis, (tuple, list)):
                    if len(axis) == 0:
                        return orig(
                            a,
                            axis=axis,
                            b=None,
                            keepdims=keepdims,
                            return_sign=False,
                            where=None,
                        )
                    axes = tuple(int(ax) for ax in axis)
                else:
                    axes = (int(axis),)

                rank = getattr(a, "ndim", None)
                if isinstance(rank, int) and axes is not None:
                    normalized: list[int] = []
                    for ax in axes:
                        axis_i = int(ax)
                        if axis_i < 0:
                            axis_i += rank
                        if axis_i < 0 or axis_i >= rank:
                            return orig(
                                a,
                                axis=axis,
                                b=None,
                                keepdims=keepdims,
                                return_sign=False,
                                where=None,
                            )
                        normalized.append(axis_i)
                    if len(set(normalized)) != len(normalized):
                        return orig(
                            a,
                            axis=axis,
                            b=None,
                            keepdims=keepdims,
                            return_sign=False,
                            where=None,
                        )
                    axes = tuple(normalized)

                return cls._PRIM.bind(
                    jnp.asarray(a),
                    axes=axes,
                    keepdims=bool(keepdims),
                )

            return _patched

        return [
            MonkeyPatchSpec(
                target="jax.nn",
                attr="logsumexp",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="jax.scipy.special",
                attr="logsumexp",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@LogSumExpPlugin._PRIM.def_impl
def _logsumexp_impl(
    a: ArrayLike,
    *,
    axes: tuple[int, ...] | None,
    keepdims: bool,
):
    axis_arg = LogSumExpPlugin._axis_arg(axes)
    return _JAX_LOGSUMEXP_ORIG(
        a,
        axis=axis_arg,
        b=None,
        keepdims=keepdims,
        return_sign=False,
        where=None,
    )


BatchDim = int | type(batching.not_mapped)


def _logsumexp_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    axes: tuple[int, ...] | None,
    keepdims: bool,
) -> tuple[jax.Array, BatchDim]:
    (operand,), (bdim,) = batched_args, batch_dims
    if bdim is batching.not_mapped:
        out = LogSumExpPlugin._PRIM.bind(operand, axes=axes, keepdims=keepdims)
        return out, batching.not_mapped

    axis_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, axis_size)
    axis_arg = LogSumExpPlugin._axis_arg(axes)
    out = jax.vmap(
        lambda x: _JAX_LOGSUMEXP_ORIG(
            x,
            axis=axis_arg,
            b=None,
            keepdims=keepdims,
            return_sign=False,
            where=None,
        )
    )(operand)
    return out, 0


batching.primitive_batchers[LogSumExpPlugin._PRIM] = _logsumexp_batch_rule
