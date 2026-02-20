# jax2onnx/plugins/jax/nn/standardize.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Primitive
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.nn._builder_utils import register_unary_elementwise_batch_rule
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_STANDARDIZE_PRIM: Final[Primitive] = Primitive("jax.nn.standardize")
_STANDARDIZE_PRIM.multiple_results = False


def _normalize_axes(axis: int | Sequence[int] | None, rank: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(rank))
    if isinstance(axis, Sequence) and not isinstance(axis, (str, bytes)):
        axes = tuple(int(a) for a in axis)
    else:
        axes = (int(axis),)
    out: list[int] = []
    for a in axes:
        aa = int(a)
        if aa < 0:
            aa += rank
        if aa < 0 or aa >= rank:
            raise ValueError(f"axis out of range for rank {rank}: {a}")
        out.append(aa)
    return tuple(out)


@register_primitive(
    jaxpr_primitive=_STANDARDIZE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.standardize.html",
    onnx=[
        {
            "component": "MeanVarianceNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__MeanVarianceNormalization.html",
        }
    ],
    since="0.12.1",
    context="primitives.nn",
    component="standardize",
    testcases=[
        {
            "testcase": "jaxnn_standardize_axis_last_eps0",
            "callable": lambda x: jax.nn.standardize(x, axis=-1, epsilon=0.0),
            "input_shapes": [(2, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["MeanVarianceNormalization:2x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_standardize_axis_tuple_eps0",
            "callable": lambda x: jax.nn.standardize(x, axis=(2, 3), epsilon=0.0),
            "input_shapes": [("B", 3, 4, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["MeanVarianceNormalization:Bx3x4x5"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "standardize_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(
                lambda y: jnp.sum(jax.nn.standardize(y, axis=1, epsilon=0.0) ** 2)
            )(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class StandardizePlugin(PrimitiveLeafPlugin):
    """Lower a safe subset of ``jax.nn.standardize`` to ONNX ``MeanVarianceNormalization``."""

    _PRIM: ClassVar[Primitive] = _STANDARDIZE_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue,
        *,
        axis: tuple[int, ...] | None = None,
        epsilon: float = 0.0,
    ) -> jax.core.ShapedArray:
        del axis, epsilon
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        params = dict(getattr(eqn, "params", {}) or {})
        axis_param = params.get("axis", None)
        epsilon = float(params.get("epsilon", 0.0))
        if not np.isclose(epsilon, 0.0):
            raise NotImplementedError(
                "jax.nn.standardize lowering supports epsilon == 0 only"
            )

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        axes = _normalize_axes(axis_param, len(x_shape))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("std_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("std_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("std_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("std_out")

        result = ctx.builder.MeanVarianceNormalization(
            x_val,
            axes=[int(a) for a in axes],
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = getattr(x_val, "type", None)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            result.shape = getattr(x_val, "shape", None)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = "__orig_impl__standardize"

        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jax.nn.standardize not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                x: ArrayLike,
                axis: int | Sequence[int] | None = -1,
                mean: ArrayLike | None = None,
                variance: ArrayLike | None = None,
                epsilon: ArrayLike = 1e-5,
                where: ArrayLike | None = None,
            ) -> ArrayLike:
                # Only intercept the subset representable via MeanVarianceNormalization.
                if mean is not None or variance is not None or where is not None:
                    return orig(
                        x,
                        axis=axis,
                        mean=mean,
                        variance=variance,
                        epsilon=epsilon,
                        where=where,
                    )
                try:
                    eps = float(np.asarray(epsilon).reshape(()))
                except Exception:
                    return orig(
                        x,
                        axis=axis,
                        mean=mean,
                        variance=variance,
                        epsilon=epsilon,
                        where=where,
                    )
                if not np.isclose(eps, 0.0):
                    return orig(
                        x,
                        axis=axis,
                        mean=mean,
                        variance=variance,
                        epsilon=epsilon,
                        where=where,
                    )
                if axis is None:
                    axis_param: tuple[int, ...] | None = None
                elif isinstance(axis, Sequence) and not isinstance(axis, (str, bytes)):
                    axis_param = tuple(int(a) for a in axis)
                else:
                    axis_param = (int(axis),)
                return cls._PRIM.bind(jnp.asarray(x), axis=axis_param, epsilon=eps)

            return _patched

        return [
            AssignSpec("jax.nn", "standardize_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="standardize",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@StandardizePlugin._PRIM.def_impl
def _standardize_impl(
    x: ArrayLike,
    *,
    axis: tuple[int, ...] | None = None,
    epsilon: float = 0.0,
) -> ArrayLike:
    orig = getattr(
        StandardizePlugin._PRIM, "__orig_impl__standardize", jax.nn.standardize
    )
    return orig(x, axis=axis, mean=None, variance=None, epsilon=epsilon, where=None)


register_unary_elementwise_batch_rule(StandardizePlugin._PRIM)


register_jvp_via_jax_jvp(StandardizePlugin._PRIM, _standardize_impl)
