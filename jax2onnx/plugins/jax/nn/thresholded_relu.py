# jax2onnx/plugins/jax/nn/thresholded_relu.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.nn._builder_utils import register_unary_elementwise_batch_rule
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_THRESHOLDED_RELU_PRIM: Final[Primitive] = Primitive("jax.nn.thresholded_relu")
_THRESHOLDED_RELU_PRIM.multiple_results = False


def _thresholded_relu_fallback(x: ArrayLike, threshold: float = 1.0) -> ArrayLike:
    arr = jnp.asarray(x)
    thr = jnp.asarray(threshold, dtype=arr.dtype)
    return jnp.where(arr > thr, arr, jnp.zeros((), dtype=arr.dtype))


def _thresholded_relu_bind(x: ArrayLike, threshold: float = 1.0) -> Any:
    return _THRESHOLDED_RELU_PRIM.bind(x, threshold=float(threshold))


@register_primitive(
    jaxpr_primitive=_THRESHOLDED_RELU_PRIM.name,
    jax_doc="https://onnx.ai/onnx/operators/onnx__ThresholdedRelu.html",
    onnx=[
        {
            "component": "ThresholdedRelu",
            "doc": "https://onnx.ai/onnx/operators/onnx__ThresholdedRelu.html",
        }
    ],
    since="0.12.1",
    context="primitives.nn",
    component="thresholded_relu",
    testcases=[
        {
            "testcase": "jaxnn_thresholded_relu_default",
            "callable": lambda x: _thresholded_relu_bind(x),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["ThresholdedRelu:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_thresholded_relu_custom",
            "callable": lambda x: _thresholded_relu_bind(x, threshold=0.2),
            "input_shapes": [("B", 3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["ThresholdedRelu:Bx3x4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class ThresholdedReluPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.thresholded_relu`` to ONNX ``ThresholdedRelu``."""

    _PRIM: ClassVar[Primitive] = _THRESHOLDED_RELU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue,
        threshold: float = 1.0,
    ) -> jax.core.ShapedArray:
        del threshold
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        threshold = float(eqn.params.get("threshold", 1.0))
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(
            x_var,
            name_hint=ctx.fresh_name("thresholded_relu_in"),
        )
        out_spec = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("thresholded_relu_out"),
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "thresholded_relu_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("thresholded_relu_out")

        result = ctx.builder.ThresholdedRelu(
            x_val,
            _outputs=[desired_name],
            alpha=threshold,
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
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_value(
            orig: Callable[..., Any] | None,
        ) -> Callable[..., Any]:
            def _patched(x: ArrayLike, threshold: float = 1.0) -> Any:
                return cls._PRIM.bind(x, threshold=float(threshold))

            # Preserve behaviour when symbol exists; otherwise provide fallback semantics.
            if orig is None:
                return _patched
            return _patched

        return [
            AssignSpec(
                "jax.nn",
                "thresholded_relu_p",
                cls._PRIM,
                delete_if_missing=True,
            ),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="thresholded_relu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="thresholded_relu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="thresholded_relu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@ThresholdedReluPlugin._PRIM.def_impl
def _thresholded_relu_impl(x: ArrayLike, threshold: float = 1.0) -> ArrayLike:
    return _thresholded_relu_fallback(x, threshold=threshold)


register_unary_elementwise_batch_rule(ThresholdedReluPlugin._PRIM)
