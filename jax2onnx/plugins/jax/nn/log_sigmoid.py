# jax2onnx/plugins/jax/nn/log_sigmoid.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
from jax2onnx._compat.jax import (
    AbstractValue,
    JaxprEqn,
    Primitive,
    ShapedArray,
)
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._precision_utils import promote_float32_operand
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax._log1p_expm1 import ElementwiseEmitter, emit_log1p
from jax2onnx.plugins.jax.nn._builder_utils import register_unary_elementwise_batch_rule
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


_LOG_SIGMOID_PRIM: Final[Primitive] = Primitive("jax.nn.log_sigmoid")
_LOG_SIGMOID_PRIM.multiple_results = False
_JAX_LOG_SIGMOID_ORIG: Final = jax.nn.log_sigmoid


@register_native_primitive(
    jaxpr_primitive=_LOG_SIGMOID_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.log_sigmoid.html",
    onnx=[
        {"component": "Abs", "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html"},
        {"component": "Neg", "doc": "https://onnx.ai/onnx/operators/onnx__Neg.html"},
        {"component": "Exp", "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Min", "doc": "https://onnx.ai/onnx/operators/onnx__Min.html"},
        {
            "component": "Equal",
            "doc": "https://onnx.ai/onnx/operators/onnx__Equal.html",
        },
        {"component": "Or", "doc": "https://onnx.ai/onnx/operators/onnx__Or.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.nn",
    component="log_sigmoid",
    testcases=[
        {
            "testcase": "jaxnn_log_sigmoid",
            "callable": lambda x: jax.nn.log_sigmoid(x),
            # Log(Sigmoid(x)) gives -inf from -20 down on ONNX Runtime, and
            # loses the tail from about 17 up, where sigmoid rounds to 1.
            "input_values": [
                np.array(
                    [[-120.0, -30.0, -20.0, -1.0, -1e-3], [0.0, 1e-3, 1.0, 17.0, 30.0]],
                    dtype=np.float32,
                )
            ],
            "rtol_f32": 1e-6,
            "rtol_f64": 1e-12,
            "atol": 0.0,
            "post_check_onnx_graph": EG(
                ["Abs:2x5 -> Neg:2x5 -> Exp:2x5 -> Add:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_log_sigmoid_dynamic",
            "callable": lambda x: jax.nn.log_sigmoid(x),
            "input_shapes": [("B", 4)],
            "post_check_onnx_graph": EG(
                ["Abs:Bx4 -> Neg:Bx4 -> Exp:Bx4 -> Add:Bx4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class LogSigmoidPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.log_sigmoid`` as ``min(x, 0) - log1p(exp(-|x|))``."""

    _PRIM: ClassVar[Primitive] = _LOG_SIGMOID_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: AbstractValue) -> ShapedArray:
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.require_value_for_var(x_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("log_sigmoid_out")
        )
        x_val = promote_float32_operand(
            ctx,
            x_val,
            target_dtype=out_spec.dtype,
            name_hint="log_sigmoid_operand_double",
        )

        emit = ElementwiseEmitter(ctx, x_val, out_spec, "log_sigmoid")
        x_val = emit.input(x_val)
        # Same formula as jax.nn.log_sigmoid, -softplus(-x): exp(-|x|) never
        # overflows, and log1p keeps the tail where sigmoid rounds to 1.
        neg_abs = emit.op("Neg", emit.op("Abs", x_val, name="abs"), name="neg_abs")
        tail = emit_log1p(
            ctx, emit.op("Exp", neg_abs, name="exp"), name_hint="log_sigmoid_log1p"
        )
        min_x_0 = emit.op("Min", x_val, emit.scalar(0.0), name="min")
        result = emit.final("Sub", min_x_0, tail)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jax.nn.log_sigmoid not found")

            def _patched(x: ArrayLike) -> ArrayLike:
                x_arr = jnp.asarray(x)
                if not jnp.issubdtype(x_arr.dtype, jnp.floating):
                    return orig(x)
                return cls._PRIM.bind(x_arr)

            return _patched

        return [
            AssignSpec("jax.nn", "log_sigmoid_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="log_sigmoid",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="log_sigmoid",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="log_sigmoid",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@LogSigmoidPlugin._PRIM.def_impl
def _log_sigmoid_impl(x: ArrayLike) -> ArrayLike:
    return _JAX_LOG_SIGMOID_ORIG(x)


register_unary_elementwise_batch_rule(LogSigmoidPlugin._PRIM)
register_jvp_via_jax_jvp(LogSigmoidPlugin._PRIM, _log_sigmoid_impl)
