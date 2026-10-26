# jax2onnx/plugins/jax/nn/log1mexp.py

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
from jax2onnx.plugins.jax._log1p_expm1 import (
    ElementwiseEmitter,
    emit_expm1,
    emit_log1p,
)
from jax2onnx.plugins.jax.nn._builder_utils import register_unary_elementwise_batch_rule
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


_LOG1MEXP_PRIM: Final[Primitive] = Primitive("jax.nn.log1mexp")
_LOG1MEXP_PRIM.multiple_results = False
_JAX_LOG1MEXP_ORIG: Final = jax.nn.log1mexp


@register_native_primitive(
    jaxpr_primitive=_LOG1MEXP_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.log1mexp.html",
    onnx=[
        {"component": "Less", "doc": "https://onnx.ai/onnx/operators/onnx__Less.html"},
        {"component": "Neg", "doc": "https://onnx.ai/onnx/operators/onnx__Neg.html"},
        {"component": "Exp", "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
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
    component="log1mexp",
    testcases=[
        {
            "testcase": "jaxnn_log1mexp_basic",
            "callable": lambda x: jax.nn.log1mexp(x),
            # Uncompensated expm1/log1p give -inf at 1e-8 and 0 at 20.
            "input_values": [
                np.array([1e-8, 1e-4, 0.2, 0.8, 2.0, 20.0], dtype=np.float32)
            ],
            "rtol_f32": 1e-6,
            "rtol_f64": 1e-12,
            "atol": 0.0,
            "post_check_onnx_graph": EG(
                ["Less -> Where:6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_log1mexp_dynamic",
            "callable": lambda x: jax.nn.log1mexp(x),
            "input_shapes": [("B",)],
            "post_check_onnx_graph": EG(
                ["Less -> Where:B"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class Log1mexpPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.log1mexp`` through compensated expm1/log1p branches."""

    _PRIM: ClassVar[Primitive] = _LOG1MEXP_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: AbstractValue) -> ShapedArray:
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        out = jax.eval_shape(_JAX_LOG1MEXP_ORIG, spec)
        return ShapedArray(out.shape, out.dtype)

    def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.require_value_for_var(x_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("log1mexp_out")
        )
        x_val = promote_float32_operand(
            ctx, x_val, target_dtype=out_spec.dtype, name_hint="log1mexp_operand_double"
        )

        emit = ElementwiseEmitter(ctx, x_val, out_spec, "log1mexp")
        x_val = emit.input(x_val)
        mask = emit.op(
            "Less", x_val, emit.scalar(np.log(2.0)), boolean=True, name="mask"
        )
        neg_x = emit.op("Neg", x_val, name="neg_x")

        # Same branches as jax.nn.log1mexp: log(-expm1(-x)) below ln 2,
        # log1p(-exp(-x)) above.
        expm1_neg_x = emit_expm1(ctx, neg_x, name_hint="log1mexp_expm1")
        left = emit.op("Log", emit.op("Neg", expm1_neg_x, name="neg_expm1"))
        neg_exp = emit.op("Neg", emit.op("Exp", neg_x, name="exp_neg_x"))
        right = emit_log1p(ctx, neg_exp, name_hint="log1mexp_log1p")

        result = emit.final("Where", mask, left, right)
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
                raise RuntimeError("Original jax.nn.log1mexp not found")

            def _patched(x: ArrayLike) -> ArrayLike:
                x_arr = jnp.asarray(x)
                if not jnp.issubdtype(x_arr.dtype, jnp.floating):
                    return orig(x)
                return cls._PRIM.bind(x_arr)

            return _patched

        return [
            AssignSpec("jax.nn", "log1mexp_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="log1mexp",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@Log1mexpPlugin._PRIM.def_impl
def _log1mexp_impl(x: ArrayLike) -> ArrayLike:
    return _JAX_LOG1MEXP_ORIG(x)


register_unary_elementwise_batch_rule(Log1mexpPlugin._PRIM)
register_jvp_via_jax_jvp(Log1mexpPlugin._PRIM, _log1mexp_impl)
