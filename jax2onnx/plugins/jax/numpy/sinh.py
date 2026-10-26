# jax2onnx/plugins/jax/numpy/sinh.py

from __future__ import annotations

from typing import ClassVar, Final

import jax
from jax2onnx._compat.jax import (
    AbstractValue,
    JaxprEqn,
    ShapedArray,
)
import jax.numpy as jnp
import numpy as np

from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax._log1p_expm1 import emit_sinh
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.jax.numpy._unary_utils import (
    abstract_eval_via_orig_unary,
    cast_input_to_output_dtype,
    register_unary_elementwise_batch_rule,
)
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


_SINH_PRIM: Final = make_jnp_primitive("jax.numpy.sinh")


@register_native_primitive(
    jaxpr_primitive=_SINH_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sinh.html",
    onnx=[
        {"component": "Abs", "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html"},
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
        {"component": "Less", "doc": "https://onnx.ai/onnx/operators/onnx__Less.html"},
        {"component": "Or", "doc": "https://onnx.ai/onnx/operators/onnx__Or.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="sinh",
    testcases=[
        {
            "testcase": "jnp_sinh_basic",
            "callable": lambda x: jnp.sinh(x),
            # (Exp(x) - Exp(-x)) / 2 returns 0 or loses most digits for the
            # small entries.
            "input_values": [
                np.array(
                    [-20.0, -1.0, -1e-3, -(2.0**-30), 0.0, 2.0**-30, 1e-6, 1e-3, 1.0],
                    dtype=np.float32,
                )
            ],
            "rtol_f32": 1e-6,
            "rtol_f64": 1e-12,
            "atol": 0.0,
            "post_check_onnx_graph": EG(
                ["Abs:9 -> Exp:9 -> Sub:9 -> Mul:9 -> Where:9 -> Where:9"],
                must_absent=["Sinh"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_sinh_int_promote",
            "callable": lambda x: jnp.sinh(x),
            "input_values": [np.array([0, 1, 2], dtype=np.int32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Abs:3 -> Exp:3 -> Sub:3 -> Mul:3 -> Where:3 -> Where:3"],
                must_absent=["Sinh"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "sinh_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.sinh)(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "sinh_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.sinh(y)))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpSinhPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SINH_PRIM
    _FUNC_NAME: ClassVar[str] = "sinh"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: AbstractValue) -> ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpSinhPlugin._PRIM,
            JnpSinhPlugin._FUNC_NAME,
            x,
        )

    def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.require_value_for_var(x_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_sinh_out")
        )
        op_input = cast_input_to_output_dtype(
            ctx,
            x_var,
            out_var,
            x_val,
            output_hint="jnp_sinh",
        )

        # Keep jnp.sinh executable on CUDA, which has no native Sinh kernel.
        result = emit_sinh(ctx, op_input, name_hint="jnp_sinh", output=out_spec)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpSinhPlugin._PRIM.def_impl
def _sinh_impl(x: object) -> object:
    orig = get_orig_impl(JnpSinhPlugin._PRIM, JnpSinhPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpSinhPlugin._PRIM, _sinh_impl)
register_unary_elementwise_batch_rule(JnpSinhPlugin._PRIM)
