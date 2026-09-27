# jax2onnx/plugins/jax/numpy/cos.py

from __future__ import annotations

from typing import Any, ClassVar, Final

import jax
from jax2onnx._compat.jax import (
    AbstractValue,
    JaxprEqn,
    ShapedArray,
)
import jax.numpy as jnp
import numpy as np
from onnx_ir import TensorType

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
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
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_COS_PRIM: Final = make_jnp_primitive("jax.numpy.cos")


@register_primitive(
    jaxpr_primitive=_COS_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cos.html",
    onnx=[
        {
            "component": "Cos",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cos.html",
        },
        {
            "component": "Sin",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sin.html",
        },
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="cos",
    testcases=[
        {
            "testcase": "jnp_cos_basic",
            "callable": lambda x: jnp.cos(x),
            "input_shapes": [(3,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Cos:3"],
                must_absent=["Sin"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_cos_int_promote",
            "callable": lambda x: jnp.cos(x),
            "input_values": [np.array([0, 1, 2], dtype=np.int32)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Cos:3"],
                must_absent=["Sin"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_cos_f64_via_sin",
            "callable": lambda x: jnp.cos(x),
            "input_values": [np.array([0.0, 1.0, 2.0], dtype=np.float64)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Mul:3 -> Sin:3 -> Mul:3 -> Mul:3 -> Sub:3"],
                must_absent=["Cos"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "cos_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.cos)(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "cos_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.cos(y)))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpCosPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _COS_PRIM
    _FUNC_NAME: ClassVar[str] = "cos"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: AbstractValue) -> ShapedArray:
        return abstract_eval_via_orig_unary(
            JnpCosPlugin._PRIM,
            JnpCosPlugin._FUNC_NAME,
            x,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("jnp_cos_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_cos_out")
        )

        op_input = cast_input_to_output_dtype(
            ctx,
            x_var,
            out_var,
            x_val,
            output_hint="jnp_cos",
        )

        out_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(out_var, "aval", None), "dtype", np.float32)
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("jnp_cos_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("jnp_cos_out")

        if out_dtype == np.float64:
            # Keep compatibility with older ORT CPU kernels using
            # cos(x) = 1 - 2*sin(x/2)**2. Scaling by 1/2 avoids the large-angle
            # phase loss of sin(x + pi/2). Near cosine zeros, cancellation limits
            # relative accuracy; the absolute error remains small.
            half = ctx.bind_const_for_var(object(), np.asarray(0.5, dtype=out_dtype))
            two = ctx.bind_const_for_var(object(), np.asarray(2.0, dtype=out_dtype))
            one = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=out_dtype))
            half_x = ctx.builder.Mul(
                op_input, half, _outputs=[ctx.fresh_name("jnp_cos_half_x")]
            )
            sin_half = ctx.builder.Sin(
                half_x, _outputs=[ctx.fresh_name("jnp_cos_sin_half")]
            )
            squared = ctx.builder.Mul(
                sin_half, sin_half, _outputs=[ctx.fresh_name("jnp_cos_sin_squared")]
            )
            twice_squared = ctx.builder.Mul(
                squared, two, _outputs=[ctx.fresh_name("jnp_cos_twice_sin_squared")]
            )
            result = ctx.builder.Sub(one, twice_squared, _outputs=[desired_name])
            for value in (half_x, sin_half, squared, twice_squared, result):
                value.type = op_input.type
                _stamp_type_and_shape(value, getattr(out_var.aval, "shape", ()))
                _ensure_value_metadata(ctx, value)
        else:
            result = ctx.builder.Cos(op_input, _outputs=[desired_name])

        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            dtype_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
            result.type = TensorType(dtype_enum)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpCosPlugin._PRIM.def_impl
def _cos_impl(x: object) -> object:
    orig = get_orig_impl(JnpCosPlugin._PRIM, JnpCosPlugin._FUNC_NAME)
    return orig(x)


register_jvp_via_jax_jvp(JnpCosPlugin._PRIM, _cos_impl)
register_unary_elementwise_batch_rule(JnpCosPlugin._PRIM)
