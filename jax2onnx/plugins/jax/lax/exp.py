# jax2onnx/plugins/jax/lax/exp.py


from jax2onnx._compat.jax import JaxprEqn
import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._complex_utils import (
    _base_dtype_for_complex,
    coerce_dim_values,
    ensure_packed_real_pair,
    pack_known_real,
    pack_real_imag_pair,
    split_packed_real_imag,
)
from jax2onnx.plugins._loop_extent_meta import propagate_axis0_override
from jax2onnx.plugins._utils import cast_value_to_dtype
from jax2onnx.plugins.jax.lax.cos import emit_real_cos
from jax2onnx.plugins.jax.lax.mul import emit_packed_complex_mul

from jax2onnx.plugins._precision_utils import require_default_accuracy
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


def _emit_complex_exp(
    ctx: LoweringContextProtocol,
    value: ir.Value,
    *,
    component_dtype: ir.DataType,
    source_shape: tuple[object, ...],
    output_name: str | None,
) -> ir.Value:
    """Emit finite complex Exp in the source component width."""
    packed, _ = ensure_packed_real_pair(ctx, value, name_hint="exp_input")
    packed = cast_value_to_dtype(
        ctx, packed, component_dtype, name_hint="exp_components"
    )
    real, imag = split_packed_real_imag(
        ctx, packed, component_dtype, prefix="exp_input"
    )

    def output(name: str) -> ir.Value:
        return ir.Value(
            name=ctx.fresh_name(name),
            type=ir.TensorType(component_dtype),
            shape=ir.Shape(coerce_dim_values(source_shape)),
        )

    magnitude = ctx.builder.op("Exp", [real], output=output("exp_magnitude"))
    cosine = emit_real_cos(
        ctx,
        imag,
        output=output("exp_cosine"),
        source_dtype=np.dtype(component_dtype.numpy()),
        source_shape=source_shape,
    )
    sine = ctx.builder.op("Sin", [imag], output=output("exp_sine"))
    magnitude_pair = pack_known_real(
        ctx, magnitude, component_dtype, name_hint="exp_magnitude"
    )
    phase_pair = pack_real_imag_pair(
        ctx,
        cosine,
        sine,
        component_dtype,
        name_hint="exp_phase",
    )
    # Keep the complete complex product, including +0 * cosine in its
    # imaginary sum: dropping it changes JAX's signed-zero result.
    result = emit_packed_complex_mul(
        ctx,
        magnitude_pair,
        phase_pair,
        target_dtype=component_dtype,
        result_shape=source_shape,
        output_name=output_name,
    )
    propagate_axis0_override(value, result)
    return result


@register_native_primitive(
    jaxpr_primitive=jax.lax.exp_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.exp.html",
    onnx=[
        {
            "component": "Exp",
            "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html",
        }
    ],
    since="0.2.0",
    context="primitives.lax",
    component="exp",
    testcases=[
        {
            "testcase": "exp",
            "callable": lambda x: jax.lax.exp(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Exp:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class ExpPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        require_default_accuracy(eqn)
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.require_value_for_var(x_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("exp_out")
        )

        if np.issubdtype(np.dtype(x_var.aval.dtype), np.complexfloating):
            result = _emit_complex_exp(
                ctx,
                x_val,
                component_dtype=_base_dtype_for_complex(
                    numpy_dtype_to_ir(out_var.aval.dtype)
                ),
                source_shape=tuple(out_var.aval.shape),
                output_name=out_spec.name,
            )
            ctx.bind_value_for_var(out_var, result)
            return

        result = ctx.builder.op("Exp", [x_val], output=out_spec)
        ctx.bind_value_for_var(out_var, result)
