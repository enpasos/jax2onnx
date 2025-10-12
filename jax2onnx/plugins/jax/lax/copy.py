# jax2onnx/plugins/jax/lax/copy.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jax import lax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=lax.copy_p.name,
    jax_doc=(
        "Handles the JAX primitive lax.copy_p. The public API `jax.lax.copy` "
        "is deprecated, but the primitive still appears in transformed jaxprs."
    ),
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="<your_current_version>",
    context="primitives.lax",
    component="copy",
    testcases=[
        {
            "testcase": "copy_float32_array",
            "callable": lambda x: lax.copy_p.bind(x),
            "input_shapes": [(2, 3)],
            "input_dtypes": [np.float32],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float32],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Identity:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "copy_int64_scalar",
            "callable": lambda x: lax.copy_p.bind(x),
            "input_values": [np.array(10, dtype=np.int64)],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.int64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Identity"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class CopyPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.copy_p`` to an ONNX Identity node in IR."""

    @staticmethod
    def abstract_eval(
        operand_aval: Any, **unused_kwargs: Any
    ):  # pragma: no cover - JAX hook
        return operand_aval

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        if len(eqn.invars) != 1 or len(eqn.outvars) != 1:
            raise ValueError("lax.copy_p expects exactly one input and one output")

        in_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        in_val = ctx.get_value_for_var(in_var, name_hint=ctx.fresh_name("copy_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("copy_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Identity")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Identity")

        result = ctx.builder.Identity(in_val, _outputs=[desired_name])

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = getattr(in_val, "type", None)
        result.shape = (
            getattr(out_spec, "shape", None) or in_val.shape or ir.Shape(out_shape)
        )
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
