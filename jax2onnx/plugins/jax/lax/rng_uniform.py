# jax2onnx/plugins/jax/lax/rng_uniform.py

from __future__ import annotations

from typing import Any, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive=jax.lax.rng_uniform_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.rng_uniform.html",
    onnx=[
        {
            "component": "RandomUniform",
            "doc": "https://onnx.ai/onnx/operators/onnx__RandomUniform.html",
        },
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    ],
    since="0.12.1",
    context="primitives.lax",
    component="rng_uniform",
    testcases=[
        {
            "testcase": "rng_uniform_f32",
            "callable": lambda lo, hi: jax.lax.rng_uniform(lo, hi, (2, 3)),
            "input_values": [
                np.asarray(0.0, dtype=np.float32),
                np.asarray(1.0, dtype=np.float32),
            ],
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                ["Sub", "RandomUniform -> Mul -> Add"],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
    ],
)
class RngUniformPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.rng_uniform`` using ONNX ``RandomUniform`` + affine scaling."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        lo_var, hi_var = eqn.invars
        out_var = eqn.outvars[0]
        params = dict(getattr(eqn, "params", {}) or {})

        lo_val = ctx.get_value_for_var(lo_var, name_hint=ctx.fresh_name("rng_lo"))
        hi_val = ctx.get_value_for_var(hi_var, name_hint=ctx.fresh_name("rng_hi"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("rng_out"))

        out_shape = tuple(
            int(d)
            for d in (params.get("shape", getattr(out_var.aval, "shape", ())) or ())
        )
        if any(d < 0 for d in out_shape):
            raise NotImplementedError(
                "rng_uniform requires static non-negative output shape"
            )

        out_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(out_var, "aval", None), "dtype", np.float32)
        )
        out_dtype_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
        if out_dtype_enum is None:
            raise TypeError(f"Unsupported rng_uniform output dtype '{out_dtype}'")

        unit_uniform = cast(
            ir.Value,
            ctx.builder.RandomUniform(
                shape=out_shape,
                dtype=int(out_dtype_enum.value),
                low=0.0,
                high=1.0,
                _outputs=[ctx.fresh_name("rng_unit")],
            ),
        )
        unit_uniform.type = ir.TensorType(out_dtype_enum)
        _stamp_type_and_shape(unit_uniform, out_shape)
        _ensure_value_metadata(ctx, unit_uniform)

        span = cast(
            ir.Value,
            ctx.builder.Sub(
                hi_val,
                lo_val,
                _outputs=[ctx.fresh_name("rng_span")],
            ),
        )
        if getattr(lo_val, "type", None) is not None:
            span.type = lo_val.type
        if getattr(lo_val, "shape", None) is not None:
            span.shape = lo_val.shape
        _ensure_value_metadata(ctx, span)

        scaled = cast(
            ir.Value,
            ctx.builder.Mul(
                unit_uniform,
                span,
                _outputs=[ctx.fresh_name("rng_scaled")],
            ),
        )
        scaled.type = ir.TensorType(out_dtype_enum)
        _stamp_type_and_shape(scaled, out_shape)
        _ensure_value_metadata(ctx, scaled)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("rng_out")
        result = cast(
            ir.Value,
            ctx.builder.Add(
                scaled,
                lo_val,
                _outputs=[desired_name],
            ),
        )
        result.type = ir.TensorType(out_dtype_enum)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)

        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
