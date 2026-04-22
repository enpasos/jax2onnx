# jax2onnx/plugins/jax/lax/rng_bit_generator.py

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.rng_bit_generator_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.rng_bit_generator.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        },
        {
            "component": "RandomUniform",
            "doc": "https://onnx.ai/onnx/operators/onnx__RandomUniform.html",
        },
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
    ],
    since="0.12.1",
    context="primitives.lax",
    component="rng_bit_generator",
    testcases=[
        {
            "testcase": "rng_bit_generator_u32",
            "callable": lambda key: jax.lax.rng_bit_generator(
                key, shape=(2, 3), dtype=np.uint32
            ),
            "input_values": [np.asarray([0, 1], dtype=np.uint32)],
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                ["Identity", "RandomUniform -> Cast"],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
    ],
)
class RngBitGeneratorPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.rng_bit_generator`` with stateless ONNX random generation."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        key_var = eqn.invars[0]
        out_key_var, out_bits_var = eqn.outvars
        params = dict(getattr(eqn, "params", {}) or {})

        key_val = ctx.get_value_for_var(
            key_var, name_hint=ctx.fresh_name("rngbg_key_in")
        )
        out_key_spec = ctx.get_value_for_var(
            out_key_var, name_hint=ctx.fresh_name("rngbg_key_out")
        )
        out_bits_spec = ctx.get_value_for_var(
            out_bits_var, name_hint=ctx.fresh_name("rngbg_bits_out")
        )

        key_name = getattr(out_key_spec, "name", None) or ctx.fresh_name("rngbg_key")
        key_out = ctx.builder.Identity(key_val, _outputs=[key_name])
        if getattr(out_key_spec, "type", None) is not None:
            key_out.type = out_key_spec.type
        if getattr(out_key_spec, "shape", None) is not None:
            key_out.shape = out_key_spec.shape
        _ensure_value_metadata(ctx, key_out)

        bits_shape = tuple(
            int(d)
            for d in (
                params.get("shape", getattr(out_bits_var.aval, "shape", ())) or ()
            )
        )
        if any(d < 0 for d in bits_shape):
            raise NotImplementedError(
                "rng_bit_generator requires static non-negative output shape"
            )

        bits_np_dtype = np.dtype(
            params.get("dtype", getattr(getattr(out_bits_var, "aval", None), "dtype"))
        )
        bits_dtype_enum = numpy_dtype_to_ir(bits_np_dtype)

        unit = ctx.builder.RandomUniform(
            shape=bits_shape,
            dtype=int(ir.DataType.FLOAT.value),
            low=0.0,
            high=1.0,
            _outputs=[ctx.fresh_name("rngbg_unit")],
        )
        unit.type = ir.TensorType(ir.DataType.FLOAT)
        _stamp_type_and_shape(unit, bits_shape)
        _ensure_value_metadata(ctx, unit)

        bits_name = getattr(out_bits_spec, "name", None) or ctx.fresh_name("rngbg_bits")
        bits_out = ctx.builder.Cast(
            unit,
            to=int(bits_dtype_enum.value),
            _outputs=[bits_name],
        )
        bits_out.type = ir.TensorType(bits_dtype_enum)
        _stamp_type_and_shape(bits_out, bits_shape)
        _ensure_value_metadata(ctx, bits_out)

        if getattr(out_bits_spec, "type", None) is not None:
            bits_out.type = out_bits_spec.type
        if getattr(out_bits_spec, "shape", None) is not None:
            bits_out.shape = out_bits_spec.shape

        ctx.bind_value_for_var(out_key_var, key_out)
        ctx.bind_value_for_var(out_bits_var, bits_out)
