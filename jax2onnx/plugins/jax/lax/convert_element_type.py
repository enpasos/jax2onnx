# jax2onnx/plugins/jax/lax/convert_element_type.py

from __future__ import annotations

from typing import Any, cast

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._complex_utils import (
    COMPLEX_DTYPES,
    _base_dtype_for_complex,
    ensure_packed_real_pair,
    pack_known_real,
    split_packed_real_imag,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._loop_extent_meta import propagate_axis0_override
from jax2onnx.plugins._utils import cast_value_to_dtype
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive="convert_element_type",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.convert_element_type.html",
    onnx=[
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
        {"component": "Clip", "doc": "https://onnx.ai/onnx/operators/onnx__Clip.html"},
        {
            "component": "IsNaN",
            "doc": "https://onnx.ai/onnx/operators/onnx__IsNaN.html",
        },
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.2.0",
    context="primitives.lax",
    component="convert_element_type",
    testcases=[
        {
            "testcase": "convert_element_type",
            "callable": lambda x: x.astype(np.int16),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Cast:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class ConvertElementTypePlugin(PrimitiveLeafPlugin):
    """Lower explicit casts, preserving JAX's saturation for 8-bit integers."""

    @staticmethod
    def _saturate_float_operand(
        ctx: LoweringContextProtocol,
        operand: ir.Value,
        target_dtype: ir.DataType,
    ) -> ir.Value:
        if ctx.opset < 11:
            raise NotImplementedError(
                "Floating-to-8-bit convert_element_type requires opset >= 11 "
                "for Clip min/max inputs"
            )
        compute_dtype = operand.dtype
        if compute_dtype == ir.DataType.BFLOAT16:
            if ctx.opset < 13:
                raise NotImplementedError(
                    "bfloat16-to-8-bit convert_element_type requires opset >= 13 "
                    "for bfloat16 Cast inputs"
                )
            # CPU Clip and Where have no bfloat16 kernels. Every bfloat16
            # value is represented exactly by float32, including the bounds.
            compute_dtype = ir.DataType.FLOAT
            widened = ir.Value(
                name=ctx.fresh_name("convert_float32"),
                type=ir.TensorType(compute_dtype),
                shape=operand.shape,
            )
            operand = ctx.builder.op(
                "Cast", [operand], {"to": int(compute_dtype.value)}, output=widened
            )
        if compute_dtype not in {
            ir.DataType.FLOAT16,
            ir.DataType.FLOAT,
            ir.DataType.DOUBLE,
        }:
            raise TypeError(
                f"Floating conversion to {target_dtype} requires a supported "
                f"floating operand dtype, got {compute_dtype}"
            )

        def scalar(value: int) -> ir.Value:
            array = np.asarray(value, dtype=compute_dtype.numpy())
            return ctx.bind_const_for_var(object(), array)

        is_nan = ctx.builder.op(
            "IsNaN",
            [operand],
            output=ir.Value(
                name=ctx.fresh_name("convert_is_nan"),
                type=ir.TensorType(ir.DataType.BOOL),
                shape=operand.shape,
            ),
        )
        finite_or_infinite = ctx.builder.op(
            "Where",
            [is_nan, scalar(0), operand],
            output=ir.Value(
                name=ctx.fresh_name("convert_nan_to_zero"),
                type=ir.TensorType(compute_dtype),
                shape=operand.shape,
            ),
        )
        limits = np.iinfo(target_dtype.numpy())
        return ctx.builder.op(
            "Clip",
            [finite_or_infinite, scalar(int(limits.min)), scalar(int(limits.max))],
            output=ir.Value(
                name=ctx.fresh_name("convert_saturated"),
                type=ir.TensorType(compute_dtype),
                shape=operand.shape,
            ),
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("convert_in")
        )
        source_val = operand_val
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("convert_out")
        )

        target_dtype = numpy_dtype_to_ir(np.dtype(out_var.aval.dtype))
        source_dtype = np.dtype(operand_var.aval.dtype)
        source_is_complex = np.issubdtype(source_dtype, np.complexfloating)
        if target_dtype in COMPLEX_DTYPES:
            component_dtype = _base_dtype_for_complex(target_dtype)
            if source_is_complex:
                packed, _ = ensure_packed_real_pair(
                    ctx, operand_val, name_hint="convert_complex"
                )
                result = cast_value_to_dtype(
                    ctx, packed, component_dtype, name_hint="convert_components"
                )
            else:
                result = pack_known_real(
                    ctx, operand_val, component_dtype, name_hint="convert_real"
                )
            propagate_axis0_override(source_val, result)
            _ensure_value_metadata(ctx, result)
            ctx.bind_value_for_var(out_var, result)
            return

        if source_is_complex:
            packed, component_dtype = ensure_packed_real_pair(
                ctx, operand_val, name_hint="convert_complex"
            )
            operand_val, _ = split_packed_real_imag(
                ctx, packed, component_dtype, prefix="convert_real"
            )
            source_dtype = np.dtype(component_dtype.numpy())
        if target_dtype in {ir.DataType.INT8, ir.DataType.UINT8} and (
            np.issubdtype(source_dtype, np.floating) or source_dtype.name == "bfloat16"
        ):
            operand_val = self._saturate_float_operand(ctx, operand_val, target_dtype)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("convert_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("convert_out")

        result = cast(
            ir.Value,
            ctx.builder.Cast(
                operand_val,
                _outputs=[desired_name],
                to=int(target_dtype.value),
            ),
        )
        result.type = ir.TensorType(target_dtype)
        result.shape = operand_val.shape
        _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
        propagate_axis0_override(source_val, result)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
