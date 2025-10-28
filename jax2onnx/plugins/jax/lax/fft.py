# jax2onnx/plugins/jax/lax/fft.py

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from jax import lax

import onnx_ir as ir

from jax2onnx.plugins._complex_utils import (
    ensure_complex_dtype,
    pack_native_complex,
    unpack_to_native_complex,
    _shape_tuple,
)
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _normalize_fft_lengths(lengths: Sequence[int] | None) -> tuple[int, ...]:
    if not lengths:
        return ()
    return tuple(int(v) for v in lengths)


def _transform_axis(packed_dims: Sequence[object]) -> int:
    if len(packed_dims) < 2:
        return 0
    return len(packed_dims) - 2


@register_primitive(
    jaxpr_primitive=lax.fft_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.fft.html",
    onnx=[{"component": "DFT", "doc": "https://onnx.ai/onnx/operators/onnx__DFT.html"}],
    since="v0.10.0",
    context="primitives.lax",
    component="fft",
    testcases=[
        {
            "testcase": "fft_complex64_1d",
            "callable": lambda x: lax.fft(x, lax.FftType.FFT, (4,)),
            "input_values": [
                np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j], dtype=np.complex64)
            ],
            "expected_output_dtypes": [np.complex64],
            "skip_numeric_validation": True,  # ORT CPU lacks Real/Imag kernels for complex inputs
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                    {"path": "Complex", "counts": {"Complex": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        }
    ],
)
class FFTPlugin(PrimitiveLeafPlugin):
    """Lower `lax.fft` (complex-to-complex, 1D) to ONNX DFT."""

    def lower(self, ctx: "IRContext", eqn):
        (x_var,) = eqn.invars
        out_var = eqn.outvars[0]
        fft_type = eqn.params.get("fft_type")
        fft_lengths = _normalize_fft_lengths(eqn.params.get("fft_lengths"))

        if fft_type is not lax.FftType.FFT:
            raise NotImplementedError("Only complex-to-complex FFT is supported currently.")
        if len(fft_lengths) not in (0, 1):
            raise NotImplementedError("Only 1D FFT with a single length is supported currently.")

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("fft_input"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("fft_output"))

        input_dtype = x_val.dtype
        if input_dtype not in (ir.DataType.COMPLEX64, ir.DataType.COMPLEX128):
            raise NotImplementedError("FFT plugin currently supports complex inputs only.")

        packed = pack_native_complex(ctx, x_val, name_hint="fft")
        packed_dims = _shape_tuple(packed)
        axis = _transform_axis(packed_dims)

        dft_inputs: list[ir.Value] = [packed]
        if fft_lengths:
            length_tensor = np.asarray(fft_lengths[0], dtype=np.int64)
            length_val = ctx.builder.add_initializer_from_array(
                name=ctx.fresh_name("fft_length"), array=length_tensor
            )
            length_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(length_val, ())
            _ensure_value_metadata(ctx, length_val)
            dft_inputs.append(length_val)

        dft_pair = ctx.builder.DFT(
            *dft_inputs,
            _outputs=[ctx.fresh_name("fft_pair")],
            axis=axis,
        )
        if packed.dtype is not None:
            dft_pair.type = ir.TensorType(packed.dtype)
        if getattr(packed, "shape", None) is not None:
            dft_pair.shape = packed.shape
            _ensure_value_metadata(ctx, dft_pair)

        complex_out = unpack_to_native_complex(
            ctx,
            dft_pair,
            name_hint="fft",
            target_dtype=input_dtype,
        )
        complex_out = ensure_complex_dtype(
            ctx,
            complex_out,
            input_dtype,
            name_hint="fft_cast",
        )

        if getattr(out_spec, "shape", None) is not None:
            complex_out.shape = out_spec.shape
        if getattr(out_spec, "type", None) is not None:
            complex_out.type = out_spec.type

        ctx.bind_value_for_var(out_var, complex_out)
