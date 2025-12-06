# jax2onnx/plugins/jax/lax/fft.py

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Sequence

import numpy as np
from jax import lax

import onnx_ir as ir

from jax2onnx.plugins._complex_utils import (
    pack_native_complex,
    _shape_tuple,
    _base_dtype_for_complex,
    conjugate_packed_tensor,
)
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _resolve_fft_constants() -> (
    tuple[type[lax.FftType] | None, object, object, object | None, object | None]
):
    try:
        return (
            lax.FftType,
            lax.FftType.FFT,
            lax.FftType.IFFT,
            getattr(lax.FftType, "RFFT", None),
            getattr(lax.FftType, "IRFFT", None),
        )
    except AttributeError:  # pragma: no cover - older JAX
        return None, 0, 1, 2, 3


FFT_CONSTANTS: Final[
    tuple[
        type[lax.FftType] | None,
        object,
        object,
        object | None,
        object | None,
    ]
] = _resolve_fft_constants()

_LAX_FFT_ENUM: Final[type[lax.FftType] | None] = FFT_CONSTANTS[0]
_LAX_FFT_KIND_FFT: Final[object] = FFT_CONSTANTS[1]
_LAX_FFT_KIND_IFFT: Final[object] = FFT_CONSTANTS[2]
_LAX_FFT_KIND_RFFT: Final[object | None] = FFT_CONSTANTS[3]
_LAX_FFT_KIND_IRFFT: Final[object | None] = FFT_CONSTANTS[4]


def _normalize_fft_kind(fft_type: object) -> str | None:
    """Return normalized kind (FFT/IFFT/RFFT/IRFFT) when supported."""
    if _LAX_FFT_ENUM is not None and isinstance(fft_type, _LAX_FFT_ENUM):
        if fft_type == _LAX_FFT_KIND_FFT:
            return "FFT"
        if fft_type == _LAX_FFT_KIND_IFFT:
            return "IFFT"
        if _LAX_FFT_KIND_RFFT is not None and fft_type == _LAX_FFT_KIND_RFFT:
            return "RFFT"
        if _LAX_FFT_KIND_IRFFT is not None and fft_type == _LAX_FFT_KIND_IRFFT:
            return "IRFFT"
        return None
    name = getattr(fft_type, "name", None)
    if isinstance(name, str):
        upper = name.upper()
        if upper in {"FFT", "IFFT", "RFFT", "IRFFT"}:
            return upper
    if isinstance(fft_type, str):
        upper = fft_type.upper()
        if upper in {"FFT", "IFFT", "RFFT", "IRFFT"}:
            return upper
    if isinstance(fft_type, (int, np.integer)):
        value = int(fft_type)
        fft_val = (
            int(_LAX_FFT_KIND_FFT.value)
            if hasattr(_LAX_FFT_KIND_FFT, "value")
            else int(_LAX_FFT_KIND_FFT)
        )
        ifft_val = (
            int(_LAX_FFT_KIND_IFFT.value)
            if hasattr(_LAX_FFT_KIND_IFFT, "value")
            else int(_LAX_FFT_KIND_IFFT)
        )
        rfft_val = (
            int(_LAX_FFT_KIND_RFFT.value)
            if hasattr(_LAX_FFT_KIND_RFFT, "value")
            else int(_LAX_FFT_KIND_RFFT)
        )
        irfft_val = (
            int(_LAX_FFT_KIND_IRFFT.value)
            if hasattr(_LAX_FFT_KIND_IRFFT, "value")
            else int(_LAX_FFT_KIND_IRFFT)
        )
        if value == fft_val:
            return "FFT"
        if value == ifft_val:
            return "IFFT"
        if value == rfft_val:
            return "RFFT"
        if value == irfft_val:
            return "IRFFT"
    return None


def _normalize_fft_lengths(lengths: Sequence[int] | None) -> tuple[int, ...]:
    if not lengths:
        return ()
    return tuple(int(v) for v in lengths)


def _transform_axis(packed_dims: Sequence[object]) -> int:
    if len(packed_dims) < 2:
        return 0
    return len(packed_dims) - 2


def _is_static_int(dim: object) -> bool:
    return isinstance(dim, (int, np.integer))


def _maybe_int(dim: object) -> int | None:
    return int(dim) if _is_static_int(dim) else None


def _reshape_tensor(
    ctx: "IRContext", value: ir.Value, target_dims: Sequence[object], *, name_hint: str
) -> ir.Value:
    target_dims = tuple(target_dims)
    if not target_dims:
        return value
    if all(_is_static_int(d) for d in target_dims):
        shape_array = np.asarray(target_dims, dtype=np.int64)
        shape_init = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name(f"{name_hint}_shape"),
            array=shape_array,
        )
        shape_init.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(shape_init, (len(target_dims),))
        _ensure_value_metadata(ctx, shape_init)
        reshaped = ctx.builder.Reshape(
            value,
            shape_init,
            _outputs=[ctx.fresh_name(name_hint)],
        )
        reshaped.type = value.type
        _stamp_type_and_shape(reshaped, target_dims)
        _ensure_value_metadata(ctx, reshaped)
        return reshaped
    value.shape = ir.Shape(target_dims)
    _ensure_value_metadata(ctx, value)
    return value


def _make_scalar_i64(ctx: "IRContext", value: int, *, name_hint: str) -> ir.Value:
    scalar = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(name_hint),
        array=np.asarray(value, dtype=np.int64),
    )
    scalar.type = ir.TensorType(ir.DataType.INT64)
    _stamp_type_and_shape(scalar, ())
    _ensure_value_metadata(ctx, scalar)
    return scalar


def _gather_channel(
    ctx: "IRContext",
    tensor: ir.Value,
    *,
    axis: int,
    index: int,
    dtype: ir.DataType,
    name_hint: str,
) -> ir.Value:
    idx_init = _make_scalar_i64(ctx, index, name_hint=f"{name_hint}_idx")
    gathered = ctx.builder.Gather(
        tensor,
        idx_init,
        axis=axis,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    gathered.type = ir.TensorType(dtype)
    dims = list(_shape_tuple(tensor))
    if dims:
        try:
            dims.pop(axis)
        except IndexError:
            dims = []
    _stamp_type_and_shape(gathered, tuple(dims))
    _ensure_value_metadata(ctx, gathered)
    return gathered


@register_primitive(
    jaxpr_primitive=lax.fft_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.fft.html",
    onnx=[{"component": "DFT", "doc": "https://onnx.ai/onnx/operators/onnx__DFT.html"}],
    since="v0.10.1",
    context="primitives.lax",
    component="fft",
    testcases=[
        {
            "testcase": "fft_complex64_1d",
            "callable": lambda x: lax.fft(x, "FFT", (4,)),
            "input_values": [
                np.array(
                    [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j],
                    dtype=np.complex64,
                )
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "fft_complex64_len8",
            "callable": lambda x: lax.fft(x, "FFT", (8,)),
            "input_values": [
                np.array(
                    [
                        0.0 + 0.0j,
                        1.0 + 1.0j,
                        2.0 + 0.0j,
                        0.0 - 3.0j,
                        -1.0 + 0.5j,
                        0.25 - 0.75j,
                        0.5 + 0.0j,
                        -0.125 + 0.25j,
                    ],
                    dtype=np.complex64,
                )
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "fft_complex64_batch",
            "callable": lambda x: lax.fft(x, "FFT", (4,)),
            "input_values": [
                np.array(
                    [
                        [0.5 + 0.0j, -1.0 + 2.0j, 0.25 - 0.75j, 3.0 + 0.0j],
                        [1.0 - 1.5j, 0.0 + 0.5j, -2.5 + 1.25j, 0.75 - 0.25j],
                    ],
                    dtype=np.complex64,
                )
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "fft_complex128_1d",
            "callable": lambda x: lax.fft(x, "FFT", (4,)),
            "input_values": [
                np.array(
                    [1.0 + 0.0j, -2.0 + 2.0j, 0.5 - 1.5j, 3.0 + 0.25j],
                    dtype=np.complex128,
                )
            ],
            "expected_output_dtypes": [np.float64],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f64_variant": True,
        },
        {
            "testcase": "fft_complex128_len8",
            "callable": lambda x: lax.fft(x, "FFT", (8,)),
            "input_values": [
                np.array(
                    [
                        0.0 + 0.0j,
                        1.5 - 0.5j,
                        -2.0 + 3.0j,
                        0.25 + 1.0j,
                        -0.75 - 0.75j,
                        2.0 + 0.0j,
                        -1.0 + 0.5j,
                        0.5 - 1.25j,
                    ],
                    dtype=np.complex128,
                )
            ],
            "expected_output_dtypes": [np.float64],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f64_variant": True,
        },
        {
            "testcase": "ifft_complex64_1d",
            "callable": lambda x: lax.fft(x, "IFFT", (4,)),
            "input_values": [
                np.array(
                    [1.0 + 0.0j, -2.0 + 2.0j, 0.5 - 1.5j, 3.0 + 0.25j],
                    dtype=np.complex64,
                )
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "ifft_complex64_len8",
            "callable": lambda x: lax.fft(x, "IFFT", (8,)),
            "input_values": [
                np.array(
                    [
                        0.0 + 0.0j,
                        -1.5 + 1.0j,
                        2.0 - 0.5j,
                        -0.25 + 0.75j,
                        1.25 - 1.5j,
                        -2.0 + 0.25j,
                        0.5 + 0.5j,
                        -1.0 - 2.0j,
                    ],
                    dtype=np.complex64,
                )
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "ifft_complex128_batch",
            "callable": lambda x: lax.fft(x, "IFFT", (4,)),
            "input_values": [
                np.array(
                    [
                        [0.5 + 0.0j, -1.0 + 2.0j, 0.25 - 0.75j, 3.0 + 0.0j],
                        [1.0 - 1.5j, 0.0 + 0.5j, -2.5 + 1.25j, 0.75 - 0.25j],
                    ],
                    dtype=np.complex128,
                )
            ],
            "expected_output_dtypes": [np.float64],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f64_variant": True,
        },
        {
            "testcase": "rfft_real32_1d",
            "callable": lambda x: lax.fft(x, "RFFT", (4,)),
            "input_values": [
                np.array([0.0, 1.0, -2.0, 3.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "rfft_real64_len8",
            "callable": lambda x: lax.fft(x, "RFFT", (8,)),
            "input_values": [
                np.array(
                    [0.5, -1.0, 2.0, -3.5, 4.0, -0.25, 1.5, -2.5],
                    dtype=np.float64,
                ),
            ],
            "expected_output_dtypes": [np.float64],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f64_variant": True,
        },
        {
            "testcase": "irfft_complex64_1d",
            "callable": lambda x: lax.fft(x, "IRFFT", (8,)),
            "input_values": [
                np.array(
                    [
                        1.0 + 0.0j,
                        -2.0 + 1.5j,
                        0.5 - 0.25j,
                        3.0 + 0.75j,
                        -1.0 + 0.0j,
                    ],
                    dtype=np.complex64,
                ),
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 8.0}},
                        "path": "Reshape:1x5x2 -> Concat:1x8x2 -> DFT -> Gather -> Reshape:1x8 -> Reshape:8",
                    }
                ],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "irfft_complex128_len8",
            "callable": lambda x: lax.fft(x, "IRFFT", (8,)),
            "input_values": [
                np.array(
                    [
                        0.25 + 0.0j,
                        -1.5 + 0.5j,
                        2.0 - 1.25j,
                        -0.75 + 0.75j,
                        0.5 + 0.0j,
                    ],
                    dtype=np.complex128,
                ),
            ],
            "expected_output_dtypes": [np.float64],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 8.0}},
                        "path": "Reshape:1x5x2 -> Concat:1x8x2 -> DFT -> Gather -> Reshape:1x8 -> Reshape:8",
                    }
                ],
                no_unused_inputs=True,
            ),
            "run_only_f64_variant": True,
        },
    ],
)
class FFTPlugin(PrimitiveLeafPlugin):
    """Lower `lax.fft` primitives to ONNX DFT."""

    def lower(self, ctx: "IRContext", eqn):
        (x_var,) = eqn.invars
        out_var = eqn.outvars[0]
        fft_type = eqn.params.get("fft_type")
        fft_lengths = _normalize_fft_lengths(eqn.params.get("fft_lengths"))

        kind = _normalize_fft_kind(fft_type)
        if kind not in {"FFT", "IFFT", "RFFT", "IRFFT"}:
            raise NotImplementedError(f"Unsupported FFT kind: {fft_type!r}")
        if len(fft_lengths) not in (0, 1):
            raise NotImplementedError(
                "Only 1D FFT with at most one explicit length is supported."
            )

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("fft_input"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("fft_output")
        )

        if kind in {"FFT", "IFFT"}:
            if x_val.dtype not in (ir.DataType.COMPLEX64, ir.DataType.COMPLEX128):
                raise NotImplementedError(
                    "FFT/IFFT require complex64 or complex128 input."
                )
            result, result_dtype = self._lower_complex_fft(
                ctx,
                x_val,
                fft_lengths,
                inverse=(kind == "IFFT"),
            )
        elif kind == "RFFT":
            if x_val.dtype not in (ir.DataType.FLOAT, ir.DataType.DOUBLE):
                raise NotImplementedError("RFFT requires float32/float64 input.")
            result, result_dtype = self._lower_rfft(ctx, x_val, fft_lengths)
        else:  # IRFFT
            if x_val.dtype not in (ir.DataType.COMPLEX64, ir.DataType.COMPLEX128):
                raise NotImplementedError(
                    "IRFFT requires complex64 or complex128 input."
                )
            result, result_dtype = self._lower_irfft(ctx, x_val, fft_lengths)

        if hasattr(out_spec, "shape") and getattr(result, "shape", None) is not None:
            out_spec.shape = result.shape
        if hasattr(out_spec, "type"):
            out_spec.type = ir.TensorType(result_dtype)

        ctx.bind_value_for_var(out_var, result)

    def _lower_complex_fft(
        self,
        ctx: "IRContext",
        x_val: ir.Value,
        fft_lengths: tuple[int, ...],
        *,
        inverse: bool,
    ) -> tuple[ir.Value, ir.DataType]:
        base_dtype = _base_dtype_for_complex(x_val.dtype)
        packed = pack_native_complex(ctx, x_val, name_hint="fft")
        original_dims = _shape_tuple(packed)

        packed_dims = original_dims
        inserted_batch = False
        if len(packed_dims) == 2:
            packed = _reshape_tensor(
                ctx,
                packed,
                (1,) + tuple(packed_dims),
                name_hint="fft_batch",
            )
            packed_dims = (1,) + tuple(packed_dims)
            inserted_batch = True

        dft_inputs: list[ir.Value] = [packed]
        if fft_lengths:
            dft_inputs.append(
                _make_scalar_i64(ctx, int(fft_lengths[0]), name_hint="fft_length")
            )
        axis_index = _transform_axis(packed_dims)
        dft_inputs.append(_make_scalar_i64(ctx, axis_index, name_hint="fft_axis"))

        dft_out = ctx.builder.DFT(
            *dft_inputs,
            _outputs=[ctx.fresh_name("fft_pair")],
            inverse=1 if inverse else 0,
            onesided=0,
        )
        dft_out.type = ir.TensorType(base_dtype)
        if all(_is_static_int(d) for d in packed_dims):
            dft_out.shape = ir.Shape(packed_dims)
            _stamp_type_and_shape(dft_out, packed_dims)
        _ensure_value_metadata(ctx, dft_out)

        result = dft_out
        if inserted_batch:
            result = _reshape_tensor(
                ctx,
                result,
                (1,) + tuple(original_dims),
                name_hint="fft_unbatch_pre",
            )
            result = _reshape_tensor(
                ctx,
                result,
                original_dims,
                name_hint="fft_unbatch",
            )
        else:
            if all(_is_static_int(d) for d in original_dims):
                _stamp_type_and_shape(result, original_dims)
                result.shape = ir.Shape(original_dims)
                _ensure_value_metadata(ctx, result)

        return result, base_dtype

    def _lower_rfft(
        self,
        ctx: "IRContext",
        x_val: ir.Value,
        fft_lengths: tuple[int, ...],
    ) -> tuple[ir.Value, ir.DataType]:
        base_dtype = x_val.dtype
        real_dims = _shape_tuple(x_val)
        if not real_dims:
            raise NotImplementedError("RFFT requires at least one signal dimension.")

        packed = _reshape_tensor(
            ctx,
            x_val,
            tuple(real_dims) + (1,),
            name_hint="rfft_pack",
        )
        packed.type = ir.TensorType(base_dtype)
        packed_dims = _shape_tuple(packed)

        inserted_batch = False
        if len(packed_dims) == 2:
            packed = _reshape_tensor(
                ctx,
                packed,
                (1,) + tuple(packed_dims),
                name_hint="rfft_batch",
            )
            packed_dims = (1,) + tuple(packed_dims)
            inserted_batch = True

        dft_inputs: list[ir.Value] = [packed]
        axis_index = _transform_axis(packed_dims)
        signal_length = (
            int(fft_lengths[0]) if fft_lengths else _maybe_int(real_dims[-1])
        )
        if fft_lengths:
            dft_inputs.append(
                _make_scalar_i64(ctx, signal_length, name_hint="rfft_length")
            )
        dft_inputs.append(_make_scalar_i64(ctx, axis_index, name_hint="rfft_axis"))

        dft_out = ctx.builder.DFT(
            *dft_inputs,
            _outputs=[ctx.fresh_name("rfft_pair")],
            inverse=0,
            onesided=1,
        )
        dft_out.type = ir.TensorType(base_dtype)
        _ensure_value_metadata(ctx, dft_out)

        onesided = signal_length // 2 + 1 if signal_length is not None else None
        output_dims = list(real_dims[:-1])
        output_dims.append(onesided if onesided is not None else real_dims[-1])
        output_dims.append(2)

        result = dft_out
        if inserted_batch and onesided is not None:
            result = _reshape_tensor(
                ctx,
                result,
                (1,) + tuple(output_dims),
                name_hint="rfft_unbatch_pre",
            )
            result = _reshape_tensor(
                ctx,
                result,
                output_dims,
                name_hint="rfft_unbatch",
            )
        elif onesided is not None and all(_is_static_int(d) for d in output_dims):
            _stamp_type_and_shape(result, tuple(output_dims))
            result.shape = ir.Shape(tuple(output_dims))
            _ensure_value_metadata(ctx, result)

        result.type = ir.TensorType(base_dtype)
        return result, base_dtype

    def _lower_irfft(
        self,
        ctx: "IRContext",
        x_val: ir.Value,
        fft_lengths: tuple[int, ...],
    ) -> tuple[ir.Value, ir.DataType]:
        if not fft_lengths:
            raise NotImplementedError(
                "IRFFT currently requires an explicit fft_lengths entry."
            )
        base_dtype = _base_dtype_for_complex(x_val.dtype)
        target_len = int(fft_lengths[0])

        complex_dims = _shape_tuple(x_val)
        packed = pack_native_complex(ctx, x_val, name_hint="irfft_pack")
        packed_dims = _shape_tuple(packed)

        inserted_batch = False
        if len(packed_dims) == 2:
            packed = _reshape_tensor(
                ctx,
                packed,
                (1,) + tuple(packed_dims),
                name_hint="irfft_batch",
            )
            packed_dims = (1,) + tuple(packed_dims)
            inserted_batch = True

        axis_index = _transform_axis(packed_dims)
        onesided_len = _maybe_int(packed_dims[axis_index])
        if onesided_len is None:
            raise NotImplementedError("IRFFT requires static onesided length.")

        # Reconstruct the full spectrum for inverse transform.
        if target_len > onesided_len:
            if target_len % 2 == 0:
                interior_start = 1
                interior_end = onesided_len - 1
            else:
                interior_start = 1
                interior_end = onesided_len
            mirror_count = max(0, interior_end - interior_start)
            if mirror_count > 0:
                mirror_indices = np.arange(
                    interior_end - 1, interior_start - 1, -1, dtype=np.int64
                )
                mirror_init = ctx.builder.add_initializer_from_array(
                    name=ctx.fresh_name("irfft_mirror_indices"),
                    array=mirror_indices,
                )
                mirror_init.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(mirror_init, (mirror_count,))
                _ensure_value_metadata(ctx, mirror_init)
                mirror_vals = ctx.builder.Gather(
                    packed,
                    mirror_init,
                    axis=axis_index,
                    _outputs=[ctx.fresh_name("irfft_mirror")],
                )
                mirror_vals.type = ir.TensorType(base_dtype)
                mirror_dims = list(packed_dims)
                mirror_dims[axis_index] = mirror_count
                _stamp_type_and_shape(mirror_vals, tuple(mirror_dims))
                _ensure_value_metadata(ctx, mirror_vals)

                mirror_conj = conjugate_packed_tensor(
                    ctx,
                    mirror_vals,
                    base_dtype,
                    prefix="irfft_mirror",
                )

                full_dims = list(packed_dims)
                full_dims[axis_index] = target_len
                packed = ctx.builder.Concat(
                    packed,
                    mirror_conj,
                    axis=axis_index,
                    _outputs=[ctx.fresh_name("irfft_full_spectrum")],
                )
                packed.type = ir.TensorType(base_dtype)
                _stamp_type_and_shape(packed, tuple(full_dims))
                _ensure_value_metadata(ctx, packed)
                packed_dims = tuple(full_dims)
        else:
            # Already full spectrum.
            pass

        dft_inputs: list[ir.Value] = [packed]
        dft_inputs.append(_make_scalar_i64(ctx, target_len, name_hint="irfft_length"))
        dft_inputs.append(_make_scalar_i64(ctx, axis_index, name_hint="irfft_axis"))

        dft_out = ctx.builder.DFT(
            *dft_inputs,
            _outputs=[ctx.fresh_name("irfft_pair")],
            inverse=1,
            onesided=0,
        )
        dft_out.type = ir.TensorType(base_dtype)
        _ensure_value_metadata(ctx, dft_out)

        real_vals = _gather_channel(
            ctx,
            dft_out,
            axis=len(packed_dims) - 1,
            index=0,
            dtype=base_dtype,
            name_hint="irfft_real",
        )

        real_dims = list(complex_dims)
        if real_dims:
            real_dims[-1] = target_len

        result = real_vals
        if inserted_batch:
            result = _reshape_tensor(
                ctx,
                result,
                (1,) + tuple(real_dims),
                name_hint="irfft_unbatch_pre",
            )
            result = _reshape_tensor(
                ctx,
                result,
                real_dims,
                name_hint="irfft_unbatch",
            )
        else:
            if all(_is_static_int(d) for d in real_dims):
                _stamp_type_and_shape(result, tuple(real_dims))
                result.shape = ir.Shape(tuple(real_dims))
                _ensure_value_metadata(ctx, result)

        result.type = ir.TensorType(base_dtype)
        return result, base_dtype
