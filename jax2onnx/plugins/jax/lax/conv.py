# jax2onnx/plugins/jax/lax/conv.py

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Sequence, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._complex_utils import (
    COMPLEX_DTYPES,
    cast_real_tensor,
    ensure_packed_real_pair,
    is_packed_complex_tensor,
    pack_real_imag_pair,
    resolve_common_real_dtype,
    split_packed_real_imag,
    coerce_dim_values,
)
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_LAYOUT_MAP: Final[dict[tuple[int, ...] | str, str]] = {
    (0, 1, 2, 3): "NCHW",
    (0, 3, 1, 2): "NHWC",
    "NCHW": "NCHW",
    "NHWC": "NHWC",
}
_FILTER_LAYOUT_MAP: Final[dict[tuple[int, ...] | str, str]] = {
    (0, 1, 2, 3): "OIHW",
    (3, 2, 0, 1): "HWIO",
    "OIHW": "OIHW",
    "HWIO": "HWIO",
}
_OUTPUT_LAYOUT_MAP: Final[dict[tuple[int, ...] | str, str]] = {
    (0, 1, 2, 3): "NCHW",
    (0, 3, 1, 2): "NHWC",
    "NCHW": "NCHW",
    "NHWC": "NHWC",
}


def _layout_from_spec(spec, mapping) -> str:
    key = spec
    if isinstance(spec, str):
        key = spec.upper()
    return mapping.get(key)


def _perm(src_layout: str, dst_layout: str) -> list[int]:
    return [src_layout.index(axis) for axis in dst_layout]


def _flatten_padding(pads: Sequence[Sequence[int]]) -> list[int]:
    befores = [int(before) for before, _ in pads]
    afters = [int(after) for _, after in pads]
    return befores + afters


def _flip_spatial_dims(
    ctx: "IRContext",
    val: ir.Value,
    shape: tuple[int, ...],
    layout: str,
    name_hint: str,
) -> ir.Value:
    spatial_axes = [i for i, c in enumerate(layout) if c not in "OI"]
    if not spatial_axes:
        return val

    starts = [shape[i] - 1 for i in spatial_axes]
    ends = [-(2**63)] * len(spatial_axes)
    axes = spatial_axes
    steps = [-1] * len(spatial_axes)

    starts_val = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(f"{name_hint}_starts"),
        array=np.array(starts, dtype=np.int64),
    )
    ends_val = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(f"{name_hint}_ends"),
        array=np.array(ends, dtype=np.int64),
    )
    axes_val = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(f"{name_hint}_axes"),
        array=np.array(axes, dtype=np.int64),
    )
    steps_val = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(f"{name_hint}_steps"),
        array=np.array(steps, dtype=np.int64),
    )

    flipped = ctx.builder.Slice(
        val,
        starts_val,
        ends_val,
        axes_val,
        steps_val,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    flipped.type = ir.TensorType(val.dtype)
    _stamp_type_and_shape(flipped, shape)
    _ensure_value_metadata(ctx, flipped)
    return flipped


@register_primitive(
    jaxpr_primitive=jax.lax.conv_general_dilated_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv.html",
    onnx=[
        {
            "component": "Conv",
            "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="conv",
    testcases=[
        {
            "testcase": "conv",
            "callable": lambda x, w: jax.lax.conv(
                x, w, window_strides=(1, 1), padding="VALID"
            ),
            "input_shapes": [(1, 2, 3, 3), (1, 2, 2, 2)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Conv:1x1x2x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv2",
            "callable": lambda x, w: jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1, 1),
                padding="VALID",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            ),
            "input_shapes": [(1, 3, 3, 2), (2, 2, 2, 1)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Transpose:1x2x3x3 -> Conv:1x1x2x2 -> Transpose:1x2x2x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv_nchw",
            "callable": lambda x, w: jax.lax.conv(
                x, w, window_strides=(1, 1), padding="VALID"
            ),
            "input_shapes": [(1, 2, 5, 5), (3, 2, 3, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Conv:1x3x3x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv_nhwc",
            "callable": lambda x, w: jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            ),
            "input_shapes": [(1, 5, 5, 3), (3, 3, 3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Transpose:1x3x5x5 -> Conv:1x4x5x5 -> Transpose:1x5x5x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv_general_dilated_nhwc_output",
            "callable": lambda x, k: jax.lax.conv_general_dilated(
                x,
                k,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            ),
            "input_values": [
                np.ones((1, 5, 5, 3), dtype=np.float32),
                np.ones((2, 2, 3, 4), dtype=np.float32),
            ],
            "expected_output_shapes": [(1, 5, 5, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Transpose:1x3x5x5 -> Conv:1x4x5x5 -> Transpose:1x5x5x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv_complex64",
            "callable": lambda x, w: jax.lax.conv(
                x, w, window_strides=(1, 1), padding="VALID"
            ),
            "input_values": [
                np.array(
                    [[[[1.0 + 0.5j, -0.25 + 1.0j], [0.75 - 0.5j, 1.5 + 0.25j]]]],
                    dtype=np.complex64,
                ),
                np.array(
                    [[[[0.5 - 1.0j, 1.0 + 0.75j], [-0.75 + 0.5j, 0.25 - 1.5j]]]],
                    dtype=np.complex64,
                ),
            ],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {"path": "Conv", "counts": {"Conv": 4}},
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv_complex64_nhwc",
            "callable": lambda x, w: jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1, 1),
                padding="VALID",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            ),
            "input_values": [
                np.array(
                    [
                        [
                            [1.0 + 0.5j, -0.25 + 0.75j],
                            [0.5 - 1.0j, 1.25 + 0.25j],
                        ],
                        [
                            [-0.5 + 0.5j, 0.75 - 0.25j],
                            [1.5 + 0.75j, -1.0 + 0.5j],
                        ],
                    ],
                    dtype=np.complex64,
                ).reshape(1, 2, 2, 2),
                np.array(
                    [
                        0.5 - 0.5j,
                        0.75 + 0.25j,
                        1.0 + 0.5j,
                        -0.25 + 1.0j,
                    ],
                    dtype=np.complex64,
                ).reshape(1, 1, 2, 2),
            ],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {"path": "Conv", "counts": {"Conv": 4}},
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv_complex128_grouped",
            "callable": lambda x, w: jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1, 1),
                padding="VALID",
                feature_group_count=2,
            ),
            "input_values": [
                np.array(
                    [
                        1.0 + 0.5j,
                        -0.75 + 0.25j,
                        0.5 - 1.25j,
                        1.25 + 0.75j,
                        0.75 + 0.5j,
                        -0.25 - 0.5j,
                        1.0 + 0.25j,
                        -1.5 + 1.0j,
                        -0.5 + 1.5j,
                        0.25 - 0.75j,
                        0.5 + 0.5j,
                        -0.25 + 1.0j,
                        1.5 - 0.5j,
                        -1.0 + 0.5j,
                        0.75 + 0.25j,
                        0.5 - 1.5j,
                    ],
                    dtype=np.complex128,
                ).reshape(1, 4, 2, 2),
                np.array(
                    [
                        0.5 + 0.75j,
                        -0.25 + 0.5j,
                        1.0 - 0.5j,
                        0.75 + 1.0j,
                        -0.5 + 0.25j,
                        1.25 - 0.75j,
                        0.5 + 1.25j,
                        -0.75 + 0.5j,
                    ],
                    dtype=np.complex128,
                ).reshape(4, 2, 1, 1),
            ],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {"path": "Conv", "counts": {"Conv": 4}},
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ConvGeneralDilatedPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.conv_general_dilated`` to ONNX ``Conv`` (2D, NHWC/NCHW)."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lhs_var, rhs_var = eqn.invars[:2]
        out_var = eqn.outvars[0]

        params = getattr(eqn, "params", {})
        dimension_numbers = params.get("dimension_numbers")
        if dimension_numbers is None:
            raise ValueError("conv_general_dilated missing dimension_numbers")

        lhs_spec, rhs_spec, out_spec = dimension_numbers
        lhs_layout = _layout_from_spec(lhs_spec, _LAYOUT_MAP)
        rhs_layout = _layout_from_spec(rhs_spec, _FILTER_LAYOUT_MAP)
        out_layout = _layout_from_spec(out_spec, _OUTPUT_LAYOUT_MAP)
        if lhs_layout is None or rhs_layout is None or out_layout is None:
            raise NotImplementedError(
                f"Unsupported conv layouts: lhs={lhs_spec}, rhs={rhs_spec}, out={out_spec}"
            )

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("conv_lhs"))
        rhs_val = ctx.get_value_for_var(rhs_var, name_hint=ctx.fresh_name("conv_rhs"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("conv_out"))

        lhs_shape = tuple(getattr(lhs_var.aval, "shape", ()))
        rhs_shape = tuple(getattr(rhs_var.aval, "shape", ()))
        out_shape = tuple(getattr(out_var.aval, "shape", ()))

        conv_kwargs: dict[str, object] = {}
        strides = params.get("window_strides", (1, 1))
        conv_kwargs["strides"] = [int(s) for s in strides]

        lhs_dilation = params.get("lhs_dilation")
        is_transpose = lhs_dilation and any(d > 1 for d in lhs_dilation)
        op_type = "ConvTranspose" if is_transpose else "Conv"
        target_kernel_layout = "IOHW" if is_transpose else "OIHW"

        padding = params.get("padding", "VALID")
        if isinstance(padding, str):
            pad_mode = padding.upper()
            if pad_mode in ("SAME", "SAME_UPPER"):
                conv_kwargs["auto_pad"] = "SAME_UPPER"
            elif pad_mode == "VALID":
                conv_kwargs["pads"] = [0, 0, 0, 0]
            else:
                raise NotImplementedError(f"Unsupported padding mode {padding}")
        else:
            num_spatial = max(len(lhs_shape) - 2, 0)
            if padding is None:
                pad_pairs: Sequence[Sequence[int]] = tuple(
                    (0, 0) for _ in range(num_spatial)
                )
            else:
                if not isinstance(padding, Sequence):
                    raise TypeError(f"Unsupported padding spec type: {type(padding)!r}")
                padding_seq = tuple(padding)
                if not padding_seq:
                    pad_pairs = tuple((0, 0) for _ in range(num_spatial))
                else:
                    first_entry = padding_seq[0]
                    if not isinstance(first_entry, Sequence):
                        raise NotImplementedError(
                            "Expected padding as sequence of (low, high) pairs"
                        )
                    pad_pairs = cast(Sequence[Sequence[int]], padding_seq)
            conv_kwargs["pads"] = _flatten_padding(pad_pairs)

        rhs_dilation = params.get("rhs_dilation")
        if rhs_dilation:
            conv_kwargs["dilations"] = [int(d) for d in rhs_dilation]

        if is_transpose and "pads" in conv_kwargs:
            # JAX conv_general_dilated padding for transpose conv is "input padding" (on dilated input),
            # while ONNX ConvTranspose pads are "output padding" (reducing output size).
            # Formula: pad_onnx = kernel_effective - 1 - pad_jax
            pads_jax = conv_kwargs["pads"]
            num_spatial = len(pads_jax) // 2
            pads_jax_starts = pads_jax[:num_spatial]
            pads_jax_ends = pads_jax[num_spatial:]

            (
                rhs_shape[2:] if rhs_layout == "OIHW" else rhs_shape[:2]
            )  # Assuming 2D spatial
            # Actually we should use rhs_layout to find spatial dims.
            # But we can infer from len(pads_jax).
            # rhs_shape is N-D.
            # If layout is OIHW, spatial is [2:].
            # If layout is HWIO, spatial is [:-2].
            spatial_dims_indices = [
                i for i, c in enumerate(rhs_layout) if c not in "OI"
            ]
            kernel_spatial = [rhs_shape[i] for i in spatial_dims_indices]

            dilations = conv_kwargs.get("dilations", [1] * num_spatial)
            kernel_effective = [
                (k - 1) * d + 1 for k, d in zip(kernel_spatial, dilations)
            ]

            pads_onnx_starts = [
                k - 1 - p for k, p in zip(kernel_effective, pads_jax_starts)
            ]
            pads_onnx_ends = [
                k - 1 - p for k, p in zip(kernel_effective, pads_jax_ends)
            ]

            conv_kwargs["pads"] = pads_onnx_starts + pads_onnx_ends

            # JAX conv_transpose implies a spatial flip of the kernel relative to conv_general_dilated.
            # We need to flip it back (or forward?) to match ONNX ConvTranspose semantics.
            # Empirical evidence shows flipping is required.
            rhs_val = _flip_spatial_dims(
                ctx, rhs_val, rhs_shape, rhs_layout, "conv_rhs_transpose"
            )

        groups = params.get("feature_group_count", 1)
        if groups != 1:
            conv_kwargs["group"] = int(groups)

        conv_dtype_enum = _dtype_to_ir(
            np.dtype(
                getattr(
                    out_var.aval, "dtype", getattr(lhs_var.aval, "dtype", np.float32)
                )
            ),
            ctx.builder.enable_double_precision,
        )
        print(f"DEBUG: conv_dtype_enum={conv_dtype_enum}")

        if self._maybe_lower_complex(
            ctx,
            lhs_var,
            rhs_var,
            lhs_val,
            rhs_val,
            out_var,
            out_spec,
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_layout,
            rhs_layout,
            out_layout,
            conv_kwargs,
            op_type,
            target_kernel_layout,
        ):
            return

        canonical_input = lhs_val
        if lhs_layout != "NCHW":
            perm = _perm(lhs_layout, "NCHW")
            transposed = ctx.builder.Transpose(
                lhs_val,
                _outputs=[ctx.fresh_name("conv_lhs_nchw")],
                perm=perm,
            )
            lhs_dtype = getattr(getattr(lhs_val, "type", None), "dtype", None)
            if lhs_dtype is not None:
                transposed.type = ir.TensorType(lhs_dtype)
            _stamp_type_and_shape(transposed, tuple(lhs_shape[i] for i in perm))
            _ensure_value_metadata(ctx, transposed)
            canonical_input = transposed

        canonical_kernel = rhs_val
        if rhs_layout != target_kernel_layout:
            perm = _perm(rhs_layout, target_kernel_layout)
            transposed = ctx.builder.Transpose(
                rhs_val,
                _outputs=[ctx.fresh_name(f"conv_rhs_{target_kernel_layout.lower()}")],
                perm=perm,
            )
            rhs_dtype = getattr(getattr(rhs_val, "type", None), "dtype", None)
            if rhs_dtype is not None:
                transposed.type = ir.TensorType(rhs_dtype)
            _stamp_type_and_shape(transposed, tuple(rhs_shape[i] for i in perm))
            _ensure_value_metadata(ctx, transposed)
            canonical_kernel = transposed

        need_output_transpose = out_layout != "NCHW"
        perm_to_nchw: Sequence[int] | None = (
            _perm(out_layout, "NCHW") if need_output_transpose else None
        )

        canonical_input = cast_real_tensor(
            ctx, canonical_input, conv_dtype_enum, name_hint="conv_lhs_cast"
        )
        canonical_kernel = cast_real_tensor(
            ctx, canonical_kernel, conv_dtype_enum, name_hint="conv_rhs_cast"
        )

        conv_output_name = (
            ctx.fresh_name("conv_out_nchw")
            if need_output_transpose
            else (getattr(out_spec, "name", None) or ctx.fresh_name(op_type))
        )
        if op_type == "ConvTranspose":
            conv_result = ctx.builder.ConvTranspose(
                canonical_input,
                canonical_kernel,
                _outputs=[conv_output_name],
                **conv_kwargs,
            )
        else:
            conv_result = ctx.builder.Conv(
                canonical_input,
                canonical_kernel,
                _outputs=[conv_output_name],
                **conv_kwargs,
            )

        if need_output_transpose:
            assert perm_to_nchw is not None
            conv_shape_intermediate = tuple(out_shape[i] for i in perm_to_nchw)
        else:
            conv_shape_intermediate = tuple(out_shape)
        conv_result.type = ir.TensorType(conv_dtype_enum)
        _stamp_type_and_shape(conv_result, conv_shape_intermediate)
        _ensure_value_metadata(ctx, conv_result)

        if need_output_transpose:
            perm_back = _perm("NCHW", out_layout)
            final_name = getattr(out_spec, "name", None) or ctx.fresh_name("conv_out")
            final_val = ctx.builder.Transpose(
                conv_result,
                _outputs=[final_name],
                perm=perm_back,
            )
            final_val.type = ir.TensorType(conv_dtype_enum)
            _stamp_type_and_shape(final_val, out_shape)
            _ensure_value_metadata(ctx, final_val)
            ctx.bind_value_for_var(out_var, final_val)
        else:
            ctx.bind_value_for_var(out_var, conv_result)

    def _maybe_lower_complex(
        self,
        ctx: "IRContext",
        lhs_var,
        rhs_var,
        lhs_val: ir.Value,
        rhs_val: ir.Value,
        out_var,
        out_spec: ir.Value,
        lhs_shape: tuple[int, ...],
        rhs_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
        lhs_layout: str,
        rhs_layout: str,
        out_layout: str,
        conv_kwargs: dict[str, object],
        op_type: str,
        target_kernel_layout: str,
    ) -> bool:
        def _is_complex_var(var) -> bool:
            aval_dtype = getattr(getattr(var, "aval", None), "dtype", None)
            if aval_dtype is None:
                return False
            try:
                return np.issubdtype(np.dtype(aval_dtype), np.complexfloating)
            except TypeError:
                return False

        lhs_dtype = getattr(lhs_val, "dtype", None)
        rhs_dtype = getattr(rhs_val, "dtype", None)
        complex_hint = (
            lhs_dtype in COMPLEX_DTYPES
            or rhs_dtype in COMPLEX_DTYPES
            or _is_complex_var(lhs_var)
            or _is_complex_var(rhs_var)
            or _is_complex_var(out_var)
        )
        packed_hint = False
        if complex_hint:
            packed_hint = is_packed_complex_tensor(lhs_val) or is_packed_complex_tensor(
                rhs_val
            )
        if not (complex_hint or packed_hint):
            return False

        lhs_packed, lhs_base = ensure_packed_real_pair(
            ctx, lhs_val, name_hint="conv_lhs_pack"
        )
        rhs_packed, rhs_base = ensure_packed_real_pair(
            ctx, rhs_val, name_hint="conv_rhs_pack"
        )
        target_dtype = resolve_common_real_dtype(lhs_base, rhs_base)

        lhs_ready = (
            lhs_packed
            if lhs_packed.dtype == target_dtype
            else cast_real_tensor(
                ctx, lhs_packed, target_dtype, name_hint="conv_lhs_cast"
            )
        )
        rhs_ready = (
            rhs_packed
            if rhs_packed.dtype == target_dtype
            else cast_real_tensor(
                ctx, rhs_packed, target_dtype, name_hint="conv_rhs_cast"
            )
        )

        lhs_real, lhs_imag = split_packed_real_imag(
            ctx, lhs_ready, target_dtype, prefix="conv_lhs"
        )
        rhs_real, rhs_imag = split_packed_real_imag(
            ctx, rhs_ready, target_dtype, prefix="conv_rhs"
        )

        perm_input = _perm(lhs_layout, "NCHW") if lhs_layout != "NCHW" else None
        perm_kernel = _perm(rhs_layout, "OIHW") if rhs_layout != "OIHW" else None
        need_output_transpose = out_layout != "NCHW"
        perm_to_nchw: Sequence[int] | None = (
            _perm(out_layout, "NCHW") if need_output_transpose else None
        )
        perm_back: Sequence[int] | None = (
            _perm("NCHW", out_layout) if need_output_transpose else None
        )
        conv_shape_nchw = (
            tuple(out_shape[i] for i in perm_to_nchw)
            if perm_to_nchw is not None
            else tuple(out_shape)
        )

        def _transpose_to_layout(
            value: ir.Value,
            perm: Sequence[int] | None,
            shape: tuple[int, ...],
            name_hint: str,
        ) -> ir.Value:
            if not perm:
                return value
            transposed = ctx.builder.Transpose(
                value,
                _outputs=[ctx.fresh_name(name_hint)],
                perm=list(perm),
            )
            perm_shape = tuple(shape[i] for i in perm)
            _stamp_type_and_shape(transposed, perm_shape)
            transposed.type = ir.TensorType(
                getattr(value, "dtype", None) or target_dtype
            )
            _ensure_value_metadata(ctx, transposed)
            return transposed

        def _cast_value(
            value: ir.Value,
            dtype: ir.DataType,
            name_hint: str,
            *,
            fallback_shape: tuple[int, ...],
        ) -> ir.Value:
            if getattr(value, "dtype", None) == dtype:
                return value
            casted = ctx.builder.Cast(
                value,
                to=int(dtype.value),
                _outputs=[ctx.fresh_name(name_hint)],
            )
            casted.type = ir.TensorType(dtype)
            shape_meta = getattr(value, "shape", None)
            if isinstance(shape_meta, ir.Shape):
                dims = coerce_dim_values(tuple(shape_meta.dims))
            elif isinstance(shape_meta, Sequence):
                dims = coerce_dim_values(tuple(shape_meta))
            else:
                dims = coerce_dim_values(fallback_shape)
            _stamp_type_and_shape(casted, dims)
            _ensure_value_metadata(ctx, casted)
            return casted

        conv_compute_dtype = target_dtype
        if target_dtype == ir.DataType.DOUBLE:
            conv_compute_dtype = ir.DataType.FLOAT

        lhs_real_canon = _transpose_to_layout(
            lhs_real, perm_input, lhs_shape, "conv_lhs_real_nchw"
        )
        lhs_imag_canon = _transpose_to_layout(
            lhs_imag, perm_input, lhs_shape, "conv_lhs_imag_nchw"
        )
        rhs_real_canon = _transpose_to_layout(
            rhs_real, perm_kernel, rhs_shape, "conv_rhs_real_oihw"
        )
        rhs_imag_canon = _transpose_to_layout(
            rhs_imag, perm_kernel, rhs_shape, "conv_rhs_imag_oihw"
        )

        if conv_compute_dtype != target_dtype:
            lhs_real_canon = _cast_value(
                lhs_real_canon,
                conv_compute_dtype,
                "conv_lhs_real_cast",
                fallback_shape=conv_shape_nchw,
            )
            lhs_imag_canon = _cast_value(
                lhs_imag_canon,
                conv_compute_dtype,
                "conv_lhs_imag_cast",
                fallback_shape=conv_shape_nchw,
            )
            rhs_real_canon = _cast_value(
                rhs_real_canon,
                conv_compute_dtype,
                "conv_rhs_real_cast",
                fallback_shape=conv_shape_nchw,
            )
            rhs_imag_canon = _cast_value(
                rhs_imag_canon,
                conv_compute_dtype,
                "conv_rhs_imag_cast",
                fallback_shape=conv_shape_nchw,
            )

        def _conv_op(lhs: ir.Value, rhs: ir.Value, name_hint: str) -> ir.Value:
            conv = ctx.builder.Conv(
                lhs,
                rhs,
                _outputs=[ctx.fresh_name(name_hint)],
                **conv_kwargs,
            )
            conv.type = ir.TensorType(conv_compute_dtype)
            _stamp_type_and_shape(conv, conv_shape_nchw)
            _ensure_value_metadata(ctx, conv)
            return conv

        ar_br = _conv_op(lhs_real_canon, rhs_real_canon, "conv_ar_br")
        ai_bi = _conv_op(lhs_imag_canon, rhs_imag_canon, "conv_ai_bi")
        ar_bi = _conv_op(lhs_real_canon, rhs_imag_canon, "conv_ar_bi")
        ai_br = _conv_op(lhs_imag_canon, rhs_real_canon, "conv_ai_br")

        real_part = ctx.builder.Sub(
            ar_br,
            ai_bi,
            _outputs=[ctx.fresh_name("conv_real_part")],
        )
        real_part.type = ir.TensorType(conv_compute_dtype)
        _stamp_type_and_shape(real_part, conv_shape_nchw)
        _ensure_value_metadata(ctx, real_part)

        imag_part = ctx.builder.Add(
            ar_bi,
            ai_br,
            _outputs=[ctx.fresh_name("conv_imag_part")],
        )
        imag_part.type = ir.TensorType(conv_compute_dtype)
        _stamp_type_and_shape(imag_part, conv_shape_nchw)
        _ensure_value_metadata(ctx, imag_part)

        if conv_compute_dtype != target_dtype:
            real_part = _cast_value(
                real_part,
                target_dtype,
                "conv_real_upcast",
                fallback_shape=conv_shape_nchw,
            )
            imag_part = _cast_value(
                imag_part,
                target_dtype,
                "conv_imag_upcast",
                fallback_shape=conv_shape_nchw,
            )

        def _from_nchw(
            value: ir.Value,
            perm: Sequence[int] | None,
            name_hint: str,
        ) -> ir.Value:
            if not perm:
                _stamp_type_and_shape(value, out_shape)
                value.type = ir.TensorType(target_dtype)
                _ensure_value_metadata(ctx, value)
                return value
            transposed = ctx.builder.Transpose(
                value,
                _outputs=[ctx.fresh_name(name_hint)],
                perm=list(perm),
            )
            _stamp_type_and_shape(transposed, out_shape)
            transposed.type = ir.TensorType(target_dtype)
            _ensure_value_metadata(ctx, transposed)
            return transposed

        real_final = _from_nchw(real_part, perm_back, "conv_real_out")
        imag_final = _from_nchw(imag_part, perm_back, "conv_imag_out")

        output_name = getattr(out_spec, "name", None) or ctx.fresh_name("conv_out")
        packed = pack_real_imag_pair(
            ctx,
            real_final,
            imag_final,
            target_dtype,
            name_hint="conv_output",
            output_name=output_name,
        )

        out_spec.type = ir.TensorType(target_dtype)
        out_spec.dtype = target_dtype
        if getattr(packed, "shape", None) is not None:
            out_spec.shape = packed.shape
        _ensure_value_metadata(ctx, packed)
        ctx.bind_value_for_var(out_var, packed)
        return True
