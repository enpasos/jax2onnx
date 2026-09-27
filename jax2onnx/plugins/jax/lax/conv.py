# jax2onnx/plugins/jax/lax/conv.py

from __future__ import annotations

from typing import Any, Final, Sequence, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
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
from jax2onnx.plugins._ir_shapes import (
    _ensure_value_metadata,
    _is_static_int,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _as_value(value: Any) -> ir.Value:
    return cast(ir.Value, value)


def _require_dtype(value: ir.Value, *, context: str) -> ir.DataType:
    dtype = value.dtype
    if dtype is None:
        raise ValueError(f"{context} requires typed IR input")
    return dtype


_LAYOUT_MAP: Final[dict[tuple[int, ...] | str, str]] = {
    (0, 1, 2): "NCW",
    (0, 2, 1): "NWC",
    (0, 1, 2, 3): "NCHW",
    (0, 3, 1, 2): "NHWC",
    "NCW": "NCW",
    "NWC": "NWC",
    "NCHW": "NCHW",
    "NHWC": "NHWC",
}
_FILTER_LAYOUT_MAP: Final[dict[tuple[int, ...] | str, str]] = {
    (0, 1, 2): "OIW",
    (2, 1, 0): "WIO",
    (0, 1, 2, 3): "OIHW",
    (3, 2, 0, 1): "HWIO",
    "OIW": "OIW",
    "WIO": "WIO",
    "OIHW": "OIHW",
    "HWIO": "HWIO",
}
_OUTPUT_LAYOUT_MAP: Final[dict[tuple[int, ...] | str, str]] = {
    (0, 1, 2): "NCW",
    (0, 2, 1): "NWC",
    (0, 1, 2, 3): "NCHW",
    (0, 3, 1, 2): "NHWC",
    "NCW": "NCW",
    "NWC": "NWC",
    "NCHW": "NCHW",
    "NHWC": "NHWC",
}


def _layout_from_spec(
    spec: object, mapping: dict[tuple[int, ...] | str, str]
) -> str | None:
    key: tuple[int, ...] | str | object = spec
    if isinstance(spec, str):
        key = spec.upper()
    elif isinstance(spec, Sequence):
        try:
            key = tuple(int(v) for v in spec)
        except (TypeError, ValueError):
            key = spec
    if isinstance(key, tuple | str):
        return mapping.get(key)
    return None


def _perm(src_layout: str, dst_layout: str) -> list[int]:
    return [src_layout.index(axis) for axis in dst_layout]


def _flatten_padding(pads: Sequence[Sequence[int]]) -> list[int]:
    befores = [int(before) for before, _ in pads]
    afters = [int(after) for _, after in pads]
    return befores + afters


def _canonical_input_layout(layout: str) -> str:
    return "NCW" if len(layout) == 3 else "NCHW"


def _canonical_kernel_layout(layout: str, *, is_transpose: bool) -> str:
    if len(layout) == 3:
        return "IOW" if is_transpose else "OIW"
    return "IOHW" if is_transpose else "OIHW"


def _flip_spatial_dims(
    ctx: LoweringContextProtocol,
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

    flipped = _as_value(
        ctx.builder.Slice(
            val,
            starts_val,
            ends_val,
            axes_val,
            steps_val,
            _outputs=[ctx.fresh_name(name_hint)],
        )
    )
    flipped.type = ir.TensorType(_require_dtype(val, context="conv spatial flip"))
    _stamp_type_and_shape(flipped, shape)
    _ensure_value_metadata(ctx, flipped)
    return flipped


def _transpose_kernel(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    perm: Sequence[int],
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    out = _as_value(
        ctx.builder.Transpose(
            val, _outputs=[ctx.fresh_name(name_hint)], perm=list(perm)
        )
    )
    out.type = ir.TensorType(_require_dtype(val, context="conv kernel transpose"))
    _stamp_type_and_shape(out, shape)
    _ensure_value_metadata(ctx, out)
    return out


def _reshape_kernel(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    shape_val = _const_i64(ctx, list(shape), name_hint=f"{name_hint}_shape")
    out = _as_value(
        ctx.builder.Reshape(val, shape_val, _outputs=[ctx.fresh_name(name_hint)])
    )
    out.type = ir.TensorType(_require_dtype(val, context="conv kernel reshape"))
    _stamp_type_and_shape(out, shape)
    _ensure_value_metadata(ctx, out)
    return out


def _conv_transpose_pads(
    pad_pairs: Sequence[Sequence[int]],
    kernel_spatial: Sequence[int],
    rhs_dilation: Sequence[int],
) -> tuple[list[int], list[int], list[int]]:
    """Map lhs-dilated conv padding onto ONNX ``ConvTranspose`` pads.

    JAX pads the dilated input, while ConvTranspose pads crop its full output, so
    ``pad_onnx = k_eff - 1 - pad_jax``. A negative result means JAX pads beyond
    the kernel's reach; those output positions are zeros, so the deficit is
    returned separately (before, after) for an explicit ``Pad`` of the output.
    ``output_padding`` cannot express it in general: ONNX Runtime requires it to
    be smaller than ``max(stride, dilation)``.
    """
    kernel_effective = [
        (int(k) - 1) * int(d) + 1
        for k, d in zip(kernel_spatial, rhs_dilation, strict=True)
    ]
    starts = [
        k - 1 - int(lo) for k, (lo, _) in zip(kernel_effective, pad_pairs, strict=True)
    ]
    ends = [
        k - 1 - int(hi) for k, (_, hi) in zip(kernel_effective, pad_pairs, strict=True)
    ]
    onnx_pads = [max(p, 0) for p in starts + ends]
    return onnx_pads, [max(-p, 0) for p in starts], [max(-p, 0) for p in ends]


def _conv_transpose_kernel(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    shape: tuple[int, ...],
    layout: str,
    groups: int,
) -> ir.Value:
    """Turn an lhs-dilated conv kernel into an ONNX ``ConvTranspose`` weight.

    ``conv_general_dilated`` correlates the dilated input with an ``(O, I/g, *k)``
    kernel; ConvTranspose scatters with an ``(I, O/g, *k)`` weight. The
    equivalent weight is the spatially flipped kernel with input and output
    channels swapped within each feature group.
    """
    kernel = _flip_spatial_dims(ctx, val, shape, layout, "conv_rhs_transpose")
    target_layout = _canonical_kernel_layout(layout, is_transpose=True)
    if groups == 1:
        perm = _perm(layout, target_layout)
        return _transpose_kernel(
            ctx,
            kernel,
            perm,
            tuple(shape[i] for i in perm),
            f"conv_rhs_{target_layout.lower()}",
        )

    if not all(_is_static_int(dim) for dim in shape):
        raise NotImplementedError(
            f"Grouped transposed convolution requires a static kernel shape; got {shape}."
        )
    oi_layout = _canonical_kernel_layout(layout, is_transpose=False)
    perm = _perm(layout, oi_layout)
    out_channels, in_per_group, *spatial = (int(shape[i]) for i in perm)
    if layout != oi_layout:
        kernel = _transpose_kernel(
            ctx,
            kernel,
            perm,
            (out_channels, in_per_group, *spatial),
            f"conv_rhs_{oi_layout.lower()}",
        )
    out_per_group = out_channels // groups
    kernel = _reshape_kernel(
        ctx,
        kernel,
        (groups, out_per_group, in_per_group, *spatial),
        "conv_rhs_grouped",
    )
    swap = [0, 2, 1, *range(3, 3 + len(spatial))]
    kernel = _transpose_kernel(
        ctx,
        kernel,
        swap,
        (groups, in_per_group, out_per_group, *spatial),
        "conv_rhs_group_swap",
    )
    return _reshape_kernel(
        ctx,
        kernel,
        (groups * in_per_group, out_per_group, *spatial),
        f"conv_rhs_{target_layout.lower()}",
    )


@register_primitive(
    jaxpr_primitive=jax.lax.conv_general_dilated_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv.html",
    onnx=[
        {
            "component": "Conv",
            "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html",
        },
        {
            "component": "ConvTranspose",
            "doc": "https://onnx.ai/onnx/operators/onnx__ConvTranspose.html",
        },
        {
            "component": "Pad",
            "doc": "https://onnx.ai/onnx/operators/onnx__Pad.html",
        },
    ],
    since="0.2.0",
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
            "testcase": "conv_general_dilated_1d",
            "callable": lambda x, w: jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1,),
                padding="VALID",
                dimension_numbers=("NCW", "OIW", "NCW"),
            ),
            "input_values": [
                np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32),
                np.array([[[0.5, -1.0]]], dtype=np.float32),
            ],
            "expected_output_shapes": [(1, 1, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {"path": "Conv", "counts": {"Conv": 1}},
                ],
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
        {
            "testcase": "conv_transpose_lhs_dilation_nchw",
            "callable": lambda x, w: jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1, 1),
                padding=((1, 1), (1, 1)),
                lhs_dilation=(2, 2),
            ),
            "input_shapes": [(1, 4, 2, 3), (4, 4, 2, 2)],
            "run_only_f32_variant": True,
            "check_deployment_readiness_report": True,
            "post_check_onnx_graph": EG(
                [("ConvTranspose:1x4x4x6", {"counts": {"ConvTranspose": 1}})],
                must_absent=["Conv", "Pad"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv_transpose_nhwc_same",
            "callable": lambda x, w: jax.lax.conv_transpose(
                x,
                w,
                strides=(2, 2),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            ),
            "input_shapes": [(1, 3, 5, 4), (3, 3, 4, 6)],
            "run_only_f32_variant": True,
            "check_deployment_readiness_report": True,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:1x4x3x5 -> ConvTranspose:1x6x6x10 -> "
                    "Transpose:1x6x10x6"
                ],
                must_absent=["Pad"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv_transpose_valid_stride_gt_kernel",
            "callable": lambda x, w: jax.lax.conv_transpose(
                x,
                w,
                strides=(3, 3),
                padding="VALID",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            ),
            "input_shapes": [(1, 2, 3, 3), (3, 2, 2, 2)],
            "run_only_f32_variant": True,
            "check_deployment_readiness_report": True,
            "post_check_onnx_graph": EG(
                [
                    (
                        "ConvTranspose:1x3x8x8 -> Pad:1x3x9x9",
                        {"inputs": {1: {"const": [0, 0, 0, 0, 0, 0, 1, 1]}}},
                    )
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv_transpose_valid_stride_gt_kernel_dynamic",
            "callable": lambda x, w: jax.lax.conv_transpose(
                x,
                w,
                strides=(3, 3),
                padding="VALID",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            ),
            "input_shapes": [("B", 2, 3, 3), (3, 2, 2, 2)],
            "run_only_f32_variant": True,
            "check_deployment_readiness_report": True,
            "post_check_onnx_graph": EG(
                ["ConvTranspose:Bx3x8x8 -> Pad:Bx3x9x9"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv_transpose_1d_leading_output_pad",
            "callable": lambda x, w: jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1,),
                padding=((3, 0),),
                lhs_dilation=(2,),
            ),
            "input_shapes": [(1, 2, 4), (3, 2, 2)],
            "run_only_f32_variant": True,
            "check_deployment_readiness_report": True,
            "post_check_onnx_graph": EG(
                [
                    (
                        "ConvTranspose:1x3x7 -> Pad:1x3x9",
                        {"inputs": {1: {"const": [0, 0, 2, 0, 0, 0]}}},
                    )
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv_transpose_grouped",
            "callable": lambda x, w: jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1, 1),
                padding=((1, 2), (1, 2)),
                lhs_dilation=(2, 2),
                feature_group_count=2,
            ),
            "input_shapes": [(1, 4, 3, 3), (6, 2, 3, 3)],
            "run_only_f32_variant": True,
            "check_deployment_readiness_report": True,
            "post_check_onnx_graph": EG(
                [
                    "Slice:6x2x3x3 -> Reshape:2x3x2x3x3 -> Transpose:2x2x3x3x3 -> "
                    "Reshape:4x3x3x3 -> ConvTranspose:1x6x6x6"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv_transpose_grouped_nhwc",
            "callable": lambda x, w: jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1, 1),
                padding=((1, 2), (1, 2)),
                lhs_dilation=(2, 2),
                feature_group_count=2,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            ),
            "input_shapes": [(1, 3, 3, 4), (3, 3, 2, 6)],
            "run_only_f32_variant": True,
            "check_deployment_readiness_report": True,
            "post_check_onnx_graph": EG(
                [
                    (
                        "Transpose:1x4x3x3 -> ConvTranspose:1x6x6x6 -> "
                        "Transpose:1x6x6x6",
                        {"counts": {"Reshape": 2, "Transpose": 4}},
                    )
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conv_transpose_asymmetric",
            "callable": lambda x, w: jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1, 1),
                padding=((0, 1), (2, 1)),
                lhs_dilation=(2, 3),
                rhs_dilation=(1, 2),
            ),
            "input_shapes": [(1, 2, 3, 4), (3, 2, 2, 3)],
            "run_only_f32_variant": True,
            "check_deployment_readiness_report": True,
            "post_check_onnx_graph": EG(
                ["ConvTranspose:1x3x5x9"],
                must_absent=["Pad"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ConvGeneralDilatedPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.conv_general_dilated`` to ONNX ``Conv``, or to ``ConvTranspose``
    when the input is dilated (``lhs_dilation > 1``)."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
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

        batch_group_count = int(params.get("batch_group_count", 1))
        if batch_group_count != 1:
            raise NotImplementedError(
                "conv_general_dilated with batch_group_count="
                f"{batch_group_count} is not supported in ONNX lowering."
            )

        strides = [int(s) for s in params.get("window_strides", (1, 1))]
        lhs_dilation = [int(d) for d in params.get("lhs_dilation") or ()]
        # lhs (input) dilation is a transposed convolution: ONNX ConvTranspose
        # strides are the input dilation, and the window stride must be 1.
        is_transpose = any(d != 1 for d in lhs_dilation)
        if is_transpose and any(s != 1 for s in strides):
            raise NotImplementedError(
                f"conv_general_dilated with lhs_dilation={tuple(lhs_dilation)} and "
                f"window_strides={tuple(strides)} has no ONNX ConvTranspose "
                "equivalent; only unit window strides are supported with input "
                "dilation."
            )
        op_type = "ConvTranspose" if is_transpose else "Conv"
        target_input_layout = _canonical_input_layout(lhs_layout)
        target_kernel_layout = _canonical_kernel_layout(
            rhs_layout, is_transpose=is_transpose
        )

        conv_kwargs: dict[str, object] = {
            "strides": lhs_dilation if is_transpose else strides
        }
        rhs_dilation = [int(d) for d in params.get("rhs_dilation") or ()]
        if rhs_dilation:
            conv_kwargs["dilations"] = rhs_dilation
        groups = int(params.get("feature_group_count", 1))
        if groups != 1:
            conv_kwargs["group"] = groups

        # Per-axis zeros padded around a ConvTranspose output when JAX pads the
        # dilated input beyond the kernel's reach (see _conv_transpose_pads).
        output_pad_before: list[int] = []
        output_pad_after: list[int] = []

        padding = params.get("padding", "VALID")
        if isinstance(padding, str):
            if is_transpose:
                raise NotImplementedError(
                    "Transposed conv_general_dilated requires explicit (low, high) "
                    f"padding pairs; got {padding!r}."
                )
            pad_mode = padding.upper()
            if pad_mode in ("SAME", "SAME_UPPER"):
                conv_kwargs["auto_pad"] = "SAME_UPPER"
            elif pad_mode == "VALID":
                num_spatial = max(len(lhs_shape) - 2, 0)
                conv_kwargs["pads"] = [0] * (2 * num_spatial)
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
            if is_transpose:
                kernel_spatial = [
                    rhs_shape[i] for i, c in enumerate(rhs_layout) if c not in "OI"
                ]
                (
                    conv_kwargs["pads"],
                    output_pad_before,
                    output_pad_after,
                ) = _conv_transpose_pads(
                    pad_pairs,
                    kernel_spatial,
                    rhs_dilation or [1] * len(kernel_spatial),
                )
            else:
                conv_kwargs["pads"] = _flatten_padding(pad_pairs)

        conv_dtype_enum = _dtype_to_ir(
            np.dtype(
                getattr(
                    out_var.aval, "dtype", getattr(lhs_var.aval, "dtype", np.float32)
                )
            ),
            ctx.builder.enable_double_precision,
        )

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
            target_input_layout,
            target_kernel_layout,
        ):
            return

        canonical_input = lhs_val
        if lhs_layout != target_input_layout:
            perm = _perm(lhs_layout, target_input_layout)
            transposed = _as_value(
                ctx.builder.Transpose(
                    lhs_val,
                    _outputs=[
                        ctx.fresh_name(f"conv_lhs_{target_input_layout.lower()}")
                    ],
                    perm=perm,
                )
            )
            lhs_dtype = getattr(getattr(lhs_val, "type", None), "dtype", None)
            if lhs_dtype is not None:
                transposed.type = ir.TensorType(lhs_dtype)
            _stamp_type_and_shape(transposed, tuple(lhs_shape[i] for i in perm))
            _ensure_value_metadata(ctx, transposed)
            canonical_input = transposed

        canonical_kernel = rhs_val
        if is_transpose:
            canonical_kernel = _conv_transpose_kernel(
                ctx, rhs_val, rhs_shape, rhs_layout, groups
            )
        elif rhs_layout != target_kernel_layout:
            perm = _perm(rhs_layout, target_kernel_layout)
            transposed = _as_value(
                ctx.builder.Transpose(
                    rhs_val,
                    _outputs=[
                        ctx.fresh_name(f"conv_rhs_{target_kernel_layout.lower()}")
                    ],
                    perm=perm,
                )
            )
            rhs_dtype = getattr(getattr(rhs_val, "type", None), "dtype", None)
            if rhs_dtype is not None:
                transposed.type = ir.TensorType(rhs_dtype)
            _stamp_type_and_shape(transposed, tuple(rhs_shape[i] for i in perm))
            _ensure_value_metadata(ctx, transposed)
            canonical_kernel = transposed

        need_output_transpose = out_layout != target_input_layout
        perm_to_nchw: Sequence[int] | None = (
            _perm(out_layout, target_input_layout) if need_output_transpose else None
        )

        canonical_input = cast_real_tensor(
            ctx, canonical_input, conv_dtype_enum, name_hint="conv_lhs_cast"
        )
        canonical_kernel = cast_real_tensor(
            ctx, canonical_kernel, conv_dtype_enum, name_hint="conv_rhs_cast"
        )

        need_output_pad = any(output_pad_before) or any(output_pad_after)
        conv_output_name = (
            ctx.fresh_name(f"conv_out_{target_input_layout.lower()}")
            if need_output_transpose or need_output_pad
            else (getattr(out_spec, "name", None) or ctx.fresh_name(op_type))
        )
        if op_type == "ConvTranspose":
            conv_result = _as_value(
                ctx.builder.ConvTranspose(
                    canonical_input,
                    canonical_kernel,
                    _outputs=[conv_output_name],
                    **conv_kwargs,
                )
            )
        else:
            conv_result = _as_value(
                ctx.builder.Conv(
                    canonical_input,
                    canonical_kernel,
                    _outputs=[conv_output_name],
                    **conv_kwargs,
                )
            )

        if need_output_transpose:
            assert perm_to_nchw is not None
            conv_shape_intermediate = tuple(out_shape[i] for i in perm_to_nchw)
        else:
            conv_shape_intermediate = tuple(out_shape)
        conv_result.type = ir.TensorType(conv_dtype_enum)

        if need_output_pad:
            spatial_pads = zip(output_pad_before, output_pad_after, strict=True)
            pad_totals = [0, 0, *(before + after for before, after in spatial_pads)]
            unpadded_shape = tuple(
                dim if not pad else int(dim) - pad if _is_static_int(dim) else None
                for dim, pad in zip(conv_shape_intermediate, pad_totals, strict=True)
            )
            _stamp_type_and_shape(conv_result, unpadded_shape)
            _ensure_value_metadata(ctx, conv_result)
            pads_val = _const_i64(
                ctx,
                [0, 0, *output_pad_before, 0, 0, *output_pad_after],
                name_hint="conv_transpose_output_pads",
            )
            padded_name = (
                ctx.fresh_name(f"conv_out_{target_input_layout.lower()}_padded")
                if need_output_transpose
                else (getattr(out_spec, "name", None) or ctx.fresh_name("Pad"))
            )
            conv_result = _as_value(
                ctx.builder.Pad(
                    conv_result, pads_val, mode="constant", _outputs=[padded_name]
                )
            )
            conv_result.type = ir.TensorType(conv_dtype_enum)

        _stamp_type_and_shape(conv_result, conv_shape_intermediate)
        _ensure_value_metadata(ctx, conv_result)

        if need_output_transpose:
            perm_back = _perm(target_input_layout, out_layout)
            final_name = getattr(out_spec, "name", None) or ctx.fresh_name("conv_out")
            final_val = _as_value(
                ctx.builder.Transpose(
                    conv_result,
                    _outputs=[final_name],
                    perm=perm_back,
                )
            )
            final_val.type = ir.TensorType(conv_dtype_enum)
            _stamp_type_and_shape(final_val, out_shape)
            _ensure_value_metadata(ctx, final_val)
            ctx.bind_value_for_var(out_var, final_val)
        else:
            ctx.bind_value_for_var(out_var, conv_result)

    def _maybe_lower_complex(
        self,
        ctx: LoweringContextProtocol,
        lhs_var: Any,
        rhs_var: Any,
        lhs_val: ir.Value,
        rhs_val: ir.Value,
        out_var: Any,
        out_spec: ir.Value,
        lhs_shape: tuple[int, ...],
        rhs_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
        lhs_layout: str,
        rhs_layout: str,
        out_layout: str,
        conv_kwargs: dict[str, object],
        op_type: str,
        target_input_layout: str,
        target_kernel_layout: str,
    ) -> bool:
        def _is_complex_var(var: Any) -> bool:
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
        if op_type == "ConvTranspose":
            raise NotImplementedError(
                "Complex transposed convolution (lhs_dilation > 1) is not supported "
                "in ONNX lowering."
            )

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

        perm_input = (
            _perm(lhs_layout, target_input_layout)
            if lhs_layout != target_input_layout
            else None
        )
        perm_kernel = (
            _perm(rhs_layout, target_kernel_layout)
            if rhs_layout != target_kernel_layout
            else None
        )
        need_output_transpose = out_layout != target_input_layout
        perm_to_nchw: Sequence[int] | None = (
            _perm(out_layout, target_input_layout) if need_output_transpose else None
        )
        perm_back: Sequence[int] | None = (
            _perm(target_input_layout, out_layout) if need_output_transpose else None
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
            transposed = _as_value(
                ctx.builder.Transpose(
                    value,
                    _outputs=[ctx.fresh_name(name_hint)],
                    perm=list(perm),
                )
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
            casted = _as_value(
                ctx.builder.Cast(
                    value,
                    to=int(dtype.value),
                    _outputs=[ctx.fresh_name(name_hint)],
                )
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
            conv = _as_value(
                ctx.builder.Conv(
                    lhs,
                    rhs,
                    _outputs=[ctx.fresh_name(name_hint)],
                    **conv_kwargs,
                )
            )
            conv.type = ir.TensorType(conv_compute_dtype)
            _stamp_type_and_shape(conv, conv_shape_nchw)
            _ensure_value_metadata(ctx, conv)
            return conv

        ar_br = _conv_op(lhs_real_canon, rhs_real_canon, "conv_ar_br")
        ai_bi = _conv_op(lhs_imag_canon, rhs_imag_canon, "conv_ai_bi")
        ar_bi = _conv_op(lhs_real_canon, rhs_imag_canon, "conv_ar_bi")
        ai_br = _conv_op(lhs_imag_canon, rhs_real_canon, "conv_ai_br")

        real_part = _as_value(
            ctx.builder.Sub(
                ar_br,
                ai_bi,
                _outputs=[ctx.fresh_name("conv_real_part")],
            )
        )
        real_part.type = ir.TensorType(conv_compute_dtype)
        _stamp_type_and_shape(real_part, conv_shape_nchw)
        _ensure_value_metadata(ctx, real_part)

        imag_part = _as_value(
            ctx.builder.Add(
                ar_bi,
                ai_br,
                _outputs=[ctx.fresh_name("conv_imag_part")],
            )
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
            transposed = _as_value(
                ctx.builder.Transpose(
                    value,
                    _outputs=[ctx.fresh_name(name_hint)],
                    perm=list(perm),
                )
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
