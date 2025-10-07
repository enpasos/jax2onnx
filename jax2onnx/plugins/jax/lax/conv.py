# jax2onnx/plugins/jax/lax/conv.py

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_LAYOUT_MAP = {
    (0, 1, 2, 3): "NCHW",
    (0, 3, 1, 2): "NHWC",
    "NCHW": "NCHW",
    "NHWC": "NHWC",
}
_FILTER_LAYOUT_MAP = {
    (0, 1, 2, 3): "OIHW",
    (3, 2, 0, 1): "HWIO",
    "OIHW": "OIHW",
    "HWIO": "HWIO",
}
_OUTPUT_LAYOUT_MAP = {
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
        },
        {
            "testcase": "conv_nchw",
            "callable": lambda x, w: jax.lax.conv(
                x, w, window_strides=(1, 1), padding="VALID"
            ),
            "input_shapes": [(1, 2, 5, 5), (3, 2, 3, 3)],
            "run_only_f32_variant": True,
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
            _ensure_value_info(ctx, transposed)
            canonical_input = transposed

        canonical_kernel = rhs_val
        if rhs_layout != "OIHW":
            perm = _perm(rhs_layout, "OIHW")
            transposed = ctx.builder.Transpose(
                rhs_val,
                _outputs=[ctx.fresh_name("conv_rhs_oihw")],
                perm=perm,
            )
            rhs_dtype = getattr(getattr(rhs_val, "type", None), "dtype", None)
            if rhs_dtype is not None:
                transposed.type = ir.TensorType(rhs_dtype)
            _stamp_type_and_shape(transposed, tuple(rhs_shape[i] for i in perm))
            _ensure_value_info(ctx, transposed)
            canonical_kernel = transposed

        need_output_transpose = out_layout != "NCHW"
        perm_to_nchw: Sequence[int] | None = (
            _perm(out_layout, "NCHW") if need_output_transpose else None
        )

        conv_kwargs: dict[str, object] = {}
        strides = params.get("window_strides", (1, 1))
        conv_kwargs["strides"] = [int(s) for s in strides]

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

        groups = params.get("feature_group_count", 1)
        if groups != 1:
            conv_kwargs["group"] = int(groups)

        conv_output_name = (
            ctx.fresh_name("conv_out_nchw")
            if need_output_transpose
            else (getattr(out_spec, "name", None) or ctx.fresh_name("Conv"))
        )
        conv_result = ctx.builder.Conv(
            canonical_input,
            canonical_kernel,
            _outputs=[conv_output_name],
            **conv_kwargs,
        )

        conv_dtype_enum = _dtype_to_ir(
            np.dtype(
                getattr(
                    out_var.aval, "dtype", getattr(lhs_var.aval, "dtype", np.float32)
                )
            ),
            ctx.builder.enable_double_precision,
        )
        if need_output_transpose:
            assert perm_to_nchw is not None
            conv_shape_intermediate = tuple(out_shape[i] for i in perm_to_nchw)
        else:
            conv_shape_intermediate = tuple(out_shape)
        conv_result.type = ir.TensorType(conv_dtype_enum)
        _stamp_type_and_shape(conv_result, conv_shape_intermediate)
        _ensure_value_info(ctx, conv_result)

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
            _ensure_value_info(ctx, final_val)
            ctx.bind_value_for_var(out_var, final_val)
        else:
            ctx.bind_value_for_var(out_var, conv_result)
