# jax2onnx/plugins/jax/lax/pad.py

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jax2onnx.converter.ir_context import IRContext


def _flatten(seq: Iterable[int]) -> list[int]:
    return [int(v) for v in seq]


@register_primitive(
    jaxpr_primitive=jax.lax.pad_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.pad.html",
    onnx=[{"component": "Pad", "doc": "https://onnx.ai/onnx/operators/onnx__Pad.html"}],
    since="v0.8.0",
    context="primitives.lax",
    component="pad",
    testcases=[
        {
            "testcase": "pad_const_1d",
            "callable": lambda x: jax.lax.pad(
                x, jnp.asarray(0.0, dtype=x.dtype), ((1, 2, 0),)
            ),
            "input_shapes": [(5,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {2: {"const": 0.0}},
                        "path": "Pad:8",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "pad_const_2d",
            "callable": lambda x: jax.lax.pad(
                x, jnp.asarray(1.0, dtype=x.dtype), ((0, 0, 0), (1, 1, 0))
            ),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {2: {"const": 1.0}},
                        "path": "Pad:2x5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "pad_const_2d_cval",
            "callable": lambda x: jax.lax.pad(
                x,
                jnp.asarray(0, dtype=x.dtype),
                ((0, 0, 0), (1, 1, 0)),
            ),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {2: {"const": 0.0}},
                        "path": "Pad:2x5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "pad_inside_scan_smoke_f64",
            "callable": lambda x: jax.lax.scan(
                lambda carry, _: (
                    carry,
                    (
                        jnp.pad(carry, ((0, 0), (0, 0), (1, 1), (1, 1)))[
                            :, :, 1:-1, 1:-1
                        ]
                        * carry
                    ),
                ),
                x,
                None,
                length=2,
            )[1],
            "input_shapes": [(1, 3, 8, 8)],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL", 1, 3, 8, 8)],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Loop"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "pad_inside_nested_scan_smoke_f64",
            "callable": lambda x: jax.lax.scan(
                lambda carry, _: jax.lax.scan(
                    lambda c2, __: (
                        c2,
                        (
                            jnp.pad(c2, ((0, 0), (0, 0), (1, 1), (1, 1)))[
                                :, :, 1:-1, 1:-1
                            ]
                            * c2
                        ),
                    ),
                    carry,
                    None,
                    length=2,
                ),
                x,
                None,
                length=1,
            )[1],
            "input_shapes": [(1, 3, 8, 8)],
            "expected_output_shapes": [
                ("JAX2ONNX_DYNAMIC_DIM_SENTINEL", 2, 1, 3, 8, 8)
            ],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Loop"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class PadPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.pad`` (constant mode, zero interior padding) to ONNX ``Pad``."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        if len(eqn.invars) < 2:
            raise ValueError("lax.pad expects data and constant value inputs")

        data_var = eqn.invars[0]
        const_var = eqn.invars[1]
        out_var = eqn.outvars[0]

        padding_config = eqn.params.get("padding_config")
        if padding_config is None:
            raise ValueError("padding_config parameter is required for lax.pad")

        interiors = [int(interior) for (_, _, interior) in padding_config]
        if any(v != 0 for v in interiors):
            raise NotImplementedError(
                "lax.pad with interior padding > 0 is not supported in ONNX Pad"
            )

        data_val = ctx.get_value_for_var(data_var, name_hint=ctx.fresh_name("pad_data"))
        prefer_dt = np.dtype(getattr(data_var.aval, "dtype", np.float32))
        pad_raw = ctx.get_value_for_var(
            const_var, name_hint=ctx.fresh_name("pad_cval"), prefer_np_dtype=prefer_dt
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("pad_out"))

        data_dtype = getattr(getattr(data_val, "type", None), "dtype", None)
        pad_dtype = getattr(getattr(pad_raw, "type", None), "dtype", None)

        pad_val = pad_raw
        if data_dtype is not None and pad_dtype is not None and data_dtype != pad_dtype:
            cast_name = ctx.fresh_name("pad_cval_like")
            pad_val = ctx.builder.Cast(
                pad_raw,
                _outputs=[cast_name],
                to=int(data_dtype.value),
            )
            pad_val.type = ir.TensorType(data_dtype)
            pad_val.shape = pad_raw.shape
            _stamp_type_and_shape(pad_val, tuple(getattr(const_var.aval, "shape", ())))
            _ensure_value_metadata(ctx, pad_val)
        else:
            pad_shape = tuple(getattr(const_var.aval, "shape", ()))
            _stamp_type_and_shape(pad_val, pad_shape)
            _ensure_value_metadata(ctx, pad_val)

        begins = _flatten(lo for (lo, _, _) in padding_config)
        ends = _flatten(hi for (_, hi, _) in padding_config)
        pads_vec = np.asarray(begins + ends, dtype=np.int64)
        pads_name = ctx.fresh_name("pad_pads")
        pads_val = ctx.builder.add_initializer_from_array(
            name=pads_name, array=pads_vec
        )
        _stamp_type_and_shape(pads_val, (pads_vec.size,))
        _ensure_value_metadata(ctx, pads_val)

        pad_inputs = [data_val, pads_val, pad_val]

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Pad")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Pad")

        result = ctx.builder.Pad(
            *pad_inputs,
            mode="constant",
            _outputs=[desired_name],
        )

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        result_dtype = getattr(getattr(data_val, "type", None), "dtype", None)
        if result_dtype is None:
            result_dtype = getattr(getattr(out_spec, "type", None), "dtype", None)
            if result_dtype is None:
                result_dtype = _dtype_to_ir(
                    np.dtype(getattr(out_var.aval, "dtype", np.float32)),
                    ctx.builder.enable_double_precision,
                )
        result.type = ir.TensorType(result_dtype)
        result.shape = ir.Shape(out_shape)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
