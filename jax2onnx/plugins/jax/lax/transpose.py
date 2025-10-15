# jax2onnx/plugins/jax/lax/transpose.py

from __future__ import annotations
from typing import Any, List

from jax import lax

import onnx_ir as ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._ir_shapes import (
    _ensure_value_metadata,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)


@register_primitive(
    jaxpr_primitive=lax.transpose_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.transpose.html",
    onnx=[
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="transpose",
    testcases=[
        {
            "testcase": "transpose_basic",
            "callable": lambda x: lax.transpose(x, (1, 0)),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(
                ["Transpose:3x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_square_matrix",
            "callable": lambda x: lax.transpose(x, (1, 0)),
            "input_shapes": [(4, 4)],
            "post_check_onnx_graph": EG(
                ["Transpose"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_3d",
            "callable": lambda x: lax.transpose(x, (1, 2, 0)),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["Transpose:3x4x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_4d",
            "callable": lambda x: lax.transpose(x, (0, 2, 3, 1)),
            "input_shapes": [(2, 3, 4, 5)],
            "post_check_onnx_graph": EG(
                ["Transpose:2x4x5x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_reverse",
            "callable": lambda x: lax.transpose(x, (2, 1, 0)),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["Transpose:4x3x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_no_axes",
            "callable": lambda x: lax.transpose(x, tuple(range(x.ndim - 1, -1, -1))),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["Transpose"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "transpose_nhwc_to_nchw",
            "callable": lambda x: lax.transpose(x, (0, 3, 1, 2)),
            "input_shapes": [(2, 28, 28, 3)],
            "post_check_onnx_graph": EG(
                ["Transpose:2x3x28x28"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class TransposePlugin(PrimitiveLeafPlugin):
    """plugins IR converter for jax.lax.transpose → ONNX Transpose."""

    def lower(self, ctx: Any, eqn):
        x_var = eqn.invars[0]
        y_var = eqn.outvars[0]

        # JAX params may be named "permutation" (current) or "dimensions" (older).
        perm = eqn.params.get("permutation", None)
        if perm is None:
            perm = eqn.params.get("dimensions", None)
        if perm is None:
            # Default to reversing dims if nothing provided (matches JAX semantics for None).
            in_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
            perm = tuple(range(len(in_shape) - 1, -1, -1))
        else:
            perm = tuple(int(p) for p in perm)

        # Inputs/outputs
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        out_spec = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("y"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Transpose")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Transpose")

        result = ctx.builder.Transpose(
            x_val,
            perm=list(perm),
            _outputs=[desired_name],
        )

        # Stamp output shape/type by permuting the input aval shape.
        in_aval = getattr(x_var, "aval", None)
        in_shape = tuple(getattr(in_aval, "shape", ()) or ())
        result_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        if result_dtype is None:
            result_dtype = getattr(getattr(out_spec, "type", None), "dtype", None)
        if result_dtype is not None:
            result.type = ir.TensorType(result_dtype)

        if in_shape:
            out_dims: List[Any] = [in_shape[i] for i in perm]
            _stamp_type_and_shape(
                result, tuple(_to_ir_dim_for_shape(d) for d in out_dims)
            )

        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(y_var, result)
