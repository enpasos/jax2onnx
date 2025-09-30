from __future__ import annotations
from typing import TYPE_CHECKING, Any, List

from jax import lax

import onnx_ir as ir
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2._ir_shapes import (
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)

if TYPE_CHECKING:
    from jax2onnx.converter2.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


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
    context="primitives2.lax",
    component="transpose",
    testcases=[
        {
            "testcase": "transpose_basic",
            "callable": lambda x: lax.transpose(x, (1, 0)),
            "input_shapes": [(2, 3)],
        },
        {
            "testcase": "transpose_square_matrix",
            "callable": lambda x: lax.transpose(x, (1, 0)),
            "input_shapes": [(4, 4)],
        },
        {
            "testcase": "transpose_3d",
            "callable": lambda x: lax.transpose(x, (1, 2, 0)),
            "input_shapes": [(2, 3, 4)],
        },
        {
            "testcase": "transpose_4d",
            "callable": lambda x: lax.transpose(x, (0, 2, 3, 1)),
            "input_shapes": [(2, 3, 4, 5)],
        },
        {
            "testcase": "transpose_reverse",
            "callable": lambda x: lax.transpose(x, (2, 1, 0)),
            "input_shapes": [(2, 3, 4)],
        },
        {
            "testcase": "transpose_no_axes",
            "callable": lambda x: lax.transpose(x, tuple(range(x.ndim - 1, -1, -1))),
            "input_shapes": [(2, 3, 4)],
        },
        {
            "testcase": "transpose_nhwc_to_nchw",
            "callable": lambda x: lax.transpose(x, (0, 3, 1, 2)),
            "input_shapes": [(2, 28, 28, 3)],
        },
    ],
)
class TransposePlugin(PrimitiveLeafPlugin):
    """plugins2 IR converter for jax.lax.transpose → ONNX Transpose."""

    def lower(self, ctx: "IRBuildContext", eqn):
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
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("y"))

        # Emit ONNX Transpose
        ctx.add_node(
            ir.Node(
                op_type="Transpose",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Transpose"),
                attributes=[ir.Attr("perm", ir.AttributeType.INTS, list(perm))],
            )
        )

        # Stamp output shape/type by permuting the input aval shape.
        in_aval = getattr(x_var, "aval", None)
        in_shape = tuple(getattr(in_aval, "shape", ()) or ())
        if in_shape:
            out_dims: List[Any] = [in_shape[i] for i in perm]
            _stamp_type_and_shape(
                y_val, tuple(_to_ir_dim_for_shape(d) for d in out_dims)
            )
