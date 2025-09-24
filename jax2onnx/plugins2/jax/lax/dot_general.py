from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.converter2.ir_builder import _dtype_to_ir
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.dot_general_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html",
    onnx=[
        {
            "component": "MatMul/Gemm",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        }
    ],
    since="v0.2.0",
    context="primitives2.lax",
    component="dot_general",
    testcases=[
        {
            "testcase": "dot_contract_nm",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (0,)), ((), ()))
            ),
            "input_shapes": [(3, 4), (4, 2)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "dot_contract_min",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (1,)), ((), ()))
            ),
            "input_shapes": [(3, 4), (2, 4)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "dot_general",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (0,)), ((), ()))
            ),
            "input_shapes": [(3, 3), (3, 3)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "dot_general_lhs1_rhs1",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (1,)), ((), ()))
            ),
            "input_shapes": [(3, 3), (3, 3)],
            "use_onnx_ir": True,
        },
    ],
)
class DotGeneralPlugin(PrimitiveLeafPlugin):
    """Lower a subset of ``lax.dot_general`` patterns to Gemm/MatMul."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        params = getattr(eqn, "params", {})
        ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = params[
            "dimension_numbers"
        ]

        if lhs_batch or rhs_batch:
            raise NotImplementedError(
                "Batched dot_general not yet supported in plugins2"
            )

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("dot_lhs"))
        rhs_val = ctx.get_value_for_var(rhs_var, name_hint=ctx.fresh_name("dot_rhs"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("dot_out"))

        tuple(getattr(lhs_var.aval, "shape", ()))
        rhs_shape = tuple(getattr(rhs_var.aval, "shape", ()))
        out_shape = tuple(getattr(out_var.aval, "shape", ()))

        def _resolve_contract_pair():
            # Supported cases mirror the legacy plugin: single axis contraction.
            if lhs_contract == (1,) and rhs_contract == (0,):
                return False  # no transpose
            if lhs_contract == (1,) and rhs_contract == (1,):
                return True  # only RHS transpose needed
            raise NotImplementedError(
                f"dot_general contraction {lhs_contract}/{rhs_contract} not supported"
            )

        transpose_rhs = _resolve_contract_pair()

        rhs_input = rhs_val
        if transpose_rhs:
            perm = list(range(len(rhs_shape)))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            transposed = ir.Value(
                name=ctx.fresh_name("dot_rhs_T"),
                type=rhs_val.type,
                shape=ir.Shape(tuple(rhs_shape[i] for i in perm)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Transpose",
                    domain="",
                    inputs=[rhs_val],
                    outputs=[transposed],
                    name=ctx.fresh_name("Transpose"),
                    attributes=[IRAttr("perm", IRAttrType.INTS, perm)],
                )
            )
            _stamp_type_and_shape(transposed, tuple(rhs_shape[i] for i in perm))
            _ensure_value_info(ctx, transposed)
            rhs_input = transposed

        out_dtype = np.dtype(
            getattr(out_var.aval, "dtype", getattr(lhs_var.aval, "dtype", np.float32))
        )
        bias_val = ctx.builder.add_initializer_from_scalar(
            ctx.fresh_name("dot_bias"), np.array(0, dtype=out_dtype)
        )

        node = ir.Node(
            op_type="Gemm",
            domain="",
            inputs=[lhs_val, rhs_input, bias_val],
            outputs=[out_val],
            name=ctx.fresh_name("Gemm"),
            attributes=[
                IRAttr("alpha", IRAttrType.FLOAT, 1.0),
                IRAttr("beta", IRAttrType.FLOAT, 0.0),
            ],
        )
        ctx.add_node(node)

        _stamp_type_and_shape(out_val, out_shape)
        out_val.type = ir.TensorType(
            _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
        )
        _ensure_value_info(ctx, out_val)
