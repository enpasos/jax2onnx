# jax2onnx/plugins/jax/lax/dot_general.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


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
    context="primitives.lax",
    component="dot_general",
    testcases=[
        {
            "testcase": "dot_contract_nm",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (0,)), ((), ()))
            ),
            "input_shapes": [(3, 4), (4, 2)],
            "post_check_onnx_graph": EG(
                [
                    {"path": "Gemm:3x2", "inputs": {2: {"const": 0.0}}},
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_contract_min",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (1,)), ((), ()))
            ),
            "input_shapes": [(3, 4), (2, 4)],
            "post_check_onnx_graph": EG(
                [
                    {"path": "Gemm:3x2", "inputs": {2: {"const": 0.0}}},
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_general",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (0,)), ((), ()))
            ),
            "input_shapes": [(3, 3), (3, 3)],
            "post_check_onnx_graph": EG(
                [
                    {"path": "Gemm:3x3", "inputs": {2: {"const": 0.0}}},
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_general_lhs1_rhs1",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (1,)), ((), ()))
            ),
            "input_shapes": [(3, 3), (3, 3)],
            "post_check_onnx_graph": EG(
                [
                    {"path": "Gemm:3x3", "inputs": {2: {"const": 0.0}}},
                ],
                no_unused_inputs=True,
            ),
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

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("dot_lhs"))
        rhs_val = ctx.get_value_for_var(rhs_var, name_hint=ctx.fresh_name("dot_rhs"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("dot_out"))

        lhs_shape = tuple(getattr(lhs_var.aval, "shape", ()))
        rhs_shape = tuple(getattr(rhs_var.aval, "shape", ()))
        out_shape = tuple(getattr(out_var.aval, "shape", ()))

        if lhs_batch or rhs_batch:
            if tuple(lhs_batch) != tuple(rhs_batch):
                raise NotImplementedError(
                    "dot_general batch dimensions must match between operands"
                )
            if len(lhs_contract) != 1 or len(rhs_contract) != 1:
                raise NotImplementedError(
                    "dot_general batching supports a single contract axis"
                )

            lhs_contract_axis = lhs_contract[0]
            rhs_contract_axis = rhs_contract[0]
            lhs_rank = len(lhs_shape)
            rhs_rank = len(rhs_shape)

            lhs_free_axes = [
                axis
                for axis in range(lhs_rank)
                if axis not in lhs_batch and axis != lhs_contract_axis
            ]
            rhs_free_axes = [
                axis
                for axis in range(rhs_rank)
                if axis not in rhs_batch and axis != rhs_contract_axis
            ]

            if len(lhs_free_axes) != 1 or len(rhs_free_axes) != 1:
                raise NotImplementedError(
                    "dot_general batching currently supports rank-2 matrix dims"
                )

            lhs_perm = list(lhs_batch) + lhs_free_axes + [lhs_contract_axis]
            rhs_perm = list(rhs_batch) + [rhs_contract_axis] + rhs_free_axes

            def _transpose_if_needed(
                value, perm, original_shape, name_hint: str
            ) -> tuple:
                if perm == list(range(len(original_shape))):
                    return value, original_shape
                permuted = ctx.builder.Transpose(
                    value,
                    _outputs=[ctx.fresh_name(name_hint)],
                    perm=perm,
                )
                val_dtype = getattr(getattr(value, "type", None), "dtype", None)
                if val_dtype is not None:
                    permuted.type = ir.TensorType(val_dtype)
                perm_shape = tuple(original_shape[i] for i in perm)
                _stamp_type_and_shape(permuted, perm_shape)
                _ensure_value_metadata(ctx, permuted)
                return permuted, perm_shape

            lhs_prepped, lhs_prep_shape = _transpose_if_needed(
                lhs_val, lhs_perm, lhs_shape, "dot_lhs_perm"
            )
            rhs_prepped, rhs_prep_shape = _transpose_if_needed(
                rhs_val, rhs_perm, rhs_shape, "dot_rhs_perm"
            )

            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("MatMul")
            result = ctx.builder.MatMul(
                lhs_prepped,
                rhs_prepped,
                _outputs=[desired_name],
            )
            out_dtype = np.dtype(
                getattr(
                    out_var.aval,
                    "dtype",
                    getattr(lhs_var.aval, "dtype", np.float32),
                )
            )
            result.type = ir.TensorType(
                _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
            )
            _stamp_type_and_shape(result, out_shape)
            _ensure_value_metadata(ctx, result)
            ctx.bind_value_for_var(out_var, result)
            return

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
            rhs_perm_shape = tuple(rhs_shape[i] for i in perm)
            transposed = ctx.builder.Transpose(
                rhs_val,
                _outputs=[ctx.fresh_name("dot_rhs_T")],
                perm=perm,
            )
            rhs_dtype = getattr(getattr(rhs_val, "type", None), "dtype", None)
            if rhs_dtype is not None:
                transposed.type = ir.TensorType(rhs_dtype)
            _stamp_type_and_shape(transposed, rhs_perm_shape)
            _ensure_value_metadata(ctx, transposed)
            rhs_input = transposed

        out_dtype = np.dtype(
            getattr(out_var.aval, "dtype", getattr(lhs_var.aval, "dtype", np.float32))
        )
        bias_val = ctx.builder.add_initializer_from_scalar(
            name=ctx.fresh_name("dot_bias"),
            value=np.array(0, dtype=out_dtype),
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Gemm")
        result = ctx.builder.Gemm(
            lhs_val,
            rhs_input,
            bias_val,
            alpha=1.0,
            beta=0.0,
            _outputs=[desired_name],
        )

        _stamp_type_and_shape(result, out_shape)
        result.type = ir.TensorType(
            _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
        )
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
