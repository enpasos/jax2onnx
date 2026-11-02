# jax2onnx/plugins/jax/lax/dot_general.py

from __future__ import annotations

import itertools
import math
import string
from typing import Any, Iterator, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._complex_utils import (
    COMPLEX_DTYPES,
    cast_real_tensor,
    coerce_dim_values,
    ensure_packed_real_pair,
    is_packed_complex_tensor,
    pack_real_imag_pair,
    resolve_common_real_dtype,
    split_packed_real_imag,
)
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.ir_utils import numpy_dtype_to_ir


def _value_ir_dtype(value: ir.Value) -> ir.DataType | None:
    return cast(
        ir.DataType | None,
        getattr(getattr(value, "type", None), "dtype", None),
    )


@register_native_primitive(
    jaxpr_primitive=jax.lax.dot_general_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html",
    onnx=[
        {
            "component": "MatMul/Gemm",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {
            "component": "Einsum",
            "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
        },
        {
            "component": "MatMulInteger",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMulInteger.html",
        },
    ],
    since="0.2.0",
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
        {
            "testcase": "dot_double_contract",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1, 2), (0, 1)), ((), ()))
            ),
            "input_shapes": [(3, 4, 5), (4, 5, 6)],
            "post_check_onnx_graph": EG(
                ["Einsum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_batched_double_contract",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((2, 3), (1, 2)), ((0,), (0,)))
            ),
            "input_shapes": [(2, 3, 4, 5), (2, 4, 5, 6)],
            "post_check_onnx_graph": EG(
                ["Einsum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_highrank_batch",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((3,), (2,)), ((0, 1), (0, 1)))
            ),
            "input_shapes": [(2, 2, 3, 4), (2, 2, 4, 5)],
            "post_check_onnx_graph": EG(
                ["MatMul"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_contract_inner_lhs_with_middle_rhs",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (0,)), ((), ()))
            ),
            "input_shapes": [(3, 4, 5), (4, 5, 6)],
            "post_check_onnx_graph": EG(
                ["Einsum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_outer_product",
            "callable": lambda x, y: jax.lax.dot_general(x, y, (((), ()), ((), ()))),
            "input_shapes": [(3,), (4,)],
            "post_check_onnx_graph": EG(
                ["Einsum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_full_contract_scalar",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((0, 1), (0, 1)), ((), ()))
            ),
            "input_shapes": [(2, 3), (2, 3)],
            "post_check_onnx_graph": EG(
                ["Einsum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_general_complex_matmul",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (0,)), ((), ()))
            ),
            "input_values": [
                np.array(
                    [[1.0 + 2.0j, -1.5 + 0.5j], [0.25 - 0.75j, 2.0 + 1.0j]],
                    dtype=np.complex64,
                ),
                np.array(
                    [[0.5 - 1.0j, 1.0 + 0.25j], [1.5 + 0.5j, -0.75 + 2.0j]],
                    dtype=np.complex64,
                ),
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [{"path": "Einsum", "counts": {"Einsum": 4}}],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_general_int8_accumulate_int32",
            "callable": lambda x, y: jax.lax.dot_general(
                x,
                y,
                (((1,), (0,)), ((), ())),
                preferred_element_type=np.int32,
            ),
            "input_values": [
                np.array([[-128, 127], [1, -1]], dtype=np.int8),
                np.array([[127, -128], [-1, 1]], dtype=np.int8),
            ],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(["MatMulInteger:2x2"], no_unused_inputs=True),
        },
    ],
)
class DotGeneralPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.dot_general`` with dtype-specific matrix accumulation."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        params = getattr(eqn, "params", {})
        ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = params[
            "dimension_numbers"
        ]

        lhs_contract = tuple(lhs_contract)
        rhs_contract = tuple(rhs_contract)
        lhs_batch = tuple(lhs_batch)
        rhs_batch = tuple(rhs_batch)

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("dot_lhs"))
        rhs_val = ctx.get_value_for_var(rhs_var, name_hint=ctx.fresh_name("dot_rhs"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("dot_out"))

        lhs_shape = tuple(getattr(lhs_var.aval, "shape", ()))
        rhs_shape = tuple(getattr(rhs_var.aval, "shape", ()))
        out_shape = tuple(getattr(out_var.aval, "shape", ()))

        if self._maybe_lower_complex(
            ctx,
            lhs_var,
            rhs_var,
            out_var,
            lhs_val,
            rhs_val,
            out_spec,
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_contract,
            rhs_contract,
            lhs_batch,
            rhs_batch,
        ):
            return

        input_dtypes = (np.dtype(lhs_var.aval.dtype), np.dtype(rhs_var.aval.dtype))
        if all(
            dtype in (np.dtype(np.int8), np.dtype(np.uint8)) for dtype in input_dtypes
        ) and np.dtype(out_var.aval.dtype) == np.dtype(np.int32):
            self._lower_integer_matmul(
                ctx,
                lhs_val,
                rhs_val,
                out_var,
                out_spec,
                lhs_shape,
                rhs_shape,
                out_shape,
                lhs_contract,
                rhs_contract,
                lhs_batch,
                rhs_batch,
            )
            return

        if self._try_lower_matmul(
            ctx,
            lhs_var,
            rhs_var,
            out_var,
            lhs_val,
            rhs_val,
            out_spec,
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_contract,
            rhs_contract,
            lhs_batch,
            rhs_batch,
        ):
            return

        self._lower_via_einsum(
            ctx,
            lhs_var,
            rhs_var,
            out_var,
            lhs_val,
            rhs_val,
            out_spec,
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_contract,
            rhs_contract,
            lhs_batch,
            rhs_batch,
        )

    @staticmethod
    def _integer_shape_tensor(
        ctx: LoweringContextProtocol,
        groups: list[tuple[ir.Value, tuple[Any, ...], tuple[int, ...]]],
        name_hint: str,
    ) -> tuple[ir.Value, tuple[Any, ...]]:
        """Build a reshape shape from products of original operand dimensions."""
        dimensions = tuple(
            (
                math.prod(int(shape[axis]) for axis in axes)
                if all(isinstance(shape[axis], (int, np.integer)) for axis in axes)
                else None
            )
            for _, shape, axes in groups
        )
        if all(dim is not None for dim in dimensions):
            return (
                _const_i64(
                    ctx,
                    np.asarray(dimensions, dtype=np.int64),
                    ctx.fresh_name(name_hint),
                ),
                dimensions,
            )

        pieces = []
        for (value, shape, axes), dimension in zip(groups, dimensions):
            if dimension is not None:
                piece = _const_i64(
                    ctx,
                    np.asarray([dimension], dtype=np.int64),
                    ctx.fresh_name(f"{name_hint}_dim"),
                )
            else:
                source_shape = ctx.builder.op("Shape", [value])
                source_shape.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(source_shape, (len(shape),))
                indices = _const_i64(
                    ctx,
                    np.asarray(axes, dtype=np.int64),
                    ctx.fresh_name(f"{name_hint}_axes"),
                )
                selected = ctx.builder.op(
                    "Gather", [source_shape, indices], {"axis": 0}
                )
                selected.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(selected, (len(axes),))
                piece = ctx.builder.op("ReduceProd", [selected], {"keepdims": 1})
                piece.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(piece, (1,))
            _ensure_value_metadata(ctx, piece)
            pieces.append(piece)
        result = ctx.builder.op("Concat", pieces, {"axis": 0})
        result.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(result, (len(groups),))
        _ensure_value_metadata(ctx, result)
        return result, dimensions

    def _lower_integer_matmul(
        self,
        ctx: LoweringContextProtocol,
        lhs: ir.Value,
        rhs: ir.Value,
        out_var: Any,
        out_spec: ir.Value,
        lhs_shape: tuple[Any, ...],
        rhs_shape: tuple[Any, ...],
        out_shape: tuple[Any, ...],
        lhs_contract: tuple[int, ...],
        rhs_contract: tuple[int, ...],
        lhs_batch: tuple[int, ...],
        rhs_batch: tuple[int, ...],
    ) -> None:
        """Flatten arbitrary contractions without losing INT32 accumulation."""
        opset = int(getattr(ctx.builder, "opset", 21))
        if opset < 10:
            raise NotImplementedError(
                "INT8/UINT8 dot_general lowering currently requires opset >= 10 "
                f"for MatMulInteger (requested opset {opset})."
            )
        lhs_free = tuple(
            axis
            for axis in range(len(lhs_shape))
            if axis not in lhs_batch and axis not in lhs_contract
        )
        rhs_free = tuple(
            axis
            for axis in range(len(rhs_shape))
            if axis not in rhs_batch and axis not in rhs_contract
        )
        output_groups: list[tuple[ir.Value, tuple[Any, ...], tuple[int, ...]]] = [
            (lhs, lhs_shape, (axis,)) for axis in (*lhs_batch, *lhs_free)
        ]
        output_groups.extend((rhs, rhs_shape, (axis,)) for axis in rhs_free)

        if any(
            isinstance(lhs_shape[axis], (int, np.integer)) and int(lhs_shape[axis]) == 0
            for axis in lhs_contract
        ):
            # ORT's CPU MatMulInteger kernel can leave K=0 outputs uninitialized.
            # An empty contraction is the additive identity, including scalar
            # results and outputs whose free dimensions are symbolic or empty.
            shape_tensor, _ = self._integer_shape_tensor(
                ctx, output_groups, "dot_integer_zero_shape"
            )
            result = ctx.builder.op(
                "ConstantOfShape",
                [shape_tensor],
                {"value": ir.AttrTensor("value", ir.tensor(np.asarray([0], np.int32)))},
            )
            result.name = out_spec.name
            result.type = ir.TensorType(ir.DataType.INT32)
            _stamp_type_and_shape(result, out_shape)
            _ensure_value_metadata(ctx, result)
            ctx.bind_value_for_var(out_var, result)
            return

        def reshape_integer(
            value: ir.Value, shape_tensor: ir.Value, shape: tuple[Any, ...]
        ) -> ir.Value:
            if opset >= 14:
                return ctx.builder.op(
                    "Reshape", [value, shape_tensor], {"allowzero": 1}
                )
            if not all(
                isinstance(dim, (int, np.integer)) and int(dim) > 0 for dim in shape
            ):
                raise NotImplementedError(
                    "The current integer dot_general grouped-reshape lowering "
                    "requires opset >= 14 for potentially zero dimensions "
                    f"(Reshape allowzero; requested opset {opset})."
                )
            return ctx.builder.op("Reshape", [value, shape_tensor])

        def prepare(
            value: ir.Value,
            shape: tuple[Any, ...],
            batch: tuple[int, ...],
            first: tuple[int, ...],
            second: tuple[int, ...],
            prefix: str,
        ) -> tuple[ir.Value, tuple[Any, ...]]:
            permutation = (*batch, *first, *second)
            prepared = value
            if permutation != tuple(range(len(shape))):
                prepared = ctx.builder.op(
                    "Transpose",
                    [value],
                    {"perm": ir.AttrInt64s("perm", permutation)},
                )
                prepared.type = value.type
                _stamp_type_and_shape(
                    prepared, tuple(shape[axis] for axis in permutation)
                )
                _ensure_value_metadata(ctx, prepared)
            groups: list[tuple[ir.Value, tuple[Any, ...], tuple[int, ...]]] = [
                (value, shape, (axis,)) for axis in batch
            ]
            groups.extend([(value, shape, first), (value, shape, second)])
            # Keep the ordinary matrix case free of extra reshape nodes.
            if all(len(axes) == 1 for _, _, axes in groups):
                return prepared, tuple(shape[axes[0]] for _, _, axes in groups)
            shape_tensor, matrix_shape = self._integer_shape_tensor(ctx, groups, prefix)
            prepared = reshape_integer(prepared, shape_tensor, matrix_shape)
            prepared.type = value.type
            _stamp_type_and_shape(prepared, matrix_shape)
            _ensure_value_metadata(ctx, prepared)
            return prepared, matrix_shape

        lhs_matrix, lhs_matrix_shape = prepare(
            lhs, lhs_shape, lhs_batch, lhs_free, lhs_contract, "dot_integer_lhs_shape"
        )
        rhs_matrix, rhs_matrix_shape = prepare(
            rhs, rhs_shape, rhs_batch, rhs_contract, rhs_free, "dot_integer_rhs_shape"
        )

        if not isinstance(lhs_matrix_shape[-1], (int, np.integer)):
            if opset < 11:
                raise NotImplementedError(
                    "The current symbolic-contraction integer dot_general lowering "
                    "requires opset >= 11 for Pad inputs "
                    f"(requested opset {opset})."
                )

            # Symbolic expressions such as K - 1 may evaluate to zero. Appending
            # one zero product preserves the exact INT32 sum while keeping K
            # positive for ORT. Pad may copy both operands; fixed K avoids it.
            def pad_contract(
                value: ir.Value,
                shape: tuple[Any, ...],
                axis: int,
                prefix: str,
            ) -> ir.Value:
                rank = len(shape)
                padding = [0] * (2 * rank)
                padding[rank + axis] = 1
                pads = _const_i64(
                    ctx,
                    np.asarray(padding, dtype=np.int64),
                    ctx.fresh_name(f"{prefix}_pads"),
                )
                zero = ctx.builder.add_initializer_from_scalar(
                    name=ctx.fresh_name(f"{prefix}_zero"),
                    value=np.asarray(
                        0,
                        dtype=(
                            np.int8
                            if _value_ir_dtype(value) == ir.DataType.INT8
                            else np.uint8
                        ),
                    ),
                )
                padded = ctx.builder.op(
                    "Pad", [value, pads, zero], {"mode": "constant"}
                )
                padded.type = value.type
                padded_shape = list(shape)
                padded_shape[axis] = None
                _stamp_type_and_shape(padded, tuple(padded_shape))
                _ensure_value_metadata(ctx, padded)
                return padded

            lhs_matrix = pad_contract(
                lhs_matrix,
                lhs_matrix_shape,
                len(lhs_matrix_shape) - 1,
                "dot_integer_lhs",
            )
            rhs_matrix = pad_contract(
                rhs_matrix,
                rhs_matrix_shape,
                len(rhs_matrix_shape) - 2,
                "dot_integer_rhs",
            )
        matrix_shape = (
            *lhs_matrix_shape[:-2],
            lhs_matrix_shape[-2],
            rhs_matrix_shape[-1],
        )
        result = ctx.builder.op("MatMulInteger", [lhs_matrix, rhs_matrix])
        result.type = ir.TensorType(ir.DataType.INT32)
        _stamp_type_and_shape(result, matrix_shape)
        _ensure_value_metadata(ctx, result)

        if len(lhs_free) != 1 or len(rhs_free) != 1:
            shape_tensor, _ = self._integer_shape_tensor(
                ctx, output_groups, "dot_integer_output_shape"
            )
            result = reshape_integer(result, shape_tensor, out_shape)
            result.type = ir.TensorType(ir.DataType.INT32)
        result.name = out_spec.name
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    def _maybe_lower_complex(
        self,
        ctx: LoweringContextProtocol,
        lhs_var: Any,
        rhs_var: Any,
        out_var: Any,
        lhs_val: ir.Value,
        rhs_val: ir.Value,
        out_spec: ir.Value,
        lhs_shape: tuple[Any, ...],
        rhs_shape: tuple[Any, ...],
        out_shape: tuple[Any, ...],
        lhs_contract: tuple[int, ...],
        rhs_contract: tuple[int, ...],
        lhs_batch: tuple[int, ...],
        rhs_batch: tuple[int, ...],
    ) -> bool:
        def _is_complex_var(var: Any) -> bool:
            aval_dtype = getattr(getattr(var, "aval", None), "dtype", None)
            if aval_dtype is None:
                return False
            try:
                return np.issubdtype(np.dtype(aval_dtype), np.complexfloating)
            except TypeError:
                return False

        complex_var_hint = (
            _is_complex_var(out_var)
            or _is_complex_var(lhs_var)
            or _is_complex_var(rhs_var)
        )
        lhs_dtype = getattr(lhs_val, "dtype", None)
        rhs_dtype = getattr(rhs_val, "dtype", None)
        complex_dtype_hint = lhs_dtype in COMPLEX_DTYPES or rhs_dtype in COMPLEX_DTYPES
        packed_hint = False
        if complex_var_hint or complex_dtype_hint:
            packed_hint = is_packed_complex_tensor(lhs_val) or is_packed_complex_tensor(
                rhs_val
            )
        if not (complex_var_hint or complex_dtype_hint or packed_hint):
            return False

        lhs_packed, lhs_base_dtype = ensure_packed_real_pair(
            ctx, lhs_val, name_hint="dot_lhs_pack"
        )
        rhs_packed, rhs_base_dtype = ensure_packed_real_pair(
            ctx, rhs_val, name_hint="dot_rhs_pack"
        )
        target_dtype = resolve_common_real_dtype(lhs_base_dtype, rhs_base_dtype)

        lhs_ready = (
            lhs_packed
            if lhs_packed.dtype == target_dtype
            else cast_real_tensor(
                ctx, lhs_packed, target_dtype, name_hint="dot_lhs_cast"
            )
        )
        rhs_ready = (
            rhs_packed
            if rhs_packed.dtype == target_dtype
            else cast_real_tensor(
                ctx, rhs_packed, target_dtype, name_hint="dot_rhs_cast"
            )
        )

        lhs_real, lhs_imag = split_packed_real_imag(
            ctx, lhs_ready, target_dtype, prefix="dot_lhs"
        )
        rhs_real, rhs_imag = split_packed_real_imag(
            ctx, rhs_ready, target_dtype, prefix="dot_rhs"
        )

        target_batch_rank = max(len(lhs_batch), len(rhs_batch))

        lhs_real_prepped, lhs_real_shape, lhs_batch_adj, lhs_contract_adj = (
            self._prepare_operand_for_einsum(
                ctx,
                lhs_real,
                lhs_shape,
                lhs_batch,
                lhs_contract,
                target_batch_rank,
                "dot_lhs_real_pad",
            )
        )
        rhs_real_prepped, rhs_real_shape, rhs_batch_adj, rhs_contract_adj = (
            self._prepare_operand_for_einsum(
                ctx,
                rhs_real,
                rhs_shape,
                rhs_batch,
                rhs_contract,
                target_batch_rank,
                "dot_rhs_real_pad",
            )
        )
        lhs_imag_prepped, _, _, _ = self._prepare_operand_for_einsum(
            ctx,
            lhs_imag,
            lhs_shape,
            lhs_batch,
            lhs_contract,
            target_batch_rank,
            "dot_lhs_imag_pad",
        )
        rhs_imag_prepped, _, _, _ = self._prepare_operand_for_einsum(
            ctx,
            rhs_imag,
            rhs_shape,
            rhs_batch,
            rhs_contract,
            target_batch_rank,
            "dot_rhs_imag_pad",
        )

        lhs_labels, rhs_labels, out_labels = self._compute_einsum_labels(
            lhs_real_shape,
            rhs_real_shape,
            lhs_contract_adj,
            rhs_contract_adj,
            lhs_batch_adj,
            rhs_batch_adj,
        )
        equation = f"{lhs_labels},{rhs_labels}->{out_labels}"
        result_dims = coerce_dim_values(out_shape)

        def _einsum(left: ir.Value, right: ir.Value, name: str) -> ir.Value:
            value = cast(
                ir.Value,
                ctx.builder.Einsum(
                    left,
                    right,
                    equation=equation,
                    _outputs=[ctx.fresh_name(name)],
                ),
            )
            value.type = ir.TensorType(target_dtype)
            value.dtype = target_dtype
            _stamp_type_and_shape(value, result_dims)
            _ensure_value_metadata(ctx, value)
            return value

        ar_br = _einsum(lhs_real_prepped, rhs_real_prepped, "dot_ar_br")
        ai_bi = _einsum(lhs_imag_prepped, rhs_imag_prepped, "dot_ai_bi")
        ar_bi = _einsum(lhs_real_prepped, rhs_imag_prepped, "dot_ar_bi")
        ai_br = _einsum(lhs_imag_prepped, rhs_real_prepped, "dot_ai_br")

        real_part = ctx.builder.Sub(
            ar_br,
            ai_bi,
            _outputs=[ctx.fresh_name("dot_real_part")],
        )
        real_part.type = ir.TensorType(target_dtype)
        real_part.dtype = target_dtype
        _stamp_type_and_shape(real_part, result_dims)
        _ensure_value_metadata(ctx, real_part)

        imag_part = ctx.builder.Add(
            ar_bi,
            ai_br,
            _outputs=[ctx.fresh_name("dot_imag_part")],
        )
        imag_part.type = ir.TensorType(target_dtype)
        imag_part.dtype = target_dtype
        _stamp_type_and_shape(imag_part, result_dims)
        _ensure_value_metadata(ctx, imag_part)

        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("dot_out")
        packed = pack_real_imag_pair(
            ctx,
            real_part,
            imag_part,
            target_dtype,
            name_hint="dot_output",
            output_name=out_name,
        )

        out_spec.type = ir.TensorType(target_dtype)
        out_spec.dtype = target_dtype
        if getattr(packed, "shape", None) is not None:
            out_spec.shape = packed.shape
        _ensure_value_metadata(ctx, packed)
        ctx.bind_value_for_var(out_var, packed)
        return True

    def _try_lower_matmul(
        self,
        ctx: LoweringContextProtocol,
        lhs_var: Any,
        rhs_var: Any,
        out_var: Any,
        lhs_val: ir.Value,
        rhs_val: ir.Value,
        out_spec: ir.Value,
        lhs_shape: tuple[Any, ...],
        rhs_shape: tuple[Any, ...],
        out_shape: tuple[Any, ...],
        lhs_contract: tuple[int, ...],
        rhs_contract: tuple[int, ...],
        lhs_batch: tuple[int, ...],
        rhs_batch: tuple[int, ...],
    ) -> bool:
        if lhs_batch or rhs_batch:
            if tuple(lhs_batch) != tuple(rhs_batch):
                return False
            if len(lhs_contract) != 1 or len(rhs_contract) != 1:
                return False

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
                return False

            lhs_perm = list(lhs_batch) + lhs_free_axes + [lhs_contract_axis]
            rhs_perm = list(rhs_batch) + [rhs_contract_axis] + rhs_free_axes

            def _transpose_if_needed(
                value: ir.Value,
                perm: list[int],
                original_shape: tuple[Any, ...],
                name_hint: str,
            ) -> tuple[ir.Value, tuple[Any, ...]]:
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

            lhs_prepped, _ = _transpose_if_needed(
                lhs_val, lhs_perm, lhs_shape, "dot_lhs_perm"
            )
            rhs_prepped, _ = _transpose_if_needed(
                rhs_val, rhs_perm, rhs_shape, "dot_rhs_perm"
            )

            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("MatMul")
            result = ctx.builder.MatMul(
                lhs_prepped,
                rhs_prepped,
                _outputs=[desired_name],
            )
            out_dtype: np.dtype[Any] = np.dtype(
                getattr(
                    out_var.aval,
                    "dtype",
                    getattr(lhs_var.aval, "dtype", np.float32),
                )
            )
            result_dtype = (
                _value_ir_dtype(lhs_prepped)
                or _value_ir_dtype(rhs_prepped)
                or numpy_dtype_to_ir(out_dtype)
            )
            result.type = ir.TensorType(result_dtype)
            _stamp_type_and_shape(result, out_shape)
            _ensure_value_metadata(ctx, result)
            ctx.bind_value_for_var(out_var, result)
            return True

        if len(lhs_contract) != 1 or len(rhs_contract) != 1:
            return False
        if len(lhs_batch) != 0 or len(rhs_batch) != 0:
            return False
        if len(lhs_shape) != 2 or len(rhs_shape) != 2:
            return False

        rhs_contract_axis = rhs_contract[0]
        rhs_rank = len(rhs_shape)
        if rhs_contract_axis not in (0, rhs_rank - 1):
            return False

        transpose_rhs = rhs_contract_axis == rhs_rank - 1
        rhs_input = rhs_val
        if transpose_rhs:
            perm = list(range(rhs_rank))
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
        result_dtype = (
            _value_ir_dtype(lhs_val)
            or _value_ir_dtype(rhs_input)
            or numpy_dtype_to_ir(out_dtype)
        )
        result.type = ir.TensorType(result_dtype)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
        return True

    def _lower_via_einsum(
        self,
        ctx: LoweringContextProtocol,
        lhs_var: Any,
        rhs_var: Any,
        out_var: Any,
        lhs_val: ir.Value,
        rhs_val: ir.Value,
        out_spec: ir.Value,
        lhs_shape: tuple[Any, ...],
        rhs_shape: tuple[Any, ...],
        out_shape: tuple[Any, ...],
        lhs_contract: tuple[int, ...],
        rhs_contract: tuple[int, ...],
        lhs_batch: tuple[int, ...],
        rhs_batch: tuple[int, ...],
    ) -> None:
        lhs_shape = tuple(lhs_shape)
        rhs_shape = tuple(rhs_shape)
        lhs_contract = tuple(lhs_contract)
        rhs_contract = tuple(rhs_contract)
        lhs_batch = tuple(lhs_batch)
        rhs_batch = tuple(rhs_batch)

        target_batch_rank = max(len(lhs_batch), len(rhs_batch))

        lhs_val_prepped, lhs_shape_prepped, lhs_batch_adj, lhs_contract_adj = (
            self._prepare_operand_for_einsum(
                ctx,
                lhs_val,
                lhs_shape,
                lhs_batch,
                lhs_contract,
                target_batch_rank,
                "dot_lhs_batch_pad",
            )
        )
        rhs_val_prepped, rhs_shape_prepped, rhs_batch_adj, rhs_contract_adj = (
            self._prepare_operand_for_einsum(
                ctx,
                rhs_val,
                rhs_shape,
                rhs_batch,
                rhs_contract,
                target_batch_rank,
                "dot_rhs_batch_pad",
            )
        )

        if len(lhs_contract_adj) != len(rhs_contract_adj):
            raise TypeError(
                "dot_general requires equal numbers of contracting dims on LHS/RHS"
            )

        for i, j in zip(lhs_batch_adj, rhs_batch_adj):
            lhs_dim = lhs_shape_prepped[i]
            rhs_dim = rhs_shape_prepped[j]
            if (
                lhs_dim is not None
                and rhs_dim is not None
                and lhs_dim not in (rhs_dim, 1)
                and rhs_dim not in (lhs_dim, 1)
            ):
                raise ValueError(
                    f"Batch dim mismatch: lhs[{i}]={lhs_dim} vs rhs[{j}]={rhs_dim}"
                )

        for i, j in zip(lhs_contract_adj, rhs_contract_adj):
            lhs_dim = lhs_shape_prepped[i]
            rhs_dim = rhs_shape_prepped[j]
            if lhs_dim is not None and rhs_dim is not None and lhs_dim != rhs_dim:
                raise ValueError(
                    f"Contract dim mismatch: lhs[{i}]={lhs_dim} vs rhs[{j}]={rhs_dim}"
                )

        lhs_labels, rhs_labels, out_labels = self._compute_einsum_labels(
            lhs_shape_prepped,
            rhs_shape_prepped,
            lhs_contract_adj,
            rhs_contract_adj,
            lhs_batch_adj,
            rhs_batch_adj,
        )
        equation = f"{lhs_labels},{rhs_labels}->{out_labels}"

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Einsum")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Einsum")

        result = ctx.builder.Einsum(
            lhs_val_prepped,
            rhs_val_prepped,
            equation=equation,
            _outputs=[desired_name],
        )

        inferred_dtype = next(
            (
                _value_ir_dtype(v)
                for v in (lhs_val_prepped, rhs_val_prepped, lhs_val, rhs_val)
                if _value_ir_dtype(v) is not None
            ),
            None,
        )
        if inferred_dtype is not None:
            result.type = ir.TensorType(inferred_dtype)
        else:
            aval_dtype = getattr(out_var.aval, "dtype", None)
            if aval_dtype is not None:
                result.type = ir.TensorType(numpy_dtype_to_ir(np.dtype(aval_dtype)))
            else:
                spec_type = getattr(out_spec, "type", None)
                if spec_type is not None:
                    result.type = spec_type

        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    def _prepare_operand_for_einsum(
        self,
        ctx: LoweringContextProtocol,
        value: ir.Value,
        shape: tuple[int, ...],
        batch_axes: tuple[int, ...],
        contract_axes: tuple[int, ...],
        target_batch_rank: int,
        name_hint: str,
    ) -> tuple[ir.Value, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        shape = tuple(shape)
        batch_axes = tuple(batch_axes)
        contract_axes = tuple(contract_axes)
        prepared = value
        prepared_shape = shape
        prepared_batch = batch_axes
        prepared_contract = contract_axes
        if len(batch_axes) < target_batch_rank:
            prepared, prepared_shape, prepared_batch, prepared_contract = (
                self._pad_operand_front(
                    ctx,
                    prepared,
                    prepared_shape,
                    prepared_batch,
                    prepared_contract,
                    target_batch_rank - len(batch_axes),
                    name_hint,
                )
            )
        return prepared, prepared_shape, prepared_batch, prepared_contract

    def _compute_einsum_labels(
        self,
        lhs_shape: tuple[int, ...],
        rhs_shape: tuple[int, ...],
        lhs_contract: tuple[int, ...],
        rhs_contract: tuple[int, ...],
        lhs_batch: tuple[int, ...],
        rhs_batch: tuple[int, ...],
    ) -> tuple[str, str, str]:
        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)

        lhs_batch_set = set(lhs_batch)
        rhs_batch_set = set(rhs_batch)
        lhs_contract_set = set(lhs_contract)
        rhs_contract_set = set(rhs_contract)

        lhs_free = [
            axis
            for axis in range(lhs_rank)
            if axis not in lhs_batch_set | lhs_contract_set
        ]
        rhs_free = [
            axis
            for axis in range(rhs_rank)
            if axis not in rhs_batch_set | rhs_contract_set
        ]

        main_gen = self._label_stream(string.ascii_lowercase + string.ascii_uppercase)
        batch_gen = self._label_stream(
            string.ascii_lowercase[::-1] + string.ascii_uppercase[::-1]
        )

        lhs_lbl: list[str | None] = [None] * lhs_rank
        rhs_lbl: list[str | None] = [None] * rhs_rank

        batch_pairs = list(zip(lhs_batch, rhs_batch))
        contract_pairs = list(zip(lhs_contract, rhs_contract))
        lhs_for_rhs_contract = {
            rhs_axis: lhs_axis for lhs_axis, rhs_axis in contract_pairs
        }

        for i, j in batch_pairs:
            label = next(batch_gen)
            lhs_lbl[i] = label
            rhs_lbl[j] = label

        for i in range(lhs_rank):
            if i in lhs_batch_set:
                continue
            if lhs_lbl[i] is None:
                lhs_lbl[i] = next(main_gen)

        for j in range(rhs_rank):
            if j in rhs_batch_set:
                continue
            if j in rhs_contract_set:
                lhs_axis = lhs_for_rhs_contract[j]
                rhs_lbl[j] = lhs_lbl[lhs_axis]
            else:
                if rhs_lbl[j] is None:
                    rhs_lbl[j] = next(main_gen)

        rhs_out_order: list[str | None] = []
        for axis in lhs_batch:
            rhs_out_order.append(lhs_lbl[axis])
        for axis in lhs_free:
            rhs_out_order.append(lhs_lbl[axis])
        for axis in rhs_free:
            rhs_out_order.append(rhs_lbl[axis])

        if not lhs_contract and not rhs_contract:
            rhs_out_order = lhs_lbl + rhs_lbl

        if not lhs_free and not rhs_free:
            if lhs_batch:
                rhs_out_order = [lhs_lbl[i] for i in lhs_batch]
            else:
                rhs_out_order = []

        if any(label is None for label in lhs_lbl):
            raise RuntimeError(f"Unlabeled LHS axes: {lhs_lbl}")
        if any(label is None for label in rhs_lbl):
            raise RuntimeError(f"Unlabeled RHS axes: {rhs_lbl}")

        lhs_labels = "".join(cast(list[str], lhs_lbl))
        rhs_labels = "".join(cast(list[str], rhs_lbl))
        out_labels = "".join(cast(list[str], rhs_out_order))
        return lhs_labels, rhs_labels, out_labels

    def _pad_operand_front(
        self,
        ctx: LoweringContextProtocol,
        value: ir.Value,
        shape: tuple[int, ...],
        batch_axes: tuple[int, ...],
        contract_axes: tuple[int, ...],
        pad_count: int,
        name_hint: str,
    ) -> tuple[ir.Value, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        if pad_count <= 0:
            return value, shape, batch_axes, contract_axes
        axes = tuple(range(pad_count))
        padded, new_shape = self._unsqueeze_value(ctx, value, shape, axes, name_hint)
        batch_axes = tuple(range(pad_count)) + tuple(
            axis + pad_count for axis in batch_axes
        )
        contract_axes = tuple(axis + pad_count for axis in contract_axes)
        return padded, new_shape, batch_axes, contract_axes

    @staticmethod
    def _unsqueeze_value(
        ctx: LoweringContextProtocol,
        value: ir.Value,
        shape: tuple[int, ...],
        axes: tuple[int, ...],
        name_hint: str,
    ) -> tuple[ir.Value, tuple[int, ...]]:
        if not axes:
            return value, shape
        axes = tuple(sorted(axes))
        axes_const = _const_i64(
            ctx,
            np.asarray(axes, dtype=np.int64),
            ctx.fresh_name(f"{name_hint}_axes"),
        )
        unsqueezed = ctx.builder.Unsqueeze(
            value,
            axes_const,
            _outputs=[ctx.fresh_name(name_hint)],
        )
        val_dtype = getattr(getattr(value, "type", None), "dtype", None)
        if val_dtype is not None:
            unsqueezed.type = ir.TensorType(val_dtype)
        new_shape = list(shape)
        for axis in axes:
            new_shape.insert(axis, 1)
        _stamp_type_and_shape(unsqueezed, tuple(new_shape))
        _ensure_value_metadata(ctx, unsqueezed)
        return unsqueezed, tuple(new_shape)

    @staticmethod
    def _label_stream(alphabet: str) -> Iterator[str]:
        length = 1
        while True:
            for combo in itertools.product(alphabet, repeat=length):
                yield "".join(combo)
            length += 1
