# jax2onnx/plugins/jax/lax/dot_general.py

from __future__ import annotations

import itertools
import string
from typing import Any, Iterator, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import is_native_precision_pair
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_deployment import (
    MatrixPrecisionRequirement,
    matrix_precision_contract,
)
from jax2onnx.converter.source_profile import (
    _Site,
    _tensor,
    _validate_dot,
    _dot_contraction_plan,
    _is_direct_tensor_matrix_dot,
    _is_mixed_fp64_dot_route,
    _is_mixed_fp32_dot_route,
)
from jax2onnx.converter.source_representation import (
    representation_for_aval,
    normalized_shape_dimensions,
    source_shape_contract,
    native_shape_contract,
    validate_source_shapes,
)
from jax2onnx.converter.typing_support import (
    LoweringContextProtocol,
    NativeLoweringContext,
)
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
    ],
)
class DotGeneralPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.dot_general`` via MatMul/Gemm fast-path or Einsum fallback."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        if ctx.precision_policy is PrecisionPolicy.SOURCE:
            matrix_output = self._lower_source(ctx, eqn)
            contract = matrix_precision_contract(
                ctx.deployment_target,
                eqn.params.get("precision"),
                tuple(
                    np.dtype(var.aval.dtype).name for var in (*eqn.invars, *eqn.outvars)
                ),
                (
                    None
                    if eqn.params.get("preferred_element_type") is None
                    else np.dtype(eqn.params["preferred_element_type"]).name
                ),
            )
            if contract is not None:
                if matrix_output is None:
                    raise NotImplementedError(
                        "SOURCE target precision lowering remains unfinished for a non-MatMul route"
                    )
                output = ctx.require_value_for_var(eqn.outvars[0])
                ctx.record_matrix_precision_requirement(
                    MatrixPrecisionRequirement(
                        source_scope=ctx.source_scope,
                        source_equation=str(eqn),
                        output=output.name or "pending",
                        matrix_output=matrix_output.name,
                        source_precision=("HIGHEST", "HIGHEST"),
                        dtypes=("float32", "float32", "float32"),
                        preferred_element_type=(
                            None
                            if eqn.params.get("preferred_element_type") is None
                            else "float32"
                        ),
                        kernel_contract=contract,
                    ),
                    output,
                    matrix_output,
                )
            return

        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        params = getattr(eqn, "params", {})
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = params[
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

    def _lower_source(self, ctx: NativeLoweringContext, eqn: Any) -> ir.Value | None:
        """Preserve native axes and declared widths; numerical evidence is separate.

        Finite FP16 operand values and products are exactly representable in
        FP32. Native mixed FP32/FP64 CPU calls locally convert the operand of a
        different width to their declared FP64 or explicit FP32 result dtype
        before computing, matching JAX.
        Original operand bindings remain unchanged. No explicit native
        DotAlgorithm or backend arithmetic is reconstructed.
        """
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]
        params = eqn.params
        unknown = set(params) - {
            "dimension_numbers",
            "precision",
            "preferred_element_type",
            "out_sharding",
        }
        if unknown:
            raise NotImplementedError(
                "SOURCE dot_general: unreviewed semantic parameters "
                f"{sorted(unknown)} are outside the audited private subset"
            )
        if params.get("out_sharding") is not None:
            raise NotImplementedError(
                "SOURCE dot_general: non-None out_sharding is outside the audited "
                "private subset; only unspecified placement is covered"
            )
        precision = params.get("precision")
        default_pair = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
        dtypes = tuple(np.dtype(var.aval.dtype) for var in (*eqn.invars, out_var))
        preferred = params.get("preferred_element_type")
        mixed_fp64 = _is_mixed_fp64_dot_route(
            tuple(dtype.name for dtype in dtypes[:2]) + ("float64",), preferred
        )
        mixed_fp32 = _is_mixed_fp32_dot_route(
            tuple(dtype.name for dtype in dtypes[:2]) + ("float32",), preferred
        )
        native_fp64 = mixed_fp64 or (
            all(dtype == np.dtype(np.float64) for dtype in dtypes)
            and (preferred is None or np.dtype(preferred) == np.dtype(np.float64))
        )
        if (
            precision is not None
            and precision != default_pair
            and not (native_fp64 and is_native_precision_pair(precision))
            and matrix_precision_contract(
                ctx.deployment_target,
                precision,
                tuple(dtype.name for dtype in dtypes),
                None if preferred is None else np.dtype(preferred).name,
            )
            is None
        ):
            raise NotImplementedError(
                "SOURCE dot_general: precision "
                f"{precision!r} is outside the audited private subset; "
                "default precision or native FP64-compute Precision enum pairs are covered, without an explicit algorithm"
            )

        same_width = (
            dtypes[0] in (np.dtype(np.float32), np.dtype(np.float64))
            and dtypes[0] == dtypes[1] == dtypes[2]
            and (preferred is None or np.dtype(preferred) == dtypes[2])
        )
        widen = (
            dtypes == (np.dtype(np.float16), np.dtype(np.float16), np.dtype(np.float32))
            and preferred is not None
            and np.dtype(preferred) == np.dtype(np.float32)
        )
        if not (same_width or widen or mixed_fp64 or mixed_fp32):
            raise NotImplementedError(
                "SOURCE dot_general: operand/result types "
                f"{dtypes} and preferred_element_type={preferred} are outside "
                "the audited private subset; same-width FP32/FP64, native mixed "
                "FP32/FP64 with FP64 or explicitly preferred FP32 work, or FP16 operands with an explicitly preferred "
                "FP32 result are covered. "
                "BF16 schema support does not establish its native accumulation/provider contract; "
                "integer, Boolean and packed-complex arithmetic require their own proof"
            )
        site = _Site(
            "SOURCE dot_general lowering",
            ctx.builder.opset,
            "dot_general",
            deployment_target=ctx.deployment_target,
        )
        _validate_dot(
            eqn,
            [_tensor(var, site) for var in eqn.invars],
            [_tensor(var, site) for var in eqn.outvars],
            site,
        )
        native_shapes = [tuple(var.aval.shape) for var in (*eqn.invars, out_var)]
        contracts = [native_shape_contract(shape) for shape in native_shapes]
        dtype = numpy_dtype_to_ir(dtypes[2])
        if (
            same_width
            and all(contract.symbol is None for contract in contracts)
            and _is_direct_tensor_matrix_dot(
                native_shapes[0], native_shapes[1], params["dimension_numbers"]
            )
        ):
            return self._lower_source_tensor_matrix(ctx, eqn, dtype)
        symbolic = any(contract.symbol is not None for contract in contracts)
        if (
            same_width
            and symbolic
            and all(
                contract.symbol is None
                or contract.symbol_axis == 0
                and all(isinstance(dim, int) and dim > 0 for dim in contract.shape[1:])
                for contract in contracts
            )
        ):
            dimensions = params["dimension_numbers"]
            if (
                contracts[0].symbol is not None
                and contracts[1].symbol is None
                and len(native_shapes[0]) in (2, 3, 4)
                and len(native_shapes[1]) == 2
                and dimensions == (((len(native_shapes[0]) - 1,), (0,)), ((), ()))
            ):
                # Keep the established one-MatMul tensor-by-static-matrix graph.
                return self._lower_source_tensor_matrix(ctx, eqn, dtype)
            if (
                len(native_shapes[0]) == len(native_shapes[1]) == 3
                and contracts[0].symbol == contracts[1].symbol
                and dimensions == (((2,), (1,)), ((0,), (0,)))
            ):
                return self._lower_source_leading_batch(ctx, eqn, dtype)
        plan = _dot_contraction_plan(
            native_shapes[0], native_shapes[1], params["dimension_numbers"]
        )
        operands = [ctx.require_value_for_var(var) for var in eqn.invars]
        for var, value in zip(eqn.invars, operands, strict=True):
            if not representation_for_aval(var.aval).matches_ir_value(value):
                raise ValueError(
                    "SOURCE dot_general: physical matrix dtype/shape differs from aval"
                )
        output = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("dot_contraction")
        )
        if not representation_for_aval(out_var.aval).matches_ir_value(output):
            raise ValueError(
                "SOURCE dot_general: physical matrix dtype/shape differs from aval"
            )

        def temporary(
            name: str, shape: tuple[int | str, ...], *, kind: ir.DataType = dtype
        ) -> ir.Value:
            return ir.Value(
                name=ctx.fresh_name(name),
                type=ir.TensorType(kind),
                shape=ir.Shape(shape),
            )

        batch_extent: ir.Value | None = None

        def runtime_shape(shape: tuple[int | str, ...]) -> ir.Value:
            nonlocal batch_extent
            if not any(isinstance(dim, str) for dim in shape):
                return _const_i64(ctx, np.asarray(shape, dtype=np.int64), "dot_shape")
            if plan.symbolic_operand is None:
                raise ValueError(
                    "SOURCE dot_general: runtime shape lacks native B provenance"
                )
            if batch_extent is None:
                side, axis = plan.symbolic_operand
                source_shape = ctx.builder.op(
                    "Shape",
                    [operands[side]],
                    output=temporary(
                        "dot_operand_shape",
                        (len(native_shapes[side]),),
                        kind=ir.DataType.INT64,
                    ),
                )
                index = _const_i64(
                    ctx, np.asarray([axis], dtype=np.int64), "dot_batch_axis"
                )
                batch_extent = ctx.builder.op(
                    "Gather",
                    [source_shape, index],
                    {"axis": 0},
                    output=temporary("dot_batch_extent", (1,), kind=ir.DataType.INT64),
                )
            pieces = [
                (
                    batch_extent
                    if isinstance(dim, str)
                    else _const_i64(
                        ctx, np.asarray([dim], dtype=np.int64), "dot_static_extent"
                    )
                )
                for dim in shape
            ]
            if len(pieces) == 1:
                return pieces[0]
            return ctx.builder.op(
                "Concat",
                pieces,
                {"axis": 0},
                output=temporary(
                    "dot_dynamic_shape", (len(shape),), kind=ir.DataType.INT64
                ),
            )

        # An empty free/static batch axis has no output elements; an empty
        # contraction instead produces the native positive-zero sum identity.
        if 0 in plan.output_shape or plan.contracting_size == 0:
            shape = runtime_shape(plan.output_shape)
            zero = ir.tensor(np.zeros((1,), dtype=dtypes[2]))
            result = ctx.builder.op(
                "ConstantOfShape",
                [shape],
                {"value": ir.AttrTensor("value", zero)},
                output=output,
            )
            ctx.bind_value_for_var(out_var, result)
            return None

        def normalize(
            value: ir.Value,
            shape: tuple[int | str, ...],
            permutation: tuple[int, ...],
            matrix_shape: tuple[int | str, ...],
        ) -> ir.Value:
            if (widen or mixed_fp64 or mixed_fp32) and _value_ir_dtype(value) != dtype:
                value = ctx.builder.op(
                    "Cast",
                    [value],
                    {"to": int(dtype)},
                    output=temporary(
                        "dot_narrow" if mixed_fp32 else "dot_widen", shape
                    ),
                )
            if permutation != tuple(range(len(shape))):
                shape = tuple(shape[axis] for axis in permutation)
                value = ctx.builder.op(
                    "Transpose",
                    [value],
                    {"perm": ir.AttrInt64s("perm", permutation)},
                    output=temporary("dot_transpose", shape),
                )
            if shape != matrix_shape:
                value = ctx.builder.op(
                    "Reshape",
                    [value, runtime_shape(matrix_shape)],
                    {"allowzero": 1},
                    output=temporary("dot_reshape", matrix_shape),
                )
            return value

        lhs = normalize(
            operands[0], contracts[0].shape, plan.lhs_permutation, plan.lhs_matrix_shape
        )
        rhs = normalize(
            operands[1], contracts[1].shape, plan.rhs_permutation, plan.rhs_matrix_shape
        )
        identity = plan.output_permutation == tuple(range(len(plan.output_shape)))
        matrix_output = (
            output
            if plan.output_shape == plan.matrix_output_shape and identity
            else temporary("dot_matrix", plan.matrix_output_shape)
        )
        # No contracting axis means one native product, without an invented +0.
        result = ctx.builder.op(
            "Mul" if plan.outer_product else "MatMul", [lhs, rhs], output=matrix_output
        )
        physical_matrix = None if plan.outer_product else result
        if plan.matrix_output_shape != plan.ungrouped_output_shape:
            result = ctx.builder.op(
                "Reshape",
                [result, runtime_shape(plan.ungrouped_output_shape)],
                {"allowzero": 1},
                output=(
                    output
                    if identity
                    else temporary("dot_ungrouped", plan.ungrouped_output_shape)
                ),
            )
        if not identity:
            result = ctx.builder.op(
                "Transpose",
                [result],
                {"perm": ir.AttrInt64s("perm", plan.output_permutation)},
                output=output,
            )
        ctx.bind_value_for_var(out_var, result)
        return physical_matrix

    def _lower_source_tensor_matrix(
        self, ctx: NativeLoweringContext, eqn: Any, dtype: ir.DataType
    ) -> ir.Value:
        """Emit one typed MatMul for a proven canonical tensor-matrix call."""
        native_shapes = [tuple(var.aval.shape) for var in (*eqn.invars, *eqn.outvars)]
        contracts = [source_shape_contract(shape) for shape in native_shapes]
        (lhs_shape, lhs_symbol), (rhs_shape, rhs_symbol), (out_shape, out_symbol) = (
            contracts
        )
        rank = len(lhs_shape)
        if (
            rank not in (2, 3, 4)
            or len(rhs_shape) != 2
            or rhs_symbol is not None
            or any(symbol is not None and symbol[1] is None for _, symbol in contracts)
        ):
            raise NotImplementedError(
                "SOURCE dot_general: tensor-matrix dimensions are outside the audited private subset"
            )
        dimensions = eqn.params["dimension_numbers"]
        if not (
            isinstance(dimensions, (tuple, list))
            and len(dimensions) == 2
            and all(
                isinstance(pair, (tuple, list))
                and len(pair) == 2
                and all(isinstance(axes, (tuple, list)) for axes in pair)
                for pair in dimensions
            )
        ):
            raise ValueError("SOURCE dot_general: malformed tensor-matrix axes")
        (lc, rc), (lb, rb) = (
            tuple(tuple(axes) for axes in pair) for pair in dimensions
        )
        if not (
            all(
                all(
                    isinstance(axis, (int, np.integer))
                    and not isinstance(axis, (bool, np.bool_))
                    and 0 <= axis < operand_rank
                    for axis in contracting + batch
                )
                and len(set(contracting + batch)) == len(contracting + batch)
                for contracting, batch, operand_rank in ((lc, lb, rank), (rc, rb, 2))
            )
            and len(lc) == len(rc)
            and len(lb) == len(rb)
        ):
            raise ValueError(
                "SOURCE dot_general: invalid or overlapping tensor-matrix axes"
            )
        if ((lc, rc), (lb, rb)) != (((rank - 1,), (0,)), ((), ())):
            raise NotImplementedError(
                "SOURCE dot_general: tensor-matrix axes are outside the audited private subset"
            )
        if any(
            isinstance(dim, (bool, np.bool_))
            or not isinstance(dim, (int, np.integer))
            or dim <= 0
            for dim in (
                *(lhs_shape if lhs_symbol is None else lhs_shape[1:]),
                *rhs_shape,
            )
        ):
            raise NotImplementedError(
                "SOURCE dot_general: tensor-matrix static extents must be positive"
            )
        validate_source_shapes(native_shapes)
        if (
            lhs_shape[-1] != rhs_shape[0]
            or out_shape != lhs_shape[:-1] + (rhs_shape[1],)
            or out_symbol != lhs_symbol
        ):
            raise ValueError(
                "SOURCE dot_general: tensor-matrix result/contracting dimensions or scope differ"
            )
        if any(
            type(var.aval.weak_type) is not bool for var in (*eqn.invars, *eqn.outvars)
        ) or (
            eqn.outvars[0].aval.weak_type
            != all(var.aval.weak_type for var in eqn.invars)
        ):
            raise ValueError(
                "SOURCE dot_general: tensor-matrix result weakness differs"
            )
        lhs, rhs = (ctx.require_value_for_var(var) for var in eqn.invars)
        output = ctx.allocate_value_for_var(
            eqn.outvars[0], name_hint=ctx.fresh_name("dot_tensor_matrix")
        )
        for value, expected in (
            (lhs, lhs_shape),
            (rhs, rhs_shape),
            (output, out_shape),
        ):
            if (
                _value_ir_dtype(value) != dtype
                or value.shape is None
                or normalized_shape_dimensions(tuple(value.shape)) != expected
            ):
                raise ValueError(
                    "SOURCE dot_general: physical matrix dtype/shape differs from aval"
                )
        result = ctx.builder.op("MatMul", [lhs, rhs], output=output)
        ctx.bind_value_for_var(eqn.outvars[0], result)
        return result

    def _lower_source_leading_batch(
        self, ctx: NativeLoweringContext, eqn: Any, dtype: ir.DataType
    ) -> ir.Value:
        """Emit one canonical MatMul after proving actual native batch scopes."""
        native_shapes = [tuple(var.aval.shape) for var in (*eqn.invars, *eqn.outvars)]
        validate_source_shapes(native_shapes)
        contracts = [source_shape_contract(shape) for shape in native_shapes]
        (lhs_shape, lhs_symbol), (rhs_shape, rhs_symbol), (out_shape, out_symbol) = (
            contracts
        )
        if any(symbol is not None and symbol[1] is None for _, symbol in contracts):
            raise NotImplementedError(
                "SOURCE dot_general: unverified symbolic declarations are outside the audited private subset"
            )
        if lhs_symbol is None or out_symbol != lhs_symbol:
            raise ValueError("SOURCE dot_general: native result batch scope differs")
        free_batch = len(lhs_shape) == len(rhs_shape) == 2 and rhs_symbol is None
        paired_batch = (
            len(lhs_shape) == len(rhs_shape) == 3 and rhs_symbol == lhs_symbol
        )
        dimensions = (
            (((1,), (0,)), ((), ())) if free_batch else (((2,), (1,)), ((0,), (0,)))
        )
        actual_dimensions = eqn.params["dimension_numbers"]
        if not (
            isinstance(actual_dimensions, (tuple, list))
            and len(actual_dimensions) == 2
            and all(
                isinstance(pair, (tuple, list))
                and len(pair) == 2
                and all(
                    isinstance(axes, (tuple, list))
                    and all(
                        isinstance(axis, (int, np.integer))
                        and not isinstance(axis, (bool, np.bool_))
                        for axis in axes
                    )
                    for axes in pair
                )
                for pair in actual_dimensions
            )
        ):
            raise ValueError("SOURCE dot_general: malformed symbolic matrix axes")
        if (
            not (free_batch or paired_batch)
            or actual_dimensions != dimensions
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                for dim in (
                    *lhs_shape[1:],
                    *(rhs_shape if free_batch else rhs_shape[1:]),
                )
            )
        ):
            raise NotImplementedError(
                "SOURCE dot_general: symbolic matrix form is outside the audited private subset"
            )
        if lhs_shape[-1] != rhs_shape[-2] or out_shape != lhs_shape[:-1] + (
            rhs_shape[-1],
        ):
            raise ValueError(
                "SOURCE dot_general: symbolic result/contracting dimensions differ"
            )
        lhs, rhs = (ctx.require_value_for_var(var) for var in eqn.invars)
        output = ctx.allocate_value_for_var(
            eqn.outvars[0], name_hint=ctx.fresh_name("dot_batch_matrix")
        )
        for value, expected in (
            (lhs, lhs_shape),
            (rhs, rhs_shape),
            (output, out_shape),
        ):
            if (
                _value_ir_dtype(value) != dtype
                or value.shape is None
                or normalized_shape_dimensions(tuple(value.shape)) != expected
            ):
                raise ValueError(
                    "SOURCE dot_general: physical symbolic dtype/shape differs from aval"
                )
        result = ctx.builder.op("MatMul", [lhs, rhs], output=output)
        ctx.bind_value_for_var(eqn.outvars[0], result)
        return result

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
