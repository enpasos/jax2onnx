# jax2onnx/plugins/jax/lax/_source_dot_structure.py

"""Frozen SOURCE contraction plans checked against the actual final IR.

This is structural evidence only. Source admission, precision policy, lowering,
optimizer behavior and independent runtime/numerical obligations are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

import jax
import numpy as np
from numpy.typing import NDArray
import onnx_ir as ir

from jax2onnx._compat.jax import JaxprEqn, is_native_precision_pair
from jax2onnx.converter.source_profile import (
    _DotContractionPlan,
    _dot_contraction_plan,
    _is_direct_tensor_matrix_dot,
    _is_mixed_fp32_dot_route,
    _is_mixed_fp64_dot_route,
)
from jax2onnx.converter.source_representation import normalized_shape_dimensions
from jax2onnx.converter.source_structure import (
    ConstantContract,
    EquationRule,
    LiteralMatcher,
    LiteralOperand,
    RuleOperand,
    StructureMismatch,
    StructureUnavailable,
    TensorContract,
)

Shape: TypeAlias = tuple[int | str, ...]
Axes: TypeAlias = tuple[
    tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]
]
_FLOATS: Final = frozenset({ir.DataType.FLOAT, ir.DataType.DOUBLE})
_KNOWN_OPERATORS: Final = frozenset(
    {
        "MatMul",
        "Transpose",
        "Reshape",
        "Cast",
        "Shape",
        "Gather",
        "Concat",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Neg",
    }
)


def _integer_literal(values: tuple[int, ...]) -> LiteralOperand:
    array: NDArray[np.int64] = np.asarray(values, dtype=np.int64)
    tensor = TensorContract(
        "int64",
        (len(values),),
        False,
        ir.DataType.INT64,
        (len(values),),
        "identity",
        None,
        None,
        None,
    )
    return LiteralOperand(tensor, ConstantContract.from_array(array))


def _direct_route(
    inputs: tuple[TensorContract, ...],
    output: TensorContract,
    native_shapes: tuple[tuple[object, ...], ...],
    axes: Axes,
    same_width: bool,
) -> bool:
    if not same_width:
        return False
    contracts = (*inputs, output)
    symbolic = any(value.symbol_token is not None for value in contracts)
    if not symbolic:
        return _is_direct_tensor_matrix_dot(native_shapes[0], native_shapes[1], axes)
    leading = all(
        value.symbol_token is None
        or value.symbol_axis == 0
        and all(type(dim) is int and dim > 0 for dim in value.shape[1:])
        for value in contracts
    )
    if not leading:
        return False
    if (
        inputs[0].symbol_token is not None
        and inputs[1].symbol_token is None
        and len(inputs[0].shape) in (2, 3, 4)
        and len(inputs[1].shape) == 2
        and axes == (((len(inputs[0].shape) - 1,), (0,)), ((), ()))
    ):
        return True
    return (
        len(inputs[0].shape) == len(inputs[1].shape) == 3
        and inputs[0].symbol_token is not None
        and inputs[0].symbol_token is inputs[1].symbol_token
        and axes == (((2,), (1,)), ((0,), (0,)))
    )


@dataclass(frozen=True)
class _DotRule:
    inputs: tuple[TensorContract, ...]
    output: TensorContract
    axes: Axes
    precision: tuple[str, str] | None
    preferred: str | None
    plan: _DotContractionPlan
    direct: bool
    shape_literals: tuple[tuple[tuple[int, ...], LiteralOperand], ...]

    def check(
        self,
        graph: ir.Graph,
        inputs: tuple[RuleOperand, ...],
        outputs: tuple[ir.Value, ...],
        match_literal: LiteralMatcher,
    ) -> tuple[ir.Node, ...]:
        if len(inputs) != 2 or len(outputs) != 1:
            raise StructureMismatch("dot-boundary-arity-differs")
        checked: set[ir.Node] = set()
        dtype = self.output.dtype

        def value(actual: ir.Value | None, shape: Shape, kind: ir.DataType) -> ir.Value:
            if actual is None:
                raise StructureMismatch("dot-input-edge-missing")
            if actual.dtype is None or actual.shape is None:
                raise StructureUnavailable("dot-value-type-unavailable")
            if any(
                not isinstance(dim, (int, np.integer, str, ir.SymbolicDim))
                or isinstance(dim, (bool, np.bool_))
                for dim in actual.shape
            ):
                raise StructureUnavailable("dot-ir-shape-unavailable")
            if (
                actual.dtype != kind
                or normalized_shape_dimensions(tuple(actual.shape)) != shape
            ):
                raise StructureMismatch("dot-value-type-or-shape-differs")
            return actual

        def node(
            actual: ir.Value | None,
            operation: str,
            arity: int,
            attributes: tuple[tuple[str, int | tuple[int, ...]], ...],
            shape: Shape,
            kind: ir.DataType = dtype,
        ) -> ir.Node:
            actual = value(actual, shape, kind)
            producer = actual.producer()
            if producer is None:
                raise StructureUnavailable("dot-producer-unavailable")
            if (
                producer.graph is not graph
                or actual.graph is not graph
                or not any(candidate is producer for candidate in graph)
                or actual.index() != 0
                or len(producer.outputs) != 1
                or producer.outputs[0] is not actual
            ):
                raise StructureUnavailable("dot-producer-port-unavailable")
            if producer.domain or producer.overload:
                raise StructureUnavailable("dot-operator-contract-unavailable")
            if producer.op_type != operation:
                if producer.op_type in _KNOWN_OPERATORS:
                    raise StructureMismatch("dot-operator-differs")
                raise StructureUnavailable("dot-lowering-form-unavailable")
            if len(producer.inputs) != arity:
                raise StructureMismatch("dot-input-arity-differs")
            if set(producer.attributes) != {name for name, _ in attributes}:
                raise StructureMismatch("dot-attribute-set-differs")
            for name, expected_attribute in attributes:
                attribute = producer.attributes[name]
                if isinstance(expected_attribute, tuple):
                    matches = (
                        attribute.type is ir.AttributeType.INTS
                        and tuple(attribute.as_ints()) == expected_attribute
                    )
                else:
                    matches = (
                        attribute.type is ir.AttributeType.INT
                        and attribute.as_int() == expected_attribute
                    )
                if not matches:
                    raise StructureMismatch("dot-attribute-differs")
            checked.add(producer)
            return producer

        def source(actual: ir.Value | None, side: int) -> None:
            expected_operand = inputs[side]
            if isinstance(expected_operand, LiteralOperand):
                match_literal(actual, expected_operand)
            elif actual is not expected_operand:
                raise StructureMismatch("dot-source-input-edge-differs")

        def integers(actual: ir.Value | None, expected_values: tuple[int, ...]) -> None:
            for key, frozen in self.shape_literals:
                if key == expected_values:
                    match_literal(actual, frozen)
                    return
            raise StructureUnavailable("dot-shape-literal-contract-unavailable")

        def batch_extent(actual: ir.Value | None) -> None:
            if self.plan.symbolic_operand is None:
                raise StructureUnavailable("dot-symbol-origin-unavailable")
            side, axis = self.plan.symbolic_operand
            gather = node(actual, "Gather", 2, (("axis", 0),), (1,), ir.DataType.INT64)
            integers(gather.inputs[1], (axis,))
            original = inputs[side]
            if not isinstance(original, ir.Value):
                raise StructureUnavailable("dot-symbolic-literal-unavailable")
            shape = node(
                gather.inputs[0],
                "Shape",
                1,
                (),
                (len(self.inputs[side].shape),),
                ir.DataType.INT64,
            )
            if shape.inputs[0] is not original:
                raise StructureMismatch("dot-symbol-source-edge-differs")

        def shape_vector(actual: ir.Value | None, expected_shape: Shape) -> None:
            if all(type(dim) is int for dim in expected_shape):
                integers(actual, tuple(int(dim) for dim in expected_shape))
                return
            if len(expected_shape) == 1:
                batch_extent(actual)
                return
            concat = node(
                actual,
                "Concat",
                len(expected_shape),
                (("axis", 0),),
                (len(expected_shape),),
                ir.DataType.INT64,
            )
            for actual_piece, expected_dimension in zip(
                concat.inputs, expected_shape, strict=True
            ):
                if isinstance(expected_dimension, str):
                    batch_extent(actual_piece)
                else:
                    integers(actual_piece, (expected_dimension,))

        if self.direct:
            matrix = node(outputs[0], "MatMul", 2, (), self.output.shape)
            for side, actual_input in enumerate(matrix.inputs):
                source(actual_input, side)
            return tuple(checked)

        current: ir.Value | None = outputs[0]
        identity = self.plan.output_permutation == tuple(
            range(len(self.plan.output_shape))
        )
        if not identity:
            transpose = node(
                current,
                "Transpose",
                1,
                (("perm", self.plan.output_permutation),),
                self.plan.output_shape,
            )
            current = transpose.inputs[0]
        if self.plan.matrix_output_shape != self.plan.ungrouped_output_shape:
            reshape = node(
                current,
                "Reshape",
                2,
                (("allowzero", 1),),
                self.plan.ungrouped_output_shape,
            )
            shape_vector(reshape.inputs[1], self.plan.ungrouped_output_shape)
            current = reshape.inputs[0]
        matrix = node(current, "MatMul", 2, (), self.plan.matrix_output_shape)
        for side, actual_input in enumerate(matrix.inputs):
            permutation = (self.plan.lhs_permutation, self.plan.rhs_permutation)[side]
            matrix_shape = (self.plan.lhs_matrix_shape, self.plan.rhs_matrix_shape)[
                side
            ]
            original_shape = self.inputs[side].shape
            permuted_shape = tuple(original_shape[axis] for axis in permutation)
            current = actual_input
            if permuted_shape != matrix_shape:
                reshape = node(current, "Reshape", 2, (("allowzero", 1),), matrix_shape)
                shape_vector(reshape.inputs[1], matrix_shape)
                current = reshape.inputs[0]
            if permutation != tuple(range(len(original_shape))):
                transpose = node(
                    current, "Transpose", 1, (("perm", permutation),), permuted_shape
                )
                current = transpose.inputs[0]
            if self.inputs[side].dtype != dtype:
                cast = node(
                    current, "Cast", 1, (("to", int(dtype.value)),), original_shape
                )
                current = cast.inputs[0]
            source(current, side)
        return tuple(checked)


def freeze_dot_structure_rule(
    eqn: JaxprEqn,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> EquationRule | None:
    if (
        eqn.primitive is not jax.lax.dot_general_p
        or eqn.effects
        or len(eqn.invars) != 2
        or len(inputs) != 2
        or len(eqn.outvars) != 1
        or len(outputs) != 1
        or type(eqn.params) is not dict
        or set(eqn.params)
        - {"dimension_numbers", "precision", "preferred_element_type", "out_sharding"}
        or "dimension_numbers" not in eqn.params
        or eqn.params.get("out_sharding") is not None
        or any(
            value.encoding != "identity" or value.dtype not in _FLOATS
            for value in (*inputs, *outputs)
        )
    ):
        return None
    tokens = [
        value.symbol_token
        for value in (*inputs, *outputs)
        if value.symbol_token is not None
    ]
    if tokens and any(token is not tokens[0] for token in tokens):
        return None
    preferred = eqn.params.get("preferred_element_type")
    dtypes = tuple(value.logical_dtype for value in (*inputs, *outputs))
    same_width = dtypes[0] == dtypes[1] == dtypes[2] and (
        preferred is None or np.dtype(preferred).name == dtypes[2]
    )
    if not (
        same_width
        or _is_mixed_fp64_dot_route(dtypes, preferred)
        or _is_mixed_fp32_dot_route(dtypes, preferred)
    ):
        return None
    precision = eqn.params.get("precision")
    if precision is not None and (
        not is_native_precision_pair(precision)
        or outputs[0].dtype != ir.DataType.DOUBLE
        and precision != (jax.lax.Precision.DEFAULT,) * 2
    ):
        return None
    native_shapes = tuple(tuple(var.aval.shape) for var in (*eqn.invars, *eqn.outvars))
    try:
        plan = _dot_contraction_plan(
            native_shapes[0], native_shapes[1], eqn.params["dimension_numbers"]
        )
    except ValueError:
        return None
    if (
        plan.outer_product
        or plan.contracting_size == 0
        or 0 in plan.output_shape
        or plan.output_shape != outputs[0].shape
    ):
        return None
    raw_axes = eqn.params["dimension_numbers"]
    axes: Axes = (
        (
            tuple(int(axis) for axis in raw_axes[0][0]),
            tuple(int(axis) for axis in raw_axes[0][1]),
        ),
        (
            tuple(int(axis) for axis in raw_axes[1][0]),
            tuple(int(axis) for axis in raw_axes[1][1]),
        ),
    )
    static_vectors: set[tuple[int, ...]] = set()
    for shape in (
        plan.lhs_matrix_shape,
        plan.rhs_matrix_shape,
        plan.ungrouped_output_shape,
    ):
        if all(type(dim) is int for dim in shape):
            static_vectors.add(tuple(int(dim) for dim in shape))
        for dim in shape:
            if type(dim) is int:
                static_vectors.add((dim,))
    if plan.symbolic_operand is not None:
        static_vectors.add((plan.symbolic_operand[1],))
    return _DotRule(
        inputs,
        outputs[0],
        axes,
        None if precision is None else (precision[0].name, precision[1].name),
        None if preferred is None else np.dtype(preferred).name,
        plan,
        _direct_route(inputs, outputs[0], native_shapes, raw_axes, same_width),
        tuple((vector, _integer_literal(vector)) for vector in sorted(static_vectors)),
    )
