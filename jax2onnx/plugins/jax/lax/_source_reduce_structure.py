# jax2onnx/plugins/jax/lax/_source_reduce_structure.py

"""Finite SOURCE reduction operation and geometry contracts.

These contracts check final wiring and representation. Accumulation order, NaN
propagation and numerical tolerances retain their independent obligations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import jax
import numpy as np
from numpy.typing import NDArray
import onnx_ir as ir

from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.converter.source_profile import _PLACEMENT_PARAMS
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
from jax2onnx.plugins.jax.lax._opset_utils import _REDUCTION_AXES_INPUT_SINCE

_OPERATIONS: Final = {
    jax.lax.reduce_sum_p: "ReduceSum",
    jax.lax.reduce_max_p: "ReduceMax",
}
_FLOATS: Final = frozenset({ir.DataType.FLOAT, ir.DataType.DOUBLE})
_KNOWN_CONTRADICTIONS: Final = frozenset(
    {
        "ReduceSum",
        "ReduceMax",
        "ReduceMin",
        "ReduceMean",
        "ReduceProd",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Neg",
    }
)


@dataclass(frozen=True)
class _ReductionRule:
    source: TensorContract
    output: TensorContract
    operation: str
    axes: tuple[int, ...]
    axes_input_since: int
    axes_literal: LiteralOperand

    def check(
        self,
        graph: ir.Graph,
        inputs: tuple[RuleOperand, ...],
        outputs: tuple[ir.Value, ...],
        match_literal: LiteralMatcher,
    ) -> tuple[ir.Node, ...]:
        if len(inputs) != 1 or len(outputs) != 1:
            raise StructureMismatch("reduce-boundary-arity-differs")
        result = outputs[0]
        if result.dtype is None or result.shape is None:
            raise StructureUnavailable("reduce-output-type-unavailable")
        if not self.output.matches(result):
            raise StructureMismatch("reduce-output-type-or-shape-differs")

        def source(actual: ir.Value | None) -> None:
            expected = inputs[0]
            if isinstance(expected, LiteralOperand):
                match_literal(actual, expected)
                return
            if actual is None:
                raise StructureMismatch("reduce-input-edge-missing")
            if actual.dtype is None or actual.shape is None:
                raise StructureUnavailable("reduce-input-type-unavailable")
            if not self.source.matches(actual):
                raise StructureMismatch("reduce-input-type-or-shape-differs")
            if actual is not expected:
                producer = actual.producer()
                if producer is not None and (
                    producer.domain
                    or producer.overload
                    or producer.op_type not in _KNOWN_CONTRADICTIONS
                ):
                    raise StructureUnavailable("reduce-input-form-unavailable")
                raise StructureMismatch("reduce-source-input-edge-differs")

        producer = result.producer()
        if not self.axes:
            operand = inputs[0]
            if isinstance(operand, ir.Value) and result is operand:
                return ()
            if isinstance(operand, LiteralOperand) and (
                producer is None or producer.op_type == "Constant"
            ):
                match_literal(result, operand)
                return ()

        if producer is None:
            raise StructureUnavailable("reduce-producer-unavailable")
        if (
            producer.graph is not graph
            or result.graph is not graph
            or not any(node is producer for node in graph)
            or result.index() != 0
            or len(producer.outputs) != 1
            or producer.outputs[0] is not result
        ):
            raise StructureUnavailable("reduce-producer-port-unavailable")
        if producer.domain or producer.overload:
            raise StructureUnavailable("reduce-operator-contract-unavailable")
        operation = self.operation if self.axes else "Identity"
        if producer.op_type != operation:
            if producer.op_type in _KNOWN_CONTRADICTIONS:
                raise StructureMismatch("reduce-operator-differs")
            raise StructureUnavailable("reduce-lowering-form-unavailable")
        if not self.axes:
            if producer.attributes or len(producer.inputs) != 1:
                raise StructureMismatch("reduce-identity-contract-differs")
            source(producer.inputs[0])
            return (producer,)

        opset = graph.opset_imports.get("")
        if not isinstance(opset, int) or isinstance(opset, bool) or opset < 1:
            raise StructureUnavailable("reduce-opset-unavailable")
        input_axes = opset >= self.axes_input_since
        expected_attributes = {"keepdims"} if input_axes else {"keepdims", "axes"}
        if len(producer.inputs) != (2 if input_axes else 1):
            raise StructureMismatch("reduce-input-arity-differs")
        if set(producer.attributes) != expected_attributes:
            raise StructureMismatch("reduce-attribute-set-differs")
        keepdims = producer.attributes["keepdims"]
        if keepdims.type is not ir.AttributeType.INT or keepdims.as_int() != 0:
            raise StructureMismatch("reduce-keepdims-differs")
        if input_axes:
            match_literal(producer.inputs[1], self.axes_literal)
        else:
            axes = producer.attributes["axes"]
            if (
                axes.type is not ir.AttributeType.INTS
                or tuple(axes.as_ints()) != self.axes
            ):
                raise StructureMismatch("reduce-axes-differ")
        source(producer.inputs[0])
        return (producer,)


def freeze_reduction_structure_rule(
    eqn: JaxprEqn,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> EquationRule | None:
    operation = next(
        (name for primitive, name in _OPERATIONS.items() if eqn.primitive is primitive),
        None,
    )
    if (
        operation is None
        or eqn.effects
        or len(eqn.invars) != 1
        or len(inputs) != 1
        or len(eqn.outvars) != 1
        or len(outputs) != 1
        or type(eqn.params) is not dict
        or set(eqn.params) - {"axes"} - _PLACEMENT_PARAMS
        or any(eqn.params.get(name) is not None for name in _PLACEMENT_PARAMS)
    ):
        return None
    left, right = inputs[0], outputs[0]
    if (
        any(
            value.encoding != "identity" or value.dtype not in _FLOATS
            for value in (left, right)
        )
        or left.dtype != right.dtype
        or left.logical_dtype != right.logical_dtype
        or left.weak_type != right.weak_type
        or 0 in left.shape
        or 0 in right.shape
    ):
        return None
    axes = eqn.params.get("axes")
    if (
        not isinstance(axes, tuple)
        or any(
            not isinstance(axis, (int, np.integer))
            or isinstance(axis, (bool, np.bool_))
            or not 0 <= axis < len(left.shape)
            for axis in axes
        )
        or len(set(axes)) != len(axes)
    ):
        return None
    axes = tuple(int(axis) for axis in axes)
    if right.shape != tuple(
        dim for axis, dim in enumerate(left.shape) if axis not in axes
    ) or right.logical_shape != tuple(
        dim for axis, dim in enumerate(left.logical_shape) if axis not in axes
    ):
        return None
    for source_axis, output_axis in (
        (left.symbol_axis, right.symbol_axis),
        (left.logical_symbol_axis, right.logical_symbol_axis),
    ):
        retained = (
            None
            if source_axis is None or source_axis in axes
            else source_axis - sum(axis < source_axis for axis in axes)
        )
        if output_axis != retained or right.symbol_token is not (
            None if retained is None else left.symbol_token
        ):
            return None
    vector: NDArray[np.int64] = np.asarray(axes, dtype=np.int64)
    literal_type = TensorContract(
        "int64",
        (len(axes),),
        False,
        ir.DataType.INT64,
        (len(axes),),
        "identity",
        None,
        None,
        None,
    )
    return _ReductionRule(
        left,
        right,
        operation,
        axes,
        _REDUCTION_AXES_INPUT_SINCE[operation],
        LiteralOperand(literal_type, ConstantContract.from_array(vector)),
    )
