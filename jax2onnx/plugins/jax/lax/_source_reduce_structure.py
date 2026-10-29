# jax2onnx/plugins/jax/lax/_source_reduce_structure.py

"""Finite SOURCE reduction operation and geometry contracts.

These contracts check final wiring and representation, including the floating
extrema correction. Actual NaN/zero-sign execution and accumulation accuracy
retain their independent numerical obligations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Final

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
    jax.lax.reduce_min_p: "ReduceMin",
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
        "Squeeze",
        "Unsqueeze",
        "Concat",
        "Gather",
        "GatherElements",
        "Cast",
        "Equal",
        "Less",
        "And",
        "Or",
        "Not",
        "Where",
        "Reciprocal",
        "IsNaN",
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
    float_correction: bool = False

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

        if self.float_correction and self.axes:
            return _check_float_extrema(
                self, graph, inputs, result, match_literal, source
            )

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


def _check_float_extrema(
    rule: _ReductionRule,
    graph: ir.Graph,
    inputs: tuple[RuleOperand, ...],
    result: ir.Value,
    match_literal: LiteralMatcher,
    source: Callable[[ir.Value | None], None],
) -> tuple[ir.Node, ...]:
    """Inspect the exact typed correction from independently frozen semantics.

    No emitted receipt or cached constant annotation supplies this contract.
    Both predicate reductions must use the original input and original axes;
    the final table selection must preserve the candidate or exact zero/NaN.
    """
    available, checked = set(graph), set[ir.Node]()
    shape, source_shape, dtype = rule.output.shape, rule.source.shape, rule.output.dtype
    selected_shape, choices_shape = (1, *shape), (2, *shape)

    def node(
        value: ir.Value | None,
        operation: str,
        arity: int,
        expected_shape: tuple[int | str, ...],
        expected_dtype: ir.DataType = dtype,
        attributes: tuple[tuple[str, int | tuple[int, ...]], ...] = (),
    ) -> ir.Node:
        if value is None:
            raise StructureMismatch("reduce-correction-input-missing")
        if value.shape is None or value.dtype is None:
            raise StructureUnavailable("reduce-correction-type-unavailable")
        if not replace(rule.output, shape=expected_shape, dtype=expected_dtype).matches(
            value
        ):
            raise StructureMismatch("reduce-correction-type-or-shape-differs")
        producer = value.producer()
        if (
            producer is None
            or producer.graph is not graph
            or value.graph is not graph
            or producer not in available
            or value.index() != 0
            or len(producer.outputs) != 1
            or producer.outputs[0] is not value
        ):
            raise StructureUnavailable("reduce-correction-producer-unavailable")
        if producer.domain or producer.overload:
            raise StructureUnavailable("reduce-correction-domain-unavailable")
        if producer.op_type != operation:
            if producer.op_type in _KNOWN_CONTRADICTIONS:
                raise StructureMismatch("reduce-correction-operator-differs")
            raise StructureUnavailable("reduce-correction-form-unavailable")
        if len(producer.inputs) != arity:
            raise StructureMismatch("reduce-correction-arity-differs")
        if set(producer.attributes) != {name for name, _ in attributes}:
            raise StructureMismatch("reduce-correction-attributes-differ")
        for name, expected in attributes:
            attribute = producer.attributes[name]
            if isinstance(expected, tuple):
                matches = (
                    attribute.type is ir.AttributeType.INTS
                    and tuple(attribute.as_ints()) == expected
                )
            else:
                matches = (
                    attribute.type is ir.AttributeType.INT
                    and attribute.as_int() == expected
                )
            if not matches:
                raise StructureMismatch("reduce-correction-attribute-differs")
        checked.add(producer)
        return producer

    def literal(value: ir.Value | None, array: np.ndarray[Any, Any]) -> None:
        dimensions = tuple(array.shape)
        tensor = TensorContract(
            array.dtype.name,
            dimensions,
            False,
            ir.DataType.from_numpy(array.dtype),
            dimensions,
            "identity",
            None,
            None,
            None,
        )
        match_literal(value, LiteralOperand(tensor, ConstantContract.from_array(array)))

    def axis(value: ir.Value | None) -> None:
        literal(value, np.asarray([0], dtype=np.int64))

    def zero(value: ir.Value | None) -> None:
        literal(value, np.asarray(0, dtype=rule.source.logical_dtype))

    def cast_input(
        value: ir.Value | None,
        expected_dtype: ir.DataType,
        expected_shape: tuple[int | str, ...],
    ) -> ir.Value | None:
        return node(
            value,
            "Cast",
            1,
            expected_shape,
            expected_dtype,
            (("to", int(expected_dtype)),),
        ).inputs[0]

    def predicate_input(value: ir.Value | None) -> ir.Value | None:
        reduced_value = cast_input(value, ir.DataType.BOOL, shape)
        opset = graph.opset_imports.get("")
        if not isinstance(opset, int) or isinstance(opset, bool) or opset < 1:
            raise StructureUnavailable("reduce-opset-unavailable")
        input_axes = opset >= _REDUCTION_AXES_INPUT_SINCE["ReduceMax"]
        attributes: tuple[tuple[str, int | tuple[int, ...]], ...] = (("keepdims", 0),)
        if not input_axes:
            attributes += (("axes", rule.axes),)
        reduced = node(
            reduced_value,
            "ReduceMax",
            2 if input_axes else 1,
            shape,
            ir.DataType.INT32,
            attributes,
        )
        if input_axes:
            match_literal(reduced.inputs[1], rule.axes_literal)
        return cast_input(reduced.inputs[0], ir.DataType.INT32, source_shape)

    squeezed = node(result, "Squeeze", 2, shape)
    axis(squeezed.inputs[1])
    gathered = node(
        squeezed.inputs[0],
        "GatherElements",
        2,
        selected_shape,
        attributes=(("axis", 0),),
    )
    choices = node(
        gathered.inputs[0],
        "Concat",
        2,
        choices_shape,
        attributes=(("axis", 0),),
    )
    base_choice = node(choices.inputs[0], "Unsqueeze", 2, selected_shape)
    axis(base_choice.inputs[1])
    candidate = base_choice.inputs[0]
    if candidate is None:
        raise StructureMismatch("reduce-correction-candidate-missing")
    checked.update(
        replace(rule, float_correction=False).check(
            graph, inputs, (candidate,), match_literal
        )
    )
    correction_choice = node(choices.inputs[1], "Unsqueeze", 2, selected_shape)
    axis(correction_choice.inputs[1])
    correction = node(
        correction_choice.inputs[0],
        "Gather",
        2,
        shape,
        attributes=(("axis", 0),),
    )
    literal(
        correction.inputs[0],
        np.asarray([0.0, -0.0, np.nan], dtype=rule.source.logical_dtype),
    )
    correction_index = node(correction.inputs[1], "Where", 3, shape, ir.DataType.INT64)
    any_nan = correction_index.inputs[0]
    isnan = node(predicate_input(any_nan), "IsNaN", 1, source_shape, ir.DataType.BOOL)
    source(isnan.inputs[0])
    literal(correction_index.inputs[1], np.asarray(2, dtype=np.int64))
    negative = cast_input(correction_index.inputs[2], ir.DataType.INT64, shape)
    if rule.operation == "ReduceMax":
        negative = node(negative, "Not", 1, shape, ir.DataType.BOOL).inputs[0]
    zero_tie = node(predicate_input(negative), "And", 2, source_shape, ir.DataType.BOOL)
    input_zero = node(zero_tie.inputs[0], "Equal", 2, source_shape, ir.DataType.BOOL)
    source(input_zero.inputs[0])
    zero(input_zero.inputs[1])
    sign = zero_tie.inputs[1]
    if rule.operation == "ReduceMax":
        sign = node(sign, "Not", 1, source_shape, ir.DataType.BOOL).inputs[0]
    less = node(sign, "Less", 2, source_shape, ir.DataType.BOOL)
    reciprocal = node(less.inputs[0], "Reciprocal", 1, source_shape)
    source(reciprocal.inputs[0])
    zero(less.inputs[1])

    indices = node(
        gathered.inputs[1], "Unsqueeze", 2, selected_shape, ir.DataType.INT64
    )
    axis(indices.inputs[1])
    replace_zero_nan = node(
        cast_input(indices.inputs[0], ir.DataType.INT64, shape),
        "Or",
        2,
        shape,
        ir.DataType.BOOL,
    )
    if replace_zero_nan.inputs[1] is not any_nan:
        raise StructureMismatch("reduce-correction-nan-predicate-edge-differs")
    base_zero = node(replace_zero_nan.inputs[0], "Equal", 2, shape, ir.DataType.BOOL)
    if base_zero.inputs[0] is not candidate:
        raise StructureMismatch("reduce-correction-candidate-edge-differs")
    zero(base_zero.inputs[1])
    return tuple(item for item in graph if item in checked)


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
        float_correction=operation in {"ReduceMin", "ReduceMax"} and bool(axes),
    )
