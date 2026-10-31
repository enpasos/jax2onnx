# jax2onnx/plugins/jax/lax/_source_shape_structure.py

"""Finite SOURCE reshape/broadcast contracts over native-frozen geometry.

Printed IR labels describe already-frozen native dimensions; they never grant
symbolic authority. A foreign runtime shape origin remains unverified until a
shared independent native-origin witness is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Final, TypeAlias

import jax
import numpy as np
from numpy.typing import NDArray
import onnx_ir as ir

from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.converter.source_profile import _PLACEMENT_PARAMS
from jax2onnx.converter.source_representation import (
    native_shape_contract,
    normalized_shape_dimensions,
    validate_native_shapes,
)
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
Attributes: TypeAlias = tuple[tuple[str, int | tuple[int, ...]], ...]
_FLOATS: Final = frozenset({ir.DataType.FLOAT, ir.DataType.DOUBLE})
_KNOWN_OPERATORS: Final = frozenset(
    {
        "Reshape",
        "Expand",
        "Transpose",
        "Shape",
        "Gather",
        "Concat",
        "Squeeze",
        "Unsqueeze",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Neg",
    }
)


def _literal(values: tuple[int, ...]) -> LiteralOperand:
    vector: NDArray[np.int64] = np.asarray(values, dtype=np.int64)
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
    return LiteralOperand(tensor, ConstantContract.from_array(vector))


class _Checker:
    """Transient final-graph traversal; no emitted IR or native objects retained."""

    def __init__(
        self,
        graph: ir.Graph,
        original: RuleOperand,
        tensor: TensorContract,
        literals: tuple[tuple[tuple[int, ...], LiteralOperand], ...],
        match_literal: LiteralMatcher,
    ) -> None:
        self.graph = graph
        self.original = original
        self.tensor = tensor
        self.literals = literals
        self.match_literal = match_literal
        self.nodes: set[ir.Node] = set()
        self.gaps: set[str] = set()

    def value(
        self, actual: ir.Value | None, shape: Shape, dtype: ir.DataType
    ) -> ir.Value:
        if actual is None:
            raise StructureMismatch("shape-input-edge-missing")
        if actual.dtype is None or actual.shape is None:
            raise StructureUnavailable("shape-value-type-unavailable")
        if any(
            not isinstance(dim, (int, np.integer, str, ir.SymbolicDim))
            or isinstance(dim, (bool, np.bool_))
            for dim in actual.shape
        ):
            raise StructureUnavailable("shape-ir-dimension-unavailable")
        if (
            actual.dtype != dtype
            or normalized_shape_dimensions(tuple(actual.shape)) != shape
        ):
            raise StructureMismatch("shape-value-type-or-shape-differs")
        return actual

    def node(
        self,
        actual: ir.Value | None,
        operation: str,
        arity: int,
        attributes: Attributes,
        shape: Shape,
        dtype: ir.DataType,
    ) -> ir.Node:
        value = self.value(actual, shape, dtype)
        node = value.producer()
        if node is None:
            raise StructureUnavailable("shape-producer-unavailable")
        if (
            node.graph is not self.graph
            or value.graph is not self.graph
            or not any(candidate is node for candidate in self.graph)
            or value.index() != 0
            or len(node.outputs) != 1
            or node.outputs[0] is not value
        ):
            raise StructureUnavailable("shape-producer-port-unavailable")
        if node.domain or node.overload:
            raise StructureUnavailable("shape-operator-contract-unavailable")
        if node.op_type != operation:
            if node.op_type in _KNOWN_OPERATORS:
                raise StructureMismatch("shape-operator-differs")
            raise StructureUnavailable("shape-lowering-form-unavailable")
        if len(node.inputs) != arity:
            raise StructureMismatch("shape-input-arity-differs")
        if set(node.attributes) != {name for name, _ in attributes}:
            raise StructureMismatch("shape-attribute-set-differs")
        for name, expected in attributes:
            attribute = node.attributes[name]
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
                raise StructureMismatch("shape-attribute-differs")
        self.nodes.add(node)
        return node

    def source(self, actual: ir.Value | None) -> None:
        if isinstance(self.original, LiteralOperand):
            self.match_literal(actual, self.original)
        elif actual is not self.original:
            producer = None if actual is None else actual.producer()
            if producer is not None and (
                producer.domain
                or producer.overload
                or producer.op_type not in _KNOWN_OPERATORS
            ):
                raise StructureUnavailable("shape-data-input-form-unavailable")
            raise StructureMismatch("shape-source-input-edge-differs")

    def integers(self, actual: ir.Value | None, expected: tuple[int, ...]) -> None:
        for key, literal in self.literals:
            if key == expected:
                self.match_literal(actual, literal)
                return
        raise StructureUnavailable("shape-integer-contract-unavailable")

    def extent(self, actual: ir.Value | None) -> None:
        # Reshape emits Gather->[Squeeze]->Unsqueeze. The exact inverse pair
        # may be removed by optimization; broadcast emits the direct Gather.
        producer = None if actual is None else actual.producer()
        if actual is None:
            raise StructureMismatch("shape-input-edge-missing")
        if producer is None or (not producer.domain and producer.op_type == "Constant"):
            raise StructureMismatch("shape-symbol-source-missing")
        if (
            producer is not None
            and not producer.domain
            and producer.op_type == "Unsqueeze"
        ):
            unsqueeze = self.node(actual, "Unsqueeze", 2, (), (1,), ir.DataType.INT64)
            self.integers(unsqueeze.inputs[1], (0,))
            squeeze = self.node(
                unsqueeze.inputs[0], "Squeeze", 2, (), (), ir.DataType.INT64
            )
            self.integers(squeeze.inputs[1], (0,))
            actual = squeeze.inputs[0]
        gather = self.node(actual, "Gather", 2, (("axis", 0),), (1,), ir.DataType.INT64)
        raw_shape = gather.inputs[0]
        shape_node = None if raw_shape is None else raw_shape.producer()
        if shape_node is None:
            raise StructureUnavailable("shape-symbol-origin-unavailable")
        if len(shape_node.inputs) != 1:
            if not shape_node.domain and shape_node.op_type in _KNOWN_OPERATORS:
                raise StructureMismatch("shape-input-arity-differs")
            raise StructureUnavailable("shape-symbol-origin-unavailable")
        origin = shape_node.inputs[0]
        own = isinstance(self.original, ir.Value) and origin is self.original
        if own:
            rank = len(self.tensor.shape)
        elif origin is not None and origin.shape is not None:
            rank = len(origin.shape)
        else:
            raise StructureUnavailable("shape-symbol-origin-unavailable")
        self.node(raw_shape, "Shape", 1, (), (rank,), ir.DataType.INT64)
        if own and self.tensor.symbol_axis is None:
            raise StructureMismatch("shape-static-symbol-origin-substitute")
        if own and self.tensor.symbol_axis is not None:
            self.integers(gather.inputs[1], (self.tensor.symbol_axis,))
        else:
            # The emitter's last-bound origin lookup is not native authority.
            # Continue other known checks so this gap cannot hide bad constants.
            self.gaps.add("shape-symbol-origin-unavailable")

    def vector(self, actual: ir.Value | None, expected: Shape) -> None:
        symbolic = any(isinstance(dim, str) for dim in expected)
        producer = None if actual is None else actual.producer()
        leaf = producer is None or (
            not producer.domain and producer.op_type == "Constant"
        )
        if leaf:
            if symbolic:
                if actual is not None and actual.is_graph_input():
                    raise StructureMismatch("shape-runtime-vector-substitute")
                raise StructureMismatch("shape-symbol-source-missing")
            self.integers(actual, tuple(int(dim) for dim in expected))
            return
        if (
            symbolic
            and len(expected) == 1
            and producer is not None
            and producer.op_type != "Concat"
        ):
            self.extent(actual)
            return
        concat = self.node(
            actual,
            "Concat",
            len(expected),
            (("axis", 0),),
            (len(expected),),
            ir.DataType.INT64,
        )
        for piece, dimension in zip(concat.inputs, expected, strict=True):
            if isinstance(dimension, str):
                self.extent(piece)
            else:
                self.integers(piece, (dimension,))

    def finish(self) -> tuple[ir.Node, ...]:
        if self.gaps:
            raise StructureUnavailable(sorted(self.gaps)[0])
        return tuple(self.nodes)


@dataclass(frozen=True)
class _ShapeRule:
    source: TensorContract
    output: TensorContract
    operation: str
    permutation: tuple[int, ...] | None
    broadcast_dimensions: tuple[int, ...] | None
    literals: tuple[tuple[tuple[int, ...], LiteralOperand], ...]

    def check(
        self,
        graph: ir.Graph,
        inputs: tuple[RuleOperand, ...],
        outputs: tuple[ir.Value, ...],
        match_literal: LiteralMatcher,
    ) -> tuple[ir.Node, ...]:
        if len(inputs) != 1 or len(outputs) != 1:
            raise StructureMismatch("shape-boundary-arity-differs")
        checker = _Checker(graph, inputs[0], self.source, self.literals, match_literal)
        checker.value(outputs[0], self.output.shape, self.output.dtype)
        opset = graph.opset_imports.get("")
        if not isinstance(opset, int) or isinstance(opset, bool) or opset < 13:
            raise StructureUnavailable("shape-opset-contract-unavailable")
        identity = self.source.same_shape(self.output) and (
            self.permutation is None
            or self.permutation == tuple(range(len(self.source.shape)))
        )
        result = outputs[0]
        producer = result.producer()
        if identity:
            if isinstance(inputs[0], ir.Value) and result is inputs[0]:
                return ()
            if isinstance(inputs[0], LiteralOperand) and (
                producer is None or producer.op_type == "Constant"
            ):
                match_literal(result, inputs[0])
                return ()
            if producer is not None and producer.op_type == "Identity":
                alias = checker.node(
                    result, "Identity", 1, (), self.output.shape, self.output.dtype
                )
                checker.source(alias.inputs[0])
                return checker.finish()
        if self.operation == "Reshape":
            if (
                self.permutation is not None
                and producer is not None
                and producer.op_type == "Transpose"
                and tuple(self.source.shape[axis] for axis in self.permutation)
                == self.output.shape
            ):
                transpose = checker.node(
                    result,
                    "Transpose",
                    1,
                    (("perm", self.permutation),),
                    self.output.shape,
                    self.output.dtype,
                )
                checker.source(transpose.inputs[0])
                return checker.finish()
            reshape = checker.node(
                result, "Reshape", 2, (), self.output.shape, self.output.dtype
            )
            checker.vector(reshape.inputs[1], self.output.shape)
            current = reshape.inputs[0]
            if self.permutation is not None:
                identity_permutation = self.permutation == tuple(
                    range(len(self.source.shape))
                )
                if not (
                    identity_permutation
                    and isinstance(inputs[0], ir.Value)
                    and current is inputs[0]
                ):
                    transposed_shape = tuple(
                        self.source.shape[axis] for axis in self.permutation
                    )
                    transpose = checker.node(
                        current,
                        "Transpose",
                        1,
                        (("perm", self.permutation),),
                        transposed_shape,
                        self.source.dtype,
                    )
                    current = transpose.inputs[0]
            checker.source(current)
        else:
            expand = checker.node(
                result, "Expand", 2, (), self.output.shape, self.output.dtype
            )
            checker.vector(expand.inputs[1], self.output.shape)
            current = expand.inputs[0]
            if self.source.shape and len(self.source.shape) != len(self.output.shape):
                if self.broadcast_dimensions is None:
                    raise StructureUnavailable("shape-broadcast-dimensions-unavailable")
                aligned: list[int | str] = [1] * len(self.output.shape)
                for axis, destination in enumerate(self.broadcast_dimensions):
                    aligned[destination] = self.source.shape[axis]
                if isinstance(inputs[0], ir.Value) and current is inputs[0]:
                    # Expand already right-aligns its operand. An optimizer may
                    # remove precisely that redundant leading-one insertion.
                    if self.broadcast_dimensions != tuple(
                        range(
                            len(self.output.shape) - len(self.source.shape),
                            len(self.output.shape),
                        )
                    ):
                        checker.gaps.add("shape-broadcast-normalization-unavailable")
                else:
                    reshape = checker.node(
                        current, "Reshape", 2, (), tuple(aligned), self.source.dtype
                    )
                    checker.vector(reshape.inputs[1], tuple(aligned))
                    current = reshape.inputs[0]
            checker.source(current)
        return checker.finish()


def freeze_shape_structure_rule(
    eqn: JaxprEqn,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> EquationRule | None:
    reshape = eqn.primitive is jax.lax.reshape_p
    if (
        not reshape
        and eqn.primitive is not jax.lax.broadcast_in_dim_p
        or eqn.effects
        or len(eqn.invars) != 1
        or len(inputs) != 1
        or len(eqn.outvars) != 1
        or len(outputs) != 1
        or type(eqn.params) is not dict
    ):
        return None
    names = (
        {"new_sizes", "dimensions"} if reshape else {"shape", "broadcast_dimensions"}
    )
    if (
        set(eqn.params) - names - _PLACEMENT_PARAMS
        or not names <= set(eqn.params)
        or any(eqn.params.get(name) is not None for name in _PLACEMENT_PARAMS)
    ):
        return None
    source, output = inputs[0], outputs[0]
    if (
        any(
            value.encoding != "identity" or value.dtype not in _FLOATS
            for value in (source, output)
        )
        or source.dtype != output.dtype
        or source.logical_dtype != output.logical_dtype
        or source.weak_type != output.weak_type
        or 0 in source.shape
        or 0 in output.shape
        or any(
            type(dim) is int and dim > np.iinfo(np.int64).max
            for value in (source, output)
            for dim in value.shape
        )
    ):
        return None
    declared = eqn.params["new_sizes" if reshape else "shape"]
    if not isinstance(declared, tuple):
        return None
    try:
        native = (
            tuple(eqn.invars[0].aval.shape),
            tuple(eqn.outvars[0].aval.shape),
            declared,
        )
        validate_native_shapes(native)
        contracts = tuple(native_shape_contract(shape) for shape in native)
    except ValueError:
        return None
    if (
        contracts[0].shape != source.shape
        or contracts[1].shape != output.shape
        or contracts[2].shape != output.shape
        or any(
            (contract.symbol is None) != (tensor.symbol_token is None)
            or contract.symbol_axis != tensor.symbol_axis
            for contract, tensor in zip(contracts[:2], (source, output), strict=True)
        )
        or source.symbol_token is not None
        and source.symbol_token is not output.symbol_token
    ):
        return None
    permutation = None
    dimensions = None
    shapes = [output.shape]
    if reshape:
        if source.symbol_token is not output.symbol_token:
            return None
        left = tuple(
            dim for axis, dim in enumerate(source.shape) if axis != source.symbol_axis
        )
        right = tuple(
            dim for axis, dim in enumerate(output.shape) if axis != output.symbol_axis
        )
        if prod(left) != prod(right):
            return None
        raw = eqn.params["dimensions"]
        if raw is not None:
            if (
                not isinstance(raw, tuple)
                or len(raw) != len(source.shape)
                or any(
                    not isinstance(axis, (int, np.integer))
                    or isinstance(axis, (bool, np.bool_))
                    for axis in raw
                )
                or set(raw) != set(range(len(source.shape)))
            ):
                return None
            permutation = tuple(int(axis) for axis in raw)
    else:
        raw = eqn.params["broadcast_dimensions"]
        if (
            not output.shape
            or not isinstance(raw, tuple)
            or len(raw) != len(source.shape)
            or any(
                not isinstance(axis, (int, np.integer))
                or isinstance(axis, (bool, np.bool_))
                or not 0 <= axis < len(output.shape)
                for axis in raw
            )
            or tuple(sorted(set(raw))) != raw
            or any(
                dim not in (1, output.shape[axis])
                for dim, axis in zip(source.shape, raw, strict=True)
            )
            or source.symbol_axis is not None
            and raw[source.symbol_axis] != output.symbol_axis
        ):
            return None
        dimensions = tuple(int(axis) for axis in raw)
        aligned: list[int | str] = [1] * len(output.shape)
        for axis, destination in enumerate(dimensions):
            aligned[destination] = source.shape[axis]
        shapes.append(tuple(aligned))
    vectors: set[tuple[int, ...]] = {(0,)}
    if source.symbol_axis is not None:
        vectors.add((source.symbol_axis,))
    for shape in shapes:
        if all(type(dim) is int for dim in shape):
            vectors.add(tuple(int(dim) for dim in shape))
        vectors.update((dim,) for dim in shape if type(dim) is int)
    return _ShapeRule(
        source,
        output,
        "Reshape" if reshape else "Expand",
        permutation,
        dimensions,
        tuple((vector, _literal(vector)) for vector in sorted(vectors)),
    )
