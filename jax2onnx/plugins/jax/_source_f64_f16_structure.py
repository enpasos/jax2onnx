# jax2onnx/plugins/jax/_source_f64_f16_structure.py

"""Exact source eligibility and opt-in mathematical RNE graph inspection.

Nothing in this module registers a lowering or authorizes SOURCE equivalence.
The mathematical spec must be declared independently of emitted graph receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import ClosedJaxpr, Jaxpr, Var
from jax2onnx.converter._source_regions import FrozenBody, FrozenEquation, FrozenUse
from jax2onnx.converter.source_structure import (
    ConstantContract,
    LiteralOperand,
    StructureMismatch,
    StructureUnavailable,
    TensorContract,
    _ShapeFreezer,
)

if TYPE_CHECKING:
    from jax2onnx.converter._source_cast_planning import WholeRootCastEligibility


def freeze_f64_f16_semantic_key(
    equation: Any,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> tuple[object, ...] | None:
    """Source-only typed semantics; this installs no mathematical graph rule."""
    from jax2onnx.converter._source_cast_planning import make_cast_eligibility

    if (
        equation.primitive is not jax.lax.convert_element_type_p
        or equation.effects
        or len(inputs) != 1
        or len(outputs) != 1
        or type(equation.params) is not dict
        or set(equation.params) != {"new_dtype", "weak_type", "sharding"}
        or not isinstance(equation.params["new_dtype"], np.dtype)
        or equation.params["new_dtype"] != np.dtype(np.float16)
        or equation.params["weak_type"] is not False
        or equation.params["sharding"] is not None
    ):
        return None
    params = (("new_dtype", "float16"), ("sharding", None), ("weak_type", False))
    body = FrozenBody(
        inputs,
        (),
        (
            FrozenEquation(
                (FrozenUse("input", 0, 0, inputs[0]),),
                outputs,
                ("native-f64-f16-convert", params),
            ),
        ),
        (FrozenUse("equation", 0, 0, outputs[0]),),
    )
    try:
        return make_cast_eligibility(body, params).body.equations[0].semantics
    except (ValueError, TypeError, AttributeError):
        return None


def freeze_root_cast(closed: object) -> WholeRootCastEligibility | None:
    """Inspect one already-produced original trace, without retaining native IR."""
    from jax2onnx.converter._source_cast_planning import make_cast_eligibility

    if type(closed) is not ClosedJaxpr:
        return None
    closed_native = cast(ClosedJaxpr, closed)
    native: Any = closed_native.jaxpr
    if type(native) is not Jaxpr:
        return None
    if (
        native.effects
        or native.constvars
        or closed_native.consts
        or len(native.invars) != 1
        or len(native.outvars) != 1
        or len(native.eqns) != 1
    ):
        return None
    equation, source, result = native.eqns[0], native.invars[0], native.outvars[0]
    if (
        equation.primitive is not jax.lax.convert_element_type_p
        or equation.effects
        or type(source) is not Var
        or type(result) is not Var
        or len(equation.invars) != 1
        or equation.invars[0] is not source
        or len(equation.outvars) != 1
        or equation.outvars[0] is not result
        or type(equation.params) is not dict
        or set(equation.params) != {"new_dtype", "weak_type", "sharding"}
        or not isinstance(equation.params["new_dtype"], np.dtype)
        or equation.params["new_dtype"] != np.dtype(np.float16)
        or equation.params["weak_type"] is not False
        or equation.params["sharding"] is not None
        or type(source.aval.weak_type) is not bool
        or source.aval.weak_type is not False
        or type(result.aval.weak_type) is not bool
        or result.aval.weak_type is not False
    ):
        return None
    for aval in (source.aval, result.aval):
        if (
            type(aval) is not jax.core.ShapedArray
            or type(aval.shape) is not tuple
            or any(type(dim) is not int or dim < 0 for dim in aval.shape)
        ):
            return None
    try:
        shapes = _ShapeFreezer()
        source_tensor = TensorContract.from_aval(source.aval, shapes)
        result_tensor = TensorContract.from_aval(result.aval, shapes)
        params = (("new_dtype", "float16"), ("sharding", None), ("weak_type", False))
        body = FrozenBody(
            (source_tensor,),
            (),
            (
                FrozenEquation(
                    (FrozenUse("input", 0, 0, source_tensor),),
                    (result_tensor,),
                    ("native-f64-f16-convert", params),
                ),
            ),
            (FrozenUse("equation", 0, 0, result_tensor),),
        )
        return make_cast_eligibility(body, params)
    except (ValueError, TypeError, AttributeError):
        return None


@dataclass(frozen=True, slots=True)
class F64F16RneSpec:
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.shape) is not tuple or any(
            type(dim) is not int or dim < 0 for dim in self.shape
        ):
            raise ValueError("RNE mathematical spec requires exact concrete dimensions")


@dataclass(frozen=True)
class RneGraphMatch:
    nodes: tuple[ir.Node, ...]
    constants: tuple[ir.Value, ...]
    literal_obligations: tuple[tuple[ir.Value, LiteralOperand], ...]


def check_f64_f16_rne_graph(
    spec: F64F16RneSpec, graph: ir.Graph, source: ir.Value, result: ir.Value
) -> RneGraphMatch:
    """Check the complete fixed mathematical graph, independently of emission."""
    if type(spec) is not F64F16RneSpec:
        raise StructureUnavailable("rne-mathematical-spec-unavailable")
    if type(spec.shape) is not tuple or any(
        type(dim) is not int or dim < 0 for dim in spec.shape
    ):
        raise StructureMismatch("rne-mathematical-shape-differs")
    opset = graph.opset_imports.get("")
    if opset is None:
        raise StructureUnavailable("rne-standard-opset-unavailable")
    if type(opset) is not int or opset < 12:
        raise StructureMismatch("rne-standard-opset-differs")
    graph_nodes = tuple(graph)
    node_set = set(graph_nodes)
    checked: set[ir.Node] = set()
    constants: set[ir.Value] = set()
    obligations: dict[tuple[str, int], tuple[ir.Value, LiteralOperand]] = {}

    def tensor(
        value: ir.Value | None, dtype: ir.DataType, shape: tuple[int, ...]
    ) -> ir.Value:
        if value is None or type(value) is not ir.Value:
            raise StructureMismatch("rne-value-missing")
        if value.type is None or value.dtype is None or value.shape is None:
            raise StructureUnavailable("rne-type-or-shape-unavailable")
        if (
            not isinstance(value.type, ir.TensorType)
            or value.dtype is not dtype
            or any(type(dim) is not int for dim in value.shape)
            or tuple(value.shape) != shape
        ):
            raise StructureMismatch("rne-type-or-shape-differs")
        if value.graph is not graph:
            raise StructureMismatch("rne-value-owner-differs")
        return value

    def operation(
        value: ir.Value | None,
        name: str,
        arity: int,
        dtype: ir.DataType = ir.DataType.DOUBLE,
        attributes: tuple[tuple[str, int], ...] = (),
    ) -> ir.Node:
        value = tensor(value, dtype, spec.shape)
        node = value.producer()
        if node is None:
            raise StructureMismatch("rne-operator-missing")
        if (
            node.graph is not graph
            or node not in node_set
            or node.domain
            or node.overload
            or node.op_type != name
            or len(node.inputs) != arity
            or len(node.outputs) != 1
            or node.outputs[0] is not value
            or set(node.attributes) != {key for key, _ in attributes}
        ):
            raise StructureMismatch("rne-operator-contract-differs")
        for key, expected in attributes:
            attribute = node.attributes[key]
            if (
                attribute.type is not ir.AttributeType.INT
                or attribute.as_int() != expected
            ):
                raise StructureMismatch("rne-attribute-differs")
        checked.add(node)
        return node

    def scalar(value: ir.Value | None, exponent: int, role: tuple[str, int]) -> None:
        value = tensor(value, ir.DataType.DOUBLE, ())
        if value.is_graph_input() or any(value is item for item in graph.inputs):
            raise StructureMismatch("rne-constant-runtime-substitute")
        producer = value.producer()
        if producer is None:
            if not any(value is item for item in graph.initializers.values()):
                raise StructureUnavailable("rne-constant-definition-unavailable")
            stored = value.const_value
        elif (
            producer.graph is graph
            and producer in node_set
            and not producer.domain
            and not producer.overload
            and producer.op_type == "Constant"
            and not producer.inputs
            and len(producer.outputs) == 1
            and producer.outputs[0] is value
            and set(producer.attributes) == {"value"}
            and producer.attributes["value"].type is ir.AttributeType.TENSOR
        ):
            stored = producer.attributes["value"].as_tensor()
            checked.add(producer)
        else:
            raise StructureMismatch("rne-constant-producer-differs")
        if stored is None:
            raise StructureUnavailable("rne-constant-storage-unavailable")
        if stored.dtype is not ir.DataType.DOUBLE or tuple(stored.shape) != ():
            raise StructureMismatch("rne-constant-tensor-metadata-differs")
        expected_array = np.asarray(2.0**exponent, dtype=np.float64)
        expected = ConstantContract.from_array(expected_array)
        try:
            actual_storage = stored.numpy()
        except OSError as exc:
            raise StructureUnavailable("rne-constant-storage-unavailable") from exc
        if ConstantContract.from_array(actual_storage) != expected:
            raise StructureMismatch("rne-constant-storage-differs")
        scalar_tensor = TensorContract(
            "float64", (), False, ir.DataType.DOUBLE, (), "identity", None, None, None
        )
        obligations[role] = (value, LiteralOperand(scalar_tensor, expected))
        constants.add(value)

    tensor(source, ir.DataType.DOUBLE, spec.shape)
    tensor(result, ir.DataType.FLOAT16, spec.shape)
    if (
        len(graph.inputs) != 1
        or graph.inputs[0] is not source
        or source.producer() is not None
    ):
        raise StructureMismatch("rne-original-dynamic-input-differs")
    if len(graph.outputs) != 1:
        raise StructureMismatch("rne-result-interface-differs")
    if graph.outputs[0] is not result:
        alias = operation(graph.outputs[0], "Identity", 1, ir.DataType.FLOAT16)
        if alias.inputs[0] is not result:
            raise StructureMismatch("rne-result-alias-edge-differs")
    cast = operation(
        result,
        "Cast",
        1,
        ir.DataType.FLOAT16,
        (("to", int(ir.DataType.FLOAT16.value)),),
    )
    divided = operation(cast.inputs[0], "Div", 2)
    rounded = operation(divided.inputs[0], "Round", 1)
    multiplied = operation(rounded.inputs[0], "Mul", 2)
    if (
        multiplied.inputs[0] is not source
        or multiplied.inputs[1] is not divided.inputs[1]
    ):
        raise StructureMismatch("rne-shared-scale-or-source-edge-differs")
    unit = divided.inputs[1]
    absolute_value: ir.Value | None = None
    for exponent in range(15, -14, -1):
        selected = operation(unit, "Where", 3)
        predicate = operation(selected.inputs[0], "GreaterOrEqual", 2, ir.DataType.BOOL)
        if absolute_value is None:
            absolute_value = predicate.inputs[0]
        elif predicate.inputs[0] is not absolute_value:
            raise StructureMismatch("rne-threshold-source-edge-differs")
        scalar(predicate.inputs[1], exponent, ("threshold", exponent))
        scalar(selected.inputs[1], 10 - exponent, ("scale", exponent))
        unit = selected.inputs[2]
    scalar(unit, 24, ("base", 0))
    absolute = operation(absolute_value, "Abs", 1)
    if absolute.inputs[0] is not source:
        raise StructureMismatch("rne-absolute-source-edge-differs")
    if checked != node_set:
        raise StructureMismatch("rne-node-coverage-differs")
    positions = {node: index for index, node in enumerate(graph_nodes)}
    for node in graph_nodes:
        for value in node.inputs:
            producer = value.producer() if value is not None else None
            if producer is not None and (
                producer not in positions or positions[producer] >= positions[node]
            ):
                raise StructureMismatch("rne-producer-position-differs")
    if set(graph.initializers.values()) - constants:
        raise StructureMismatch("rne-initializer-coverage-differs")
    ordered_roles = (("base", 0),) + tuple(
        role
        for exponent in range(-13, 16)
        for role in (("threshold", exponent), ("scale", exponent))
    )
    return RneGraphMatch(
        tuple(node for node in graph_nodes if node in checked),
        tuple(
            dict.fromkeys(
                value for value, _ in (obligations[role] for role in ordered_roles)
            )
        ),
        tuple(obligations[role] for role in ordered_roles),
    )
