# jax2onnx/plugins/_source_control_flow_structure.py

"""Original dynamic If contracts; neither source admission nor graph rewriting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import ClosedJaxpr, Jaxpr, Literal
from jax2onnx.converter.source_structure import (
    ConstantContract,
    EquationHandle,
    LiteralOperand,
    RegionKey,
    StructureMismatch,
    StructureUnavailable,
    TensorContract,
    _ORIGIN,
    _OWNER,
    _ValueWitness,
)


@dataclass(frozen=True)
class BooleanIndexRule:
    """Exact scalar BOOL -> strong INT32, including a proved composed use."""

    def check(self, graph: Any, inputs: Any, outputs: Any, match_literal: Any) -> Any:
        if len(inputs) != 1 or len(outputs) != 1 or not isinstance(inputs[0], ir.Value):
            raise StructureUnavailable("boolean-index-ports-unavailable")
        node = outputs[0].producer()
        if (
            node is None
            or node.graph is not graph
            or node not in set(graph)
            or node.domain
            or node.overload
            or node.op_type != "Cast"
            or len(node.inputs) != 1
            or node.inputs[0] is not inputs[0]
            or len(node.outputs) != 1
            or node.outputs[0] is not outputs[0]
            or set(node.attributes) != {"to"}
            or node.attributes["to"].type is not ir.AttributeType.INT
            or node.attributes["to"].as_int() != int(ir.DataType.INT32.value)
        ):
            raise StructureMismatch("boolean-index-cast-differs")
        return (node,)


@dataclass(frozen=True)
class IfRule:
    """Child body content lives once in the frozen equation's native roles."""

    def check(self, graph: Any, inputs: Any, outputs: Any, match_literal: Any) -> Any:
        raise StructureUnavailable("dynamic-if-attachment-unavailable")


@dataclass(frozen=True)
class LiteralIfRule:
    """Original strong scalar selector; no final selector tensor is implied."""

    index: int
    selector: LiteralOperand

    def check(self, graph: Any, inputs: Any, outputs: Any, match_literal: Any) -> Any:
        raise StructureUnavailable("literal-if-attachment-unavailable")


@dataclass(frozen=True)
class LiteralIfBody:
    invocation: object
    owner: object
    body: Any
    equations: tuple[Any, ...]
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    input_values: tuple[Any, ...]
    closed_constant_values: tuple[_ValueWitness, ...]
    variables: tuple[tuple[int, Any], ...]
    receipts: tuple[tuple[int, Any], ...]
    result_values: tuple[Any, ...] | None
    region: Any


@dataclass(frozen=True)
class LiteralIfDischarge:
    """Disjoint original obligations; unreachable bodies claim no physical IR."""

    selector: LiteralOperand
    inactive_role: str
    inactive_invocation: object
    inactive_body: Any
    graph_matched_equations: int
    semantically_discharged_equations: int
    inactive_nodes: tuple[()] = ()
    inactive_storage: tuple[()] = ()


@dataclass(frozen=True)
class LiteralIfAttachment:
    handle: EquationHandle
    owner: object
    rule: LiteralIfRule
    false: LiteralIfBody
    true: LiteralIfBody
    values: tuple[tuple[Any, Any], ...]
    node_origins: tuple[object, ...]
    node_ports: tuple[int, ...]
    original_returned: tuple[Any, ...]
    returned: tuple[Any, ...]
    results: tuple[Any, ...]
    discharge: LiteralIfDischarge


def literal_if_body(scope: Any) -> LiteralIfBody:
    return LiteralIfBody(
        scope.invocation,
        scope.owner,
        scope.body,
        scope.equations,
        scope.inputs,
        scope.outputs,
        scope.input_values,
        scope.closed_constant_values,
        tuple(sorted(scope.variables.items())),
        tuple(sorted(scope.receipts.items())),
        scope.result_values,
        scope.region,
    )


@dataclass(frozen=True)
class IfAttachment:
    origin: object
    owner: object
    true: RegionKey
    false: RegionKey
    true_owner: object
    false_owner: object


def freeze_control_flow_rule(
    equation: Any,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> BooleanIndexRule | IfRule | LiteralIfRule | None:
    if equation.effects:
        return None
    if equation.primitive is jax.lax.convert_element_type_p:
        params = equation.params
        if (
            len(inputs) == len(outputs) == 1
            and inputs[0].logical_dtype == "bool"
            and inputs[0].dtype is ir.DataType.BOOL
            and outputs[0].logical_dtype == "int32"
            and outputs[0].dtype is ir.DataType.INT32
            and inputs[0].shape == outputs[0].shape == ()
            and inputs[0].logical_shape == outputs[0].logical_shape == ()
            and inputs[0].encoding == outputs[0].encoding == "identity"
            and outputs[0].weak_type is False
            and type(params) is dict
            and {"new_dtype", "weak_type"}
            <= set(params)
            <= {"new_dtype", "weak_type", "sharding"}
            and type(params["new_dtype"]) is type(np.dtype(np.int32))
            and params["new_dtype"] == np.dtype(np.int32)
            and params["weak_type"] is False
            and params.get("sharding") is None
        ):
            return BooleanIndexRule()
        return None
    if equation.primitive is not jax.lax.cond_p:
        return None
    if (
        not inputs
        or not outputs
        or type(equation.params) is not dict
        or set(equation.params) != {"branches"}
        or inputs[0].logical_dtype != "int32"
        or inputs[0].dtype is not ir.DataType.INT32
        or inputs[0].weak_type is not False
        or inputs[0].shape != ()
        or inputs[0].logical_shape != ()
        or inputs[0].encoding != "identity"
        or any(value.symbol_token is not None for value in (*inputs, *outputs))
    ):
        return None
    branches = equation.params["branches"]
    if type(branches) is not tuple or len(branches) != 2:
        return None
    from jax2onnx.converter.source_structure import _ShapeFreezer

    for closed in branches:
        if type(closed) is not ClosedJaxpr and type(closed) is not Jaxpr:
            return None
        body = closed.jaxpr
        if body.effects or len(body.constvars) != len(closed.consts):
            return None
        shapes = _ShapeFreezer()
        if (
            tuple(TensorContract.from_aval(var.aval, shapes) for var in body.invars)
            != inputs[1:]
            or tuple(TensorContract.from_aval(var.aval, shapes) for var in body.outvars)
            != outputs
        ):
            return None
    if isinstance(equation.invars[0], Literal):
        selector = equation.invars[0]
        if type(selector.val) is int:
            index = selector.val
        else:
            payload = np.asarray(selector.val)
            if payload.dtype != np.dtype(np.int32) or payload.shape != ():
                return None
            index = int(payload.item())
        if index not in (0, 1):
            return None
        payload = np.asarray(index, dtype=np.int32)
        return LiteralIfRule(
            index, LiteralOperand(inputs[0], ConstantContract.from_array(payload))
        )
    return IfRule()


def record_if_attachment(
    ctx: Any,
    equation: Any,
    true: RegionKey,
    false: RegionKey,
    true_graph: ir.Graph,
    false_graph: ir.Graph,
) -> None:
    ledger = ctx._source_structure
    if not ledger.handles or type(ledger.handles[-1]) is not EquationHandle:
        raise StructureUnavailable("original-if-handle-unavailable")
    handle = ledger.handles[-1]
    if handle in ledger.if_calls or true.parent != handle or false.parent != handle:
        raise StructureUnavailable("if-attachment-identity-unavailable")
    outputs = tuple(ctx.lookup_value_for_var(var) for var in equation.outvars)
    if not outputs or outputs[0] is None or outputs[0].producer() is None:
        raise StructureUnavailable("if-attachment-output-unavailable")
    node = outputs[0].producer()
    ledger.if_calls[handle] = IfAttachment(
        node.meta.setdefault(_ORIGIN, object()),
        ctx.builder.graph.meta[_OWNER],
        true,
        false,
        true_graph.meta[_OWNER],
        false_graph.meta[_OWNER],
    )


def prepare_literal_if_clone(ctx: Any, graph: ir.Graph) -> None:
    """Stamp actual staged definitions before the existing numerical clone."""
    from jax2onnx.converter.source_structure import _witness

    for node in graph:
        for value in node.outputs:
            _witness(value)


def record_literal_if_attachment(
    ctx: Any,
    equation: Any,
    graph: ir.Graph,
    nodes: tuple[ir.Node, ...],
    returned: tuple[ir.Value, ...],
) -> None:
    """Observe the producer's actual clone map after its existing attachment."""
    from jax2onnx.converter.source_structure import _witness

    ledger = ctx._source_structure
    handle = ledger.handles[-1] if ledger.handles else None
    if type(handle) is not EquationHandle or handle in ledger.literal_if_calls:
        raise StructureUnavailable("literal-if-handle-unavailable")
    parent = next(
        (
            item
            for item in ledger.active
            if item is not None and item.invocation is handle.invocation
        ),
        None,
    )
    if parent is None:
        raise StructureUnavailable("literal-if-parent-unavailable")
    rule = parent.equations[handle.index].rule
    if type(rule) is not LiteralIfRule:
        raise StructureUnavailable("literal-if-rule-unavailable")
    bodies = []
    for role in ("false", "true"):
        invocation = ledger.regions.get(RegionKey(handle, role))
        matches = [scope for scope in ledger.scopes if scope.invocation is invocation]
        if len(matches) != 1:
            raise StructureUnavailable("literal-if-original-body-unavailable")
        scope = matches[0]
        if (
            scope.body is None
            or scope.body.gaps
            or any(item.children for item in scope.body.equations)
            or any(item.semantics is None for item in scope.body.equations)
            or set(scope.receipts) != set(range(len(scope.equations)))
            or scope.result_values is None
            or scope.pending
            or scope.region is None
        ):
            raise StructureUnavailable("literal-if-childless-contract-unavailable")
        bodies.append(literal_if_body(scope))
    selected, inactive = bodies[rule.index], bodies[1 - rule.index]
    if graph.meta.get(_OWNER) is not selected.owner:
        raise StructureMismatch("literal-if-selected-owner-differs")
    originals = tuple(graph)
    if len(originals) != len(nodes) or len(graph.outputs) != len(returned):
        raise StructureMismatch("literal-if-selected-clone-arity-differs")
    mapping = []
    for original, clone in zip(originals, nodes, strict=True):
        if (
            original.meta.get(_ORIGIN) is not clone.meta.get(_ORIGIN)
            or clone.graph is not ctx.builder.graph
            or len(original.outputs) != len(clone.outputs)
        ):
            raise StructureMismatch("literal-if-selected-clone-origin-differs")
        for before, after in zip(original.outputs, clone.outputs, strict=True):
            source, target = _witness(before), _witness(after)
            mapping.append((source, target))
            if source in ledger.constants:
                if target in ledger.constants:
                    raise StructureUnavailable("literal-if-constant-replay-unavailable")
                ledger.constants[target] = ledger.constants[source]
    values = dict(mapping)
    original_returns = tuple(_witness(value) for value in graph.outputs)
    actual_returns = tuple(_witness(value) for value in returned)
    if any(
        values.get(before) != after
        for before, after in zip(original_returns, actual_returns, strict=True)
    ):
        raise StructureMismatch("literal-if-selected-result-map-differs")
    results = tuple(_witness(ctx.lookup_value_for_var(var)) for var in equation.outvars)
    ledger.literal_if_calls[handle] = LiteralIfAttachment(
        handle,
        ctx.builder.graph.meta[_OWNER],
        rule,
        bodies[0],
        bodies[1],
        tuple(mapping),
        tuple(node.meta[_ORIGIN] for node in nodes),
        tuple(len(node.outputs) for node in originals),
        original_returns,
        actual_returns,
        results,
        LiteralIfDischarge(
            rule.selector,
            "true" if rule.index == 0 else "false",
            inactive.invocation,
            inactive.body,
            len(selected.equations),
            1 + len(inactive.equations),
        ),
    )
