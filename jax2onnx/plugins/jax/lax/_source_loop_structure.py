# jax2onnx/plugins/jax/lax/_source_loop_structure.py

"""Original While/Scan roles and independent final Loop port/shape proofs.

This module observes existing traces and final IR. It neither admits new source
programs nor changes graph metadata. Each original invocation stays distinct.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import (
    ClosedJaxpr,
    Jaxpr,
    native_scan_flat_tree_partitions,
    native_scan_parameter_names,
    scan_arity,
)
from jax2onnx.converter.source_structure import (
    EquationHandle,
    LiteralOperand,
    RegionKey,
    StructureMismatch,
    StructureUnavailable,
    TensorContract,
    _ORIGIN,
    _OWNER,
    _ShapeFreezer,
    _ValueWitness,
    _resolve_bindings,
    _unused_inputs,
    _witness,
)


@dataclass(frozen=True)
class WhileRule:
    condition_constants: int
    body_constants: int
    carries: int

    def check(self, graph: Any, inputs: Any, outputs: Any, match_literal: Any) -> Any:
        raise StructureUnavailable("while-attachment-unavailable")

    def input_ports(self, role: str, count: int) -> tuple[int, ...]:
        nc, nb = self.condition_constants, self.body_constants
        carry = tuple(range(nc + nb, count))
        if role in {"initial-condition", "next-condition"}:
            return (*range(nc), *carry)
        if role == "body":
            return (*range(nc, nc + nb), *carry)
        raise StructureUnavailable("while-original-role-unavailable")


@dataclass(frozen=True)
class ScanRule:
    constants: int
    carries: int
    sequences: int
    results: int
    length: int
    unroll: int
    linear: tuple[bool, ...] | None
    split_transpose: bool | None
    partitions: tuple[tuple[int, ...], tuple[int, ...]] | None

    def check(self, graph: Any, inputs: Any, outputs: Any, match_literal: Any) -> Any:
        raise StructureUnavailable("scan-attachment-unavailable")


def _freeze_scan_rule(
    equation: Any,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> ScanRule | None:
    params = equation.params
    fields = native_scan_parameter_names()
    if (
        equation.effects
        or type(params) is not dict
        or not fields <= set(params) <= fields | {"sharding", "out_sharding"}
        or any(params.get(key) is not None for key in ("sharding", "out_sharding"))
        or type(params["reverse"]) is not bool
        or params["reverse"]
        or type(params["length"]) is not int
        or params["length"] < 0
        or type(params["unroll"]) is not int
        or params["unroll"] < 0
        or any(
            value.encoding != "identity" or value.symbol_token is not None
            for value in (*inputs, *outputs)
        )
    ):
        return None
    closed = params["jaxpr"]
    if type(closed) is not ClosedJaxpr and type(closed) is not Jaxpr:
        return None
    body = closed.jaxpr  # Native JAX 0.11+ may return self.
    if body.effects or len(body.constvars) != len(closed.consts):
        return None
    partitions = native_scan_flat_tree_partitions(params)
    if partitions is None:
        if any(
            type(params[key]) is not int or params[key] < 0
            for key in ("num_consts", "num_carry")
        ):
            return None
    elif not partitions[2]:
        return None
    nc, nk, nx = scan_arity(params, len(body.invars))
    ny = len(body.outvars) - nk
    if (
        ny < 0
        or len(inputs) != nc + nk + nx
        or len(outputs) != nk + ny
        or partitions is not None
        and (partitions[0] != (nc, nk, nx) or partitions[1] != (nk, ny))
    ):
        return None
    linear = params.get("linear")
    split = params.get("_split_transpose")
    if "linear" in fields and (
        type(linear) is not tuple
        or len(linear) != len(inputs)
        or any(type(value) is not bool for value in linear)
        or type(split) is not bool
    ):
        return None
    length = params["length"]
    sequence_inputs = inputs[nc + nk :]
    if any(
        not value.shape or value.shape[0] != length or value.logical_shape[0] != length
        for value in sequence_inputs
    ):
        return None
    shapes = _ShapeFreezer()
    incoming = tuple(TensorContract.from_aval(var.aval, shapes) for var in body.invars)
    outgoing = tuple(TensorContract.from_aval(var.aval, shapes) for var in body.outvars)
    expected_inputs = (
        *inputs[: nc + nk],
        *(
            replace(value, shape=value.shape[1:], logical_shape=value.logical_shape[1:])
            for value in sequence_inputs
        ),
    )
    if incoming != expected_inputs:
        return None
    # Native result weakness is independently validated by SOURCE admission.
    # Carries preserve value dtype/shape even if a weak initial value strengthens.
    for initial, final in zip(inputs[nc : nc + nk], outgoing[:nk], strict=True):
        if replace(initial, weak_type=final.weak_type) != final:
            return None
    expected_outputs = (
        *outgoing[:nk],
        *(
            replace(
                value,
                shape=(length, *value.shape),
                logical_shape=(length, *value.logical_shape),
            )
            for value in outgoing[nk:]
        ),
    )
    if any(
        replace(expected, weak_type=actual.weak_type) != actual
        for expected, actual in zip(expected_outputs, outputs, strict=True)
    ):
        return None
    return ScanRule(
        nc,
        nk,
        nx,
        ny,
        length,
        params["unroll"],
        linear,
        split,
        None if partitions is None else partitions[:2],
    )


def freeze_loop_rule(
    equation: Any,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> WhileRule | ScanRule | None:
    if equation.primitive is jax.lax.scan_p:
        return _freeze_scan_rule(equation, inputs, outputs)
    if equation.primitive is not jax.lax.while_p or equation.effects:
        return None
    params = equation.params
    fields = {"cond_jaxpr", "body_jaxpr", "cond_nconsts", "body_nconsts"}
    if (
        type(params) is not dict
        or not fields <= set(params) <= fields | {"sharding", "out_sharding"}
        or params.get("sharding") is not None
        or params.get("out_sharding") is not None
        or any(
            value.encoding != "identity" or value.symbol_token is not None
            for value in (*inputs, *outputs)
        )
    ):
        return None
    nc, nb = params["cond_nconsts"], params["body_nconsts"]
    if (
        type(nc) is not int
        or type(nb) is not int
        or nc < 0
        or nb < 0
        or nc + nb >= len(inputs)
        or len(outputs) != len(inputs) - nc - nb
    ):
        return None
    rule = WhileRule(nc, nb, len(outputs))
    for key, role in (("cond_jaxpr", "initial-condition"), ("body_jaxpr", "body")):
        closed = params[key]
        if type(closed) is not ClosedJaxpr and type(closed) is not Jaxpr:
            return None
        body = closed.jaxpr
        if body.effects or len(body.constvars) != len(closed.consts):
            return None
        shapes = _ShapeFreezer()
        incoming = tuple(
            TensorContract.from_aval(var.aval, shapes) for var in body.invars
        )
        outgoing = tuple(
            TensorContract.from_aval(var.aval, shapes) for var in body.outvars
        )
        if incoming != tuple(
            inputs[port] for port in rule.input_ports(role, len(inputs))
        ):
            return None
        if role == "body":
            if outgoing != outputs:
                return None
        elif (
            len(outgoing) != 1
            or outgoing[0].dtype is not ir.DataType.BOOL
            or outgoing[0].logical_dtype != "bool"
            or outgoing[0].shape != ()
            or outgoing[0].logical_shape != ()
            or outgoing[0].encoding != "identity"
        ):
            return None
    for carry_input, carry_output in zip(inputs[nc + nb :], outputs, strict=True):
        if (
            carry_input.dtype != carry_output.dtype
            or carry_input.logical_dtype != carry_output.logical_dtype
            or carry_input.shape != carry_output.shape
            or carry_input.logical_shape != carry_output.logical_shape
        ):
            return None
    return rule


@dataclass(frozen=True)
class WhileAttachment:
    origin: object
    owner: object
    body_owner: object
    regions: tuple[RegionKey, RegionKey, RegionKey]
    lazy: tuple[tuple[_ValueWitness, _ValueWitness], ...]


def record_while_attachment(
    ctx: Any,
    equation: Any,
    initial: RegionKey,
    body: RegionKey,
    next_condition: RegionKey,
    body_graph: ir.Graph,
    call_parameters: Any,
) -> None:
    ledger = ctx._source_structure
    handle = ledger.handles[-1] if ledger.handles else None
    if type(handle) is not EquationHandle or handle in ledger.while_calls:
        raise StructureUnavailable("while-original-handle-unavailable")
    keys = (initial, body, next_condition)
    if any(type(key) is not RegionKey or key.parent != handle for key in keys):
        raise StructureUnavailable("while-attachment-roles-unavailable")
    outputs = tuple(ctx.lookup_value_for_var(var) for var in equation.outvars)
    if not outputs or outputs[0] is None or outputs[0].producer() is None:
        raise StructureUnavailable("while-attachment-output-unavailable")
    node = outputs[0].producer()
    requested = call_parameters.port_bindings
    first_lazy = len(body_graph.inputs) - len(requested)
    lazy = tuple(
        (_witness(outer), _witness(body_graph.inputs[first_lazy + index]))
        for index, (outer, _) in enumerate(requested)
    )
    ledger.while_calls[handle] = WhileAttachment(
        node.meta.setdefault(_ORIGIN, object()),
        ctx.builder.graph.meta[_OWNER],
        body_graph.meta[_OWNER],
        keys,
        lazy,
    )


@dataclass(frozen=True)
class _Fact:
    dtype: ir.DataType
    shape: tuple[int, ...]


class ActualShapeFacts:
    """Read-only shape induction from actual ports, storage and operator edges.

    Internal ValueInfo is checked for contradictions; it supplies no missing
    dimension. No expected source tensor is an input to this fact computation.
    """

    def __init__(self, model: ir.Model, definitions: dict[object, ir.Function]) -> None:
        self.values: dict[ir.Value, _Fact] = {}
        self.required_graphs: set[ir.Graph] = set()
        self.required_values: set[ir.Value] = set()
        self.definitions = {
            (fn.domain, fn.name, getattr(fn, "overload", "")): fn.graph
            for fn in definitions.values()
        }
        for value in model.graph.inputs:
            if (
                value.dtype is not None
                and value.shape is not None
                and all(
                    isinstance(dim, (int, np.integer))
                    and not isinstance(dim, (bool, np.bool_))
                    for dim in value.shape
                )
            ):
                self._record(
                    value,
                    _Fact(
                        value.dtype,
                        tuple(
                            int(cast("int | np.integer[Any]", dim))
                            for dim in value.shape
                        ),
                    ),
                )
        self._graph(model.graph, frozenset(), False)

    def _record(self, value: ir.Value, fact: _Fact) -> None:
        if value.dtype is not None and value.dtype != fact.dtype:
            raise StructureMismatch("while-derived-dtype-differs")
        if value.shape is not None:
            if len(value.shape) != len(fact.shape):
                raise StructureMismatch("while-derived-rank-differs")
            for actual, derived in zip(value.shape, fact.shape, strict=True):
                if isinstance(actual, (int, np.integer)) and not isinstance(
                    actual, (bool, np.bool_)
                ):
                    if int(actual) != derived:
                        raise StructureMismatch(
                            "while-derived-concrete-dimension-differs"
                        )
                elif actual is not None and not (
                    type(actual) is ir.SymbolicDim and actual.value is None
                ):
                    raise StructureUnavailable("while-derived-metadata-unavailable")
        previous = self.values.get(value)
        if previous is not None and previous != fact:
            raise StructureMismatch("while-derived-facts-conflict")
        self.values[value] = fact

    def matches(self, tensor: TensorContract, value: ir.Value) -> bool:
        fact = self.values.get(value)
        required = value.graph in self.required_graphs or value in self.required_values
        if fact is None:
            if required:
                raise StructureUnavailable("while-actual-shape-fact-unavailable")
            return tensor.matches(value)
        self._record(value, fact)
        if value.dtype is None or value.shape is None:
            raise StructureUnavailable("while-actual-type-metadata-unavailable")
        return tensor.dtype == fact.dtype and tensor.shape == fact.shape

    @staticmethod
    def _storage(value: ir.Value) -> Any:
        node = value.producer()
        if node is None:
            return value.const_value if value.is_initializer() else None
        if (
            not node.domain
            and not node.overload
            and node.op_type == "Constant"
            and not node.inputs
            and len(node.outputs) == 1
            and set(node.attributes) == {"value"}
            and node.attributes["value"].type is ir.AttributeType.TENSOR
        ):
            return node.attributes["value"].as_tensor()
        return None

    def _graph(
        self, graph: ir.Graph, ancestors: frozenset[ir.Graph], required: bool
    ) -> None:
        if graph in ancestors or len(ancestors) >= 64:
            raise StructureUnavailable("while-shape-recursive-graph-unavailable")
        if required:
            self.required_graphs.add(graph)
        ancestors = ancestors | {graph}
        for value in (
            *graph.initializers.values(),
            *(out for node in graph for out in node.outputs),
        ):
            tensor = self._storage(value)
            if tensor is not None:
                self._record(
                    value, _Fact(tensor.dtype, tuple(int(dim) for dim in tensor.shape))
                )
        unary = {"Identity", "Neg", "Exp", "Sin", "Cos", "Sqrt", "Not"}
        binary = {
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Less",
            "LessOrEqual",
            "Greater",
            "GreaterOrEqual",
            "Equal",
            "And",
            "Or",
        }
        predicates = {
            "Less",
            "LessOrEqual",
            "Greater",
            "GreaterOrEqual",
            "Equal",
            "And",
            "Or",
        }
        for node in graph:
            arguments = tuple(
                self.values.get(value) if value is not None else None
                for value in node.inputs
            )
            results: tuple[_Fact | None, ...] = ()
            target = self.definitions.get((node.domain, node.op_type, node.overload))
            if (
                target is not None
                and len(target.inputs) == len(arguments)
                and all(fact is not None for fact in arguments)
            ):
                for formal, fact in zip(target.inputs, arguments, strict=True):
                    assert fact is not None
                    self._record(formal, fact)
                self._graph(target, ancestors, required)
                results = tuple(self.values.get(value) for value in target.outputs)
            elif node.domain or node.overload:
                continue
            elif node.op_type == "If" and set(node.attributes) == {
                "then_branch",
                "else_branch",
            }:
                branches = []
                for name in ("then_branch", "else_branch"):
                    attribute = node.attributes[name]
                    if attribute.type is not ir.AttributeType.GRAPH:
                        break
                    child = attribute.as_graph()
                    self._graph(child, ancestors, required)
                    branches.append(
                        tuple(self.values.get(value) for value in child.outputs)
                    )
                if len(branches) == 2 and branches[0] == branches[1]:
                    results = branches[0]
            elif (
                node.op_type == "Loop"
                and set(node.attributes) == {"body"}
                and len(arguments) >= 2
            ):
                attribute = node.attributes["body"]
                if attribute.type is not ir.AttributeType.GRAPH:
                    continue
                child = attribute.as_graph()
                self.required_values.update(node.outputs)
                if len(child.inputs) != len(arguments) or not all(
                    fact is not None for fact in arguments[2:]
                ):
                    continue
                self._record(child.inputs[0], _Fact(ir.DataType.INT64, ()))
                self._record(child.inputs[1], _Fact(ir.DataType.BOOL, ()))
                for formal, fact in zip(child.inputs[2:], arguments[2:], strict=True):
                    assert fact is not None
                    self._record(formal, fact)
                self._graph(child, ancestors, True)
                outgoing = tuple(self.values.get(value) for value in child.outputs)
                carried_count = len(arguments) - 2
                if len(outgoing) < 1 + carried_count:
                    continue
                for initial, final in zip(
                    arguments[2:], outgoing[1 : 1 + carried_count], strict=True
                ):
                    if final is not None and initial != final:
                        raise StructureMismatch("while-carried-shape-induction-differs")
                results = outgoing[1 : 1 + carried_count]
                scanned = outgoing[1 + carried_count :]
                if scanned:
                    trip = (
                        self._storage(node.inputs[0])
                        if node.inputs[0] is not None
                        else None
                    )
                    condition = (
                        self._storage(node.inputs[1])
                        if node.inputs[1] is not None
                        else None
                    )
                    if trip is None or condition is None:
                        continue
                    trip_array, condition_array = trip.numpy(), condition.numpy()
                    if (
                        trip.dtype is not ir.DataType.INT64
                        or tuple(trip.shape) != ()
                        or condition.dtype is not ir.DataType.BOOL
                        or tuple(condition.shape) != ()
                        or not bool(condition_array.item())
                    ):
                        continue
                    _identity_edge(child, child.outputs[0], child.inputs[1])
                    length = int(trip_array.item())
                    if length < 0:
                        continue
                    results = (
                        *results,
                        *(
                            (
                                _Fact(fact.dtype, (length, *fact.shape))
                                if fact is not None
                                else None
                            )
                            for fact in scanned
                        ),
                    )
            elif (
                node.op_type == "Gather"
                and len(arguments) == 2
                and all(fact is not None for fact in arguments)
                and set(node.attributes) == {"axis"}
            ):
                data, index = arguments
                axis = node.attributes["axis"]
                assert data is not None and index is not None
                if (
                    axis.type is ir.AttributeType.INT
                    and axis.as_int() == 0
                    and data.shape
                    and index.dtype is ir.DataType.INT64
                    and index.shape == ()
                ):
                    results = (_Fact(data.dtype, data.shape[1:]),)
            elif (
                node.op_type in unary
                and len(arguments) == 1
                and arguments[0] is not None
            ):
                results = (arguments[0],)
            elif (
                node.op_type == "Cast"
                and len(arguments) == 1
                and arguments[0] is not None
                and set(node.attributes) == {"to"}
            ):
                attribute = node.attributes["to"]
                if attribute.type is ir.AttributeType.INT:
                    results = (
                        _Fact(ir.DataType(attribute.as_int()), arguments[0].shape),
                    )
            elif (
                node.op_type in binary
                and len(arguments) == 2
                and all(fact is not None for fact in arguments)
            ):
                left, right = arguments
                assert left is not None and right is not None
                if left.dtype != right.dtype:
                    continue
                shape = []
                width = max(len(left.shape), len(right.shape))
                for a, b in zip(
                    (1,) * (width - len(left.shape)) + left.shape,
                    (1,) * (width - len(right.shape)) + right.shape,
                    strict=True,
                ):
                    if a != b and a != 1 and b != 1:
                        break
                    shape.append(b if a == 1 else a)
                else:
                    results = (
                        _Fact(
                            (
                                ir.DataType.BOOL
                                if node.op_type in predicates
                                else left.dtype
                            ),
                            tuple(shape),
                        ),
                    )
            elif (
                node.op_type == "Transpose"
                and len(arguments) == 1
                and arguments[0] is not None
                and set(node.attributes) == {"perm"}
            ):
                attribute = node.attributes["perm"]
                if attribute.type is ir.AttributeType.INTS:
                    permutation = tuple(attribute.as_ints())
                    if sorted(permutation) == list(range(len(arguments[0].shape))):
                        results = (
                            _Fact(
                                arguments[0].dtype,
                                tuple(arguments[0].shape[axis] for axis in permutation),
                            ),
                        )
            if len(results) == len(node.outputs):
                for value, fact in zip(node.outputs, results, strict=True):
                    if fact is not None:
                        self._record(value, fact)


@dataclass(frozen=True)
class WhileAudit:
    node: ir.Node
    regions: tuple[Any, Any, Any]
    inputs: tuple[ir.Value, ...]
    region_inputs: tuple[tuple[ir.Value, ...], ...]
    region_outputs: tuple[tuple[ir.Value, ...], ...]
    adapters: tuple[ir.Node, ...]


def _identity_edge(
    graph: ir.Graph, actual: ir.Value, expected: ir.Value
) -> tuple[ir.Node, ...]:
    if actual is expected:
        return ()
    node = actual.producer()
    if (
        node is None
        or node.graph is not graph
        or node not in set(graph)
        or node.domain
        or node.overload
        or node.op_type != "Identity"
        or node.attributes
        or len(node.inputs) != 1
        or node.inputs[0] is not expected
        or len(node.outputs) != 1
        or node.outputs[0] is not actual
    ):
        raise StructureMismatch("while-passthrough-edge-differs")
    return (node,)


def plan_while_calls(
    ledger: Any, scopes: Any, graphs: Any, parents: Any, matches: Any
) -> dict[EquationHandle, WhileAudit]:
    wanted = {
        EquationHandle(scope.invocation, index)
        for scope in scopes.values()
        for index, equation in enumerate(scope.equations)
        if type(equation.rule) is WhileRule
    }
    if wanted != set(ledger.while_calls):
        raise StructureUnavailable("while-call-coverage-incomplete")
    plans = {}
    for handle, attachment in ledger.while_calls.items():
        if type(attachment) is not WhileAttachment:
            raise StructureUnavailable("while-attachment-unavailable")
        scope = scopes[handle.invocation]
        rule = scope.equations[handle.index].rule
        receipt = scope.receipts.get(handle.index)
        if receipt is None:
            raise StructureUnavailable("while-source-receipt-unavailable")
        graph = graphs[scope.owner]
        nodes = [node for node in graph if node.meta.get(_ORIGIN) is attachment.origin]
        if len(nodes) != 1:
            raise StructureUnavailable("while-final-node-unavailable")
        node = nodes[0]
        body_graph = graphs.get(attachment.body_owner)
        state_count = rule.body_constants + rule.condition_constants + rule.carries
        count = state_count + len(attachment.lazy)
        if (
            attachment.owner is not scope.owner
            or node.domain
            or node.overload
            or node.op_type != "Loop"
            or set(node.attributes) != {"body"}
            or node.attributes["body"].type is not ir.AttributeType.GRAPH
            or node.attributes["body"].as_graph() is not body_graph
            or len(node.inputs) != 2 + count
            or len(node.outputs) != count
            or node.inputs[0] is not None
            or any(value is None for value in node.inputs[1:])
            or body_graph is None
            or len(body_graph.inputs) != 2 + count
            or len(body_graph.outputs) != 1 + count
        ):
            raise StructureMismatch("while-loop-protocol-differs")
        link = parents.get(attachment.body_owner)
        if link is None or link[0] is not graph or link[1] is not node:
            raise StructureMismatch("while-body-ancestry-differs")
        for value, dtype in (
            (node.inputs[1], ir.DataType.BOOL),
            (body_graph.inputs[0], ir.DataType.INT64),
            (body_graph.inputs[1], ir.DataType.BOOL),
            (body_graph.outputs[0], ir.DataType.BOOL),
        ):
            if (
                value is None
                or value.dtype is not dtype
                or value.shape is None
                or tuple(value.shape) != ()
            ):
                raise StructureMismatch("while-header-type-differs")
        nb, nc = rule.body_constants, rule.condition_constants
        outer = tuple(node.inputs[2:])
        formals = tuple(body_graph.inputs[2:])
        assert all(value is not None for value in outer)
        adapters: list[ir.Node] = []
        for index in (*range(nb + nc), *range(state_count, count)):
            adapters.extend(
                _identity_edge(
                    body_graph, body_graph.outputs[1 + index], formals[index]
                )
            )
        parent_inputs = (
            *outer[nb : nb + nc],
            *outer[:nb],
            *outer[nb + nc : state_count],
        )
        resolved = _resolve_bindings(graph, set(receipt.outputs))
        for expected, actual in zip(
            receipt.outputs, node.outputs[nb + nc : state_count], strict=True
        ):
            if resolved[expected] is not actual:
                raise StructureMismatch("while-parent-result-order-differs")
        for index, (outer_witness, local_witness) in enumerate(attachment.lazy):
            outer_values = _resolve_bindings(graph, {outer_witness})
            local_values = _resolve_bindings(body_graph, {local_witness})
            if (
                outer_values[outer_witness] is not outer[state_count + index]
                or local_values[local_witness] is not formals[state_count + index]
            ):
                raise StructureMismatch("while-lazy-parameter-order-differs")
            for value in (
                outer[state_count + index],
                formals[state_count + index],
                body_graph.outputs[1 + state_count + index],
                node.outputs[state_count + index],
            ):
                if (
                    value.dtype is not ir.DataType.BOOL
                    or value.shape is None
                    or tuple(value.shape) != ()
                ):
                    raise StructureMismatch("while-lazy-parameter-type-differs")
        children = []
        frozen_children = dict(scope.body.equations[handle.index].children)
        for role, key, owner in zip(
            ("initial-condition", "body", "next-condition"),
            attachment.regions,
            (scope.owner, attachment.body_owner, attachment.body_owner),
            strict=True,
        ):
            if key != RegionKey(handle, role) or key not in ledger.regions:
                raise StructureUnavailable("while-region-role-unavailable")
            child = scopes.get(ledger.regions[key])
            if (
                child is None
                or child.owner is not owner
                or child.region is None
                or child.region.key != key
            ):
                raise StructureUnavailable("while-region-owner-unavailable")
            if (
                child.body != frozen_children.get(role)
                or child.region.expected != child.body
                or child.body.gaps
            ):
                raise StructureMismatch("while-original-child-body-differs")
            children.append(child)
        body_scope = children[1]
        if body_scope.result_values is None:
            raise StructureUnavailable("while-body-results-unavailable")
        body_values = _resolve_bindings(
            body_graph,
            {
                value
                for value in body_scope.result_values
                if type(value) is _ValueWitness
            },
        )
        updated = []
        for binding, actual in zip(
            body_scope.result_values,
            body_graph.outputs[1 + nb + nc : 1 + state_count],
            strict=True,
        ):
            if type(binding) is _ValueWitness:
                candidate = body_values[binding]
                adapters.extend(_identity_edge(body_graph, actual, candidate))
            elif type(binding) is LiteralOperand:
                candidate = actual
                producer = candidate.producer()
                if producer is not None and producer.op_type == "Identity":
                    candidate = producer.inputs[0]
                    if candidate is None:
                        raise StructureMismatch("while-literal-carry-edge-differs")
                    adapters.extend(_identity_edge(body_graph, actual, candidate))
            else:
                raise StructureUnavailable("while-body-result-binding-unavailable")
            updated.append(candidate)
        role_inputs = (
            (*outer[nb : nb + nc], *outer[nb + nc : state_count]),
            (*formals[:nb], *formals[nb + nc : state_count]),
            (*formals[nb : nb + nc], *updated),
        )
        role_outputs = (
            (node.inputs[1],),
            tuple(body_graph.outputs[1 + nb + nc : 1 + state_count]),
            (body_graph.outputs[0],),
        )
        for role, child, actual_inputs in zip(
            ("initial-condition", "body", "next-condition"),
            children,
            role_inputs,
            strict=True,
        ):
            ports = child.region.captures
            source_bindings = tuple(
                receipt.inputs[port]
                for port in rule.input_ports(role, len(receipt.inputs))
            )
            if role == "next-condition":
                source_bindings = (*receipt.inputs[:nc], *body_scope.result_values)
            if (
                len(ports) != len(actual_inputs)
                or tuple(port.binding for port in ports) != source_bindings
            ):
                raise StructureMismatch("while-original-input-partition-differs")
            unused = _unused_inputs(child.body)
            for index, (port, actual) in enumerate(
                zip(ports, actual_inputs, strict=True)
            ):
                values = _resolve_bindings(graphs[port.outer.owner], {port.outer})
                if (
                    port.outer != port.local
                    or port.local != child.input_values[index]
                    or values[port.outer] is not actual
                    or port.tensor != child.body.inputs[index]
                    or type(port.unused) is not bool
                    or port.unused != (index in unused)
                    or not matches(port.tensor, actual)
                ):
                    raise StructureMismatch("while-region-input-edge-or-type-differs")
        for initial, formal, result, final in zip(
            outer, formals, body_graph.outputs[1:], node.outputs, strict=True
        ):
            fact_tensor = next(
                (
                    use.tensor
                    for use, value in zip(
                        scope.equations[handle.index].inputs, parent_inputs, strict=True
                    )
                    if value is initial
                ),
                None,
            )
            if fact_tensor is not None and any(
                not matches(fact_tensor, value)
                for value in (initial, formal, result, final)
            ):
                raise StructureMismatch("while-carried-type-or-shape-differs")
        plans[handle] = WhileAudit(
            node,
            tuple(children),
            tuple(parent_inputs),
            tuple(role_inputs),
            tuple(role_outputs),
            tuple(adapters),
        )
    return plans


@dataclass(frozen=True)
class ScanAttachment:
    origin: object
    owner: object
    body_owner: object
    region: RegionKey
    lazy: tuple[tuple[_ValueWitness, _ValueWitness], ...]


def record_scan_attachment(
    ctx: Any, region: RegionKey, body_graph: ir.Graph, call_parameters: Any
) -> None:
    ledger = ctx._source_structure
    handle = ledger.handles[-1] if ledger.handles else None
    if type(handle) is not EquationHandle or handle in ledger.scan_calls:
        raise StructureUnavailable("scan-original-handle-unavailable")
    if type(region) is not RegionKey or region != RegionKey(handle, "scan-body"):
        raise StructureUnavailable("scan-attachment-role-unavailable")
    # Observe the emitted node directly: zero carries/ys need no user result,
    # and zero-trip ys may subsequently be replaced by static empty storage.
    nodes = [
        node
        for node in ctx.builder.graph
        if not node.domain
        and node.op_type == "Loop"
        and "body" in node.attributes
        and node.attributes["body"].type is ir.AttributeType.GRAPH
        and node.attributes["body"].as_graph() is body_graph
    ]
    if len(nodes) != 1:
        raise StructureUnavailable("scan-emitted-loop-unavailable")
    requested = call_parameters.port_bindings
    first_lazy = len(body_graph.inputs) - len(requested)
    ledger.scan_calls[handle] = ScanAttachment(
        nodes[0].meta.setdefault(_ORIGIN, object()),
        ctx.builder.graph.meta[_OWNER],
        body_graph.meta[_OWNER],
        region,
        tuple(
            (_witness(outer), _witness(body_graph.inputs[first_lazy + index]))
            for index, (outer, _) in enumerate(requested)
        ),
    )


@dataclass(frozen=True)
class ScanAudit:
    node: ir.Node
    regions: tuple[Any, ...]
    inputs: tuple[ir.Value, ...]
    region_outputs: tuple[tuple[ir.Value, ...], ...]
    adapters: tuple[ir.Node, ...]
    constants: tuple[ir.Value, ...]


def _scan_storage(graph: ir.Graph, value: ir.Value) -> Any:
    tensor = ActualShapeFacts._storage(value)
    producer = value.producer()
    if tensor is None:
        raise StructureMismatch("scan-required-literal-storage-differs")
    if (
        value.graph is not graph
        or (
            producer is not None
            and (producer.graph is not graph or producer not in set(graph))
        )
        or (
            producer is None
            and not any(item is value for item in graph.initializers.values())
        )
    ):
        raise StructureUnavailable("scan-literal-owner-unavailable")
    # Tensor storage supplies dimensions. Loop postprocessing may erase these
    # dimensions from ValueInfo; ActualShapeFacts independently checks every
    # retained concrete dimension and rejects conflicting or named metadata.
    if (
        value.dtype is not tensor.dtype
        or value.shape is None
        or len(value.shape) != len(tensor.shape)
    ):
        raise StructureMismatch("scan-literal-type-or-shape-differs")
    return tensor


def plan_scan_calls(
    ledger: Any, scopes: Any, graphs: Any, parents: Any, matches: Any
) -> dict[EquationHandle, ScanAudit]:
    wanted = {
        EquationHandle(scope.invocation, index)
        for scope in scopes.values()
        for index, equation in enumerate(scope.equations)
        if type(equation.rule) is ScanRule
    }
    if wanted != set(ledger.scan_calls):
        raise StructureUnavailable("scan-call-coverage-incomplete")
    plans = {}
    for handle, attachment in ledger.scan_calls.items():
        if type(attachment) is not ScanAttachment:
            raise StructureUnavailable("scan-attachment-unavailable")
        scope = scopes[handle.invocation]
        equation = scope.equations[handle.index]
        rule = equation.rule
        receipt = scope.receipts.get(handle.index)
        if receipt is None:
            raise StructureUnavailable("scan-source-receipt-unavailable")
        graph = graphs[scope.owner]
        nodes = [node for node in graph if node.meta.get(_ORIGIN) is attachment.origin]
        if len(nodes) != 1:
            raise StructureUnavailable("scan-final-node-unavailable")
        node = nodes[0]
        body = graphs.get(attachment.body_owner)
        original_count = rule.constants + rule.carries + rule.sequences
        count = original_count + len(attachment.lazy)
        if (
            attachment.owner is not scope.owner
            or node.domain
            or node.overload
            or node.op_type != "Loop"
            or set(node.attributes) != {"body"}
            or node.attributes["body"].type is not ir.AttributeType.GRAPH
            or node.attributes["body"].as_graph() is not body
            or len(node.inputs) != 2 + count
            or len(node.outputs) != count + rule.results
            or any(value is None for value in node.inputs)
            or body is None
            or len(body.inputs) != 2 + count
            or len(body.outputs) != 1 + count + rule.results
        ):
            raise StructureMismatch("scan-loop-protocol-differs")
        link = parents.get(attachment.body_owner)
        if link is None or link[0] is not graph or link[1] is not node:
            raise StructureMismatch("scan-body-ancestry-differs")
        for value, dtype in (
            (node.inputs[0], ir.DataType.INT64),
            (node.inputs[1], ir.DataType.BOOL),
            (body.inputs[0], ir.DataType.INT64),
            (body.inputs[1], ir.DataType.BOOL),
            (body.outputs[0], ir.DataType.BOOL),
        ):
            if (
                value.dtype is not dtype
                or value.shape is None
                or tuple(value.shape) != ()
            ):
                raise StructureMismatch("scan-header-type-differs")
        constants = [node.inputs[0], node.inputs[1]]
        trip = _scan_storage(graph, node.inputs[0]).numpy()
        condition = _scan_storage(graph, node.inputs[1]).numpy()
        if (
            trip.dtype != np.dtype(np.int64)
            or trip.shape != ()
            or int(trip.item()) != rule.length
        ):
            raise StructureMismatch("scan-original-length-differs")
        if (
            condition.dtype != np.dtype(np.bool_)
            or condition.shape != ()
            or condition.tobytes() != np.asarray(True, np.bool_).tobytes()
        ):
            raise StructureMismatch("scan-initial-condition-differs")
        outer, formals = tuple(node.inputs[2:]), tuple(body.inputs[2:])
        adapters = list(_identity_edge(body, body.outputs[0], body.inputs[1]))
        nc, nk, _nx = rule.constants, rule.carries, rule.sequences
        for index in (*range(nc), *range(nc + nk, count)):
            adapters.extend(
                _identity_edge(body, body.outputs[1 + index], formals[index])
            )
        for index, (outer_witness, local_witness) in enumerate(attachment.lazy):
            outer_values = _resolve_bindings(graph, {outer_witness})
            local_values = _resolve_bindings(body, {local_witness})
            if (
                outer_values[outer_witness] is not outer[original_count + index]
                or local_values[local_witness] is not formals[original_count + index]
            ):
                raise StructureMismatch("scan-lazy-parameter-order-differs")
            for value in (
                outer[original_count + index],
                formals[original_count + index],
                body.outputs[1 + original_count + index],
                node.outputs[original_count + index],
            ):
                if (
                    value.dtype is not ir.DataType.BOOL
                    or value.shape is None
                    or tuple(value.shape) != ()
                ):
                    raise StructureMismatch("scan-lazy-parameter-type-differs")
        key = attachment.region
        if key != RegionKey(handle, "scan-body") or key not in ledger.regions:
            raise StructureUnavailable("scan-region-role-unavailable")
        child = scopes.get(ledger.regions[key])
        if (
            child is None
            or child.owner is not attachment.body_owner
            or child.region is None
            or child.region.key != key
        ):
            raise StructureUnavailable("scan-region-owner-unavailable")
        original_body = dict(scope.body.equations[handle.index].children).get(
            "scan-body"
        )
        if (
            child.body != original_body
            or child.region.expected != child.body
            or child.body.gaps
        ):
            raise StructureMismatch("scan-original-child-body-differs")
        if child.result_values is None:
            raise StructureUnavailable("scan-body-results-unavailable")
        ports = child.region.captures
        if (
            len(ports) != original_count
            or tuple(port.binding for port in ports) != receipt.inputs
        ):
            raise StructureMismatch("scan-original-input-partition-differs")
        unused = _unused_inputs(child.body)
        for index, port in enumerate(ports):
            if (
                port.outer != port.local
                or port.local != child.input_values[index]
                or port.tensor != child.body.inputs[index]
                or type(port.unused) is not bool
                or port.unused != (index in unused)
            ):
                raise StructureMismatch("scan-original-input-port-differs")
            # Ordinary DCE can remove an unused per-step Gather while the
            # full sequence remains mandatory explicit Loop state. Only the
            # independently frozen body's use-def proof permits this absence;
            # retained markers still require unique definitions and exact edges.
            optional = (
                frozenset({port.local})
                if index >= nc + nk and index in unused
                else frozenset()
            )
            actual = _resolve_bindings(body, {port.local}, optional).get(port.local)
            if actual is None:
                continue
            if index < nc + nk:
                if actual is not formals[index]:
                    raise StructureMismatch("scan-body-input-edge-differs")
            else:
                gather = actual.producer()
                if (
                    gather is None
                    or gather.graph is not body
                    or gather not in set(cast(ir.Graph, body))
                    or gather.domain
                    or gather.overload
                    or gather.op_type != "Gather"
                    or len(gather.inputs) != 2
                    or gather.inputs[0] is not formals[index]
                    or gather.inputs[1] is not cast(ir.Graph, body).inputs[0]
                    or len(gather.outputs) != 1
                    or gather.outputs[0] is not actual
                    or set(gather.attributes) != {"axis"}
                    or gather.attributes["axis"].type is not ir.AttributeType.INT
                    or gather.attributes["axis"].as_int() != 0
                ):
                    raise StructureMismatch("scan-sequence-gather-edge-or-axis-differs")
                adapters.append(gather)
            if not matches(port.tensor, actual):
                raise StructureMismatch("scan-body-input-type-or-shape-differs")
        for index, use in enumerate(equation.inputs):
            for actual in (
                outer[index],
                formals[index],
                body.outputs[1 + index],
                node.outputs[index],
            ):
                if not matches(use.tensor, actual):
                    raise StructureMismatch("scan-carried-type-or-shape-differs")
        actual_outputs = (
            *body.outputs[1 + nc : 1 + nc + nk],
            *body.outputs[1 + count :],
        )
        projected = (*node.outputs[nc : nc + nk], *node.outputs[count:])
        resolved = _resolve_bindings(graph, set(receipt.outputs))
        for index, (binding, stacked) in enumerate(
            zip(receipt.outputs, projected, strict=True)
        ):
            actual = resolved[binding]
            if index >= nk and rule.length == 0:
                tensor = _scan_storage(graph, actual)
                storage = tensor.numpy()
                if (
                    not storage.shape
                    or storage.shape[0] != 0
                    or storage.size != 0
                    or storage.nbytes != 0
                    or storage.tobytes() != b""
                    or not matches(equation.outputs[index].tensor, actual)
                    or not matches(equation.outputs[index].tensor, stacked)
                ):
                    raise StructureMismatch("scan-zero-trip-empty-storage-differs")
                constants.append(actual)
            elif actual is not stacked:
                raise StructureMismatch("scan-parent-result-order-differs")
        for value in constants:
            producer = value.producer()
            if producer is not None:
                adapters.append(producer)
        plans[handle] = ScanAudit(
            node,
            (child,),
            outer[:original_count],
            (tuple(actual_outputs),),
            tuple(adapters),
            tuple(value for value in constants if value.is_initializer()),
        )
    return plans
