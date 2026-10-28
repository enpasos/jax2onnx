# jax2onnx/converter/_source_identity_dce.py

"""Observed dead Identity edges, separate from source and CSE obligations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from weakref import WeakSet

import onnx_ir as ir

from . import _source_identity_cse as cse

if TYPE_CHECKING:
    from .source_structure import _ValueWitness as Witness, TensorContract

_RECEIPTS: str = "jax2onnx.source.structure.identity_dce"
_GAPS: str = "jax2onnx.source.structure.identity_dce_gaps"


@dataclass(frozen=True)
class _Row:
    value: Witness
    input: Witness
    anchor: Witness
    schema: cse._Schema
    ordinal: int


@dataclass(frozen=True)
class _Interface:
    origin: object
    schema: cse._NodeSchema
    inputs: tuple[tuple[Witness, cse._Schema] | None, ...]
    outputs: tuple[tuple[Witness, cse._Schema], ...]


@dataclass(frozen=True, eq=False)
class IdentityDCE:
    owner: object
    rows: tuple[_Row, ...]
    anchor: Witness
    schema: cse._Schema
    uses: tuple[cse._Port, ...]
    outputs: tuple[tuple[Witness, cse._Schema], ...]
    interfaces: tuple[_Interface, ...]
    cse_receipts: tuple[cse.IdentityCSE, ...]
    constant_lifts: tuple[Any, ...]

    def __deepcopy__(self, memo: Any) -> IdentityDCE:
        return self


_REGISTERED: WeakSet[IdentityDCE] = WeakSet()


@dataclass
class _TransientRow:
    node: ir.Node
    value: ir.Value
    input: ir.Value
    record: _Row
    uses: tuple[tuple[ir.Node, int], ...]


@dataclass
class _Before:
    graph: ir.Graph
    rows: tuple[_TransientRow, ...]
    anchor: ir.Value
    anchor_uses: tuple[tuple[ir.Node, int], ...]
    projected_uses: tuple[tuple[ir.Node, int], ...]
    inputs: tuple[ir.Value, ...]
    outputs: tuple[ir.Value, ...]
    initializers: tuple[ir.Value, ...]
    nodes: frozenset[ir.Node]
    interfaces: tuple[
        tuple[ir.Node, tuple[ir.Value | None, ...], tuple[ir.Value, ...]], ...
    ]
    receipt: IdentityDCE


def _current(value: ir.Value) -> Witness:
    """Bind this physical role; historical ConstantLift roles are separate."""
    from .source_structure import (
        StructureUnavailable,
        _OWNER,
        _VALUES,
        _ValueWitness,
        _resolve_direct_bindings,
        _witness,
    )

    graph = value.graph
    if graph is None or type(graph.meta.get(_OWNER)) is not object:
        raise StructureUnavailable("identity-dce-current-owner-unavailable")
    markers = value.meta.get(_VALUES, ())
    if type(markers) is not tuple or any(
        type(item) is not _ValueWitness for item in markers
    ):
        raise StructureUnavailable("identity-dce-current-markers-unavailable")
    result = _witness(value)
    if _resolve_direct_bindings(graph, {result}).get(result) is not value:
        raise StructureUnavailable("identity-dce-current-definition-unavailable")
    return result


def _prior(
    graph: ir.Graph,
) -> tuple[
    tuple[cse.IdentityCSE, ...],
    tuple[Any, ...],
    set[Witness],
    set[tuple[object, object]],
]:
    from .source_structure import _constant_lift_receipts

    identities = cse.receipts(graph)
    lifts = _constant_lift_receipts(graph)
    values = {
        witness
        for receipt in identities
        for row in receipt.rows
        for witness in (row.value, row.input, row.normalized_input, row.canonical)
    }
    values.update(
        witness
        for receipt in identities
        for output in receipt.outputs
        for witness in (output.actual, output.canonical)
    )
    values.update(witness for receipt in identities for witness, _ in receipt.uses)
    values.update(
        witness
        for receipt in lifts
        for witness in (receipt.before, receipt.after, *receipt.markers)
    )
    ports = {
        (port.owner, port.origin)
        for receipt in identities
        for _, uses in receipt.uses
        for port in uses
    }
    ports.update(
        (owner, origin)
        for receipt in identities
        for owner, origins in receipt.orders
        for origin in origins
    )
    ports.update(
        (port.owner, port.origin) for receipt in lifts for port in receipt.uses
    )
    return identities, lifts, values, ports


def _interface(node: ir.Node, graph: ir.Graph) -> _Interface:
    from .source_structure import StructureUnavailable, _ORIGIN

    if node.graph is not graph or node not in set(graph):
        raise StructureUnavailable("identity-dce-retained-interface-owner-unavailable")
    cse._origin(node)
    if any(
        value is not None
        and (value.graph is not graph or not cse._visible(value, node))
        for value in node.inputs
    ):
        raise StructureUnavailable("identity-dce-local-interface-unavailable")
    return _Interface(
        node.meta[_ORIGIN],
        cse._node_schema(node),
        tuple(
            None if value is None else (_current(value), cse._schema(value))
            for value in node.inputs
        ),
        tuple((_current(value), cse._schema(value)) for value in node.outputs),
    )


def _uses(value: ir.Value) -> tuple[tuple[ir.Node, int], ...]:
    from .source_structure import StructureUnavailable

    result = tuple((use.node, use.idx) for use in value.uses())
    if len(result) != len(set(result)) or any(
        node.graph is None
        or node not in set(node.graph)
        or type(port) is not int
        or port < 0
        or port >= len(node.inputs)
        or node.inputs[port] is not value
        for node, port in result
    ):
        raise StructureUnavailable("identity-dce-actual-use-port-unavailable")
    return result


def _capture(graph: ir.Graph, graphs: tuple[ir.Graph, ...]) -> tuple[_Before, ...]:
    from .source_structure import _OWNER

    if type(graph.meta.get(_OWNER)) is not object:
        return ()
    nodes = tuple(graph)
    boundaries = {
        value
        for child in graphs
        for value in (*child.inputs, *child.outputs, *child.initializers.values())
    }
    eligible: dict[ir.Value, tuple[ir.Node, ir.Value, cse._Schema, int]] = {}
    for ordinal, node in enumerate(nodes):
        if node.op_type != "Identity":
            continue
        try:
            input_value, output, schema = cse._identity(node, graph)
            if input_value.graph is graph and cse._visible(input_value, node):
                eligible[output] = (node, input_value, schema, ordinal)
        except (ValueError, TypeError):
            continue
    dead: set[ir.Value] = set()
    for output, (node, _, _, _) in reversed(tuple(eligible.items())):
        uses = _uses(output)
        if output in boundaries or len(uses) != len(set(uses)):
            continue
        if all(
            consumer.graph is graph
            and consumer in set(nodes)
            and port == 0
            and len(consumer.outputs) == 1
            and consumer.outputs[0] in dead
            and consumer.inputs[0] is output
            for consumer, port in uses
        ):
            dead.add(output)
    if not dead:
        return ()
    live_nodes: set[ir.Node] = set()
    pending = list(graph.outputs)
    while pending:
        value = pending.pop()
        producer = value.producer()
        if (
            producer is not None
            and producer.graph is graph
            and producer not in live_nodes
        ):
            live_nodes.add(producer)
            pending.extend(value for value in producer.inputs if value is not None)
    roots: dict[ir.Value, ir.Value] = {}
    groups: dict[ir.Value, list[ir.Value]] = {}
    for value, (_, input_value, _, _) in eligible.items():
        if value in dead:
            root = roots.get(input_value, input_value)
            roots[value] = root
            groups.setdefault(root, []).append(value)
    protected_values: set[Witness] = set()
    protected_ports: set[tuple[object, object]] = set()
    for child in graphs:
        _, _, values, ports = _prior(child)
        protected_values.update(values)
        protected_ports.update(ports)
    identities, lifts, _, _ = _prior(graph)
    result = []
    for anchor, members in groups.items():
        removed_nodes = {eligible[value][0] for value in members}
        anchor_uses = _uses(anchor)
        projected = tuple(
            (node, port) for node, port in anchor_uses if node not in removed_nodes
        )
        # An already optional unused capture must not gain a replacement value.
        if anchor not in graph.outputs and not any(
            node in live_nodes for node, _ in projected
        ):
            continue
        records = tuple(
            _Row(
                _current(value),
                _current(eligible[value][1]),
                _current(anchor),
                eligible[value][2],
                eligible[value][3],
            )
            for value in members
        )
        if any(
            row.value in protected_values
            or (row.value.owner, row.value.origin) in protected_ports
            for row in records
        ):
            from .source_structure import StructureUnavailable

            raise StructureUnavailable("identity-dce-prior-receipt-dependency")
        relevant = {node for node, _ in projected}
        anchor_producer = anchor.producer()
        if anchor_producer is not None:
            relevant.add(anchor_producer)
        ordered = tuple(node for node in nodes if node in relevant)
        interfaces = tuple(_interface(node, graph) for node in ordered)
        use_order = {node: index for index, node in enumerate(nodes)}
        retained_ports: tuple[cse._Port, ...] = tuple(
            cse._port(node, port)
            for node, port in sorted(
                projected, key=lambda item: (use_order[item[0]], item[1])
            )
        )
        receipt = IdentityDCE(
            graph.meta[_OWNER],
            records,
            _current(anchor),
            cse._schema(anchor),
            retained_ports,
            tuple((_current(value), cse._schema(value)) for value in graph.outputs),
            interfaces,
            identities,
            lifts,
        )
        rows = []
        for record, value in zip(records, members, strict=True):
            node, input_value, _, _ = eligible[value]
            actual_uses = _uses(value)
            expected_uses = tuple(
                (eligible[other][0], 0)
                for other in members
                if eligible[other][1] is value
            )
            if (
                len(actual_uses) != len(expected_uses)
                or set(actual_uses) != set(expected_uses)
                or value in boundaries
            ):
                from .source_structure import StructureUnavailable

                raise StructureUnavailable(
                    "identity-dce-component-use-closure-unavailable"
                )
            rows.append(_TransientRow(node, value, input_value, record, actual_uses))
        result.append(
            _Before(
                graph,
                tuple(rows),
                anchor,
                anchor_uses,
                projected,
                tuple(graph.inputs),
                tuple(graph.outputs),
                tuple(graph.initializers.values()),
                frozenset(nodes),
                tuple(
                    (node, tuple(node.inputs), tuple(node.outputs)) for node in ordered
                ),
                receipt,
            )
        )
    return tuple(result)


def _relation(receipt: IdentityDCE) -> dict[Witness, Witness]:
    from .source_structure import StructureUnavailable, _ValueWitness

    if (
        type(receipt.owner) is not object
        or type(receipt.anchor) is not _ValueWitness
        or receipt.anchor.owner is not receipt.owner
        or type(receipt.anchor.origin) is not object
        or receipt.anchor.role not in ("input", "initializer", "producer")
        or type(receipt.anchor.port) is not int
        or receipt.anchor.port < 0
        or type(receipt.schema) is not cse._Schema
        or not receipt.rows
    ):
        raise StructureUnavailable("identity-dce-anchor-contract-unavailable")
    result: dict[Witness, Witness] = {}
    last = -1
    for row in receipt.rows:
        if (
            type(row) is not _Row
            or type(row.value) is not _ValueWitness
            or type(row.input) is not _ValueWitness
            or row.value.owner is not receipt.owner
            or row.input.owner is not receipt.owner
            or type(row.value.origin) is not object
            or type(row.input.origin) is not object
            or row.value.role != "producer"
            or type(row.value.port) is not int
            or row.value.port != 0
            or row.input.role not in ("input", "initializer", "producer")
            or type(row.input.port) is not int
            or row.input.port < 0
            or row.value == receipt.anchor
            or row.value in result
            or row.anchor != receipt.anchor
            or row.schema != receipt.schema
            or type(row.ordinal) is not int
            or row.ordinal <= last
            or (row.input != receipt.anchor and row.input not in result)
        ):
            raise StructureUnavailable(
                "identity-dce-ordered-immediate-edges-unavailable"
            )
        result[row.value] = receipt.anchor
        last = row.ordinal
    return result


def receipts(graph: ir.Graph) -> tuple[IdentityDCE, ...]:
    from .source_structure import StructureUnavailable, _OWNER

    attached = graph.meta.get(_RECEIPTS, ())
    if type(attached) is not tuple or any(
        type(item) is not IdentityDCE or item not in _REGISTERED for item in attached
    ):
        raise StructureUnavailable("identity-dce-registered-receipts-unavailable")
    if len(set(attached)) != len(attached) or any(
        item.owner is not graph.meta.get(_OWNER) for item in attached
    ):
        raise StructureUnavailable("identity-dce-receipt-owner-or-replay-unavailable")
    _check_catalog(graph, attached)
    return attached


def _check_catalog(graph: ir.Graph, attached: tuple[IdentityDCE, ...]) -> None:
    from .source_structure import StructureUnavailable

    aliases: set[Witness] = set()
    dependencies: set[Witness] = set()
    for item in attached:
        relation = _relation(item)
        if aliases & set(relation):
            raise StructureUnavailable("identity-dce-alias-domain-conflict")
        aliases.update(relation)
        dependencies.add(item.anchor)
        dependencies.update(witness for witness, _ in item.outputs)
        dependencies.update(
            witness
            for interface in item.interfaces
            for pair in (*interface.inputs, *interface.outputs)
            if pair is not None
            for witness in (pair[0],)
        )
    if attached:
        identities, lifts, protected, ports = _prior(graph)
        cse_aliases = {old for item in identities for old in cse._congruence(item)}
        lift_aliases = {item.before for item in lifts}
        if aliases & (dependencies | protected) or dependencies & (
            cse_aliases | lift_aliases
        ):
            raise StructureUnavailable("identity-dce-cross-receipt-dependency")
        if any((old.owner, old.origin) in ports for old in aliases):
            raise StructureUnavailable("identity-dce-prior-consumer-dependency")


def _finish(before: _Before) -> None:
    from .source_structure import StructureMismatch, _ORIGIN

    graph = before.graph
    current = set(graph)
    missing = [row for row in before.rows if row.node not in current]
    if not missing:
        return
    if len(missing) != len(before.rows) or current - before.nodes:
        raise StructureMismatch("identity-dce-incomplete-component-removal")
    if (
        tuple(graph.inputs) != before.inputs
        or tuple(graph.outputs) != before.outputs
        or tuple(graph.initializers.values()) != before.initializers
    ):
        raise StructureMismatch("identity-dce-graph-boundary-changed")
    removed_anchor_edges = {
        (row.node, 0) for row in before.rows if row.input is before.anchor
    }
    if set(before.anchor_uses) - set(before.projected_uses) != removed_anchor_edges:
        raise StructureMismatch("identity-dce-original-anchor-edge-coverage-differs")
    for row in before.rows:
        expected_uses = {
            (other.node, 0) for other in before.rows if other.input is row.value
        }
        if len(row.uses) != len(expected_uses) or set(row.uses) != expected_uses:
            raise StructureMismatch(
                "identity-dce-original-internal-edge-coverage-differs"
            )
        if (
            row.node.graph is not None
            or row.value.uses()
            or row.value in graph.outputs
            or (row.node.domain, row.node.op_type, row.node.overload)
            != ("", "Identity", "")
            or row.node.attributes
            or tuple(row.node.inputs) != (None,)
            or tuple(row.node.outputs) != (row.value,)
            or row.value.index() != 0
            or row.node.meta.get(_ORIGIN) is not row.record.value.origin
            or row.value.producer() is not row.node
            or cse._schema(row.value) != row.record.schema
        ):
            raise StructureMismatch("identity-dce-old-definition-not-detached")
    for node, inputs, outputs in before.interfaces:
        if (
            node not in current
            or node.graph is not graph
            or tuple(node.inputs) != inputs
            or tuple(node.outputs) != outputs
        ):
            raise StructureMismatch("identity-dce-retained-interface-changed")
    actual_uses = _uses(before.anchor)
    if len(actual_uses) != len(before.projected_uses) or set(actual_uses) != set(
        before.projected_uses
    ):
        raise StructureMismatch("identity-dce-projected-anchor-uses-differ")
    from .source_structure import _resolve_direct_bindings

    existing = receipts(graph)
    _check_catalog(graph, (*existing, before.receipt))
    _validate(graph, before.receipt, _resolve_direct_bindings)
    _REGISTERED.add(before.receipt)
    graph.meta[_RECEIPTS] = (*existing, before.receipt)


def _validate(
    graph: ir.Graph,
    receipt: IdentityDCE,
    resolve: cse.Resolver,
    graphs: tuple[ir.Graph, ...] | None = None,
) -> dict[Witness, ir.Value]:
    from .source_structure import (
        StructureMismatch,
        StructureUnavailable,
        _ORIGIN,
        _OWNER,
        _VALUES,
        _resolve_direct_bindings,
    )

    aliases = _relation(receipt)
    if graph.meta.get(_OWNER) is not receipt.owner:
        raise StructureUnavailable("identity-dce-final-owner-unavailable")
    identities, lifts, _, _ = _prior(graph)
    if identities != receipt.cse_receipts or lifts != receipt.constant_lifts:
        raise StructureUnavailable("identity-dce-prior-receipt-set-differs")
    actual = resolve(graph, {receipt.anchor}, frozenset())[receipt.anchor]
    if (
        _resolve_direct_bindings(graph, {receipt.anchor})[receipt.anchor] is not actual
        or cse._schema(actual) != receipt.schema
    ):
        raise StructureMismatch("identity-dce-final-anchor-differs")
    scan = cse._graphs(graph) if graphs is None else graphs
    old_origins = {old.origin for old in aliases}
    for child in scan:
        for node in child:
            if node.meta.get(_ORIGIN) in old_origins:
                raise StructureUnavailable("identity-dce-old-producer-reappeared")
        for value in (
            *child.inputs,
            *child.outputs,
            *child.initializers.values(),
            *(
                value
                for node in child
                for value in (*node.inputs, *node.outputs)
                if value is not None
            ),
        ):
            if any(marker in aliases for marker in value.meta.get(_VALUES, ())):
                raise StructureUnavailable("identity-dce-old-marker-reappeared")
    ports = tuple(cse._port(node, port) for node, port in _uses(actual))
    if len(ports) != len(receipt.uses) or set(ports) != set(receipt.uses):
        raise StructureMismatch("identity-dce-final-anchor-uses-differ")
    wanted = {witness for witness, _ in receipt.outputs}
    wanted.update(
        witness
        for interface in receipt.interfaces
        for pair in (*interface.inputs, *interface.outputs)
        if pair is not None
        for witness in (pair[0],)
    )
    found = _resolve_direct_bindings(graph, wanted)
    if len(graph.outputs) != len(receipt.outputs) or any(
        value is not found[witness] or cse._schema(value) != schema
        for value, (witness, schema) in zip(graph.outputs, receipt.outputs, strict=True)
    ):
        raise StructureMismatch("identity-dce-final-output-slots-differ")
    origins = tuple(interface.origin for interface in receipt.interfaces)
    if (
        tuple(
            node.meta.get(_ORIGIN)
            for node in graph
            if node.meta.get(_ORIGIN) in origins
        )
        != origins
    ):
        raise StructureMismatch("identity-dce-final-retained-order-differs")
    for interface in receipt.interfaces:
        nodes = [node for node in graph if node.meta.get(_ORIGIN) is interface.origin]
        if len(nodes) != 1 or cse._node_schema(nodes[0]) != interface.schema:
            raise StructureMismatch("identity-dce-final-consumer-schema-differs")
        node = nodes[0]
        inputs = tuple(
            None if pair is None else found[pair[0]] for pair in interface.inputs
        )
        outputs = tuple(found[witness] for witness, _ in interface.outputs)
        if tuple(node.inputs) != inputs or tuple(node.outputs) != outputs:
            raise StructureMismatch("identity-dce-final-retained-ports-differ")
        for input_value, pair in zip(node.inputs, interface.inputs, strict=True):
            if pair is not None and (
                input_value is None
                or cse._schema(input_value) != pair[1]
                or not cse._visible(input_value, node)
            ):
                raise StructureMismatch(
                    "identity-dce-final-input-schema-or-visibility-differs"
                )
        if any(
            cse._schema(value) != pair[1]
            for value, pair in zip(node.outputs, interface.outputs, strict=True)
        ):
            raise StructureMismatch("identity-dce-final-output-schema-differs")
    return {old: actual for old in aliases}


def _problem(graph: ir.Graph, stage: str, exc: Exception) -> None:
    from .source_structure import StructureMismatch, StructureUnavailable

    reason = (
        str(exc)
        if isinstance(exc, (StructureMismatch, StructureUnavailable))
        else type(exc).__name__
    )
    previous = graph.meta.get(_GAPS, ())
    graph.meta[_GAPS] = (
        (*previous, stage + ":" + reason)
        if type(previous) is tuple
        else (stage + ":" + reason,)
    )


@contextmanager
def observe_identity_dce(model: ir.Model) -> Iterator[None]:
    """Observe the existing call; no observation failure prevents its execution."""
    from jax2onnx.ir_utils import iter_ir_functions

    snapshots: list[_Before] = []
    try:
        roots = (
            model.graph,
            *(function.graph for function in iter_ir_functions(model.functions)),
        )
        graphs = tuple(graph for root in roots for graph in cse._graphs(root))
        if len(set(graphs)) != len(graphs):
            raise ValueError("identity-dce-repeated-graph-owner")
        for graph in graphs:
            snapshots.extend(_capture(graph, graphs))
    except Exception as exc:
        _problem(model.graph, "identity-dce-before", exc)
    yield
    for snapshot in snapshots:
        try:
            _finish(snapshot)
        except Exception as exc:
            _problem(snapshot.graph, "identity-dce-after", exc)


def validate_identity_dce(
    graph: ir.Graph, resolve: cse.Resolver, graphs: tuple[ir.Graph, ...] | None = None
) -> None:
    from .source_structure import StructureUnavailable

    problems = graph.meta.get(_GAPS, ())
    if type(problems) is not tuple or any(type(item) is not str for item in problems):
        raise StructureUnavailable("identity-dce-problem-metadata-unavailable")
    if problems:
        raise StructureUnavailable(problems[0])
    for receipt in receipts(graph):
        _validate(graph, receipt, resolve, graphs)


def resolve_dead_identity_bindings(
    graph: ir.Graph,
    wanted: set[Witness],
    optional: frozenset[Witness],
    resolve: cse.Resolver,
) -> dict[Witness, ir.Value] | None:
    attached = receipts(graph)
    aliases = {old for item in attached for old in _relation(item)}
    redirected = wanted & aliases
    if not redirected:
        return None
    result = resolve(graph, wanted - redirected, optional - redirected)
    for receipt in attached:
        if redirected & set(_relation(receipt)):
            found = _validate(graph, receipt, resolve)
            result.update((old, found[old]) for old in redirected & set(found))
    return result


def eliminated_identity_edge(
    graph: ir.Graph,
    input_witness: Witness,
    output_witness: Witness,
    source: ir.Value,
    output: ir.Value,
    tensor: TensorContract,
    resolve: cse.Resolver,
) -> bool:
    from .source_structure import StructureMismatch, _OWNER

    for receipt in receipts(graph):
        for row in receipt.rows:
            if row.value != output_witness:
                continue
            if (
                row.input != input_witness
                or input_witness.owner is not graph.meta.get(_OWNER)
                or output_witness.owner is not graph.meta.get(_OWNER)
            ):
                raise StructureMismatch(
                    "literal-if-deleted-adapter-immediate-edge-differs"
                )
            found = _validate(graph, receipt, resolve)
            if (
                found[output_witness] is not source
                or output is not source
                or not tensor.matches(source)
                or not tensor.matches(output)
            ):
                raise StructureMismatch(
                    "literal-if-deleted-adapter-anchor-or-type-differs"
                )
            return True
    return False
