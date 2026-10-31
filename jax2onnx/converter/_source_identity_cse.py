# jax2onnx/converter/_source_identity_cse.py

"""Finite, producer-bound Identity substitutions from one unchanged CSE pass.

Only the transient snapshots retain IR objects. Registered receipts preserve the
old-to-canonical value relation and ordered output bridges separately. They do
not authorize arithmetic CSE, source admission, or additional artifact coverage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import struct
from typing import Any, TYPE_CHECKING
from weakref import WeakSet

import onnx_ir as ir

from . import _source_cse_output_aliases as output_aliases

if TYPE_CHECKING:
    from .source_structure import _ValueWitness as Witness

_RECEIPTS: str = "jax2onnx.source.structure.identity_cse"
_GAPS: str = "jax2onnx.source.structure.identity_cse_gaps"
Resolver = Callable[
    [ir.Graph, set["Witness"], frozenset["Witness"]], dict["Witness", ir.Value]
]


@dataclass(frozen=True)
class _Schema:
    dtype: ir.DataType
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _NodeSchema:
    domain: str
    operation: str
    overload: str
    inputs: int
    outputs: int
    attributes: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class _Port:
    owner: object
    origin: object
    port: int
    schema: _NodeSchema


@dataclass(frozen=True)
class _Row:
    value: Witness
    input: Witness
    normalized_input: Witness
    canonical: Witness
    schema: _Schema
    ordinal: int


@dataclass(frozen=True)
class _Output:
    slot: int
    original: Witness
    canonical: Witness
    actual: Witness
    schema: _Schema
    bridge: bool
    slots: tuple[int, ...]
    name: str | None
    normalized: bool


@dataclass(frozen=True, eq=False)
class IdentityCSE:
    owner: object
    rows: tuple[_Row, ...]
    uses: tuple[tuple[Witness, tuple[_Port, ...]], ...]
    outputs: tuple[_Output, ...]
    output_count: int
    orders: tuple[tuple[object, tuple[object, ...]], ...]
    output_classes: tuple[tuple[int, ...], ...]
    output_names: tuple[str | None, ...]
    output_schemas: tuple[_Schema, ...]
    normalizations: tuple[output_aliases.Normalization, ...] = ()

    def __deepcopy__(self, memo: Any) -> IdentityCSE:
        return self


_REGISTERED: WeakSet[IdentityCSE] = WeakSet()


@dataclass
class _TransientRow:
    node: ir.Node
    value: ir.Value
    input: ir.Value
    normalized_input: ir.Value
    canonical: ir.Value
    schema: _Schema
    ordinal: int


@dataclass
class _Before:
    graph: ir.Graph
    rows: tuple[_TransientRow, ...]
    records: tuple[_Row, ...]
    uses: tuple[tuple[ir.Value, ir.Node, int, _Port], ...]
    outputs: tuple[ir.Value, ...]
    nodes: frozenset[ir.Node]
    originals: tuple[Witness, ...]
    output_classes: tuple[tuple[int, ...], ...]
    output_names: tuple[str | None, ...]
    output_schemas: tuple[_Schema, ...]
    normalizations: tuple[output_aliases.Normalization, ...] = ()


def _graphs(root: ir.Graph) -> tuple[ir.Graph, ...]:
    result = []
    pending = [root]
    while pending:
        graph = pending.pop()
        if graph in result:
            from .source_structure import StructureUnavailable

            raise StructureUnavailable("identity-cse-graph-cycle-unavailable")
        result.append(graph)
        for node in graph:
            for attribute in node.attributes.values():
                if attribute.type is ir.AttributeType.GRAPH:
                    pending.append(attribute.as_graph())
                elif attribute.type is ir.AttributeType.GRAPHS:
                    pending.extend(attribute.as_graphs())
    return tuple(result)


def _schema(value: ir.Value) -> _Schema:
    from .source_representation import normalized_shape_dimensions
    from .source_structure import StructureUnavailable

    if type(value.dtype) is not ir.DataType or value.shape is None:
        raise StructureUnavailable("identity-cse-type-unavailable")
    shape = normalized_shape_dimensions(tuple(value.shape))
    if any(type(dim) is not int for dim in shape):
        raise StructureUnavailable("identity-cse-concrete-shape-unavailable")
    return _Schema(value.dtype, tuple(int(dim) for dim in shape))


def _identity(node: ir.Node, graph: ir.Graph) -> tuple[ir.Value, ir.Value, _Schema]:
    from .source_structure import StructureMismatch

    if (
        node.graph is not graph
        or node not in set(graph)
        or node.domain
        or node.overload
        or node.op_type != "Identity"
        or node.attributes
        or len(node.inputs) != 1
        or node.inputs[0] is None
        or len(node.outputs) != 1
        or node.outputs[0].producer() is not node
        or node.outputs[0].index() != 0
        or _schema(node.inputs[0]) != _schema(node.outputs[0])
    ):
        raise StructureMismatch("identity-cse-identity-schema-differs")
    return node.inputs[0], node.outputs[0], _schema(node.outputs[0])


def _origin(node: ir.Node) -> object:
    from .source_structure import StructureUnavailable, _ORIGIN

    origin = node.meta.setdefault(_ORIGIN, object())
    if type(origin) is not object:
        raise StructureUnavailable("identity-cse-producer-origin-unavailable")
    return origin


def _node_schema(node: ir.Node) -> _NodeSchema:
    from .source_structure import StructureUnavailable, _OWNER

    if any(
        type(item) is not str for item in (node.domain, node.op_type, node.overload)
    ):
        raise StructureUnavailable("identity-cse-operation-identifier-unavailable")
    attributes: list[tuple[str, object]] = []
    for name, attribute in sorted(node.attributes.items()):
        kind = attribute.type
        if kind is ir.AttributeType.INT:
            value: object = ("int", attribute.as_int())
        elif kind is ir.AttributeType.FLOAT:
            value = ("float", struct.pack(">d", attribute.as_float()))
        elif kind is ir.AttributeType.STRING:
            value = ("string", attribute.as_string())
        elif kind is ir.AttributeType.INTS:
            value = ("ints", tuple(attribute.as_ints()))
        elif kind is ir.AttributeType.FLOATS:
            value = (
                "floats",
                tuple(struct.pack(">d", item) for item in attribute.as_floats()),
            )
        elif kind is ir.AttributeType.STRINGS:
            value = ("strings", tuple(attribute.as_strings()))
        elif kind is ir.AttributeType.GRAPH:
            owner = attribute.as_graph().meta.get(_OWNER)
            if type(owner) is not object:
                raise StructureUnavailable("identity-cse-attribute-owner-unavailable")
            value = ("graph", owner)
        elif kind is ir.AttributeType.GRAPHS:
            owners = tuple(child.meta.get(_OWNER) for child in attribute.as_graphs())
            if any(type(owner) is not object for owner in owners):
                raise StructureUnavailable("identity-cse-attribute-owner-unavailable")
            value = ("graphs", owners)
        else:
            # Tensor-valued consumer attributes need their own finite boundary.
            # No tensor materialization is performed by this Identity observer.
            raise StructureUnavailable("identity-cse-consumer-attribute-unavailable")
        if type(name) is not str:
            raise StructureUnavailable("identity-cse-attribute-name-unavailable")
        attributes.append((name, value))
    return _NodeSchema(
        node.domain,
        node.op_type,
        node.overload,
        len(node.inputs),
        len(node.outputs),
        tuple(attributes),
    )


def _visible(value: ir.Value, node: ir.Node) -> bool:
    from .source_structure import _OWNER, _visible_before

    root = value.graph
    if root is None or node.graph is None:
        return False
    parents = {}
    for graph in _graphs(root):
        for candidate in graph:
            for attribute in candidate.attributes.values():
                if attribute.type is ir.AttributeType.GRAPH:
                    parents[attribute.as_graph().meta.get(_OWNER)] = (graph, candidate)
                elif attribute.type is ir.AttributeType.GRAPHS:
                    for child in attribute.as_graphs():
                        parents[child.meta.get(_OWNER)] = (graph, candidate)
    return _visible_before(value, node, parents)


def _current(value: ir.Value) -> Witness:
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
        raise StructureUnavailable("identity-cse-current-owner-unavailable")
    witness = _witness(value)
    markers = value.meta.get(_VALUES, ())
    if type(markers) is not tuple or any(
        type(item) is not _ValueWitness for item in markers
    ):
        raise StructureUnavailable("identity-cse-current-markers-unavailable")
    wanted = {item for item in markers if item.owner is graph.meta[_OWNER]}
    found = _resolve_direct_bindings(graph, wanted)
    if not wanted or any(found[item] is not value for item in wanted):
        raise StructureUnavailable("identity-cse-original-definition-ambiguous")
    return witness


def _capture(graph: ir.Graph) -> tuple[_Before, ...]:
    """Compute only ordered, acyclic Identity congruence from actual pre-values."""
    from .source_structure import _OWNER

    if type(graph.meta.get(_OWNER)) is not object:
        return ()
    nodes = tuple(graph)
    rows = []
    canonical: dict[ir.Value, ir.Value] = {}
    classes: dict[tuple[ir.Value, _Schema], ir.Value] = {}
    for ordinal, node in enumerate(nodes):
        if node.op_type != "Identity":
            continue
        try:
            input_value, output, schema = _identity(node, graph)
            if not _visible(input_value, node):
                continue
        except (ValueError, TypeError):
            continue
        normalized = canonical.get(input_value, input_value)
        representative = classes.setdefault((normalized, schema), output)
        canonical[output] = representative
        rows.append(
            _TransientRow(
                node, output, input_value, normalized, representative, schema, ordinal
            )
        )
    by_value = {row.value: row for row in rows}
    remaining = set(by_value)
    components = []
    while remaining:
        seed = next(iter(remaining))
        component = {seed}
        pending = [seed]
        while pending:
            value = pending.pop()
            row = by_value[value]
            linked = {
                other.value
                for other in rows
                if other.input is value
                or row.input is other.value
                or other.canonical is row.canonical
            }
            for other in linked - component:
                component.add(other)
                pending.append(other)
        remaining -= component
        selected = tuple(row for row in rows if row.value in component)
        if not any(row.canonical is not row.value for row in selected):
            continue
        records = tuple(
            _Row(
                _current(row.value),
                _current(row.input),
                _current(row.normalized_input),
                _current(row.canonical),
                row.schema,
                row.ordinal,
            )
            for row in selected
        )
        uses = []
        descendants = set(_graphs(graph))
        for row in selected:
            for use in row.value.uses():
                consumer = use.node
                if consumer.graph not in descendants or consumer not in set(
                    consumer.graph
                ):
                    from .source_structure import StructureUnavailable

                    raise StructureUnavailable(
                        "identity-cse-original-use-owner-unavailable"
                    )
                _origin(consumer)
                uses.append((row.value, consumer, use.idx, _port(consumer, use.idx)))
        components.append(
            _Before(
                graph,
                selected,
                records,
                tuple(uses),
                tuple(graph.outputs),
                frozenset(nodes),
                tuple(_current(value) for value in graph.outputs),
                *output_aliases.output_contract(graph),
            )
        )
    return tuple(components)


def _port(node: ir.Node, port: int) -> _Port:
    from .source_structure import StructureUnavailable, _ORIGIN, _OWNER

    graph = node.graph
    origin = node.meta.get(_ORIGIN)
    if (
        graph is None
        or type(graph.meta.get(_OWNER)) is not object
        or type(origin) is not object
    ):
        raise StructureUnavailable("identity-cse-use-owner-unavailable")
    return _Port(graph.meta[_OWNER], origin, port, _node_schema(node))


def _finish(before: _Before) -> None:
    from .source_structure import StructureMismatch, _ORIGIN, _OWNER, _witness

    graph = before.graph
    if graph.meta.get(_OWNER) is not before.records[0].value.owner:
        raise StructureMismatch("identity-cse-post-owner-differs")
    if not any(row.node.graph is not graph for row in before.rows):
        # An unchanged/no-op pass creates no substitution authority.
        return
    if len(graph.outputs) != len(before.outputs):
        raise StructureMismatch("identity-cse-output-count-differs")
    output_aliases.validate_outputs(
        graph, before.output_classes, before.output_names, before.output_schemas
    )
    aliases = {row.value: row.canonical for row in before.rows}
    by_node = {row.node: row for row in before.rows}
    keepers = {row.canonical for row in before.rows}
    for row, record in zip(before.rows, before.records, strict=True):
        if row.value is row.canonical:
            if (
                row.node.meta.get(_ORIGIN) is not record.value.origin
                or _current(row.normalized_input) != record.normalized_input
            ):
                raise StructureMismatch("identity-cse-canonical-authority-differs")
            actual_input, output, schema = _identity(row.node, graph)
            if (
                actual_input is not row.normalized_input
                or output is not row.value
                or schema != row.schema
            ):
                raise StructureMismatch("identity-cse-canonical-edge-differs")
        elif (
            row.node.graph is not None
            or row.node in set(graph)
            or row.value.uses()
            or any(value is row.value for value in graph.outputs)
        ):
            raise StructureMismatch("identity-cse-old-definition-not-removed")
    expected_uses: dict[ir.Value, set[tuple[ir.Node, int]]] = {
        value: set() for value in keepers
    }
    related = {row.node for row in before.rows if row.value is row.canonical}
    for old_value, consumer, port, contract in before.uses:
        if (
            consumer in by_node
            and by_node[consumer].value is not by_node[consumer].canonical
        ):
            continue
        if (
            consumer.graph is None
            or consumer not in set(consumer.graph)
            or _port(consumer, port) != contract
            or port < 0
            or port >= len(consumer.inputs)
            or consumer.inputs[port] is not aliases[old_value]
            or not _visible(aliases[old_value], consumer)
        ):
            raise StructureMismatch("identity-cse-projected-consumer-differs")
        expected_uses[aliases[old_value]].add((consumer, port))
        related.add(consumer)
    outputs = []
    schemas = {row.value: row.schema for row in before.rows}
    for slot, old in enumerate(before.outputs):
        if old not in aliases:
            continue
        canonical = aliases[old]
        actual = graph.outputs[slot]
        bridge = actual is not canonical
        if bridge:
            node = actual.producer()
            if node is None or node in before.nodes:
                raise StructureMismatch("identity-cse-output-bridge-not-fresh")
            bridge_input, bridge_output, schema = _identity(node, graph)
            if (
                bridge_input is not canonical
                or bridge_output is not actual
                or schema != schemas[old]
                or actual.uses()
                or tuple(i for i, value in enumerate(graph.outputs) if value is actual)
                != next(group for group in before.output_classes if slot in group)
                or not _visible(canonical, node)
            ):
                raise StructureMismatch("identity-cse-output-bridge-differs")
            expected_uses[canonical].add((node, 0))
            related.add(node)
        outputs.append(
            _Output(
                slot,
                before.originals[slot],
                _current(canonical),
                _current(actual),
                schemas[old],
                bridge,
                next(group for group in before.output_classes if slot in group),
                before.output_names[slot],
                any(
                    fact.original == before.originals[slot]
                    for fact in before.normalizations
                ),
            )
        )
    for value, expected in expected_uses.items():
        actual_uses = {(use.node, use.idx) for use in value.uses()}
        if actual_uses != expected or len(actual_uses) != len(value.uses()):
            raise StructureMismatch("identity-cse-projected-uses-differ")
    orders = []
    for child in _graphs(graph):
        selected = tuple(_origin(node) for node in child if node in related)
        if selected:
            orders.append((child.meta[_OWNER], selected))
    use_order = {
        node: (graph_index, node_index)
        for graph_index, child in enumerate(_graphs(graph))
        for node_index, node in enumerate(child)
    }
    uses = tuple(
        (
            _witness(value),
            tuple(
                _port(node, port)
                for node, port in sorted(
                    expected_uses[value],
                    key=lambda item: (*use_order[item[0]], item[1]),
                )
            ),
        )
        for value in (row.value for row in before.rows if row.value is row.canonical)
    )
    receipt = IdentityCSE(
        graph.meta[_OWNER],
        before.records,
        uses,
        tuple(outputs),
        len(before.outputs),
        tuple(orders),
        before.output_classes,
        before.output_names,
        before.output_schemas,
        tuple(
            fact
            for fact in before.normalizations
            if any(output.original == fact.original for output in outputs)
        ),
    )
    # No receipt exists until every row, use and ordered output relation passed.
    _REGISTERED.add(receipt)
    graph.meta[_RECEIPTS] = (*receipts(graph), receipt)


def _problem(graph: ir.Graph, stage: str, exc: Exception) -> None:
    from .source_structure import StructureMismatch, StructureUnavailable

    reason = (
        str(exc)
        if isinstance(exc, (StructureMismatch, StructureUnavailable))
        else type(exc).__name__
    )
    previous = graph.meta.get(_GAPS, ())
    if type(previous) is not tuple:
        previous = ()
    graph.meta[_GAPS] = (*previous, stage + ":" + reason)


@contextmanager
def observe_identity_cse(model: ir.Model) -> Iterator[output_aliases.Observation]:
    """Observe the real pass once; preserve its result and exception ownership."""
    from jax2onnx.ir_utils import iter_ir_functions
    from .source_structure import _OWNER

    observation = output_aliases.Observation()
    snapshots: list[_Before] = []
    try:
        roots = [
            model.graph,
            *(function.graph for function in iter_ir_functions(model.functions)),
        ]
        graphs = tuple(graph for root in roots for graph in _graphs(root))
    except Exception as exc:
        _problem(model.graph, "identity-cse-before", exc)
        graphs = ()
    for graph in graphs:
        if type(graph.meta.get(_OWNER)) is not object:
            continue
        try:
            snapshots.extend(_capture(graph))
        except Exception as exc:
            _problem(graph, "identity-cse-before", exc)
    for snapshot in snapshots:
        observation.originals[snapshot.graph] = snapshot.originals
    yield observation
    for snapshot in snapshots:
        try:
            snapshot.normalizations = observation.facts(snapshot.graph)
            _finish(snapshot)
        except Exception as exc:
            _problem(snapshot.graph, "identity-cse-after", exc)


def receipts(graph: ir.Graph) -> tuple[IdentityCSE, ...]:
    from .source_structure import StructureUnavailable, _OWNER

    result = graph.meta.get(_RECEIPTS, ())
    if (
        type(result) is not tuple
        or any(
            type(item) is not IdentityCSE or item not in _REGISTERED for item in result
        )
        or len(set(result)) != len(result)
        or any(item.owner is not graph.meta.get(_OWNER) for item in result)
    ):
        raise StructureUnavailable("identity-cse-registered-receipts-unavailable")
    before = [
        row.value for item in result for row in item.rows if row.value != row.canonical
    ]
    if len(set(before)) != len(before):
        raise StructureUnavailable("identity-cse-alias-receipt-conflict")
    # One pass certifies disjoint connected components, not receipt chains.
    # Check the whole catalog before any retained-definition resolver callback;
    # genuine receipts from opposite-order clones must not create recursion.
    aliases = {old for item in result for old in _congruence(item)}
    dependencies = {
        witness
        for item in result
        for row in item.rows
        for witness in (row.canonical, row.normalized_input)
    }
    dependencies.update(
        witness
        for item in result
        for output in item.outputs
        for witness in (output.canonical, output.actual)
    )
    if aliases & dependencies:
        raise StructureUnavailable("identity-cse-cross-component-dependency")
    return result


def _congruence(receipt: IdentityCSE) -> dict[Witness, Witness]:
    from .source_structure import StructureUnavailable, _ValueWitness

    classes: dict[tuple[Witness, _Schema], Witness] = {}
    normalized: dict[Witness, Witness] = {}
    if any(
        type(row) is not _Row
        or type(row.schema) is not _Schema
        or type(row.ordinal) is not int
        or row.ordinal < 0
        for row in receipt.rows
    ):
        raise StructureUnavailable("identity-cse-row-schema-unavailable")
    ordinals = [row.ordinal for row in receipt.rows]
    if ordinals != sorted(set(ordinals)) or not receipt.rows:
        raise StructureUnavailable("identity-cse-preorder-unavailable")
    for row in receipt.rows:
        if (
            any(
                type(item) is not _ValueWitness
                for item in (row.value, row.input, row.normalized_input, row.canonical)
            )
            or row.value.owner is not receipt.owner
            or row.canonical.owner is not receipt.owner
            or row.value.role != "producer"
            or row.value.port != 0
            or row.canonical.role != "producer"
            or row.canonical.port != 0
            or row.value in normalized
        ):
            raise StructureUnavailable("identity-cse-row-authority-unavailable")
        input_value = normalized.get(row.input, row.input)
        canonical = classes.setdefault((input_value, row.schema), row.value)
        if input_value != row.normalized_input or canonical != row.canonical:
            raise StructureUnavailable("identity-cse-precongruence-differs")
        normalized[row.value] = canonical
    aliases = {
        value: canonical
        for value, canonical in normalized.items()
        if value != canonical
    }
    if set(aliases) & set(aliases.values()):
        raise StructureUnavailable("identity-cse-alias-cycle-unavailable")
    return aliases


def _nodes(graph: ir.Graph, owner: object, origin: object) -> list[ir.Node]:
    from .source_structure import _ORIGIN, _OWNER

    return [
        node
        for child in _graphs(graph)
        if child.meta.get(_OWNER) is owner
        for node in child
        if node.meta.get(_ORIGIN) is origin
    ]


def _resolve_input(
    graph: ir.Graph, witness: Witness, actual: ir.Value, resolve: Resolver
) -> ir.Value:
    from .source_structure import StructureUnavailable, _OWNER

    owner_graph = actual.graph
    if owner_graph is None or owner_graph.meta.get(_OWNER) is not witness.owner:
        raise StructureUnavailable("identity-cse-final-input-owner-unavailable")
    value = resolve(owner_graph, {witness}, frozenset()).get(witness)
    if value is not actual:
        raise StructureUnavailable("identity-cse-final-input-edge-unavailable")
    return value


def _validate(
    graph: ir.Graph, receipt: IdentityCSE, resolve: Resolver
) -> dict[Witness, ir.Value]:
    from .source_structure import (
        StructureMismatch,
        StructureUnavailable,
        _ORIGIN,
        _OWNER,
        _VALUES,
    )

    output_aliases.validate_outputs(
        graph, receipt.output_classes, receipt.output_names, receipt.output_schemas
    )
    aliases = _congruence(receipt)
    canonical_rows = [row for row in receipt.rows if row.value == row.canonical]
    canonical_wanted = {row.value for row in canonical_rows}
    bridge_wanted = {output.actual for output in receipt.outputs if output.bridge}
    if (canonical_wanted | bridge_wanted) & set(aliases):
        raise StructureUnavailable("identity-cse-canonical-alias-cycle")
    resolved = resolve(graph, canonical_wanted | bridge_wanted, frozenset())
    for value in (
        *graph.inputs,
        *graph.initializers.values(),
        *(value for node in graph for value in node.outputs),
    ):
        if any(marker in aliases for marker in value.meta.get(_VALUES, ())):
            raise StructureUnavailable("identity-cse-removed-marker-reappeared")
    for row in canonical_rows:
        actual = resolved[row.value]
        node = actual.producer()
        if node is None:
            raise StructureMismatch("identity-cse-final-canonical-producer-missing")
        input_value, output, schema = _identity(node, graph)
        if output is not actual or schema != row.schema:
            raise StructureMismatch("identity-cse-final-canonical-schema-differs")
        _resolve_input(graph, row.normalized_input, input_value, resolve)
        if not _visible(input_value, node):
            raise StructureMismatch("identity-cse-final-input-not-visible")
    if len(graph.outputs) != receipt.output_count:
        raise StructureMismatch("identity-cse-final-output-count-differs")
    expected_uses = dict(receipt.uses)
    if set(expected_uses) != canonical_wanted or len(expected_uses) != len(
        receipt.uses
    ):
        raise StructureUnavailable(
            "identity-cse-final-canonical-use-coverage-incomplete"
        )
    for witness, ports in receipt.uses:
        value = resolved[witness]
        actual_ports = tuple(_port(use.node, use.idx) for use in value.uses())
        if len(actual_ports) != len(ports) or set(actual_ports) != set(ports):
            raise StructureMismatch("identity-cse-final-uses-differ")
        for port in ports:
            nodes = _nodes(graph, port.owner, port.origin)
            if (
                len(nodes) != 1
                or port.port < 0
                or port.port >= len(nodes[0].inputs)
                or nodes[0].inputs[port.port] is not value
                or _node_schema(nodes[0]) != port.schema
                or not _visible(value, nodes[0])
            ):
                raise StructureMismatch("identity-cse-final-consumer-differs")
    slots = []
    for output_record in receipt.outputs:
        if (
            type(output_record.slot) is not int
            or output_record.slot < 0
            or output_record.slot >= len(graph.outputs)
            or output_record.slot in slots
            or output_record.canonical not in canonical_wanted
            or type(output_record.original) is not type(receipt.rows[0].value)
            or output_record.original not in {row.value for row in receipt.rows}
            or output_record.canonical
            != next(
                row.canonical
                for row in receipt.rows
                if row.value == output_record.original
            )
            or type(output_record.normalized) is not bool
            or output_record.slots
            != next(
                group for group in receipt.output_classes if output_record.slot in group
            )
        ):
            raise StructureUnavailable("identity-cse-final-output-slots-unavailable")
        slots.append(output_record.slot)
        actual = resolved[output_record.actual]
        canonical = resolved[output_record.canonical]
        if (
            graph.outputs[output_record.slot] is not actual
            or _schema(actual) != output_record.schema
            or actual.name != output_record.name
            or tuple(i for i, value in enumerate(graph.outputs) if value is actual)
            != output_record.slots
        ):
            raise StructureMismatch("identity-cse-final-output-slot-differs")
        if output_record.bridge:
            node = actual.producer()
            if node is None:
                raise StructureMismatch("identity-cse-final-bridge-missing")
            input_value, value, schema = _identity(node, graph)
            if (
                input_value is not canonical
                or value is not actual
                or schema != output_record.schema
                or actual.uses()
                or not _visible(canonical, node)
            ):
                raise StructureMismatch("identity-cse-final-bridge-differs")
        elif actual is not canonical:
            raise StructureMismatch("identity-cse-final-direct-output-differs")
    output_aliases.validate_facts(
        graph, receipt.normalizations, receipt.outputs, resolved
    )
    component_values = set(resolved.values())
    if {
        index for index, value in enumerate(graph.outputs) if value in component_values
    } != set(slots):
        raise StructureMismatch("identity-cse-final-component-output-slots-differ")
    for owner, origins in receipt.orders:
        matching = [
            child for child in _graphs(graph) if child.meta.get(_OWNER) is owner
        ]
        if len(matching) != 1 or len(set(origins)) != len(origins):
            raise StructureUnavailable("identity-cse-final-order-owner-unavailable")
        actual_origins = tuple(
            node.meta.get(_ORIGIN)
            for node in matching[0]
            if node.meta.get(_ORIGIN) in origins
        )
        if actual_origins != origins:
            raise StructureMismatch("identity-cse-final-producer-order-differs")
    return {old: resolved[canonical] for old, canonical in aliases.items()}


def validate_identity_cse(graph: ir.Graph, resolve: Resolver) -> None:
    from .source_structure import StructureUnavailable

    problems = graph.meta.get(_GAPS, ())
    if type(problems) is not tuple or any(type(item) is not str for item in problems):
        raise StructureUnavailable("identity-cse-problem-metadata-unavailable")
    if problems:
        raise StructureUnavailable(problems[0])
    for receipt in receipts(graph):
        _validate(graph, receipt, resolve)


def resolve_identity_bindings(
    graph: ir.Graph,
    wanted: set[Witness],
    optional: frozenset[Witness],
    resolve: Resolver,
) -> dict[Witness, ir.Value] | None:
    attached = receipts(graph)
    aliases = {
        row.value
        for item in attached
        for row in item.rows
        if row.value != row.canonical
    }
    redirected = wanted & aliases
    if not redirected:
        return None
    result = resolve(graph, wanted - redirected, optional - redirected)
    for item in attached:
        resolved = _validate(graph, item, resolve)
        result.update(
            (witness, resolved[witness]) for witness in redirected & set(resolved)
        )
    return result
