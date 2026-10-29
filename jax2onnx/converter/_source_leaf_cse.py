# jax2onnx/converter/_source_leaf_cse.py

"""Observed, finite Constant/Sub substitutions from the unchanged CSE pass.

Receipts certify substitution of actual old definitions. They never discharge a
source equation, an unused capture, or physical artifact coverage. Only transient
snapshots retain IR objects; registered receipts survive authentic graph clones.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from math import prod
from typing import Any, TYPE_CHECKING
from weakref import WeakSet

import onnx_ir as ir

from . import _source_identity_cse as identity
from . import _source_cse_output_aliases as output_aliases

if TYPE_CHECKING:
    from .source_structure import _ValueWitness as Witness

_RECEIPTS: str = "jax2onnx.source.structure.leaf_cse"
_GAPS: str = "jax2onnx.source.structure.leaf_cse_gaps"


@dataclass(frozen=True)
class _Storage:
    dtype: str
    shape: tuple[int, ...]
    payload: bytes


@dataclass(frozen=True)
class _Row:
    value: Witness
    operation: str
    inputs: tuple[Witness, ...]
    normalized_inputs: tuple[Witness, ...]
    input_schemas: tuple[identity._Schema, ...]
    canonical: Witness
    schema: identity._Schema
    storage: _Storage | None
    ordinal: int


@dataclass(frozen=True)
class _Interface:
    owner: object
    origin: object
    schema: identity._NodeSchema
    inputs: tuple[tuple[Witness, identity._Schema] | None, ...]
    outputs: tuple[tuple[Witness, identity._Schema], ...]


@dataclass(frozen=True)
class _Output:
    slot: int
    original: Witness
    canonical: Witness
    actual: Witness
    schema: identity._Schema
    bridge: bool
    slots: tuple[int, ...]
    name: str | None
    normalized: bool


@dataclass(frozen=True, eq=False)
class LeafCSE:
    owner: object
    rows: tuple[_Row, ...]
    uses: tuple[tuple[Witness, tuple[identity._Port, ...]], ...]
    outputs: tuple[_Output, ...]
    interfaces: tuple[_Interface, ...]
    orders: tuple[tuple[object, tuple[object, ...]], ...]
    identities: tuple[identity.IdentityCSE, ...]
    output_classes: tuple[tuple[int, ...], ...]
    output_names: tuple[str | None, ...]
    output_schemas: tuple[identity._Schema, ...]
    normalizations: tuple[output_aliases.Normalization, ...] = ()

    def __deepcopy__(self, memo: Any) -> LeafCSE:
        return self


_REGISTERED: WeakSet[LeafCSE] = WeakSet()


@dataclass
class _TransientRow:
    node: ir.Node
    value: ir.Value
    inputs: tuple[ir.Value, ...]
    canonical: ir.Value
    record: _Row


@dataclass
class _Before:
    graph: ir.Graph
    rows: tuple[_TransientRow, ...]
    nodes: tuple[ir.Node, ...]
    node_orders: tuple[tuple[ir.Graph, tuple[ir.Node, ...]], ...]
    interfaces: tuple[
        tuple[ir.Node, tuple[ir.Value | None, ...], tuple[ir.Value, ...]], ...
    ]
    inputs: tuple[ir.Value, ...]
    initializers: tuple[ir.Value, ...]
    outputs: tuple[ir.Value, ...]
    output_schemas: tuple[identity._Schema, ...]
    witnesses: dict[ir.Value, Witness]
    contracts: dict[ir.Node, _Interface]
    identities: tuple[identity.IdentityCSE, ...]
    output_classes: tuple[tuple[int, ...], ...]
    output_names: tuple[str | None, ...]
    normalizations: tuple[output_aliases.Normalization, ...] = ()


def _current(value: ir.Value) -> Witness:
    """Stamp only the current physical definition, never an old alias."""
    from .source_structure import (
        _resolve_direct_bindings,
        _witness,
        StructureUnavailable,
    )

    result = _witness(value)
    graph = value.graph
    if (
        graph is None
        or _resolve_direct_bindings(graph, {result}).get(result) is not value
    ):
        raise StructureUnavailable("leaf-cse-current-definition-unavailable")
    return result


def _storage(value: ir.Value, tensor: ir.TensorProtocol) -> _Storage:
    from .source_structure import StructureMismatch, StructureUnavailable

    schema = identity._schema(value)
    if (
        any(dim < 0 for dim in schema.shape)
        or prod(schema.shape) > 10
        or tensor.dtype is not schema.dtype
        or tuple(tensor.shape) != schema.shape
    ):
        raise StructureUnavailable("leaf-cse-small-typed-storage-unavailable")
    array = tensor.numpy()
    if tuple(array.shape) != schema.shape or array.dtype.hasobject:
        raise StructureMismatch("leaf-cse-backing-storage-differs")
    return _Storage(array.dtype.str, tuple(array.shape), array.tobytes(order="C"))


def _leaf(
    node: ir.Node, graph: ir.Graph
) -> tuple[ir.Value, tuple[ir.Value, ...], identity._Schema, _Storage | None]:
    from .source_structure import StructureUnavailable

    if (
        node.graph is not graph
        or node not in set(graph)
        or any(
            type(item) is not str for item in (node.domain, node.op_type, node.overload)
        )
        or node.domain
        or node.overload
        or len(node.outputs) != 1
        or node.outputs[0].producer() is not node
        or node.outputs[0].index() != 0
    ):
        raise StructureUnavailable("leaf-cse-definition-schema-unavailable")
    value = node.outputs[0]
    schema = identity._schema(value)
    if any(dim < 0 for dim in schema.shape):
        raise StructureUnavailable("leaf-cse-concrete-shape-unavailable")
    if node.op_type == "Constant":
        if (
            node.inputs
            or set(node.attributes) != {"value"}
            or node.attributes["value"].type is not ir.AttributeType.TENSOR
        ):
            raise StructureUnavailable("leaf-cse-constant-schema-unavailable")
        return value, (), schema, _storage(value, node.attributes["value"].as_tensor())
    if (
        node.op_type != "Sub"
        or node.attributes
        or len(node.inputs) != 2
        or any(value is None for value in node.inputs)
        or schema.dtype not in (ir.DataType.FLOAT, ir.DataType.DOUBLE)
    ):
        raise StructureUnavailable("leaf-cse-sub-schema-unavailable")
    inputs = tuple(value for value in node.inputs if value is not None)
    if any(
        identity._schema(value).dtype is not schema.dtype
        or any(dim < 0 for dim in identity._schema(value).shape)
        or not identity._visible(value, node)
        for value in inputs
    ):
        raise StructureUnavailable("leaf-cse-sub-operands-unavailable")
    return value, inputs, schema, None


def _capture(graph: ir.Graph) -> _Before | None:
    from .source_structure import StructureUnavailable, _OWNER

    if type(graph.meta.get(_OWNER)) is not object:
        return None
    nodes = tuple(graph)
    rows = []
    normalized: dict[ir.Value, ir.Value] = {}
    classes: dict[object, ir.Value] = {}
    for ordinal, node in enumerate(nodes):
        if node.op_type not in ("Constant", "Sub"):
            continue
        try:
            value, inputs, schema, storage = _leaf(node, graph)
        except (ValueError, TypeError):
            continue
        operands = tuple(normalized.get(value, value) for value in inputs)
        input_schemas = tuple(identity._schema(value) for value in inputs)
        key = (node.op_type, operands, input_schemas, schema, storage)
        canonical = classes.setdefault(key, value)
        normalized[value] = canonical
        rows.append(
            _TransientRow(
                node,
                value,
                inputs,
                canonical,
                _Row(
                    _current(value),
                    node.op_type,
                    tuple(_current(value) for value in inputs),
                    tuple(_current(value) for value in operands),
                    input_schemas,
                    _current(canonical),
                    schema,
                    storage,
                    ordinal,
                ),
            )
        )
    if not any(row.value is not row.canonical for row in rows):
        return None
    # Keep duplicate classes and their finite leaf prerequisites, rather than
    # claiming unrelated leaves that may later be removed by ordinary DCE.
    selected = {row.value for row in rows if row.value is not row.canonical}
    by_value = {row.value: row for row in rows}
    pending = list(selected)
    while pending:
        row = by_value[pending.pop()]
        for value in (row.canonical, *row.inputs):
            if value in by_value and value not in selected:
                selected.add(value)
                pending.append(value)
    rows = [row for row in rows if row.value in selected]
    descendants = identity._graphs(graph)
    for row in rows:
        uses = tuple((use.node, use.idx) for use in row.value.uses())
        if len(uses) != len(set(uses)) or any(
            node.graph not in descendants
            or node not in set(node.graph)
            or port < 0
            or port >= len(node.inputs)
            or node.inputs[port] is not row.value
            for node, port in uses
        ):
            raise StructureUnavailable("leaf-cse-original-use-closure-unavailable")
    interfaces = tuple(
        (node, tuple(node.inputs), tuple(node.outputs))
        for child in descendants
        for node in child
    )
    values = {
        value
        for _, inputs, outputs in interfaces
        for value in (*inputs, *outputs)
        if value is not None
    }
    values.update((*graph.inputs, *graph.initializers.values(), *graph.outputs))
    witnesses = {value: _current(value) for value in values}
    touched = {row.node for row in rows}
    touched.update(use.node for row in rows for use in row.value.uses())
    contracts = {
        node: _interface(node) for node in touched if node.op_type != "Constant"
    }
    return _Before(
        graph,
        tuple(rows),
        nodes,
        tuple((child, tuple(child)) for child in descendants),
        interfaces,
        tuple(graph.inputs),
        tuple(graph.initializers.values()),
        tuple(graph.outputs),
        tuple(identity._schema(value) for value in graph.outputs),
        witnesses,
        contracts,
        identity.receipts(graph),
        output_aliases.slots(tuple(graph.outputs)),
        tuple(value.name for value in graph.outputs),
    )


def _interface(node: ir.Node) -> _Interface:
    from .source_structure import StructureUnavailable, _OWNER

    graph = node.graph
    if graph is None:
        raise StructureUnavailable("leaf-cse-interface-owner-unavailable")
    return _Interface(
        graph.meta[_OWNER],
        identity._origin(node),
        identity._node_schema(node),
        tuple(
            None if value is None else (_current(value), identity._schema(value))
            for value in node.inputs
        ),
        tuple((_current(value), identity._schema(value)) for value in node.outputs),
    )


def _finish(before: _Before) -> None:
    from .source_structure import StructureMismatch, _ORIGIN, _OWNER

    graph = before.graph
    current = set(graph)
    removed = {row.node for row in before.rows if row.node not in current}
    if not removed:
        return
    if graph.meta.get(_OWNER) is not before.rows[0].record.value.owner:
        raise StructureMismatch("leaf-cse-post-owner-differs")
    if (
        tuple(graph.inputs) != before.inputs
        or tuple(graph.initializers.values()) != before.initializers
    ):
        raise StructureMismatch("leaf-cse-post-boundary-differs")
    if len(graph.outputs) != len(before.outputs):
        raise StructureMismatch("leaf-cse-output-count-differs")
    output_aliases.validate_outputs(
        graph, before.output_classes, before.output_names, before.output_schemas
    )
    identities = identity.receipts(graph)
    if identities[: len(before.identities)] != before.identities:
        raise StructureMismatch("leaf-cse-identity-catalog-replaced")
    new_identities = identities[len(before.identities) :]
    # The inner observer has already authenticated these exact Identity changes.
    identity_aliases = {
        old: canonical
        for receipt in new_identities
        for old, canonical in identity._congruence(receipt).items()
    }
    actual_by_witness = {witness: value for value, witness in before.witnesses.items()}
    aliases = {row.value: row.canonical for row in before.rows}
    for old_witness, canonical_witness in identity_aliases.items():
        if (
            old_witness not in actual_by_witness
            or canonical_witness not in actual_by_witness
        ):
            raise StructureMismatch("leaf-cse-identity-old-definition-unavailable")
        aliases[actual_by_witness[old_witness]] = actual_by_witness[canonical_witness]

    def projected(value: ir.Value) -> ir.Value:
        seen = set()
        while value in aliases and aliases[value] is not value:
            if value in seen:
                raise StructureMismatch("leaf-cse-post-substitution-cycle")
            seen.add(value)
            value = aliases[value]
        return value

    expected_removed = {
        row.node for row in before.rows if row.value is not row.canonical
    }
    identity_removed = {
        node
        for node in before.nodes
        if node.outputs and before.witnesses[node.outputs[0]] in identity_aliases
    }
    if (
        set(before.nodes) - current != expected_removed | identity_removed
        or removed != expected_removed
    ):
        raise StructureMismatch("leaf-cse-unexplained-definition-removal")
    for child, original_nodes in before.node_orders:
        retained = tuple(node for node in original_nodes if node.graph is child)
        if tuple(node for node in child if node in set(original_nodes)) != retained:
            raise StructureMismatch("leaf-cse-post-original-order-differs")
    for row in before.rows:
        record = row.record
        if row.value is not row.canonical:
            if (
                row.node.graph is not None
                or row.value.uses()
                or row.value in graph.outputs
                or row.node.meta.get(_ORIGIN) is not record.value.origin
            ):
                raise StructureMismatch("leaf-cse-old-definition-not-removed")
        else:
            leaf_value, leaf_inputs, leaf_schema, leaf_storage = _leaf(row.node, graph)
            if (
                leaf_value is not row.value
                or leaf_inputs != tuple(projected(value) for value in row.inputs)
                or leaf_schema != record.schema
                or leaf_storage != record.storage
                or _current(leaf_value) != record.value
            ):
                raise StructureMismatch("leaf-cse-canonical-definition-differs")
    # Verify all old retained interfaces, including operands outside the class.
    # The only supported simultaneous removal is an authentic Identity receipt.
    for interface_node, interface_inputs, interface_outputs in before.interfaces:
        if interface_node in expected_removed | identity_removed:
            continue
        if (
            interface_node.graph is None
            or interface_node not in set(interface_node.graph)
            or tuple(interface_node.inputs)
            != tuple(
                None if value is None else projected(value)
                for value in interface_inputs
            )
            or tuple(interface_node.outputs) != interface_outputs
        ):
            raise StructureMismatch("leaf-cse-retained-interface-differs")
        contract = before.contracts.get(interface_node)
        if contract is not None:
            if (
                identity._node_schema(interface_node) != contract.schema
                or interface_node.graph.meta.get(_OWNER) is not contract.owner
                or interface_node.meta.get(_ORIGIN) is not contract.origin
            ):
                raise StructureMismatch("leaf-cse-post-consumer-schema-differs")
            for interface_value, interface_pair in zip(
                (*interface_node.inputs, *interface_node.outputs),
                (*contract.inputs, *contract.outputs),
                strict=True,
            ):
                if interface_pair is not None and (
                    interface_value is None
                    or identity._schema(interface_value) != interface_pair[1]
                ):
                    raise StructureMismatch("leaf-cse-post-consumer-type-differs")
    keepers = {row.value for row in before.rows if row.value is row.canonical}
    expected_uses: dict[ir.Value, set[tuple[ir.Node, int]]] = {
        value: set() for value in keepers
    }
    related = {row.node for row in before.rows if row.value in keepers}
    for use_node, use_inputs, _ in before.interfaces:
        if use_node in expected_removed | identity_removed:
            continue
        for use_port, old_operand in enumerate(use_inputs):
            if old_operand is not None and projected(old_operand) in keepers:
                projected_value = projected(old_operand)
                if not identity._visible(projected_value, use_node):
                    raise StructureMismatch("leaf-cse-post-use-not-visible")
                expected_uses[projected_value].add((use_node, use_port))
                related.add(use_node)
    outputs: list[_Output] = []
    new_bridges = set()
    identity_bridges = {
        output.actual
        for receipt in new_identities
        for output in receipt.outputs
        if output.bridge
    }
    for output_slot, old_output in enumerate(before.outputs):
        original_schema = before.output_schemas[output_slot]
        canonical_output = projected(old_output)
        actual_output = graph.outputs[output_slot]
        bridge = actual_output is not canonical_output
        if bridge:
            bridge_node = actual_output.producer()
            if bridge_node is None or bridge_node in set(before.nodes):
                raise StructureMismatch("leaf-cse-output-bridge-not-fresh")
            bridge_input, bridge_output, bridge_schema = identity._identity(
                bridge_node, graph
            )
            if (
                bridge_input is not canonical_output
                or bridge_output is not actual_output
                or bridge_schema != original_schema
                or actual_output.uses()
                or tuple(
                    i for i, value in enumerate(graph.outputs) if value is actual_output
                )
                != next(
                    group for group in before.output_classes if output_slot in group
                )
                or not identity._visible(canonical_output, bridge_node)
                or (
                    old_output not in aliases
                    and _current(actual_output) not in identity_bridges
                )
            ):
                raise StructureMismatch("leaf-cse-output-bridge-differs")
            new_bridges.add(bridge_node)
            if canonical_output in keepers:
                expected_uses[canonical_output].add((bridge_node, 0))
                related.add(bridge_node)
        if identity._schema(actual_output) != original_schema:
            raise StructureMismatch("leaf-cse-output-schema-differs")
        outputs.append(
            _Output(
                output_slot,
                before.witnesses[old_output],
                _current(canonical_output),
                _current(actual_output),
                identity._schema(actual_output),
                bridge,
                next(group for group in before.output_classes if output_slot in group),
                before.output_names[output_slot],
                any(
                    fact.original == before.witnesses[old_output]
                    for fact in before.normalizations
                ),
            )
        )
    if current - set(before.nodes) != new_bridges:
        raise StructureMismatch("leaf-cse-unexplained-new-definition")
    for canonical_value, expected_ports in expected_uses.items():
        actual_ports = tuple((use.node, use.idx) for use in canonical_value.uses())
        if (
            len(actual_ports) != len(expected_ports)
            or set(actual_ports) != expected_ports
        ):
            raise StructureMismatch("leaf-cse-projected-uses-differ")
    # Freeze complete interfaces only for the touched closure. Constants have
    # their separate typed storage rule because a later authentic lift removes
    # their producer. No old alias is stamped onto any post-pass value.
    interfaces = tuple(
        _interface(node)
        for child in identity._graphs(graph)
        for node in child
        if node in related and node.op_type != "Constant"
    )
    orders = tuple(
        (
            child.meta[_OWNER],
            tuple(identity._origin(node) for node in child if node in related),
        )
        for child in identity._graphs(graph)
        if any(node in related for node in child)
    )
    order = {
        node: (i, j)
        for i, child in enumerate(identity._graphs(graph))
        for j, node in enumerate(child)
    }
    receipt = LeafCSE(
        graph.meta[_OWNER],
        tuple(row.record for row in before.rows),
        tuple(
            (
                _current(value),
                tuple(
                    identity._port(node, port)
                    for node, port in sorted(
                        uses, key=lambda item: (*order[item[0]], item[1])
                    )
                ),
            )
            for value, uses in expected_uses.items()
        ),
        tuple(outputs),
        interfaces,
        orders,
        identities,
        before.output_classes,
        before.output_names,
        before.output_schemas,
        before.normalizations,
    )
    existing = receipts(graph)
    _validate_catalogs(graph, (*existing, receipt))
    _REGISTERED.add(receipt)
    graph.meta[_RECEIPTS] = (*existing, receipt)


def _congruence(receipt: LeafCSE) -> dict[Witness, Witness]:
    from .source_structure import StructureUnavailable, _ValueWitness

    if type(receipt.owner) is not object or not receipt.rows:
        raise StructureUnavailable("leaf-cse-row-owner-unavailable")
    normalized: dict[Witness, Witness] = {}
    classes: dict[object, Witness] = {}
    last = -1
    for row in receipt.rows:
        if (
            type(row) is not _Row
            or type(row.value) is not _ValueWitness
            or type(row.canonical) is not _ValueWitness
            or row.value.owner is not receipt.owner
            or row.canonical.owner is not receipt.owner
            or row.value.role != "producer"
            or row.value.port != 0
            or row.canonical.role != "producer"
            or row.canonical.port != 0
            or row.value in normalized
            or type(row.ordinal) is not int
            or row.ordinal <= last
            or type(row.schema) is not identity._Schema
            or any(
                type(value) is not _ValueWitness
                for value in (*row.inputs, *row.normalized_inputs)
            )
            or any(type(schema) is not identity._Schema for schema in row.input_schemas)
        ):
            raise StructureUnavailable("leaf-cse-row-authority-unavailable")
        if row.operation == "Constant":
            valid = (
                not row.inputs
                and not row.normalized_inputs
                and not row.input_schemas
                and type(row.storage) is _Storage
            )
        else:
            valid = (
                row.operation == "Sub"
                and len(row.inputs)
                == len(row.normalized_inputs)
                == len(row.input_schemas)
                == 2
                and row.storage is None
                and row.schema.dtype in (ir.DataType.FLOAT, ir.DataType.DOUBLE)
            )
        if not valid:
            raise StructureUnavailable("leaf-cse-row-profile-unavailable")
        inputs = tuple(normalized.get(value, value) for value in row.inputs)
        key = (row.operation, inputs, row.input_schemas, row.schema, row.storage)
        canonical = classes.setdefault(key, row.value)
        if inputs != row.normalized_inputs or canonical != row.canonical:
            raise StructureUnavailable("leaf-cse-precongruence-differs")
        normalized[row.value] = canonical
        last = row.ordinal
    aliases = {
        old: canonical for old, canonical in normalized.items() if old != canonical
    }
    if not aliases or set(aliases) & set(aliases.values()):
        raise StructureUnavailable("leaf-cse-substitution-domain-unavailable")
    return aliases


def _dependencies(receipt: LeafCSE) -> set[Witness]:
    return (
        {row.canonical for row in receipt.rows}
        | {
            value
            for row in receipt.rows
            if row.value == row.canonical
            for value in row.normalized_inputs
        }
        | {
            value
            for output in receipt.outputs
            for value in (output.canonical, output.actual)
        }
        | {
            pair[0]
            for interface in receipt.interfaces
            for pair in (*interface.inputs, *interface.outputs)
            if pair is not None
        }
    )


def _check_domains(
    entries: tuple[tuple[str, dict[Witness, Witness], set[Witness]], ...],
) -> None:
    from .source_structure import StructureUnavailable

    domains: dict[Witness, tuple[str, set[Witness]]] = {}
    for kind, relation, dependencies in entries:
        for old in relation:
            if old in domains:
                raise StructureUnavailable("leaf-cse-alias-domain-conflict")
            domains[old] = (kind, dependencies)
    # Iterative DFS also rejects genuine opposite-order clone/catalog cycles.
    finished: set[Witness] = set()
    active: set[Witness] = set()
    for start in domains:
        pending = [(start, False)]
        while pending:
            value, leaving = pending.pop()
            if leaving:
                active.remove(value)
                finished.add(value)
            elif value not in finished:
                if value in active:
                    raise StructureUnavailable("leaf-cse-cross-catalog-cycle")
                active.add(value)
                pending.append((value, True))
                pending.extend(
                    (dependency, False)
                    for dependency in domains[value][1]
                    if dependency in domains
                )


def _check_catalog(
    graph: ir.Graph, leaves: tuple[LeafCSE, ...]
) -> tuple[tuple[str, dict[Witness, Witness], set[Witness]], ...]:
    """Check every finite substitution domain before any resolver callback."""
    from . import _source_identity_dce as dce
    from .source_structure import StructureUnavailable, _constant_lift_receipts

    identities = identity.receipts(graph)
    dead = dce.receipts(graph)
    lifts = _constant_lift_receipts(graph)
    entries: list[tuple[str, dict[Witness, Witness], set[Witness]]] = []
    for leaf_receipt in leaves:
        entries.append(("leaf", _congruence(leaf_receipt), _dependencies(leaf_receipt)))
    for identity_receipt in identities:
        dependencies = {row.canonical for row in identity_receipt.rows}
        dependencies.update(
            row.normalized_input
            for row in identity_receipt.rows
            if row.value == row.canonical
        )
        dependencies.update(
            value
            for output in identity_receipt.outputs
            for value in (output.canonical, output.actual)
        )
        entries.append(
            ("identity", identity._congruence(identity_receipt), dependencies)
        )
    for dead_receipt in dead:
        dependencies = {dead_receipt.anchor}
        dependencies.update(value for value, _ in dead_receipt.outputs)
        dependencies.update(
            pair[0]
            for interface in dead_receipt.interfaces
            for pair in (*interface.inputs, *interface.outputs)
            if pair is not None
        )
        entries.append(("dce", dce._relation(dead_receipt), dependencies))
    for lift_receipt in lifts:
        entries.append(
            ("lift", {lift_receipt.before: lift_receipt.after}, {lift_receipt.after})
        )
    _check_domains(tuple(entries))
    if leaves:
        # The sole leaf definition-role transition is the authentic later lift
        # of a retained Constant, checked directly before resolver recursion.
        from .source_structure import _resolve_constant_lift, _resolve_direct_bindings

        canonical_rows = {
            row.value: row
            for receipt in leaves
            for row in receipt.rows
            if row.value == row.canonical
        }
        for lift in lifts:
            row = canonical_rows.get(lift.before)
            if row is not None:
                if row.operation != "Constant" or row.value not in lift.markers:
                    raise StructureUnavailable(
                        "leaf-cse-nonchronological-lift-dependency"
                    )
                actual = _resolve_direct_bindings(graph, {lift.after})[lift.after]
                _resolve_constant_lift(graph, lift, actual)
                if (
                    actual.const_value is None
                    or _storage(actual, actual.const_value) != row.storage
                ):
                    raise StructureUnavailable(
                        "leaf-cse-lift-storage-dependency-differs"
                    )
        # Existing DCE never observed leaf receipts. A later deleted witness or
        # changed use of this closure therefore has no leaf-specific authority.
        protected = set().union(
            *(_dependencies(receipt) | set(_congruence(receipt)) for receipt in leaves)
        )
        touched_origins = {
            (interface.owner, interface.origin)
            for receipt in leaves
            for interface in receipt.interfaces
        }
        for removed_receipt in dead:
            removed = set(dce._relation(removed_receipt))
            if removed & protected or any(
                (value.owner, value.origin) in touched_origins for value in removed
            ):
                raise StructureUnavailable("leaf-cse-unproved-dce-overlap")
    return tuple(entries)


def receipts(graph: ir.Graph) -> tuple[LeafCSE, ...]:
    from .source_structure import StructureUnavailable, _OWNER

    result = graph.meta.get(_RECEIPTS, ())
    if (
        type(result) is not tuple
        or any(
            type(receipt) is not LeafCSE or receipt not in _REGISTERED
            for receipt in result
        )
        or len(set(result)) != len(result)
        or any(receipt.owner is not graph.meta.get(_OWNER) for receipt in result)
    ):
        raise StructureUnavailable("leaf-cse-registered-receipts-unavailable")
    return result


def _catalog_graphs(graph: ir.Graph) -> dict[object, ir.Graph]:
    """Discover the actual finite owner closure without resolving a witness.

    Descendant attributes expose consumers; actual operand/output definition
    graphs expose lexical ancestors. Walking graph identity once permits valid
    captures while rejecting an ambiguous owner shared by separate live graphs.
    """
    from .source_structure import StructureUnavailable, _OWNER

    owners: dict[object, ir.Graph] = {}
    seen: set[ir.Graph] = set()
    pending = [graph]
    while pending:
        root = pending.pop()
        if root in seen:
            continue
        for child in identity._graphs(root):
            if child in seen:
                continue
            seen.add(child)
            owner = child.meta.get(_OWNER)
            if type(owner) is not object or (
                owner in owners and owners[owner] is not child
            ):
                raise StructureUnavailable("leaf-cse-catalog-owner-unavailable")
            owners[owner] = child
            values = (
                *child.inputs,
                *child.outputs,
                *child.initializers.values(),
                *(
                    value
                    for node in child
                    for value in (*node.inputs, *node.outputs)
                    if value is not None
                ),
            )
            for value in values:
                definition_graph = value.graph
                if definition_graph is None:
                    # Discovery adds actual defining graphs; detached operands
                    # gain no validation or substitution authority here.
                    continue
                if definition_graph not in seen:
                    pending.append(definition_graph)
    return owners


def _validate_catalogs(
    graph: ir.Graph, proposed: tuple[LeafCSE, ...] | None = None
) -> None:
    """Check all reachable catalogs before any recursive resolver callback.

    A pending receipt is checked by the same global gate before registration;
    the graph's authentic attached catalog remains unchanged throughout.
    """
    from .source_structure import StructureUnavailable

    owners = _catalog_graphs(graph)
    combined: list[tuple[str, dict[Witness, Witness], set[Witness]]] = []
    for owner, child in owners.items():
        problems = child.meta.get(_GAPS, ())
        if type(problems) is not tuple or any(
            type(item) is not str for item in problems
        ):
            raise StructureUnavailable("leaf-cse-problem-metadata-unavailable")
        if problems:
            raise StructureUnavailable(problems[0])
        attached = receipts(child)
        leaves = proposed if child is graph and proposed is not None else attached
        entries = _check_catalog(child, leaves)
        for _, relation, dependencies in entries:
            if any(old.owner is not owner for old in relation):
                raise StructureUnavailable("leaf-cse-catalog-alias-owner-unavailable")
            if any(dependency.owner not in owners for dependency in dependencies):
                raise StructureUnavailable(
                    "leaf-cse-catalog-dependency-owner-unavailable"
                )
        combined.extend(entries)
    _check_domains(tuple(combined))


def validate_leaf_catalog(graph: ir.Graph) -> None:
    _validate_catalogs(graph)


def _validate(
    graph: ir.Graph, receipt: LeafCSE, resolve: identity.Resolver
) -> dict[Witness, ir.Value]:
    from .source_structure import (
        StructureMismatch,
        StructureUnavailable,
        _ORIGIN,
        _OWNER,
        _VALUES,
        _constant_lift_receipts,
        _resolve_direct_bindings,
    )

    validate_leaf_catalog(graph)
    output_aliases.validate_outputs(
        graph, receipt.output_classes, receipt.output_names, receipt.output_schemas
    )
    aliases = _congruence(receipt)
    if (
        graph.meta.get(_OWNER) is not receipt.owner
        or identity.receipts(graph) != receipt.identities
    ):
        raise StructureUnavailable("leaf-cse-final-owner-or-identity-catalog-differs")
    lifts = {item.before: item for item in _constant_lift_receipts(graph)}
    canonical_rows = [row for row in receipt.rows if row.value == row.canonical]
    wanted = _dependencies(receipt)
    if wanted & set(aliases):
        raise StructureUnavailable("leaf-cse-final-self-dependency")
    resolved: dict[Witness, ir.Value] = {}
    # Dependencies can belong to a lexical ancestor or descendant graph. Bind
    # through the matching authentic owner rather than reconstructing by name.
    scan = identity._graphs(graph)
    for witness in wanted:
        owners = [child for child in scan if child.meta.get(_OWNER) is witness.owner]
        if not owners:
            # Outer values are reached only through actual touched operands.
            owners = list(
                {
                    value.graph
                    for node in graph
                    for value in node.inputs
                    if value is not None
                    and value.graph is not None
                    and value.graph.meta.get(_OWNER) is witness.owner
                }
            )
        if len(owners) != 1:
            raise StructureUnavailable("leaf-cse-final-dependency-owner-unavailable")
        resolved[witness] = resolve(owners[0], {witness}, frozenset())[witness]
    old_origins = {old.origin for old in aliases}
    for child in scan:
        for node in child:
            if node.meta.get(_ORIGIN) in old_origins:
                raise StructureUnavailable("leaf-cse-removed-origin-reappeared")
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
                raise StructureUnavailable("leaf-cse-removed-marker-reappeared")
    lifted_origins = set()
    for row in canonical_rows:
        actual = resolved[row.value]
        lift = lifts.get(row.value)
        if lift is not None:
            if (
                row.operation != "Constant"
                or _resolve_direct_bindings(graph, {lift.after})[lift.after]
                is not actual
                or actual.const_value is None
            ):
                raise StructureMismatch("leaf-cse-final-lift-differs")
            if _storage(actual, actual.const_value) != row.storage:
                raise StructureMismatch("leaf-cse-final-lift-storage-differs")
            lifted_origins.add(row.value.origin)
        else:
            if _resolve_direct_bindings(graph, {row.value})[row.value] is not actual:
                raise StructureMismatch("leaf-cse-final-canonical-binding-differs")
            canonical_node = actual.producer()
            if canonical_node is None:
                raise StructureMismatch("leaf-cse-final-canonical-producer-missing")
            value, inputs, schema, storage = _leaf(canonical_node, graph)
            if (
                value is not actual
                or canonical_node.op_type != row.operation
                or schema != row.schema
                or storage != row.storage
                or inputs
                != tuple(resolved[witness] for witness in row.normalized_inputs)
                or tuple(identity._schema(value) for value in inputs)
                != row.input_schemas
            ):
                raise StructureMismatch(
                    "leaf-cse-final-canonical-schema-or-edges-differ"
                )
    keepers = {row.value for row in canonical_rows}
    if set(dict(receipt.uses)) != keepers or len(dict(receipt.uses)) != len(
        receipt.uses
    ):
        raise StructureUnavailable("leaf-cse-canonical-use-coverage-incomplete")
    for witness, ports in receipt.uses:
        value = resolved[witness]
        actual_ports = tuple(identity._port(use.node, use.idx) for use in value.uses())
        if len(actual_ports) != len(ports) or set(actual_ports) != set(ports):
            raise StructureMismatch("leaf-cse-final-uses-differ")
        for port in ports:
            nodes = identity._nodes(graph, port.owner, port.origin)
            if (
                len(nodes) != 1
                or port.port < 0
                or port.port >= len(nodes[0].inputs)
                or nodes[0].inputs[port.port] is not value
                or not identity._visible(value, nodes[0])
            ):
                raise StructureMismatch("leaf-cse-final-consumer-differs")
    for interface in receipt.interfaces:
        nodes = identity._nodes(graph, interface.owner, interface.origin)
        if len(nodes) != 1 or identity._node_schema(nodes[0]) != interface.schema:
            raise StructureMismatch("leaf-cse-final-interface-schema-differs")
        node = nodes[0]
        if tuple(node.inputs) != tuple(
            None if pair is None else resolved[pair[0]] for pair in interface.inputs
        ) or tuple(node.outputs) != tuple(
            resolved[pair[0]] for pair in interface.outputs
        ):
            raise StructureMismatch("leaf-cse-final-interface-edges-differ")
        for interface_value, pair in zip(
            (*node.inputs, *node.outputs),
            (*interface.inputs, *interface.outputs),
            strict=True,
        ):
            if pair is not None and (
                interface_value is None or identity._schema(interface_value) != pair[1]
            ):
                raise StructureMismatch("leaf-cse-final-interface-type-differs")
        if any(
            value is not None and not identity._visible(value, node)
            for value in node.inputs
        ):
            raise StructureMismatch("leaf-cse-final-interface-input-not-visible")
    if len(graph.outputs) != len(receipt.outputs):
        raise StructureMismatch("leaf-cse-final-output-count-differs")
    for slot, output in enumerate(receipt.outputs):
        actual, canonical = resolved[output.actual], resolved[output.canonical]
        if (
            output.slot != slot
            or graph.outputs[slot] is not actual
            or identity._schema(actual) != output.schema
            or actual.name != output.name
            or type(output.normalized) is not bool
            or output.slots
            != next(group for group in receipt.output_classes if slot in group)
            or tuple(i for i, value in enumerate(graph.outputs) if value is actual)
            != output.slots
        ):
            raise StructureMismatch("leaf-cse-final-output-slot-differs")
        if output.bridge:
            bridge_node = actual.producer()
            if bridge_node is None:
                raise StructureMismatch("leaf-cse-final-bridge-missing")
            input_value, value, schema = identity._identity(bridge_node, graph)
            if (
                input_value is not canonical
                or value is not actual
                or schema != output.schema
                or actual.uses()
                or tuple(i for i, value in enumerate(graph.outputs) if value is actual)
                != output.slots
                or not identity._visible(canonical, bridge_node)
            ):
                raise StructureMismatch("leaf-cse-final-bridge-differs")
        elif actual is not canonical:
            raise StructureMismatch("leaf-cse-final-direct-output-differs")
    output_aliases.validate_facts(
        graph, receipt.normalizations, receipt.outputs, resolved
    )
    for owner, origins in receipt.orders:
        matching = [child for child in scan if child.meta.get(_OWNER) is owner]
        expected = tuple(origin for origin in origins if origin not in lifted_origins)
        if len(matching) != 1 or len(set(origins)) != len(origins):
            raise StructureUnavailable("leaf-cse-final-order-owner-unavailable")
        if (
            tuple(
                node.meta.get(_ORIGIN)
                for node in matching[0]
                if node.meta.get(_ORIGIN) in origins
            )
            != expected
        ):
            raise StructureMismatch("leaf-cse-final-producer-order-differs")
    return {old: resolved[canonical] for old, canonical in aliases.items()}


def validate_leaf_cse(graph: ir.Graph, resolve: identity.Resolver) -> None:
    validate_leaf_catalog(graph)
    for receipt in receipts(graph):
        _validate(graph, receipt, resolve)


def resolve_leaf_bindings(
    graph: ir.Graph,
    wanted: set[Witness],
    optional: frozenset[Witness],
    resolve: identity.Resolver,
) -> dict[Witness, ir.Value] | None:
    attached = receipts(graph)
    aliases = {old for receipt in attached for old in _congruence(receipt)}
    redirected = wanted & aliases
    if not redirected:
        return None
    validate_leaf_catalog(graph)
    result = resolve(graph, wanted - redirected, optional - redirected)
    for receipt in attached:
        if redirected & set(_congruence(receipt)):
            found = _validate(graph, receipt, resolve)
            result.update((old, found[old]) for old in redirected & set(found))
    return result


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
def observe_leaf_cse(model: ir.Model) -> Iterator[output_aliases.Observation]:
    """Outer observation leaves the original pass and exception in control."""
    from jax2onnx.ir_utils import iter_ir_functions

    observation = output_aliases.Observation()
    snapshots = []
    try:
        roots = (
            model.graph,
            *(function.graph for function in iter_ir_functions(model.functions)),
        )
        graphs = tuple(graph for root in roots for graph in identity._graphs(root))
        if len(set(graphs)) != len(graphs):
            raise ValueError("leaf-cse-repeated-graph-owner")
        for graph in graphs:
            snapshot = _capture(graph)
            if snapshot is not None:
                snapshots.append(snapshot)
    except Exception as exc:
        _problem(model.graph, "leaf-cse-before", exc)
    for snapshot in snapshots:
        observation.originals[snapshot.graph] = tuple(
            snapshot.witnesses[value] for value in snapshot.outputs
        )
    yield observation
    for snapshot in snapshots:
        try:
            snapshot.normalizations = observation.facts(snapshot.graph)
            _finish(snapshot)
        except Exception as exc:
            _problem(snapshot.graph, "leaf-cse-after", exc)
