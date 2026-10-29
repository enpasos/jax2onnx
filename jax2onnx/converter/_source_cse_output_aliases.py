# jax2onnx/converter/_source_cse_output_aliases.py

"""Restore original SOURCE output Value classes after the one actual CSE pass.

Only the transaction holds IR objects. Frozen facts are inert historical data
inside authenticated parent receipts; they provide no substitution authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import onnx_ir as ir

from .precision_policy import PrecisionPolicy

if TYPE_CHECKING:
    from ._source_identity_cse import _NodeSchema, _Schema, _Output as IdentityOutput
    from ._source_leaf_cse import _Output as LeafOutput
    from .source_structure import _ValueWitness as Witness


@dataclass(frozen=True)
class _RawBridge:
    value: Witness
    input: Witness
    ordinal: int
    slots: tuple[int, ...]
    name: str
    schema: _Schema
    operation: _NodeSchema


@dataclass(frozen=True)
class Normalization:
    owner: object
    original: Witness
    slots: tuple[int, ...]
    name: str
    schema: _Schema
    canonical: Witness
    survivor: Witness
    raw: tuple[_RawBridge, ...]


@dataclass
class Observation:
    originals: dict[ir.Graph, tuple[Witness, ...]] = field(default_factory=dict)
    bound: dict[ir.Graph, tuple[Normalization, ...]] = field(default_factory=dict)
    transaction: Transaction | None = None
    failure: str | None = None

    def record(self, transaction: Transaction) -> None:
        """Optional witness recording cannot veto the physical transaction."""
        from . import _source_identity_cse as identity
        from .source_structure import _OWNER, StructureUnavailable

        self.transaction = transaction
        if transaction.failure is not None:
            return
        try:
            for graph, originals in self.originals.items():
                selected = tuple(
                    plan for plan in transaction.plans if plan.before.graph is graph
                )
                facts = []
                for plan in selected:
                    original = originals[plan.slots[0]]
                    owner = graph.meta.get(_OWNER)
                    if type(owner) is not object or original.owner is not owner:
                        raise StructureUnavailable(
                            "cse-output-alias-recording-owner-unavailable"
                        )
                    raw = tuple(
                        _RawBridge(
                            identity._current(item.value),
                            identity._current(item.input),
                            item.ordinal,
                            item.slots,
                            item.name,
                            item.schema,
                            item.operation,
                        )
                        for item in plan.raw
                    )
                    facts.append(
                        Normalization(
                            owner,
                            original,
                            plan.slots,
                            plan.raw[0].name,
                            plan.raw[0].schema,
                            identity._current(plan.canonical),
                            identity._current(plan.survivor),
                            raw,
                        )
                    )
                self.bound[graph] = tuple(facts)
        except Exception as exc:
            self.bound.clear()
            self.failure = "cse-output-alias-recording-unavailable:" + str(exc)

    def facts(self, graph: ir.Graph) -> tuple[Normalization, ...]:
        from .source_structure import StructureUnavailable

        if self.transaction is None:
            return ()
        if self.failure is not None:
            raise StructureUnavailable(self.failure)
        if self.transaction.failure is not None:
            raise StructureUnavailable(self.transaction.failure)
        if not self.transaction.finished:
            raise StructureUnavailable("cse-output-alias-transaction-incomplete")
        return self.bound.get(graph, ())


def slots(values: tuple[ir.Value, ...]) -> tuple[tuple[int, ...], ...]:
    """Partition every ordered occurrence by actual Value identity only."""
    result: list[tuple[int, ...]] = []
    seen: set[ir.Value] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(tuple(i for i, item in enumerate(values) if item is value))
    return tuple(result)


def output_contract(
    graph: ir.Graph,
) -> tuple[tuple[tuple[int, ...], ...], tuple[str | None, ...], tuple[_Schema, ...]]:
    from . import _source_identity_cse as identity

    values = tuple(graph.outputs)
    return (
        slots(values),
        tuple(value.name for value in values),
        tuple(identity._schema(value) for value in values),
    )


def validate_outputs(
    graph: ir.Graph,
    classes: tuple[tuple[int, ...], ...],
    names: tuple[str | None, ...],
    schemas: tuple[_Schema, ...],
) -> None:
    from .source_structure import StructureMismatch

    if output_contract(graph) != (classes, names, schemas):
        raise StructureMismatch("cse-output-alias-current-interface-differs")


@dataclass
class _Before:
    graph: ir.Graph
    nodes: tuple[ir.Node, ...]
    values: frozenset[ir.Value]
    named_values: tuple[tuple[ir.Value, str | None], ...]
    definitions: dict[ir.Value, tuple[ir.Graph | None, ir.Node | None, int | None]]
    outputs: tuple[ir.Value, ...]
    classes: tuple[tuple[int, ...], ...]
    names: tuple[str | None, ...]
    schemas: tuple[_Schema, ...]


@dataclass(frozen=True)
class _Scope:
    roots: tuple[ir.Graph, ...]
    graphs: tuple[ir.Graph, ...]
    holders: tuple[tuple[ir.Graph, ir.Node, str, int, ir.Graph], ...]


def _scope(model: ir.Model) -> _Scope:
    """Actual graph/holder closure, including actual lexical ancestors."""
    from jax2onnx.ir_utils import iter_ir_functions
    from .source_structure import StructureUnavailable

    roots = (
        model.graph,
        *(function.graph for function in iter_ir_functions(model.functions)),
    )
    if len(set(roots)) != len(roots):
        raise StructureUnavailable("cse-output-alias-repeated-root")
    graphs: list[ir.Graph] = []
    active: set[ir.Graph] = set()
    holders: list[tuple[ir.Graph, ir.Node, str, int, ir.Graph]] = []
    parents: dict[ir.Graph, tuple[ir.Graph, ir.Node, str, int]] = {}

    def visit(graph: ir.Graph) -> None:
        if graph in active:
            raise StructureUnavailable("cse-output-alias-graph-cycle")
        if graph in graphs:
            return
        active.add(graph)
        graphs.append(graph)
        for node in graph:
            if node.graph is not graph:
                raise StructureUnavailable("cse-output-alias-node-owner-differs")
            for name, attribute in node.attributes.items():
                if name != attribute.name:
                    raise StructureUnavailable(
                        "cse-output-alias-attribute-key-name-differs"
                    )
                children = (
                    (attribute.as_graph(),)
                    if attribute.type is ir.AttributeType.GRAPH
                    else (
                        tuple(attribute.as_graphs())
                        if attribute.type is ir.AttributeType.GRAPHS
                        else ()
                    )
                )
                for index, child in enumerate(children):
                    holder = (graph, node, name, index)
                    if child in parents and parents[child] != holder:
                        raise StructureUnavailable("cse-output-alias-holder-ambiguous")
                    parents[child] = holder
                    holders.append((*holder, child))
                    visit(child)
        active.remove(graph)

    for root in roots:
        visit(root)
    # A model used as an actual child may retain legal references to its real
    # parent's Values. Follow actual graph references, never printed names.
    index = 0
    while index < len(graphs):
        graph = graphs[index]
        index += 1
        for value in (
            *graph.outputs,
            *(value for node in graph for value in node.inputs if value is not None),
        ):
            if value.graph is not None and value.graph not in graphs:
                visit(value.graph)
    for child in graphs:
        seen: set[ir.Graph] = set()
        cursor = child
        while cursor in parents:
            if cursor in seen:
                raise StructureUnavailable("cse-output-alias-holder-cycle")
            seen.add(cursor)
            cursor = parents[cursor][0]
    return _Scope(roots, tuple(graphs), tuple(holders))


def _parents(scope: _Scope) -> dict[ir.Graph, tuple[ir.Graph, ir.Node]]:
    return {child: (parent, holder) for parent, holder, _, _, child in scope.holders}


def _visible(value: ir.Value, node: ir.Node, scope: _Scope) -> bool:
    graph = value.graph
    consumer = node.graph
    if (
        graph is None
        or consumer is None
        or graph not in scope.graphs
        or consumer not in scope.graphs
    ):
        return False
    producer = value.producer()
    port = value.index()
    if producer is None:
        if value not in graph.inputs and value not in graph.initializers.values():
            return False
    elif (
        producer.graph is not graph
        or producer not in set(graph)
        or port is None
        or port < 0
        or port >= len(producer.outputs)
        or producer.outputs[port] is not value
    ):
        return False
    parents = _parents(scope)
    point = node
    visited: set[ir.Graph] = set()
    while consumer is not graph:
        if consumer in visited or consumer not in parents:
            return False
        visited.add(consumer)
        consumer, point = parents[consumer]
    return producer is None or tuple(graph).index(producer) < tuple(graph).index(point)


def _descendant(child: ir.Graph, parent: ir.Graph, scope: _Scope) -> bool:
    parents = _parents(scope)
    visited: set[ir.Graph] = set()
    while child is not parent:
        if child in visited or child not in parents:
            return False
        visited.add(child)
        child = parents[child][0]
    return True


def _binding_visible(
    value: ir.Value, graph: ir.Graph, point: ir.Node | None, scope: _Scope
) -> bool:
    """The nearest named binding must be this actual Value at this occurrence."""
    name = value.name
    parents = _parents(scope)
    visited: set[ir.Graph] = set()
    while graph not in visited:
        visited.add(graph)
        nodes = tuple(graph)
        if point is not None and point not in nodes:
            return False
        bindings = tuple(
            dict.fromkeys(
                (
                    *graph.inputs,
                    *graph.initializers.values(),
                    *(output for node in nodes for output in node.outputs),
                )
            )
        )
        named = tuple(item for item in bindings if item.name == name)
        if named:
            if len(named) != 1 or named[0] is not value or value.graph is not graph:
                return False
            producer = value.producer()
            if producer is None:
                return value in graph.inputs or value in graph.initializers.values()
            port = value.index()
            return (
                producer.graph is graph
                and producer in nodes
                and port is not None
                and 0 <= port < len(producer.outputs)
                and producer.outputs[port] is value
                and (point is None or nodes.index(producer) < nodes.index(point))
            )
        if graph not in parents:
            return False
        graph, point = parents[graph]
    return False


def _hidden(value: ir.Value, name: str, graph: ir.Graph, scope: _Scope) -> bool:
    for child in scope.graphs:
        occurrences: tuple[tuple[ir.Node | None, ir.Value], ...] = (
            *(
                (node, item)
                for node in child
                for item in node.inputs
                if item is not None
            ),
            *((None, output) for output in child.outputs),
        )
        all_values = (
            *child.inputs,
            *child.outputs,
            *child.initializers.values(),
            *(
                item
                for node in child
                for item in (*node.inputs, *node.outputs)
                if item is not None
            ),
        )
        if child is not graph and any(item is value for item in all_values):
            return True
        if child is graph or not _descendant(child, graph, scope):
            continue
        # Printed names only identify ambiguity for rejection. A definition in
        # another graph is not a binding at this particular serialized use.
        for point, item in occurrences:
            if (
                item is not value
                and item.name == name
                and not _binding_visible(item, child, point, scope)
            ):
                return True
    return False


@dataclass(frozen=True)
class _AttributeState:
    key: str
    attribute: ir.Attr
    name: str
    type: ir.AttributeType
    reference: str | None
    children: tuple[ir.Graph, ...]
    documentation: str | None


def _attributes(node: ir.Node) -> tuple[_AttributeState, ...]:
    return tuple(
        _AttributeState(
            key,
            attribute,
            attribute.name,
            attribute.type,
            attribute.ref_attr_name,
            (
                (attribute.as_graph(),)
                if attribute.type is ir.AttributeType.GRAPH
                else (
                    tuple(attribute.as_graphs())
                    if attribute.type is ir.AttributeType.GRAPHS
                    else ()
                )
            ),
            attribute.doc_string,
        )
        for key, attribute in node.attributes.items()
    )


@dataclass
class _NodeState:
    node: ir.Node
    graph: ir.Graph | None
    name: str | None
    domain: str
    operation: str
    overload: str
    attributes: tuple[_AttributeState, ...]


@dataclass
class _ValueState:
    value: ir.Value
    graph: ir.Graph | None
    name: str | None
    dtype: ir.DataType | None
    shape: tuple[object, ...] | None
    producer: ir.Node | None
    port: int | None
    uses: frozenset[tuple[ir.Node, int]]
    constant: ir.TensorProtocol | None


@dataclass
class _Local:
    graph: ir.Graph
    nodes: tuple[ir.Node, ...]
    inputs: tuple[ir.Value, ...]
    outputs: tuple[ir.Value, ...]
    initializers: tuple[tuple[str, ir.Value], ...]
    ports: tuple[tuple[ir.Node, tuple[ir.Value | None, ...], tuple[ir.Value, ...]], ...]
    node_fields: tuple[_NodeState, ...]
    values: tuple[_ValueState, ...]


def _physical(graphs: tuple[ir.Graph, ...]) -> tuple[_Local, ...]:
    result = []
    for graph in graphs:
        nodes = tuple(graph)
        values = dict.fromkeys(
            (
                *graph.inputs,
                *graph.outputs,
                *graph.initializers.values(),
                *(
                    value
                    for node in nodes
                    for value in (*node.inputs, *node.outputs)
                    if value is not None
                ),
            )
        )
        result.append(
            _Local(
                graph,
                nodes,
                tuple(graph.inputs),
                tuple(graph.outputs),
                tuple(graph.initializers.items()),
                tuple(
                    (node, tuple(node.inputs), tuple(node.outputs)) for node in nodes
                ),
                tuple(
                    _NodeState(
                        node,
                        node.graph,
                        node.name,
                        node.domain,
                        node.op_type,
                        node.overload,
                        _attributes(node),
                    )
                    for node in nodes
                ),
                tuple(
                    _ValueState(
                        value,
                        value.graph,
                        value.name,
                        value.dtype,
                        None if value.shape is None else tuple(value.shape),
                        value.producer(),
                        value.index(),
                        frozenset((use.node, use.idx) for use in value.uses()),
                        value.const_value,
                    )
                    for value in values
                ),
            )
        )
    return tuple(result)


def _unchanged(before: tuple[_Local, ...]) -> bool:
    return _physical(tuple(item.graph for item in before)) == before


@dataclass
class _Plan:
    before: _Before
    slots: tuple[int, ...]
    original: ir.Value
    canonical: ir.Value
    survivor: ir.Value
    bridges: tuple[ir.Node, ...]
    raw: tuple[_PhysicalBridge, ...]


@dataclass
class _PhysicalBridge:
    value: ir.Value
    input: ir.Value
    ordinal: int
    slots: tuple[int, ...]
    name: str
    schema: _Schema
    operation: _NodeSchema


def _capture(graph: ir.Graph) -> _Before:
    nodes = tuple(graph)
    outputs = tuple(graph.outputs)
    classes, names, schemas = output_contract(graph)
    values = frozenset(
        (
            *graph.inputs,
            *graph.outputs,
            *graph.initializers.values(),
            *(
                value
                for node in nodes
                for value in (*node.inputs, *node.outputs)
                if value is not None
            ),
        )
    )
    return _Before(
        graph,
        nodes,
        values,
        tuple((value, value.name) for value in values),
        {value: (value.graph, value.producer(), value.index()) for value in values},
        outputs,
        classes,
        names,
        schemas,
    )


def _plan(before: _Before, scope: _Scope) -> tuple[_Plan, ...]:
    from . import _source_identity_cse as identity
    from .source_structure import StructureMismatch

    graph = before.graph
    current_nodes = tuple(graph)
    if len(graph.outputs) != len(before.outputs):
        raise StructureMismatch("cse-output-alias-owner-or-slot-count-differs")
    plans = []
    for group in before.classes:
        actual = tuple(graph.outputs[i] for i in group)
        distinct = tuple(dict.fromkeys(actual))
        if len(distinct) == 1:
            continue
        if len(group) < 2:
            raise StructureMismatch("cse-output-alias-single-slot-split")
        original = before.outputs[group[0]]
        producer = original.producer()
        if producer is None or producer.graph is not None or producer in current_nodes:
            raise StructureMismatch("cse-output-alias-original-definition-retained")
        name, schema = before.names[group[0]], before.schemas[group[0]]
        if type(name) is not str or not name:
            raise StructureMismatch("cse-output-alias-original-name-unavailable")
        if any(
            value is not original and old_name == name
            for value, old_name in before.named_values
        ):
            raise StructureMismatch("cse-output-alias-original-name-ambiguous")
        nodes = []
        canonical = None
        for value in distinct:
            node = value.producer()
            if node is None or node in before.nodes or node not in current_nodes:
                raise StructureMismatch("cse-output-alias-bridge-not-fresh")
            input_value, output, output_schema = identity._identity(node, graph)
            input_producer = input_value.producer()
            if (
                output is not value
                or output_schema != schema
                or value.name != name
                or value.uses()
                or input_value not in before.values
                or (input_value.graph, input_value.producer(), input_value.index())
                != before.definitions.get(input_value)
                or (
                    input_producer is not None
                    and (
                        input_producer.graph is None
                        or input_producer not in set(input_producer.graph)
                    )
                )
                or not _visible(input_value, node, scope)
            ):
                raise StructureMismatch("cse-output-alias-bridge-interface-differs")
            if canonical is None:
                canonical = input_value
            elif canonical is not input_value:
                raise StructureMismatch("cse-output-alias-canonical-input-differs")
            if _hidden(value, name, graph, scope):
                raise StructureMismatch("cse-output-alias-hidden-interface-use")
            nodes.append(node)
        if canonical is None:
            raise StructureMismatch("cse-output-alias-canonical-input-missing")
        if (
            tuple(i for i, value in enumerate(graph.outputs) if value in distinct)
            != group
        ):
            raise StructureMismatch("cse-output-alias-unrecorded-output-occurrence")
        all_values = (
            *graph.inputs,
            *graph.initializers.values(),
            *(value for node in current_nodes for value in node.outputs),
        )
        if any(value not in distinct and value.name == name for value in all_values):
            raise StructureMismatch("cse-output-alias-current-name-ambiguous")
        raw = tuple(
            _PhysicalBridge(
                value,
                canonical,
                current_nodes.index(node),
                tuple(i for i, item in enumerate(graph.outputs) if item is value),
                name,
                schema,
                identity._NodeSchema("", "Identity", "", 1, 1, ()),
            )
            for value, node in zip(distinct, nodes, strict=True)
        )
        plans.append(
            _Plan(before, group, original, canonical, actual[0], tuple(nodes), raw)
        )
    return tuple(plans)


def _verify_normalized(
    raw: tuple[_Local, ...], plans: tuple[_Plan, ...], scope: _Scope
) -> None:
    from . import _source_identity_cse as identity
    from .source_structure import StructureMismatch

    # Value.uses() is global. Authenticate one complete frozen port domain
    # before deriving the same planned edge delta for every occurrence of a
    # captured Value, including occurrences in planless descendant snapshots.
    raw_graphs: dict[ir.Graph, _Local] = {}
    raw_ports: dict[
        ir.Node, tuple[ir.Graph, tuple[ir.Value | None, ...], tuple[ir.Value, ...]]
    ] = {}
    raw_values: dict[ir.Value, _ValueState] = {}
    for raw_local in raw:
        if (
            raw_local.graph in raw_graphs
            or len(set(raw_local.nodes)) != len(raw_local.nodes)
            or tuple(node for node, _, _ in raw_local.ports) != raw_local.nodes
            or tuple(item.node for item in raw_local.node_fields) != raw_local.nodes
            or any(item.graph is not raw_local.graph for item in raw_local.node_fields)
        ):
            raise StructureMismatch("cse-output-alias-raw-node-record-differs")
        raw_graphs[raw_local.graph] = raw_local
        for raw_node, raw_inputs, raw_outputs in raw_local.ports:
            if raw_node in raw_ports:
                raise StructureMismatch("cse-output-alias-raw-node-record-differs")
            raw_ports[raw_node] = (raw_local.graph, raw_inputs, raw_outputs)
        for saved_value in raw_local.values:
            prior_value = raw_values.get(saved_value.value)
            if prior_value is not None and (
                prior_value.graph is not saved_value.graph
                or prior_value.producer is not saved_value.producer
                or prior_value.port != saved_value.port
                or prior_value.name != saved_value.name
                or prior_value.dtype is not saved_value.dtype
                or prior_value.shape != saved_value.shape
                or prior_value.constant is not saved_value.constant
                or prior_value.uses != saved_value.uses
            ):
                raise StructureMismatch("cse-output-alias-raw-value-record-differs")
            raw_values[saved_value.value] = saved_value
    # The frozen Value usage catalog must agree with every frozen in-closure
    # input port. External uses remain in the complete saved set unchanged.
    for catalog_node, (_, catalog_inputs, catalog_outputs) in raw_ports.items():
        for catalog_index, catalog_operand in enumerate(catalog_inputs):
            if catalog_operand is not None and (
                catalog_operand not in raw_values
                or (catalog_node, catalog_index) not in raw_values[catalog_operand].uses
            ):
                raise StructureMismatch("cse-output-alias-raw-value-record-differs")
        for catalog_index, catalog_output in enumerate(catalog_outputs):
            saved_output = raw_values.get(catalog_output)
            if (
                saved_output is None
                or saved_output.producer is not catalog_node
                or saved_output.port != catalog_index
                or saved_output.graph is not raw_ports[catalog_node][0]
            ):
                raise StructureMismatch("cse-output-alias-raw-value-record-differs")
    for catalog_value, saved_catalog_value in raw_values.items():
        for consumer, port in saved_catalog_value.uses:
            if consumer in raw_ports:
                inputs = raw_ports[consumer][1]
                if port < 0 or port >= len(inputs) or inputs[port] is not catalog_value:
                    raise StructureMismatch("cse-output-alias-raw-value-record-differs")

    all_bridges: set[ir.Node] = set()
    all_removed: set[ir.Node] = set()
    for global_plan in plans:
        plan_local = raw_graphs.get(global_plan.before.graph)
        if (
            plan_local is None
            or len(global_plan.bridges) != len(global_plan.raw)
            or sum(item.value is global_plan.survivor for item in global_plan.raw) != 1
        ):
            raise StructureMismatch("cse-output-alias-raw-plan-binding-differs")
        for bridge, bridge_raw in zip(
            global_plan.bridges, global_plan.raw, strict=True
        ):
            record = raw_ports.get(bridge)
            if (
                bridge in all_bridges
                or record is None
                or record[0] is not global_plan.before.graph
                or record[1] != (global_plan.canonical,)
                or record[2] != (bridge_raw.value,)
                or bridge_raw.input is not global_plan.canonical
                or bridge_raw.ordinal < 0
                or bridge_raw.ordinal >= len(plan_local.nodes)
                or plan_local.nodes[bridge_raw.ordinal] is not bridge
                or bridge_raw.value.producer() is not bridge
                or bridge_raw.value.index() != 0
            ):
                raise StructureMismatch("cse-output-alias-raw-plan-binding-differs")
            all_bridges.add(bridge)
            if bridge_raw.value is not global_plan.survivor:
                all_removed.add(bridge)
    removed_edges: dict[ir.Value, set[tuple[ir.Node, int]]] = {}
    for removed_node in all_removed:
        for index, operand in enumerate(raw_ports[removed_node][1]):
            if operand is not None:
                removed_edges.setdefault(operand, set()).add((removed_node, index))

    by_graph: dict[ir.Graph, list[_Plan]] = {}
    for plan in plans:
        by_graph.setdefault(plan.before.graph, []).append(plan)
    for local in raw:
        graph = local.graph
        selected = by_graph.get(graph, [])
        removed = {
            node
            for plan in selected
            for node in plan.bridges
            if node.outputs[0] is not plan.survivor
        }
        expected_outputs = list(local.outputs)
        for plan in selected:
            for slot in plan.slots:
                expected_outputs[slot] = plan.survivor
        if (
            tuple(graph) != tuple(node for node in local.nodes if node not in removed)
            or tuple(graph.outputs) != tuple(expected_outputs)
            or tuple(graph.inputs) != local.inputs
            or tuple(graph.initializers.items()) != local.initializers
        ):
            raise StructureMismatch("cse-output-alias-normalized-physical-differs")
        for node, inputs, outputs in local.ports:
            if (
                tuple(node.outputs) != outputs
                or (node not in removed and tuple(node.inputs) != inputs)
                or (
                    node in removed
                    and (
                        node.graph is not None
                        or any(value is not None for value in node.inputs)
                    )
                )
            ):
                raise StructureMismatch("cse-output-alias-normalized-ports-differ")
        for fields in local.node_fields:
            node = fields.node
            expected = _NodeState(
                node,
                None if node in removed else fields.graph,
                fields.name,
                fields.domain,
                fields.operation,
                fields.overload,
                fields.attributes,
            )
            actual = _NodeState(
                node,
                node.graph,
                node.name,
                node.domain,
                node.op_type,
                node.overload,
                _attributes(node),
            )
            if actual != expected:
                raise StructureMismatch(
                    "cse-output-alias-normalized-node-fields-differ"
                )
        for plan in selected:
            validate_outputs(
                graph, plan.before.classes, plan.before.names, plan.before.schemas
            )
            producer = plan.survivor.producer()
            if producer is None:
                raise StructureMismatch("cse-output-alias-normalized-survivor-missing")
            input_value, output, schema = identity._identity(producer, graph)
            if (
                input_value is not plan.canonical
                or output is not plan.survivor
                or schema != plan.before.schemas[plan.slots[0]]
                or output.uses()
                or not _visible(input_value, producer, scope)
            ):
                raise StructureMismatch("cse-output-alias-normalized-survivor-differs")
        for value_fields in local.values:
            value = value_fields.value
            expected_uses = value_fields.uses - removed_edges.get(value, set())
            expected_graph = (
                None if value_fields.producer in all_removed else value_fields.graph
            )
            if (
                value.graph is not expected_graph
                or value.name != value_fields.name
                or value.dtype is not value_fields.dtype
                or (None if value.shape is None else tuple(value.shape))
                != value_fields.shape
                or value.producer() is not value_fields.producer
                or value.index() != value_fields.port
                or frozenset((use.node, use.idx) for use in value.uses())
                != expected_uses
                or value.const_value is not value_fields.constant
            ):
                raise StructureMismatch(
                    "cse-output-alias-normalized-value-fields-differ"
                )


def _rollback(raw: tuple[_Local, ...]) -> None:
    from .source_structure import StructureMismatch

    for local in raw:
        graph = local.graph
        for fields in local.node_fields:
            node = fields.node
            node.name, node.domain, node.op_type, node.overload = (
                fields.name,
                fields.domain,
                fields.operation,
                fields.overload,
            )
            for saved in fields.attributes:
                saved.attribute.name = saved.name
                saved.attribute.doc_string = saved.documentation
                children = (
                    (saved.attribute.as_graph(),)
                    if saved.attribute.type is ir.AttributeType.GRAPH
                    else (
                        tuple(saved.attribute.as_graphs())
                        if saved.attribute.type is ir.AttributeType.GRAPHS
                        else ()
                    )
                )
                if (
                    saved.attribute.type is not saved.type
                    or saved.attribute.ref_attr_name != saved.reference
                    or children != saved.children
                ):
                    raise StructureMismatch(
                        "cse-output-alias-rollback-attribute-unrestorable"
                    )
            if _attributes(node) != fields.attributes:
                node.attributes.clear()
                node.attributes.update(
                    {saved.key: saved.attribute for saved in fields.attributes}
                )
        graph.outputs[:] = list(local.outputs)
        for node, inputs, _ in local.ports:
            for index, value in enumerate(inputs):
                if node.inputs[index] is not value:
                    node.replace_input_with(index, value)
        # Insert each saved node before its saved successor, including detached
        # nodes whose safe removal disconnected their original input ports.
        successor = None
        for node in reversed(local.nodes):
            if successor is None:
                graph.append(node)
            else:
                graph.insert_before(successor, node)
            successor = node
    if not _unchanged(raw):
        raise StructureMismatch("cse-output-alias-rollback-incomplete")


@dataclass
class Transaction:
    model: ir.Model
    enabled: bool
    scope: _Scope | None = None
    snapshots: tuple[_Before, ...] = ()
    graphs: tuple[ir.Graph, ...] = ()
    raw: tuple[_Local, ...] = ()
    plans: tuple[_Plan, ...] = ()
    failure: str | None = None
    planned: bool = False
    finished: bool = False
    affected: bool = False

    def prepare(self) -> None:
        from .source_structure import StructureMismatch

        if not self.enabled:
            self.planned = True
            return
        # This latch precedes all detailed postpass capture and survives failed
        # planning. It depends only on captured classes and current slot identity.
        self.affected = self.affected or any(
            len(group) > 1
            and all(slot < len(before.graph.outputs) for slot in group)
            and len({before.graph.outputs[slot] for slot in group}) > 1
            for before in self.snapshots
            for group in before.classes
        )
        if self.failure is not None:
            self.planned = True
            return
        try:
            current = _scope(self.model)
            self.raw = _physical(current.graphs)
            if current != self.scope:
                raise StructureMismatch("cse-output-alias-current-attachments-differ")
            self.plans = tuple(
                plan for before in self.snapshots for plan in _plan(before, current)
            )
            if not _unchanged(self.raw) or _scope(self.model) != current:
                raise StructureMismatch("cse-output-alias-preflight-mutated")
        except Exception as exc:
            self._failed(exc)
        self.planned = True

    def _failed(self, exc: Exception) -> None:
        from .source_structure import StructureMismatch

        self.plans = ()
        self.failure = "cse-output-alias-unavailable:" + str(exc)
        try:
            if self.raw and not _unchanged(self.raw):
                _rollback(self.raw)
        except Exception as rollback_error:
            raise StructureMismatch(
                "cse-output-alias-rollback-failed"
            ) from rollback_error
        if self.affected:
            raise StructureMismatch(
                "cse-output-alias-required-repair:" + str(exc)
            ) from exc

    def finish(self) -> None:
        from .source_structure import StructureMismatch

        if not self.enabled or self.failure is not None:
            self.finished = True
            return
        try:
            if not self.planned or self.scope is None:
                raise StructureMismatch("cse-output-alias-plan-unavailable")
            if not _unchanged(self.raw) or _scope(self.model) != self.scope:
                raise StructureMismatch("cse-output-alias-precommit-mutated")
            for plan in self.plans:
                for slot in plan.slots:
                    plan.before.graph.outputs[slot] = plan.survivor
                for node in plan.bridges:
                    if node.outputs[0] is not plan.survivor:
                        plan.before.graph.remove(node, safe=True)
            _verify_normalized(self.raw, self.plans, self.scope)
            if _scope(self.model) != self.scope:
                raise StructureMismatch(
                    "cse-output-alias-postcommit-attachments-differ"
                )
        except Exception as exc:
            self._failed(exc)
        self.finished = True


def capture(
    model: ir.Model, precision_policy: PrecisionPolicy = PrecisionPolicy.LEGACY
) -> Transaction:
    result = Transaction(model, precision_policy is PrecisionPolicy.SOURCE)
    if not result.enabled:
        return result
    try:
        result.scope = _scope(model)
        result.graphs = result.scope.graphs
        result.snapshots = tuple(_capture(graph) for graph in result.graphs)
    except Exception as exc:
        result.failure = "cse-output-alias-before:" + str(exc)
    return result


def validate_facts(
    graph: ir.Graph,
    facts: tuple[Normalization, ...],
    outputs: tuple[IdentityOutput | LeafOutput, ...],
    resolved: dict[Witness, ir.Value],
) -> None:
    """Only a registered parent calls this; facts never resolve a witness."""
    from . import _source_identity_cse as identity
    from .source_structure import (
        StructureUnavailable,
        StructureMismatch,
        _OWNER,
        _ORIGIN,
        _VALUES,
    )

    if type(facts) is not tuple or any(
        type(item) is not Normalization for item in facts
    ):
        raise StructureUnavailable("cse-output-alias-facts-unavailable")
    seen: set[int] = set()
    normalized_slots = {output.slot for output in outputs if output.normalized}
    for fact in facts:
        records = tuple(
            output for output in outputs if output.original == fact.original
        )
        if (
            fact.owner is not graph.meta.get(_OWNER)
            or type(fact.slots) is not tuple
            or not fact.slots
            or tuple(output.slot for output in records) != fact.slots
            or any(slot in seen for slot in fact.slots)
            or len(fact.slots) < 2
            or type(fact.name) is not str
            or not fact.name
            or type(fact.raw) is not tuple
            or len(fact.raw) < 2
            or any(type(item) is not _RawBridge for item in fact.raw)
        ):
            raise StructureUnavailable("cse-output-alias-fact-class-unavailable")
        seen.update(fact.slots)
        if fact.canonical not in resolved or fact.survivor not in resolved:
            raise StructureUnavailable("cse-output-alias-unbound-live-dependency")
        canonical, actual = resolved[fact.canonical], resolved[fact.survivor]
        if (
            any(
                not output.bridge
                or not output.normalized
                or output.actual != fact.survivor
                or output.canonical != fact.canonical
                or output.schema != fact.schema
                for output in records
            )
            or tuple(i for i, value in enumerate(graph.outputs) if value is actual)
            != fact.slots
            or actual.name != fact.name
            or actual.uses()
        ):
            raise StructureMismatch("cse-output-alias-current-class-differs")
        raw_slots = tuple(sorted(slot for item in fact.raw for slot in item.slots))
        if (
            raw_slots != fact.slots
            or len(set(raw_slots)) != len(raw_slots)
            or len({item.value for item in fact.raw}) != len(fact.raw)
            or any(
                type(item.ordinal) is not int or item.ordinal < 0 for item in fact.raw
            )
            or tuple(item.ordinal for item in fact.raw)
            != tuple(sorted(set(item.ordinal for item in fact.raw)))
            or any(
                item.input != fact.canonical
                or item.schema != fact.schema
                or item.name != fact.name
                or not item.slots
                or tuple(sorted(set(item.slots))) != item.slots
                or item.operation != identity._NodeSchema("", "Identity", "", 1, 1, ())
                for item in fact.raw
            )
            or fact.raw[0].value != fact.survivor
            or fact.raw[0].slots[0] != fact.slots[0]
        ):
            raise StructureUnavailable("cse-output-alias-raw-class-differs")
        node = actual.producer()
        if node is None:
            raise StructureMismatch("cse-output-alias-survivor-missing")
        input_value, value, schema = identity._identity(node, graph)
        if (
            input_value is not canonical
            or value is not actual
            or schema != fact.schema
            or not identity._visible(canonical, node)
        ):
            raise StructureMismatch("cse-output-alias-survivor-differs")
        removed = {item.value for item in fact.raw if item.value != fact.survivor}
        for child in identity._graphs(graph):
            for candidate in child:
                if candidate.meta.get(_ORIGIN) in {value.origin for value in removed}:
                    raise StructureUnavailable(
                        "cse-output-alias-removed-bridge-reappeared"
                    )
            for value in (
                *child.inputs,
                *child.outputs,
                *child.initializers.values(),
                *(
                    value
                    for candidate in child
                    for value in (*candidate.inputs, *candidate.outputs)
                    if value is not None
                ),
            ):
                if any(marker in removed for marker in value.meta.get(_VALUES, ())):
                    raise StructureUnavailable(
                        "cse-output-alias-removed-bridge-marker-reappeared"
                    )
                if child is not graph and value is actual:
                    raise StructureMismatch(
                        "cse-output-alias-current-hidden-interface-use"
                    )
    if seen != normalized_slots:
        raise StructureUnavailable("cse-output-alias-normalization-coverage-differs")
