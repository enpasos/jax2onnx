# jax2onnx/converter/initializer_metadata.py

"""IR-only, export-local evidence for captured initializer preservation.

The registry owns strong references, never writes tracking markers into IR, and
never attempts to infer source identity from a name, annotation, or payload.
Only explicit capture, reshape, and graph-clone operations create lineage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx_ir as ir

from jax2onnx.initializer_metadata import (
    InitializerMetadataError,
    initializer_metadata_session as initializer_metadata_session,
)


@dataclass
class _Source:
    source: object
    snapshot: dict[str, str]
    handle: str


@dataclass
class _Capture:
    source: _Source
    dtype: str
    shape: tuple[int, ...]
    payload: bytes
    handle: str


_ACTIVE_REGISTRY: ContextVar[InitializerMetadataRegistry | None] = ContextVar(
    "jax2onnx_initializer_metadata", default=None
)


def active_registry() -> InitializerMetadataRegistry | None:
    return _ACTIVE_REGISTRY.get()


def is_annotated(value: object) -> bool:
    registry = active_registry()
    return registry is not None and registry.is_annotated(value)


def register_reshape(source: ir.Value, derived: ir.Value) -> None:
    registry = active_registry()
    if registry is not None:
        registry.register_reshape(source, derived)


def register_graph_clone(original: ir.Graph, cloned: ir.Graph) -> None:
    registry = active_registry()
    if registry is not None:
        registry.register_graph_clone(original, cloned)


def _normalized_result(result: object) -> dict[str, str]:
    if result is None:
        return {}
    if not isinstance(result, Mapping):
        raise TypeError("initializer_metadata must return a Mapping[str, str] or None")
    snapshot: dict[str, str] = {}
    for key, value in result.items():
        if not isinstance(key, str):
            raise TypeError("initializer_metadata keys must be nonempty strings")
        # Copy exact immutable text, including for str subclasses whose custom
        # equality/encoding methods must not control the accepted snapshot.
        key = str.__str__(key)
        if not isinstance(value, str):
            raise TypeError(f"initializer_metadata value for {key!r} must be a string")
        value = str.__str__(value)
        if not key:
            raise ValueError("initializer_metadata keys must be nonempty strings")
        try:
            key.encode("utf-8")
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "initializer_metadata text must be encodable as UTF-8"
            ) from exc
        snapshot[key] = value
    return snapshot


def _array_storage(array: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    # Normalize byte order, without numeric conversion or NaN canonicalization.
    dtype = array.dtype.newbyteorder("=")
    native = array.astype(dtype, copy=False)
    return str(dtype), tuple(int(dim) for dim in array.shape), native.tobytes(order="C")


def _tensor_storage(tensor: ir.TensorProtocol) -> tuple[str, tuple[int, ...], bytes]:
    return _array_storage(np.asarray(tensor.numpy()))


def _subgraphs(node: ir.Node) -> list[ir.Graph]:
    result: list[ir.Graph] = []
    for attr in node.attributes.values():
        if attr.is_ref():
            continue
        if attr.type == ir.AttributeType.GRAPH:
            result.append(attr.as_graph())
        elif attr.type == ir.AttributeType.GRAPHS:
            result.extend(attr.as_graphs())
    return result


def _stored(graph: ir.Graph) -> list[tuple[ir.Value | None, ir.TensorProtocol]]:
    result: list[tuple[ir.Value | None, ir.TensorProtocol]] = []
    for value in graph.initializers.values():
        if value.const_value is not None:
            result.append((value, value.const_value))
    for node in graph:
        for attr in node.attributes.values():
            if attr.is_ref():
                continue
            if attr.type == ir.AttributeType.TENSOR:
                tensor = attr.as_tensor()
                owner = (
                    node.outputs[0]
                    if node.op_type == "Constant" and node.outputs
                    else None
                )
                result.append((owner, tensor))
            elif attr.type == ir.AttributeType.TENSORS:
                result.extend((None, tensor) for tensor in attr.as_tensors())
        for child in _subgraphs(node):
            result.extend(_stored(child))
    return result


class InitializerMetadataRegistry:
    def __init__(
        self, callback: Callable[[object], Mapping[str, str] | None] | None
    ) -> None:
        self.callback = callback
        self.sources: dict[int, _Source] = {}
        self.captures: list[_Capture] = []
        self._values: dict[ir.Value, frozenset[int]] = {}
        self._tensors: dict[int, tuple[ir.TensorProtocol, frozenset[int]]] = {}
        self._expected: dict[ir.Value, frozenset[int]] = {}
        self._functions: dict[tuple[str, str, str], ir.Function] = {}
        self._sealed_topology: object = None
        self._has_annotations = False
        self._scope_outputs: dict[ir.Graph, tuple[ir.Value, ...]] = {}
        self._prepared_graphs: dict[ir.Graph, tuple[object, tuple[object, ...]]] = {}
        self._node_witnesses: dict[ir.Node, object] = {}
        self._unreviewed_lowering = False

    @property
    def has_annotations(self) -> bool:
        return self._has_annotations

    def _fail(self, obligation: str, stage: str, capture: int | None = None) -> None:
        handle = self.captures[capture].handle if capture is not None else "export"
        raise InitializerMetadataError(
            "preservation_failed",
            f"Initializer metadata {obligation}; stage={stage}, scope/source={handle}",
        )

    def admit_lowering(self, *, audited: bool) -> None:
        self._unreviewed_lowering |= not audited
        if self.has_annotations and self._unreviewed_lowering:
            self._fail(
                "unverified lowering implementation cannot preserve source provenance",
                "lowering admission",
            )

    def observe(self, source: object) -> _Source:
        snapshot = _normalized_result(
            self.callback(source) if self.callback is not None else None
        )
        record = self.sources.get(id(source))
        if record is not None:
            if record.source is not source:
                self._fail("source identity cannot be verified", "capture")
            if record.snapshot != snapshot:
                raise InitializerMetadataError(
                    "inconsistent_callback",
                    f"Initializer metadata callback disagrees; stage=capture, source={record.handle}",
                )
            return record
        record = _Source(source, snapshot, f"source-{len(self.sources)}")
        self.sources[id(source)] = record
        return record

    def check_supported(self, source: _Source, dtype: np.dtype) -> None:
        if source.snapshot and str(dtype) not in {
            "bool",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "float16",
            "bfloat16",
            "float32",
            "float64",
        }:
            raise InitializerMetadataError(
                "unsupported_representation",
                f"Initializer metadata requires dense supported storage; stage=capture, source={source.handle}",
            )

    def capture(self, source: _Source, value: ir.Value, array: np.ndarray) -> None:
        dtype, shape, payload = (
            _array_storage(array)
            if source.snapshot
            else (str(array.dtype), tuple(array.shape), b"")
        )
        self._has_annotations |= bool(source.snapshot)
        capture_id = len(self.captures)
        self.captures.append(
            _Capture(
                source, dtype, shape, payload, f"{source.handle}/capture-{capture_id}"
            )
        )
        if source.snapshot and self._unreviewed_lowering:
            self._fail(
                "unverified lowering implementation cannot preserve source provenance",
                "capture",
                capture_id,
            )
        self._register(value, frozenset({capture_id}), "capture")
        if source.snapshot:
            assert value.const_value is not None
            value.const_value.metadata_props.update(source.snapshot)

    def _register(self, value: ir.Value, captures: frozenset[int], stage: str) -> None:
        previous = self._values.get(value, frozenset())
        combined = previous | captures
        source_ids = {id(self.captures[index].source.source) for index in combined}
        if len(source_ids) > 1:
            self._fail("independent sources share storage", stage)
        self._values[value] = combined
        if value.const_value is not None:
            tensor = value.const_value
            prior = self._tensors.get(id(tensor))
            tensor_captures = combined | (
                prior[1] if prior is not None else frozenset()
            )
            if (
                len(
                    {
                        id(self.captures[index].source.source)
                        for index in tensor_captures
                    }
                )
                > 1
            ):
                self._fail("independent sources share a tensor", stage)
            self._tensors[id(tensor)] = (tensor, tensor_captures)

    def _accepted(self, captures: frozenset[int]) -> frozenset[int]:
        return frozenset(
            index for index in captures if self.captures[index].source.snapshot
        )

    def is_annotated(self, value: object) -> bool:
        if not self.has_annotations:
            return False
        if isinstance(value, ir.Value):
            return bool(self.dependencies(value))
        item = self._tensors.get(id(value))
        return item is not None and bool(self._accepted(item[1]))

    def register_reshape(self, source: ir.Value, derived: ir.Value) -> None:
        captures = self._values.get(source, frozenset())
        if not captures:
            return
        self._register(derived, captures, "reshape")
        for capture in self._accepted(captures):
            self._validate_tensor(derived.const_value, capture, "reshape")

    def register_function(self, function: ir.Function) -> None:
        self._functions[function.identifier()] = function

    def _node_witness(self, node: ir.Node) -> object:
        tensors: list[object] = []
        for name, attr in node.attributes.items():
            if attr.is_ref():
                continue
            if attr.type == ir.AttributeType.TENSOR:
                tensors.append((name, id(attr.as_tensor())))
            elif attr.type == ir.AttributeType.TENSORS:
                tensors.append(
                    (name, tuple(id(tensor) for tensor in attr.as_tensors()))
                )
        return (
            node.domain,
            node.op_type,
            node.overload,
            tuple(node.inputs),
            tuple(node.outputs),
            tuple(tensors),
            tuple(_subgraphs(node)),
        )

    def _verify_node_witnesses(self, graph: ir.Graph) -> None:
        for node in graph.all_nodes():
            expected = self._node_witnesses.get(node)
            if expected is not None and expected != self._node_witness(node):
                self._fail(
                    "lowered node source/use association changed", "graph assembly"
                )

    def _assembly_witness(self, graph: ir.Graph) -> tuple[object, tuple[object, ...]]:
        references: list[object] = [
            graph,
            *graph.inputs,
            *graph.outputs,
            *graph.initializers.values(),
        ]
        nodes = []
        for node in graph:
            references.extend([node, *node.outputs])
            references.extend(value for value in node.inputs if value is not None)
            tensors: list[object] = []
            for name, attr in node.attributes.items():
                if attr.is_ref():
                    continue
                if attr.type == ir.AttributeType.TENSOR:
                    tensor = attr.as_tensor()
                    references.append(tensor)
                    tensors.append((name, id(tensor)))
                elif attr.type == ir.AttributeType.TENSORS:
                    tensor_list = tuple(attr.as_tensors())
                    references.extend(tensor_list)
                    tensors.append((name, tuple(id(tensor) for tensor in tensor_list)))
            children = []
            for child in _subgraphs(node):
                child_signature, child_references = self._assembly_witness(child)
                children.append(child_signature)
                references.extend(child_references)
            nodes.append(
                (
                    id(node),
                    node.domain,
                    node.op_type,
                    node.overload,
                    tuple(
                        id(value) if value is not None else None
                        for value in node.inputs
                    ),
                    tuple(id(value) for value in node.outputs),
                    tuple(tensors),
                    tuple(children),
                )
            )
        signature = (
            id(graph),
            tuple(id(value) for value in graph.inputs),
            tuple(id(value) for value in graph.outputs),
            tuple(
                (id(value), id(value.const_value))
                for value in graph.initializers.values()
            ),
            tuple(nodes),
        )
        return signature, tuple(references)

    def _verify_scope_outputs(self, graph: ir.Graph) -> None:
        expected_outputs = self._scope_outputs.get(graph, ())
        if not expected_outputs:
            return
        live: set[ir.Value] = set()
        pending = list(graph.outputs)
        while pending:
            value = pending.pop()
            if value in live:
                continue
            live.add(value)
            producer = value.producer()
            if producer is not None and producer.graph is not None:
                pending.extend(item for item in producer.inputs if item is not None)
                for child in _subgraphs(producer):
                    pending.extend(child.outputs)
        for value in expected_outputs:
            expected = self._expected.get(value, frozenset())
            if expected and value not in live:
                self._fail(
                    "scope assembly replaced a source-dependent output",
                    "graph assembly",
                    min(expected),
                )

    def prepare_graph(self, graph: ir.Graph) -> None:
        if not self.has_annotations:
            return
        self._verify_scope_outputs(graph)
        self._verify_node_witnesses(graph)
        previous = self._prepared_graphs.get(graph)
        witness = self._assembly_witness(graph)
        if previous is not None and previous[0] != witness[0]:
            self._fail(
                "prepared graph changed before controlled assembly", "graph assembly"
            )
        self._prepared_graphs[graph] = witness

    def verify_assembled_graph(self, graph: ir.Graph) -> None:
        if not self.has_annotations:
            return
        expected = self._prepared_graphs.get(graph)
        if expected is None or expected[0] != self._assembly_witness(graph)[0]:
            self._fail(
                "graph assembly has no matching source/use witness", "graph assembly"
            )

    def register_graph_clone(self, original: ir.Graph, cloned: ir.Graph) -> None:
        if self.callback is None:
            return
        # Establish positional correspondence independently of source metadata.
        # Graph.clone promises unchanged topology and shared tensor objects.
        # Verify that promise before assigning provenance to any cloned value.
        self._verify_scope_outputs(original)
        self._verify_node_witnesses(original)
        if original in self._prepared_graphs:
            self.verify_assembled_graph(original)
        value_map: dict[ir.Value, ir.Value] = {}
        graph_pairs: list[tuple[ir.Graph, ir.Graph]] = []

        def collect(first: ir.Graph, second: ir.Graph) -> None:
            graph_pairs.append((first, second))
            if (
                len(first.inputs),
                len(first.outputs),
                len(first.initializers),
                len(first),
            ) != (
                len(second.inputs),
                len(second.outputs),
                len(second.initializers),
                len(second),
            ):
                self._fail("graph clone changed structural counts", "graph clone")
            for left, right in zip(first.inputs, second.inputs):
                value_map[left] = right
            for left, right in zip(
                first.initializers.values(), second.initializers.values()
            ):
                value_map[left] = right
                if left.const_value is not right.const_value:
                    self._fail(
                        "graph clone replaced initializer storage", "graph clone"
                    )
            for before, after in zip(first, second):
                if (
                    before.domain,
                    before.op_type,
                    before.overload,
                    len(before.inputs),
                    len(before.outputs),
                ) != (
                    after.domain,
                    after.op_type,
                    after.overload,
                    len(after.inputs),
                    len(after.outputs),
                ):
                    self._fail("graph clone changed node structure", "graph clone")
                for left, right in zip(before.outputs, after.outputs):
                    value_map[left] = right
                    if left.const_value is not right.const_value:
                        self._fail(
                            "graph clone replaced constant storage", "graph clone"
                        )
                if tuple(before.attributes) != tuple(after.attributes):
                    self._fail("graph clone changed attributes", "graph clone")
                for name, attribute in before.attributes.items():
                    copied = after.attributes[name]
                    if (
                        attribute.type != copied.type
                        or attribute.is_ref() != copied.is_ref()
                    ):
                        self._fail(
                            "graph clone changed an attribute representation",
                            "graph clone",
                        )
                    if attribute.is_ref():
                        if attribute.ref_attr_name != copied.ref_attr_name:
                            self._fail(
                                "graph clone changed an attribute reference",
                                "graph clone",
                            )
                    elif attribute.type == ir.AttributeType.TENSOR:
                        if attribute.as_tensor() is not copied.as_tensor():
                            self._fail(
                                "graph clone replaced tensor attribute storage",
                                "graph clone",
                            )
                    elif attribute.type == ir.AttributeType.TENSORS:
                        originals = tuple(attribute.as_tensors())
                        copies = tuple(copied.as_tensors())
                        if len(originals) != len(copies) or any(
                            left is not right for left, right in zip(originals, copies)
                        ):
                            self._fail(
                                "graph clone replaced tensor-list storage",
                                "graph clone",
                            )
                first_children, second_children = _subgraphs(before), _subgraphs(after)
                if len(first_children) != len(second_children):
                    self._fail("graph clone changed body count", "graph clone")
                for first_child, second_child in zip(first_children, second_children):
                    collect(first_child, second_child)

        collect(original, cloned)
        for first_graph, second_graph in graph_pairs:
            for left, right in zip(first_graph.outputs, second_graph.outputs):
                if value_map.get(left, left) is not right:
                    self._fail("graph clone redirected a graph output", "graph clone")
            for before, after in zip(first_graph, second_graph):
                for original_input, cloned_input in zip(before.inputs, after.inputs):
                    if (
                        value_map.get(original_input, original_input)
                        if original_input is not None
                        else None
                    ) is not cloned_input:
                        self._fail("graph clone redirected a node input", "graph clone")
        for first_graph, second_graph in graph_pairs:
            for before, after in zip(first_graph, second_graph):
                if before in self._node_witnesses:
                    self._node_witnesses[after] = self._node_witness(after)
        for first, second in value_map.items():
            if first in self._values:
                self._register(second, self._values[first], "graph clone")
            if first in self._expected:
                self._expected[second] = self._expected[first]
        for first_graph, second_graph in graph_pairs:
            outputs = self._scope_outputs.get(first_graph)
            if outputs is not None:
                self._scope_outputs[second_graph] = tuple(
                    value_map.get(value, value) for value in outputs
                )
            if self.has_annotations:
                self._prepared_graphs[second_graph] = self._assembly_witness(
                    second_graph
                )

    def dependencies(
        self, value: ir.Value, seen: set[ir.Value] | None = None
    ) -> frozenset[int]:
        if seen is None:
            seen = set()
        pending = [value]
        result: frozenset[int] = frozenset()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            result |= self._accepted(self._values.get(current, frozenset()))
            producer = current.producer()
            if producer is None or producer.graph is None:
                continue
            pending.extend(item for item in producer.inputs if item is not None)
            for graph in _subgraphs(producer):
                pending.extend(graph.outputs)
            function = self._functions.get(
                (producer.domain, producer.op_type, producer.overload)
            )
            if function is not None:
                pending.extend(function.graph.outputs)
        return result

    def variable_dependencies(
        self, ctx: Any, var: object, seen: set[int] | None = None
    ) -> frozenset[int]:
        if seen is None:
            seen = set()
        if id(var) in seen:
            return frozenset()
        seen.add(id(var))
        value = ctx.lookup_value_for_var(var)
        result = self.dependencies(value) if value is not None else frozenset()
        producer = ctx._const_folder._producer.get(id(var))
        if producer is not None:
            for item in producer[0].invars:
                result |= self.variable_dependencies(ctx, item, seen)
        return result

    def equation_inputs(self, ctx: Any, eqn: Any) -> frozenset[int]:
        if not self.has_annotations:
            return frozenset()
        result: frozenset[int] = frozenset()
        for var in eqn.invars:
            value = ctx.lookup_value_for_var(var)
            if value is not None:
                result |= self.dependencies(value)
        return result

    def prove_scope_outputs(self, ctx: Any, jaxpr: Any) -> None:
        if self.has_annotations:
            for var in jaxpr.outvars:
                value = ctx.require_value_for_var(var)
                self._expected[value] = self.dependencies(value)
            self._scope_outputs[ctx.builder.graph] = tuple(
                ctx.require_value_for_var(var) for var in jaxpr.outvars
            )
            self._verify_node_witnesses(ctx.builder.graph)
            for node in ctx.builder.graph.all_nodes():
                self._node_witnesses[node] = self._node_witness(node)

    def equation_outputs(self, ctx: Any, eqn: Any, expected: frozenset[int]) -> None:
        if not self.has_annotations:
            return
        actual: frozenset[int] = frozenset()
        outputs: list[ir.Value] = []
        for var in eqn.outvars:
            value = ctx.lookup_value_for_var(var)
            if value is not None:
                outputs.append(value)
                output_dependencies = self.dependencies(value)
                # An inlined nested scope proves its outputs independently,
                # including outputs independent of annotated operands. A new
                # unproved output must preserve all source input dependencies.
                output_expected = self._expected.get(value, expected)
                if not output_expected <= output_dependencies:
                    self._fail(
                        "lowering output lost a captured input dependency",
                        "lowering",
                        min(output_expected - output_dependencies),
                    )
                actual |= output_dependencies
        if not expected <= actual:
            self._fail(
                "lowering lost a captured input dependency",
                "lowering",
                min(expected - actual),
            )
        for value in outputs:
            self._expected[value] = self.dependencies(value)

    def _validate_tensor(
        self, tensor: ir.TensorProtocol | None, capture: int, stage: str
    ) -> None:
        if tensor is None:
            self._fail("stored representation has no tensor", stage, capture)
        assert tensor is not None
        reference = self.captures[capture]
        try:
            dtype, shape, payload = _tensor_storage(tensor)
        except Exception as exc:
            raise InitializerMetadataError(
                "preservation_failed",
                f"Initializer metadata storage cannot be decoded; stage={stage}, source={reference.handle}",
            ) from exc
        if (
            dtype != reference.dtype
            or payload != reference.payload
            or np.prod(shape, dtype=object) != np.prod(reference.shape, dtype=object)
        ):
            self._fail("reference dtype/order/element bits changed", stage, capture)
        if any(
            tensor.metadata_props.get(key) != value
            for key, value in reference.source.snapshot.items()
        ):
            self._fail("accepted entries changed or disappeared", stage, capture)

    def _live(
        self, model: ir.Model
    ) -> tuple[set[ir.Value], set[ir.Node], set[ir.Graph]]:
        self._functions.update(model.functions)
        values: set[ir.Value] = set()
        nodes: set[ir.Node] = set()
        graphs: set[ir.Graph] = set()

        pending_graphs = [model.graph]
        pending_values: list[ir.Value] = []
        while pending_graphs or pending_values:
            while pending_graphs:
                graph = pending_graphs.pop()
                if graph not in graphs:
                    graphs.add(graph)
                    pending_values.extend(graph.outputs)
            if not pending_values:
                continue
            value = pending_values.pop()
            if value in values:
                continue
            values.add(value)
            node = value.producer()
            if node is None or node.graph is None or node in nodes:
                continue
            nodes.add(node)
            pending_values.extend(item for item in node.inputs if item is not None)
            pending_graphs.extend(_subgraphs(node))
            function = self._functions.get((node.domain, node.op_type, node.overload))
            if function is not None:
                pending_graphs.append(function.graph)
        return values, nodes, graphs

    def _topology(self, model: ir.Model) -> object:
        values, nodes, graphs = self._live(model)
        node_items = []
        for node in nodes:
            tensors: list[object] = []
            for name, attr in node.attributes.items():
                if attr.is_ref():
                    continue
                if attr.type == ir.AttributeType.TENSOR:
                    tensors.append((name, id(attr.as_tensor())))
                elif attr.type == ir.AttributeType.TENSORS:
                    tensors.append(
                        (name, tuple(id(tensor) for tensor in attr.as_tensors()))
                    )
            node_items.append(
                (
                    id(node),
                    node.domain,
                    node.op_type,
                    node.overload,
                    tuple(
                        id(value) if value is not None else None
                        for value in node.inputs
                    ),
                    tuple(id(value) for value in node.outputs),
                    tuple(tensors),
                    tuple(id(graph) for graph in _subgraphs(node)),
                )
            )
        graph_items = tuple(
            sorted(
                (
                    id(graph),
                    tuple(id(value) for value in graph.outputs),
                    tuple(
                        (id(value), id(value.const_value))
                        for value in graph.initializers.values()
                        if value in values
                    ),
                )
                for graph in graphs
            )
        )
        return (tuple(sorted(node_items)), graph_items)

    def seal_ir(self, model: ir.Model) -> None:
        if self.has_annotations:
            self.validate_ir(model, stage="lowering completion")
            self._sealed_topology = self._topology(model)

    def validate_ir(self, model: ir.Model, *, stage: str = "final IR") -> None:
        if not self.has_annotations:
            return
        if (
            self._sealed_topology is not None
            and self._topology(model) != self._sealed_topology
        ):
            self._fail(
                "unregistered storage or source/use topology transformation", stage
            )
        values, nodes, graphs = self._live(model)
        live_captures: set[int] = set()
        all_graphs = [
            model.graph,
            *(function.graph for function in model.functions.values()),
        ]
        for graph in all_graphs:
            for value, tensor in _stored(graph):
                record = self._tensors.get(id(tensor))
                value_captures = (
                    self._accepted(self._values.get(value, frozenset()))
                    if value is not None
                    else frozenset()
                )
                if record is None:
                    if value_captures:
                        self._fail(
                            "registered source acquired an unregistered tensor",
                            stage,
                            min(value_captures),
                        )
                    continue
                captures = self._accepted(record[1])
                if not captures:
                    continue
                if value is None or value not in self._values:
                    self._fail(
                        "stored source has no registered use association",
                        stage,
                        next(iter(captures)),
                    )
                assert value is not None
                value_captures = self._accepted(self._values[value])
                if not captures <= value_captures:
                    self._fail(
                        "stored source use was reassigned", stage, next(iter(captures))
                    )
                for capture in captures:
                    self._validate_tensor(tensor, capture, stage)
                if value in values:
                    live_captures.update(captures)
        required = {
            index
            for index, capture in enumerate(self.captures)
            if capture.source.snapshot
        }
        missing = required - live_captures
        if missing:
            self._fail(
                "accepted capture has no live stored representation",
                stage,
                min(missing),
            )
        for value in values:
            accepted = self._accepted(self._values.get(value, frozenset()))
            if accepted:
                tensor_record = self._tensors.get(id(value.const_value))
                if tensor_record is None or not accepted <= tensor_record[1]:
                    self._fail(
                        "live source value lost its registered storage",
                        stage,
                        min(accepted),
                    )
            expected = self._expected.get(value, frozenset())
            actual = self.dependencies(value)
            if not expected <= actual:
                self._fail(
                    "recorded source/use association was lost",
                    stage,
                    min(expected - actual),
                )

    def snapshot_ir(self, model: ir.Model) -> object:
        self.validate_ir(model, stage="serialization input")
        return self._serialization_inventory(model) if self.has_annotations else None

    def validate_serialized_ir(
        self, model: ir.Model, snapshot: object, *, stage: str = "serialization"
    ) -> None:
        if snapshot is None:
            return
        try:
            inventory = self._serialization_inventory(model)
        except InitializerMetadataError:
            raise
        except Exception as exc:
            raise InitializerMetadataError(
                "preservation_failed",
                f"Initializer metadata serialized storage cannot be decoded; stage={stage}, scope/source=export",
            ) from exc
        if inventory != snapshot:
            self._fail("serialized storage or source/use topology differs", stage)

    def _serialization_inventory(self, model: ir.Model) -> object:
        # Positional correspondence to a fully verified IR model is the witness
        # here. Include every edge and tensor, so moving labels among equal
        # sources or redirecting uses cannot pass a metadata/count-only check.
        def graph_inventory(graph: ir.Graph) -> object:
            locations: dict[ir.Value, tuple[str, int, int]] = {}
            for index, value in enumerate(graph.inputs):
                locations[value] = ("input", index, 0)
            for index, value in enumerate(graph.initializers.values()):
                locations[value] = ("initializer", index, 0)
            for index, node in enumerate(graph):
                for output_index, value in enumerate(node.outputs):
                    locations[value] = ("node", index, output_index)

            def edge(value: ir.Value | None) -> object:
                if value is None:
                    return None
                # Lexical captures are resolved by the complete parent graph
                # naming as well as positional edges; names do not infer source
                # identity anywhere in the registry.
                return locations.get(value, ("outer", value.name))

            def tensor_inventory(tensor: ir.TensorProtocol) -> object:
                return (
                    _tensor_storage(tensor),
                    tuple(sorted(tensor.metadata_props.items())),
                )

            initializers = tuple(
                tensor_inventory(value.const_value)
                for value in graph.initializers.values()
                if value.const_value is not None
            )
            node_items = []
            for node in graph:
                attributes: list[object] = []
                for name, attr in node.attributes.items():
                    if attr.is_ref():
                        attributes.append((name, "ref", attr.ref_attr_name))
                    elif attr.type == ir.AttributeType.TENSOR:
                        attributes.append(
                            (name, "tensor", tensor_inventory(attr.as_tensor()))
                        )
                    elif attr.type == ir.AttributeType.TENSORS:
                        attributes.append(
                            (
                                name,
                                "tensors",
                                tuple(
                                    tensor_inventory(tensor)
                                    for tensor in attr.as_tensors()
                                ),
                            )
                        )
                    elif attr.type == ir.AttributeType.GRAPH:
                        attributes.append(
                            (name, "graph", graph_inventory(attr.as_graph()))
                        )
                    elif attr.type == ir.AttributeType.GRAPHS:
                        attributes.append(
                            (
                                name,
                                "graphs",
                                tuple(
                                    graph_inventory(child) for child in attr.as_graphs()
                                ),
                            )
                        )
                    elif attr.type == ir.AttributeType.FLOAT:
                        attributes.append(
                            (
                                name,
                                "float32",
                                np.asarray(attr.as_float(), dtype=np.float32).tobytes(),
                            )
                        )
                    elif attr.type == ir.AttributeType.FLOATS:
                        attributes.append(
                            (
                                name,
                                "float32s",
                                np.asarray(
                                    attr.as_floats(), dtype=np.float32
                                ).tobytes(),
                            )
                        )
                    elif attr.type == ir.AttributeType.INT:
                        attributes.append((name, "int", attr.as_int()))
                    elif attr.type == ir.AttributeType.INTS:
                        attributes.append((name, "ints", tuple(attr.as_ints())))
                    elif attr.type == ir.AttributeType.STRING:
                        string = attr.value
                        attributes.append(
                            (
                                name,
                                "string",
                                (
                                    string
                                    if isinstance(string, bytes)
                                    else str(string).encode("utf-8")
                                ),
                            )
                        )
                    elif attr.type == ir.AttributeType.STRINGS:
                        attributes.append(
                            (
                                name,
                                "strings",
                                tuple(
                                    string.encode("utf-8")
                                    for string in attr.as_strings()
                                ),
                            )
                        )
                    else:
                        attributes.append((name, int(attr.type), repr(attr.value)))
                node_items.append(
                    (
                        node.domain,
                        node.op_type,
                        node.overload,
                        tuple(edge(value) for value in node.inputs),
                        len(node.outputs),
                        tuple(attributes),
                    )
                )
            return (
                len(graph.inputs),
                initializers,
                tuple(node_items),
                tuple(edge(value) for value in graph.outputs),
            )

        return (
            graph_inventory(model.graph),
            tuple(
                (key, graph_inventory(function.graph))
                for key, function in model.functions.items()
            ),
        )
