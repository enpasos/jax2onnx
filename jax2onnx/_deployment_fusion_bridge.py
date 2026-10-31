# jax2onnx/_deployment_fusion_bridge.py

"""Copied ONNX artifact adapter for exact private deployment fusion proofs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import onnx
import onnx.inliner

from jax2onnx._deployment_fusions import (
    Attribute,
    FusionUnfinished,
    Graph,
    Node,
    TensorData,
    ValueType,
    expand_proven_fusions,
    require,
    refine_type,
    _FP32_POST_MUL_SINGLE_PANEL,
)
from jax2onnx.converter.source_deployment import (
    DEPLOYMENT_METADATA_KEY,
    deployment_requirements_from_json,
    reviewed_fp32_contract,
)


def contains_fusions(model: onnx.ModelProto) -> bool:
    def body(owner: Any) -> bool:
        for node in owner.node:
            if node.op_type == "Gemm" or (
                node.domain == "com.microsoft" and node.op_type == "QuickGelu"
            ):
                return True
            if (
                node.domain == "com.microsoft"
                and node.op_type == "FusedMatMul"
                and any(
                    (
                        attribute.name in {"transBatchA", "transBatchB"}
                        and attribute.i != 0
                    )
                    or (
                        attribute.name == "alpha"
                        and (
                            attribute.type != onnx.AttributeProto.FLOAT
                            or bool(attribute.ref_attr_name)
                            or attribute.f != 1.0
                        )
                    )
                    for attribute in node.attribute
                )
            ):
                return True
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH and body(attribute.g):
                    return True
                if attribute.type == onnx.AttributeProto.GRAPHS and any(
                    body(child) for child in attribute.graphs
                ):
                    return True
        return False

    return body(model.graph) or any(body(function) for function in model.functions)


def _reviewed_post_mul_contract(original: onnx.ModelProto) -> str | None:
    declarations = [
        item.value
        for item in original.metadata_props
        if item.key == DEPLOYMENT_METADATA_KEY
    ]
    require(len(declarations) <= 1, "ambiguous source deployment declaration")
    required = deployment_requirements_from_json(
        declarations[0] if declarations else None
    )
    if required is None:
        return None
    target = required.target
    if reviewed_fp32_contract(
        target
    ) == "ort-cpu-x86-fp32-sgemm-v1" and target.runtime_build in {
        "ORT Build Info: git-branch=HEAD, git-commit-id=402979a, fp8-kv-cache=1, build type=Release",
        "ORT Build Info: git-branch=HEAD, git-commit-id=470ae16099, build type=Release",
    }:
        # This is a declared structural capability. Actual runtime/provider/
        # library/hardware/session observation is still an independent gate.
        return _FP32_POST_MUL_SINGLE_PANEL
    return None


def _tensor(value: onnx.TensorProto) -> TensorData:
    array = onnx.numpy_helper.to_array(value)
    if array.dtype.hasobject:
        # String/object arrays must never hash process-specific pointer storage.
        # They remain opaque to the floating fusion and layout evaluators.
        chunks = []
        for item in array.reshape(-1):
            require(isinstance(item, bytes), "opaque tensor storage is not bytes")
            chunks.append(len(item).to_bytes(8, "little") + item)
        data = b"".join(chunks)
    else:
        if array.dtype.byteorder == ">" or (
            array.dtype.byteorder == "=" and not np.little_endian
        ):
            array = array.byteswap().view(array.dtype.newbyteorder("<"))
        data = array.tobytes(order="C")
    return TensorData(value.data_type, tuple(value.dims), data, deepcopy(value))


def _type(value: onnx.ValueInfoProto) -> ValueType:
    kind = value.type.WhichOneof("value")
    if kind not in (None, "tensor_type"):
        # A known non-tensor is never refined into a floating matrix.
        return ValueType(-1, None, deepcopy(value))
    tensor = value.type.tensor_type
    shape = (
        tuple(
            (
                dimension.dim_value
                if dimension.HasField("dim_value")
                else (
                    dimension.dim_param or None
                    if dimension.HasField("dim_param")
                    else None
                )
            )
            for dimension in tensor.shape.dim
        )
        if tensor.HasField("shape")
        else None
    )
    return ValueType(tensor.elem_type or None, shape, deepcopy(value))


def _standard_gemm_defaults(
    node: onnx.NodeProto,
    attributes: dict[str, Attribute],
    imports: Mapping[str, int],
) -> None:
    if node.op_type != "Gemm" or node.domain not in ("", "ai.onnx"):
        return
    version = imports.get(node.domain, imports.get(""))
    require(version is not None and version > 0, "Gemm standard opset is absent")
    assert version is not None
    schema = onnx.defs.get_schema("Gemm", version, "")
    for name, kind in (
        ("alpha", "FLOAT"),
        ("beta", "FLOAT"),
        ("transA", "INT"),
        ("transB", "INT"),
    ):
        require(
            name in schema.attributes, "installed Gemm schema lacks required attribute"
        )
        default = schema.attributes[name].default_value
        expected_type = getattr(onnx.AttributeProto, kind)
        require(
            default.type == expected_type, "installed Gemm default schema type differs"
        )
        if name not in attributes:
            attributes[name] = Attribute(kind, onnx.helper.get_attribute_value(default))


def _graph(owner: onnx.GraphProto, imports: Mapping[str, int]) -> Graph:
    types: dict[str, ValueType] = {}
    for value in (*owner.input, *owner.output, *owner.value_info):
        current = _type(value)
        if value.name in types:
            current = refine_type(types[value.name], current)
        types[value.name] = current
    initializers = {value.name: _tensor(value) for value in owner.initializer}
    require(len(initializers) == len(owner.initializer), "duplicate initializer name")
    nodes = []
    for node in owner.node:
        if node.domain == "com.microsoft" and node.op_type == "FusedMatMul":
            require(
                imports.get("com.microsoft") == 1, "unproved FusedMatMul domain opset"
            )
        attributes = {}
        for attribute in node.attribute:
            require(attribute.name not in attributes, "duplicate operator attribute")
            kind = onnx.AttributeProto.AttributeType.Name(attribute.type)
            if attribute.type == onnx.AttributeProto.GRAPH:
                attribute_value: Any = _graph(attribute.g, imports)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                attribute_value = tuple(
                    _graph(child, imports) for child in attribute.graphs
                )
            elif attribute.type == onnx.AttributeProto.TENSOR:
                attribute_value = _tensor(attribute.t)
            elif attribute.ref_attr_name:
                attribute_value = None
            else:
                attribute_value = onnx.helper.get_attribute_value(attribute)
            attributes[attribute.name] = Attribute(
                kind, attribute_value, attribute.ref_attr_name
            )
        _standard_gemm_defaults(node, attributes, imports)
        nodes.append(
            Node(
                node.name,
                node.domain,
                node.op_type,
                node.overload,
                tuple(node.input),
                tuple(node.output),
                attributes,
                deepcopy(node),
            )
        )
    return Graph(
        owner.name,
        tuple(value.name for value in owner.input),
        tuple(value.name for value in owner.output),
        types,
        initializers,
        nodes,
        deepcopy(owner),
    )


def _write_type(value: onnx.ValueInfoProto, observed: ValueType) -> None:
    if observed.dtype is None or observed.dtype < 0:
        return
    value.type.tensor_type.elem_type = observed.dtype
    if observed.shape is not None:
        shape = value.type.tensor_type.shape
        shape.ClearField("dim")
        shape.SetInParent()
        for size in observed.shape:
            dimension = shape.dim.add()
            if type(size) is int:
                dimension.dim_value = size
            elif isinstance(size, str):
                dimension.dim_param = size
            else:
                require(size is None, "unproved dimension descriptor")


def _materialize(graph: Graph) -> onnx.GraphProto:
    body: onnx.GraphProto = deepcopy(graph.payload)
    del body.node[:]
    for node in graph.nodes:
        proto = deepcopy(node.payload)
        proto.name, proto.domain, proto.op_type, proto.overload = (
            node.name,
            node.domain,
            node.op,
            node.overload,
        )
        del proto.input[:]
        proto.input.extend(node.inputs)
        del proto.output[:]
        proto.output.extend(node.outputs)
        for attribute in proto.attribute:
            value = node.attributes[attribute.name]
            if value.kind == "GRAPH":
                attribute.g.CopyFrom(_materialize(value.value))
            elif value.kind == "GRAPHS":
                del attribute.graphs[:]
                attribute.graphs.extend(_materialize(child) for child in value.value)
        body.node.append(proto)
    del body.initializer[:]
    for name, tensor in graph.initializers.items():
        proto = deepcopy(tensor.payload)
        proto.name = name
        body.initializer.append(proto)
    infos: dict[str, list[onnx.ValueInfoProto]] = {}
    for value in (*body.input, *body.output, *body.value_info):
        infos.setdefault(value.name, []).append(value)
    for name, observed in graph.types.items():
        if name in infos:
            for value in infos[name]:
                _write_type(value, observed)
        elif observed.dtype is not None and observed.dtype > 0:
            value = body.value_info.add()
            value.name = name
            _write_type(value, observed)
    return body


def _non_tensor_annotation(value: onnx.ValueInfoProto) -> onnx.ValueInfoProto:
    result = deepcopy(value)
    result.type.ClearField("tensor_type")
    if result.type.ByteSize() == 0:
        result.ClearField("type")
    return result


def _require_type_refresh_structure(
    before: onnx.ModelProto, inferred: onnx.ModelProto
) -> None:
    """Only graph tensor annotations may differ, including lexical children."""

    def normalize(owner: onnx.GraphProto, returned: onnx.GraphProto) -> None:
        existing = {value.name: value for value in owner.value_info}
        require(len(existing) == len(owner.value_info), "ambiguous current value_info")
        observed = {value.name: value for value in returned.value_info}
        require(
            len(observed) == len(returned.value_info), "ambiguous inferred value_info"
        )
        names = (
            {value.name for value in (*owner.input, *owner.output, *owner.initializer)}
            | {value.values.name for value in owner.sparse_initializer}
            | {
                name
                for node in owner.node
                for name in (*node.input, *node.output)
                if name
            }
            | set(existing)
        )
        for name, value in observed.items():
            require(
                bool(name) and name in names,
                "inferred annotation has no current SSA binding",
            )
            declared_info = existing.get(name, onnx.ValueInfoProto(name=name))
            require(
                _non_tensor_annotation(declared_info).SerializeToString(
                    deterministic=True
                )
                == _non_tensor_annotation(value).SerializeToString(deterministic=True),
                "type refresh changed non-type value_info metadata",
            )
        for graph in (owner, returned):
            del graph.value_info[:]
            for values in (graph.input, graph.output):
                require(
                    len({value.name for value in values}) == len(values),
                    "ambiguous graph interface annotation",
                )
                for value in values:
                    value.CopyFrom(_non_tensor_annotation(value))
        require(
            len(owner.node) == len(returned.node), "type refresh changed node count"
        )
        for left, right in zip(owner.node, returned.node, strict=True):
            require(
                len(left.attribute) == len(right.attribute),
                "type refresh changed attributes",
            )
            for original, actual in zip(left.attribute, right.attribute, strict=True):
                require(
                    original.name == actual.name and original.type == actual.type,
                    "type refresh changed attribute ownership/schema",
                )
                if original.type == onnx.AttributeProto.GRAPH:
                    normalize(original.g, actual.g)
                elif original.type == onnx.AttributeProto.GRAPHS:
                    require(
                        len(original.graphs) == len(actual.graphs),
                        "type refresh changed graph count",
                    )
                    for a, b in zip(original.graphs, actual.graphs, strict=True):
                        normalize(a, b)

    original, actual = deepcopy(before), deepcopy(inferred)
    normalize(original.graph, actual.graph)
    require(
        original.SerializeToString(deterministic=True)
        == actual.SerializeToString(deterministic=True),
        "type refresh changed non-type model structure or payload",
    )


def _merge_refreshed_types(current: Graph, inferred: onnx.GraphProto) -> Graph:
    result = deepcopy(current)

    def local_names(graph: Graph) -> set[str]:
        return (
            set(graph.inputs)
            | set(graph.initializers)
            | {name for node in graph.nodes for name in node.outputs}
        )

    def known_type(graph: Graph, name: str) -> ValueType:
        known = graph.types.get(name, ValueType(None, None))
        if name in graph.initializers and name not in graph.inputs:
            tensor = graph.initializers[name]
            known = refine_type(known, ValueType(tensor.dtype, tensor.shape))
        return known

    def merge(graph: Graph, owner: onnx.GraphProto, parents: tuple[Graph, ...]) -> None:
        local = local_names(graph)
        for value in (*owner.input, *owner.output, *owner.value_info):
            known = known_type(graph, value.name)
            defining_parent: Graph | None = None
            if value.name not in local:
                for parent in reversed(parents):
                    if value.name in local_names(parent):
                        known = refine_type(known_type(parent, value.name), known)
                        defining_parent = parent
                        break
            refined = refine_type(known, _type(value))
            if defining_parent is not None:
                defining_parent.types[value.name] = refine_type(
                    known_type(defining_parent, value.name), refined
                )
            graph.types[value.name] = refined
        for node, returned in zip(graph.nodes, owner.node, strict=True):
            for attribute in returned.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    merge(
                        node.attributes[attribute.name].value,
                        attribute.g,
                        (*parents, graph),
                    )
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child, actual in zip(
                        node.attributes[attribute.name].value,
                        attribute.graphs,
                        strict=True,
                    ):
                        merge(child, actual, (*parents, graph))

    merge(result, inferred, ())
    return result


_SymbolOccurrence = tuple[Graph, str, ValueType, int]


def _descriptor_graphs(graph: Graph) -> tuple[Graph, ...]:
    result = [graph]
    for node in graph.nodes:
        for attribute in node.attributes.values():
            if attribute.kind == "GRAPH":
                result.extend(_descriptor_graphs(attribute.value))
            elif attribute.kind == "GRAPHS":
                for child in attribute.value:
                    result.extend(_descriptor_graphs(child))
    return tuple(result)


def _dimension_symbols(*models: onnx.ModelProto) -> set[str]:
    """Protect declarations in every raw type, function and lexical body."""
    result: set[str] = set()

    def visit(message: Any) -> None:
        if message.DESCRIPTOR.full_name == "onnx.TensorShapeProto.Dimension":
            if message.HasField("dim_param") and message.dim_param:
                result.add(message.dim_param)
        for field, value in message.ListFields():
            if field.message_type is not None:
                if field.is_repeated:
                    for item in value:
                        visit(item)
                else:
                    visit(value)

    for model in models:
        visit(model)
    return result


def _symbol_occurrences(graph: Graph) -> dict[str, tuple[_SymbolOccurrence, ...]]:
    result: dict[str, list[_SymbolOccurrence]] = {}
    for owner in _descriptor_graphs(graph):
        for name, known in owner.types.items():
            for axis, size in enumerate(known.shape or ()):
                if isinstance(size, str):
                    result.setdefault(size, []).append((owner, name, known, axis))
    return {name: tuple(values) for name, values in result.items()}


def _returned_types(
    graph: Graph, inferred: onnx.GraphProto
) -> dict[tuple[int, str], ValueType]:
    result: dict[tuple[int, str], ValueType] = {}

    def visit(current: Graph, owner: onnx.GraphProto) -> None:
        for value in (*owner.input, *owner.output, *owner.value_info):
            key = (id(current), value.name)
            observed = _type(value)
            result[key] = refine_type(result.get(key, ValueType(None, None)), observed)
        for node, returned in zip(current.nodes, owner.node, strict=True):
            for attribute in returned.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit(node.attributes[attribute.name].value, attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child, actual in zip(
                        node.attributes[attribute.name].value,
                        attribute.graphs,
                        strict=True,
                    ):
                        visit(child, actual)

    visit(graph, inferred)
    return result


class _TargetInferenceSymbols:
    """Fresh target variables; strong references retain exact occurrence lineage.

    The proof engine mutates the Graph returned by a refresh until the next
    callback. Replacing a ValueType during source restoration loses permission,
    even if its spelling and shape happen to equal a previously inferred type.
    Verified initial inference can grant single-use bootstrap permission on the
    pristine engine copy, before any source annotation is restored into it.
    """

    def __init__(self, protected: set[str]) -> None:
        self._protected = set(protected)
        self._active: dict[str, tuple[int, tuple[_SymbolOccurrence, ...]]] = {}
        self._next_generation = 0
        self._initial_graph: Graph | None = None

    def initialize(self, current: Graph) -> None:
        expected = self._initial_graph
        require(expected is not None, "initial target provenance is not verified")
        assert expected is not None
        require(
            current is not expected
            and current == expected
            and _materialize(current).SerializeToString(deterministic=True)
            == _materialize(expected).SerializeToString(deterministic=True),
            "initial target provenance has no exact pristine copy",
        )
        require(
            not self._active and self._next_generation == 0,
            "initial target provenance was already used",
        )
        active: dict[str, tuple[int, tuple[_SymbolOccurrence, ...]]] = {}
        generation = 0
        for symbol, occurrences in _symbol_occurrences(current).items():
            if symbol not in self._protected:
                active[symbol] = (generation, occurrences)
                generation += 1
        self._active, self._next_generation, self._initial_graph = (
            active,
            generation,
            None,
        )

    def merge(self, current: Graph, inferred: onnx.GraphProto) -> Graph:
        existing = _symbol_occurrences(current)
        returned = _returned_types(current, inferred)
        protected = set(self._protected)
        generations: dict[str, int] = {}
        bindings: dict[str, int] = {}
        for symbol, occurrences in existing.items():
            previous = self._active.get(symbol)
            if (
                previous is None
                or len(occurrences) != len(previous[1])
                or not all(
                    any(
                        owner is prior_owner
                        and name == prior_name
                        and known is prior_known
                        and axis == prior_axis
                        for prior_owner, prior_name, prior_known, prior_axis in previous[
                            1
                        ]
                    )
                    for owner, name, known, axis in occurrences
                )
            ):
                protected.add(symbol)
            if symbol in protected:
                continue
            assert previous is not None
            generations[symbol] = previous[0]
            integers = {
                observed.shape[axis]
                for owner, name, _, axis in occurrences
                if (observed := returned.get((id(owner), name))) is not None
                and observed.shape is not None
                and axis < len(observed.shape)
                and type(observed.shape[axis]) is int
            }
            if not integers:
                continue
            require(
                len(integers) == 1, "fresh inference variable has conflicting values"
            )
            integer = next(iter(integers))
            assert type(integer) is int
            for owner, name, known, axis in occurrences:
                observed = returned.get((id(owner), name))
                require(
                    observed is not None
                    and observed.dtype == known.dtype
                    and observed.shape is not None
                    and known.shape is not None
                    and len(observed.shape) == len(known.shape)
                    and observed.shape[axis] == integer,
                    "fresh inference variable specialization is partial or inconsistent",
                )
            bindings[symbol] = integer

        candidate = current
        if bindings:
            candidate = deepcopy(current)
            for owner in _descriptor_graphs(candidate):
                for name, known in tuple(owner.types.items()):
                    if known.shape is not None:
                        shape = tuple(
                            bindings.get(size, size) if isinstance(size, str) else size
                            for size in known.shape
                        )
                        if shape != known.shape:
                            owner.types[name] = ValueType(
                                known.dtype, shape, known.payload
                            )
        result = _merge_refreshed_types(candidate, inferred)
        next_active: dict[str, tuple[int, tuple[_SymbolOccurrence, ...]]] = {}
        next_generation = self._next_generation
        for symbol, occurrences in _symbol_occurrences(result).items():
            if symbol in protected:
                continue
            if symbol in existing:
                # A same-round new use of a just-bound spelling cannot inherit
                # its old binding or origin. Keep it protected thereafter.
                if symbol in bindings or symbol not in generations:
                    protected.add(symbol)
                    continue
                generation = generations[symbol]
            else:
                generation = next_generation
                next_generation += 1
            next_active[symbol] = (generation, occurrences)
        self._protected, self._active, self._next_generation = (
            protected,
            next_active,
            next_generation,
        )
        return result


def _initial_target_types(
    before: onnx.ModelProto,
    inferred: onnx.ModelProto,
    imports: Mapping[str, int],
    protected: set[str],
) -> tuple[Graph, _TargetInferenceSymbols]:
    _require_type_refresh_structure(before, inferred)
    target = _merge_refreshed_types(_graph(before.graph, imports), inferred.graph)
    provenance = _TargetInferenceSymbols(protected | _dimension_symbols(before))
    provenance._initial_graph = target
    return target, provenance


def _refresh_virtual_types(
    current: Graph,
    envelope: onnx.ModelProto,
    provenance: _TargetInferenceSymbols | None = None,
) -> Graph:
    model = deepcopy(envelope)
    model.graph.CopyFrom(_materialize(current))
    before = deepcopy(model)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    _require_type_refresh_structure(before, inferred)
    if provenance is None:
        return _merge_refreshed_types(current, inferred.graph)
    return provenance.merge(current, inferred.graph)


def prove_deployment_fusions(
    original: onnx.ModelProto,
    optimized: onnx.ModelProto,
    roles: dict[str, tuple[str, str, str]],
) -> tuple[onnx.ModelProto, tuple[str, ...]]:
    """Expand proved fusions in a private copy; actual artifact bytes stay exact."""
    before = (original.SerializeToString(), optimized.SerializeToString())
    try:
        # Validate original declarations before inlining can discard them.
        onnx.shape_inference.infer_shapes(
            deepcopy(original), strict_mode=True, data_prop=True
        )
        onnx.shape_inference.infer_shapes(
            deepcopy(optimized), strict_mode=True, data_prop=True
        )
        source = onnx.shape_inference.infer_shapes(
            onnx.inliner.inline_local_functions(deepcopy(original)),
            strict_mode=True,
            data_prop=True,
        )
        inlined_target = onnx.inliner.inline_local_functions(deepcopy(optimized))
        initial_target = deepcopy(inlined_target)
        target = onnx.shape_inference.infer_shapes(
            inlined_target, strict_mode=True, data_prop=True
        )
        require(
            inlined_target.SerializeToString(deterministic=True)
            == initial_target.SerializeToString(deterministic=True),
            "initial target inference changed its detached input",
        )
        imports = {item.domain: item.version for item in target.opset_import}
        require(
            len(imports) == len(target.opset_import),
            "ambiguous optimized opset imports",
        )
        source_imports = {item.domain: item.version for item in source.opset_import}
        require(
            len(source_imports) == len(source.opset_import),
            "ambiguous source opset imports",
        )
        target_graph, provenance = _initial_target_types(
            initial_target,
            target,
            imports,
            _dimension_symbols(original, optimized, initial_target),
        )
        virtual, kinds = expand_proven_fusions(
            _graph(source.graph, source_imports),
            target_graph,
            roles,
            fp32_post_mul_contract=_reviewed_post_mul_contract(original),
            initialize_types=provenance.initialize,
            refresh_types=lambda current: _refresh_virtual_types(
                current, target, provenance
            ),
        )
        result = deepcopy(target)
        result.graph.CopyFrom(_materialize(virtual))
        return result, kinds
    finally:
        if before != (original.SerializeToString(), optimized.SerializeToString()):
            raise FusionUnfinished("deployment inspection changed an actual artifact")
