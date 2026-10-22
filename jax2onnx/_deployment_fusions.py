# jax2onnx/_deployment_fusions.py

"""Private exact virtual fusion proof; descriptors contain no framework objects.

Runtime adapters alone supply artifact tensor storage. This module performs
structural and byte/index correspondence; it grants no numerical acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
from itertools import product
import math
import re
import struct
from typing import Any, Callable


class FusionUnfinished(RuntimeError):
    """An actual optimizer transformation still lacks an exact proof."""


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise FusionUnfinished(detail)


@dataclass(frozen=True)
class ValueType:
    dtype: int | None
    # None is absent/unknown rank. () is an explicitly present scalar shape.
    shape: tuple[int | str | None, ...] | None
    payload: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class TensorData:
    dtype: int
    shape: tuple[int, ...]
    data: bytes = field(repr=False)
    payload: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        require(type(self.dtype) is int and self.dtype > 0, "constant dtype is invalid")
        require(
            all(type(size) is int and size >= 0 for size in self.shape),
            "constant has nonconcrete dimensions",
        )
        if self.dtype in self.storage_sizes():
            require(
                len(self.data) == self.count * self.itemsize,
                "constant storage extent differs from its declared shape",
            )

    @staticmethod
    def storage_sizes() -> dict[int, int]:
        return {
            1: 4,
            2: 1,
            3: 1,
            4: 2,
            5: 2,
            6: 4,
            7: 8,
            9: 1,
            10: 2,
            11: 8,
            12: 4,
            13: 8,
            14: 8,
            15: 16,
            16: 2,
            17: 1,
            18: 1,
            19: 1,
            20: 1,
            24: 1,
        }

    @property
    def itemsize(self) -> int:
        require(
            self.dtype in self.storage_sizes(),
            "constant layout storage dtype remains unproved",
        )
        return self.storage_sizes()[self.dtype]

    @property
    def count(self) -> int:
        return math.prod(self.shape)

    @property
    def key(self) -> tuple[Any, ...]:
        return self.dtype, self.shape, hashlib.sha256(self.data).hexdigest()

    def integers(self) -> tuple[int, ...]:
        require(
            self.dtype in {6, 7} and len(self.shape) <= 1,
            "shape carrier is not an exact integer scalar/vector",
        )
        return tuple(
            value[0]
            for value in struct.iter_unpack(
                "<i" if self.dtype == 6 else "<q", self.data
            )
        )

    def reshape(self, dimensions: tuple[int, ...], allowzero: int = 0) -> TensorData:
        require(allowzero in (0, 1), "Reshape allowzero is not Boolean")
        shape = list(dimensions)
        require(
            all(type(size) is int and size >= -1 for size in shape),
            "invalid Reshape dimensions",
        )
        if not allowzero:
            for index, size in enumerate(shape):
                if size == 0:
                    require(
                        index < len(self.shape),
                        "Reshape zero copies a missing input dimension",
                    )
                    shape[index] = self.shape[index]
        require(shape.count(-1) <= 1, "ambiguous Reshape inference")
        if -1 in shape:
            known = math.prod(size for size in shape if size != -1)
            require(
                known > 0 and self.count % known == 0,
                "unproved Reshape inferred extent",
            )
            shape[shape.index(-1)] = self.count // known
        require(math.prod(shape) == self.count, "Reshape changes storage extent")
        return TensorData(self.dtype, tuple(shape), self.data)

    def broadcast(self, shape: tuple[int, ...]) -> TensorData:
        require(len(shape) >= len(self.shape), "broadcast lowers rank")
        padded = (1,) * (len(shape) - len(self.shape)) + self.shape
        require(
            all(
                before == after or before == 1
                for before, after in zip(padded, shape, strict=True)
            ),
            "constant broadcast index map differs",
        )
        chunks = []
        for coordinate in product(*(range(size) for size in shape)):
            index = 0
            for before, position in zip(padded, coordinate, strict=True):
                index = index * before + (0 if before == 1 else position)
            chunks.append(
                self.data[index * self.itemsize : (index + 1) * self.itemsize]
            )
        return TensorData(self.dtype, shape, b"".join(chunks))

    def transpose(self, permutation: tuple[int, ...]) -> TensorData:
        require(
            sorted(permutation) == list(range(len(self.shape))),
            "invalid constant Transpose permutation",
        )
        shape = tuple(self.shape[index] for index in permutation)
        chunks = []
        for output_coordinate in product(*(range(size) for size in shape)):
            coordinate = [0] * len(shape)
            for axis, source_axis in enumerate(permutation):
                coordinate[source_axis] = output_coordinate[axis]
            index = 0
            for size, position in zip(self.shape, coordinate, strict=True):
                index = index * size + position
            chunks.append(
                self.data[index * self.itemsize : (index + 1) * self.itemsize]
            )
        return TensorData(self.dtype, shape, b"".join(chunks))

    def slice(
        self,
        starts: tuple[int, ...],
        ends: tuple[int, ...],
        axes: tuple[int, ...],
        steps: tuple[int, ...],
    ) -> TensorData:
        itemsize = self.itemsize
        ranges = slice_index_ranges(self.shape, starts, ends, axes, steps)
        shape = tuple(len(indices) for indices in ranges)
        chunks = []
        for coordinate in product(*ranges):
            index = 0
            for size, position in zip(self.shape, coordinate, strict=True):
                index = index * size + position
            chunks.append(self.data[index * itemsize : (index + 1) * itemsize])
        return TensorData(self.dtype, shape, b"".join(chunks))


@dataclass(frozen=True)
class Attribute:
    kind: str
    value: Any
    reference: str = ""


@dataclass
class Node:
    name: str
    domain: str
    op: str
    overload: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: dict[str, Attribute]
    payload: Any = field(default=None, repr=False, compare=False)


@dataclass
class Graph:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    types: dict[str, ValueType]
    initializers: dict[str, TensorData]
    nodes: list[Node]
    payload: Any = field(default=None, repr=False, compare=False)


STANDARD = {"", "ai.onnx"}
LAYOUT = {"Identity", "Transpose", "Reshape"}
_FP32_POST_MUL_SINGLE_PANEL = "ort-cpu-x86-fp32-post-mul-single-k-panel-v1"


def slice_index_ranges(
    shape: tuple[int | str | None, ...],
    starts: tuple[int, ...],
    ends: tuple[int, ...],
    axes: tuple[int, ...],
    steps: tuple[int, ...],
) -> tuple[range, ...]:
    """Exact ONNX Slice indices; no tensor element is numerically evaluated."""
    require(
        all(type(size) is int and size >= 0 for size in shape),
        "Slice input bounds remain unproved",
    )
    require(
        len(starts) == len(ends) == len(axes) == len(steps)
        and all(
            type(value) is int
            for values in (starts, ends, axes, steps)
            for value in values
        ),
        "Slice control index lengths/types differ",
    )
    normalized = tuple(axis + len(shape) if axis < 0 else axis for axis in axes)
    require(
        len(set(normalized)) == len(normalized)
        and all(0 <= axis < len(shape) for axis in normalized),
        "Slice axes are invalid",
    )
    require(all(step != 0 for step in steps), "Slice step is zero")
    ranges = []
    for size in shape:
        assert type(size) is int
        ranges.append(range(size))
    for start, end, axis, step in zip(starts, ends, normalized, steps, strict=True):
        size = shape[axis]
        assert type(size) is int
        if size == 0:
            ranges[axis] = range(0)
            continue
        start = start + size if start < 0 else start
        end = end + size if end < 0 else end
        if step > 0:
            start = min(max(start, 0), size)
            end = min(max(end, 0), size)
        else:
            # ONNX clips negative-step starts to [0, dim-1], unlike Python's
            # slice.indices for extreme negative starts. Ends clip to -1.
            start = min(max(start, 0), size - 1)
            end = min(max(end, -1), size - 1)
        ranges[axis] = range(start, end, step)
    return tuple(ranges)


def broadcast_shape(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    result = []
    for index in range(max(len(left), len(right))):
        a = left[-1 - index] if index < len(left) else 1
        b = right[-1 - index] if index < len(right) else 1
        require(a == b or a == 1 or b == 1, "bias broadcast dimensions differ")
        result.append(b if a == 1 else a)
    return tuple(reversed(result))


def children(node: Node) -> list[tuple[str, int, Graph]]:
    result = []
    for name, attribute in node.attributes.items():
        if attribute.kind == "GRAPH":
            result.append((name, 0, attribute.value))
        elif attribute.kind == "GRAPHS":
            result.extend(
                (name, index, graph) for index, graph in enumerate(attribute.value)
            )
    return result


def free_names(graph: Graph) -> set[str]:
    local = (
        set(graph.inputs)
        | set(graph.initializers)
        | {name for node in graph.nodes for name in node.outputs}
    )
    used = {name for node in graph.nodes for name in node.inputs if name} | set(
        graph.outputs
    )
    for node in graph.nodes:
        for _, _, child in children(node):
            used.update(free_names(child))
    return used - local


class Registry:
    def __init__(self, roles: dict[str, tuple[str, str, str]]) -> None:
        self.roles = roles

    def tag(self, node: Node, *, fusion: bool = False) -> str:
        found: set[str] = set()
        for tag in re.findall(r"jax2onnx_src_[0-9]{6}", node.name):
            owner = self.roles.get(tag)
            if owner is None:
                continue
            source_op, source_domain, _ = owner
            standard = source_domain in STANDARD and node.domain in STANDARD
            exact = source_op == node.op and (source_domain == node.domain or standard)
            fused_matrix = (
                source_op == "MatMul"
                and source_domain in STANDARD
                and (
                    (node.op == "FusedMatMul" and node.domain == "com.microsoft")
                    or (fusion and node.op == "Gemm" and node.domain in STANDARD)
                )
            )
            fused_gelu = (
                fusion
                and source_op == "Mul"
                and source_domain in STANDARD
                and node.op == "QuickGelu"
                and node.domain == "com.microsoft"
            )
            if exact or fused_matrix or fused_gelu:
                found.add(tag)
        require(len(found) == 1, "optimizer source role is missing or ambiguous")
        return next(iter(found))


class View:
    def __init__(
        self,
        graph: Graph,
        scope: tuple[str, ...],
        registry: Registry,
        original_constants: set[Any],
        parent: View | None = None,
    ) -> None:
        self.graph, self.scope, self.registry, self.original_constants, self.parent = (
            graph,
            scope,
            registry,
            original_constants,
            parent,
        )
        self.producers: dict[str, tuple[Node, int]] = {}
        self.consumers: dict[str, list[tuple[Node, int]]] = {}
        known = set(graph.inputs) | set(graph.initializers)
        require(len(graph.inputs) == len(set(graph.inputs)), "duplicate graph input")
        for node in graph.nodes:
            require(bool(node.outputs) and all(node.outputs), "missing node result")
            for index, name in enumerate(node.outputs):
                require(name not in known, "duplicate value definition")
                known.add(name)
                self.producers[name] = (node, index)
            for index, name in enumerate(node.inputs):
                if name:
                    self.consumers.setdefault(name, []).append((node, index))
        self.constant_cache: dict[str, TensorData | None] = {}
        self.anchor_cache: dict[str, tuple[Any, ...]] = {}
        self.child_views: list[View] = []
        self.repeated_ifs: dict[tuple[str, tuple[Any, ...]], Node] = {}
        if_roles: dict[str, int] = {}
        for node in graph.nodes:
            if node.op == "If" and children(node):
                role = registry.tag(node)
                if_roles[role] = if_roles.get(role, 0) + 1
        for node in graph.nodes:
            nested = children(node)
            if not nested:
                continue
            role = registry.tag(node)
            occurrence: tuple[str, ...] = (role,)
            if node.op == "If" and if_roles[role] > 1:
                signature = self.if_binding(node)
                key = (role, signature)
                require(
                    key not in self.repeated_ifs,
                    "ambiguous repeated If binding discriminator",
                )
                self.repeated_ifs[key] = node
                # Full structural text, not a digest or optimizer-derived ordinal.
                occurrence += ("binding:" + repr(signature),)
            for attribute, index, child in nested:
                child_scope = scope + occurrence + (attribute, str(index))
                self.child_views.append(
                    View(child, child_scope, registry, original_constants, self)
                )

    def routing_origin(
        self,
        name: str,
        active: frozenset[tuple[tuple[str, ...], str]] = frozenset(),
        *,
        predicate: bool = False,
    ) -> tuple[Any, ...]:
        """Exact ordered origin used only to route repeated control-flow bodies."""
        owner = self.owner(name)
        if owner is not self:
            return owner.routing_origin(name, active, predicate=predicate)
        key = (self.scope, name)
        require(key not in active, "cyclic If binding provenance")
        next_active = active | {key}
        if name in self.graph.inputs:
            return (
                "input",
                self.scope,
                self.graph.inputs.index(name),
                self.value_type(name).dtype,
            )
        constant = self.constant(name)
        if constant is not None:
            return (
                ("constant", constant.dtype, constant.shape, constant.data)
                if predicate
                else ("constant-operand",)
            )
        node, port = self.producers[name]
        if node.op == "Identity" and node.domain in STANDARD:
            self.layout_step(node)
            return self.routing_origin(node.inputs[0], next_active, predicate=predicate)
        if not predicate and node.op in LAYOUT and node.domain in STANDARD:
            step = self.layout_step(node)
            origin = self.routing_origin(node.inputs[0], next_active)
            return origin if step is None else ("layout", step, origin)
        require(not children(node), "nested controller origin remains unproved")

        def attribute(value: Any) -> Any:
            if type(value) in (str, bytes, int, bool, type(None)):
                return value
            if type(value) is float:
                return ("float64-words", struct.pack("<d", value))
            if type(value) in (tuple, list):
                return tuple(attribute(item) for item in value)
            if type(value) is TensorData:
                return ("tensor", value.dtype, value.shape, value.data)
            raise FusionUnfinished("If binding attribute descriptor remains unproved")

        return (
            "producer",
            self.scope,
            self.registry.tag(node),
            "" if node.domain in STANDARD else node.domain,
            node.op,
            node.overload,
            port,
            len(node.outputs),
            self.value_type(name).dtype,
            tuple(
                (
                    self.routing_origin(value, next_active, predicate=predicate)
                    if value
                    else ("missing",)
                )
                for value in node.inputs
            ),
            tuple(
                sorted(
                    (name, item.kind, item.reference, attribute(item.value))
                    for name, item in node.attributes.items()
                )
            ),
        )

    def if_binding(self, node: Node) -> tuple[Any, ...]:
        require(
            node.op == "If"
            and node.domain in STANDARD
            and not node.overload
            and len(node.inputs) == 1
            and bool(node.inputs[0])
            and set(node.attributes) == {"then_branch", "else_branch"}
            and all(
                item.kind == "GRAPH" and not item.reference
                for item in node.attributes.values()
            ),
            "repeated If schema remains unproved",
        )
        captures = []
        for attribute, index, graph in children(node):
            require(
                not graph.inputs and len(graph.outputs) == len(node.outputs),
                "repeated If branch interface differs",
            )
            # Routing ignores constant-only captures and duplicate transparent
            # aliases. Existing per-consumer constant and branch proofs do not.
            dynamic = {
                self.routing_origin(name)
                for name in free_names(graph)
                if self.constant(name) is None
            }
            captures.append((attribute, index, tuple(sorted(dynamic, key=repr))))
        return (
            self.scope,
            self.registry.tag(node),
            tuple(self.routing_origin(name, predicate=True) for name in node.inputs),
            tuple(sorted(captures)),
        )

    def owner(self, name: str) -> View:
        if (
            name in self.producers
            or name in self.graph.inputs
            or name in self.graph.initializers
        ):
            return self
        require(self.parent is not None, "unbound graph value")
        assert self.parent is not None
        return self.parent.owner(name)

    def value_type(self, name: str) -> ValueType:
        owner = self.owner(name)
        declared = self.graph.types.get(name, ValueType(None, None))
        if owner is not self:
            return refine_type(owner.value_type(name), declared)
        # A graph input can override its initializer; its default payload cannot
        # establish a dynamic formal's shape or role.
        if name in self.graph.inputs:
            return declared
        constant = self.constant(name)
        if constant is not None:
            return refine_type(ValueType(constant.dtype, constant.shape), declared)
        return declared

    def constant(
        self, name: str, active: frozenset[tuple[tuple[str, ...], str]] = frozenset()
    ) -> TensorData | None:
        owner = self.owner(name)
        if owner is not self:
            return owner.constant(name, active)
        if name in self.constant_cache:
            return self.constant_cache[name]
        key = (self.scope, name)
        require(key not in active, "cyclic constant provenance")
        if name in self.graph.inputs:
            return None
        if name in self.graph.initializers:
            return self.graph.initializers[name]
        node, _ = self.producers[name]
        if (
            node.domain not in STANDARD
            or node.overload
            or len(node.outputs) != 1
            or any(attribute.reference for attribute in node.attributes.values())
        ):
            return None
        next_active = active | {key}
        result: TensorData | None = None
        attributes = node.attributes

        def arg(index: int) -> TensorData | None:
            return self.constant(node.inputs[index], next_active)

        if node.op == "Constant":
            if (
                not node.inputs
                and set(attributes) == {"value"}
                and attributes["value"].kind == "TENSOR"
            ):
                result = attributes["value"].value
        elif node.op == "Identity" and len(node.inputs) == 1 and not attributes:
            result = arg(0)
        elif (
            node.op == "Slice"
            and len(node.inputs) in (3, 4, 5)
            and all(node.inputs[:3])
            and not attributes
        ):
            data = arg(0)
            if data is not None:
                controls = [
                    arg(index) if value else None
                    for index, value in enumerate(node.inputs[1:], 1)
                ]
                if all(
                    not value or control is not None
                    for value, control in zip(node.inputs[1:], controls, strict=True)
                ):
                    _, slice_starts, slice_ends, slice_axes, slice_steps = (
                        slice_controls(self, node, next_active)
                    )
                    result = data.slice(
                        slice_starts, slice_ends, slice_axes, slice_steps
                    )
        elif (
            node.op == "Reshape"
            and len(node.inputs) == 2
            and set(attributes) <= {"allowzero"}
        ):
            data, shape = arg(0), arg(1)
            if data is not None and shape is not None:
                require(
                    shape.dtype == 7 and len(shape.shape) == 1,
                    "Reshape parameter is not an exact INT64 vector",
                )
                allowzero = attributes.get("allowzero", Attribute("INT", 0))
                require(
                    allowzero.kind == "INT" and type(allowzero.value) is int,
                    "invalid Reshape allowzero schema",
                )
                result = data.reshape(shape.integers(), allowzero.value)
        elif node.op == "Expand" and len(node.inputs) == 2 and not attributes:
            data, shape = arg(0), arg(1)
            if data is not None and shape is not None:
                require(
                    shape.dtype == 7 and len(shape.shape) == 1,
                    "Expand parameter is not an exact INT64 vector",
                )
                expanded_shape = shape.integers()
                require(
                    all(size >= 0 for size in expanded_shape), "invalid Expand shape"
                )
                result = data.broadcast(broadcast_shape(data.shape, expanded_shape))
        elif (
            node.op == "Transpose"
            and len(node.inputs) == 1
            and set(attributes) <= {"perm"}
        ):
            data = arg(0)
            if data is not None:
                permutation = attributes.get(
                    "perm", Attribute("INTS", tuple(reversed(range(len(data.shape)))))
                )
                require(
                    permutation.kind == "INTS", "invalid Transpose permutation schema"
                )
                result = data.transpose(tuple(permutation.value))
        elif node.op == "Concat" and node.inputs and set(attributes) == {"axis"}:
            parts = [arg(index) for index in range(len(node.inputs))]
            if all(part is not None for part in parts):
                require(
                    attributes["axis"].kind == "INT"
                    and attributes["axis"].value in (0, -1),
                    "shape Concat axis differs",
                )
                require(
                    all(
                        part is not None
                        and part.dtype in {6, 7}
                        and len(part.shape) == 1
                        for part in parts
                    ),
                    "Concat is not an integer shape-vector carrier",
                )
                concrete = [part for part in parts if part is not None]
                require(
                    len({part.dtype for part in concrete}) == 1,
                    "shape Concat dtypes differ",
                )
                result = TensorData(
                    concrete[0].dtype,
                    (sum(part.shape[0] for part in concrete),),
                    b"".join(part.data for part in concrete),
                )
        elif (
            node.op in {"Unsqueeze", "Squeeze"}
            and len(node.inputs) == 2
            and not attributes
        ):
            data, axes = arg(0), arg(1)
            if data is not None and axes is not None:
                require(
                    axes.dtype == 7 and len(axes.shape) == 1,
                    "Squeeze/Unsqueeze axes are not an exact INT64 vector",
                )
                dimensions = list(data.shape)
                rank = len(dimensions) + (
                    len(axes.integers()) if node.op == "Unsqueeze" else 0
                )
                axis_values = tuple(
                    axis + rank if axis < 0 else axis for axis in axes.integers()
                )
                require(
                    len(set(axis_values)) == len(axis_values)
                    and all(0 <= axis < rank for axis in axis_values),
                    "invalid squeeze/unsqueeze axes",
                )
                if node.op == "Unsqueeze":
                    for axis in sorted(axis_values):
                        dimensions.insert(axis, 1)
                else:
                    for axis in sorted(axis_values, reverse=True):
                        require(
                            dimensions[axis] == 1, "Squeeze removes nonunit dimension"
                        )
                        dimensions.pop(axis)
                result = data.reshape(tuple(dimensions), 1)
        self.constant_cache[name] = result
        return result

    def anchors(
        self, name: str, active: frozenset[tuple[tuple[str, ...], str]] = frozenset()
    ) -> tuple[Any, ...]:
        owner = self.owner(name)
        if owner is not self:
            return owner.anchors(name, active)
        if name in self.anchor_cache:
            return self.anchor_cache[name]
        key = (self.scope, name)
        require(key not in active, "cyclic dynamic provenance")
        result: tuple[Any, ...]
        if name in self.graph.inputs:
            result = (
                (
                    "input",
                    self.scope,
                    self.graph.inputs.index(name),
                    name if not self.scope else "",
                ),
            )
        elif name in self.graph.initializers:
            constant_key = self.graph.initializers[name].key
            result = (
                (("initializer", constant_key),)
                if constant_key in self.original_constants
                else ()
            )
        else:
            node, _ = self.producers[name]
            arguments = {argument for argument in node.inputs if argument}
            for _, _, child in children(node):
                arguments.update(free_names(child))
            result = tuple(
                sorted(
                    {
                        anchor
                        for argument in arguments
                        for anchor in self.anchors(argument, active | {key})
                    },
                    key=repr,
                )
            )
        self.anchor_cache[name] = result
        return result

    def layout_step(self, node: Node) -> tuple[Any, ...] | None:
        require(
            node.domain in STANDARD
            and not node.overload
            and len(node.outputs) == 1
            and bool(node.inputs)
            and not any(a.reference for a in node.attributes.values()),
            "unproved layout operator schema",
        )
        before, after = self.value_type(node.inputs[0]), self.value_type(
            node.outputs[0]
        )
        require(
            before.dtype is None or after.dtype is None or before.dtype == after.dtype,
            "layout changes known compute dtype",
        )
        if node.op == "Identity":
            require(
                len(node.inputs) == 1 and not node.attributes,
                "unproved Identity schema",
            )
            refine_type(before, after)
            return None
        if node.op == "Transpose":
            require(
                len(node.inputs) == 1 and set(node.attributes) <= {"perm"},
                "unproved Transpose schema",
            )
            attribute = node.attributes.get("perm")
            if attribute is None:
                require(before.shape is not None, "default Transpose rank is unknown")
                assert before.shape is not None
                permutation = tuple(reversed(range(len(before.shape))))
            else:
                require(attribute.kind == "INTS", "Transpose permutation is not INTS")
                permutation = tuple(attribute.value)
            require(
                all(type(axis) is int for axis in permutation)
                and sorted(permutation) == list(range(len(permutation))),
                "invalid ordered Transpose mapping",
            )
            if before.shape is not None:
                require(
                    len(before.shape) == len(permutation),
                    "Transpose permutation rank differs",
                )
                refine_type(
                    ValueType(
                        before.dtype, tuple(before.shape[axis] for axis in permutation)
                    ),
                    after,
                )
            elif after.shape is not None:
                require(
                    len(after.shape) == len(permutation),
                    "Transpose result rank differs",
                )
            return (
                None
                if permutation == tuple(range(len(permutation)))
                else ("transpose", permutation)
            )
        require(
            node.op == "Reshape"
            and len(node.inputs) == 2
            and set(node.attributes) <= {"allowzero"},
            "unproved Reshape schema",
        )
        allowzero = node.attributes.get("allowzero", Attribute("INT", 0))
        require(
            allowzero.kind == "INT"
            and type(allowzero.value) is int
            and allowzero.value in (0, 1),
            "invalid Reshape allowzero schema",
        )
        parameter = self.constant(node.inputs[1])
        require(
            parameter is not None
            and parameter.dtype == 7
            and len(parameter.shape) == 1,
            "Reshape needs an exact INT64 vector parameter",
        )
        require(
            before.shape is not None, "Reshape input shape correspondence is unknown"
        )
        assert before.shape is not None and parameter is not None
        shape = reshape_shape(before.shape, parameter.integers(), allowzero.value)
        refine_type(ValueType(before.dtype, shape), after)
        return None if shape == before.shape else ("reshape", shape)

    def layout(self, name: str) -> tuple[Any, ...]:
        owner = self.owner(name)
        if owner is not self:
            return owner.layout(name)
        steps = []
        active = set()
        while name in self.producers and self.producers[name][0].op in LAYOUT:
            require(name not in active, "cyclic layout provenance")
            active.add(name)
            node, _ = self.producers[name]
            step = self.layout_step(node)
            if step is not None:
                steps.append(step)
            name = node.inputs[0]
        owner = self.owner(name)
        if owner is not self:
            nested = owner.layout(name)
            return nested[0], nested[1] + tuple(reversed(steps))
        item = self.producers.get(name)
        origin = (
            ("boundary", self.anchors(name))
            if item is None
            else ("producer", self.registry.tag(item[0]), item[1], self.anchors(name))
        )
        return origin, tuple(reversed(steps))

    def uses(self, name: str) -> list[Any]:
        result: list[Any] = list(self.consumers.get(name, ()))
        result.extend(
            ("output", index)
            for index, output in enumerate(self.graph.outputs)
            if output == name
        )
        for node in self.graph.nodes:
            for attribute, index, graph in children(node):
                if name in free_names(graph):
                    result.append(
                        ("capture", self.registry.tag(node), attribute, index)
                    )
        return result

    def consumer_operand(self, node: Node, name: str) -> tuple[Any, ...]:
        if not name:
            return ("missing",)
        constant = self.constant(name)
        if constant is not None:
            shape = constant.shape
            if node.domain in STANDARD and node.op in {
                "Add",
                "Sub",
                "Mul",
                "Div",
                "Min",
                "Max",
                "Equal",
                "Less",
                "LessOrEqual",
                "Greater",
                "GreaterOrEqual",
                "And",
                "Or",
                "Xor",
            }:
                # Leading unit axes have the same exact trailing broadcast map.
                # No value reduction, arithmetic or dynamic substitution occurs.
                while shape and shape[0] == 1:
                    shape = shape[1:]
            return (
                "constant",
                constant.dtype,
                shape,
                hashlib.sha256(constant.data).hexdigest(),
            )
        return ("data", self.layout(name))

    def edges(
        self, name: str, active: frozenset[tuple[tuple[str, ...], str]] = frozenset()
    ) -> tuple[Any, ...]:
        key = (self.scope, name)
        require(key not in active, "cyclic result consumer provenance")
        next_active = active | {key}
        result: list[tuple[Any, ...]] = []
        for node, port in self.consumers.get(name, ()):
            if node.op in LAYOUT and node.domain in STANDARD and port == 0:
                step = self.layout_step(node)
                edges = self.edges(node.outputs[0], next_active)
                result.extend(
                    edges
                    if step is None
                    else (("layout", step, edge) for edge in edges)
                )
            else:
                result.append(
                    (
                        "consumer",
                        self.registry.tag(node),
                        port,
                        tuple(
                            self.consumer_operand(node, argument)
                            for argument in node.inputs
                        ),
                        tuple(self.anchors(output) for output in node.outputs),
                    )
                )
        result.extend(
            ("output", self.scope, index)
            for index, output in enumerate(self.graph.outputs)
            if output == name
        )
        for child in self.child_views:
            if name in free_names(child.graph):
                result.extend(child.edges(name, next_active))
        return tuple(sorted(result, key=repr))


def all_views(root: View) -> dict[tuple[str, ...], View]:
    result = {root.scope: root}
    for child in root.child_views:
        for scope, view in all_views(child).items():
            require(scope not in result, "ambiguous nested scope correspondence")
            result[scope] = view
    return result


def prove_repeated_if_bindings(
    source_views: dict[tuple[str, ...], View], target_views: dict[tuple[str, ...], View]
) -> None:
    """Routing never substitutes for ordered containing-result/caller proofs."""
    for scope in set(source_views) | set(target_views):
        source, target = source_views.get(scope), target_views.get(scope)
        left = {} if source is None else source.repeated_ifs
        right = {} if target is None else target.repeated_ifs
        require(
            left.keys() == right.keys(),
            "repeated If has no unique exact call binding counterpart",
        )
        if not left:
            continue
        assert source is not None and target is not None
        for key, original in left.items():
            actual = right[key]
            require(
                len(original.outputs) == len(actual.outputs),
                "repeated If output arity differs",
            )
            for name, output in zip(original.outputs, actual.outputs, strict=True):
                require(
                    source.edges(name) == target.edges(output),
                    "repeated If containing outputs, caller ports or sibling edges changed",
                )


def agree_type(
    source: ValueType, target: ValueType, *, rank: int | None = None
) -> ValueType:
    require(
        source.dtype is not None and target.dtype in (None, source.dtype),
        "fusion compute dtype differs from exact source",
    )
    require(source.shape is not None, "source fusion value rank is unknown")
    assert source.shape is not None
    if rank is not None:
        require(len(source.shape) == rank, "fusion matrix is not rank2")
    if target.shape is not None:
        require(
            len(target.shape) == len(source.shape),
            "known fusion rank contradicts source",
        )
        require(
            all(
                a is None or b is None or a == b
                for a, b in zip(source.shape, target.shape, strict=True)
            ),
            "known fusion dimension contradicts source",
        )
    return ValueType(source.dtype, source.shape, source.payload)


def reshape_shape(
    before: tuple[int | str | None, ...], dimensions: tuple[int, ...], allowzero: int
) -> tuple[int | str, ...]:
    require(
        all(type(size) is int and size >= -1 for size in dimensions),
        "invalid Reshape dimensions",
    )
    shape: list[int | str] = list(dimensions)
    for index, size in enumerate(shape):
        if size == 0 and not allowzero:
            require(
                index < len(before) and before[index] is not None,
                "Reshape copies unknown/missing dimension",
            )
            copied = before[index]
            assert copied is not None
            shape[index] = copied
    require(shape.count(-1) <= 1, "ambiguous Reshape inference")

    def factors(sizes: tuple[Any, ...]) -> tuple[int, tuple[str, ...]]:
        require(
            all(
                type(size) is int and size >= 0 or isinstance(size, str) and size
                for size in sizes
            ),
            "Reshape extent has unproved dimensions",
        )
        return math.prod(size for size in sizes if type(size) is int), tuple(
            sorted(size for size in sizes if isinstance(size, str))
        )

    original_count, original_symbols = factors(before)
    if -1 in shape:
        known_count, known_symbols = factors(
            tuple(size for size in shape if size != -1)
        )
        require(
            known_count > 0 and original_count % known_count == 0,
            "unproved Reshape inferred extent",
        )
        remaining = list(original_symbols)
        for symbol in known_symbols:
            require(symbol in remaining, "Reshape inference changes symbolic factors")
            remaining.remove(symbol)
        quotient = original_count // known_count
        require(
            not remaining or len(remaining) == 1 and quotient == 1,
            "Reshape inferred symbolic expression remains unproved",
        )
        shape[shape.index(-1)] = remaining[0] if remaining else quotient
    require(
        factors(tuple(shape)) == (original_count, original_symbols),
        "Reshape changes exact storage extent",
    )
    return tuple(shape)


def refine_type(actual: ValueType, declared: ValueType) -> ValueType:
    require(
        actual.dtype is None
        or declared.dtype is None
        or actual.dtype == declared.dtype,
        "known value dtype contradicts exact correspondence",
    )
    shape = actual.shape if actual.shape is not None else declared.shape
    if actual.shape is not None and declared.shape is not None:
        require(
            len(actual.shape) == len(declared.shape),
            "known value rank contradicts exact correspondence",
        )
        require(
            all(
                a is None or b is None or a == b
                for a, b in zip(actual.shape, declared.shape, strict=True)
            ),
            "known value dimension contradicts exact correspondence",
        )
        shape = tuple(
            a if a is not None else b
            for a, b in zip(actual.shape, declared.shape, strict=True)
        )
    return ValueType(
        actual.dtype if actual.dtype is not None else declared.dtype,
        shape,
        actual.payload if actual.payload is not None else declared.payload,
    )


def matrix_dimensions(a: ValueType, b: ValueType, output: ValueType) -> None:
    require(
        all(
            value.shape is not None and len(value.shape) == 2
            for value in (a, b, output)
        ),
        "source matrix rank2 proof is absent",
    )
    assert a.shape is not None and b.shape is not None and output.shape is not None
    for first, second in (
        (a.shape[1], b.shape[0]),
        (a.shape[0], output.shape[0]),
        (b.shape[1], output.shape[1]),
    ):
        require(
            first is None or second is None or first == second,
            "source matrix dimensions contradict its result",
        )


def broadcastable_bias(
    shape: tuple[int | str | None, ...], output: tuple[int | str | None, ...]
) -> None:
    require(
        len(shape) <= 2 and len(output) == 2,
        "bias rank is outside scalar/vector/matrix broadcasting",
    )
    padded = (1,) * (2 - len(shape)) + shape
    require(
        all(
            before == 1 or before is not None and before == after
            for before, after in zip(padded, output, strict=True)
        ),
        "bias broadcast mapping is unproved",
    )


@dataclass(frozen=True)
class Match:
    source: View
    nodes: tuple[Node, ...]
    data_inputs: tuple[str, ...]
    target: Node
    terminal: str
    kind: str


def equal_operand(
    source: View, name: str, target: View, actual: str, *, rank: int | None = None
) -> None:
    left, right = source.constant(name), target.constant(actual)
    if left is not None or right is not None:
        require(
            left is not None and right is not None and left.key == right.key,
            "constant matrix/activation operand differs",
        )
    else:
        require(
            source.layout(name) == target.layout(actual),
            "ordered operand origin/layout or call binding differs",
        )
    agree_type(source.value_type(name), target.value_type(actual), rank=rank)


def match_gemm(source: View, target: View, node: Node) -> Match:
    require(
        node.domain in STANDARD
        and not node.overload
        and len(node.inputs) == 3
        and all(node.inputs)
        and len(node.outputs) == 1,
        "Gemm needs proved standard three-input zero-transpose form",
    )
    require(
        set(node.attributes) == {"alpha", "beta", "transA", "transB"}
        and not any(attribute.reference for attribute in node.attributes.values()),
        "unknown Gemm attribute schema",
    )
    for name in ("alpha", "beta"):
        attribute = node.attributes[name]
        require(
            attribute.kind == "FLOAT"
            and type(attribute.value) is float
            and struct.pack("<f", attribute.value) == struct.pack("<f", 1.0),
            "Gemm scale is not exact unit FLOAT",
        )
    for name in ("transA", "transB"):
        attribute = node.attributes[name]
        require(
            attribute.kind == "INT"
            and type(attribute.value) is int
            and attribute.value == 0,
            "Gemm transpose form remains unproved",
        )
    tag = source.registry.tag(node, fusion=True)
    matches = []
    for matrix in source.graph.nodes:
        if (
            matrix.op != "MatMul"
            or matrix.domain not in STANDARD
            or source.registry.tag(matrix) != tag
        ):
            continue
        try:
            require(
                not matrix.overload
                and not matrix.attributes
                and len(matrix.inputs) == 2
                and len(matrix.outputs) == 1,
                "source MatMul schema differs",
            )
            uses = source.uses(matrix.outputs[0])
            require(
                len(uses) == 1 and isinstance(uses[0][0], Node),
                "removed matrix result has another live use",
            )
            add, port = uses[0]
            require(
                add.op == "Add"
                and add.domain in STANDARD
                and not add.overload
                and not add.attributes
                and len(add.inputs) == 2
                and len(add.outputs) == 1
                and port in (0, 1),
                "Gemm source is not direct MatMul plus Add",
            )
            bias = add.inputs[1 - port]
            for original, actual in zip(matrix.inputs, node.inputs[:2], strict=True):
                equal_operand(source, original, target, actual, rank=2)
            matrix_type = source.value_type(matrix.outputs[0])
            add_type = source.value_type(add.outputs[0])
            require(
                matrix_type.dtype in {1, 10, 11}
                and matrix_type
                == ValueType(add_type.dtype, add_type.shape, add_type.payload),
                "source Add changes matrix compute type or rank/layout",
            )
            matrix_dimensions(
                source.value_type(matrix.inputs[0]),
                source.value_type(matrix.inputs[1]),
                matrix_type,
            )
            agree_type(add_type, target.value_type(node.outputs[0]), rank=2)
            source_bias, target_bias = source.constant(bias), target.constant(
                node.inputs[2]
            )
            bias_type = source.value_type(bias)
            require(
                bias_type.dtype == matrix_type.dtype and bias_type.shape is not None,
                "source bias compute dtype/rank is unknown",
            )
            assert matrix_type.shape is not None
            assert bias_type.shape is not None
            broadcastable_bias(bias_type.shape, matrix_type.shape)
            if source_bias is not None or target_bias is not None:
                require(
                    source_bias is not None
                    and target_bias is not None
                    and source_bias.dtype == target_bias.dtype == matrix_type.dtype,
                    "constant bias cannot acquire dynamic provenance",
                )
                assert source_bias is not None and target_bias is not None
                broadcastable_bias(target_bias.shape, matrix_type.shape)
                shape = broadcast_shape(source_bias.shape, target_bias.shape)
                require(
                    source_bias.broadcast(shape).key
                    == target_bias.broadcast(shape).key,
                    "constant bias storage/index/broadcast mapping differs",
                )
            else:
                equal_operand(source, bias, target, node.inputs[2])
            matches.append(
                Match(
                    source,
                    (matrix, add),
                    (*matrix.inputs, bias),
                    node,
                    add.outputs[0],
                    "Gemm",
                )
            )
        except FusionUnfinished:
            continue
    require(
        len(matches) == 1,
        "Gemm has no unique exact source MatMul/Add/operand correspondence",
    )
    return matches[0]


def match_quick_gelu(source: View, target: View, node: Node) -> Match:
    require(
        node.domain == "com.microsoft"
        and node.op == "QuickGelu"
        and not node.overload
        and len(node.inputs) == len(node.outputs) == 1,
        "unproved QuickGelu schema",
    )
    require(
        set(node.attributes) == {"alpha"},
        "QuickGelu needs an explicit exact coefficient",
    )
    alpha = node.attributes["alpha"]
    require(
        alpha.kind == "FLOAT"
        and not alpha.reference
        and type(alpha.value) is float
        and math.isfinite(alpha.value),
        "QuickGelu coefficient schema/words differ",
    )
    tag = source.registry.tag(node, fusion=True)
    matches = []
    for terminal in source.graph.nodes:
        if (
            terminal.op != "Mul"
            or terminal.domain not in STANDARD
            or source.registry.tag(terminal) != tag
        ):
            continue
        try:
            require(
                not terminal.overload
                and not terminal.attributes
                and len(terminal.inputs) == 2
                and len(terminal.outputs) == 1,
                "terminal Mul schema differs",
            )
            for sigmoid_port in (0, 1):
                sigmoid_item = source.producers.get(terminal.inputs[sigmoid_port])
                if sigmoid_item is None or sigmoid_item[0].op != "Sigmoid":
                    continue
                sigmoid = sigmoid_item[0]
                require(
                    sigmoid.domain in STANDARD
                    and not sigmoid.overload
                    and not sigmoid.attributes
                    and len(sigmoid.inputs) == len(sigmoid.outputs) == 1,
                    "source Sigmoid schema differs",
                )
                scaled_item = source.producers.get(sigmoid.inputs[0])
                require(scaled_item is not None, "source coefficient Mul is absent")
                assert scaled_item is not None
                scaled = scaled_item[0]
                require(
                    scaled.op == "Mul"
                    and scaled.domain in STANDARD
                    and not scaled.overload
                    and not scaled.attributes
                    and len(scaled.inputs) == 2
                    and len(scaled.outputs) == 1,
                    "coefficient Mul schema differs",
                )
                x = terminal.inputs[1 - sigmoid_port]
                require(
                    scaled.inputs.count(x) == 1,
                    "QuickGelu occurrences do not share the same actual x",
                )
                coefficient = scaled.inputs[1 - scaled.inputs.index(x)]
                require(
                    source.uses(scaled.outputs[0]) == [(sigmoid, 0)]
                    and source.uses(sigmoid.outputs[0]) == [(terminal, sigmoid_port)],
                    "removed activation intermediate has another live use",
                )
                equal_operand(source, x, target, node.inputs[0])
                x_type = source.value_type(x)
                require(
                    x_type.dtype in {1, 10, 11},
                    "QuickGelu floating compute type is unknown",
                )
                for value in (
                    scaled.outputs[0],
                    sigmoid.outputs[0],
                    terminal.outputs[0],
                ):
                    agree_type(x_type, source.value_type(value))
                agree_type(x_type, target.value_type(node.outputs[0]))
                constant = source.constant(coefficient)
                require(
                    constant is not None
                    and constant.dtype == x_type.dtype
                    and constant.shape == (),
                    "QuickGelu coefficient is not the exact typed scalar",
                )
                assert constant is not None
                encoded = struct.pack(
                    {1: "<f", 10: "<e", 11: "<d"}[constant.dtype], alpha.value
                )
                require(
                    encoded == constant.data,
                    "QuickGelu alpha does not match source scalar IEEE words",
                )
                matches.append(
                    Match(
                        source,
                        (scaled, sigmoid, terminal),
                        (x, coefficient),
                        node,
                        terminal.outputs[0],
                        "QuickGelu",
                    )
                )
        except (FusionUnfinished, OverflowError, struct.error):
            continue
    require(
        len(matches) == 1,
        "QuickGelu has no unique complete source-role/operand correspondence",
    )
    return matches[0]


class Rewriter:
    def __init__(self, target: View, match: Match, serial: int) -> None:
        self.target, self.match, self.serial = target, match, serial
        self.prefix = f"__jax2onnx_virtual_fusion_{serial}_"
        self.added: list[Node] = []
        self.cloned: dict[tuple[tuple[str, ...], str], str] = {}
        self.names = (
            set(target.graph.inputs)
            | set(target.graph.initializers)
            | set(target.producers)
        )

    def fresh(self, label: str) -> str:
        candidate = self.prefix + str(len(self.names)) + "_" + label
        require(candidate not in self.names, "virtual name collision")
        self.names.add(candidate)
        return candidate

    def clone_constant(self, name: str, view: View | None = None) -> str:
        source = (self.match.source if view is None else view).owner(name)
        key = (source.scope, name)
        if key in self.cloned:
            return self.cloned[key]
        require(
            source.constant(name) is not None,
            "dynamic value cannot be cloned as a constant",
        )
        if name in source.graph.initializers:
            target = self.fresh("constant")
            self.target.graph.initializers[target] = source.graph.initializers[name]
            self.target.graph.types[target] = source.value_type(name)
            self.cloned[key] = target
            return target
        node, index = source.producers[name]
        require(
            len(node.outputs) == 1 and index == 0,
            "constant closure has multiple results",
        )
        # Evaluator approval above proves each data and integer shape carrier;
        # this copies its exact role, attributes and ordered ports, not values
        # synthesized from an unrelated optimized constant.
        arguments = tuple(
            self.clone_constant(argument, source) if argument else ""
            for argument in node.inputs
        )
        output = self.fresh("constant_result")
        cloned = deepcopy(node)
        cloned.name = self.prefix + node.name
        cloned.inputs, cloned.outputs = arguments, (output,)
        self.added.append(cloned)
        self.target.graph.types[output] = source.value_type(name)
        self.cloned[key] = output
        return output

    def operand(self, original: str, actual: str) -> str:
        if self.match.source.constant(original) is not None:
            return self.clone_constant(original)
        source_type = self.match.source.value_type(original)
        self.target.graph.types[actual] = agree_type(
            source_type, self.target.value_type(actual)
        )
        return actual

    def emit(
        self, source: Node, inputs: tuple[str, ...], output: str, source_output: str
    ) -> None:
        cloned = deepcopy(source)
        cloned.name = self.prefix + source.name
        cloned.inputs, cloned.outputs = inputs, (output,)
        self.added.append(cloned)
        self.target.graph.types[output] = self.match.source.value_type(source_output)

    def apply(self) -> None:
        match, node = self.match, self.match.target
        if match.kind == "Gemm":
            matrix, add = match.nodes
            a = self.operand(matrix.inputs[0], node.inputs[0])
            b = self.operand(matrix.inputs[1], node.inputs[1])
            bias = self.operand(match.data_inputs[2], node.inputs[2])
            intermediate = self.fresh("matrix_result")
            self.emit(matrix, (a, b), intermediate, matrix.outputs[0])
            matrix_port = add.inputs.index(matrix.outputs[0])
            arguments = (
                (intermediate, bias) if matrix_port == 0 else (bias, intermediate)
            )
            self.emit(add, arguments, node.outputs[0], add.outputs[0])
        else:
            scaled, sigmoid, terminal = match.nodes
            x = self.operand(match.data_inputs[0], node.inputs[0])
            coefficient = self.clone_constant(match.data_inputs[1])
            scaled_arguments = tuple(
                x if name == match.data_inputs[0] else coefficient
                for name in scaled.inputs
            )
            scaled_output, sigmoid_output = self.fresh("scaled"), self.fresh("sigmoid")
            self.emit(scaled, scaled_arguments, scaled_output, scaled.outputs[0])
            self.emit(sigmoid, (scaled_output,), sigmoid_output, sigmoid.outputs[0])
            terminal_arguments = tuple(
                x if name == match.data_inputs[0] else sigmoid_output
                for name in terminal.inputs
            )
            self.emit(
                terminal, terminal_arguments, node.outputs[0], terminal.outputs[0]
            )
        position = next(
            index
            for index, current in enumerate(self.target.graph.nodes)
            if current is node
        )
        self.target.graph.nodes[position : position + 1] = self.added


def constant_closure(view: View, name: str) -> tuple[Any, ...]:
    """Retain exact lexical identities of every proved original constant root."""
    seen: set[tuple[tuple[str, ...], str]] = set()
    roots: set[tuple[Any, ...]] = set()

    def visit(current: View, value: str) -> None:
        owner = current.owner(value)
        identity = (owner.scope, value)
        require(identity not in seen, "cyclic original constant closure")
        seen.add(identity)
        constant = owner.constant(value)
        require(constant is not None, "original constant closure is not exact")
        assert constant is not None
        agree_type(ValueType(constant.dtype, constant.shape), owner.value_type(value))
        if value in owner.graph.initializers:
            require(value not in owner.graph.inputs, "overridable constant origin")
            roots.add(("initializer", owner.scope, value, constant.key))
        else:
            node, index = owner.producers[value]
            if node.op == "Constant":
                roots.add(
                    (
                        "Constant",
                        owner.scope,
                        owner.registry.tag(node),
                        index,
                        constant.key,
                    )
                )
            else:
                for argument in node.inputs:
                    if argument:
                        visit(owner, argument)
        seen.remove(identity)

    visit(view, name)
    require(bool(roots), "original constant closure has no exact root")
    return tuple(sorted(roots, key=repr))


def constant_consumer_binding(view: View, node: Node, port: int) -> tuple[Any, ...]:
    """Use ordered bindings, retaining independent arithmetic as it actually is."""

    def binding(name: str) -> tuple[Any, ...]:
        if not name:
            return ("missing",)
        constant = view.constant(name)
        if constant is not None:
            return ("constant", constant.key)
        anchors = tuple(anchor for anchor in view.anchors(name) if anchor[0] == "input")
        require(bool(anchors), "dynamic consumer has no proved input binding")
        return ("dynamic", anchors)

    return (
        tuple(binding(name) for index, name in enumerate(node.inputs) if index != port),
        tuple(
            sorted(
                (
                    (
                        attribute,
                        index,
                        tuple(
                            sorted(
                                (binding(name) for name in free_names(child)), key=repr
                            )
                        ),
                    )
                    for attribute, index, child in children(node)
                ),
                key=repr,
            )
        ),
    )


def restore_constant_origins(
    source_views: dict[tuple[str, ...], View],
    virtual: Graph,
    registry: Registry,
    original_constants: set[Any],
) -> None:
    """Restore only a unique exact constant closure at its actual consumer port."""
    serial = 0
    claimed: set[tuple[tuple[str, ...], int, int]] = set()
    while True:
        target_views = all_views(View(virtual, (), registry, original_constants))
        restored = False
        for scope, target in target_views.items():
            require(
                scope in source_views, "constant origin has no original lexical scope"
            )
            source = source_views[scope]
            for node in target.graph.nodes:
                if node.domain not in STANDARD:
                    continue
                try:
                    tag = registry.tag(node)
                except FusionUnfinished:
                    continue
                for port, actual in enumerate(node.inputs):
                    if not actual:
                        continue
                    owner = target.owner(actual)
                    if (
                        actual not in owner.graph.initializers
                        or actual in owner.graph.inputs
                    ):
                        continue
                    actual_constant = owner.graph.initializers[actual]
                    agree_type(
                        ValueType(actual_constant.dtype, actual_constant.shape),
                        target.value_type(actual),
                    )
                    matches: list[tuple[Node, str, tuple[Any, ...]]] = []
                    for candidate in source.graph.nodes:
                        if (
                            candidate.op != node.op
                            or candidate.domain != node.domain
                            or candidate.overload != node.overload
                            or candidate.attributes != node.attributes
                            or len(candidate.inputs) != len(node.inputs)
                            or len(candidate.outputs) != len(node.outputs)
                            or registry.tag(candidate) != tag
                        ):
                            continue
                        original = candidate.inputs[port]
                        if not original:
                            continue
                        try:
                            constant = source.constant(original)
                            if constant != actual_constant:
                                continue
                            closure = constant_closure(source, original)
                            if constant_consumer_binding(
                                source, candidate, port
                            ) != constant_consumer_binding(target, node, port):
                                continue
                            for index, (left, right) in enumerate(
                                zip(candidate.inputs, node.inputs, strict=True)
                            ):
                                if index != port and left and right:
                                    refine_type(
                                        source.value_type(left),
                                        target.value_type(right),
                                    )
                            matches.append((candidate, original, closure))
                        except FusionUnfinished:
                            continue
                    require(
                        len(matches) <= 1,
                        "constant consumer port has ambiguous original closure/binding",
                    )
                    if not matches:
                        continue
                    candidate, original, _ = matches[0]
                    original_owner = source.owner(original)
                    if original in original_owner.graph.initializers:
                        # The same exact typed initializer already retains its anchor.
                        continue
                    identity = (scope, id(candidate), port)
                    require(
                        identity not in claimed,
                        "original constant consumer port reused by optimized bindings",
                    )
                    match = Match(
                        source,
                        (candidate,),
                        (original,),
                        node,
                        original,
                        "ConstantOrigin",
                    )
                    rewriter = Rewriter(target, match, serial)
                    rewriter.prefix = f"__jax2onnx_virtual_constant_{serial}_"
                    replacement = rewriter.clone_constant(original)
                    position = next(
                        index
                        for index, current in enumerate(target.graph.nodes)
                        if current is node
                    )
                    target.graph.nodes[position:position] = rewriter.added
                    node.inputs = tuple(
                        replacement if index == port else name
                        for index, name in enumerate(node.inputs)
                    )
                    claimed.add(identity)
                    serial += 1
                    restored = True
                    break
                if restored:
                    break
            if restored:
                break
        if not restored:
            return


def fused_permutation(rank: int, transpose: int, batch: int) -> tuple[int, ...]:
    require(rank >= 2, "FusedMatMul operand rank is unproved")
    require(transpose in (0, 1) and batch in (0, 1), "invalid FusedMatMul flags")
    if batch:
        return tuple(range(1, rank - 1)) + (
            (rank - 1, 0) if transpose else (0, rank - 1)
        )
    if transpose:
        return tuple(range(rank - 2)) + (rank - 1, rank - 2)
    return tuple(range(rank))


def bound_attributes(node: Node) -> dict[str, Attribute]:
    require(
        node.domain in STANDARD
        and not node.overload
        and not any(attribute.reference for attribute in node.attributes.values()),
        "unproved bound producer schema",
    )
    attributes = dict(node.attributes)
    if node.op == "Cast":
        require(
            set(attributes) <= {"to", "saturate"} and "to" in attributes,
            "unproved Cast attribute schema",
        )
        attributes.setdefault("saturate", Attribute("INT", 1))
        require(
            all(
                attribute.kind == "INT" and type(attribute.value) is int
                for attribute in attributes.values()
            )
            and attributes["saturate"].value in (0, 1),
            "unproved Cast attribute value",
        )
        require(attributes["to"].value > 0, "Cast target dtype is invalid")
    elif node.op == "ReduceSum":
        require(
            set(attributes) <= {"keepdims", "noop_with_empty_axes"},
            "unproved ReduceSum attribute schema",
        )
        attributes.setdefault("keepdims", Attribute("INT", 1))
        attributes.setdefault("noop_with_empty_axes", Attribute("INT", 0))
        require(
            all(
                attribute.kind == "INT"
                and type(attribute.value) is int
                and attribute.value in (0, 1)
                for attribute in attributes.values()
            ),
            "unproved ReduceSum attribute value",
        )
    elif node.op in {"Div", "Exp", "Identity"}:
        require(not attributes, "unproved bound operator attributes")
    return attributes


def bound_producer(
    source: View, name: str, target: View, actual: str
) -> tuple[Node, Node]:
    left_owner, right_owner = source.owner(name), target.owner(actual)
    require(
        left_owner.scope == right_owner.scope, "bound producer lexical scope differs"
    )
    require(
        name in left_owner.producers and actual in right_owner.producers,
        "bound data producer is missing",
    )
    left, left_port = left_owner.producers[name]
    right, right_port = right_owner.producers[actual]
    require(
        left.op == right.op
        and left.domain in STANDARD
        and right.domain in STANDARD
        and left_port == right_port == 0
        and len(left.outputs) == len(right.outputs) == 1
        and len(left.inputs) == len(right.inputs)
        and left_owner.registry.tag(left) == right_owner.registry.tag(right)
        and bound_attributes(left) == bound_attributes(right),
        "bound data producer role/schema/output port differs",
    )
    for original, optimized in zip(left.inputs, right.inputs, strict=True):
        require(bool(original) == bool(optimized), "bound optional operand differs")
        if original:
            equal_operand(left_owner, original, right_owner, optimized)
            # Preserve the immediate ordered producers as well as their root
            # bindings; equal names or a union of anchors cannot establish this.
            original_owner, optimized_owner = left_owner.owner(
                original
            ), right_owner.owner(optimized)
            if (
                original in original_owner.producers
                and optimized in optimized_owner.producers
            ):
                a, ap = original_owner.producers[original]
                b, bp = optimized_owner.producers[optimized]
                if a.op not in LAYOUT and b.op not in LAYOUT:
                    require(
                        a.op == b.op
                        and ap == bp
                        and len(a.inputs) == len(b.inputs)
                        and len(a.outputs) == len(b.outputs)
                        and original_owner.registry.tag(a)
                        == optimized_owner.registry.tag(b)
                        and bound_attributes(a) == bound_attributes(b),
                        "bound ordered operand producer changed",
                    )
                    for a_input, b_input in zip(a.inputs, b.inputs, strict=True):
                        require(
                            bool(a_input) == bool(b_input),
                            "bound producer optional input differs",
                        )
                        if a_input:
                            equal_operand(
                                original_owner, a_input, optimized_owner, b_input
                            )
    return left, right


def prove_bound_type(
    source: View,
    name: str,
    target: View,
    actual: str,
    updates: list[tuple[View, str, ValueType]],
) -> ValueType:
    left, right = bound_producer(source, name, target, actual)
    require(left.op in {"Div", "Cast"}, "missing bound producer type remains unproved")
    require(
        len(left.inputs) == (2 if left.op == "Div" else 1),
        "bound producer arity differs",
    )
    if left.op == "Cast":
        input_source, input_target = left.inputs[0], right.inputs[0]
        known = target.value_type(input_target)
        if known.dtype is None or known.shape is None:
            known = prove_bound_type(
                source, input_source, target, input_target, updates
            )
        require(known.shape is not None, "Cast input rank remains unproved")
        inferred = ValueType(bound_attributes(right)["to"].value, known.shape)
        inferred = refine_type(inferred, source.value_type(name))
    else:
        actual_inputs = tuple(target.value_type(argument) for argument in right.inputs)
        if all(
            value.dtype is not None and value.shape is not None
            for value in actual_inputs
        ):
            first, second = actual_inputs
            assert first.shape is not None and second.shape is not None
            require(first.dtype == second.dtype, "Div operand dtypes differ")
            dimensions: list[int | str | None] = []
            for index in range(max(len(first.shape), len(second.shape))):
                a = first.shape[-1 - index] if index < len(first.shape) else 1
                b = second.shape[-1 - index] if index < len(second.shape) else 1
                require(
                    a == 1 or b == 1 or a is not None and a == b,
                    "Div operand broadcast dimensions are unproved",
                )
                dimensions.append(b if a == 1 else a)
            inferred = refine_type(
                ValueType(first.dtype, tuple(reversed(dimensions))),
                source.value_type(name),
            )
        else:
            # Source-derived refinement follows the exact Div schema, ordered
            # operand layouts/producers and actual call bindings above.
            inferred = source.value_type(name)
        require(
            inferred.dtype is not None and inferred.shape is not None,
            "corresponding Div source type is absent",
        )
    inferred = refine_type(inferred, target.value_type(actual))
    updates.append((target.owner(actual), actual, inferred))
    return inferred


def slice_controls(
    view: View,
    node: Node,
    active: frozenset[tuple[tuple[str, ...], str]] = frozenset(),
) -> tuple[tuple[int, ...], ...]:
    require(
        node.domain in STANDARD
        and node.op == "Slice"
        and not node.overload
        and not node.attributes
        and len(node.inputs) in (3, 4, 5)
        and all(node.inputs[:3])
        and len(node.outputs) == 1,
        "unproved Slice schema",
    )
    controls = [
        view.constant(name, active) if name else None for name in node.inputs[1:]
    ]
    require(
        all(
            not name or value is not None
            for name, value in zip(node.inputs[1:], controls, strict=True)
        ),
        "Slice provided controls are not exact constants",
    )
    require(
        controls[0] is not None and controls[1] is not None,
        "Slice starts/ends are not exact constants",
    )
    starts = controls[0]
    assert starts is not None
    require(
        starts.dtype in {6, 7} and len(starts.shape) == 1,
        "Slice controls are not typed integer vectors",
    )
    count = starts.shape[0]
    for value in controls:
        if value is not None:
            require(
                value.dtype == starts.dtype and value.shape == (count,),
                "Slice control dtype/shape differs",
            )
    ends = controls[1]
    assert ends is not None
    axes = (
        controls[2].integers()
        if len(controls) > 2 and controls[2] is not None
        else tuple(range(count))
    )
    steps = (
        controls[3].integers()
        if len(controls) > 3 and controls[3] is not None
        else (1,) * count
    )
    require(all(step != 0 for step in steps), "Slice step is zero")
    return ((starts.dtype,), starts.integers(), ends.integers(), axes, steps)


def prove_slice_type(
    source: View,
    name: str,
    target: View,
    actual: str,
    updates: list[tuple[View, str, ValueType]],
) -> ValueType:
    source_owner, target_owner = source.owner(name), target.owner(actual)
    require(source_owner.scope == target_owner.scope, "Slice lexical scope differs")
    left, left_port = source_owner.producers[name]
    right, right_port = target_owner.producers[actual]
    require(
        left_port == right_port == 0
        and source_owner.registry.tag(left) == target_owner.registry.tag(right),
        "Slice role/output port differs",
    )
    original, controls = slice_controls(source_owner, left), slice_controls(
        target_owner, right
    )
    require(original == controls, "Slice typed ordered controls differ")
    require(
        source_owner.layout(left.inputs[0]) == target_owner.layout(right.inputs[0]),
        "Slice data origin/layout/call binding differs",
    )
    data = prove_bound_type(
        source_owner, left.inputs[0], target_owner, right.inputs[0], updates
    )
    require(
        data.shape is not None
        and all(type(size) is int and size >= 0 for size in data.shape),
        "Slice input bounds remain unproved",
    )
    assert data.shape is not None
    _, starts, ends, axes, steps = controls
    dimensions = tuple(
        len(indices)
        for indices in slice_index_ranges(data.shape, starts, ends, axes, steps)
    )
    inferred = refine_type(
        ValueType(data.dtype, dimensions), source_owner.value_type(name)
    )
    inferred = refine_type(inferred, target_owner.value_type(actual))
    updates.append((target_owner, actual, inferred))
    return inferred


@dataclass(frozen=True)
class FusedMatMulMatch:
    source: View
    matrix: Node
    target: Node
    wrappers: tuple[tuple[Node, ...], ...]
    updates: tuple[tuple[View, str, ValueType], ...]
    operand_types: tuple[ValueType, ...]
    result_type: ValueType
    scale: Node | None = None

    @property
    def terminal(self) -> str:
        return (self.matrix if self.scale is None else self.scale).outputs[0]


def fused_matmul_needs_proof(node: Node) -> bool:
    alpha = node.attributes.get("alpha", Attribute("FLOAT", 1.0))
    return any(
        node.attributes.get(flag, Attribute("INT", 0)).value != 0
        for flag in ("transBatchA", "transBatchB")
    ) or not (
        alpha.kind == "FLOAT"
        and not alpha.reference
        and type(alpha.value) is float
        and alpha.value == 1.0
    )


def match_fused_post_mul(source: View, matrix: Node, alpha: Attribute) -> Node:
    uses = source.uses(matrix.outputs[0])
    require(
        len(uses) == 1 and isinstance(uses[0][0], Node) and uses[0][1] in (0, 1),
        "scaled MatMul has another consumer, direct output or lexical capture",
    )
    scale: Node = uses[0][0]
    port: int = uses[0][1]
    require(
        scale.domain in STANDARD
        and scale.op == "Mul"
        and not scale.overload
        and not scale.attributes
        and len(scale.inputs) == 2
        and len(scale.outputs) == 1
        and scale.inputs[port] == matrix.outputs[0]
        and scale.inputs.count(matrix.outputs[0]) == 1,
        "nonunit FusedMatMul has no exact source post-Mul placement",
    )
    source.registry.tag(scale)
    constant = source.constant(scale.inputs[1 - port])
    require(
        constant is not None
        and constant.dtype == 1
        and constant.shape == ()
        and constant.data == struct.pack("<f", alpha.value),
        "FusedMatMul alpha differs from the exact FP32 scalar post-Mul storage",
    )
    require(
        source.value_type(matrix.outputs[0]).dtype == 1
        and source.value_type(scale.outputs[0]).dtype == 1,
        "post-Mul rounding proof requires explicit FP32 matrix/result types",
    )
    return scale


def match_fused_matmul(
    source: View,
    target: View,
    node: Node,
    fp32_post_mul_contract: str | None = None,
) -> FusedMatMulMatch:
    require(
        node.domain == "com.microsoft"
        and node.op == "FusedMatMul"
        and not node.overload
        and len(node.inputs) == 2
        and all(node.inputs)
        and len(node.outputs) == 1,
        "unproved FusedMatMul schema",
    )
    require(
        set(node.attributes)
        <= {"alpha", "transA", "transB", "transBatchA", "transBatchB"}
        and not any(attribute.reference for attribute in node.attributes.values()),
        "unproved FusedMatMul attribute schema",
    )
    alpha = node.attributes.get("alpha", Attribute("FLOAT", 1.0))
    require(
        alpha.kind == "FLOAT"
        and type(alpha.value) is float
        and math.isfinite(alpha.value)
        and alpha.value != 0.0,
        "FusedMatMul alpha is not finite nonzero FLOAT",
    )
    scaled = struct.pack("<f", alpha.value) != struct.pack("<f", 1.0)
    if scaled:
        require(
            fp32_post_mul_contract == _FP32_POST_MUL_SINGLE_PANEL,
            "nonunit FusedMatMul lacks a reviewed FP32 single-K-panel rounding contract",
        )
    flags = {}
    for name in ("transA", "transB", "transBatchA", "transBatchB"):
        value = node.attributes.get(name, Attribute("INT", 0))
        require(
            value.kind == "INT" and type(value.value) is int and value.value in (0, 1),
            "FusedMatMul flag is not Boolean INT",
        )
        flags[name] = value.value
    tag = source.registry.tag(node)
    matches: list[FusedMatMulMatch] = []
    failures: list[str] = []
    for matrix in source.graph.nodes:
        if (
            matrix.op != "MatMul"
            or matrix.domain not in STANDARD
            or source.registry.tag(matrix) != tag
        ):
            continue
        try:
            require(
                not matrix.overload
                and not matrix.attributes
                and len(matrix.inputs) == 2
                and len(matrix.outputs) == 1,
                "source MatMul schema differs",
            )
            updates: list[tuple[View, str, ValueType]] = []
            wrappers: list[tuple[Node, ...]] = []
            operand_types: list[ValueType] = []
            for port, (original, actual) in enumerate(
                zip(matrix.inputs, node.inputs, strict=True)
            ):
                letter = "A" if port == 0 else "B"
                declared = target.value_type(actual)
                logical = source.value_type(original)
                shape = declared.shape if declared.shape is not None else logical.shape
                require(shape is not None, "FusedMatMul operand rank remains unknown")
                assert shape is not None
                permutation = fused_permutation(
                    len(shape), flags["trans" + letter], flags["transBatch" + letter]
                )
                selected: list[Node] = []
                physical = original
                physical_owner = source.owner(physical)
                while (
                    physical in physical_owner.producers
                    and physical_owner.producers[physical][0].op == "Identity"
                ):
                    identity = physical_owner.producers[physical][0]
                    source.layout_step(identity)
                    selected.append(identity)
                    physical = identity.inputs[0]
                    physical_owner = source.owner(physical)
                if permutation != tuple(range(len(permutation))):
                    require(
                        physical in physical_owner.producers,
                        "fused transpose has no exact original layout",
                    )
                    transpose = physical_owner.producers[physical][0]
                    require(
                        transpose.op == "Transpose",
                        "fused operand lost an original Cast/layout boundary",
                    )
                    source.layout_step(transpose)
                    attribute = transpose.attributes.get("perm")
                    expected = (
                        tuple(attribute.value)
                        if attribute is not None
                        else tuple(reversed(range(len(permutation))))
                    )
                    require(
                        expected == permutation,
                        "fused ordered permutation differs from original",
                    )
                    selected.append(transpose)
                    physical = transpose.inputs[0]
                require(
                    source.layout(physical) == target.layout(actual),
                    "fused physical operand layout/call binding differs",
                )
                if declared.dtype is None or declared.shape is None:
                    require(
                        physical in source.owner(physical).producers
                        and actual in target.owner(actual).producers,
                        "missing fused operand type has no exact Slice producer",
                    )
                    declared = prove_slice_type(
                        source, physical, target, actual, updates
                    )
                physical_type = refine_type(source.value_type(physical), declared)
                require(
                    physical_type.dtype in {1, 10, 11}
                    and physical_type.shape is not None
                    and len(physical_type.shape) == len(permutation),
                    "fused physical operand type/rank is unproved",
                )
                assert physical_type.shape is not None
                effective = ValueType(
                    physical_type.dtype,
                    tuple(physical_type.shape[index] for index in permutation),
                )
                agree_type(logical, effective)
                wrappers.append(tuple(reversed(selected)))
                operand_types.append(effective)
            a, b = operand_types
            assert a.shape is not None and b.shape is not None
            if scaled:
                # The reviewed x86 FMA3 single-K-panel kernel completes its
                # FP32 accumulation before a separate FP32 alpha multiply.
                # This preserves the post-Mul rounding boundary; actual dot
                # schedules and numerical acceptance remain separate gates.
                inner = a.shape[-1]
                require(
                    a.dtype == b.dtype == 1
                    and type(inner) is int
                    and 1 <= inner <= 128,
                    "nonunit FusedMatMul rounding requires FP32 and proved integer 1<=K<=128",
                )
            require(
                a.dtype == b.dtype
                and a.shape[-1] == b.shape[-2]
                and a.shape[-1] is not None,
                "FusedMatMul inner dimensions/dtypes differ",
            )
            batch: list[int | str | None] = []
            for index in range(max(len(a.shape), len(b.shape)) - 2):
                first = a.shape[-3 - index] if index < len(a.shape) - 2 else 1
                second = b.shape[-3 - index] if index < len(b.shape) - 2 else 1
                require(
                    first == 1 or second == 1 or first is not None and first == second,
                    "FusedMatMul batch broadcast is unproved",
                )
                batch.append(second if first == 1 else first)
            result = ValueType(a.dtype, (*reversed(batch), a.shape[-2], b.shape[-1]))
            result = refine_type(result, source.value_type(matrix.outputs[0]))
            scale = match_fused_post_mul(source, matrix, alpha) if scaled else None
            if scale is not None:
                result = refine_type(result, source.value_type(scale.outputs[0]))
            result = refine_type(result, target.value_type(node.outputs[0]))
            matches.append(
                FusedMatMulMatch(
                    source,
                    matrix,
                    node,
                    tuple(wrappers),
                    tuple(updates),
                    tuple(operand_types),
                    result,
                    scale,
                )
            )
        except FusionUnfinished as exc:
            if scaled:
                failures.append(str(exc))
            continue
    require(
        len(matches) == 1,
        "FusedMatMul has no unique exact operand/permutation/type correspondence"
        + (": " + "; ".join(sorted(set(failures))) if scaled and not matches else ""),
    )
    return matches[0]


def project_layout_types(view: View, names: tuple[str, ...]) -> None:
    queue = list(names)
    seen: set[str] = set()
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        before = view.value_type(name)
        if before.shape is None or before.dtype is None:
            continue
        for node, port in view.consumers.get(name, ()):
            if port != 0 or node.op not in LAYOUT or node.domain not in STANDARD:
                continue
            view.layout_step(node)
            if node.op == "Transpose":
                attribute = node.attributes.get("perm")
                permutation = (
                    tuple(attribute.value)
                    if attribute is not None
                    else tuple(reversed(range(len(before.shape))))
                )
                shape = tuple(before.shape[index] for index in permutation)
            elif node.op == "Reshape":
                parameter = view.constant(node.inputs[1])
                assert parameter is not None
                allowzero = node.attributes.get("allowzero", Attribute("INT", 0)).value
                shape = reshape_shape(before.shape, parameter.integers(), allowzero)
            else:
                shape = before.shape
            output = node.outputs[0]
            view.graph.types[output] = refine_type(
                ValueType(before.dtype, shape), view.value_type(output)
            )
            queue.append(output)


def restore_fused_matmul(target: View, match: FusedMatMulMatch, serial: int) -> None:
    node, source = match.target, match.source
    placeholder = Match(
        source,
        (match.matrix,) if match.scale is None else (match.matrix, match.scale),
        match.matrix.inputs,
        node,
        match.terminal,
        "FusedMatMul",
    )
    rewriter = Rewriter(target, placeholder, serial)
    inputs: list[str] = []
    for port, original_wrappers in enumerate(match.wrappers):
        argument = node.inputs[port]
        for original in original_wrappers:
            cloned = deepcopy(original)
            output = rewriter.fresh("operand_layout")
            cloned.name = rewriter.prefix + original.name
            cloned.inputs, cloned.outputs = (argument,), (output,)
            rewriter.added.append(cloned)
            target.graph.types[output] = source.value_type(original.outputs[0])
            argument = output
        inputs.append(argument)
    matrix_output = (
        node.outputs[0] if match.scale is None else rewriter.fresh("matrix_result")
    )
    rewriter.emit(match.matrix, tuple(inputs), matrix_output, match.matrix.outputs[0])
    if match.scale is not None:
        scale = match.scale
        matrix_port = scale.inputs.index(match.matrix.outputs[0])
        coefficient = rewriter.clone_constant(scale.inputs[1 - matrix_port])
        arguments = (
            (matrix_output, coefficient)
            if matrix_port == 0
            else (coefficient, matrix_output)
        )
        rewriter.emit(scale, arguments, node.outputs[0], scale.outputs[0])
    target.graph.types[node.outputs[0]] = match.result_type
    for owner, name, value in match.updates:
        owner.graph.types[name] = value
    position = next(
        index for index, current in enumerate(target.graph.nodes) if current is node
    )
    target.graph.nodes[position : position + 1] = rewriter.added
    project_layout_types(target, (node.outputs[0],))


class _IntegerCone:
    """An exact ordered integer program rooted at one actual local Cast SSA."""

    operations = {
        "Less",
        "Greater",
        "Add",
        "Where",
        "Max",
        "Min",
        "Or",
        "Unsqueeze",
        "Reshape",
        "Expand",
    }

    def __init__(self, view: View, root: Node, token: tuple[Any, ...]) -> None:
        self.view, self.root, self.token = view, root, token
        self.nodes: dict[str, Node] = {}
        self.endpoints: dict[tuple[str, int], tuple[Node, str]] = {}
        self.keys: dict[str, tuple[Any, ...]] = {}
        pending = [root.outputs[0]]
        seen: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            for use in view.uses(name):
                require(
                    isinstance(use[0], Node),
                    "integer cone output/capture remains unproved",
                )
                node, port = use
                require(
                    node.domain in STANDARD
                    and not node.overload
                    and len(node.outputs) == 1,
                    "integer cone schema differs",
                )
                observed = view.value_type(node.outputs[0])
                boundary = (
                    node.op == "Gather"
                    and port == 1
                    or node.op == "Where"
                    and port == 0
                    and observed.dtype in {1, 10, 11, 16}
                )
                if boundary:
                    key = (view.registry.tag(node), port)
                    require(key not in self.endpoints, "ambiguous integer endpoint")
                    self.endpoints[key] = (node, name)
                else:
                    require(
                        node.op in self.operations, "unsupported external integer user"
                    )
                    require(
                        node.outputs[0] not in self.nodes
                        or self.nodes[node.outputs[0]] is node,
                        "ambiguous integer definition",
                    )
                    self.nodes[node.outputs[0]] = node
                    pending.extend(node.outputs)
        for name in self.nodes:
            self.expression(name)
        require(bool(self.endpoints), "integer cone has no live endpoint")
        for _, name in self.endpoints.values():
            self.expression(name)

    def known(self, name: str) -> tuple[int, tuple[int, ...]]:
        value = self.view.value_type(name)
        require(
            value.dtype in {6, 7, 9}
            and value.shape is not None
            and all(type(size) is int and size >= 0 for size in value.shape),
            "integer cone type/shape remains unproved",
        )
        assert value.dtype is not None and value.shape is not None
        return value.dtype, tuple(size for size in value.shape if isinstance(size, int))

    def expression(
        self, name: str, active: frozenset[str] = frozenset()
    ) -> tuple[Any, ...]:
        if name in self.keys:
            return self.keys[name]
        require(name not in active, "cyclic integer expression")
        dtype, shape = self.known(name)
        constant = self.view.constant(name)
        result: tuple[Any, ...]
        if name == self.root.outputs[0]:
            result = (dtype, shape, ("Cast-root", self.token))
        elif constant is not None:
            result = (
                dtype,
                shape,
                ("constant", constant.dtype, constant.shape, constant.data),
            )
        else:
            require(
                name in self.nodes and self.view.owner(name) is self.view,
                "integer operand has an unrelated owner/definition",
            )
            node = self.nodes[name]
            require(
                node.domain in STANDARD
                and not node.overload
                and len(node.outputs) == 1
                and all(node.inputs)
                and not any(item.reference for item in node.attributes.values()),
                "unproved integer operator schema",
            )
            require(
                node.op in self.operations,
                "integer expression operator remains unproved",
            )
            next_active = active | {name}
            arguments = tuple(self.expression(arg, next_active) for arg in node.inputs)
            if node.op in {"Reshape", "Expand", "Unsqueeze"}:
                require(len(arguments) == 2, "integer view arity differs")
                control = self.view.constant(node.inputs[1])
                require(
                    control is not None
                    and control.dtype == 7
                    and len(control.shape) == 1,
                    "integer view control is not exact INT64 vector",
                )
                assert control is not None
                values = control.integers()
                before = arguments[0][1]
                require(dtype == arguments[0][0], "integer view changes dtype")
                if node.op == "Reshape":
                    require(
                        set(node.attributes) <= {"allowzero"},
                        "integer Reshape attribute differs",
                    )
                    allowzero = node.attributes.get("allowzero", Attribute("INT", 0))
                    require(
                        allowzero.kind == "INT"
                        and type(allowzero.value) is int
                        and allowzero.value in (0, 1),
                        "integer Reshape allowzero schema differs",
                    )
                    require(
                        reshape_shape(before, values, allowzero.value) == shape,
                        "integer Reshape extent differs",
                    )
                    tree = arguments[0][2]
                elif node.op == "Expand":
                    require(
                        not node.attributes
                        and all(size >= 0 for size in values)
                        and broadcast_shape(before, values) == before == shape,
                        "integer Expand is not identity",
                    )
                    tree = arguments[0][2]
                else:
                    require(not node.attributes, "integer Unsqueeze attributes differ")
                    rank = len(before) + len(values)
                    axes = tuple(axis + rank if axis < 0 else axis for axis in values)
                    require(
                        len(set(axes)) == len(axes)
                        and all(0 <= axis < rank for axis in axes),
                        "integer Unsqueeze axes differ",
                    )
                    dimensions = list(before)
                    for axis in sorted(axes):
                        dimensions.insert(axis, 1)
                    require(
                        tuple(dimensions) == shape, "integer Unsqueeze shape differs"
                    )
                    tree = (node.op, arguments)
                result = (dtype, shape, tree)
            else:
                require(
                    not node.attributes,
                    "integer arithmetic/comparison attributes differ",
                )
                arity = 3 if node.op == "Where" else 2
                require(
                    len(arguments) == arity, "integer arithmetic ordered arity differs"
                )
                if node.op == "Where":
                    require(
                        arguments[0][0] == 9
                        and arguments[1][0] == arguments[2][0] == dtype
                        and dtype in {6, 7},
                        "integer Where arms/types differ",
                    )
                elif node.op == "Or":
                    require(
                        dtype == 9 and all(arg[0] == 9 for arg in arguments),
                        "integer mask Or type differs",
                    )
                else:
                    require(
                        arguments[0][0] == arguments[1][0]
                        and arguments[0][0] in {6, 7}
                        and dtype
                        == (9 if node.op in {"Less", "Greater"} else arguments[0][0]),
                        "integer arithmetic/comparison dtype differs",
                    )
                expected: tuple[int, ...] = ()
                for arg in arguments:
                    expected = broadcast_shape(expected, arg[1])
                require(expected == shape, "integer arithmetic broadcast differs")
                result = (dtype, shape, (node.op, arguments))
        self.keys[name] = result
        return result

    def triples(self) -> list[tuple[Node, Node, Node]]:
        root = self.root.outputs[0]
        triples = []
        claimed: set[int] = set()
        for node in self.nodes.values():
            if node.op != "Where" or len(node.inputs) != 3 or node.inputs[2] != root:
                continue
            less = self.view.producers.get(node.inputs[0])
            add = self.view.producers.get(node.inputs[1])
            require(
                less is not None and add is not None,
                "integer normalization producer missing",
            )
            assert less is not None and add is not None
            a, b = less[0], add[0]
            require(
                a.op == "Less"
                and b.op == "Add"
                and a.inputs[0] == b.inputs[0] == root
                and self.view.constant(a.inputs[1]) is not None
                and self.view.constant(b.inputs[1]) is not None,
                "integer normalization ordered operands differ",
            )
            require(
                not claimed.intersection({id(a), id(b), id(node)}),
                "integer normalization roles overlap",
            )
            claimed.update((id(a), id(b), id(node)))
            triples.append((a, b, node))
        require(
            claimed == {id(node) for node, _ in self.view.consumers.get(root, ())},
            "other direct Cast users differ",
        )
        return triples

    def ancestors(self, name: str) -> set[str]:
        result: set[str] = set()
        pending = [name]
        while pending:
            value = pending.pop()
            if value in result or value not in self.nodes:
                continue
            result.add(value)
            pending.extend(self.nodes[value].inputs)
        return result


def _integer_endpoint_binding(
    source: View,
    left: Node,
    target: View,
    right: Node,
    port: int,
    gathers: set[tuple[str, str]],
) -> None:
    require(
        left.op == right.op
        and left.domain in STANDARD
        and right.domain in STANDARD
        and not left.overload
        and not right.overload
        and left.attributes == right.attributes
        and not any(a.reference for a in left.attributes.values())
        and len(left.inputs) == len(right.inputs)
        and len(left.outputs) == len(right.outputs) == 1,
        "integer endpoint schema/attributes differ",
    )
    require(
        (
            left.op == "Gather"
            and port == 1
            and len(left.inputs) == 2
            and set(left.attributes) <= {"axis"}
        )
        or (
            left.op == "Where"
            and port == 0
            and len(left.inputs) == 3
            and not left.attributes
        ),
        "unproved integer endpoint",
    )
    if left.op == "Gather":
        axis = left.attributes.get("axis", Attribute("INT", 0))
        require(
            axis.kind == "INT" and type(axis.value) is int,
            "integer Gather axis schema differs",
        )
    for index, (a, b) in enumerate(zip(left.inputs, right.inputs, strict=True)):
        if index != port:
            require(bool(a) and bool(b), "integer endpoint optional input differs")
            if left.op == "Where" and index == 2:
                require(
                    (a, b) in gathers, "integer mask data is not its proved Gather pair"
                )
            else:
                equal_operand(source, a, target, b)
            first, second = source.value_type(a), target.value_type(b)
            require(
                first.dtype == second.dtype and first.shape == second.shape,
                "integer endpoint side type/shape differs",
            )
    first, second = source.value_type(left.outputs[0]), target.value_type(
        right.outputs[0]
    )
    require(
        first.dtype == second.dtype
        and first.shape == second.shape
        and first.dtype is not None
        and first.shape is not None,
        "integer endpoint result type/shape differs",
    )


def _integer_normalizations(view: View, root: Node) -> int:
    count = 0
    for node, port in view.consumers.get(root.outputs[0], ()):
        if node.op != "Where" or port != 2 or len(node.inputs) != 3:
            continue
        less, add = view.producers.get(node.inputs[0]), view.producers.get(
            node.inputs[1]
        )
        if (
            less is not None
            and add is not None
            and less[0].op == "Less"
            and add[0].op == "Add"
            and less[0].inputs
            and add[0].inputs
            and less[0].inputs[0] == add[0].inputs[0] == root.outputs[0]
        ):
            count += 1
    return count


def _integer_cast_binding(source: View, left: Node, target: View, right: Node) -> bool:
    try:
        for owner, node in ((source, left), (target, right)):
            require(
                len(node.inputs) == len(node.outputs) == 1
                and owner.value_type(node.inputs[0]).dtype in {6, 7}
                and owner.value_type(node.outputs[0]).dtype in {6, 7},
                "integer root is not an integer conversion",
            )
            attributes = node.attributes
            require(
                set(attributes) <= {"to", "saturate", "round_mode"}
                and "to" in attributes
                and not any(item.reference for item in attributes.values()),
                "integer root Cast attributes differ",
            )
            to = attributes["to"]
            require(
                to.kind == "INT"
                and type(to.value) is int
                and to.value == owner.value_type(node.outputs[0]).dtype,
                "integer root Cast to/type differs",
            )
            if "saturate" in attributes:
                item = attributes["saturate"]
                require(
                    item.kind == "INT" and type(item.value) is int and item.value == 1,
                    "integer root Cast saturate differs",
                )
            if "round_mode" in attributes:
                item = attributes["round_mode"]
                require(
                    item.kind == "STRING" and item.value == b"up",
                    "integer root Cast round_mode differs",
                )
        require(
            len(left.inputs)
            == len(right.inputs)
            == len(left.outputs)
            == len(right.outputs)
            == 1
            and not left.overload
            and not right.overload,
            "integer root Cast schema differs",
        )
        equal_operand(source, left.inputs[0], target, right.inputs[0])
        a, b = source.value_type(left.inputs[0]), target.value_type(right.inputs[0])
        c, d = source.value_type(left.outputs[0]), target.value_type(right.outputs[0])
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and c.dtype == d.dtype
            and c.shape == d.shape
        )
    except FusionUnfinished:
        return False


def restore_integer_branches(
    source_views: dict[tuple[str, ...], View],
    virtual: Graph,
    registry: Registry,
    original_constants: set[Any],
) -> None:
    """Plan complete live integer CSE restoration, then commit without proof work."""
    targets = all_views(View(virtual, (), registry, original_constants))
    plans: dict[
        int,
        tuple[
            Graph,
            dict[int, list[Node]],
            dict[str, ValueType],
            dict[int, tuple[str, ...]],
        ],
    ] = {}
    serial = 0
    for scope, target in targets.items():
        require(scope in source_views, "integer restoration lexical scope differs")
        source = source_views[scope]
        for root in target.graph.nodes:
            if (
                root.op != "Cast"
                or root.domain not in STANDARD
                or len(root.outputs) != 1
                or target.value_type(root.outputs[0]).dtype not in {6, 7}
            ):
                continue
            tag = registry.tag(root)
            matches = [
                node
                for node in source.graph.nodes
                if node.op == "Cast" and registry.tag(node) == tag
            ]
            if not any(_integer_normalizations(source, node) >= 2 for node in matches):
                continue

            def direct(view: View, node: Node) -> tuple[Any, ...]:
                return tuple(
                    sorted(
                        (registry.tag(user), port)
                        for user, port in view.consumers.get(node.outputs[0], ())
                    )
                )

            if all(direct(source, node) == direct(target, root) for node in matches):
                continue
            matches = [
                node
                for node in matches
                if _integer_cast_binding(source, node, target, root)
            ]
            if matches and all(
                direct(source, node) == direct(target, root) for node in matches
            ):
                continue
            require(
                len(matches) == 1,
                "integer root Cast source binding is missing or ambiguous",
            )
            left = matches[0]
            require(
                sum(
                    node.op == "Cast"
                    and registry.tag(node) == tag
                    and _integer_cast_binding(source, left, target, node)
                    for node in target.graph.nodes
                )
                == 1,
                "integer target Cast binding is ambiguous",
            )
            token = (scope, tag, root.outputs[0])
            original, actual = _IntegerCone(source, left, token), _IntegerCone(
                target, root, token
            )
            branches, representatives = original.triples(), actual.triples()
            require(
                len(branches) >= 2 and len(representatives) == 1,
                "integer CSE has no unique shared triple",
            )
            representative = representatives[0]

            def roles(triple: tuple[Node, Node, Node]) -> tuple[str, ...]:
                return tuple(registry.tag(node) for node in triple)

            kept = [
                branch for branch in branches if roles(branch) == roles(representative)
            ]
            require(
                len(kept) == 1
                and len({roles(branch) for branch in branches}) == len(branches),
                "integer CSE representative source roles differ",
            )
            require(
                original.endpoints.keys() == actual.endpoints.keys(),
                "integer endpoint coverage differs",
            )
            gathers = {
                (node.outputs[0], actual.endpoints[key][0].outputs[0])
                for key, (node, _) in original.endpoints.items()
                if node.op == "Gather"
            }
            partitions: dict[int, list[tuple[str, int]]] = {
                id(branch[2]): [] for branch in branches
            }
            for key, (endpoint, value) in original.endpoints.items():
                other, current = actual.endpoints[key]
                require(
                    original.expression(value) == actual.expression(current),
                    "integer endpoint ordered expression differs",
                )
                _integer_endpoint_binding(
                    source, endpoint, target, other, key[1], gathers
                )
                parents = original.ancestors(value)
                owners = [
                    branch for branch in branches if branch[2].outputs[0] in parents
                ]
                require(
                    len(owners) == 1, "integer endpoint has ambiguous source branch"
                )
                partitions[id(owners[0][2])].append(key)
            for branch in branches:
                require(
                    all(
                        original.expression(a.outputs[0])
                        == actual.expression(b.outputs[0])
                        for a, b in zip(branch, representative, strict=True)
                    ),
                    "integer source triple does not correspond",
                )
                endpoints = partitions[id(branch[2])]
                require(
                    len(endpoints) == 2 and {key[1] for key in endpoints} == {0, 1},
                    "integer branch index/mask endpoint partition differs",
                )
                require(
                    {original.endpoints[key][0].op for key in endpoints}
                    == {"Gather", "Where"},
                    "integer branch endpoint kinds differ",
                )
            # Build only target-derived live closures; no graph object is changed here.
            _, before, updates, rewires = plans.setdefault(
                id(target.graph), (target.graph, {}, {}, {})
            )
            names = (
                set(target.graph.inputs)
                | set(target.graph.initializers)
                | set(target.producers)
                | set(target.graph.types)
            )
            for branch in branches:
                if branch is kept[0]:
                    continue
                serial += 1
                prefix = f"__jax2onnx_virtual_integer_{serial}_"
                copied: dict[str, str] = {}
                triple_roles = {
                    id(b): registry.tag(a)
                    for a, b in zip(branch, representative, strict=True)
                }
                added: list[Node] = []

                def clone(name: str) -> str:
                    if name == root.outputs[0]:
                        return name
                    if name in copied:
                        return copied[name]
                    owner = target.owner(name)
                    constant = target.constant(name)
                    if name in owner.graph.initializers:
                        require(
                            name not in owner.graph.inputs and constant is not None,
                            "integer clone constant binding differs",
                        )
                        return name
                    require(
                        owner is target
                        and name in target.producers
                        and (name in actual.nodes or constant is not None),
                        "integer clone has unproved dependency",
                    )
                    template, port = target.producers[name]
                    require(
                        template.op != "Cast"
                        and port == 0
                        and len(template.outputs) == 1,
                        "integer clone has runtime Cast or multiple outputs",
                    )
                    arguments = tuple(
                        clone(arg) if arg else "" for arg in template.inputs
                    )
                    fresh = prefix + str(len(names))
                    require(fresh not in names, "integer clone name collision")
                    names.add(fresh)
                    node = deepcopy(template)
                    node.name = prefix + triple_roles.get(id(template), template.name)
                    node.inputs, node.outputs = arguments, (fresh,)
                    added.append(node)
                    updates[fresh] = target.value_type(name)
                    copied[name] = fresh
                    return fresh

                endpoint_ids = {
                    id(actual.endpoints[key][0]): key
                    for key in partitions[id(branch[2])]
                }
                for endpoint in target.graph.nodes:
                    endpoint_key = endpoint_ids.get(id(endpoint))
                    if endpoint_key is None:
                        continue
                    require(
                        id(endpoint) not in rewires,
                        "integer endpoint planned more than once",
                    )
                    start = len(added)
                    replacement = clone(actual.endpoints[endpoint_key][1])
                    before.setdefault(id(endpoint), []).extend(added[start:])
                    rewires[id(endpoint)] = tuple(
                        replacement if index == endpoint_key[1] else arg
                        for index, arg in enumerate(endpoint.inputs)
                    )
    for graph, before, updates, rewires in plans.values():
        nodes: list[Node] = []
        for node in graph.nodes:
            nodes.extend(before.get(id(node), ()))
            if id(node) in rewires:
                node.inputs = rewires[id(node)]
            nodes.append(node)
        graph.nodes = nodes
        graph.types.update(updates)


def expand_proven_fusions(
    original: Graph,
    optimized: Graph,
    roles: dict[str, tuple[str, str, str]],
    *,
    fp32_post_mul_contract: str | None = None,
    refresh_types: Callable[[Graph], Graph] | None = None,
    initialize_types: Callable[[Graph], None] | None = None,
) -> tuple[Graph, tuple[str, ...]]:
    """Return a private virtual graph; neither supplied artifact is mutated."""
    registry = Registry(roles)
    original_constants: set[Any] = set()

    def constants(graph: Graph) -> None:
        original_constants.update(value.key for value in graph.initializers.values())
        for node in graph.nodes:
            for _, _, child in children(node):
                constants(child)

    constants(original)
    source_views = all_views(View(original, (), registry, original_constants))
    virtual = deepcopy(optimized)
    if initialize_types is not None:
        initialize_types(virtual)
    restore_constant_origins(source_views, virtual, registry, original_constants)
    matched: list[tuple[tuple[str, ...], Match, str]] = []
    while True:
        target_views = all_views(View(virtual, (), registry, original_constants))
        pending = [
            (scope, view, node)
            for scope, view in target_views.items()
            for node in view.graph.nodes
            if node.op == "Gemm"
            or node.domain == "com.microsoft"
            and (node.op == "QuickGelu" or node.op == "FusedMatMul")
        ]
        if not pending:
            break
        scope, target, node = pending[0]
        require(scope in source_views, "fusion has no exact original nested scope")
        source = source_views[scope]
        if node.op == "FusedMatMul":
            fused = match_fused_matmul(source, target, node, fp32_post_mul_contract)
            actual_output = node.outputs[0]
            restore_fused_matmul(target, fused, len(matched))
            matched.append(
                (
                    scope,
                    Match(
                        source,
                        (
                            (fused.matrix,)
                            if fused.scale is None
                            else (fused.matrix, fused.scale)
                        ),
                        fused.matrix.inputs,
                        node,
                        fused.terminal,
                        "FusedMatMul",
                    ),
                    actual_output,
                )
            )
            if refresh_types is not None:
                virtual = refresh_types(virtual)
            continue
        match = (
            match_gemm(source, target, node)
            if node.op == "Gemm"
            else match_quick_gelu(source, target, node)
        )
        actual_output = node.outputs[0]
        Rewriter(target, match, len(matched)).apply()
        matched.append((scope, match, actual_output))
        if refresh_types is not None:
            virtual = refresh_types(virtual)
    restore_integer_branches(source_views, virtual, registry, original_constants)
    final_views = all_views(View(virtual, (), registry, original_constants))
    prove_repeated_if_bindings(source_views, final_views)
    for scope, match, actual_output in matched:
        require(
            match.source.edges(match.terminal)
            == final_views[scope].edges(actual_output),
            "fusion terminal consumers, direct outputs or sibling edges changed",
        )
    return virtual, tuple(match.kind for _, match, _ in matched)
