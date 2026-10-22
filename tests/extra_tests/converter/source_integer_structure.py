# tests/extra_tests/converter/source_integer_structure.py

"""Prospective SOURCE integer structure; original callbacks remain observations.

This semantic counterpart supplies no numerical/state criterion or primitive
admission. Its qualification anchors are the original native INT32 iota rows
and the same native integer/index family inside the unchanged GPT source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from types import FunctionType
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import onnx

from jax2onnx.converter.source_representation import native_shape_contract
from jax2onnx.precision import CheckStatus
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from tests.extra_tests.converter.source_native_input_usage import (
    NativeInputUsage,
    check_native_input_usage,
)
from tests.extra_tests.converter.source_precision_corpus_oracles import (
    CorpusStructureEvidence,
)


VERSION = "source-int32-range-select-gather-structure-v1"


def _body(value: Any) -> Any:
    inner = getattr(value, "jaxpr", None)
    return inner if inner is not None and inner is not value else value


def _equations(value: Any, ancestors: frozenset[int] = frozenset()) -> list[Any]:
    body = _body(value)
    if id(body) in ancestors or not hasattr(body, "eqns"):
        return []
    result = []
    for equation in body.eqns:
        result.append(equation)
        for parameter in equation.params.values():
            candidates = (
                parameter if isinstance(parameter, (tuple, list)) else (parameter,)
            )
            for candidate in candidates:
                result.extend(_equations(candidate, ancestors | {id(body)}))
    return result


def _properties(value: Any) -> tuple[int, tuple[int | str, ...]]:
    tensor = value.type.tensor_type
    return tensor.elem_type, tuple(
        d.dim_param if d.dim_param else d.dim_value for d in tensor.shape.dim
    )


def _attribute(node: Any, name: str) -> Any:
    value = next(a for a in node.attribute if a.name == name)
    return onnx.helper.get_attribute_value(value)


def _called_scopes(
    model: Any, *, native_input_usage: NativeInputUsage | None = None
) -> list[tuple[Any, bool, set[str]]]:
    definitions = {(f.domain, f.name, f.overload): f for f in model.functions}
    required: dict[tuple[str, str, str], set[int]] = {}
    pending = [(model.graph, False, set(v.name for v in model.graph.output))]
    result: dict[int, tuple[Any, bool, set[str]]] = {}
    while pending:
        scope, function, outputs = pending.pop()
        producers = {v: n for n in scope.node for v in n.output}
        live = set(outputs)
        queue = list(outputs)
        while queue:
            node = producers.get(queue.pop())
            if node is None:
                continue
            for name in node.input:
                if name and name not in live:
                    live.add(name)
                    queue.append(name)
        previous = result.get(id(scope))
        if previous is not None:
            live.update(previous[2])
        result[id(scope)] = (scope, function, live)
        for node in scope.node:
            if not any(name in live for name in node.output):
                continue
            key = node.domain, node.op_type, node.overload
            if key in definitions:
                positions = {i for i, name in enumerate(node.output) if name in live}
                extra = positions - required.get(key, set())
                if extra:
                    required.setdefault(key, set()).update(extra)
                    definition = definitions[key]
                    pending.append(
                        (
                            definition,
                            True,
                            {definition.output[i] for i in required[key]},
                        )
                    )
            for attribute in node.attribute:
                graphs = (
                    (attribute.g,)
                    if attribute.type == onnx.AttributeProto.GRAPH
                    else (
                        attribute.graphs
                        if attribute.type == onnx.AttributeProto.GRAPHS
                        else ()
                    )
                )
                for graph in graphs:
                    pending.append((graph, function, {v.name for v in graph.output}))
    # Default remains strict. Only exact actual native declaration/body/call
    # correspondence can factor the original no-unused convention.
    check_native_input_usage(model, native_input_usage)
    return list(result.values())


class _Scope:
    def __init__(self, scope: Any, live: set[str]) -> None:
        self.scope = scope
        self.live = live
        self.producers = {name: n for n in scope.node for name in n.output}
        self.types = {v.name: _properties(v) for v in scope.value_info}
        self.types.update(
            {v.name: _properties(v) for v in scope.input if not isinstance(v, str)}
        )
        self.types.update(
            {v.name: _properties(v) for v in scope.output if not isinstance(v, str)}
        )
        self.initializers = {v.name: v for v in getattr(scope, "initializer", ())}

    def scalar(self, name: str, dtype: int) -> Any:
        tensor = self.initializers.get(name)
        if tensor is None:
            node = self.producers[name]
            if node.op_type != "Constant":
                raise ValueError("Integer proof requires the actual scalar constant")
            tensor = _attribute(node, "value")
        if tensor.data_type != dtype or tuple(tensor.dims) != ():
            raise ValueError("Integer proof scalar constant dtype/rank changed")
        return onnx.numpy_helper.to_array(tensor).item()

    def vector(self, name: str, dtype: int, length: int) -> tuple[int, ...]:
        tensor = self.initializers.get(name)
        if tensor is None:
            node = self.producers[name]
            if node.op_type != "Constant":
                raise ValueError(
                    "Integer proof requires the actual axis/shape constant"
                )
            tensor = _attribute(node, "value")
        if tensor.data_type != dtype or tuple(tensor.dims) != (length,):
            raise ValueError("Integer proof vector dtype/shape changed")
        return tuple(int(v) for v in onnx.numpy_helper.to_array(tensor))

    def integer_data(self, name: str, dtype: int = onnx.TensorProto.INT32) -> None:
        if self.types.get(name, (None, ()))[0] != dtype:
            raise ValueError("Native integer data dtype changed")

    def descendants(self, name: str) -> set[str]:
        # Only data-preserving integer paths. A predicate is not an index arm.
        reached = {name}
        for node in self.scope.node:
            data = node.input[1:] if node.op_type == "Where" else node.input[:1]
            if node.op_type == "Add":
                data = node.input
            if not any(v in reached for v in data):
                continue
            if node.op_type == "Cast":
                raise ValueError("SOURCE INT32 index data acquired a conversion")
            if node.op_type in {
                "Identity",
                "Unsqueeze",
                "Expand",
                "Reshape",
                "Add",
                "Where",
                "Max",
                "Min",
            }:
                for output in node.output:
                    self.integer_data(output)
                    reached.add(output)
        return reached

    def clipped_gather(
        self, gather: Any, prefix: tuple[int | str, ...], native: Any
    ) -> None:
        native_data = native.invars[0].aval
        data_shape = native_shape_contract(native_data.shape).shape
        dtype = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(native_data.dtype))
        if self.types[gather.input[0]] != (dtype, data_shape):
            raise ValueError("Native embedding operand dtype/shape changed")
        dimensions = native.params["dimension_numbers"]
        if (
            tuple(dimensions.offset_dims)
            != tuple(range(len(prefix), len(prefix) + len(data_shape) - 1))
            or tuple(dimensions.collapsed_slice_dims) != (0,)
            or tuple(dimensions.start_index_map) != (0,)
            or tuple(dimensions.operand_batching_dims)
            or tuple(dimensions.start_indices_batching_dims)
            or tuple(native.params["slice_sizes"]) != (1, *data_shape[1:])
        ):
            raise ValueError("Native embedding gather dimension/slice schema changed")
        if (
            _attribute(gather, "axis") != 0
            or not data_shape
            or not isinstance(data_shape[0], int)
        ):
            raise ValueError("Native embedding gather axis/extent changed")
        maximum = data_shape[0] - 1
        if not 0 <= maximum < (1 << 31):
            raise ValueError("Native INT32 gather bound is not representable")
        minimum = self.producers[gather.input[1]]
        if minimum.op_type != "Min":
            raise ValueError("Native gather lost its upper clipping guard")
        lower = self.producers[minimum.input[0]]
        if lower.op_type != "Max":
            raise ValueError("Native gather lost its lower clipping guard")
        indices, zero = lower.input
        bound = minimum.input[1]
        self.integer_data(indices)
        if (
            self.scalar(zero, onnx.TensorProto.INT32) != 0
            or self.scalar(bound, onnx.TensorProto.INT32) != maximum
        ):
            raise ValueError("Native INT32 gather clipping bounds changed")
        if self.types[indices][1] != prefix:
            raise ValueError("Native embedding index shape changed")
        below = [
            n
            for n in self.scope.node
            if n.op_type == "Less" and list(n.input) == [indices, zero]
        ]
        above = [
            n
            for n in self.scope.node
            if n.op_type == "Greater" and list(n.input) == [indices, bound]
        ]
        if len(below) != 1 or len(above) != 1:
            raise ValueError("Native fill/drop invalid-index comparisons changed")
        invalid = [
            n
            for n in self.scope.node
            if n.op_type == "Or"
            and list(n.input) == [below[0].output[0], above[0].output[0]]
        ]
        if len(invalid) != 1:
            raise ValueError("Native fill/drop invalid-index mask changed")
        masks = {invalid[0].output[0]}
        for node in self.scope.node:
            if (
                node.op_type in {"Unsqueeze", "Expand", "Reshape", "Identity"}
                and node.input[0] in masks
            ):
                masks.update(node.output)
        fills = [
            n
            for n in self.scope.node
            if n.op_type == "Where"
            and n.input[0] in masks
            and n.input[2] == gather.output[0]
        ]
        if len(fills) != 1:
            raise ValueError("Native fill/drop result guard changed")
        fill = fills[0]
        if fill.output[0] not in self.live:
            raise ValueError(
                "Native fill/drop chain does not reach a live scope/root output"
            )
        dtype, shape = self.types[gather.output[0]]
        if self.types[fill.output[0]] != (dtype, shape) or shape != (
            *prefix,
            *data_shape[1:],
        ):
            raise ValueError("Native fill/drop output dtype/shape changed")
        if dtype not in (
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
        ) or not np.isnan(self.scalar(fill.input[1], dtype)):
            raise ValueError("Native default fill NaN changed")


def has_integer_source_family(native_trace: Any) -> bool:
    return any(
        e.primitive is jax.lax.iota_p
        and np.dtype(e.params["dtype"]) == np.dtype(np.int32)
        and isinstance(e.params["shape"][e.params["dimension"]], int)
        and 0 <= e.params["shape"][e.params["dimension"]] <= (1 << 24)
        for e in _equations(native_trace)
    )


def _native_integer_roles(native_trace: Any) -> list[tuple[Any, list[Any]]]:
    roles: dict[int, tuple[Any, dict[int, Any]]] = {}
    copies = {
        jax.lax.broadcast_in_dim_p,
        jax.lax.reshape_p,
        jax.lax.squeeze_p,
        jax.lax.transpose_p,
        jax.lax.add_p,
        jax.lax.convert_element_type_p,
    }

    def visit(
        closed: Any, arguments: list[set[int]], ancestors: frozenset[int]
    ) -> list[set[int]]:
        body = _body(closed)
        if id(body) in ancestors:
            raise ValueError("Recursive native integer proof")
        if len(arguments) != len(body.invars):
            raise ValueError("Native integer carrier operand/formal count differs")
        provenance = {
            id(v): set(origins)
            for v, origins in zip(body.invars, arguments, strict=True)
        }
        for equation in body.eqns:
            incoming = [provenance.get(id(v), set()) for v in equation.invars]
            outgoing: list[set[int]] = [set() for _ in equation.outvars]
            if equation.primitive is jax.lax.iota_p and np.dtype(
                equation.params["dtype"]
            ) == np.dtype(np.int32):
                extent = equation.params["shape"][equation.params["dimension"]]
                if isinstance(extent, int) and 0 <= extent <= (1 << 24):
                    origin = id(equation)
                    roles.setdefault(origin, (equation, {}))
                    outgoing = [{origin}]
            elif equation.primitive.name == "jit" and "jaxpr" in equation.params:
                outgoing = visit(
                    equation.params["jaxpr"], incoming, ancestors | {id(body)}
                )
            elif equation.primitive is jax.lax.gather_p:
                for origin in incoming[1]:
                    roles[origin][1][id(equation)] = equation
            elif (
                equation.primitive in copies or equation.primitive is jax.lax.select_n_p
            ):
                operands = (
                    incoming[1:]
                    if equation.primitive is jax.lax.select_n_p
                    else incoming
                )
                origins = set().union(*operands)
                if all(
                    np.dtype(v.aval.dtype) == np.dtype(np.int32)
                    for v in equation.outvars
                ):
                    outgoing = [set(origins) for _ in equation.outvars]
            for variable, origins in zip(equation.outvars, outgoing, strict=True):
                provenance[id(variable)] = origins
        return [provenance.get(id(v), set()) for v in body.outvars]

    body = _body(native_trace)
    visit(native_trace, [set() for _ in body.invars], frozenset())
    return [(equation, list(gathers.values())) for equation, gathers in roles.values()]


def check_integer_source_structure(
    native_trace: Any,
    model: Any,
    *,
    require_function: bool = False,
    native_input_usage: NativeInputUsage | None = None,
) -> dict[str, Any]:
    specs = []
    for equation, native_gathers in _native_integer_roles(native_trace):
        if equation.primitive is not jax.lax.iota_p or np.dtype(
            equation.params["dtype"]
        ) != np.dtype(np.int32):
            continue
        shape = native_shape_contract(equation.params["shape"]).shape
        axis = equation.params["dimension"]
        extent = shape[axis]
        if not isinstance(extent, int) or not 0 <= extent <= (1 << 24):
            continue
        if (
            equation.invars
            or len(equation.outvars) != 1
            or equation.outvars[0].aval.weak_type is not False
        ):
            raise ValueError("Original native INT32 iota schema changed")
        specs.append((shape, axis, extent, native_gathers))
    if not specs:
        raise ValueError(
            "SOURCE integer counterpart requires an actual bounded native INT32 iota"
        )
    scopes = [
        (_Scope(scope, live), function)
        for scope, function, live in _called_scopes(
            model, native_input_usage=native_input_usage
        )
    ]
    checked = []
    for shape, axis, extent, gathers in specs:
        matches = []
        for scope, function in scopes:
            if require_function and not function:
                continue
            for node in scope.scope.node:
                if node.op_type != "Range" or scope.types.get(node.output[0]) != (
                    onnx.TensorProto.INT32,
                    (extent,),
                ):
                    continue
                if [scope.scalar(n, onnx.TensorProto.INT32) for n in node.input] != [
                    0,
                    extent,
                    1,
                ]:
                    continue
                reachable = scope.descendants(node.output[0])
                if not any(
                    scope.types.get(name) == (onnx.TensorProto.INT32, shape)
                    and name in scope.live
                    for name in reachable
                ):
                    continue
                if len(shape) > 1:
                    unsqueezes = [
                        n
                        for n in scope.scope.node
                        if n.op_type == "Unsqueeze" and n.input[0] == node.output[0]
                    ]
                    if len(unsqueezes) != 1:
                        raise ValueError("Native broadcasted iota lost its Unsqueeze")
                    if scope.vector(
                        unsqueezes[0].input[1], onnx.TensorProto.INT64, len(shape) - 1
                    ) != tuple(i for i in range(len(shape)) if i != axis):
                        raise ValueError("Native broadcasted iota axis changed")
                    if not any(
                        n.op_type == "Expand"
                        and n.input[0] == unsqueezes[0].output[0]
                        and scope.types.get(n.output[0])
                        == (onnx.TensorProto.INT32, shape)
                        for n in scope.scope.node
                    ):
                        raise ValueError(
                            "Native broadcasted iota lost its typed Expand"
                        )
                nodes = [
                    n
                    for n in scope.scope.node
                    if n.op_type == "Gather" and n.input[1] in reachable
                ]
                if not gathers and nodes:
                    raise ValueError(
                        "SOURCE Range reaches a gather unrelated to its exact native iota"
                    )
                if gathers:
                    if any(
                        e.params["mode"] is not jax.lax.GatherScatterMode.FILL_OR_DROP
                        or not np.isnan(e.params["fill_value"])
                        for e in gathers
                    ):
                        raise ValueError(
                            "Native index counterpart needs the exact default fill/drop gather schema"
                        )
                    if len(nodes) != 1:
                        raise ValueError(
                            "Native iota does not feed one actual embedding gather"
                        )
                    if len(gathers) != 1:
                        raise ValueError("Native embedding gather role is not unique")
                    scope.clipped_gather(nodes[0], shape, gathers[0])
                matches.append(
                    {
                        "function": function,
                        "shape": shape,
                        "axis": axis,
                        "extent": extent,
                        "fill_drop_gather": bool(gathers),
                    }
                )
        if len(matches) != 1:
            raise ValueError(
                "Native INT32 iota has no unique called SOURCE typed counterpart"
            )
        checked.extend(matches)
    return {
        "version": VERSION,
        "native_specs": checked,
        "source_structure": "validated",
        "numerical_acceptance": "UNCHANGED",
    }


def check_integer_callback_counterpart(
    callback: Any, model: Any, *, native_input_usage: NativeInputUsage | None = None
) -> dict[str, Any]:
    if (
        type(callback) is not FunctionType
        or callback.__code__ is not expect_graph([]).__code__
    ):
        raise ValueError(
            "Integer structural amendment requires the original expect_graph callback"
        )
    captured = inspect.getclosurevars(callback).nonlocals
    specifications = []
    amended = []
    for original in captured["specs"]:
        if isinstance(original, str):
            spec = {"path": original}
        elif type(original) is dict:
            spec = dict(original)
        elif type(original) is tuple:
            spec = {"path": original[0], **original[1]}
        else:
            raise ValueError("Unreviewed integer callback specification")
        steps = [v.strip() for v in spec["path"].split("->")]
        operations = [v.split(":", 1)[0].strip() for v in steps]
        reviewed_cast_paths = {
            ("Range", "Cast"),
            ("Range", "Unsqueeze", "Expand", "Cast"),
        }
        if tuple(operations) == ("Range", "Unsqueeze", "Expand", "Gather"):
            if any(k in spec for k in ("attrs", "inputs")):
                raise ValueError(
                    "Integer callback has an unreviewed operator-specific condition"
                )
            # Only this reviewed path syntax is replaced. Typed native dataflow,
            # bounds, mask and fill are independently mandatory; keep the exact
            # Range suffix and every selector/global/other-clause requirement.
            spec["path"] = steps[0]
        elif tuple(operations) in reviewed_cast_paths:
            spec["path"] = " -> ".join(steps[:-1])
        # Every unreviewed sequence stays with the original matcher, including
        # an unrelated operation embedded between Range and Gather/Cast.
        if spec["path"] != " -> ".join(steps):
            amended.append({"original": original, "source": spec})
        specifications.append(spec)
    options = {
        k: captured[k]
        for k in (
            "mode",
            "must_absent",
            "no_unused_inputs",
            "no_unused_function_inputs",
            "search_functions",
            "symbols",
            "explain_on_fail",
        )
    }
    unused_correspondence = None
    if native_input_usage is not None:
        # Revalidate the exact artifact before factoring these two options. All
        # other original callback selectors, flags and conditions stay exact.
        unused_correspondence = check_native_input_usage(model, native_input_usage)
        if options["no_unused_function_inputs"] and not options["search_functions"]:
            raise ValueError(
                "Original no-unused Function option requires search_functions"
            )
        options["no_unused_inputs"] = False
        options["no_unused_function_inputs"] = False
    if not expect_graph(specifications, **options)(model):
        raise ValueError(
            "SOURCE counterpart failed a preserved original callback obligation"
        )
    return {
        "version": VERSION,
        "amended_integer_paths": amended,
        "other_callback_obligations": "preserved and passed",
        "native_unused_correspondence": unused_correspondence,
    }


@dataclass(frozen=True)
class Int32IndexCounterpart:
    """The same semantic policy is used by raw iota and named embedding scopes."""

    def check(self, prepared: Any, model: Any) -> CorpusStructureEvidence:
        callback = prepared.metadata.get("post_check_onnx_graph")
        raw_status, raw_detail = "absent", "No original callback"
        if callback is not None:
            try:
                result = bool(callback(model))
                raw_status, raw_detail = (
                    "passed" if result else "failed"
                ), f"Original callback returned {result}"
            except Exception as exc:
                raw_status, raw_detail = "error", f"{type(exc).__name__}: {exc}"
        try:
            result = check_integer_source_structure(prepared.original_jaxpr, model)
            result["callback_counterpart"] = check_integer_callback_counterpart(
                callback, model
            )
            status, detail = CheckStatus.VALIDATED, json.dumps(result, sort_keys=True)
        except Exception as exc:
            status, detail = CheckStatus.FAILED, str(exc)
        digest = hashlib.sha256(
            VERSION.encode() + Path(__file__).read_bytes()
        ).hexdigest()
        return CorpusStructureEvidence(
            VERSION, digest, raw_status, raw_detail, status, detail
        )
