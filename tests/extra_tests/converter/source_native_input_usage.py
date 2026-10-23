# tests/extra_tests/converter/source_native_input_usage.py

"""Exact test-side native declaration correspondence; no numerical amendment."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from typing import Any
from unittest.mock import patch

import onnx
import onnx_ir as ir

from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_representation import representation_for_aval


VERSION = "source-native-unused-declaration-correspondence-v1"


def _body(value: Any) -> Any:
    inner = getattr(value, "jaxpr", None)
    return inner if inner is not None and inner is not value else value


def native_unused_positions(closed: Any) -> frozenset[int]:
    """Count identity references, including results, lexical captures and effects."""
    body = _body(closed)
    formals = {id(var): index for index, var in enumerate(body.invars)}
    used: set[int] = set()
    visited: set[int] = set()

    def visit(value: Any) -> None:
        identity = id(value)
        if identity in formals:
            used.add(formals[identity])
            return
        if identity in visited:
            return
        visited.add(identity)
        candidate = _body(value)
        if hasattr(candidate, "eqns") and hasattr(candidate, "invars"):
            if candidate.effects:
                # Explicit state/key passthrough is already a result reference.
                # Other effect dependence has no unused-input proof here.
                used.update(formals.values())
            for result in candidate.outvars:
                visit(result)
            for equation in candidate.eqns:
                if equation.effects:
                    used.update(formals.values())
                for operand in equation.invars:
                    visit(operand)
                visit(equation.params)
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(closed)
    return frozenset(formals.values()) - used


def _children(scope: Any) -> Iterator[Any]:
    for node in scope.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield attribute.g
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                yield from attribute.graphs


def _name(value: Any) -> str:
    return value if isinstance(value, str) else value.name


def _lexical_usage(scope: Any) -> tuple[set[str], set[str]]:
    """Return own declared unused inputs and free names captured from ancestors."""
    inputs = {_name(value) for value in scope.input}
    bound = inputs | {value.name for value in getattr(scope, "initializer", ())}
    bound.update(name for node in scope.node for name in node.output if name)
    used = {_name(value) for value in scope.output}
    used.update(name for node in scope.node for name in node.input if name)
    for child in _children(scope):
        _, captures = _lexical_usage(child)
        used.update(captures)
    return inputs - used, used - bound


def _scopes(scope: Any) -> Iterator[Any]:
    yield scope
    for child in _children(scope):
        yield from _scopes(child)


def _identifier(value: Any) -> tuple[str, str, str]:
    return value.domain, value.name, value.overload


def _node_identifier(value: Any) -> tuple[str, str, str]:
    return value.domain, value.op_type, value.overload


def _bytes(value: Any) -> bytes:
    return value.SerializeToString(deterministic=True)


@dataclass(eq=False)
class _BuildFrame:
    plugin: Any
    ctx: Any
    callee: Any
    equation: Any
    captures: Any
    source_trace: Any
    body: Any = None
    definition: Any = None
    parent_values: tuple[Any, ...] = ()
    unused: frozenset[int] = frozenset()
    signature: tuple[Any, ...] = ()


@dataclass(eq=False)
class _Call:
    enclosing: _BuildFrame | None
    definition: _BuildFrame
    equation: Any
    node: Any
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]


@dataclass(eq=False)
class NativeInputUsage:
    """Observe exact nested build frames, body Vars, definitions and call edges.

    Names locate an already identity-proven emitted edge in its final serialized
    scope; names/types/shapes never pair a native body with a definition. Root
    unused declarations remain strict: the fixed-kwargs model oracle is not a
    runtime-operand root trace.
    """

    _stack: list[_BuildFrame] = field(default_factory=list)
    _definitions: dict[int, _BuildFrame] = field(default_factory=dict)
    _calls: list[_Call] = field(default_factory=list)
    _artifact: Any = None
    _artifact_digest: str | None = None
    _ir_model: Any = None

    @contextmanager
    def build(
        self,
        plugin: Any,
        ctx: Any,
        callee: Any,
        eqn: Any,
        captures: Any,
        *,
        source_trace: Any = None,
    ) -> Iterator[_BuildFrame]:
        if ctx.precision_policy is not PrecisionPolicy.SOURCE or source_trace is None:
            raise ValueError(
                "Native declaration proof requires the actual SOURCE call-site context"
            )
        frame = _BuildFrame(plugin, ctx, callee, eqn, captures, source_trace)
        frame.parent_values = (
            *(ctx.require_value_for_var(var) for var in eqn.invars),
            *(capture.value for capture in captures.dynamic),
        )
        self._stack.append(frame)
        try:
            yield frame
        finally:
            if not self._stack or self._stack.pop() is not frame:
                raise ValueError("Native build-frame stack lost exact ownership")

    def retain_body(self, plugin: Any, body: Any) -> None:
        if not self._stack or self._stack[-1].plugin is not plugin:
            raise ValueError("Native Function body has no exact active build frame")
        frame = self._stack[-1]
        if frame.body is not None:
            raise ValueError("One Function build returned multiple native bodies")
        frame.body = body

    def retain_definition(self, frame: _BuildFrame, definition: Any) -> None:
        if not self._stack or self._stack[-1] is not frame or frame.body is None:
            raise ValueError("Returned Function has no exact native build/body pair")
        if id(definition) in self._definitions:
            raise ValueError("Returned Function definition was already paired")
        body = _body(frame.body)
        inputs, outputs = tuple(definition.graph.inputs), tuple(
            definition.graph.outputs
        )
        if len(inputs) != len(body.invars) or len(inputs) != len(frame.parent_values):
            raise ValueError("Native/body/Function input arity or captures changed")
        if len(outputs) != len(body.outvars):
            raise ValueError("Native/Function output arity changed")
        for variables, values in ((body.invars, inputs), (body.outvars, outputs)):
            for var, value in zip(variables, values, strict=True):
                if not representation_for_aval(var.aval).matches_ir_value(value):
                    raise ValueError("Native logical-to-physical Function port changed")
        for var, value in zip(body.invars, frame.parent_values, strict=True):
            if not representation_for_aval(var.aval).matches_ir_value(value):
                raise ValueError(
                    "Reordered source operands/captures changed representation"
                )
        frame.definition = definition
        frame.unused = native_unused_positions(frame.body)
        frame.signature = (
            _identifier(definition),
            tuple(value.name for value in inputs),
            tuple(value.name for value in outputs),
        )
        self._definitions[id(definition)] = frame

    def emit(
        self, original: Any, ctx: Any, eqn: Any, definition: Any, captures: Any
    ) -> Any:
        frame = self._definitions.get(id(definition))
        if frame is None or frame.definition is not definition:
            raise ValueError(
                "Cached Function has no exact native definition association"
            )
        inputs = (
            *(ctx.require_value_for_var(var) for var in eqn.invars),
            *(capture.value for capture in captures.dynamic),
        )
        native_formals = _body(frame.body).invars
        if len(inputs) != len(native_formals) or any(
            not representation_for_aval(var.aval).matches_ir_value(value)
            for var, value in zip(native_formals, inputs, strict=True)
        ):
            raise ValueError(
                "Cached call changed native logical-to-physical input ports"
            )
        before = {id(node) for node in ctx.builder.graph}
        result = original(ctx, eqn, definition, captures)
        emitted = [node for node in ctx.builder.graph if id(node) not in before]
        if len(emitted) != 1 or _node_identifier(emitted[0]) != frame.signature[0]:
            raise ValueError("Actual native Function emission changed exact identifier")
        node = emitted[0]
        outputs = tuple(ctx.require_value_for_var(var) for var in eqn.outvars)
        if len(node.inputs) != len(inputs) or any(
            left is not right for left, right in zip(node.inputs, inputs, strict=True)
        ):
            raise ValueError("Actual Function call reordered native operand identities")
        if len(node.outputs) != len(outputs) or any(
            left is not right for left, right in zip(node.outputs, outputs, strict=True)
        ):
            raise ValueError("Actual Function call changed native output identities")
        self._calls.append(
            _Call(
                self._stack[-1] if self._stack else None,
                frame,
                eqn,
                node,
                inputs,
                outputs,
            )
        )
        return result

    def retain_model(self, model: Any) -> None:
        if self._ir_model is not None:
            raise ValueError("Native declaration observer returned multiple IR models")
        self._ir_model = model

    @contextmanager
    def observe(self) -> Iterator[NativeInputUsage]:
        from jax2onnx.converter import conversion_api
        from jax2onnx.plugins.plugin_system import FunctionPlugin

        original_build = FunctionPlugin._build_function
        original_body = FunctionPlugin._trace_function_body
        original_emit = FunctionPlugin._emit_function_call
        original_finalize = conversion_api._build_and_finalize_ir_model

        def finalize(*args: Any, **kwargs: Any) -> Any:
            model = original_finalize(*args, **kwargs)
            self.retain_model(model)
            return model

        def build(*args: Any, **kwargs: Any) -> Any:
            with self.build(*args, **kwargs) as frame:
                definition = original_build(*args, **kwargs)
                self.retain_definition(frame, definition)
                return definition

        def body(*args: Any, **kwargs: Any) -> Any:
            closed = original_body(*args, **kwargs)
            self.retain_body(args[0], closed)
            return closed

        def emit(*args: Any, **kwargs: Any) -> Any:
            return self.emit(original_emit, *args, **kwargs)

        with (
            patch.object(FunctionPlugin, "_build_function", build),
            patch.object(FunctionPlugin, "_trace_function_body", body),
            patch.object(FunctionPlugin, "_emit_function_call", staticmethod(emit)),
            patch.object(conversion_api, "_build_and_finalize_ir_model", finalize),
        ):
            yield self
        if self._stack:
            raise ValueError("Native build-frame observer leaked an active frame")

    def seal(self, model: Any) -> dict[str, Any]:
        if self._artifact is not None:
            raise ValueError("Native declaration proof already bound to an artifact")
        result = self.check(model)
        self._artifact, self._artifact_digest = model, result["artifact_sha256"]
        return result

    def check(self, model: Any) -> dict[str, Any]:
        if self._stack:
            raise ValueError("Native declaration proof has an unfinished build frame")
        digest = hashlib.sha256(_bytes(model)).hexdigest()
        if self._artifact is not None and (
            model is not self._artifact or digest != self._artifact_digest
        ):
            raise ValueError("Native declaration proof artifact is stale or different")
        if (
            self._ir_model is None
            or hashlib.sha256(_bytes(ir.to_proto(self._ir_model))).hexdigest() != digest
        ):
            raise ValueError("Artifact differs from exact observed final IR model")
        definitions = {_identifier(value): value for value in model.functions}
        if len(definitions) != len(model.functions):
            raise ValueError("Artifact has ambiguous Function identifiers/overloads")
        paired = {frame.signature[0]: frame for frame in self._definitions.values()}
        if len(paired) != len(self._definitions) or definitions.keys() != paired.keys():
            raise ValueError(
                "Artifact Function has no unique actual native definition pair"
            )
        allowed = []
        for key, frame in paired.items():
            actual = definitions[key]
            if (tuple(actual.input), tuple(actual.output)) != frame.signature[1:]:
                raise ValueError("Artifact Function formal arity/order changed")
            # The exact returned IR Function remains the assembly owner. The
            # full final-IR serialization above binds every actual body and call.
            if self._ir_model.functions.get(key) is not frame.definition:
                raise ValueError(
                    "Final IR Function lost its exact native build association"
                )
            unused, free = _lexical_usage(actual)
            if free:
                raise ValueError("Function contains an unbound lexical capture")
            for position, name in enumerate(actual.input):
                if name in unused:
                    if position not in frame.unused:
                        raise ValueError(
                            "Native-used Function formal became disconnected"
                        )
                    allowed.append(
                        dict(definition=list(key), position=position, name=name)
                    )
            for child in _scopes(actual):
                if child is not actual and _lexical_usage(child)[0]:
                    raise ValueError(
                        "Unused subgraph formal has no actual native mapping"
                    )
        if _lexical_usage(model.graph)[0]:
            raise ValueError("Unused root input has no runtime-operand native mapping")
        expected: dict[tuple[Any, str], _Call] = {}
        for call in self._calls:
            owner = call.enclosing.signature[0] if call.enclosing is not None else None
            key = owner, call.node.name
            if not call.node.name or key in expected:
                raise ValueError(
                    "Actual emitted Function edge has an ambiguous scope/name"
                )
            if (
                len(call.node.inputs) != len(call.inputs)
                or len(call.node.outputs) != len(call.outputs)
                or any(
                    left is not right
                    for left, right in zip(call.node.inputs, call.inputs, strict=True)
                )
                or any(
                    left is not right
                    for left, right in zip(call.node.outputs, call.outputs, strict=True)
                )
            ):
                raise ValueError(
                    "Observed native Function operand/output identity changed"
                )
            expected[key] = call
        actual_calls = set()
        for owner, scope in [(None, model.graph), *definitions.items()]:
            for child in _scopes(scope):
                if child is not scope and _lexical_usage(child)[0]:
                    raise ValueError(
                        "Unused subgraph formal has no actual native mapping"
                    )
                for node in child.node:
                    identity = _node_identifier(node)
                    if identity not in definitions:
                        if node.domain in {key[0] for key in definitions}:
                            raise ValueError(
                                "Local call identifier/overload has no exact definition"
                            )
                        continue
                    edge = owner, node.name
                    if edge in actual_calls or edge not in expected:
                        raise ValueError(
                            "Artifact call has no unique actual native emitted edge"
                        )
                    actual_calls.add(edge)
                    call = expected[edge]
                    if (
                        identity != call.definition.signature[0]
                        or tuple(node.input)
                        != tuple(value.name for value in call.inputs)
                        or tuple(node.output)
                        != tuple(value.name for value in call.outputs)
                        or len(node.input) != len(definitions[identity].input)
                        or len(node.output) != len(definitions[identity].output)
                    ):
                        raise ValueError(
                            "Artifact call changed exact native operands/arity/order"
                        )
        if actual_calls != expected.keys():
            raise ValueError("Observed native Function call disappeared from artifact")
        return dict(
            version=VERSION,
            artifact_sha256=digest,
            actual_native_function_definitions=len(paired),
            actual_native_function_calls=len(actual_calls),
            native_unused_formals=allowed,
            root_unused_exemptions=[],
        )


def check_native_input_usage(
    model: Any, proof: NativeInputUsage | None = None
) -> dict[str, Any]:
    if proof is not None:
        return proof.check(model)
    for scope in (model.graph, *model.functions):
        for child in _scopes(scope):
            if _lexical_usage(child)[0]:
                raise ValueError(
                    "SOURCE counterpart preserves unused-input rejection without native proof"
                )
    return dict(version=VERSION, native_unused_formals=[], mode="strict-no-proof")
