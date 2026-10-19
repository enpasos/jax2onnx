# jax2onnx/converter/function_scope.py

from __future__ import annotations
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Literal
import onnx_ir as ir
from onnx_ir.convenience import rename_values

from .typing_support import ConverterContext


@dataclass(frozen=True)
class FunctionKey:
    qualified_name: str  # e.g. "pkg.module.SuperBlock"
    input_sig: tuple[tuple[tuple[object, ...], str], ...]
    capture_sig: tuple[object, ...]  # instance/config hash or tuple of static fields
    tracing_sig: tuple[object, ...] = ()


class FunctionRegistry:
    def __init__(self) -> None:
        self._defs: dict[FunctionKey, ir.Function] = {}

    def get(self, key: FunctionKey) -> ir.Function | None:
        return self._defs.get(key)

    def put(self, key: FunctionKey, function: ir.Function) -> None:
        self._defs[key] = function

    def all(self) -> list[ir.Function]:
        return list(self._defs.values())


class FunctionScope:
    """Stage a function body in its own IRContext.

    The scope borrows the parent converter settings but switches `function_mode`
    so constants are emitted as `Constant` nodes (FunctionProto cannot own
    initializers). Each parent value handed to :meth:`begin` is mirrored with a
    fresh `ir.Value` that becomes a function input. :meth:`end` finalizes the
    child graph once and returns an independent native `ir.Function`, shared by
    the registry and the enclosing model. Inputs, outputs, and nodes live only
    in the child graph while its body is being built.
    """

    def __init__(self, parent: ConverterContext, name: str, domain: str = "") -> None:
        self.parent = parent
        self.name = name
        self.domain = domain

        # Hex encodes the exact pair without collapsing punctuation or Unicode.
        namespace = (
            f"function_d{domain.encode('utf-8').hex()}_n{name.encode('utf-8').hex()}"
        )
        self.ctx = parent.create_child_context(scope="function", prefix=namespace)
        self._value_namespace = self.ctx.fresh_name("local")
        self._state: Literal["new", "building", "sealed"] = "new"

    def begin(self, inputs: list[ir.Value]) -> list[ir.Value]:
        if self._state != "new":
            raise RuntimeError("FunctionScope.begin() may only be called once.")
        self._state = "building"
        formals = [
            ir.Value(name=f"f_in_{i}", type=value.type, shape=value.shape)
            for i, value in enumerate(inputs)
        ]
        self.ctx.builder.inputs.extend(formals)
        parent_origin = self.parent.get_symbolic_dim_origin
        for vin, fin in zip(inputs, formals):
            vin_shape = vin.shape
            dims = tuple(vin_shape.dims) if vin_shape is not None else ()
            for axis, dim in enumerate(dims):
                origin = parent_origin(dim)
                if origin is not None:
                    self.ctx.record_symbolic_dim_origin(dim, fin, axis)
        return formals

    def _namespace_owned_values(self) -> None:
        """Rename local definitions by identity, retaining lexical outer captures."""
        graph = self.ctx.builder.graph
        graphs = [graph, *graph.subgraphs()]
        formals = {id(value) for value in graph.inputs}
        owned: list[ir.Value] = []
        seen: set[int] = set()
        for local_graph in graphs:
            definitions = [
                *local_graph.inputs,
                *local_graph.initializers.values(),
                *(value for node in local_graph for value in node.outputs),
            ]
            for value in definitions:
                identity = id(value)
                if (
                    value.graph is local_graph
                    and identity not in formals
                    and identity not in seen
                ):
                    seen.add(identity)
                    owned.append(value)

        # Include parent definitions as well as all free operands/results. The
        # latter can originate above the immediate parent, and must keep names.
        visible = {value.name for value in graph.inputs if value.name}
        parent_graph = self.parent.builder.graph
        for local_graph in [parent_graph, *parent_graph.subgraphs(), *graphs]:
            values = [
                *local_graph.inputs,
                *local_graph.outputs,
                *local_graph.initializers.values(),
                *(value for node in local_graph for value in node.outputs),
                *(
                    value
                    for node in local_graph
                    for value in node.inputs
                    if value is not None
                ),
            ]
            visible.update(
                value.name for value in values if id(value) not in seen and value.name
            )
        namespace = self._value_namespace
        names = [f"{namespace}_{index}" for index in range(len(owned))]
        while visible.intersection(names):
            namespace = self.ctx.fresh_name("local")
            names = [f"{namespace}_{index}" for index in range(len(owned))]
        self._value_namespace = namespace

        # Value.name also renames const_value. Proto-backed tensors store the
        # name in their backing proto, so those require an independent backing
        # object; other public tensor wrappers keep the name on the wrapper.
        for value in owned:
            tensor = value.const_value
            if tensor is not None:
                detached = (
                    deepcopy(tensor)
                    if isinstance(tensor, ir.TensorProtoTensor)
                    else copy(tensor)
                )
                if detached is tensor:
                    raise TypeError(
                        "Function-local tensor wrapper must be independent."
                    )
                value.const_value = detached
        rename_values(owned, names)

    def end(self, outputs: list[ir.Value]) -> ir.Function:
        """Seal the body and return its final, independently owned IR function."""
        if self._state == "sealed":
            raise RuntimeError("FunctionScope already sealed.")
        if self._state != "building":
            raise RuntimeError("FunctionScope.begin() must be called before end().")
        self.ctx.builder.outputs.clear()
        self.ctx.builder.outputs.extend(outputs)
        self._namespace_owned_values()
        body_graph = self.ctx.builder.graph.clone(allow_outer_scope_values=True)
        body_graph.name = self.name
        body_opset = int(self.ctx.builder.opset)
        fn_domain = (self.domain or "").strip()
        if fn_domain:
            body_graph.opset_imports.setdefault(fn_domain, body_opset)
        for node in body_graph.all_nodes():
            if node.domain:
                body_graph.opset_imports.setdefault(node.domain, body_opset)
        body_graph.opset_imports.setdefault("", body_opset)

        function = ir.Function(
            domain=self.domain,
            name=self.name,
            graph=body_graph,
            attributes=[],
        )
        setattr(function, "_attr_overrides", dict(self.ctx.attr_overrides))
        self._state = "sealed"
        return function
