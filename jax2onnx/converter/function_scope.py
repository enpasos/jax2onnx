# jax2onnx/converter/function_scope.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import onnx_ir as ir

from .typing_support import ConverterContext


@dataclass(frozen=True)
class FunctionKey:
    qualified_name: str  # e.g. "pkg.module.SuperBlock"
    input_sig: tuple[tuple[tuple[object, ...], str], ...]
    capture_sig: tuple[object, ...]  # instance/config hash or tuple of static fields


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

        self.ctx = parent.create_child_context(scope="function")
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

    def end(self, outputs: list[ir.Value]) -> ir.Function:
        """Seal the body and return its final, independently owned IR function."""
        if self._state == "sealed":
            raise RuntimeError("FunctionScope already sealed.")
        if self._state != "building":
            raise RuntimeError("FunctionScope.begin() must be called before end().")
        self.ctx.builder.outputs.clear()
        self.ctx.builder.outputs.extend(outputs)
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
