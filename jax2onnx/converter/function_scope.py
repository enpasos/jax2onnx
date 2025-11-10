# jax2onnx/converter/function_scope.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx_ir as ir

from .ir_context import IRContext
from .ir_clone import clone_graph
from .typing_support import SymbolicDimOrigin


@dataclass(frozen=True)
class FunctionKey:
    qualified_name: str  # e.g. "pkg.module.SuperBlock"
    input_sig: Tuple[
        Tuple[Any, ...], ...
    ]  # shapes/dtypes signature (symbolic tokens allowed)
    capture_sig: Tuple[Any, ...]  # instance/config hash or tuple of static fields


@dataclass
class FunctionDef:
    name: str
    domain: str
    inputs: List[ir.Value]
    outputs: List[ir.Value]
    nodes: List[ir.Node]
    # late attribute overrides captured from the child IRContext
    attr_overrides: dict[str, dict[str, object]] | None = None


class FunctionRegistry:
    def __init__(self) -> None:
        # key -> FunctionDef
        self._defs: dict[FunctionKey, FunctionDef] = {}

    def get(self, key: FunctionKey) -> Optional[FunctionDef]:
        return self._defs.get(key)

    def put(self, key: FunctionKey, fdef: FunctionDef) -> None:
        self._defs[key] = fdef

    def all(self) -> List[FunctionDef]:
        return list(self._defs.values())


class FunctionScope:
    """Stage a function body in its own IRContext.

    The scope borrows the parent converter settings but switches `function_mode`
    so constants are emitted as `Constant` nodes (FunctionProto cannot own
    initializers). Each parent value handed to :meth:`begin` is mirrored with a
    fresh `ir.Value` that becomes a function input; `end` snapshots the child
    graph and returns a `FunctionDef` that still references those mirrored
    values. Call-sites can then emit a single node with `domain`/`op_type`
    pointing at the produced function while reusing the parent graph values.
    """

    def __init__(self, parent: IRContext, name: str, domain: str = "") -> None:
        self.parent = parent
        self.name = name
        self.domain = domain

        # Mirror parent converter settings so the function body behaves identically.
        parent_opset = parent.opset
        parent_x64 = parent.enable_double_precision

        # child context buffers
        self.ctx = IRContext(
            opset=parent_opset,
            enable_double_precision=parent_x64,
            input_specs=[],  # set on begin()
        )
        self.ctx._function_mode = True  # tell constant binder to emit Constant nodes
        self.ctx.builder._function_mode = True
        self._inputs: List[ir.Value] = []
        self._outputs: List[ir.Value] = []
        self._sealed = False
        self._attr_overrides: Dict[str, Dict[str, object]] = {}
        self._prev_inside: bool = False

        self.fn_def: Optional[FunctionDef] = None

    def begin(self, inputs: List[ir.Value]) -> List[ir.Value]:
        # Mark we are inside a function while building its body
        self._prev_inside = self.ctx._inside_function_scope
        self.ctx._inside_function_scope = True
        if self.fn_def is None:
            self.fn_def = FunctionDef(
                name=self.name,
                domain=self.domain,
                inputs=[],
                outputs=[],
                nodes=[],
            )
        fn_def = self.fn_def
        fn_def.inputs = []
        parent_origin = self.parent.get_symbolic_dim_origin
        for i, vin in enumerate(inputs):
            fin = ir.Value(
                name=f"f_in_{i}",
                type=vin.type,
                shape=vin.shape,
            )
            # Register as graph input in child
            self.ctx._inputs.append(fin)
            fn_def.inputs.append(fin)

            vin_shape = vin.shape
            dims = tuple(vin_shape.dims) if vin_shape is not None else ()
            for axis, dim in enumerate(dims):
                origin = parent_origin(dim)
                if origin is None and isinstance(dim, str):
                    origin = parent_origin(str(dim))
                if origin is not None:
                    sym_key = str(dim)
                    self.ctx.builder.record_symbol_origin(sym_key, fin, axis)
                    try:
                        if not isinstance(dim, (int, np.integer)):
                            self.ctx._sym_origin[dim] = SymbolicDimOrigin(
                                value=fin, axis=axis
                            )
                    except Exception:
                        pass
                    try:
                        self.ctx._sym_origin_str[str(dim)] = SymbolicDimOrigin(
                            value=fin, axis=axis
                        )
                    except Exception:
                        pass
        return fn_def.inputs

    def end(self, outputs: List[ir.Value]) -> FunctionDef:
        if self._sealed:
            raise RuntimeError("FunctionScope already sealed.")
        if self.fn_def is None:
            raise RuntimeError("FunctionScope.begin() must be called before end().")
        self._sealed = True
        # Snapshot child inputs/outputs/nodes/overrides
        fn_def = self.fn_def
        inputs = list(fn_def.inputs)
        self._outputs = list(outputs)
        nodes = list(self.ctx._nodes)
        overrides = dict(self.ctx._attr_overrides)
        self._attr_overrides = overrides
        # Restore previous scope flag
        self.ctx._inside_function_scope = self._prev_inside
        return FunctionDef(
            name=self.name,
            domain=self.domain,
            inputs=inputs,
            outputs=self._outputs,
            nodes=nodes,
            attr_overrides=overrides,
        )

    def to_ir_function(self) -> ir.Function:
        # Pick an opset for the body; prefer parent/child builder opset
        body_opset = int(self.ctx.builder.opset)

        # Ensure the builder graph carries the final outputs before cloning.
        if self._outputs:
            graph_outputs = self.ctx.builder.outputs
            graph_outputs.clear()
            graph_outputs.extend(self._outputs)

        body_graph = clone_graph(self.ctx.builder.graph)
        body_graph.name = self.name

        opset_imports = dict(body_graph.opset_imports)
        fn_domain = (self.domain or "").strip()
        if fn_domain:
            opset_imports.setdefault(fn_domain, body_opset)
        try:
            nodes_iter = list(body_graph.all_nodes())
        except AttributeError:
            try:
                nodes_iter = list(body_graph.node)
            except AttributeError:
                nodes_iter = []
        for node in nodes_iter:
            dom = getattr(node, "domain", "") or ""
            if dom:
                opset_imports.setdefault(dom, body_opset)
        opset_imports.setdefault("", body_opset)
        body_graph.opset_imports.clear()
        body_graph.opset_imports.update(opset_imports)

        # Create the Function (domain/name must match the call-site)
        fn = ir.Function(
            domain=self.domain,
            name=self.name,
            graph=body_graph,
            attributes=[],
        )
        setattr(fn, "_attr_overrides", dict(self._attr_overrides or {}))
        return fn
