# file: jax2onnx/converter2/function_scope.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import onnx_ir as ir
import numpy as np

from .ir_context import IRContext


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
    def __init__(self):
        # key -> FunctionDef
        self._defs: dict[FunctionKey, FunctionDef] = {}

    def get(self, key: FunctionKey) -> Optional[FunctionDef]:
        return self._defs.get(key)

    def put(self, key: FunctionKey, fdef: FunctionDef) -> None:
        self._defs[key] = fdef

    def all(self) -> List[FunctionDef]:
        return list(self._defs.values())


class FunctionScope:
    """
    Capture a function body into a child IRContext. In function-mode:
    - constants must be emitted as Constant nodes (not initializers)
    - inputs/outputs are local to the function
    """

    def __init__(self, parent: IRContext, name: str, domain: str = ""):
        self.parent = parent
        self.name = name
        self.domain = domain

        # Be defensive: parent may not expose `opset` / `enable_double_precision`
        parent_builder = getattr(parent, "builder", None)
        parent_opset = getattr(parent, "opset", None)
        if parent_opset is None and parent_builder is not None:
            parent_opset = getattr(parent_builder, "opset", None)
        if parent_opset is None:
            parent_opset = 21  # safe default; tests set opset explicitly

        parent_x64 = getattr(parent, "enable_double_precision", None)
        if parent_x64 is None and parent_builder is not None:
            parent_x64 = getattr(parent_builder, "enable_double_precision", False)
        if parent_x64 is None:
            parent_x64 = False

        # child context buffers
        self.ctx = IRContext(
            opset=parent_opset,
            enable_double_precision=parent_x64,
            input_specs=[],  # set on begin()
        )
        self.ctx._function_mode = True  # tell constant binder to emit Constant nodes
        self._inputs: List[ir.Value] = []
        self._outputs: List[ir.Value] = []
        self._sealed = False

        self.fn_def: Optional[FunctionDef] = None

    def begin(self, inputs: List[ir.Value]) -> List[ir.Value]:
        # Mark we are inside a function while building its body
        self._prev_inside = getattr(self.ctx, "_inside_function_scope", False)
        setattr(self.ctx, "_inside_function_scope", True)
        if not hasattr(self, "fn_def") or self.fn_def is None:
            self.fn_def = FunctionDef(
                name=getattr(self, "name", "Function"),
                domain=getattr(self, "domain", "custom"),
                inputs=[],
                outputs=[],
                nodes=[],
            )
        self.fn_def.inputs = []
        for i, vin in enumerate(inputs):
            fin = ir.Value(
                name=f"f_in_{i}",
                type=vin.type,
                shape=vin.shape,
            )
            # Register as graph input in child
            self.ctx._inputs.append(fin)
            self.fn_def.inputs.append(fin)
        return self.fn_def.inputs

    def end(self, outputs: List[ir.Value]) -> FunctionDef:
        if self._sealed:
            raise RuntimeError("FunctionScope already sealed.")
        self._sealed = True
        # Snapshot child inputs/outputs/nodes/overrides
        inputs = list(getattr(self.fn_def, "inputs", []) or [])
        self._outputs = list(outputs)
        nodes = list(getattr(self.ctx, "_nodes", []) or [])
        overrides = dict(getattr(self.ctx, "_attr_overrides", {}) or {})
        # Restore previous scope flag
        setattr(
            self.ctx, "_inside_function_scope", getattr(self, "_prev_inside", False)
        )
        return FunctionDef(
            name=self.name,
            domain=self.domain,
            inputs=inputs,
            outputs=self._outputs,
            nodes=nodes,
            attr_overrides=overrides,
        )

    # NEW: materialize a native onnx_ir.Function from the child context
    def to_ir_function(self) -> ir.Function:
        # Pick an opset for the body; prefer parent/child builder opset
        try:
            body_opset = int(getattr(getattr(self.ctx, "builder", None), "opset", 21))
        except Exception:
            body_opset = 21

        # Build an IR graph for the function body
        g = ir.Graph(
            inputs=list(self.ctx.builder.inputs or []),
            outputs=list(self._outputs or []),
            nodes=list(self.ctx.builder.nodes or []),
            initializers=list(self.ctx.builder.initializers or []),
            name=self.name,
            opset_imports={"": body_opset},
        )
        # Create the Function (domain/name must match the call-site)
        return ir.Function(
            domain=self.domain,
            name=self.name,
            graph=g,
            attributes=[],
        )


def attach_functions_to_model(*args, **kwargs):
    """
    Deprecated in IR2: use native onnx_ir.Function and attach to ir.Model.
    This is left as a no-op shim to keep imports from breaking while migrating.
    """
    return None
