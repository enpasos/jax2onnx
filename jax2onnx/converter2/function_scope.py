# file: jax2onnx/converter2/function_scope.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import onnx
import onnx.helper as oh
import onnx.onnx_ml_pb2 as onnx_ml
import onnx_ir as ir

from .ir_context import IRContext


@dataclass(frozen=True)
class FunctionKey:
    qualified_name: str           # e.g. "pkg.module.SuperBlock"
    input_sig: Tuple[Tuple[Any, ...], ...]  # shapes/dtypes signature (symbolic tokens allowed)
    capture_sig: Tuple[Any, ...]  # instance/config hash or tuple of static fields


@dataclass
class FunctionDef:
    name: str
    domain: str
    inputs: List[ir.Value]
    outputs: List[ir.Value]
    nodes: List[ir.Node]
    # We keep opset imports minimal for now; parent model imports apply.


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
        # directly; some builds keep them on `parent.builder`.
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
            input_specs=[],                  # set on begin()
        )
        self.ctx._function_mode = True      # tell constant binder to emit Constant nodes
        self._inputs: List[ir.Value] = []
        self._outputs: List[ir.Value] = []
        self._sealed = False

    def begin(self, inputs: List[ir.Value]) -> List[ir.Value]:
        # function inputs are Values in the child context with the same meta
        self._inputs = []
        for i, vin in enumerate(inputs):
            fin = ir.Value(
                name=f"f_in_{i}",
                type=vin.type,
                shape=vin.shape,
            )
            # Register as graph input in child
            self.ctx._inputs.append(fin)
            self._inputs.append(fin)
        return self._inputs

    def end(self, outputs: List[ir.Value]) -> FunctionDef:
        if self._sealed:
            raise RuntimeError("FunctionScope already sealed.")
        self._sealed = True
        # Outputs are child Values; ensure they exist
        self._outputs = []
        for i, vout in enumerate(outputs):
            # If vout came from parent, we need to map to the child-produced value;
            # here we assume lowering wrote into child values directly.
            self._outputs.append(vout)
        # Snapshot nodes produced in the child context
        nodes = list(getattr(self.ctx, "_nodes", []) or [])
        return FunctionDef(
            name=self.name,
            domain=self.domain,
            inputs=self._inputs,
            outputs=self._outputs,
            nodes=nodes,
        )


def attach_functions_to_model(model: onnx.ModelProto, fdefs: List[FunctionDef]) -> None:
    """
    Append FunctionProto to the serialized model.
    Note: onnx-ir isn't used here; we use ONNX helpers to materialize FunctionProto.
    """
    for f in fdefs:
        # ONNX FunctionProto needs input/output names; nodes are NodeProto
        input_names = [v.name for v in f.inputs]
        output_names = [v.name for v in f.outputs]

        # Build NodeProto list from ir.Node (ir.to_node_proto() would be ideal; fallback map)
        nodes: List[onnx_ml.NodeProto] = []
        for n in f.nodes:
            node = onnx_ml.NodeProto()
            node.op_type = n.op_type
            node.domain = n.domain or ""
            node.name = getattr(n, "name", "") or ""
            # inputs/outputs are named Values
            node.input[:] = [getattr(v, "name", "") for v in (getattr(n, "inputs", []) or [])]
            node.output[:] = [getattr(v, "name", "") for v in (getattr(n, "outputs", []) or [])]
            # attributes
            for a in (getattr(n, "attributes", []) or []):
                attr = onnx_ml.AttributeProto()
                # best-effort name
                attr.name = getattr(a, "name", "") or getattr(a, "key", "") or "attr"
                # support INT, INTS, FLOAT, STRING, INTS via common fields
                if getattr(a, "ints", None) is not None:
                    attr.ints[:] = [int(x) for x in a.ints]
                    attr.type = onnx_ml.AttributeProto.INTS
                elif getattr(a, "int", None) is not None:
                    attr.i = int(a.int)
                    attr.type = onnx_ml.AttributeProto.INT
                elif getattr(a, "f", None) is not None:
                    attr.f = float(a.f)
                    attr.type = onnx_ml.AttributeProto.FLOAT
                elif getattr(a, "s", None) is not None:
                    s = a.s if isinstance(a.s, (bytes, bytearray)) else str(a.s).encode("utf-8")
                    attr.s = s
                    attr.type = onnx_ml.AttributeProto.STRING
                elif getattr(a, "value", None) is not None and getattr(a.value, "ints", None) is not None:
                    attr.ints[:] = [int(x) for x in a.value.ints]
                    attr.type = onnx_ml.AttributeProto.INTS
                node.attribute.append(attr)
            nodes.append(node)

        fproto = oh.make_function(
            domain=f.domain,
            fname=f.name,
            inputs=input_names,
            outputs=output_names,
            nodes=nodes,
            opset_imports=list(model.opset_import),  # reuse model imports
        )
        model.functions.append(fproto)
