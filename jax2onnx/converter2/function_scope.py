# file: jax2onnx/converter2/function_scope.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import onnx
import onnx.helper as oh
import onnx.onnx_ml_pb2 as onnx_ml
import onnx_ir as ir
import numpy as np
from onnx import numpy_helper

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


def attach_functions_to_model(model: onnx.ModelProto, fdefs: List[FunctionDef]) -> None:
    """
    Append FunctionProto to the serialized model.
    Note: onnx-ir isn't used here; we use ONNX helpers to materialize FunctionProto.
    """
    # ---- Helpers: dtype/shape & tensor conversion ------------------------
    from onnx import TensorProto

    def _to_onnx_dtype(v: ir.Value) -> int:
        t = getattr(v, "type", None)
        et = getattr(t, "elem_type", None) if t is not None else None
        if et is None:
            et = getattr(t, "dtype", None) if t is not None else None
        mapping = {
            getattr(ir.DataType, "FLOAT", None): TensorProto.FLOAT,
            getattr(ir.DataType, "DOUBLE", None): TensorProto.DOUBLE,
            getattr(ir.DataType, "BOOL", None): TensorProto.BOOL,
            getattr(ir.DataType, "INT64", None): TensorProto.INT64,
            getattr(ir.DataType, "INT32", None): TensorProto.INT32,
            getattr(ir.DataType, "INT16", None): TensorProto.INT16,
            getattr(ir.DataType, "INT8", None): TensorProto.INT8,
            getattr(ir.DataType, "UINT64", None): TensorProto.UINT64,
            getattr(ir.DataType, "UINT32", None): TensorProto.UINT32,
            getattr(ir.DataType, "UINT16", None): TensorProto.UINT16,
            getattr(ir.DataType, "UINT8", None): TensorProto.UINT8,
        }
        return mapping.get(et, TensorProto.FLOAT)

    def _to_onnx_shape(v: ir.Value) -> List[Any]:
        shp = getattr(v, "shape", None)
        dims = getattr(shp, "dims", None)
        if dims is None and isinstance(shp, (list, tuple)):
            dims = shp
        out: List[Any] = []
        for d in dims or []:
            out.append(int(d) if isinstance(d, (int, np.integer)) else str(d))
        return out

    def _to_tensor_proto_from_any(x: Any) -> Optional[onnx_ml.TensorProto]:
        """Best-effort: onnx_ir.Tensor / numpy / TensorProto -> TensorProto."""
        if x is None:
            return None
        if isinstance(x, onnx_ml.TensorProto):
            return x
        # onnx_ir.Tensor exposes .to_numpy() and often .to_proto()
        if hasattr(x, "to_proto"):
            try:
                tp = x.to_proto()
                if isinstance(tp, onnx_ml.TensorProto):
                    return tp
            except Exception:
                pass
        if hasattr(x, "to_numpy"):
            try:
                return numpy_helper.from_array(np.asarray(x.to_numpy()))
            except Exception:
                pass
        # common fallbacks
        try:
            return numpy_helper.from_array(np.asarray(x))
        except Exception:
            return None

    for f in fdefs:
        input_names = [v.name for v in f.inputs]
        output_names = [v.name for v in f.outputs]

        # Build NodeProto list from ir.Node (fallback conversion)
        nodes: List[onnx_ml.NodeProto] = []

        # Constant tensors we can reuse when Constant nodes lack explicit 'value' attr.
        const_values: dict[str, onnx_ml.TensorProto] = {}
        for n in f.nodes:
            if getattr(n, "op_type", "") == "Constant" and getattr(n, "outputs", None):
                for out_val in n.outputs:
                    tp = _to_tensor_proto_from_any(
                        getattr(out_val, "const_value", None)
                    )
                    if tp is not None:
                        const_values[out_val.name] = tp

        # Track which names are already produced; start with function inputs.
        produced: set[str] = set(input_names)

        # ValueInfos for inputs, outputs, and intermediates
        vi_map: dict[str, onnx_ml.ValueInfoProto] = {}
        for vin in f.inputs:
            vi_map[vin.name] = oh.make_tensor_value_info(
                vin.name, _to_onnx_dtype(vin), _to_onnx_shape(vin)
            )
        for vout in f.outputs:
            vi_map[vout.name] = oh.make_tensor_value_info(
                vout.name, _to_onnx_dtype(vout), _to_onnx_shape(vout)
            )

        for n in f.nodes:
            # Inject Constant nodes if a node consumes a const Value that hasn't been produced yet
            pre_nodes: List[onnx_ml.NodeProto] = []
            for vin in getattr(n, "inputs", []) or []:
                vname = getattr(vin, "name", "")
                if not vname or vname in produced:
                    continue
                if hasattr(vin, "const_value") and vin.const_value is not None:
                    tp = _to_tensor_proto_from_any(vin.const_value)
                    if tp is not None:
                        cnode = onnx_ml.NodeProto()
                        cnode.op_type = "Constant"
                        cnode.domain = ""
                        cnode.name = f"Constant__{vname}"
                        cnode.output[:] = [vname]
                        ap = onnx_ml.AttributeProto()
                        ap.name = "value"
                        ap.type = onnx_ml.AttributeProto.TENSOR
                        ap.t.CopyFrom(tp)
                        cnode.attribute.append(ap)
                        pre_nodes.append(cnode)
                        produced.add(vname)
                        # also stamp shape on that edge
                        if vname not in vi_map:
                            vi_map[vname] = oh.make_tensor_value_info(
                                vname, _to_onnx_dtype(vin), _to_onnx_shape(vin)
                            )

            node = onnx_ml.NodeProto()
            node.op_type = n.op_type
            node.domain = n.domain or ""
            node.name = getattr(n, "name", "") or ""
            node.input[:] = [
                getattr(v, "name", "") for v in (getattr(n, "inputs", []) or [])
            ]
            node.output[:] = [
                getattr(v, "name", "") for v in (getattr(n, "outputs", []) or [])
            ]

            # ---- Attributes (handle dict/list and scalars) ----
            attrs = getattr(n, "attributes", None)

            # If this is a Constant with a 'value' that isn't a TensorProto yet, fix it
            if isinstance(attrs, dict) and n.op_type == "Constant" and "value" in attrs:
                tp = _to_tensor_proto_from_any(attrs["value"])
                if tp is not None:
                    ap = onnx_ml.AttributeProto()
                    ap.name = "value"
                    ap.type = onnx_ml.AttributeProto.TENSOR
                    ap.t.CopyFrom(tp)
                    node.attribute.append(ap)

            def _add_attr(name: str, value: Any):
                try:
                    node.attribute.append(oh.make_attribute(name, value))
                except Exception:
                    # Last resort: encode simple ints/floats/strings only
                    ap = onnx_ml.AttributeProto()
                    ap.name = name
                    if isinstance(value, (int, np.integer)):
                        ap.type = onnx_ml.AttributeProto.INT
                        ap.i = int(value)
                    elif isinstance(value, (float, np.floating)):
                        ap.type = onnx_ml.AttributeProto.FLOAT
                        ap.f = float(value)
                    elif isinstance(value, (bytes, bytearray, str)):
                        ap.type = onnx_ml.AttributeProto.STRING
                        ap.s = (
                            value
                            if isinstance(value, (bytes, bytearray))
                            else str(value).encode("utf-8")
                        )
                    else:
                        return
                    node.attribute.append(ap)

            if isinstance(attrs, dict):
                for k, v in attrs.items():
                    if n.op_type == "Constant" and k == "value":
                        continue  # already handled above
                    _add_attr(k, v)
            elif isinstance(attrs, (list, tuple)):
                for a in attrs:
                    k = getattr(a, "name", "") or getattr(a, "key", "") or "attr"
                    _add_attr(k, a)

            # Ensure Constant nodes carry a 'value' tensor attribute (fallback from outputs)
            if n.op_type == "Constant" and node.output:
                has_value_attr = any(
                    a.name == "value" and a.type == onnx_ml.AttributeProto.TENSOR
                    for a in node.attribute
                )
                if not has_value_attr:
                    tname = node.output[0]
                    if tname in const_values:
                        ap = onnx_ml.AttributeProto()
                        ap.name = "value"
                        ap.type = onnx_ml.AttributeProto.TENSOR
                        ap.t.CopyFrom(const_values[tname])
                        node.attribute.append(ap)

            # Append pre-constants, then the actual node
            nodes.extend(pre_nodes)
            nodes.append(node)

            # Mark produced outputs and stamp value_info for internal edges
            for vout in getattr(n, "outputs", []) or []:
                vname = getattr(vout, "name", "")
                if vname:
                    produced.add(vname)
                    if vname not in vi_map:
                        vi_map[vname] = oh.make_tensor_value_info(
                            vname, _to_onnx_dtype(vout), _to_onnx_shape(vout)
                        )

        # Materialize the FunctionProto
        fproto = oh.make_function(
            domain=f.domain,
            fname=f.name,
            inputs=input_names,
            outputs=output_names,
            nodes=nodes,
            opset_imports=list(model.opset_import),  # reuse model imports
        )

        # Apply any late attribute overrides (from child IRContext)
        if f.attr_overrides:
            for n in fproto.node:
                overrides = f.attr_overrides.get(n.name)
                if not overrides:
                    continue
                # Drop any existing attrs with same keys, then re-add via helper
                keep = [a for a in n.attribute if a.name not in overrides]
                del n.attribute[:]
                n.attribute.extend(keep)
                for k, v in overrides.items():
                    n.attribute.append(oh.make_attribute(k, v))

        # Attach value_info for all internal wires (enables ORT shape/type inference)
        if vi_map:
            # Do not duplicate signature IO in value_info
            for nm, vi in vi_map.items():
                if nm in input_names or nm in output_names:
                    continue
                fproto.value_info.append(vi)

        model.functions.append(fproto)
