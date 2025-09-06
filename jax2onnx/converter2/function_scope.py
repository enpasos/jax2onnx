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
            input_specs=[],  # set on begin()
        )
        self.ctx._function_mode = True  # tell constant binder to emit Constant nodes
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
                out.append(int(d) if isinstance(d, int) else str(d))
            return out

        def _to_tensor_proto_from_any(x: Any) -> Optional[onnx_ml.TensorProto]:
            """Best-effort: onnx_ir.Tensor / numpy / TensorProto -> TensorProto."""
            if x is None:
                return None
            if isinstance(x, onnx_ml.TensorProto):
                return x
            if hasattr(x, "proto") and isinstance(
                getattr(x, "proto"), onnx_ml.TensorProto
            ):
                return getattr(x, "proto")
            if hasattr(x, "to_numpy") and callable(getattr(x, "to_numpy")):
                try:
                    return numpy_helper.from_array(np.asarray(x.to_numpy()))
                except Exception:
                    pass
            for cand in ("np", "array", "value", "ndarray", "data"):
                if hasattr(x, cand):
                    arr = getattr(x, cand)
                    if arr is not None:
                        try:
                            return numpy_helper.from_array(np.asarray(arr))
                        except Exception:
                            pass
            try:
                return numpy_helper.from_array(np.asarray(x))
            except Exception:
                return None

        # ---- Start building the FunctionProto --------------------------------
        input_names = [v.name for v in f.inputs]
        output_names = [v.name for v in f.outputs]

        # Build NodeProto list from ir.Node (fallback conversion)
        nodes: List[onnx_ml.NodeProto] = []

        # Constant tensors we can reuse when Constant nodes lack explicit 'value' attr.
        const_values: dict[str, onnx_ml.TensorProto] = {}
        for n in f.nodes:
            if getattr(n, "op_type", "") == "Constant" and getattr(n, "outputs", None):
                for out_val in n.outputs:
                    if (
                        hasattr(out_val, "const_value")
                        and out_val.const_value is not None
                    ):
                        tp = _to_tensor_proto_from_any(out_val.const_value)
                        if tp is not None:
                            const_values[out_val.name] = tp

        produced: set[str] = set(input_names)  # names already available for use

        # ValueInfos (shapes on edges) — we collect for inputs/outputs and all intermediates.
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
            # Inject missing Constant nodes for any constant inputs not yet produced.
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
            # inputs/outputs are named Values
            node.input[:] = [
                getattr(v, "name", "") for v in (getattr(n, "inputs", []) or [])
            ]
            node.output[:] = [
                getattr(v, "name", "") for v in (getattr(n, "outputs", []) or [])
            ]

            # ---- Attributes (handle dict or list; add Constant.value if missing) ----
            attrs = getattr(n, "attributes", None)

            def _emit_attr(name: str, aobj: Any):
                ap = onnx_ml.AttributeProto()
                ap.name = name or "attr"
                # Common scalar/list cases from onnx-ir
                if hasattr(aobj, "ints") and aobj.ints is not None:
                    ap.type = onnx_ml.AttributeProto.INTS
                    ap.ints[:] = [int(x) for x in aobj.ints]
                elif hasattr(aobj, "int"):
                    ap.type = onnx_ml.AttributeProto.INT
                    ap.i = int(aobj.int)
                elif hasattr(aobj, "f"):
                    ap.type = onnx_ml.AttributeProto.FLOAT
                    ap.f = float(aobj.f)
                elif hasattr(aobj, "s"):
                    ap.type = onnx_ml.AttributeProto.STRING
                    s = (
                        aobj.s
                        if isinstance(aobj.s, (bytes, bytearray))
                        else str(aobj.s).encode("utf-8")
                    )
                    ap.s = s
                elif hasattr(aobj, "t"):
                    # tensor attribute (e.g., Constant.value)
                    ap.type = onnx_ml.AttributeProto.TENSOR
                    ap.t.CopyFrom(aobj.t)
                elif hasattr(aobj, "value") and hasattr(aobj.value, "ints"):
                    ap.type = onnx_ml.AttributeProto.INTS
                    ap.ints[:] = [int(x) for x in aobj.value.ints]
                # NEW: accept plain Python scalars as attrs
                elif isinstance(aobj, (int, np.integer)):
                    ap.type = onnx_ml.AttributeProto.INT
                    ap.i = int(aobj)
                elif isinstance(aobj, (float, np.floating)):
                    ap.type = onnx_ml.AttributeProto.FLOAT
                    ap.f = float(aobj)
                else:
                    return  # skip unknown
                node.attribute.append(ap)

            # Dict-style attributes
            if isinstance(attrs, dict):
                # If it's a Constant node, treat 'value' specially and force a TensorProto
                if n.op_type == "Constant" and "value" in attrs:
                    tp = None
                    v = attrs["value"]
                    if hasattr(v, "t"):
                        tp = _to_tensor_proto_from_any(getattr(v, "t"))
                    else:
                        tp = _to_tensor_proto_from_any(v)
                    if tp is not None:
                        ap = onnx_ml.AttributeProto()
                        ap.name = "value"
                        ap.type = onnx_ml.AttributeProto.TENSOR
                        ap.t.CopyFrom(tp)
                        node.attribute.append(ap)
                # Emit the rest
                for k, v in attrs.items():
                    if n.op_type == "Constant" and k == "value":
                        continue
                    _emit_attr(str(k), v)
            # List-style attributes
            elif isinstance(attrs, (list, tuple)):
                for a in attrs:
                    name = getattr(a, "name", "") or getattr(a, "key", "")
                    if n.op_type == "Constant" and name == "value":
                        tp = None
                        # a may have .t or be the tensor itself
                        if hasattr(a, "t"):
                            tp = _to_tensor_proto_from_any(getattr(a, "t"))
                        else:
                            tp = _to_tensor_proto_from_any(a)
                        if tp is not None:
                            ap = onnx_ml.AttributeProto()
                            ap.name = "value"
                            ap.type = onnx_ml.AttributeProto.TENSOR
                            ap.t.CopyFrom(tp)
                            node.attribute.append(ap)
                        continue
                    _emit_attr(name, a)

            # Ensure Constant nodes carry a 'value' tensor attribute (fallback from output const_value)
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

            # Append any pre-Constant nodes, then the actual node
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

        fproto = oh.make_function(
            domain=f.domain,
            fname=f.name,
            inputs=input_names,
            outputs=output_names,
            nodes=nodes,
            opset_imports=list(model.opset_import),  # reuse model imports
        )
        # Add value_info so shapes show up on function-body edges
        if vi_map:
            fproto.value_info.extend(list(vi_map.values()))
        model.functions.append(fproto)
