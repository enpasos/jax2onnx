# jax2onnx/serde_onnx.py

from __future__ import annotations

from typing import Set

import onnx
import onnx_ir as ir
import tempfile
import os

from onnx import TensorProto


def _align_cast_value_info_types(model: "onnx.ModelProto") -> None:
    """Ensure Cast/CastLike outputs have matching value_info dtypes."""

    int64_outputs: Set[str] = set()
    int32_outputs: Set[str] = set()

    def _record(outputs, dtype_idx):
        if dtype_idx == TensorProto.INT64:
            int64_outputs.update(outputs)
        elif dtype_idx == TensorProto.INT32:
            int32_outputs.update(outputs)

    initializer_dtypes = {init.name: init.data_type for init in model.graph.initializer}
    value_info_dtypes = {}
    for vi in list(model.graph.value_info) + list(model.graph.input):
        type_proto = vi.type
        if type_proto and type_proto.WhichOneof("value") == "tensor_type":
            value_info_dtypes[vi.name] = type_proto.tensor_type.elem_type

    for node in model.graph.node:
        if node.op_type == "Cast":
            to_attr = next((attr for attr in node.attribute if attr.name == "to"), None)
            if to_attr is not None:
                _record(node.output, to_attr.i)
        elif node.op_type == "CastLike" and len(node.input) >= 2:
            exemplar = node.input[1]
            dtype_idx = initializer_dtypes.get(exemplar) or value_info_dtypes.get(
                exemplar
            )
            if dtype_idx is not None:
                _record(node.output, dtype_idx)

    if not int64_outputs and not int32_outputs:
        return

    for vi in model.graph.value_info:
        type_proto = vi.type
        if not type_proto or type_proto.WhichOneof("value") != "tensor_type":
            continue
        tensor_type = type_proto.tensor_type
        if vi.name in int64_outputs and tensor_type.elem_type != TensorProto.INT64:
            tensor_type.elem_type = TensorProto.INT64
        elif vi.name in int32_outputs and tensor_type.elem_type != TensorProto.INT32:
            tensor_type.elem_type = TensorProto.INT32


def ir_to_onnx(ir_model: "ir.Model") -> onnx.ModelProto:
    """
    Convert an onnx-ir Model to an ONNX ModelProto by saving to a temp file.
    This module is the only place where 'onnx' is imported for converter.
    """
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            tmp = f.name
        ir.save(ir_model, tmp)
        model = onnx.load_model(tmp)
        _align_cast_value_info_types(model)
        return model
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
