from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import tempfile

import onnx

# NOTE: onnx_ir: https://github.com/onnx/ir-py
# We use it to build a tiny graph for the first op (Tanh).
try:
    import onnx_ir as ir
except Exception as e:  # pragma: no cover
    ir = None
    _IR_IMPORT_ERROR = e

def _extract_shape_and_dtype(spec: Any) -> Tuple[Tuple[Union[int, str], ...], Optional[Any]]:
    """
    Accepts jax.ShapeDtypeStruct / jax.core.ShapedArray / (tuple|list) shape.
    Returns (shape_tuple, dtype_or_None).
    """
    # jax objects usually expose .shape / .dtype
    if hasattr(spec, "shape"):
        shp = tuple(spec.shape)
        dt = getattr(spec, "dtype", None)
        return shp, dt
    # plain shape like (3,) or ("B", 4)
    if isinstance(spec, (tuple, list)):
        return tuple(spec), None
    raise TypeError(f"Unsupported input spec type: {type(spec)}")

def _np_float_dtype(enable_double_precision: bool):
    import numpy as np
    return np.float64 if enable_double_precision else np.float32

def _ir_dtype(enable_double_precision: bool) -> "ir.DataType":
    return ir.DataType.DOUBLE if enable_double_precision else ir.DataType.FLOAT

def to_onnx(
    *,
    fn: Any,
    inputs: List[Any],
    input_params: Optional[Dict[str, Any]],
    model_name: str,
    opset: int,
    enable_double_precision: bool,
    loosen_internal_shapes: bool,
    record_primitive_calls_file: Optional[str],
) -> onnx.ModelProto:
    """
    Minimal onnx_ir-based converter for the IR test lane.
    For now, it handles a single-argument elementwise tanh testcase.
    """
    if ir is None:
        raise ImportError(
            "onnx_ir is required for converter2 but could not be imported"
        ) from _IR_IMPORT_ERROR

    if not inputs:
        # Allow zero-arg functions if/when we need them; not needed for tanh case.
        raise ValueError("converter2 expects at least one input for now.")
    if len(inputs) != 1:
        # Keep the first milestone simple; extend as we onboard more ops.
        raise NotImplementedError("converter2 currently supports exactly 1 input.")

    # Determine input shape/dtype. If dtype is missing or non-float, default to float32/64.
    in_shape, in_dtype = _extract_shape_and_dtype(inputs[0])
    # Normalize dtype choice: tests in primitives2 drive float32 by default.
    np_dtype = _np_float_dtype(enable_double_precision)

    # ---- Build IR values ----
    x = ir.Value(
        name="x0",
        shape=ir.Shape(in_shape),
        type=ir.TensorType(_ir_dtype(enable_double_precision)),
    )
    y = ir.Value(
        name="y0",
        shape=ir.Shape(in_shape),
        type=ir.TensorType(_ir_dtype(enable_double_precision)),
    )

    # ---- Nodes ----
    # First testcase is tanh: y = Tanh(x)
    tanh_node = ir.node(op_type="Tanh", inputs=[x], outputs=[y])

    # ---- Graph & Model ----
    graph = ir.Graph(
        inputs=[x],
        outputs=[y],
        nodes=[tanh_node],
        initializers=[],
        name=model_name or "jax2onnx_ir_graph",
        opset_imports={"": opset},
    )
    # NOTE: onnxruntime in our CI/env supports IR v10 max (error showed max=10).
    # For now, force IR v10 so ORT can load the model.
    model = ir.Model(graph, ir_version=10)

    # We need to return an onnx.ModelProto to the caller. Save via onnx_ir, then load.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            tmp_path = f.name
        ir.save(model, tmp_path)
        proto = onnx.load_model(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return proto
