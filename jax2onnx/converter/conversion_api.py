# file: jax2onnx/converter/conversion_api.py

from typing import Any, Dict, Optional, Sequence, Union
import onnx
import logging
import numpy as np
from onnx import helper, mapping
from jax import config as jax_config
import jax.numpy as jnp

from .dynamic_utils import _create_symbolic_input_avals
from .jaxpr_converter import Jaxpr2OnnxConverter
from .name_generator import UniqueNameGenerator
from .onnx_builder import OnnxBuilder
from .optimize_onnx_graph import improve_onnx_model

logger = logging.getLogger("jax2onnx.converter.conversion_api")


def _elem_type_from_numpy(arr: np.ndarray) -> int:
    return mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]


def _promote_params_to_inputs(model: onnx.ModelProto, params: Dict[str, Any] | None):
    if not params:
        return
    for name, value in params.items():
        kept = [init for init in model.graph.initializer if init.name != name]
        model.graph.ClearField("initializer")
        model.graph.initializer.extend(kept)
        kept = [vi for vi in model.graph.value_info if vi.name != name]
        model.graph.ClearField("value_info")
        model.graph.value_info.extend(kept)
        if any(inp.name == name for inp in model.graph.input):
            continue
        dtype = _elem_type_from_numpy(np.asarray(value))
        vi = helper.make_tensor_value_info(name, dtype, [])
        model.graph.input.append(vi)


def to_onnx(
    fn: Any,
    inputs: Sequence[Sequence[Union[int, str]]],
    input_params: Dict[str, Any] | None = None,
    model_name: str = "jax_model",
    opset: int = 21,
    *,
    enable_double_precision: bool = False,
    default_dtype: Any | None = None,
    record_primitive_calls_file: Optional[str] = None,
) -> onnx.ModelProto:
    logger.info(f"Starting JAX to ONNX conversion for '{model_name}'")
    logger.debug(f"Received raw inputs (shapes): {inputs}")
    logger.debug(f"Received input_params: {input_params.keys() if input_params else 'None'}")

    if enable_double_precision:
        jax_config.update("jax_enable_x64", True)
        working_dtype = jnp.float64
    else:
        jax_config.update("jax_enable_x64", False)
        working_dtype = jnp.float32 if default_dtype is None else default_dtype

    logger.debug(f"🔧 enable_double_precision = {enable_double_precision} → working dtype = {working_dtype}")

    from jax import ShapeDtypeStruct

    normalized_specs = []
    for spec in inputs:
        if isinstance(spec, ShapeDtypeStruct):
            normalized_specs.append((spec.shape, spec.dtype))
        elif hasattr(spec, "shape") and hasattr(spec, "dtype"):
            normalized_specs.append((tuple(spec.shape), spec.dtype))
        elif isinstance(spec, (tuple, list)):
            normalized_specs.append((tuple(spec), working_dtype))
        else:
            raise TypeError(
                f"Unsupported inputs element: {type(spec)}. "
                "Must be shape tuple, Array, or ShapeDtypeStruct."
            )

    logger.debug(f"Normalized input_specs: {normalized_specs}")

    symbolic_avals, var_to_symbol_map = _create_symbolic_input_avals(normalized_specs)

    unique_name_generator = UniqueNameGenerator()
    builder = OnnxBuilder(
        unique_name_generator,
        opset=opset,
        converter=None,
        enable_double_precision=enable_double_precision,
    )
    builder.var_to_symbol_map = var_to_symbol_map

    converter = Jaxpr2OnnxConverter(
        builder,
        record_primitive_calls_file=record_primitive_calls_file,
        function_context_for_recording=getattr(fn, "__name__", model_name),
    )
    builder.converter = converter
    converter.call_params = input_params or {}

    logger.info("Initiating JAX tracing with symbolic abstract values...")
    converter.trace_jaxpr(fn, symbolic_avals, params=input_params)
    logger.info("JAX tracing finished.")

    logger.info("Building ONNX model...")
    builder.filter_unused_initializers()
    model = builder.create_onnx_model(model_name)
    _promote_params_to_inputs(model, input_params)

    logger.info("Optimizing ONNX model...")
    model = improve_onnx_model(model)

    target_ir_version = 10
    if model.ir_version > target_ir_version:
        logger.info(
            f"Current model IR version is {model.ir_version}. "
            f"Setting IR version to {target_ir_version} for compatibility."
        )
        model.ir_version = target_ir_version

    logger.info("ONNX model conversion complete.")
    logger.debug(onnx.helper.printable_graph(model.graph))

    if record_primitive_calls_file and hasattr(converter, "recorded_calls_log"):
        from jax2onnx.utils.debug import save_primitive_calls_log
        save_primitive_calls_log(
            converter.recorded_calls_log,
            record_primitive_calls_file,
        )

    return model