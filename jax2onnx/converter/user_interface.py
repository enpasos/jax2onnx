# file: jax2onnx/converter/user_interface.py

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import argparse
import logging

import onnx
from jax import config, core
from jax2onnx.converter.conversion_api import to_onnx as to_onnx_impl
from jax2onnx.converter.validation import allclose as allclose_impl
from jax2onnx.plugin_system import onnx_function as onnx_function_impl

# Added imports for the bridge handler
from flax import nnx
from . import linen_handler


config.update("jax_dynamic_shapes", True)

_FLOAT64_HELP = (
    "Export the entire ONNX graph in double precision (tensor(double)). "
    "If omitted, tensors are exported in single precision (tensor(float))."
)


def to_onnx(
    fn: Callable,
    inputs: List[Any],
    input_params: Optional[Dict[str, Any]] = None,
    model_name: str = "jax_model",
    opset: int = 21,
    *,
    enable_double_precision: bool = False,
    record_primitive_calls_file: Optional[str] = None,
) -> onnx.ModelProto:
    """
    Converts a JAX function or model into an ONNX model.
    """
    target_fn = fn
    target_inputs = inputs
    is_bridged_model = isinstance(fn, nnx.bridge.ToNNX)

    if is_bridged_model:
        pure_fn, trace_args = linen_handler.to_onnx_for_linen_bridge(
            fn,
            inputs,
            model_name=model_name,
            opset=opset,
            input_params=input_params,
            enable_double_precision=enable_double_precision,
            record_primitive_calls_file=record_primitive_calls_file,
        )
        target_fn = pure_fn
        target_inputs = trace_args

    logging.info(
        f"Converting JAX function to ONNX model with parameters: "
        f"model_name={model_name}, opset={opset}, input_shapes={target_inputs}, "
        f"input_params={input_params}, enable_double_precision={enable_double_precision}, "
        f"record_primitive_calls_file={record_primitive_calls_file}"
    )

    processed_inputs_for_impl: list

    if not target_inputs:
        processed_inputs_for_impl = []
    else:
        # If we used the bridge, separate the params pytree from the actual inputs.
        if is_bridged_model:
            params_pytree = target_inputs[0]
            actual_inputs = target_inputs[1:]
            processed_inputs_for_impl = [params_pytree]
        else:
            actual_inputs = target_inputs
            processed_inputs_for_impl = []

        # Now, process only the actual data inputs into ShapeDtypeStructs or shapes.
        if actual_inputs:
            if all(isinstance(x, core.ShapedArray) for x in actual_inputs):
                processed_inputs_for_impl.extend(list(actual_inputs))
            else:
                def is_shape_tuple(item):
                    return isinstance(item, (tuple, list)) and all(
                        isinstance(dim, (int, str)) for dim in item
                    )

                if all(is_shape_tuple(x) for x in actual_inputs):
                    processed_inputs_for_impl.extend(list(actual_inputs))
                else:
                    try:
                        import jax
                        processed_inputs_for_impl.extend(
                            [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in actual_inputs]
                        )
                    except AttributeError as e:
                        raise ValueError(
                            "Invalid 'inputs' argument. Expected a list of JAX/NumPy arrays, "
                            "jax.ShapeDtypeStruct objects, or shape tuples. "
                            f"Got an element of type {type(actual_inputs[0]) if actual_inputs else 'Unknown'} in the list. Error: {e}"
                        )

    return to_onnx_impl(
        fn=target_fn,
        inputs=processed_inputs_for_impl,
        input_params=input_params,
        model_name=model_name,
        opset=opset,
        enable_double_precision=enable_double_precision,
        record_primitive_calls_file=record_primitive_calls_file,
    )


# ... (The rest of your user_interface.py file remains unchanged)
# ... (build_arg_parser, run_command_line, convert, onnx_function, allclose)
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jax2onnx",
        description="Convert a JAX function to an ONNX model.",
    )
    p.add_argument("module", help="Python module containing the JAX function")
    p.add_argument("fn", help="Name of the JAX function inside the module")
    p.add_argument("--out", help="Output .onnx file", default="model.onnx")
    p.add_argument("--opset", type=int, default=21, help="ONNX opset version")
    p.add_argument(
        "--float64",
        dest="enable_double_precision",
        action="store_true",
        default=False,
        help=_FLOAT64_HELP,
    )
    p.add_argument(
        "--record-primitives",
        dest="record_primitive_calls_file",
        help="File path to record JAX primitive calls during conversion",
        default=None,
    )
    return p

def run_command_line():
    args = build_arg_parser().parse_args()
    import importlib
    import sys
    sys.path.append(".")
    try:
        module = importlib.import_module(args.module)
        function = getattr(module, args.fn)
    except (ImportError, AttributeError) as e:
        logging.error(f"Error loading function: {e}")
        sys.exit(1)
    input_specs = []
    if hasattr(args, "input_shapes") and args.input_shapes:
        try:
            input_specs = eval(args.input_shapes)
        except Exception as e:
            logging.error(f"Error parsing input shapes: {e}")
            sys.exit(1)
    to_onnx(
        function,
        inputs=input_specs,
        model_name=args.fn,
        opset=args.opset,
        enable_double_precision=args.enable_double_precision,
        record_primitive_calls_file=args.record_primitive_calls_file,
    )

def convert(*, enable_double_precision: bool = False, record_primitive_calls_file: Optional[str] = None, **kwargs):
    return to_onnx(
        enable_double_precision=enable_double_precision,
        record_primitive_calls_file=record_primitive_calls_file,
        **kwargs,
    )

def onnx_function(target: Union[Callable, type]) -> Union[Callable, type]:
    return onnx_function_impl(target)

def allclose(
    fn: Callable,
    onnx_model_path: str,
    inputs: List[Any],
    input_params: Optional[Dict[str, Any]] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Tuple[bool, str]:
    import numpy as np
    logging.info(
        f"Comparing JAX and ONNX outputs with parameters: "
        f"onnx_model_path={onnx_model_path}, inputs={inputs}, "
        f"input_params={input_params}, rtol={rtol}, atol={atol}"
    )
    def is_shape(x):
        return isinstance(x, (tuple, list)) and all(
            isinstance(dim, (int, str)) for dim in x
        )
    if all(is_shape(x) for x in inputs):
        xs = tuple(
            np.random.rand(*[d if isinstance(d, int) else 2 for d in shape]).astype(
                np.float32
            )
            for shape in inputs
        )
    else:
        xs = tuple(inputs)
    if input_params is None:
        return allclose_impl(fn, onnx_model_path, *xs, rtol=rtol, atol=atol)
    else:
        return allclose_impl(
            fn, onnx_model_path, *xs, rtol=rtol, atol=atol, **input_params
        )