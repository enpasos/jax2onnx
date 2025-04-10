# file: jax2onnx/converter/user_interface.py

import onnx
from typing import Any, Optional, Dict
import numpy as np
import onnxruntime as ort
from jax2onnx.converter.jax_to_onnx import to_onnx as core_to_onnx
from collections import defaultdict
import jax.numpy as jnp


def to_onnx(
    fn: Any,
    input_shapes: Any,
    model_name: str = "jax_model",
    opset: int = 21,
    *,
    kwargs: Optional[Dict[str, Any]] = None,
) -> onnx.ModelProto:
    """Convert a JAX function or Module to an ONNX model."""
    kwargs = kwargs or {}
    input_params = kwargs.pop("input_params", {})

    if input_params:
        if not isinstance(input_shapes, (list, tuple)):
            input_shapes = [input_shapes]
        for _ in input_params.values():
            input_shapes.append(())

        old_fn = fn

        def wrapped_fn(*args):
            n_tensor_args = len(args) - len(input_params)
            call_args = args[:n_tensor_args]
            param_keys = list(input_params.keys())
            param_values = args[n_tensor_args:]

            converted_kwargs = {}
            for k, v in zip(param_keys, param_values):
                expected_type = type(input_params[k])
                if expected_type is bool:
                    converted_kwargs[k] = jnp.asarray(v, dtype=jnp.bool_).reshape(())
                elif expected_type is int:
                    converted_kwargs[k] = jnp.asarray(v, dtype=jnp.int32).reshape(())
                elif expected_type is float:
                    converted_kwargs[k] = jnp.asarray(v, dtype=jnp.float32).reshape(())
                else:
                    raise ValueError(f"Unsupported input_param type: {expected_type}")
            return old_fn(*call_args, **converted_kwargs)

        fn = wrapped_fn

    return core_to_onnx(
        fn=fn,
        input_shapes=input_shapes,
        model_name=model_name,
        opset=opset,
    )


def save_onnx(
    fn: Any,
    input_shapes: Any,
    output_path: str = "model.onnx",
    model_name: str = "jax_model",
    opset: int = 21,
    hierarchical: bool = False,
):
    onnx_model = to_onnx(
        fn,
        input_shapes,
        model_name=model_name,
        opset=opset,
        hierarchical=hierarchical,
    )
    onnx.save_model(onnx_model, output_path)


def allclose(callable, onnx_model_path, *xs):
    # Load ONNX model and create inference session
    session = ort.InferenceSession(onnx_model_path)

    # Extract actual input names from model
    input_names = [inp.name for inp in session.get_inputs()]

    if len(input_names) != len(xs):
        raise ValueError(f"Expected {len(input_names)} inputs, but got {len(xs)}.")

    # Run ONNX
    p = {name: np.array(x) for name, x in zip(input_names, xs)}
    onnx_output = session.run(None, p)

    # Split positional inputs vs dynamic kwargs (heuristic: last N are scalars)
    def is_scalar(x):
        return isinstance(x, jnp.ndarray) and x.shape == ()

    tensor_args = [x for x in xs if not is_scalar(x)]
    scalar_args = [x for x in xs if is_scalar(x)]

    # Dynamically guess input_param names based on Dropout.__call__ signature
    import inspect

    sig = inspect.signature(callable.__call__)
    param_names = list(sig.parameters)[1:]  # skip self
    tensor_names = param_names[: len(tensor_args)]
    scalar_names = param_names[len(tensor_args) :]

    dynamic_kwargs = dict(zip(scalar_names, scalar_args))

    try:
        jax_output = callable(*tensor_args, **dynamic_kwargs)
    except TypeError:
        # fallback for lambdas or callables that don't accept kwargs
        jax_output = callable(*xs)

    if not isinstance(jax_output, list):
        jax_output = [jax_output]
    if not isinstance(onnx_output, list):
        onnx_output = [onnx_output]

    isOk = np.allclose(onnx_output, jax_output, rtol=1e-3, atol=1e-5)

    return (
        isOk,
        (
            "ONNX and JAX outputs match :-)"
            if isOk
            else "ONNX and JAX outputs do not match :-("
        ),
    )


class ModelExportContext:
    """
    Holds model-specific state for naming and caching.
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id: str = model_id or "default_model"
        self.function_cache: Dict[str, Any] = {}
        self.instance_counters: Dict[str, int] = defaultdict(int)

    def next_function_name(self, base_name: str) -> str:
        """
        Generates a unique ONNX function name scoped to this model.
        E.g., TransformerBlock_1, TransformerBlock_2, ...
        """
        self.instance_counters[base_name] += 1
        return f"{base_name}_{self.instance_counters[base_name]}"
