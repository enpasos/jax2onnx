# file: jax2onnx/converter/user_interface.py
"""
High-level public façade around the core converter.

Key additions
-------------
* If the incoming object is a raw *initialised* **or** *un-initialised*
  `flax.linen.Module`, we:
    1. Wrap it in `nnx.bridge.ToNNX`
    2. Lazily initialise that bridge from the first input spec
    3. Hand the bridged version to the dedicated Linen-bridge handler.
* Existing behaviour for an already-bridged model (instance of
  `nnx.bridge.ToNNX`) and for plain JAX callables is unchanged.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import argparse
import logging

import jax.numpy as jnp
from jax import config, core
import onnx

from flax import nnx
from flax.linen import Module as LinenModule

from jax2onnx.converter.conversion_api import to_onnx as _impl_to_onnx
from jax2onnx.converter.validation     import allclose as _impl_allclose
from jax2onnx.plugin_system            import onnx_function as _impl_onnx_function

from . import linen_handler                        # bridge logic
from .linen_handler import _suspend_flax_shape_check      # silences Flax param checks

config.update("jax_dynamic_shapes", True)

_FLOAT64_HELP = (
    "Export the ONNX graph in double precision (tensor(double)). "
    "If omitted, tensors are exported in single precision (tensor(float))."
)

# --------------------------------------------------------------------------- #
# Helper: build a dummy array from a shape tuple that may contain symbols.
# --------------------------------------------------------------------------- #
def _dummy_from_shape(shape: tuple[Any, ...]) -> jnp.ndarray:
    concrete = [d if isinstance(d, int) else 2 for d in shape]
    return jnp.ones(tuple(concrete), jnp.float32)


# --------------------------------------------------------------------------- #
# Public entry-point
# --------------------------------------------------------------------------- #
def to_onnx(                           # noqa: C901 – single public API is OK long
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
    Convert *fn* (plain JAX callable **or** Flax model) into an ONNX graph.

    The function transparently handles three cases

    1. **Already bridged** – `isinstance(fn, nnx.bridge.ToNNX)`
    2. **Raw Linen module** – `isinstance(fn, flax.linen.Module)`
       (we create & initialise the bridge here)
    3. **Plain callable** – hands straight to the converter.
    """
    target_fn   = fn
    target_inps = inputs
    is_bridge   = False

    # ------------------------------------------------------------------ #
    # Case 1 – fn is *already* a nnx.bridge.ToNNX wrapper
    # ------------------------------------------------------------------ #
    if isinstance(fn, nnx.bridge.ToNNX):
        is_bridge = True

    # ------------------------------------------------------------------ #
    # Case 2 – raw Linen module ➜ build + init bridge now
    # ------------------------------------------------------------------ #
    elif isinstance(fn, LinenModule):
        logging.info("Wrapping raw Flax Linen module in nnx.bridge.ToNNX…")

        # 2-a) Create the bridge
        bridged = nnx.bridge.ToNNX(fn, rngs=nnx.Rngs(0))

        # 2-b) Lazy-initialise using a dummy from the *first* input spec
        if not inputs:
            raise ValueError(
                "When passing a Flax Linen module you must also supply a non-empty "
                "`inputs` list so a dummy batch can be constructed for lazy "
                "initialisation."
            )

        first = inputs[0]
        if isinstance(first, (tuple, list)):
            dummy = _dummy_from_shape(tuple(first))
        else:
            dummy = jnp.ones(first.shape, first.dtype)

        bridged = nnx.bridge.lazy_init(bridged, dummy)

        # Replace *fn* with the freshly initialised bridge
        fn          = bridged
        target_fn   = bridged      # will be unwrapped by linen_handler next
        target_inps = inputs
        is_bridge   = True

    # ------------------------------------------------------------------ #
    # (Case 3 – plain callable) nothing special to do
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # If we have a bridge of any sort, hand it to the dedicated handler
    # ------------------------------------------------------------------ #
    if is_bridge:
        pure_fn, trace_args = linen_handler.to_onnx_for_linen_bridge(
            fn,
            inputs,
            model_name=model_name,
            opset=opset,
            input_params=input_params,
            enable_double_precision=enable_double_precision,
            record_primitive_calls_file=record_primitive_calls_file,
        )
        target_fn   = pure_fn
        target_inps = trace_args

    # ------------------------------------------------------------------ #
    # Build inputs list (ShapeDtypeStructs etc.) for the low-level converter
    # ------------------------------------------------------------------ #
    processed: list[Any] = []

    if target_inps:
        if is_bridge:
            # 1st entry = params pytree:
            processed.append(target_inps[0])
            user_inputs = target_inps[1:]
        else:
            user_inputs = target_inps

        if all(isinstance(x, core.ShapedArray) for x in user_inputs):
            processed.extend(user_inputs)
        else:
            def _is_shape_tuple(x):
                return isinstance(x, (tuple, list)) and all(
                    isinstance(dim, (int, str)) for dim in x
                )

            if all(_is_shape_tuple(x) for x in user_inputs):
                processed.extend(user_inputs)
            else:
                import jax
                processed.extend(
                    [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in user_inputs]
                )

    # ------------------------------------------------------------------ #
    # Finally call the core implementation *inside* the lenient-shape context
    # so Flax does not choke on symbolic dimensions.
    # ------------------------------------------------------------------ #
    with _suspend_flax_shape_check():
        return _impl_to_onnx(
            fn=target_fn,
            inputs=processed,
            input_params=input_params,
            model_name=model_name,
            opset=opset,
            enable_double_precision=enable_double_precision,
            record_primitive_calls_file=record_primitive_calls_file,
        )


# --------------------------------------------------------------------------- #
# Boiler-plate command-line & helper wrappers (unchanged)
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jax2onnx",
        description="Convert a JAX function to an ONNX model.",
    )
    p.add_argument("module", help="Python module containing the JAX function")
    p.add_argument("fn",     help="Name of the JAX function inside the module")
    p.add_argument("--out",   default="model.onnx", help="Output .onnx file")
    p.add_argument("--opset", type=int, default=21,       help="ONNX opset version")
    p.add_argument(
        "--float64", dest="enable_double_precision",
        action="store_true", default=False, help=_FLOAT64_HELP,
    )
    p.add_argument(
        "--record-primitives", dest="record_primitive_calls_file",
        default=None, help="File path to dump primitive call trace",
    )
    return p


def run_command_line():
    args = _build_arg_parser().parse_args()
    import importlib, sys
    sys.path.append(".")
    try:
        mod = importlib.import_module(args.module)
        fn  = getattr(mod, args.fn)
    except (ImportError, AttributeError) as err:
        logging.error(f"Error loading function: {err}")
        raise SystemExit(1)

    to_onnx(
        fn,
        inputs=[],                  # CLI mode expects shape inference inside fn
        model_name=args.fn,
        opset=args.opset,
        enable_double_precision=args.enable_double_precision,
        record_primitive_calls_file=args.record_primitive_calls_file,
    )


def convert(
    *, enable_double_precision: bool = False,
    record_primitive_calls_file: Optional[str] = None,
    **kwargs,
):
    return to_onnx(
        enable_double_precision=enable_double_precision,
        record_primitive_calls_file=record_primitive_calls_file,
        **kwargs,
    )


def onnx_function(target: Union[Callable, type]) -> Union[Callable, type]:
    return _impl_onnx_function(target)


def allclose(
    fn: Callable,
    onnx_model_path: str,
    inputs: List[Any],
    input_params: Optional[Dict[str, Any]] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Tuple[bool, str]:
    """Shell around the validator that generates dummy data for shape specs."""
    import numpy as np

    def _is_shape(x):
        return isinstance(x, (tuple, list)) and all(
            isinstance(dim, (int, str)) for dim in x
        )

    if all(_is_shape(x) for x in inputs):
        xs = tuple(
            np.random.randn(*[d if isinstance(d, int) else 2 for d in shape]).astype(
                np.float32
            )
            for shape in inputs
        )
    else:
        xs = tuple(inputs)

    if input_params is None:
        return _impl_allclose(fn, onnx_model_path, *xs, rtol=rtol, atol=atol)
    return _impl_allclose(
        fn, onnx_model_path, *xs, rtol=rtol, atol=atol, **input_params
    )