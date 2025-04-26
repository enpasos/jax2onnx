# file: jax2onnx/converter/dynamic_utils.py

import numpy as np

import jax
from jax import export as jax_export  # Use alias to avoid conflict if needed
from jax import core, ShapeDtypeStruct
import logging
from typing import List, Sequence, Tuple, Any, Dict, Union

INT64_MAX = np.iinfo(np.int64).max


def encode_dims(seq):
    return np.asarray(
        [d if isinstance(d, int) else INT64_MAX for d in seq], dtype=np.int64
    )


logger_api = logging.getLogger("jax2onnx.conversion_api")  # Or a more specific name


def _create_symbolic_input_avals(
    input_specs: Sequence[Tuple[Sequence[Union[int, str]], Any]],
) -> Tuple[List[ShapeDtypeStruct], Dict[Any, str]]:
    """
    Converts input shape specifications containing strings into abstract
    ShapeDtypeStruct objects containing JAX symbolic dimension objects.

    Args:
        input_specs: A sequence of tuples, where each tuple contains
                     (shape_tuple, dtype). Shape tuples can contain
                     integers or strings representing symbolic dimensions.

    Returns:
        A tuple containing:
        - List[ShapeDtypeStruct]: Abstract values with JAX symbolic objects.
        - Dict[Any, str]: Map from JAX symbolic object back to original string name.
    """
    symbolic_avals = []
    symbol_map: Dict[str, Any] = {}  # Map string name -> JAX symbolic object
    var_to_symbol_map: Dict[Any, str] = {}  # Map JAX object -> string name

    logger_api.debug(f"Creating symbolic avals from input_specs: {input_specs}")

    if not hasattr(jax_export, "symbolic_shape"):
        raise RuntimeError(
            "jax.export.symbolic_shape not found. "
            "Please use JAX version supporting shape polymorphism export APIs."
        )

    for shape_spec, dtype in input_specs:
        processed_shape = []
        if not isinstance(shape_spec, (tuple, list)):
            # Handle scalar shapes potentially passed as single elements
            shape_spec = (shape_spec,)

        for dim in shape_spec:
            if isinstance(dim, str):
                if dim not in symbol_map:
                    try:
                        # Create the JAX symbolic object (e.g., _DimExpr/DimVar)
                        symbol_obj = jax_export.symbolic_shape(dim)[0]
                        logger_api.info(
                            f"Created JAX symbolic object for '{dim}': {symbol_obj} (type: {type(symbol_obj)})"
                        )
                        symbol_map[dim] = symbol_obj
                        # Store reverse mapping as well
                        var_to_symbol_map[symbol_obj] = dim
                        # Add mappings by id and str for potential lookup flexibility later
                        # var_to_symbol_map[id(symbol_obj)] = dim # Maybe less reliable?
                        # var_to_symbol_map[str(symbol_obj)] = dim
                    except Exception as e:
                        logger_api.error(
                            f"Failed to create symbolic shape for dimension '{dim}'. Error: {e}",
                            exc_info=True,
                        )
                        raise ValueError(
                            f"Invalid symbolic dimension specification: '{dim}'"
                        ) from e
                # Use the retrieved/created symbolic object
                processed_shape.append(symbol_map[dim])
            elif isinstance(dim, int):
                processed_shape.append(dim)
            else:
                raise TypeError(
                    f"Invalid dimension type in shape {shape_spec}. "
                    f"Expected int or str, got {type(dim)} ({dim})"
                )

        symbolic_avals.append(ShapeDtypeStruct(tuple(processed_shape), dtype))

    logger_api.debug(f"Created symbolic avals: {symbolic_avals}")
    logger_api.debug(f"Symbol map (str -> obj): {symbol_map}")
    logger_api.debug(f"Reverse map (obj -> str): {var_to_symbol_map}")
    return symbolic_avals, var_to_symbol_map
