# file: jax2onnx/converter/dynamic_utils.py

import numpy as np
from jax import export as jax_export
from jax import ShapeDtypeStruct
import logging
from typing import List, Sequence, Tuple, Any, Dict, Union

INT64_MAX = np.iinfo(np.int64).max


def encode_dims(seq: Sequence[Union[int, Any]]):  # Added type hint for seq
    return np.asarray(
        [d if isinstance(d, int) else INT64_MAX for d in seq], dtype=np.int64
    )


logger_api = logging.getLogger("jax2onnx.conversion_api")


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
    symbolic_avals: List[ShapeDtypeStruct] = []
    symbol_map: Dict[str, Any] = {}  # Map string name -> JAX symbolic object
    var_to_symbol_map: Dict[Any, str] = {}  # Map JAX object -> string name

    logger_api.debug(f"Creating symbolic avals from input_specs: {input_specs}")

    if not hasattr(jax_export, "symbolic_shape"):
        raise RuntimeError(
            "jax.export.symbolic_shape not found. "
            "Please use JAX version supporting shape polymorphism export APIs."
        )

    for shape_spec, dtype in input_specs:
        # shape_spec is already Sequence[Union[int, str]], e.g., ('B', 10)
        processed_shape: List[Union[int, Any]] = (
            []
        )  # Store ints or JAX symbolic objects

        if not isinstance(shape_spec, (tuple, list)):
            # Handle scalar shapes potentially passed as single elements
            # Allow either a flat tuple or a tuple-of-tuples here
            shape_spec_iterable: Union[
                Sequence[Union[int, str]], Sequence[Sequence[Union[int, str]]]
            ] = (shape_spec,)
        else:
            shape_spec_iterable = shape_spec

        for dim in shape_spec_iterable:  # dim is Union[int, str]
            if isinstance(dim, str):
                if dim not in symbol_map:
                    try:
                        symbol_tuple: Tuple[Any, ...] = jax_export.symbolic_shape(dim)
                        symbol_obj: Any = symbol_tuple[
                            0
                        ]  # This is line 55 in new context
                        logger_api.info(
                            f"Created JAX symbolic object for '{dim}': {symbol_obj} (type: {type(symbol_obj)})"
                        )
                        symbol_map[dim] = symbol_obj
                        var_to_symbol_map[symbol_obj] = dim
                    except Exception as e:
                        logger_api.error(
                            f"Failed to create symbolic shape for dimension '{dim}'. Error: {e}",
                            exc_info=True,
                        )
                        raise ValueError(
                            f"Invalid symbolic dimension specification: '{dim}'"
                        ) from e
                processed_shape.append(symbol_map[dim])  # Appends the symbolic object
            elif isinstance(dim, int):
                processed_shape.append(dim)  # Appends the integer
            else:
                # This case should not be reached if shape_spec_iterable is Sequence[Union[int, str]]
                raise TypeError(
                    f"Invalid dimension type in shape {shape_spec}. "
                    f"Expected int or str, got {type(dim)} ({dim})"
                )

        current_shape_for_struct: Tuple[Union[int, Any], ...] = tuple(processed_shape)
        symbolic_avals.append(ShapeDtypeStruct(current_shape_for_struct, dtype))

    logger_api.debug(f"Created symbolic avals: {symbolic_avals}")
    logger_api.debug(f"Symbol map (str -> obj): {symbol_map}")
    logger_api.debug(f"Reverse map (obj -> str): {var_to_symbol_map}")
    return symbolic_avals, var_to_symbol_map
