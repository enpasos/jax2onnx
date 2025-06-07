import numpy as np
from jax import export as jax_export
from jax import ShapeDtypeStruct
import logging
from typing import List, Sequence, Tuple, Any, Dict, Union
import jax.numpy as jnp

INT64_MAX = np.iinfo(np.int64).max


def encode_dims(seq: Sequence[Union[int, Any]]):  # Added type hint for seq
    return np.asarray(
        [d if isinstance(d, int) else INT64_MAX for d in seq], dtype=np.int64
    )


logger_api = logging.getLogger("jax2onnx.conversion_api")


def _create_symbolic_input_avals(
    input_specs: Sequence[
        Union[
            Sequence[Union[int, str]],  # shape-only
            Tuple[Sequence[Union[int, str]], Any],  # (shape, dtype)
        ]
    ],
) -> Tuple[List[ShapeDtypeStruct], Dict[Any, str]]:
    """
    Converts input shape specifications into abstract ShapeDtypeStructs
    containing JAX symbolic dimension objects, ensuring a unified scope.
    """
    logger_api.debug(f"Creating symbolic avals from input_specs: {input_specs}")

    if not hasattr(jax_export, "symbolic_shape"):
        raise RuntimeError(
            "jax.export.symbolic_shape not found. "
            "Please use JAX version supporting shape polymorphism export APIs."
        )

    # 1. Collect all unique symbolic dimension names from all input specs
    all_symbol_names = set()
    for spec in input_specs:
        if (
            isinstance(spec, (tuple, list))
            and len(spec) == 2
            and isinstance(spec[0], (tuple, list))
        ):
            shape_seq, _ = spec
        else:
            shape_seq = spec

        shape_seq_iterable = (
            (shape_seq,) if not isinstance(shape_seq, (tuple, list)) else shape_seq
        )
        for dim in shape_seq_iterable:
            if isinstance(dim, str):
                all_symbol_names.add(dim)

    # 2. Create all symbolic objects in a single call to ensure one scope
    symbol_map: Dict[str, Any] = {}
    if all_symbol_names:
        sorted_symbols = sorted(list(all_symbol_names))
        combined_spec = ",".join(sorted_symbols)
        try:
            created_symbol_objects = jax_export.symbolic_shape(combined_spec)
            symbol_map = dict(zip(sorted_symbols, created_symbol_objects))
            for dim_name, sym_obj in symbol_map.items():
                logger_api.info(
                    f"Created JAX symbolic object for '{dim_name}': {sym_obj} (type: {type(sym_obj)})"
                )
        except Exception as e:
            logger_api.error(
                f"Failed to create symbolic shapes for spec '{combined_spec}'. Error: {e}",
                exc_info=True,
            )
            raise ValueError(
                f"Invalid symbolic dimension specification: '{combined_spec}'"
            ) from e

    # 3. Build the avals using the pre-built, correctly-scoped symbol map
    symbolic_avals: List[ShapeDtypeStruct] = []
    for spec in input_specs:
        if (
            isinstance(spec, (tuple, list))
            and len(spec) == 2
            and isinstance(spec[0], (tuple, list))
        ):
            shape_seq, dtype = spec
        else:
            shape_seq, dtype = spec, jnp.float32

        shape_seq_iterable = (
            (shape_seq,) if not isinstance(shape_seq, (tuple, list)) else shape_seq
        )

        processed_shape = [symbol_map.get(d, d) for d in shape_seq_iterable]
        current_shape_for_struct = tuple(processed_shape)
        symbolic_avals.append(ShapeDtypeStruct(current_shape_for_struct, dtype))

    var_to_symbol_map: Dict[Any, str] = {v: k for k, v in symbol_map.items()}

    logger_api.debug(f"Created symbolic avals: {symbolic_avals}")
    logger_api.debug(f"Symbol map (str -> obj): {symbol_map}")
    logger_api.debug(f"Reverse map (obj -> str): {var_to_symbol_map}")
    return symbolic_avals, var_to_symbol_map
