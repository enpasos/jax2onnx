# poc_symbolic_primitive.py
import jax
import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive
from jax import export  # Use jax.export for symbolic shapes
import logging
from typing import Sequence

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("POC_SymbolicPrimitive")

# --- 1. Define Custom Primitive ---
poc_concat_p = Primitive("poc_concat")
poc_concat_p.multiple_results = False


# --- 2. Define Implementation (not critical for shape logic) ---
def poc_concat_impl(*arrays, dimension):
    # A real implementation isn't needed for abstract eval testing
    # We can just use lax.concatenate here for a placeholder impl
    logger.debug(
        f"poc_concat_impl called with {len(arrays)} arrays, dimension={dimension}"
    )
    from jax import lax  # Import inside if needed

    if not arrays:
        return jnp.array([], dtype=jnp.float32)  # Or appropriate default/error
    # lax.concatenate expects a sequence
    return lax.concatenate(arrays, dimension=dimension)


poc_concat_p.def_impl(poc_concat_impl)


# --- 3. Define Abstract Eval Rule (Manual Calculation Logic) ---
def poc_concat_abstract_eval(*avals: core.AbstractValue, dimension: int):
    """
    Abstract evaluation for poc_concat, performs manual shape calculation,
    expecting operable symbolic dimension objects in input avals.
    """
    logger.debug("--- poc_concat_abstract_eval START ---")
    logger.debug(f"Received dimension type: {type(dimension)}, value: {dimension}")
    logger.debug(f"Received *avals tuple (length {len(avals)}): {avals}")

    # --- Input Validation ---
    if not avals:
        raise ValueError("poc_concat requires at least one input array.")
    if not all(isinstance(a, core.ShapedArray) for a in avals):
        logger.error(f"Received non-ShapedArray arguments: {[type(a) for a in avals]}")
        raise TypeError("Inputs to abstract_eval must be ShapedArray instances.")

    first_aval = avals[0]

    # --- Dtype Resolution ---
    try:
        output_dtype = jnp.result_type(*[a.dtype for a in avals])
    except Exception as e:
        logger.warning(
            f"Could not compute result_type, falling back to first aval dtype. Error: {e}"
        )
        output_dtype = first_aval.dtype

    # --- Shape Calculation ---
    rank = first_aval.ndim
    # Verify ranks match
    for i, aval in enumerate(avals[1:], 1):
        if aval.ndim != rank:
            all_shapes_repr = [repr(a.shape) for a in avals]
            raise TypeError(
                f"Concatenate inputs must have same rank ({rank}). Got shapes {all_shapes_repr} at index {i}"
            )

    # Ensure dimension is a valid integer (should be passed concretely)
    if not isinstance(dimension, int):
        raise TypeError(f"Dimension must be an integer, got {type(dimension)}")
    if not -rank <= dimension < rank:
        raise ValueError(f"Dimension {dimension} out of range for rank {rank}")
    axis = dimension % rank

    # Helper to get dimension object from shape
    def get_dim(aval, idx):
        # Assume aval.shape contains int or hashable/operable symbolic objects (e.g., DimVar)
        return aval.shape[idx]

    output_shape_list = []
    logger.debug("Calculating output shape dimension by dimension:")
    for i in range(rank):
        dims_at_i = [get_dim(aval, i) for aval in avals]
        logger.debug(
            f"  Axis {i}: Dimensions = {dims_at_i} (Types: {[type(d) for d in dims_at_i]})"
        )

        if i == axis:
            # Concatenation axis: Sum integers, try symbolic math using '+'
            final_axis_dim = 0
            for d_idx, d in enumerate(dims_at_i):
                try:
                    logger.debug(
                        f"    Axis {i} (Concat): Adding dim {d} (type {type(d)}) to current sum {final_axis_dim} (type {type(final_axis_dim)})"
                    )
                    if d_idx == 0:
                        final_axis_dim = d
                    else:
                        # Attempt addition - relies on JAX symbolic objects supporting '+'
                        final_axis_dim = final_axis_dim + d
                    logger.debug(
                        f"      -> New sum: {final_axis_dim} (type {type(final_axis_dim)})"
                    )
                except TypeError as e:
                    logger.error(
                        f"    Failed adding dimensions on axis {i}: {final_axis_dim} + {d}. Error: {e}"
                    )
                    raise TypeError(
                        f"Cannot add dimensions of type {type(final_axis_dim)} and {type(d)} on axis {i}"
                    ) from e
                except Exception as e:
                    logger.error(
                        f"    Unexpected error adding dimensions on axis {i}: {final_axis_dim} + {d}. Error: {e}",
                        exc_info=True,
                    )
                    raise
            output_shape_list.append(final_axis_dim)
            logger.debug(f"  Axis {i} (Concat) Result: {final_axis_dim}")
        else:
            # Non-concatenation axis: Check consistency using '=='
            representative_dim = dims_at_i[0]
            for k in range(1, len(dims_at_i)):
                current_dim = dims_at_i[k]
                try:
                    # Attempt comparison - relies on JAX symbolic objects supporting '=='
                    logger.debug(
                        f"    Axis {i} (Non-Concat): Comparing {representative_dim} ({type(representative_dim)}) == {current_dim} ({type(current_dim)})"
                    )
                    are_equal = representative_dim == current_dim
                    logger.debug(f"      -> Comparison result: {are_equal}")
                    if not are_equal:
                        raise TypeError(
                            f"Concat incompatible dimensions at non-concat axis {i}: "
                            f"{representative_dim} vs {current_dim}"
                        )
                except TypeError as e:
                    logger.error(
                        f"    Failed comparing dimensions {representative_dim} ({type(representative_dim)}) and "
                        f"{current_dim} ({type(current_dim)}) on axis {i}: {e}"
                    )
                    raise TypeError(f"Cannot compare dimensions on axis {i}") from e
                except Exception as e:
                    logger.error(
                        f"    Unexpected error comparing dimensions on axis {i}: {representative_dim} vs {current_dim}. Error: {e}",
                        exc_info=True,
                    )
                    raise
            output_shape_list.append(representative_dim)
            logger.debug(f"  Axis {i} (Non-Concat) Result: {representative_dim}")

    # --- Return ShapedArray with computed shape tuple ---
    final_output_shape_tuple = tuple(output_shape_list)
    logger.debug(
        f"poc_concat_abstract_eval - Final output shape tuple: {final_output_shape_tuple}"
    )

    try:
        # This tuple should contain integers and JAX's hashable symbolic objects (DimVar/DimExpr)
        result_aval = core.ShapedArray(final_output_shape_tuple, output_dtype)
        logger.debug("--- poc_concat_abstract_eval END (Success) ---")
        return result_aval
    except TypeError as e:
        logger.error(
            f"TypeError creating ShapedArray with computed shape tuple: {final_output_shape_tuple}. Error: {e}",
            exc_info=True,
        )
        logger.debug(
            "--- poc_concat_abstract_eval END (ShapedArray Creation Failed) ---"
        )
        raise


# --- 4. Register Primitive Rule ---
poc_concat_p.def_abstract_eval(poc_concat_abstract_eval)


# --- 5. Define JAX Function using the Primitive ---
def poc_concat_wrapper(arrays: Sequence[jax.Array], dimension: int):
    # Helper to call the primitive bind
    return poc_concat_p.bind(*arrays, dimension=dimension)


def test_func(a, b):
    # Example function concatenating along axis 1
    logger.info("Entering test_func (JAX function being traced)")
    result = poc_concat_wrapper((a, b), dimension=1)
    logger.info("Exiting test_func")
    return result


# --- 6. Main PoC Logic ---
if __name__ == "__main__":
    logger.info("--- Starting PoC Script ---")

    # Create symbolic dimension 'B' using jax.export API
    try:
        # symbolic_shape returns a tuple, e.g., ('B',) for "B"
        # We need the element itself, often a DimVar
        B_var = export.symbolic_shape("B")[0]
        logger.info(f"Created symbolic dimension 'B': {B_var} (type: {type(B_var)})")
    except Exception as e:
        logger.error(f"Failed to create symbolic dimension 'B': {e}", exc_info=True)
        exit()

    # Create input ShapeDtypeStruct using the symbolic dimension
    aval1 = core.ShapedArray((B_var, 1, 8), jnp.float32)
    aval2 = core.ShapedArray((B_var, 10, 8), jnp.float32)
    logger.info(f"Input aval1: {aval1}")
    logger.info(f"Input aval2: {aval2}")

    # Use jax.eval_shape to trace test_func with symbolic inputs
    logger.info("Calling jax.eval_shape...")
    try:
        output_aval = jax.eval_shape(test_func, aval1, aval2)
        logger.info("--- jax.eval_shape SUCCESS ---")
        logger.info(f"Output Aval: {output_aval}")
        logger.info(f"Output Shape: {output_aval.shape}")
        # Expected output shape: (B, 11, 8) where B is the symbolic object
        # And the 11 is computed by the abstract_eval rule (1 + 10)
    except Exception as e:
        logger.error(f"!!! jax.eval_shape FAILED: {e}", exc_info=True)

    logger.info("--- PoC Script Finished ---")
