# file: jax2onnx/sandbox/poc_symbolic_primitive2.py
import jax
import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive
from jax import export  # Use jax.export for symbolic shapes
import logging
import numpy as np
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
def poc_concat_impl(*arrays, axis):
    # A real implementation isn't needed for abstract eval testing
    # We can just use lax.concatenate here for a placeholder impl
    logger.debug(f"poc_concat_impl called with {len(arrays)} arrays, axis={axis}")
    from jax import lax  # Import inside if needed

    if not arrays:
        return jnp.array([], dtype=jnp.float32)  # Or appropriate default/error
    # lax.concatenate expects a sequence
    return lax.concatenate(arrays, axis=axis)


poc_concat_p.def_impl(poc_concat_impl)


# --- 3. Define Abstract Eval Rule (Manual Calculation Logic) ---
def poc_concat_abstract_eval(*avals: core.AbstractValue, axis: int):
    """
    Abstract evaluation for poc_concat, performs manual shape calculation,
    expecting operable symbolic dimension objects in input avals.
    """
    logger.debug("--- poc_concat_abstract_eval START ---")
    logger.debug(f"Received dimension type: {type(axis)}, value: {axis}")
    logger.debug(f"Received *avals tuple (length {len(avals)}): {avals}")

    # Retrieve the original function from a global variable
    orig_fn = globals().get("_POC_ORIG_FN", None)
    if orig_fn is None:
        raise RuntimeError(
            "Original function not set. Please set _POC_ORIG_FN before tracing."
        )

    shape_dtype_structs = [
        (
            jax.ShapeDtypeStruct(a.shape, jnp.float32)
            if isinstance(a, core.ShapedArray)
            else a
        )
        for a in avals
    ]

    f = lambda a, b: orig_fn(a, b, axis=axis)

    try:
        # Ensure dimension is a Python int (not a tracer or symbolic value)
        if not isinstance(axis, int):
            raise TypeError(
                f"Expected axis/dimension to be a Python int, got {type(axis)}: {axis}"
            )
        exported = export.export(jax.jit(f))(*shape_dtype_structs)
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise

    print("Export complete!")
    print(f"Input shapes: {exported.in_avals}")
    print(f"Output shapes: {exported.out_avals}")

    # convert exported.out_avals to a list of tuples
    final_output_shape_tuple = []
    for aval in exported.out_avals:
        if isinstance(aval, core.ShapedArray):
            final_output_shape_tuple.append(aval.shape)
        else:
            raise TypeError(f"Unsupported output type: {type(aval)}")
    logger.debug(f"Final output shape tuple: {final_output_shape_tuple}")
    logger.debug(f"Final output dtype: {exported.out_avals[0].dtype}")
    output_dtype = exported.out_avals[0].dtype
    logger.debug(f"Output dtype: {output_dtype}")

    result_aval = core.ShapedArray(final_output_shape_tuple, output_dtype)
    return result_aval


poc_concat_p.def_abstract_eval(poc_concat_abstract_eval)


# --- 5. Define JAX Function using the Primitive ---
def poc_concat_wrapper(arrays: Sequence[jax.Array], axis: int):
    # Helper to call the primitive bind
    return poc_concat_p.bind(*arrays, axis=axis)


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

    fn_input = lambda a, b: jnp.concatenate((a, b), axis=1)

    # patch function
    def fn(a, b, *kwargs):
        logger.info("Calling fn (patched JAX function)")
        return poc_concat_wrapper((a, b), dimension=1)

    patched_concat_fn = fn

    target = jax.numpy
    attr = "concatenate"  # The attribute to patch

    # Save the original attribute
    original_concat_fn = getattr(target, attr)

    # Apply the patch - either directly or by calling with the original
    # We determine which approach to use by checking if patch_func accepts parameters

    setattr(target, attr, patched_concat_fn)

    # Set the original function globally for abstract eval
    globals()["_POC_ORIG_FN"] = original_concat_fn

    # Create input ShapeDtypeStructs for tracing
    a_struct = jax.ShapeDtypeStruct((B_var, 1, 8), jnp.float32)
    b_struct = jax.ShapeDtypeStruct((B_var, 10, 8), jnp.float32)

    logger.info("Tracing jaxpr with symbolic shapes...")

    jaxpr = jax.make_jaxpr(fn_input)(a_struct, b_struct)
    logger.info(f"JAXPR:\n{jaxpr}")

    logger.info("--- PoC Script Finished ---")
