# file: jax2onnx/plugins/jax/numpy/concat.py

# --- Imports (Ensure these are present) ---
from typing import TYPE_CHECKING, Callable, Any, Sequence  # Added Sequence
import functools
import inspect
import jax
from jax import core
from jax import lax
from jax import numpy as jnp
from jax.extend.core import Primitive
from onnx import helper

# --- Import jax.export ---
from jax import export

# -------------------------
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
import logging
import numpy as np

# --- Import the wrapper used by the patching mechanism ---
# (Make sure this path is correct relative to your project structure)
from jax2onnx.converter.patched_callable_wrapper import PatchedCallableWrapper

# -------------------------------------------------------

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.concat")

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# --- Primitive Definition (likely already exists) ---
if not hasattr(jnp, "concat_p"):
    jnp.concat_p = Primitive("jnp.concat")
    jnp.concat_p.multiple_results = False


# --- Registration (likely already exists, keep testcases) ---
@register_primitive(
    jaxpr_primitive=jnp.concat_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concat.html",
    onnx=[
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="concat",
    testcases=[  # Make sure testcases are defined correctly
        {
            "testcase": "concat",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=0),
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "concat_abstract_middle_dim",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=1),
            "input_shapes": [("B", 1, 8), ("B", 10, 8)],
            # Add expected shape for clarity, matching the PoC's log
            "expected_output_shapes": [("B", 11, 8)],
        },
    ],
)
class ConcatPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.numpy.concatenate using the jax.export approach
    for robust symbolic shape handling.
    """

    # --- Store original function reference here ---
    _ORIGINAL_CONCAT: Callable | None = None
    # --------------------------------------------

    # === NEW abstract_eval using jax.export ===
    @staticmethod
    def abstract_eval(
        *avals: core.ShapedArray, axis: int
    ):  # Signature assumes axis is static
        """
        Compute output shape/dtype using `jax.export`. Relies on the original
        function being stored in ConcatPlugin._ORIGINAL_CONCAT by patch_info.
        """
        logger.debug("ConcatPlugin: Entering NEW abstract_eval (using jax.export)")
        logger.debug(f"Received avals: {avals}")
        logger.debug(f"Received axis: {axis} (type: {type(axis)})")

        # --- Input Validation ---
        if not avals:
            raise ValueError("Concat requires at least one input array.")
        if not all(isinstance(a, core.ShapedArray) for a in avals):
            logger.error(f"Received non-ShapedArray args: {[type(a) for a in avals]}")
            raise TypeError("Inputs must be ShapedArray instances.")
        if not isinstance(axis, int):
            # Should not happen if axis is static, but good practice to check
            raise TypeError(f"Axis must be an integer, got {type(axis)}")

        # --- Retrieve the original function ---
        if ConcatPlugin._ORIGINAL_CONCAT is None:
            raise RuntimeError(
                "Original jnp.concatenate reference not set in ConcatPlugin._ORIGINAL_CONCAT. Patching might have failed."
            )
        orig_concat = ConcatPlugin._ORIGINAL_CONCAT
        logger.debug(f"Using original function: {orig_concat.__name__}")

        # --- Build ShapeDtypeStructs for export ---
        # jax.export needs ShapeDtypeStruct, not ShapedArray
        try:
            shapespecs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in avals]
            logger.debug(f"Created ShapeDtypeStructs: {shapespecs}")
        except Exception as e:
            logger.error(
                f"Failed to create ShapeDtypeStructs from avals: {avals}. Error: {e}",
                exc_info=True,
            )
            raise

        # --- Define the helper function for export ---
        # It takes individual arrays (*args) and concatenates them
        def f(*args_for_f):
            logger.debug(f"  Helper 'f' called with {len(args_for_f)} args")
            # Ensure args are passed as a sequence to the original concat
            return orig_concat(args_for_f, axis=axis)

        # --- Call jax.export ---
        logger.debug(
            f"Calling jax.export.export(jax.jit(f)) with {len(shapespecs)} specs..."
        )
        try:
            # jax.jit might help optimize the helper function before export
            exported = export.export(jax.jit(f))(*shapespecs)
            out_aval = exported.out_avals[0]  # Assuming single result
            logger.debug(f"jax.export successful. Output aval: {out_aval}")
        except Exception as e:
            logger.error(f"jax.export.export failed: {e}", exc_info=True)
            # Provide more context on failure
            logger.error(f"  Original Function: {orig_concat}")
            logger.error(f"  Axis: {axis}")
            logger.error(f"  Input ShapeSpecs: {shapespecs}")
            raise

        # --- Return the result as ShapedArray ---
        logger.debug(
            f"Returning ShapedArray: shape={out_aval.shape}, dtype={out_aval.dtype}"
        )
        # The output from export is already an AbstractValue (like ShapedArray or ShapeDtypeStruct)
        # but we explicitly return ShapedArray as expected by the abstract_eval contract.
        return core.ShapedArray(out_aval.shape, out_aval.dtype)

    # === End NEW abstract_eval ===

    # === Keep existing to_onnx ===
    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles ONNX conversion for jnp.concat_p."""
        logger.debug(f"ConcatPlugin: Converting node to ONNX.")

        # --- Retrieve axis ---
        # It should be passed as a static parameter from abstract_eval
        if "axis" not in params:
            logger.warning(
                "Axis parameter not found in 'to_onnx' params, defaulting to 0."
            )
            axis = 0
        else:
            axis = params["axis"]
            # Ensure axis is int here too
            if not isinstance(axis, int):
                try:
                    axis = int(axis)  # Should be safe if static
                except Exception:
                    raise TypeError(
                        f"Axis param in to_onnx must be int, got {type(axis)}"
                    )

        logger.debug(f"Using axis: {axis}")

        input_names = [s.get_name(var) for var in node_inputs]
        output_name = s.get_name(node_outputs[0])

        # Add ONNX node
        concat_node = helper.make_node(
            "Concat",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("concat"),
            axis=axis,
        )
        s.add_node(concat_node)
        logger.debug(f"Added ONNX Concat node for output {output_name}")

        # Add shape info - Retrieve from the output variable's aval
        # This aval was computed by our new abstract_eval method
        out_aval = node_outputs[0].aval
        output_shape = out_aval.shape
        dtype = out_aval.dtype
        logger.debug(
            f"Adding shape info for {output_name}: shape={output_shape}, dtype={dtype}"
        )
        s.add_shape_info(output_name, tuple(output_shape), dtype)

    # === End to_onnx ===

    # === MODIFIED patch_info to store original function ===
    @staticmethod
    def patch_info() -> dict[str, Any]:
        """Returns parameters needed for monkey patching jnp.concatenate."""
        logger.debug("ConcatPlugin: Getting patch_info")

        # This function is called by the patching utility.
        # It receives the original function to be patched.
        def patch_creator(original_jnp_concatenate: Callable):
            logger.info(
                f"Patching '{original_jnp_concatenate.__name__}'. Storing original reference."
            )
            # --- Store the original function reference ---
            ConcatPlugin._ORIGINAL_CONCAT = original_jnp_concatenate
            # --------------------------------------------
            # Return an instance of the wrapper, initialized with the original function
            # and the primitive this plugin handles (jnp.concat_p)
            return PatchedCallableWrapper(original_jnp_concatenate, jnp.concat_p)

        return {
            "patch_targets": [jnp],  # Target module: jax.numpy
            "patch_function": patch_creator,  # Function that creates the patch object
            "target_attribute": "concatenate",  # Function name within jax.numpy
        }

    # === End MODIFIED patch_info ===


# --- Register the NEW abstract evaluation rule ---
jnp.concat_p.def_abstract_eval(ConcatPlugin.abstract_eval)
# -------------------------------------------------
