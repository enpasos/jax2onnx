# file: jax2onnx/plugins/jax/numpy/concat.py

# --- Imports ---
from typing import TYPE_CHECKING, Callable, Any
import functools
import inspect
import jax
from jax import core
from jax import lax  # Import lax for potential use if needed
from jax import numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
import logging
import numpy as np

from jax import export
from jax2onnx.converter.patched_callable_wrapper import PatchedCallableWrapper

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.concat")

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# --- Wrapper Definition (place PatchedCallableWrapper class here) ---
# class PatchedCallableWrapper: ... (definition from above)

# --- Primitive Definition ---
if not hasattr(jnp, "concat_p"):
    jnp.concat_p = Primitive("jnp.concat")
    jnp.concat_p.multiple_results = False


# --- Registration ---
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
            "expected_output_shapes": [("B", 11, 8)],
        },
    ],
)
class ConcatPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.numpy.concatenate using a custom primitive,
    a PatchedCallableWrapper, and abstract_eval calling jax.eval_shape
    on the original function passed via params.
    """

    @staticmethod
    def abstract_eval(*avals, **params):  # Signature receives *avals and **params
        """
        Performs abstract evaluation by retrieving the original function
        from params and calling jax.eval_shape. (Re-testing this approach).
        """
        logger.debug("ConcatPlugin: Entering abstract_eval (eval_shape attempt)")
        logger.debug(f"Received avals: {avals}")
        logger.debug(f"Received params: {params.keys()}")

        # --- Retrieve original function and axis from params ---
        if "_original_fn" not in params:
            raise ValueError("'_original_fn' not found in abstract_eval params.")
        original_fn = params["_original_fn"]

        if "axis" not in params:
            raise ValueError("'axis' not found in abstract_eval params.")
        axis = params["axis"]

        # Ensure axis is concrete integer (check just in case)
        if isinstance(axis, core.Tracer):
            raise TypeError("Axis cannot be tracer in abstract_eval params.")
        if not isinstance(axis, int):
            try:
                axis = int(axis)
            except Exception:
                raise TypeError(f"Axis param must be int, got {type(axis)}")

        # The inputs *avals are already the abstract values
        if not all(isinstance(a, core.AbstractValue) for a in avals):
            logger.error(
                f"Received non-AbstractValue arguments in *avals: {[type(a) for a in avals]}"
            )
            raise TypeError("Inputs to abstract_eval must be AbstractValue instances.")
        if not avals:
            raise ValueError(
                "abstract_eval received no AbstractValue arguments in *avals."
            )

        logger.debug(
            f"Calling jax.eval_shape on {original_fn.__name__} with avals: {avals}, axis: {axis}"
        )

        def f(*args):
            return original_fn(args, axis=axis)

        try:

            # avals come in ShapedArray form, so we need to convert them to  jax.ShapeDtypeStruct(a_shape, jnp.float32)

            shape_dtype_structs = [
                (
                    jax.ShapeDtypeStruct(a.shape, jnp.float32)
                    if isinstance(a, core.ShapedArray)
                    else a
                )
                for a in avals
            ]

            exported = export.export(jax.jit(f))(*shape_dtype_structs)
            result_avals = exported.out_avals

        except TypeError as e:
            # Catching the expected TypeError
            logger.error(
                f"Caught expected TypeError from jax.eval_shape: {e}", exc_info=True
            )
            logger.error(
                "This confirms jax.eval_shape cannot be called with AbstractValue inputs from within abstract_eval."
            )
            # Re-raise the original error to stop the test execution clearly
            raise
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error during jax.eval_shape: {e}", exc_info=True)
            # logger.error(f"Inputs to eval_shape - Partial Func: {partial_fn}, Avals Tuple: {tuple(avals)}")
            raise

        logger.debug(f"Result aval from jax.eval_shape (if successful): {result_avals}")
        return result_avals  # This line likely won't be reached

    def get_patch_params(self) -> tuple[Any, str, Callable]:
        """Returns parameters needed for monkey patching jnp.concatenate with the wrapper."""
        logger.debug("ConcatPlugin: Getting patch params")

        # This function will be called by the patching utility (e.g., _temporary_patch).
        # It receives the original jnp.concatenate function.
        def patch_creator(original_jnp_concatenate: Callable):
            logger.info(
                f"Creating PatchedCallableWrapper for {original_jnp_concatenate.__name__}"
            )
            # Return an instance of the wrapper, initialized with the original function
            # and the primitive this plugin handles (jnp.concat_p)
            return PatchedCallableWrapper(original_jnp_concatenate, jnp.concat_p)

        # Target module, attribute name, and the function that creates the patch object
        return (jnp, "concatenate", patch_creator)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles ONNX conversion for jnp.concat_p."""
        logger.debug(
            f"ConcatPlugin: Converting node to ONNX. Params keys: {list(params.keys())}"
        )

        # --- Retrieve axis, ignoring internal _original_fn ---
        if "axis" not in params:
            logger.warning(
                "Axis parameter not found in 'to_onnx' params, defaulting to 0."
            )
            axis = 0
        else:
            axis = params["axis"]

        # Filter out internal param before potentially using others
        onnx_params = {k: v for k, v in params.items() if k != "_original_fn"}
        # Log if other unexpected params exist
        if len(onnx_params) > 1:  # We only expect 'axis' besides '_original_fn'
            logger.warning(f"Unexpected params in to_onnx: {list(onnx_params.keys())}")

        input_names = [s.get_name(var) for var in node_inputs]
        output_name = s.get_name(node_outputs[0])

        # Get shape/dtype info from output aval (computed by abstract_eval)
        out_aval = node_outputs[0].aval
        output_shape = out_aval.shape
        dtype = out_aval.dtype
        logger.debug(f"  Output aval shape: {output_shape}, dtype: {dtype}")

        # Add ONNX node
        concat_node = helper.make_node(
            "Concat",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("concat"),
            axis=axis,  # Use axis retrieved from params
        )
        s.add_node(concat_node)

        # Add shape info to converter
        s.add_shape_info(output_name, tuple(output_shape), dtype)
        logger.debug(f"  Added shape info for {output_name}: {output_shape}")


# Register the abstract evaluation rule AFTER the class definition
jnp.concat_p.def_abstract_eval(ConcatPlugin.abstract_eval)
