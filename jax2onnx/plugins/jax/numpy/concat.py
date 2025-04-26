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

    # In file: jax2onnx/plugins/jax/numpy/concat.py
    # Inside class ConcatPlugin:

    @staticmethod
    def abstract_eval(*avals, **params):  # Keep **params to receive axis
        """
        Performs abstract evaluation manually, assuming input dimension objects
        (in aval.shape) support required arithmetic and comparison.
        Replicates expected JAX internal logic for concatenate.
        """
        import jax
        import jax.numpy as jnp
        from jax import core
        import logging
        import numpy as np

        logger = logging.getLogger("jax2onnx.plugins.jax.numpy.concat")

        # --- Extract axis from params ---
        # (Assumes PatchedCallableWrapper is still in use for now to pass params)
        if "axis" not in params:
            raise ValueError("'axis' parameter not found in abstract_eval params.")
        axis = params["axis"]

        logger.debug(
            "ConcatPlugin: Entering abstract_eval (Manual Calculation - Final)"
        )
        logger.debug(f"Received axis: {axis} (type: {type(axis)})")
        logger.debug(f"Received *avals tuple (length {len(avals)}): {avals}")

        # --- Input Validation and Aval Extraction ---
        if not avals:
            raise ValueError(
                "abstract_eval received no AbstractValue arguments in *avals."
            )
        if not all(isinstance(a, core.AbstractValue) for a in avals):
            logger.error(
                f"Received non-AbstractValue arguments: {[type(a) for a in avals]}"
            )
            raise TypeError("Inputs to abstract_eval must be AbstractValue instances.")

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
        if not hasattr(first_aval, "ndim") or not hasattr(first_aval, "shape"):
            raise TypeError(f"Input aval {first_aval} does not have ndim or shape.")
        rank = first_aval.ndim

        # Verify ranks match
        for i, aval in enumerate(avals[1:], 1):
            if not hasattr(aval, "ndim") or aval.ndim != rank:
                all_shapes_repr = [repr(getattr(a, "shape", "N/A")) for a in avals]
                raise TypeError(
                    f"Concatenate inputs must have same rank ({rank}). Got shapes {all_shapes_repr} at index {i}"
                )

        # Ensure axis is a valid integer
        if isinstance(axis, core.Tracer):  # Should not happen if from params
            raise TypeError("Axis for concatenate cannot be a tracer.")
        if not isinstance(axis, int):
            try:
                axis = int(axis)
            except (ValueError, TypeError):
                raise TypeError(f"Axis must be an integer, got {type(axis)}")
        if not -rank <= axis < rank:
            raise ValueError(f"Axis {axis} out of range for rank {rank}")
        axis = axis % rank

        # Helper to get dimension object from shape
        def get_dim(aval, idx):
            try:
                # Assume aval.shape contains int or operable symbolic objects (DimVar/_DimExpr)
                return aval.shape[idx]
            except IndexError as e:
                raise ValueError(
                    f"Cannot access dimension {idx} in shape {aval.shape}"
                ) from e
            except TypeError as e:
                logger.error(
                    f"TypeError accessing aval.shape[{idx}] for aval {aval}. Shape: {getattr(aval, 'shape', 'N/A')}, Error: {e}"
                )
                raise TypeError(f"Cannot index shape of aval {aval}") from e

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
                            # Attempt addition - relies on JAX DimVar/_DimExpr supporting '+'
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
                        # Attempt comparison - relies on JAX DimVar/_DimExpr supporting '=='
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
            f"Manual calculation - Final output shape tuple: {final_output_shape_tuple}"
        )

        try:
            # This tuple should contain integers and JAX's hashable symbolic objects (DimVar/DimExpr)
            result_aval = core.ShapedArray(final_output_shape_tuple, output_dtype)
            logger.debug(f"--- abstract_eval END (Success) ---")
            return result_aval
        except TypeError as e:
            # If this still fails (e.g., unhashable DimExpr), JAX internal handling is complex.
            logger.error(
                f"TypeError creating ShapedArray with computed shape tuple: {final_output_shape_tuple}. Error: {e}",
                exc_info=True,
            )
            logger.debug(f"--- abstract_eval END (ShapedArray Creation Failed) ---")
            raise

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
