# file: jax2onnx/plugins/jax/numpy/squeeze.py

from typing import TYPE_CHECKING

import jax
from jax import core
from jax import numpy as jnp
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.converter.dynamic_utils import encode_dims
from jax2onnx.converter.patched_callable_wrapper import PatchedCallableWrapper
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive


import logging

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.squeeze")

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# Define a new primitive for squeeze
jnp.squeeze_p = Primitive("jnp.squeeze")
jnp.squeeze_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jnp.squeeze_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.squeeze.html",
    onnx=[
        {
            "component": "Squeeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="squeeze",
    testcases=[
        {
            "testcase": "squeeze_single_dim",
            "callable": lambda a: jnp.squeeze(a, axis=0),
            "input_shapes": [(1, 49, 10)],
        },
        {
            "testcase": "squeeze_multiple_dims",
            "callable": lambda a: jnp.squeeze(a, axis=(0, 2)),
            "input_shapes": [(1, 49, 1, 10)],
        },
        {
            "testcase": "squeeze_vit_output",
            "callable": lambda a: jnp.squeeze(a, axis=1),
            "input_shapes": [(1, 1, 10)],
        },
        {
            "testcase": "squeeze_dynamic_batch",
            "callable": lambda a: jnp.squeeze(a, axis=1),
            "input_shapes": [("B", 1, 10)],
        },
        {
            "testcase": "squeeze_all_dims",
            "callable": lambda a: jnp.squeeze(a),
            "input_shapes": [(1, 1, 1)],
        },
        {
            "testcase": "squeeze_negative_axis",
            "callable": lambda a: jnp.squeeze(a, axis=-1),
            "input_shapes": [(1, 49, 1)],
        },
        {
            "testcase": "squeeze_negative_axis_tuple",
            "callable": lambda a: jnp.squeeze(a, axis=(-1, -3)),
            "input_shapes": [(1, 49, 1)],
        },
        {
            "testcase": "squeeze_dynamic_and_negative_axis",
            "callable": lambda a: jnp.squeeze(a, axis=(-1, -3)),
            "input_shapes": [(1, "B", 1)],
        },
    ],
)
class SqueezePlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.numpy.squeeze to ONNX.
    """

    _ORIGINAL_OP = None  # Will be filled by patch_info

    @staticmethod
    def abstract_eval(x, axes=None, *, axis=None):
        """
        Compute the output shape for squeeze using jax.eval_shape to handle symbolic dimensions.

        This delegates shape computation to JAX itself, ensuring correct behavior with symbolic shapes.

        Parameters:
            x: The input array to squeeze
            axes: The axes to squeeze (expected from _squeeze)
            axis: Alternative param name (expected from the patched_callable_wrapper)
        """
        # Handle either 'axes' or 'axis' parameter
        if axis is not None:
            if axes is not None:
                logger.warning(
                    "Both 'axes' and 'axis' provided to abstract_eval; using 'axis'"
                )
            axes = axis

        # 1. Sanity checks
        if not isinstance(x, core.ShapedArray):
            raise TypeError("expected ShapedArray input")

        # 2. Specs for eval_shape
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)

        # 3. Helper using the un-patched op
        orig = SqueezePlugin._ORIGINAL_OP
        if orig is None:
            logger.warning(
                "Original squeeze op not available, using fallback shape computation."
            )
            # Fallback if original op is not available
            shape = list(x.shape)
            if axes is None:
                new_shape = [d for d in shape if not (isinstance(d, int) and d == 1)]
            else:
                # Handle negative indices
                ndim = len(shape)
                normalized_axes = [(ax + ndim) if ax < 0 else ax for ax in axes]
                new_shape = [d for i, d in enumerate(shape) if i not in normalized_axes]
            return core.ShapedArray(tuple(new_shape), x.dtype)

        # 4. Convert axes to the form expected by the original op
        axis_arg = None
        if axes is not None:
            # Handle both single int and tuple/list cases
            if isinstance(axes, (tuple, list)):
                # If length is 1, pass as single int for better compatibility
                if len(axes) == 1:
                    axis_arg = int(axes[0])
                else:
                    # Otherwise keep as tuple
                    axis_arg = axes
            else:
                # Single integer case
                axis_arg = int(axes)

        def _helper(x):
            return orig(x, axis=axis_arg)

        # 5. Use JAX's eval_shape for correct symbolic dimension handling
        try:
            result = jax.eval_shape(_helper, spec)
            return core.ShapedArray(result.shape, result.dtype)
        except Exception as e:
            logger.error(f"Error in abstract_eval for squeeze: {e}")
            # Fallback if eval_shape fails
            shape = list(x.shape)
            if axes is None:
                new_shape = [d for d in shape if not (isinstance(d, int) and d == 1)]
            else:
                # Handle negative indices
                ndim = len(shape)
                normalized_axes = [(ax + ndim) if ax < 0 else ax for ax in axes]
                new_shape = [d for i, d in enumerate(shape) if i not in normalized_axes]
            return core.ShapedArray(tuple(new_shape), x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles ONNX conversion for jnp.squeeze."""

        # Handle either 'axes' or 'axis' parameter name
        if "axes" in params:
            axes = params["axes"]
        elif "axis" in params:
            # Convert the axis parameter to a tuple of axes
            axis_val = params["axis"]
            if isinstance(axis_val, (tuple, list)):
                axes = axis_val  # Already a tuple/list
            else:
                axes = (axis_val,)  # Make single value into a tuple
            logger.debug(f"Converting 'axis' parameter to 'axes': {axes}")
        else:
            # No axes specified, squeeze all size-1 dimensions
            axes = None

        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])

        # Use symbolic shape if available (e.g. batch dim "B")
        var = node_inputs[0]
        var_name = s.get_var_name(var)
        input_shape = s.symbolic_shapes.get(var_name, var.aval.shape)
        logger.debug(
            f"SqueezePlugin.to_onnx: input_name={input_name}, input_shape={input_shape}, axes={axes}"
        )

        # Normalize axes into positive indices; collect any symbolic axes
        normalized_axes = []
        symbolic_axes = []
        if axes is not None:
            for axis in axes:
                axis_val = axis if axis >= 0 else axis + len(input_shape)
                if 0 <= axis_val < len(input_shape):
                    if isinstance(input_shape[axis_val], int):
                        normalized_axes.append(axis_val)
                    else:
                        symbolic_axes.append(axis_val)

        # Special case: if all dimensions are size-1, perform a direct reshape to scalar
        if all(isinstance(dim, int) and dim == 1 for dim in input_shape):
            # Direct reshape to scalar is more reliable than squeeze for this case
            shape_name = s.get_unique_name("scalar_shape")
            s.builder.add_initializer(name=shape_name, vals=[], dims=[0])
            reshape_node = helper.make_node(
                "Reshape",
                inputs=[input_name, shape_name],
                outputs=[output_name],
                name=s.get_unique_name("reshape_to_scalar"),
            )
            s.add_node(reshape_node)
            s.add_shape_info(output_name, ())
            return

        # Identify which of those are actual size-1 dims
        static_axes = [i for i in normalized_axes if input_shape[i] == 1]

        # If user specified axes but none are size-1 at trace-time, just identity
        if axes is not None and not static_axes:
            identity = helper.make_node(
                "Identity",
                inputs=[input_name],
                outputs=[output_name],
                name=s.get_unique_name("identity"),
            )
            s.add_node(identity)
            s.add_shape_info(output_name, tuple(input_shape))
            return

        # If there are static axes, supply them as an initializer
        if static_axes:
            axes_name = s.get_unique_name("squeeze_axes")
            s.builder.add_initializer(
                name=axes_name, vals=encode_dims(static_axes), dims=[len(static_axes)]
            )
            squeeze_inputs = [input_name, axes_name]
            output_shape = tuple(
                dim for i, dim in enumerate(input_shape) if i not in static_axes
            )
        else:
            # No axes specified, squeeze all size-1 dimensions with ONNX Squeeze
            squeeze_inputs = [input_name]
            output_shape = tuple(
                dim for dim in input_shape if not (isinstance(dim, int) and dim == 1)
            )
            # If we end up with an empty shape after removing all size-1 dims,
            # ONNX can't represent a true scalar, so default to a 1D tensor
            if not output_shape:
                output_shape = (1,)

        squeeze_node = helper.make_node(
            "Squeeze",
            inputs=squeeze_inputs,
            outputs=[output_name],
            name=s.get_unique_name("squeeze"),
        )
        s.add_node(squeeze_node)
        s.add_shape_info(output_name, output_shape)

    @staticmethod
    def _squeeze(a, axis: int | tuple[int, ...] | None = None):
        """Defines the primitive binding for Squeeze."""
        if axis is None:
            axes = tuple(
                i for i, dim in enumerate(a.shape) if isinstance(dim, int) and dim == 1
            )
        elif isinstance(axis, int):
            axes = (axis,)
        else:
            axes = tuple(axis)
        return jnp.squeeze_p.bind(a, axes=axes)

    @staticmethod
    def patch_info():
        """Provides patching information for Squeeze."""

        def _creator(orig_fn):
            SqueezePlugin._ORIGINAL_OP = orig_fn
            return PatchedCallableWrapper(orig_fn, jnp.squeeze_p)

        return {
            "patch_targets": [jnp],
            "target_attribute": "squeeze",
            "patch_function": _creator,
        }


# Register abstract evaluation function
jnp.squeeze_p.def_abstract_eval(SqueezePlugin.abstract_eval)
