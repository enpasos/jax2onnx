# file: jax2onnx/plugins/jax/numpy/linspace.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence, Callable

import numpy as np
import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive  # Changed from jax.core for Primitive
from onnx import helper  # Removed TensorProto as it's part of helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.linspace")


jnp.linspace_p = Primitive("jnp.linspace")
jnp.linspace_p.multiple_results = False


def _abstract_eval_linspace_static(
    *, static_start, static_stop, static_num, endpoint, dtype, axis
):
    """
    Abstract evaluation function for the static linspace primitive.
    All arguments are expected to be passed as keyword arguments from `bind` (params).
    """
    if not (isinstance(static_num, int) and not isinstance(static_num, bool)):
        raise ValueError(
            f"For static linspace, 'static_num' (value: {static_num}, type: {type(static_num)}) "
            "must be a concrete Python integer."
        )
    if static_num < 0:
        raise ValueError("Number of samples, `static_num`, must be non-negative.")

    # JAX's jnp.linspace has an axis parameter. This static plugin creates a 1D constant.
    # If axis is not 0, the original JAX function might behave differently (e.g. inserting
    # the 1D array into a higher-dimensional one). We restrict this static plugin.
    if axis != 0:
        # This error will be caught during jax.make_jaxpr tracing.
        raise NotImplementedError(
            f"Static linspace plugin currently only supports axis=0. Got axis={axis}."
        )

    output_dtype_np: np.dtype
    if dtype is not None:
        output_dtype_np = np.dtype(dtype)
    else:
        # Determine output dtype based on JAX's behavior for jnp.linspace:
        # - If start/stop are floats, result is float (promoted by np.result_type).
        # - If start/stop are ints, JAX defaults to a float type (e.g., float32 or float64 depending on config).
        is_start_float = isinstance(static_start, (float, np.floating))
        is_stop_float = isinstance(static_stop, (float, np.floating))

        if is_start_float or is_stop_float:
            # np.result_type handles promotion like (float32, int32) -> float32
            # or (float64, float32) -> float64
            # or (float32, float32) -> float32
            intermediate_dtype = np.result_type(static_start, static_stop)
            # Ensure it's a JAX default float if it ended up integer (e.g. np.array(1, dtype=object))
            if not jnp.issubdtype(intermediate_dtype, np.floating):
                output_dtype_np = np.dtype(jnp.float32)  # Default promotion
            else:
                output_dtype_np = intermediate_dtype

        else:  # Both start and stop are integer types (Python int or np.integer)
            # JAX's linspace promotes to float if dtype is not specified.
            # Default to float32, common for JAX unless x64 is enabled (then float64).
            output_dtype_np = np.dtype(jnp.float32)

    # JAX linspace behavior for num=0: returns an empty array.
    # JAX linspace behavior for num=1: returns array([start]).
    # Shape is (static_num,)
    return core.ShapedArray((static_num,), output_dtype_np, weak_type=False)


jnp.linspace_p.def_abstract_eval(_abstract_eval_linspace_static)


@register_primitive(
    jaxpr_primitive=jnp.linspace_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linspace.html (Static Version for jax2onnx)",
    onnx=[
        {
            "component": "Constant",
            "doc": "https://onnx.ai/onnx/operators/onnx__Constant.html",
        },
    ],
    since="v0.5.2",
    context="primitives.jnp",
    component="linspace",
    testcases=[
        {
            "testcase": "linspace_static_basic",
            "callable": lambda: jnp.linspace(0.0, 10.0, num=5, dtype=jnp.float32),
            "input_values": [],
            "expected_output_shapes": [(5,)],  # Output dtype will be float32
        },
        {
            "testcase": "linspace_static_endpoint_false",
            "callable": lambda: jnp.linspace(
                0, 4, num=4, endpoint=False, dtype=jnp.int32
            ),
            "input_values": [],
            "expected_output_shapes": [(4,)],  # Output dtype will be int32
        },
        {
            "testcase": "linspace_static_num_1",
            "callable": lambda: jnp.linspace(3.0, 10.0, num=1, dtype=jnp.float64),
            "input_values": [],
            "expected_output_shapes": [(1,)],  # Output dtype will be float64
        },
        {
            "testcase": "linspace_static_num_0",
            # dtype=None, inputs are float and int, result_type will be float, abstract eval default float32
            "callable": lambda: jnp.linspace(0.0, 10.0, num=0),
            "input_values": [],
            "expected_output_shapes": [(0,)],
        },
        {  # Test with integer inputs and no dtype specified (should become float32)
            "testcase": "linspace_static_int_inputs_default_dtype",
            "callable": lambda: jnp.linspace(0, 10, num=5),
            "input_values": [],
            "expected_output_shapes": [(5,)],  # Expect float32 from abstract_eval logic
        },
    ],
)
class LinspacePlugin(PrimitiveLeafPlugin):
    _ORIGINAL_LINSPACE: Callable[..., Any] | None = None

    @staticmethod
    def abstract_eval(  # For plugin system, matches primitive's abstract_eval params
        *, static_start, static_stop, static_num, endpoint, dtype, axis
    ):
        return jnp.linspace_p.abstract_eval(
            static_start=static_start,
            static_stop=static_stop,
            static_num=static_num,
            endpoint=endpoint,
            dtype=dtype,
            axis=axis,
        )

    @staticmethod
    def get_monkey_patch(orig_fn: Callable[..., Any]):
        LinspacePlugin._ORIGINAL_LINSPACE = orig_fn

        def patched_linspace_static_impl(
            start, stop, num=50, *, endpoint=True, dtype=None, axis=0
        ):
            # This static plugin creates a 1D constant.
            # JAX's jnp.linspace itself has an axis parameter, which could imply embedding.
            # We simplify and restrict this static plugin to axis=0.
            if axis != 0:
                logger.warning(
                    f"LinspaceStaticPlugin received axis={axis}, but only supports axis=0. "
                    "Falling back to original JAX linspace if available."
                )
                if LinspacePlugin._ORIGINAL_LINSPACE:
                    return LinspacePlugin._ORIGINAL_LINSPACE(
                        start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis
                    )
                # If no fallback, error will be raised by abstract_eval during tracing.
                # Or we can raise it here directly.
                raise NotImplementedError(
                    f"Static linspace plugin currently only supports axis=0. Got axis={axis}, and fallback is unavailable."
                )

            is_static_start = isinstance(start, (int, float, np.number))
            is_static_stop = isinstance(stop, (int, float, np.number))
            # 'num' must be a Python int for the static version (not a JAX tracer or np.ndarray).
            # Exclude bools as they are instances of int but not valid for 'num'.
            is_static_num = isinstance(num, int) and not isinstance(num, bool)

            if is_static_start and is_static_stop and is_static_num:
                # All key inputs are Python constants.
                # Bind them as static keyword parameters to the primitive.
                return jnp.linspace_p.bind(
                    static_start=start,
                    static_stop=stop,
                    static_num=num,
                    endpoint=endpoint,
                    dtype=dtype,
                    axis=axis,  # Pass axis along; abstract_eval will validate it.
                )
            else:
                logger.warning(
                    "LinspaceStaticPlugin expects static Python numbers for start/stop, and a Python int for num. "
                    f"Received types: start={type(start)}, stop={type(stop)}, num={type(num)}. "
                    "Falling back to original JAX linspace if available."
                )
                if LinspacePlugin._ORIGINAL_LINSPACE:
                    return LinspacePlugin._ORIGINAL_LINSPACE(
                        start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis
                    )
                raise ValueError(
                    "Static linspace plugin requires static inputs (Python numbers for start/stop, Python int for num), "
                    "but received dynamic JAX types or non-integer num. Fallback to original JAX linspace is unavailable."
                )

        return patched_linspace_static_impl

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jnp],
            "target_attribute": "linspace",
            "patch_function": LinspacePlugin.get_monkey_patch,
        }

    def to_onnx(
        self,
        s: Jaxpr2OnnxConverter,
        node_inputs: Sequence[core.Var],  # Should be empty due to static bind
        node_outputs: Sequence[core.Var],
        params: dict[str, Any],  # All static values are in params
    ) -> None:
        if node_inputs:
            raise ValueError(
                "LinspaceStaticPlugin's to_onnx expects no node_inputs for its "
                "current static implementation, as all arguments should be in params."
            )

        output_var = node_outputs[0]
        output_name = s.get_name(output_var)

        # Retrieve static values from params, consistent with abstract_eval and bind
        start_val = params["static_start"]
        stop_val = params["static_stop"]
        num_val = params["static_num"]
        endpoint = params["endpoint"]
        # axis = params["axis"] # Validated to be 0 by abstract_eval / patcher

        # The dtype for np.linspace and the ONNX tensor should be the one
        # determined by abstract_eval. This is available in output_var.aval.dtype.
        dtype_np = output_var.aval.dtype
        if dtype_np is None:  # Should not happen if abstract_eval is correct
            raise ValueError("Output variable dtype is None after abstract evaluation.")

        # Precompute the linspace values using NumPy
        linspace_values: np.ndarray
        if num_val == 0:
            linspace_values = np.array([], dtype=dtype_np)
        elif num_val == 1:
            # For num=1, JAX's jnp.linspace(start, stop, 1) consistently returns [start].
            # This holds true for both endpoint=True and endpoint=False.
            linspace_values = np.array([start_val], dtype=dtype_np)
        else:
            linspace_values = np.linspace(
                start_val, stop_val, num_val, endpoint=endpoint, dtype=dtype_np
            )

        # Create an ONNX Constant node with the precomputed values
        onnx_tensor = helper.make_tensor(
            name=s.get_unique_name(
                f"{output_name}_value"
            ),  # Ensure unique tensor name within the node
            data_type=s._ensure_onnx_dtype(dtype_np),
            dims=linspace_values.shape,
            # Ensure vals are Python built-in types (float, int) for make_tensor
            vals=linspace_values.flatten().tolist(),
        )

        constant_node = helper.make_node(
            "Constant",
            inputs=[],  # No inputs for a constant node
            outputs=[output_name],
            value=onnx_tensor,  # The tensor proto is passed as 'value' attribute
            name=s.get_unique_name(
                f"linspace_static_const_{output_name}"
            ),  # Unique node name
        )
        s.add_node(constant_node)
        # s.add_shape_info is usually not needed here as the converter infers from output_var.aval
