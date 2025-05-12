# file: jax2onnx/plugins/jax/numpy/arange.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence, Callable

import numpy as np
import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive, Literal, Var  # type: ignore

from onnx import helper
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.arange")


# --- JAX-side Sentinel for Data-Dependent Dynamic Dimensions ---
class Jax2OnnxDynamicDimSentinel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Jax2OnnxDynamicDimSentinel, cls).__new__(cls)
        return cls._instance

    def __repr__(self):
        return "JAX2ONNX_DYNAMIC_DIM_SENTINEL"

    def dimension_as_value(self):
        logger.error("Jax2OnnxDynamicDimSentinel.dimension_as_value() called.")
        raise TypeError(
            "Jax2OnnxDynamicDimSentinel cannot be converted to a concrete dimension value."
        )

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return isinstance(other, Jax2OnnxDynamicDimSentinel)


DATA_DEPENDENT_DYNAMIC_DIM = Jax2OnnxDynamicDimSentinel()
# --- End Sentinel Definition ---

if not hasattr(jnp, "arange_p_jax2onnx"):
    jnp.arange_p_jax2onnx = Primitive("jnp.arange_jax2onnx")
    jnp.arange_p_jax2onnx.multiple_results = False
else:
    jnp.arange_p_jax2onnx = getattr(jnp, "arange_p_jax2onnx")


def abstract_eval_arange_dynamic(*in_avals: core.AbstractValue, dtype: Any = None):
    # 1. Determine final_dtype
    if dtype is not None:
        final_dtype = np.dtype(dtype)
    else:
        is_float = False
        # Inspect input avals for dtype inference if dtype is not explicitly provided
        # This mimics JAX's behavior: jnp.arange(5) -> int; jnp.arange(5.0) -> float
        # jnp.arange(0, 5.0) -> float
        for aval_for_dtype in in_avals:
            # Use .val if Literal, otherwise try to access .dtype if ShapedArray (less common for arange inputs)
            val_to_check_for_dtype = None
            if isinstance(aval_for_dtype, Literal):
                val_to_check_for_dtype = aval_for_dtype.val
            elif hasattr(aval_for_dtype, "dtype"):  # E.g. if a tracer has a fixed dtype
                # If any input implies float, the output should be float
                if jnp.issubdtype(aval_for_dtype.dtype, np.floating):
                    is_float = True
                    break
                # If it's a 0-dim array, we might get its value
                if not aval_for_dtype.shape and hasattr(
                    aval_for_dtype, "val"
                ):  # Check for concrete val in 0-d array avals
                    val_to_check_for_dtype = aval_for_dtype.val

            if val_to_check_for_dtype is not None:
                if isinstance(val_to_check_for_dtype, (float, np.floating)):
                    is_float = True
                    break
        # JAX default dtypes: integers default to int32/int64, floats to float32/float64
        # For arange, if no dtype and no float inputs, it's typically int.
        # If any input is float, or dtype implies float, then float.
        # We'll default to int32/float32 here for simplicity if not specified.
        final_dtype = np.dtype(np.float32 if is_float else np.int32)
        logger.debug(
            f"Arange abstract_eval: dtype from bind was None, inferred as {final_dtype} from input avals."
        )

    try:
        # Attempt to extract concrete values from Literals
        concrete_vals = [
            float(aval.val) for aval in in_avals if isinstance(aval, Literal)
        ]

        # Check if all inputs were concrete and convertible to float
        if len(concrete_vals) != len(in_avals):
            logger.debug(
                "Arange abstract_eval: Not all inputs are concrete Literals. Defaulting to dynamic shape."
            )
            return core.ShapedArray(
                (DATA_DEPENDENT_DYNAMIC_DIM,), final_dtype, weak_type=False
            )

        py_start, py_stop, py_step = 0.0, 0.0, 1.0  # Default values
        if len(concrete_vals) == 1:
            py_stop = concrete_vals[0]
        elif len(concrete_vals) == 2:
            py_start = concrete_vals[0]
            py_stop = concrete_vals[1]
        elif len(concrete_vals) == 3:
            py_start = concrete_vals[0]
            py_stop = concrete_vals[1]
            py_step = concrete_vals[2]
        else:
            # This case should ideally be caught by the patcher's arg count check.
            logger.error(
                f"Internal error: abstract_eval for arange received {len(concrete_vals)} concrete vals from {len(in_avals)} avals. Defaulting to dynamic."
            )
            return core.ShapedArray(
                (DATA_DEPENDENT_DYNAMIC_DIM,), final_dtype, weak_type=False
            )

        if py_step == 0.0:
            # JAX raises TypeError: jax.numpy.arange: parameter step cannot be zero.
            # We'll return a dynamic shape as ONNX Range with step=0 might be problematic or undefined.
            logger.warning(
                "arange step is zero. JAX usually errors. Using dynamic sentinel for output shape."
            )
            return core.ShapedArray(
                (DATA_DEPENDENT_DYNAMIC_DIM,), final_dtype, weak_type=False
            )

        # Calculate size using NumPy's logic for arange length
        # size = np.arange(py_start, py_stop, py_step).size # This is safer
        # Or, the formula: ceil((stop - start) / step) for positive step
        # and ceil((start - stop) / (-step)) for negative step, ensuring max(0, result)
        # For floating point, precision can be an issue with direct formula.
        # np.ceil can handle this.
        if (py_step > 0 and py_start >= py_stop) or (
            py_step < 0 and py_start <= py_stop
        ):
            size = 0
        else:
            size = int(np.ceil((py_stop - py_start) / py_step))
            # Correction for floating point issues where ceil might over-count slightly
            # if last element is exactly stop.
            # e.g. arange(0, 3, 1.5) -> [0, 1.5]. stop-start/step = 3/1.5 = 2. ceil(2)=2.
            # e.g. arange(0, 2, 0.5) -> [0, 0.5, 1.0, 1.5]. stop-start/step = 2/0.5 = 4. ceil(4)=4.
            # The direct formula can sometimes be off by one for exact multiples due to floating point.
            # A more robust way for concrete values if precision is a concern:
            # temp_arange = np.arange(py_start, py_stop, step=py_step, dtype=float) # Use float for calculation robustness
            # size = temp_arange.size
            # This is what JAX itself does internally for concrete values.
            # However, for abstract eval, we might not want to instantiate a full array.
            # The formula max(0, ceil((stop-start)/step)) is standard for positive steps.
            # Let's stick to the standard formula carefully.
            if py_step == 0:  # Should be caught earlier
                size = 0  # Or raise, JAX errors.
            elif (py_stop > py_start and py_step < 0) or (
                py_stop < py_start and py_step > 0
            ):
                size = 0
            else:  # step direction aligns with start/stop
                # np.longdouble for precision in calculation
                size = int(
                    np.ceil(
                        (np.longdouble(py_stop) - np.longdouble(py_start))
                        / np.longdouble(py_step)
                    )
                )

        size = max(0, size)

        logger.debug(
            f"Arange abstract_eval: concrete case, computed size={size} for start={py_start}, stop={py_stop}, step={py_step}. Final dtype={final_dtype}"
        )
        return core.ShapedArray((size,), final_dtype, weak_type=False)

    except (AttributeError, TypeError, ValueError) as e:
        logger.debug(
            f"Arange abstract_eval: dynamic case due to error ({e}), or input not a numeric Literal, or calculation error. Using sentinel. Dtype={final_dtype}."
        )
        return core.ShapedArray(
            (DATA_DEPENDENT_DYNAMIC_DIM,), final_dtype, weak_type=False
        )


jnp.arange_p_jax2onnx.def_abstract_eval(abstract_eval_arange_dynamic)


@register_primitive(
    jaxpr_primitive=jnp.arange_p_jax2onnx.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arange.html",
    onnx=[
        {"component": "Range", "doc": "https://onnx.ai/onnx/operators/onnx__Range.html"}
    ],
    since="v0.5.2",  # Ensure this is accurate or update if version changed
    context="primitives.jnp",
    component="arange",
    testcases=[
        # Existing dynamic test cases (expecting DATA_DEPENDENT_DYNAMIC_DIM)
        {
            "testcase": "arange_stop_only_concrete_input_val",  # Renamed for clarity
            "callable": lambda stop: jnp.arange(stop, dtype=jnp.float32),
            "input_values": [
                np.array(5.0, dtype=np.float32)
            ],  # Input is a JAX array tracer
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_start_stop_concrete_input_val",  # Renamed for clarity
            "callable": lambda start, stop: jnp.arange(start, stop, dtype=jnp.float32),
            "input_values": [
                np.array(2.0, dtype=np.float32),
                np.array(7.0, dtype=np.float32),
            ],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_start_stop_step_concrete_input_val",  # Renamed for clarity
            "callable": lambda start, stop, step: jnp.arange(
                start, stop, step, dtype=jnp.float32
            ),
            "input_values": [
                np.array(1.0, dtype=np.float32),
                np.array(7.0, dtype=np.float32),
                np.array(2.0, dtype=np.float32),
            ],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        # This test seems redundant if input_values make them tracers anyway
        # {
        # "testcase": "arange_start_stop_dynamic_via_tracers",
        # "callable": lambda start, stop: jnp.arange(start, stop, dtype=jnp.float32),
        # "input_values": [
        # np.array(2.0, dtype=np.float32), # These become tracers
        # np.array(7.0, dtype=np.float32),
        # ],
        # "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        # },
        {
            "testcase": "arange_float_concrete_input_val",  # Renamed for clarity
            "callable": lambda start, stop, step: jnp.arange(
                start, stop, step, dtype=jnp.float32  # Explicit dtype
            ),
            "input_values": [
                np.array(1.0, dtype=np.float32),
                np.array(4.5, dtype=np.float32),
                np.array(0.5, dtype=np.float32),
            ],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        # --- New Static Test Cases ---
        # Inputs are Python literals directly in the lambda.
        # `input_values` should be empty for these.
        {
            "testcase": "arange_static_stop_only_int",
            "callable": lambda: jnp.arange(
                5
            ),  # dtype will be inferred as int32 by abstract_eval
            "input_values": [],
            "expected_output_shapes": [(5,)],  # np.arange(5) -> [0,1,2,3,4]
        },
        {
            "testcase": "arange_static_stop_only_float",
            "callable": lambda: jnp.arange(5.0),  # dtype will be inferred as float32
            "input_values": [],
            "expected_output_shapes": [(5,)],  # np.arange(5.0) -> [0.,1.,2.,3.,4.]
        },
        {
            "testcase": "arange_static_start_stop_int",
            "callable": lambda: jnp.arange(2, 7),  # dtype int32
            "input_values": [],
            "expected_output_shapes": [(5,)],  # np.arange(2,7) -> [2,3,4,5,6]
        },
        {
            "testcase": "arange_static_start_stop_step_int",
            "callable": lambda: jnp.arange(1, 10, 2),  # dtype int32
            "input_values": [],
            "expected_output_shapes": [(5,)],  # np.arange(1,10,2) -> [1,3,5,7,9]
        },
        {
            "testcase": "arange_static_empty_result_pos_step",
            "callable": lambda: jnp.arange(5, 2, 1),  # dtype int32
            "input_values": [],
            "expected_output_shapes": [(0,)],
        },
        {
            "testcase": "arange_static_empty_result_neg_step",
            "callable": lambda: jnp.arange(2, 5, -1),  # dtype int32
            "input_values": [],
            "expected_output_shapes": [(0,)],
        },
        {
            "testcase": "arange_static_negative_step",
            "callable": lambda: jnp.arange(5, 0, -1),  # dtype int32
            "input_values": [],
            "expected_output_shapes": [(5,)],  # np.arange(5,0,-1) -> [5,4,3,2,1]
        },
        {
            "testcase": "arange_static_float_step_explicit_dtype",
            "callable": lambda: jnp.arange(1.0, 2.0, 0.25, dtype=jnp.float32),
            "input_values": [],
            "expected_output_shapes": [(4,)],  # [1.0, 1.25, 1.5, 1.75]
        },
        {
            "testcase": "arange_static_float_step_inferred_dtype",
            "callable": lambda: jnp.arange(0.0, 1.0, 0.3),  # dtype float32 inferred
            "input_values": [],
            "expected_output_shapes": [(4,)],  # [0.0, 0.3, 0.6, 0.9]
        },
        {
            "testcase": "arange_static_stop_zero",
            "callable": lambda: jnp.arange(0),  # dtype int32
            "input_values": [],
            "expected_output_shapes": [(0,)],
        },
        {
            "testcase": "arange_static_start_equals_stop",
            "callable": lambda: jnp.arange(5, 5, 1),  # dtype int32
            "input_values": [],
            "expected_output_shapes": [(0,)],
        },
        {
            "testcase": "arange_static_large_numbers_int",
            "callable": lambda: jnp.arange(1000, 1010, 1, dtype=jnp.int32),
            "input_values": [],
            "expected_output_shapes": [(10,)],
        },
    ],
)
class ArangePlugin(PrimitiveLeafPlugin):
    _ORIGINAL_ARANGE: Callable[..., Any] | None = None

    @staticmethod
    def abstract_eval(*in_avals, dtype=None):
        # Ensure this matches the primitive's abstract_eval signature
        return jnp.arange_p_jax2onnx.abstract_eval(*in_avals, dtype=dtype)

    @staticmethod
    def get_monkey_patch(orig_fn: Callable[..., Any]):
        ArangePlugin._ORIGINAL_ARANGE = orig_fn

        def patched_arange(*args, **kwargs):
            dtype_param = kwargs.pop("dtype", None)
            if kwargs:
                logger.warning(
                    f"jnp.arange patched call received unexpected kwargs: {kwargs}. "
                    "These will be ignored by the primitive binding but passed to original if fallback occurs."
                )
            # jnp.arange can take 1, 2, or 3 positional arguments:
            # arange(stop)
            # arange(start, stop)
            # arange(start, stop, step)
            num_pos_args = len(args)
            if not (1 <= num_pos_args <= 3):
                logger.debug(
                    f"Calling original arange due to invalid number of positional args: {num_pos_args}."
                )
                if ArangePlugin._ORIGINAL_ARANGE:
                    return ArangePlugin._ORIGINAL_ARANGE(
                        *args, dtype=dtype_param, **kwargs  # Pass original kwargs back
                    )
                # JAX itself would raise a TypeError here.
                raise TypeError(
                    f"arange takes 1 to 3 positional arguments but {num_pos_args} were given"
                )

            # Bind the arguments to the custom primitive
            # The arguments (args) could be Python scalars, JAX arrays, or tracers.
            bind_args = args[:num_pos_args]
            return jnp.arange_p_jax2onnx.bind(*bind_args, dtype=dtype_param)

        return patched_arange

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jnp],
            "target_attribute": "arange",
            "patch_function": ArangePlugin.get_monkey_patch,
        }

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Var],  # These are jax.core.Var instances from the jaxpr
        node_outputs: Sequence[Var],
        params: dict[str, Any],  # Contains 'dtype' from bind
    ) -> None:
        output_var = node_outputs[0]
        output_aval = output_var.aval
        # The dtype of the output is determined by abstract_eval (which considers the 'dtype' param)
        dtype_np = np.dtype(output_aval.dtype)
        output_name = s.get_name(output_var)

        # The shape of the output is also from abstract_eval
        output_shape_tuple_from_aval = output_aval.shape
        # This shape can be concrete (e.g., (5,)) or dynamic (e.g., (DATA_DEPENDENT_DYNAMIC_DIM,))
        onnx_shape_representation: tuple[Any, ...] = output_shape_tuple_from_aval

        if DATA_DEPENDENT_DYNAMIC_DIM in output_shape_tuple_from_aval:
            logger.info(
                f"arange.to_onnx: Output '{output_name}' has a data-dependent dynamic dimension. "
                f"ONNX shape info: {output_shape_tuple_from_aval}."
            )
            # The JaxprConverter's add_shape_info will handle stringifying the sentinel if necessary
        else:
            logger.debug(
                f"arange.to_onnx: Output shape for '{output_name}' is concrete: {output_shape_tuple_from_aval}."
            )

        input_count = len(node_inputs)  # Number of start/stop/step vars in jaxpr
        onnx_input_names: list[str] = []

        # Default ONNX Range inputs if fewer than 3 are provided to jnp.arange
        # These defaults must match the types expected by ONNX Range (often same as output)
        default_start_val = np.array(0, dtype=dtype_np)
        default_step_val = np.array(1, dtype=dtype_np)

        if input_count == 1:  # arange(stop)
            # ONNX Range: start, limit, delta
            onnx_input_names.append(s.get_constant_name(default_start_val))  # start = 0
            onnx_input_names.append(s.get_name(node_inputs[0]))  # limit = stop
            onnx_input_names.append(s.get_constant_name(default_step_val))  # delta = 1
        elif input_count == 2:  # arange(start, stop)
            onnx_input_names.append(s.get_name(node_inputs[0]))  # start
            onnx_input_names.append(s.get_name(node_inputs[1]))  # limit = stop
            onnx_input_names.append(s.get_constant_name(default_step_val))  # delta = 1
        elif input_count == 3:  # arange(start, stop, step)
            onnx_input_names.append(s.get_name(node_inputs[0]))  # start
            onnx_input_names.append(s.get_name(node_inputs[1]))  # limit = stop
            onnx_input_names.append(s.get_name(node_inputs[2]))  # delta = step
        else:
            # This should not be reached if patcher and abstract_eval are correct.
            raise ValueError(
                f"Arange plugin received unexpected number of inputs: {input_count}"
            )

        range_node = helper.make_node(
            "Range", inputs=onnx_input_names, outputs=[output_name]
        )
        s.add_node(range_node)
        # add_shape_info uses the shape determined by abstract_eval
        s.add_shape_info(output_name, onnx_shape_representation, dtype_np)
        logger.debug(
            f"arange.to_onnx: add_shape_info for '{output_name}' with shape "
            f"{onnx_shape_representation} (from aval {output_shape_tuple_from_aval}), dtype {dtype_np}."
        )
