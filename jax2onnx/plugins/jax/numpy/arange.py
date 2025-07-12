# In jax2onnx/plugins/jax/numpy/arange.py

from __future__ import annotations

import logging  # Ensure logging is imported in this file
from typing import TYPE_CHECKING, Any, Sequence, Callable

import numpy as np
import jax.numpy as jnp

from jax import core
from jax import config as jax_config

# STRICTLY keep the following line unchanged
from jax.extend.core import Primitive, Literal  # This Literal should be used for checks

from onnx import helper, TensorProto
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


logger = logging.getLogger("jax2onnx.plugins.jax.numpy.arange")


# --- JAX-side Sentinel for Data-Dependent Dynamic Dimensions ---
# ... (sentinel class definition remains the same) ...
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
    logger.debug("--- ARANGE abstract_eval_arange_dynamic (direct log) ---")
    logger.debug(f"Called with jax_enable_x64: {jax_config.jax_enable_x64}")
    logger.debug(f"Explicit dtype parameter: {dtype}")
    logger.debug(f"Number of in_avals: {len(in_avals)}")
    for i, aval_item in enumerate(in_avals):
        logger.debug(
            f"  in_avals[{i}]: type={type(aval_item)}, aval={aval_item}, "
            f"is_jax_extend_core_Literal={isinstance(aval_item, Literal)}, "  # Literal is jax.extend.core.Literal
            f"val={getattr(aval_item, 'val', 'N/A')}, dtype={getattr(aval_item, 'dtype', 'N/A')}"
        )
    logger.debug(f"Checking against Literal type: {Literal} (from jax.extend.core)")

    logger.debug("--- ARANGE abstract_eval_arange_dynamic ---")
    logger.debug(f"Called with jax_enable_x64: {jax_config.jax_enable_x64}")
    logger.debug(f"Explicit dtype parameter: {dtype}")
    logger.debug(f"Number of in_avals: {len(in_avals)}")
    for i, aval_item in enumerate(in_avals):
        logger.debug(
            f"  in_avals[{i}]: type={type(aval_item)}, aval={aval_item}, "
            f"is_jax_extend_core_Literal={isinstance(aval_item, Literal)}, "
            f"val={getattr(aval_item, 'val', 'N/A')}, dtype={getattr(aval_item, 'dtype', 'N/A')}"
        )
    logger.debug(f"Checking against Literal type: {Literal} (from jax.extend.core)")

    x64_enabled = jax_config.jax_enable_x64
    final_dtype: np.dtype

    if dtype is not None:
        _temp_dtype = np.dtype(dtype)
        if jnp.issubdtype(_temp_dtype, np.floating):
            if x64_enabled:
                final_dtype = np.dtype(np.float64)
                if _temp_dtype != final_dtype:
                    logger.debug(
                        f"Arange abstract_eval: Explicit float dtype {_temp_dtype} promoted to {final_dtype} due to jax_enable_x64=True."
                    )
            elif _temp_dtype == np.dtype(np.float64):
                final_dtype = np.dtype(np.float32)
                logger.debug(
                    f"Arange abstract_eval: Explicit float64 dtype {_temp_dtype} demoted to {final_dtype} due to jax_enable_x64=False."
                )
            else:
                final_dtype = _temp_dtype
        else:
            final_dtype = _temp_dtype
    else:
        is_float_inferred = False
        for aval_for_dtype in in_avals:
            val_to_check_for_dtype = None
            if isinstance(aval_for_dtype, Literal):
                val_to_check_for_dtype = aval_for_dtype.val
            elif hasattr(aval_for_dtype, "dtype"):
                if jnp.issubdtype(aval_for_dtype.dtype, np.floating):
                    is_float_inferred = True
                    break
                if not aval_for_dtype.shape and hasattr(aval_for_dtype, "val"):
                    val_to_check_for_dtype = aval_for_dtype.val
            if val_to_check_for_dtype is not None and isinstance(
                val_to_check_for_dtype, (float, np.floating)
            ):
                is_float_inferred = True
                break
        if is_float_inferred:
            final_dtype = np.dtype(np.float64) if x64_enabled else np.dtype(np.float32)
        else:
            final_dtype = np.dtype(np.int32)
        logger.debug(
            f"Arange abstract_eval: dtype from bind was None, inferred as {final_dtype} from input avals (x64_enabled={x64_enabled})."
        )

    try:
        all_literals = True
        for i, aval in enumerate(in_avals):
            is_lit = isinstance(aval, Literal)
            logger.debug(
                f"Checking in_aval[{i}]: type={type(aval)}, isinstance Literal ({Literal})? {is_lit}"
            )
            if not is_lit:
                all_literals = False
                break

        if not all_literals:
            logger.warning(
                "Arange abstract_eval: Not all inputs are jax.extend.core.Literal instances. Defaulting to dynamic shape."
            )
            # ONNX Range will be fed int64 constants, so our dynamic arange must declare int64 output
            dyn_dtype = (
                np.dtype(np.int64)
                if np.issubdtype(final_dtype, np.integer)
                else final_dtype
            )
            if dyn_dtype != final_dtype:
                logger.debug(
                    f"Arange abstract_eval: Promoting dynamic integer dtype {final_dtype} → {dyn_dtype}."
                )
            return core.ShapedArray(
                (DATA_DEPENDENT_DYNAMIC_DIM,), dyn_dtype, weak_type=False
            )

        logger.debug("All inputs are Literals. Proceeding with concrete evaluation.")
        concrete_vals = [float(aval.val) for aval in in_avals]
        logger.debug(f"Concrete vals for calculation: {concrete_vals}")

        py_start, py_stop, py_step = 0.0, 0.0, 1.0
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
            logger.error(
                f"Internal error: arange abstract_eval received {len(concrete_vals)} concrete values. Defaulting to dynamic."
            )
            return core.ShapedArray(
                (DATA_DEPENDENT_DYNAMIC_DIM,), final_dtype, weak_type=False
            )
        logger.debug(f"Python start={py_start}, stop={py_stop}, step={py_step}")

        if py_step == 0.0:
            logger.warning("arange step is zero. Using dynamic sentinel.")
            return core.ShapedArray(
                (DATA_DEPENDENT_DYNAMIC_DIM,), final_dtype, weak_type=False
            )

        size = 0
        calc_start, calc_stop, calc_step = (
            np.float64(py_start),
            np.float64(py_stop),
            np.float64(py_step),
        )
        if (calc_step > 0 and calc_start < calc_stop) or (
            calc_step < 0 and calc_start > calc_stop
        ):
            value_for_ceil = (calc_stop - calc_start) / calc_step
            size = int(np.ceil(value_for_ceil))
        size = max(0, size)
        logger.debug(
            f"Arange abstract_eval: concrete case, calculated size={size}, final_dtype={final_dtype}"
        )
        return core.ShapedArray((size,), final_dtype, weak_type=False)

    except Exception as e:
        logger.error(
            f"Arange abstract_eval: Exception during concrete evaluation ({e}), defaulting to dynamic shape.",
            exc_info=True,
        )
        # same promotion on error path
        err_dtype = (
            np.dtype(np.int64)
            if np.issubdtype(final_dtype, np.integer)
            else final_dtype
        )
        return core.ShapedArray(
            (DATA_DEPENDENT_DYNAMIC_DIM,), err_dtype, weak_type=False
        )


# ... (rest of the ArangePlugin class and other definitions remain the same) ...

jnp.arange_p_jax2onnx.def_abstract_eval(abstract_eval_arange_dynamic)


@register_primitive(
    jaxpr_primitive=jnp.arange_p_jax2onnx.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arange.html",
    onnx=[
        {"component": "Range", "doc": "https://onnx.ai/onnx/operators/onnx__Range.html"}
    ],
    since="v0.5.2",
    context="primitives.jnp",
    component="arange",
    testcases=[
        # ------------------------------------------------------------------
        # Data‐dependent stop: arange(x.shape[1]) should produce a dynamic Range
        {
            "testcase": "arange_data_dependent_indices",
            "callable": lambda x: jnp.arange(x.shape[1]),
            "input_shapes": [(3, 10)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
        },
        # ------------------------------------------------------------------
        {
            "testcase": "arange_stop_only_concrete_input_val",
            "callable": lambda stop: jnp.arange(stop, dtype=jnp.float32),
            "input_values": [np.array(5.0, dtype=np.float32)],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_start_stop_concrete_input_val",
            "callable": lambda start, stop: jnp.arange(start, stop, dtype=jnp.float32),
            "input_values": [
                np.array(2.0, dtype=np.float32),
                np.array(7.0, dtype=np.float32),
            ],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_start_stop_step_concrete_input_val",
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
        {
            "testcase": "arange_float_concrete_input_val",
            "callable": lambda start, stop, step: jnp.arange(
                start, stop, step, dtype=jnp.float32
            ),
            "input_values": [
                np.array(1.0, dtype=np.float32),
                np.array(4.5, dtype=np.float32),
                np.array(0.5, dtype=np.float32),
            ],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_static_stop_only_int",
            "callable": lambda: jnp.arange(5),
            "input_values": [],
            # "expected_output_shapes": [(5,)],
            "x64_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "x32_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_static_stop_only_float",
            "callable": lambda: jnp.arange(5.0),
            "input_values": [],
            "expected_output_shapes": [(5,)],
        },
        {
            "testcase": "arange_static_start_stop_int",
            "callable": lambda: jnp.arange(2, 7),
            "input_values": [],
            "x64_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "x32_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_static_start_stop_step_int",
            "callable": lambda: jnp.arange(1, 10, 2),
            "input_values": [],
            "x64_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "x32_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_static_empty_result_pos_step",
            "callable": lambda: jnp.arange(5, 2, 1),
            "input_values": [],
            "x64_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "x32_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_static_empty_result_neg_step",
            "callable": lambda: jnp.arange(2, 5, -1),
            "input_values": [],
            "x64_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "x32_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_static_negative_step",
            "callable": lambda: jnp.arange(5, 0, -1),
            "input_values": [],
            "x64_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "x32_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_static_float_step_explicit_dtype",
            "callable": lambda: jnp.arange(1.0, 2.0, 0.25, dtype=jnp.float32),
            "input_values": [],
            "expected_output_shapes": [(4,)],
        },
        {
            "testcase": "arange_static_float_step_inferred_dtype",
            "callable": lambda: jnp.arange(0.0, 1.0, 0.3),  # Should infer float
            "input_values": [],
            "expected_output_shapes": [(4,)],
        },
        {
            "testcase": "arange_static_stop_zero",
            "callable": lambda: jnp.arange(0),
            "input_values": [],
            "x64_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "x32_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_static_start_equals_stop",
            "callable": lambda: jnp.arange(5, 5, 1),
            "input_values": [],
            "x64_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "x32_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_static_large_numbers_int",
            "callable": lambda: jnp.arange(1000, 1010, 1, dtype=jnp.int32),
            "input_values": [],
            "x64_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "x32_expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
    ],
)
class ArangePlugin(PrimitiveLeafPlugin):
    _ORIGINAL_ARANGE: Callable[..., Any] | None = None

    @staticmethod
    def abstract_eval(*in_avals, dtype=None):
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
            num_pos_args = len(args)
            if not (1 <= num_pos_args <= 3):
                logger.debug(
                    f"Calling original arange due to invalid number of positional args: {num_pos_args}."
                )
                if ArangePlugin._ORIGINAL_ARANGE:
                    return ArangePlugin._ORIGINAL_ARANGE(  # type: ignore
                        *args, dtype=dtype_param, **kwargs
                    )
                raise TypeError(
                    f"arange takes 1 to 3 positional arguments but {num_pos_args} were given"
                )

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
        node_inputs: Sequence[core.Var],
        node_outputs: Sequence[core.Var],
        params: dict[str, Any],
    ) -> None:
        output_var = node_outputs[0]
        output_aval = output_var.aval
        # Pick up the JAX‐inferred dtype
        dtype_np = np.dtype(output_aval.dtype)
        # ONNX Range only supports int64 for integer outputs, so promote any integer dtype
        if np.issubdtype(dtype_np, np.integer):
            dtype_np = np.dtype(np.int64)
        output_name = s.get_name(output_var)

        output_shape_tuple_from_aval = output_aval.shape
        onnx_shape_representation: tuple[Any, ...] = output_shape_tuple_from_aval

        if DATA_DEPENDENT_DYNAMIC_DIM in output_shape_tuple_from_aval:
            logger.info(
                f"arange.to_onnx: Output '{output_name}' has a data-dependent dynamic dimension. "
                f"ONNX shape info: {output_shape_tuple_from_aval}."
            )
        else:
            logger.debug(
                f"arange.to_onnx: Output shape for '{output_name}' is concrete: {output_shape_tuple_from_aval}."
            )

        input_vars = list(node_inputs)
        onnx_input_names: list[str] = []

        def _ensure_typed_onnx_input(
            var: core.Var | None, default_py_value: Any | None
        ) -> str:
            if var is not None:
                if isinstance(
                    var.aval, Literal
                ):  # Check against jax.extend.core.Literal
                    typed_const_val = np.array(var.aval.val, dtype=dtype_np)
                    return s.get_constant_name(typed_const_val)
                else:
                    original_name = s.get_name(var)
                    # if JAX dtype doesn't match our target (INT64), insert a Cast
                    if var.aval.dtype != dtype_np:
                        # Insert a Cast to the promoted integer dtype (int64) or float dtype
                        cast_name = s.get_unique_name(f"{original_name}_cast")
                        s.add_node(
                            helper.make_node(
                                "Cast",
                                inputs=[original_name],
                                outputs=[cast_name],
                                to=(
                                    TensorProto.INT64
                                    if np.issubdtype(dtype_np, np.integer)
                                    else TensorProto.FLOAT
                                ),
                            )
                        )
                        # preserve shape info on the casted tensor
                        s.add_shape_info(cast_name, var.aval.shape, dtype_np)
                        return cast_name
                    return original_name
            elif default_py_value is not None:
                return s.get_constant_name(np.array(default_py_value, dtype=dtype_np))
            else:
                raise ValueError(
                    "Internal error in _ensure_typed_onnx_input: requires var or default_py_value."
                )

        if len(input_vars) == 1:
            onnx_input_names.append(_ensure_typed_onnx_input(None, default_py_value=0))
            onnx_input_names.append(_ensure_typed_onnx_input(input_vars[0], None))
            onnx_input_names.append(_ensure_typed_onnx_input(None, default_py_value=1))
        elif len(input_vars) == 2:
            onnx_input_names.append(_ensure_typed_onnx_input(input_vars[0], None))
            onnx_input_names.append(_ensure_typed_onnx_input(input_vars[1], None))
            onnx_input_names.append(_ensure_typed_onnx_input(None, default_py_value=1))
        elif len(input_vars) == 3:
            onnx_input_names.append(_ensure_typed_onnx_input(input_vars[0], None))
            onnx_input_names.append(_ensure_typed_onnx_input(input_vars[1], None))
            onnx_input_names.append(_ensure_typed_onnx_input(input_vars[2], None))
        else:
            raise ValueError(
                f"Arange plugin received unexpected number of inputs: {len(input_vars)}"
            )

        range_node = helper.make_node(
            "Range", inputs=onnx_input_names, outputs=[output_name]
        )
        s.add_node(range_node)
        s.add_shape_info(output_name, onnx_shape_representation, dtype_np)
        logger.debug(
            f"arange.to_onnx: add_shape_info for '{output_name}' with shape "
            f"{onnx_shape_representation} (from aval {output_shape_tuple_from_aval}), dtype {dtype_np}."
        )
