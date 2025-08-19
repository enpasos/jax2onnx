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
            f"is_jax_extend_core_Literal={isinstance(aval_item, Literal)}, "
            f"val={getattr(aval_item, 'val', 'N/A')}, dtype={getattr(aval_item, 'dtype', 'N/A')}"
        )
    logger.debug(f"Checking against Literal type: {Literal} (from jax.extend.core)")

    x64_enabled = jax_config.jax_enable_x64
    final_dtype: np.dtype

    # Determine dtype as before...
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
        # Infer dtype from avals as before...
        is_float_inferred = False
        for aval in in_avals:
            val_to_check = None
            if isinstance(aval, Literal):
                val_to_check = aval.val
            elif hasattr(aval, "dtype") and jnp.issubdtype(aval.dtype, np.floating):
                is_float_inferred = True
            elif not aval.shape and hasattr(aval, "val"):
                val_to_check = aval.val
            if isinstance(val_to_check, (float, np.floating)):
                is_float_inferred = True
                break
        if is_float_inferred:
            final_dtype = np.dtype(np.float64) if x64_enabled else np.dtype(np.float32)
        else:
            final_dtype = np.dtype(np.int32)
        logger.debug(
            f"Arange abstract_eval: dtype inferred as {final_dtype} from input avals (x64={x64_enabled})."
        )

    try:
        # --- New: accept any concrete scalar aval ---
        concrete_vals: list[float] = []
        all_concrete = True
        for i, aval in enumerate(in_avals):
            # 1) Literal from either core or extend.core
            if isinstance(aval, (core.Literal, Literal)):
                concrete_vals.append(float(aval.val))
                continue

            # 2) Any scalar-shaped aval with a .val attribute
            if not aval.shape and hasattr(aval, "val"):
                v = aval.val
                if np.isscalar(v):
                    concrete_vals.append(float(v))
                    continue

            all_concrete = False
            logger.debug(
                f"in_aval[{i}] of type {type(aval)} is not concrete; falling back to dynamic."
            )
            break

        if not all_concrete:
            logger.warning(
                "Arange abstract_eval: inputs not all concrete; defaulting to dynamic shape."
            )
            dyn_dtype = (
                np.dtype(np.int64)
                if np.issubdtype(final_dtype, np.integer)
                else final_dtype
            )
            return core.ShapedArray(
                (DATA_DEPENDENT_DYNAMIC_DIM,), dyn_dtype, weak_type=False
            )

        logger.debug("All inputs are concrete. Proceeding with concrete evaluation.")
        # Determine size by numpy semantics
        start = concrete_vals[0] if len(concrete_vals) > 1 else 0.0
        if len(concrete_vals) == 1:
            stop = concrete_vals[0]
            step = 1.0
        elif len(concrete_vals) == 2:
            stop = concrete_vals[1]
            step = 1.0
        else:
            stop = concrete_vals[1]
            step = concrete_vals[2]

        if step == 0.0:
            return core.ShapedArray(
                (DATA_DEPENDENT_DYNAMIC_DIM,), final_dtype, weak_type=False
            )

        # Compute size
        num = max(0, int(np.ceil((stop - start) / step)))
        return core.ShapedArray((num,), final_dtype, weak_type=False)

    except Exception:
        # Fallback to dynamic on any error
        err_dtype = (
            np.dtype(np.int64)
            if np.issubdtype(final_dtype, np.integer)
            else final_dtype
        )
        return core.ShapedArray(
            (DATA_DEPENDENT_DYNAMIC_DIM,), err_dtype, weak_type=False
        )


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
                    f"jnp.arange patched call received unexpected kwargs: {kwargs}."
                )
            num_pos = len(args)
            if not (1 <= num_pos <= 3):
                if ArangePlugin._ORIGINAL_ARANGE:
                    return ArangePlugin._ORIGINAL_ARANGE(
                        *args, dtype=dtype_param, **kwargs
                    )
                raise TypeError(
                    f"arange takes 1 to 3 positional arguments but {num_pos} were given"
                )
            return jnp.arange_p_jax2onnx.bind(*args[:num_pos], dtype=dtype_param)

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
