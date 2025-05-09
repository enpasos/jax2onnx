# file: jax2onnx/plugins/jax/numpy/arange.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence, Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive, Literal, Var

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
        avals_to_inspect_for_dtype = list(in_avals)
        for aval_for_dtype in avals_to_inspect_for_dtype:
            val_to_check = (
                aval_for_dtype.val
                if isinstance(aval_for_dtype, Literal)
                else aval_for_dtype
            )
            if isinstance(val_to_check, float):
                is_float = True
                break
            if hasattr(val_to_check, "dtype") and jnp.issubdtype(
                val_to_check.dtype, np.floating
            ):
                is_float = True
                break
        final_dtype = np.dtype(np.float32 if is_float else np.int32)
        logger.debug(
            f"Arange abstract_eval: dtype from bind was None, inferred as {final_dtype}."
        )

    try:
        py_start, py_stop, py_step = 0.0, 0.0, 1.0
        if len(in_avals) == 1:
            py_stop = float(in_avals[0].val)
        elif len(in_avals) == 2:
            py_start = float(in_avals[0].val)
            py_stop = float(in_avals[1].val)
        elif len(in_avals) == 3:
            py_start = float(in_avals[0].val)
            py_stop = float(in_avals[1].val)
            py_step = float(in_avals[2].val)
        else:
            logger.error(
                f"Internal error: abstract_eval for arange received {len(in_avals)} avals. Defaulting to dynamic."
            )
            return core.ShapedArray((DATA_DEPENDENT_DYNAMIC_DIM,), final_dtype)

        if py_step == 0.0:
            logger.warning(
                "arange step is zero. JAX usually errors. Using dynamic sentinel for output shape."
            )
            return core.ShapedArray((DATA_DEPENDENT_DYNAMIC_DIM,), final_dtype)
        size = max(0, int(np.ceil((py_stop - py_start) / py_step)))
        logger.debug(
            f"Arange abstract_eval: concrete case, computed size={size}, dtype={final_dtype}"
        )
        return core.ShapedArray((size,), final_dtype)
    except (AttributeError, TypeError, ValueError):
        logger.debug(
            f"Arange abstract_eval: dynamic case (input not a numeric Literal, or calculation error), using sentinel, dtype={final_dtype}."
        )
        return core.ShapedArray((DATA_DEPENDENT_DYNAMIC_DIM,), final_dtype)


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
        {
            "testcase": "arange_stop_only_concrete",
            "callable": lambda stop: jnp.arange(stop, dtype=jnp.float32),
            "input_values": [np.array(5.0, dtype=np.float32)],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_start_stop_concrete",
            "callable": lambda start, stop: jnp.arange(start, stop, dtype=jnp.float32),
            "input_values": [
                np.array(2.0, dtype=np.float32),
                np.array(7.0, dtype=np.float32),
            ],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_start_stop_step_concrete",
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
            "testcase": "arange_start_stop_dynamic_via_tracers",
            "callable": lambda start, stop: jnp.arange(start, stop, dtype=jnp.float32),
            "input_values": [
                np.array(2.0, dtype=np.float32),
                np.array(7.0, dtype=np.float32),
            ],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
        {
            "testcase": "arange_float_concrete",
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
                    return ArangePlugin._ORIGINAL_ARANGE(
                        *args, dtype=dtype_param, **kwargs
                    )
                raise TypeError(
                    f"arange takes from 1 to 3 positional arguments but {num_pos_args} were given"
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
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ) -> None:
        output_var = node_outputs[0]
        output_aval = output_var.aval
        dtype_np = np.dtype(output_aval.dtype)
        output_name = s.get_name(output_var)

        output_shape_tuple_from_aval = output_aval.shape
        onnx_shape_representation: tuple[Any, ...] = output_shape_tuple_from_aval

        if output_shape_tuple_from_aval == (DATA_DEPENDENT_DYNAMIC_DIM,):
            logger.info(
                f"arange.to_onnx: Output '{output_name}' uses DATA_DEPENDENT_DYNAMIC_DIM. "
                f"ONNX dim_param will be '{str(DATA_DEPENDENT_DYNAMIC_DIM)}' (via JaxprConverter stringification)."
            )
        else:
            logger.debug(
                f"arange.to_onnx: Output shape for '{output_name}': {output_shape_tuple_from_aval}."
            )

        input_count = len(node_inputs)
        start_val_for_onnx, stop_val_for_onnx, step_val_for_onnx = None, None, None

        if input_count == 1:
            start_val_for_onnx = np.array(0, dtype=dtype_np)
            stop_val_for_onnx = node_inputs[0]
            step_val_for_onnx = np.array(1, dtype=dtype_np)
        elif input_count == 2:
            start_val_for_onnx = node_inputs[0]
            stop_val_for_onnx = node_inputs[1]
            step_val_for_onnx = np.array(1, dtype=dtype_np)
        elif input_count == 3:
            start_val_for_onnx = node_inputs[0]
            stop_val_for_onnx = node_inputs[1]
            step_val_for_onnx = node_inputs[2]

        start_name = (
            s.get_name(start_val_for_onnx)
            if isinstance(start_val_for_onnx, Var)
            else s.get_constant_name(start_val_for_onnx)
        )
        stop_name = (
            s.get_name(stop_val_for_onnx)
            if isinstance(stop_val_for_onnx, Var)
            else s.get_constant_name(stop_val_for_onnx)
        )
        step_name = (
            s.get_name(step_val_for_onnx)
            if isinstance(step_val_for_onnx, Var)
            else s.get_constant_name(step_val_for_onnx)
        )

        range_node = helper.make_node(
            "Range", inputs=[start_name, stop_name, step_name], outputs=[output_name]
        )
        s.add_node(range_node)
        s.add_shape_info(output_name, onnx_shape_representation, dtype_np)
        logger.debug(
            f"arange.to_onnx: add_shape_info for '{output_name}' with internal shape "
            f"{output_shape_tuple_from_aval} (passed as {onnx_shape_representation}), dtype {dtype_np}."
        )
