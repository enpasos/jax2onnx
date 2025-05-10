from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence, Callable, Union

import numpy as np
import jax
import jax.numpy as jnp
from jax import core
from jax.core import ShapedArray
from jax.extend.core import Primitive, Literal, Var
from onnx import helper, TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
    from jax2onnx.converter.onnx_builder import OnnxBuilder

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.linspace")


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
        raise TypeError(
            "Jax2OnnxDynamicDimSentinel cannot be converted to a concrete dimension value."
        )

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return isinstance(other, Jax2OnnxDynamicDimSentinel)


DATA_DEPENDENT_DYNAMIC_DIM = Jax2OnnxDynamicDimSentinel()


# --- Abstract Evaluation for Linspace ---
def abstract_eval_linspace(start_aval, stop_aval, num_aval, *, endpoint, dtype, axis):
    if dtype is not None:
        final_dtype = np.dtype(dtype)
    else:
        final_dtype = np.float32  # Default dtype if none is specified

    if axis != 0:
        raise NotImplementedError("linspace with axis != 0 is not supported.")

    if isinstance(num_aval, Literal) and isinstance(num_aval.val, int):
        num_val = num_aval.val
        if num_val < 0:
            raise ValueError("linspace num must be non-negative.")
        return core.ShapedArray((num_val,), final_dtype)
    else:
        return core.ShapedArray((DATA_DEPENDENT_DYNAMIC_DIM,), final_dtype)


# Registering the JAX primitive for linspace
if not hasattr(jnp, "linspace_p_jax2onnx"):
    jnp.linspace_p_jax2onnx = Primitive("jnp.linspace_jax2onnx")
    jnp.linspace_p_jax2onnx.multiple_results = False

jnp.linspace_p_jax2onnx.def_abstract_eval(abstract_eval_linspace)


@register_primitive(
    jaxpr_primitive=jnp.linspace_p_jax2onnx.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linspace.html",
    onnx=[{"component": "Range"}, {"component": "Add"}],
    since="0.5.2",
    context="primitives.jnp",
    component="linspace",
    testcases=[  # Test cases adjusted from your last version
        {
            "testcase": "linspace_basic_concrete_num",
            "callable": lambda start, stop: jnp.linspace(
                start, stop, num=5, dtype=jnp.float32
            ),
            "input_values": [
                np.array(0.0, dtype=jnp.float32),
                np.array(10.0, dtype=jnp.float32),
            ],
            "expected_output_shapes": [(5,)],
        },
        {
            "testcase": "linspace_endpoint_false_concrete_num",
            "callable": lambda start, stop: jnp.linspace(
                start, stop, num=5, endpoint=False, dtype=jnp.float32
            ),
            "input_values": [
                np.array(0.0, dtype=jnp.float32),
                np.array(10.0, dtype=jnp.float32),
            ],
            "expected_output_shapes": [(5,)],
        },
        {
            "testcase": "linspace_num_1_concrete",
            "callable": lambda start, stop: jnp.linspace(
                start, stop, num=1, dtype=jnp.float32
            ),
            "input_values": [
                np.array(3.0, dtype=jnp.float32),
                np.array(10.0, dtype=jnp.float32),
            ],
            "expected_output_shapes": [(1,)],
        },
        {
            "testcase": "linspace_num_0_concrete",
            "callable": lambda start, stop: jnp.linspace(
                start, stop, num=0, dtype=jnp.float32
            ),
            "input_values": [
                np.array(0.0, dtype=jnp.float32),
                np.array(10.0, dtype=jnp.float32),
            ],
            "expected_output_shapes": [(0,)],
        },
        {
            "testcase": "linspace_dynamic_num",
            "callable": lambda start, stop, n: jnp.linspace(
                start, stop, num=n, dtype=jnp.float32
            ),
            # Assuming your test infrastructure/to_onnx can handle this input_shapes format
            # to correctly trace `n` with int32 dtype.
            "input_values": [
                np.array(0.0, dtype=jnp.float32),
                np.array(1.0, dtype=jnp.float32),
                np.array(7, dtype=np.int32),
            ],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
        },
    ],
)
class LinspacePlugin(PrimitiveLeafPlugin):

    @staticmethod
    def abstract_eval(*args, **kwargs):
        return jnp.linspace_p_jax2onnx.abstract_eval(*args, **kwargs)

    @staticmethod
    def get_monkey_patch(orig_fn: Callable[..., Any]):
        def patched_linspace(
            start, stop, num=50, *, endpoint=True, dtype=None, axis=0, retstep=False
        ):
            if retstep:
                raise NotImplementedError(
                    "jnp.linspace with retstep=True is not supported."
                )
            if axis != 0:
                return orig_fn(
                    start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis
                )
            return jnp.linspace_p_jax2onnx.bind(
                start, stop, num, endpoint=endpoint, dtype=dtype, axis=axis
            )

        return patched_linspace

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jnp],
            "target_attribute": "linspace",
            "patch_function": LinspacePlugin.get_monkey_patch,
        }

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ) -> None:
        start_var, stop_var, num_var = node_inputs
        output_var = node_outputs[0]

        final_dtype_np = np.dtype(output_var.aval.dtype)
        final_onnx_dtype_enum = s._ensure_onnx_dtype(final_dtype_np)
        output_name = s.get_name(output_var)

        # Convert inputs to the correct dtype for ONNX
        start_name = s.get_name(start_var)
        stop_name = s.get_name(stop_var)
        num_name = s.get_name(num_var)

        # Create the Range node (simulating linspace behavior)
        range_node_name = s.get_unique_name("linspace_range")
        s.add_node(
            helper.make_node(
                "Range",
                inputs=[start_name, stop_name, num_name],
                outputs=[range_node_name],
            )
        )
        s.add_shape_info(range_node_name, output_var.aval.shape, final_dtype_np)

        # Final cast to ensure the correct dtype
        s.add_node(
            helper.make_node(
                "Cast",
                inputs=[range_node_name],
                outputs=[output_name],
                to=final_onnx_dtype_enum,
            )
        )
        s.add_shape_info(output_name, output_var.aval.shape, final_dtype_np)
