"""
This module contains the ONNX plugin for the `jax.numpy.stack` function,
following the standard jax2onnx pattern for `jnp` functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# ---------------------------------------------------------------------- #
# 1.  A dedicated primitive for jnp.stack                                #
# ---------------------------------------------------------------------- #
jnp_stack_p = Primitive("jnp.stack")
jnp_stack_p.multiple_results = False


# ---------------------------------------------------------------------- #
# 2.  Plugin registration                                                #
# ---------------------------------------------------------------------- #
@register_primitive(
    primitive_obj=jnp_stack_p,
    binding_factory=lambda: jnp.stack,
    jaxpr_primitive=jnp_stack_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.stack.html",
    onnx=[
        {
            "component": "Unsqueeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Unsqueeze.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="v0.7.1",
    context="primitives.jnp",
    component="stack",
    testcases=[
        {
            "testcase": "stack_axis_0",
            "callable": lambda *args: jnp.stack(args, axis=0),
            "input_values": [
                np.array([1, 2], dtype=np.float32),
                np.array([3, 4], dtype=np.float32),
            ],
            "expected_output_shapes": [(2, 2)],
        },
        {
            "testcase": "stack_axis_1",
            "callable": lambda *args: jnp.stack(args, axis=1),
            "input_values": [
                np.array([[1, 2], [3, 4]], dtype=np.float32),
                np.array([[5, 6], [7, 8]], dtype=np.float32),
            ],
            "expected_output_shapes": [(2, 2, 2)],
        },
        {
            "testcase": "stack_negative_axis",
            "callable": lambda *args: jnp.stack(args, axis=-1),
            "input_values": [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([4, 5, 6], dtype=np.int32),
            ],
            "expected_output_shapes": [(3, 2)],
        },
        {
            "testcase": "stack_scalars",
            "callable": lambda *args: jnp.stack(args, axis=0),
            "input_values": [
                np.array(10, dtype=np.float32),
                np.array(20, dtype=np.float32),
            ],
            "expected_output_shapes": [(2,)],
        },
    ],
)
class StackPlugin(PrimitiveLeafPlugin):
    """ONNX plugin for jnp.stack."""

    _ORIG_CALL: Callable[..., Any] | None = None
    primitive = jnp_stack_p

    @staticmethod
    def abstract_eval(*in_avals: core.ShapedArray, axis: int, **_):
        """Abstract evaluation implementation for the jnp.stack primitive."""
        if not in_avals:
            raise ValueError("must supply arrays to stack")

        shape = in_avals[0].shape
        for aval in in_avals[1:]:
            if aval.shape != shape:
                msg = "all input arrays must have the same shape. Got {} and {}"
                raise ValueError(msg.format(shape, aval.shape))

        # Manually normalize the axis relative to the *output* rank.
        output_rank = len(shape) + 1
        if axis < 0:
            axis += output_rank

        output_shape = list(shape)
        output_shape.insert(axis, len(in_avals))

        output_dtype = in_avals[0].dtype
        return core.ShapedArray(tuple(output_shape), output_dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[jax.core.Var],
        node_outputs: Sequence[jax.core.Var],
        params: Dict[str, Any],
    ):
        """Converts the JAX `stack` primitive to ONNX `Unsqueeze` and `Concat` ops."""
        axis = params["axis"]

        unsqueezed_inputs = []
        for var in node_inputs:
            input_name = s.get_name(var)
            unsqueezed_name = s.get_unique_name(f"{input_name}_unsqueezed")

            # ONNX Unsqueeze uses the axis value directly.
            axes_val = np.array([axis], dtype=np.int64)
            axes_const = s.get_constant_name(axes_val)

            node = helper.make_node(
                "Unsqueeze",
                inputs=[input_name, axes_const],
                outputs=[unsqueezed_name],
                name=s.get_unique_name("Unsqueeze"),
            )
            s.add_node(node)

            # Add shape info for the newly created intermediate tensor
            input_shape = list(var.aval.shape)
            # Manually normalize axis to correctly calculate the unsqueezed shape
            rank = len(input_shape) + 1
            unsqueezed_axis = axis if axis >= 0 else axis + rank
            input_shape.insert(unsqueezed_axis, 1)
            s.add_shape_info(unsqueezed_name, tuple(input_shape), var.aval.dtype)

            unsqueezed_inputs.append(unsqueezed_name)

        output_name = s.get_name(node_outputs[0])

        # Normalize axis for Concat to be non-negative
        concat_rank = len(node_outputs[0].aval.shape)
        concat_axis = axis if axis >= 0 else axis + concat_rank

        node = helper.make_node(
            "Concat",
            inputs=unsqueezed_inputs,
            outputs=[output_name],
            name=s.get_unique_name("Concat"),
            axis=concat_axis,
        )
        s.add_node(node)

    @staticmethod
    def _stack_binding(arrays, axis=0):
        """Binds the inputs to the custom jnp.stack primitive."""
        if not isinstance(arrays, (list, tuple)):
            raise TypeError(f"stack expects a sequence of arrays, got {type(arrays)}")

        flat_arrays, _ = jax.tree_util.tree_flatten(arrays)
        return jnp_stack_p.bind(*flat_arrays, axis=axis)

    @staticmethod
    def get_monkey_patch(orig_fn: Callable):
        """Returns the patched version of jnp.stack."""
        StackPlugin._ORIG_CALL = orig_fn

        def patched_stack(arrays, axis=0):
            return StackPlugin._stack_binding(arrays, axis=axis)

        return patched_stack

    @staticmethod
    def patch_info():
        """Provides patching information to the plugin system."""
        return {
            "patch_targets": [jnp],
            "target_attribute": "stack",
            "patch_function": StackPlugin.get_monkey_patch,
        }


jnp_stack_p.def_abstract_eval(StackPlugin.abstract_eval)
