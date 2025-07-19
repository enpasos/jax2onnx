"""
This module contains the ONNX plugin for the `jax.numpy.unstack` function,
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
# 1.  A dedicated primitive for jnp.unstack                              #
# ---------------------------------------------------------------------- #
jnp_unstack_p = Primitive("jnp.unstack")
jnp_unstack_p.multiple_results = True


# ---------------------------------------------------------------------- #
# 2.  Plugin registration                                                #
# ---------------------------------------------------------------------- #
@register_primitive(
    primitive_obj=jnp_unstack_p,
    binding_factory=lambda: jnp.unstack,
    jaxpr_primitive=jnp_unstack_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.unstack.html",
    onnx=[
        {
            "component": "Split",
            "doc": "https://onnx.ai/onnx/operators/onnx__Split.html",
        },
        {
            "component": "Squeeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
        },
    ],
    since="v0.7.1",
    context="primitives.jnp",
    component="unstack",
    testcases=[
        {
            "testcase": "unstack_axis_0",
            "callable": lambda x: jnp.unstack(x, axis=0),
            "input_values": [np.array([[1, 2], [3, 4]], dtype=np.float32)],
            "expected_output_shapes": [(2,), (2,)],
        },
        {
            "testcase": "unstack_axis_1",
            "callable": lambda x: jnp.unstack(x, axis=1),
            "input_values": [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)],
            "expected_output_shapes": [(2,), (2,), (2,)],
        },
        {
            "testcase": "unstack_negative_axis",
            "callable": lambda x: jnp.unstack(x, axis=-1),
            "input_values": [np.array([[[1, 2], [3, 4]]], dtype=np.float32)],
            "expected_output_shapes": [(1, 2), (1, 2)],
        },
    ],
)
class UnstackPlugin(PrimitiveLeafPlugin):
    """ONNX plugin for jnp.unstack."""

    _ORIG_CALL: Callable[..., Any] | None = None
    primitive = jnp_unstack_p

    @staticmethod
    def abstract_eval(x: core.ShapedArray, *, axis: int, **_):
        """
        Produce the result *avals* for `jnp.unstack`.

        The output is a tuple of `core.ShapedArray`s – **not** concrete
        `ShapeDtypeStruct`s – so that JAX’s tracing machinery can manipulate
        them without ever touching real values.
        """
        rank = len(x.shape)
        axis = axis if axis >= 0 else axis + rank
        if not 0 <= axis < rank:
            raise ValueError(f"axis {axis} out of bounds for rank-{rank} tensor")

        size = x.shape[axis]
        if not isinstance(size, (int, np.integer)):
            # `unstack` needs the length along `axis` at trace-time.
            raise core.InconclusiveDimensionOperation(
                "jnp.unstack requires a statically known dimension length."
            )

        out_shape = x.shape[:axis] + x.shape[axis + 1 :]
        return tuple(core.ShapedArray(out_shape, x.dtype) for _ in range(int(size)))

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[jax.core.Var],
        node_outputs: Sequence[jax.core.Var],
        params: Dict[str, Any],
    ):
        """Converts the JAX `unstack` primitive to ONNX `Split` and `Squeeze` ops."""
        axis = params["axis"]
        input_aval = node_inputs[0].aval

        # Infer the number of outputs from the input shape
        rank = len(input_aval.shape)
        norm_axis = axis if axis >= 0 else axis + rank
        num = input_aval.shape[norm_axis]

        input_name = s.get_name(node_inputs[0])

        # 1. Split the tensor into `num` chunks along the specified axis.
        split_output_names = [s.get_unique_name(f"split_chunk_{i}") for i in range(num)]
        splits_const = s.get_constant_name(np.array([1] * num, dtype=np.int64))

        split_node = helper.make_node(
            "Split",
            inputs=[input_name, splits_const],
            outputs=split_output_names,
            name=s.get_unique_name("Split"),
            axis=axis,
        )
        s.add_node(split_node)

        # 2. Squeeze each chunk to remove the singleton dimension.
        axes_to_squeeze = s.get_constant_name(np.array([axis], dtype=np.int64))

        for i in range(num):
            split_chunk_name = split_output_names[i]
            final_output_name = s.get_name(node_outputs[i])

            # Add shape info for the intermediate split tensor
            original_shape = list(input_aval.shape)
            original_shape[norm_axis] = 1
            s.add_shape_info(split_chunk_name, tuple(original_shape), input_aval.dtype)

            squeeze_node = helper.make_node(
                "Squeeze",
                inputs=[split_chunk_name, axes_to_squeeze],
                outputs=[final_output_name],
                name=s.get_unique_name("Squeeze"),
            )
            s.add_node(squeeze_node)

    @staticmethod
    def _unstack_binding(x, axis=0):
        """Binds the inputs to the custom jnp.unstack primitive."""
        return jnp_unstack_p.bind(x, axis=axis)

    @staticmethod
    def get_monkey_patch(orig_fn: Callable):
        """Returns the patched version of jnp.unstack."""
        UnstackPlugin._ORIG_CALL = orig_fn

        def patched_unstack(x, axis=0):
            return UnstackPlugin._unstack_binding(x, axis=axis)

        return patched_unstack

    @staticmethod
    def patch_info():
        """Provides patching information to the plugin system."""
        return {
            "patch_targets": [jnp],
            "target_attribute": "unstack",
            "patch_function": UnstackPlugin.get_monkey_patch,
        }


jnp_unstack_p.def_abstract_eval(UnstackPlugin.abstract_eval)
