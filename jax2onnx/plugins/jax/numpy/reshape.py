from collections.abc import Sequence
from typing import TYPE_CHECKING

from jax import core
from jax import numpy as jnp
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

import numpy as np

# Define the reshape primitive
jnp.reshape_p = Primitive("jnp.reshape")
jnp.reshape_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=jnp.reshape_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.reshape.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="reshape",
    testcases=[
        {
            "testcase": "reshape_1",
            "callable": lambda a: jnp.reshape(a, (2, 6)),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "reshape_2",
            "callable": lambda a: jnp.reshape(a, (-1, 2)),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "reshape_3",
            "callable": lambda a: jnp.reshape(a, (2, -1)),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "reshape_4",
            "callable": lambda a: jnp.reshape(a, (-1, 4)),
            "input_shapes": [("B", 3, 4)],
        },
        {
            "testcase": "reshape_to_scalar",
            "callable": lambda a: jnp.reshape(a, ()),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "reshape_from_scalar",
            "callable": lambda a: jnp.reshape(a, (1,)),
            "input_shapes": [()],
        },
    ],
)
class ReshapePlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.numpy.reshape to ONNX.
    """

    @staticmethod
    def _process_newshape(newshape: Sequence[int | str]) -> list[int | str]:
        """Validates and processes the newshape argument for reshape."""
        if isinstance(newshape, (int, str)):
            newshape = [newshape]
        else:
            newshape = list(newshape)

        neg_one_count = sum(1 for dim in newshape if dim == -1)
        if neg_one_count > 1:
            raise ValueError("Only one dimension can be -1 (inferred).")

        return newshape

    @staticmethod
    def _get_dynamic_output_shape(
        input_shape: tuple[int | str, ...], newshape: Sequence[int | str]
    ) -> tuple[int | str, ...]:
        """Computes the output shape for jnp.reshape while handling dynamic dimensions."""
        newshape = ReshapePlugin._process_newshape(newshape)
        input_shape_list = list(input_shape)

        dummy_input_shape = [1 if isinstance(s, str) else s for s in input_shape_list]
        dummy_newshape = [1 if isinstance(s, str) else s for s in newshape]

        if -1 in dummy_newshape:
            neg_one_index = dummy_newshape.index(-1)
            known_dims_product = np.prod([dim for dim in dummy_newshape if dim != -1])
            # Avoid ZeroDivisionError
            if known_dims_product == 0 and np.prod(dummy_input_shape) != 0:
                raise ValueError(
                    f"Cannot reshape array of shape {input_shape} into shape {newshape}"
                )
            inferred_dim = (
                int(np.prod(dummy_input_shape) / known_dims_product)
                if known_dims_product != 0
                else 0
            )
            dummy_newshape[neg_one_index] = inferred_dim

        if np.prod(dummy_input_shape) != np.prod(dummy_newshape):
            raise ValueError(
                f"Cannot reshape array of shape {input_shape} into shape {newshape}"
            )

        output_shape = [
            orig if isinstance(orig, str) else dummy
            for orig, dummy in zip(newshape, dummy_newshape, strict=False)
        ]
        return tuple(output_shape)

    @staticmethod
    def abstract_eval(a, newshape):
        """Abstract evaluation function for Reshape."""
        newshape_processed = ReshapePlugin._process_newshape(newshape)
        output_shape = ReshapePlugin._get_dynamic_output_shape(
            a.shape, newshape_processed
        )
        return core.ShapedArray(tuple(output_shape), a.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of Reshape to ONNX format."""
        newshape = params["newshape"]
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])

        input_shape = node_inputs[0].aval.shape
        output_shape = ReshapePlugin._get_dynamic_output_shape(input_shape, newshape)
        processed_newshape = ReshapePlugin._process_newshape(newshape)

        shape_tensor_name = s.builder.get_constant_name(
            np.array(
                [dim if isinstance(dim, int) else -1 for dim in processed_newshape],
                dtype=np.int64,
            )
        )

        reshape_node = helper.make_node(
            "Reshape",
            inputs=[input_name, shape_tensor_name],
            outputs=[output_name],
            name=s.get_unique_name("reshape"),
            allowzero=0,  # Explicit allowzero=0
        )
        s.add_node(reshape_node)

        # Ensure shape contains only integers by replacing dynamic dimensions with -1
        sanitized_output_shape = tuple(
            dim if isinstance(dim, int) else -1 for dim in output_shape
        )
        s.builder.add_value_info(
            output_name, shape=sanitized_output_shape, dtype=node_inputs[0].aval.dtype
        )

    @staticmethod
    def _reshape(a, newshape, order="C"):
        """Defines the primitive binding for Reshape."""
        if order != "C":
            raise NotImplementedError("Only C-style reshape is supported.")
        return jnp.reshape_p.bind(a, newshape=newshape)

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for Reshape."""

        def patched_reshape(a, newshape, order="C"):
            return ReshapePlugin._reshape(a, newshape, order)

        return patched_reshape

    @staticmethod
    def patch_info():
        """Provides patching information for Reshape."""
        return {
            "patch_targets": [jnp],
            "patch_function": lambda _: ReshapePlugin.get_monkey_patch(),
            "target_attribute": "reshape",
        }


# Register abstract evaluation function
jnp.reshape_p.def_abstract_eval(ReshapePlugin.abstract_eval)
