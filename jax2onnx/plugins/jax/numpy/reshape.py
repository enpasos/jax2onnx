from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
from typing import TYPE_CHECKING, Tuple, List, Union, Sequence
from jax2onnx.plugin_system import register_primitive, PrimitiveLeafPlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

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
    def _process_newshape(newshape: Sequence[Union[int, str]]) -> List[Union[int, str]]:
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
        input_shape: Tuple[Union[int, str], ...], newshape: Sequence[Union[int, str]]
    ) -> Tuple[Union[int, str], ...]:
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
            for orig, dummy in zip(newshape, dummy_newshape)
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

        # Create a shape tensor for ONNX
        shape_tensor_name = s.get_unique_name("reshape_shape")
        onnx_shape = [dim if isinstance(dim, int) else -1 for dim in processed_newshape]
        # Create a NumPy array with the correct dtype (int64)
        onnx_shape_array = np.array(onnx_shape, dtype=np.int64)
        s.add_initializer(name=shape_tensor_name, vals=onnx_shape_array)

        reshape_node = helper.make_node(
            "Reshape",
            inputs=[input_name, shape_tensor_name],
            outputs=[output_name],
            name=s.get_unique_name("reshape"),
            allowzero=0,  # Explicit allowzero=0
        )
        s.add_node(reshape_node)
        s.add_shape_info(output_name, output_shape)

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
