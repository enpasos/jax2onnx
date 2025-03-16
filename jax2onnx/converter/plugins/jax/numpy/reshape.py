from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
import contextlib
from typing import TYPE_CHECKING, Tuple, List, Union, Sequence

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

import numpy as np

# Define a new primitive for reshape.
jnp.reshape_p = Primitive("jnp.reshape")


def get_primitive():
    return jnp.reshape_p


def _process_newshape(newshape: Sequence[Union[int, str]]) -> List[Union[int, str]]:
    """Processes and validates the newshape argument of reshape.

    - Converts the input sequence to a list.
    - Ensures that at most one dimension is -1 (inferred dimension).
    - Allows string values for dynamic dimensions.
    - Raises ValueError for invalid shapes.
    """
    if isinstance(newshape, (int, str)):
        newshape = [newshape]
    else:
        newshape = list(newshape)  # Convert to list for modification

    neg_one_count = 0
    for dim in newshape:
        if isinstance(dim, int):
            if dim == -1:
                neg_one_count += 1
            elif dim < 0:
                raise ValueError("Invalid shape dimension: {}".format(dim))
        elif not isinstance(dim, str):
            raise ValueError("Invalid shape dimension: {}".format(dim))

    if neg_one_count > 1:
        raise ValueError("Only one dimension can be -1 (inferred).")

    return newshape


def _get_dynamic_output_shape(
    input_shape: Tuple[Union[int, str], ...], newshape: Sequence[Union[int, str]]
) -> Tuple[Union[int, str], ...]:
    """Calculates the output shape of jnp.reshape, handling dynamic dimensions."""
    newshape = _process_newshape(newshape)
    input_shape_list = list(input_shape)  # Convert tuple to list

    # Replace dynamic dimensions in input_shape with 1 for calculation.
    dummy_input_shape = [1 if isinstance(s, str) else s for s in input_shape_list]
    dummy_newshape = [1 if isinstance(s, str) else s for s in newshape]

    # Find the index of -1 in newshape, if it exists.
    neg_one_index = -1
    try:
        neg_one_index = dummy_newshape.index(-1)
    except ValueError:
        pass

    # Calculate the size of the inferred dimension if -1 is present.
    if neg_one_index != -1:
        known_dims_product = 1
        for i, dim in enumerate(dummy_newshape):
            if i != neg_one_index:
                known_dims_product *= dim
        if known_dims_product == 0:
            raise ValueError("Cannot infer shape with zero-sized dimensions.")
        inferred_dim = int(np.prod(dummy_input_shape) / known_dims_product)
        dummy_newshape[neg_one_index] = inferred_dim

    if np.prod(dummy_input_shape) != np.prod(dummy_newshape):
        raise ValueError(
            f"Cannot reshape array of shape {input_shape} into shape {newshape}"
        )

    # Build output shape, substituting dynamic dimensions back in.
    output_shape: List[Union[int, str]] = []
    for orig, dummy in zip(newshape, dummy_newshape):
        if isinstance(orig, str):
            output_shape.append(orig)
        else:
            output_shape.append(dummy)
    return tuple(output_shape)


def _concretize_shape(
    shape: Tuple[Union[int, str], ...], concrete_value: int = 2
) -> Tuple[int, ...]:
    """Replaces any dynamic marker (string) in a shape with a concrete value."""
    return tuple(concrete_value if isinstance(dim, str) else dim for dim in shape)


def _get_monkey_patch():
    def reshape(a, newshape, order="C"):
        if order != "C":
            raise NotImplementedError("Only C-style reshape is supported.")

        def reshape_abstract_eval(a, newshape):
            newshape_processed = _process_newshape(newshape)
            output_shape = _get_dynamic_output_shape(a.shape, newshape_processed)
            # For abstract evaluation, return a concrete shape.
            concrete_shape = _concretize_shape(output_shape)
            return core.ShapedArray(concrete_shape, a.dtype)

        jnp.reshape_p.multiple_results = False
        jnp.reshape_p.def_abstract_eval(reshape_abstract_eval)
        return jnp.reshape_p.bind(a, newshape=newshape)

    return reshape


@contextlib.contextmanager
def temporary_patch():
    original_fn = jnp.reshape
    jnp.reshape = _get_monkey_patch()
    try:
        yield
    finally:
        jnp.reshape = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_reshape(node_inputs, node_outputs, params):
        newshape = params["newshape"]
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])

        input_shape = node_inputs[0].aval.shape
        output_shape = _get_dynamic_output_shape(input_shape, newshape)
        processed_newshape = _process_newshape(newshape)

        # Compute a concrete shape for the initializer by replacing dynamic markers.
        concrete_shape = _concretize_shape(tuple(processed_newshape))

        # Create a shape tensor for ONNX.
        shape_tensor_name = s.get_unique_name("reshape_shape")
        onnx_shape = list(concrete_shape)

        # Use the new add_initializer method instead of directly accessing builder.initializers
        s.add_initializer(name=shape_tensor_name, vals=onnx_shape)

        reshape_node = helper.make_node(
            "Reshape",
            inputs=[input_name, shape_tensor_name],
            outputs=[output_name],
            name=s.get_unique_name("reshape"),
            allowzero=0,  # Explicit allowzero=0
        )
        s.add_node(reshape_node)
        s.add_shape_info(output_name, output_shape)

    return handle_reshape


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "jnp.reshape",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.reshape.html",
        "onnx": [
            {
                "component": "Reshape",
                "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.jnp",
        "testcases": [
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
            {  # Edge case
                "testcase": "reshape_from_scalar",
                "callable": lambda a: jnp.reshape(a, (1,)),
                "input_shapes": [()],
            },
        ],
    }
