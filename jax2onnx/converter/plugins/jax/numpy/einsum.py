# file: jax2onnx/converter/plugins/flax/nnx/einsum.py
import numpy as np
from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
import contextlib
from typing import TYPE_CHECKING, Tuple, List, Union, Dict

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define a new primitive for einsum.
jnp.einsum_p = Primitive("jnp.einsum")


def get_primitive():
    return jnp.einsum_p


def _parse_einsum_equation(equation: str) -> Tuple[List[str], str]:
    """Parses the einsum equation into input and output terms."""
    parts = equation.split("->")
    if len(parts) != 2:
        raise ValueError("Einsum equation must contain '->'.")
    input_terms, output_term = parts
    input_terms_list = input_terms.split(",")
    return input_terms_list, output_term


def _get_dynamic_output_shape(
    input_shapes: List[Tuple[Union[int, str], ...]], equation: str
) -> Tuple[Union[int, str], ...]:
    """Calculates the output shape, correctly handling dynamic dimensions.

    Args:
        input_shapes: List of input shapes (tuples). Can contain ints or strs.
        equation: The einsum equation string (e.g., "bij,bjk->bik").

    Returns:
        The output shape, as a tuple. May contain ints or strs.
    """

    # 1. Create dummy inputs (replace dynamic dims with 1).
    dummy_inputs = [
        np.zeros([1 if isinstance(d, str) else d for d in shape])
        for shape in input_shapes
    ]

    # 2. Calculate the output shape using numpy.einsum.
    dummy_output = np.einsum(equation, *dummy_inputs)
    output_shape = list(dummy_output.shape)

    # 3. Parse the equation to map indices to labels.
    input_terms, output_term = _parse_einsum_equation(equation)
    index_to_label: Dict[str, Union[int, str]] = {}

    # 3a. Build the index_to_label map from inputs.
    for term, shape in zip(input_terms, input_shapes):
        for i, label in enumerate(term):
            if label not in index_to_label:
                try:
                    dim_value = shape[i]
                    index_to_label[label] = dim_value
                except IndexError:
                    index_to_label[label] = -1

    # 3b. Substitute dynamic labels into the output shape.
    for i, label in enumerate(output_term):
        if label in index_to_label and isinstance(index_to_label[label], str):
            output_shape[i] = index_to_label[label]

    return tuple(output_shape)


def _get_monkey_patch():
    def einsum(equation, *operands, precision=None):
        def einsum_abstract_eval(*operands, equation, precision):
            input_shapes = [op.shape for op in operands]
            output_shape = _get_dynamic_output_shape(input_shapes, equation)
            return core.ShapedArray(output_shape, operands[0].dtype)

        jnp.einsum_p.multiple_results = False
        jnp.einsum_p.def_abstract_eval(einsum_abstract_eval)
        return jnp.einsum_p.bind(*operands, equation=equation, precision=precision)

    return einsum


@contextlib.contextmanager
def temporary_patch():
    original_fn = jnp.einsum
    jnp.einsum = _get_monkey_patch()
    try:
        yield
    finally:
        jnp.einsum = original_fn


def _process_einsum_equation(equation: str, num_inputs: int) -> str:
    parts = equation.split("->")
    if len(parts) != 2:
        raise ValueError("Einsum equation must contain '->'.")
    input_terms, output_term = parts
    input_terms_list = input_terms.split(",")
    if len(input_terms_list) != num_inputs:
        raise ValueError(
            f"Einsum equation has {len(input_terms_list)} input terms, expected {num_inputs}."
        )
    return equation


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_einsum(node_inputs, node_outputs, params):
        equation = params.get("equation")
        input_names = [s.get_name(var) for var in node_inputs]
        output_name = s.get_name(node_outputs[0])
        processed_equation = _process_einsum_equation(equation, len(node_inputs))
        input_shapes = [inp.aval.shape for inp in node_inputs]
        output_shape = _get_dynamic_output_shape(input_shapes, equation)
        einsum_node = helper.make_node(
            "Einsum",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("einsum"),
            equation=processed_equation,
        )
        s.add_node(einsum_node)
        s.add_shape_info(output_name, output_shape)

    return handle_einsum


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "jnp.einsum",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum.html",
        "onnx": [
            {
                "component": "Einsum",
                "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.jnp",
        "testcases": [
            {
                "testcase": "einsum",  # Matrix-vector multiplication
                "callable": lambda a, b: jnp.einsum("ij,j->i", a, b, precision=None),
                "input_shapes": [(3, 3), (3,)],
            },
            {
                "testcase": "einsum_matmul",  # Matrix multiplication
                "callable": lambda a, b: jnp.einsum("ij,jk->ik", a, b, precision=None),
                "input_shapes": [(4, 3), (3, 5)],
            },
            {
                "testcase": "einsum_dynamic",  # Dynamic dimension test
                "callable": lambda a, b: jnp.einsum("ij,j->i", a, b, precision=None),
                "input_shapes": [("B", 3), (3,)],
            },
            {
                "testcase": "einsum_dynamic_matmul",  # Dynamic batch, matrix multiplication
                "callable": lambda a, b: jnp.einsum(
                    "bij,jk->bik", a, b, precision=None
                ),
                "input_shapes": [("B", 5, 3), (3, 4)],
            },
            {
                "testcase": "einsum_transpose",  # Simple transpose
                "callable": lambda a: jnp.einsum("ij->ji", a, precision=None),
                "input_shapes": [(2, 3)],
            },
            {
                "testcase": "einsum_dynamic_transpose",  # Dynamic transpose
                "callable": lambda a: jnp.einsum("bij->bji", a, precision=None),
                "input_shapes": [("B", 2, 3)],
            },
            {
                "testcase": "einsum_dynamic_matmul2",  # different dynamic dim
                "callable": lambda a, b: jnp.einsum(
                    "bij,jk->bik", a, b, precision=None
                ),
                "input_shapes": [("B", 5, 3), (3, 4)],
            },
            {
                "testcase": "einsum_dynamic_matmul3",  # different dynamic dim
                "callable": lambda a, b: jnp.einsum(
                    "bij,bjk->bik", a, b, precision=None
                ),
                "input_shapes": [("B", 5, 3), ("B", 3, 4)],
            },
        ],
    }
