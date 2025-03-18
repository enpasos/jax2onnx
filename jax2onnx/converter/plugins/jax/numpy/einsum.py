import numpy as np
from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
from typing import TYPE_CHECKING, Tuple, List, Union, Dict

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the Einsum primitive
jnp.einsum_p = Primitive("jnp.einsum")


def get_primitive():
    """Returns the jnp.einsum primitive."""
    return jnp.einsum_p


def parse_einsum_equation(equation: str) -> Tuple[List[str], str]:
    """Parses the einsum equation into input and output terms."""
    parts = equation.split("->")
    if len(parts) != 2:
        raise ValueError("Einsum equation must contain '->'.")
    input_terms, output_term = parts
    return input_terms.split(","), output_term


def get_dynamic_output_shape(
    input_shapes: List[Tuple[Union[int, str], ...]], equation: str
) -> Tuple[Union[int, str], ...]:
    """Calculates the output shape while handling dynamic dimensions."""

    dummy_inputs = [
        np.zeros([1 if isinstance(d, str) else d for d in shape])
        for shape in input_shapes
    ]
    dummy_output = np.einsum(equation, *dummy_inputs)
    output_shape = list(dummy_output.shape)

    input_terms, output_term = parse_einsum_equation(equation)
    index_to_label: Dict[str, Union[int, str]] = {}

    for term, shape in zip(input_terms, input_shapes):
        for i, label in enumerate(term):
            if label not in index_to_label:
                try:
                    index_to_label[label] = shape[i]
                except IndexError:
                    index_to_label[label] = -1

    for i, label in enumerate(output_term):
        if label in index_to_label and isinstance(index_to_label[label], str):
            output_shape[i] = index_to_label[label]

    return tuple(output_shape)


def einsum_abstract_eval(*operands, equation, precision):
    """Abstract evaluation function for Einsum."""
    input_shapes = [op.shape for op in operands]
    output_shape = get_dynamic_output_shape(input_shapes, equation)
    return core.ShapedArray(output_shape, operands[0].dtype)


# Register abstract evaluation function
jnp.einsum_p.def_abstract_eval(einsum_abstract_eval)


def einsum(equation, *operands, precision=None):
    """Defines the primitive binding for Einsum."""
    return jnp.einsum_p.bind(*operands, equation=equation, precision=precision)


def patch_info():
    """Provides patching information for Einsum."""
    return {
        "patch_targets": [jnp],
        "patch_function": lambda _: einsum,
        "target_attribute": "einsum",
    }


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_einsum(node_inputs, node_outputs, params):
        equation = params.get("equation")
        input_names = [s.get_name(var) for var in node_inputs]
        output_name = s.get_name(node_outputs[0])

        einsum_node = helper.make_node(
            "Einsum",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("einsum"),
            equation=equation,
        )
        s.add_node(einsum_node)

        input_shapes = [inp.aval.shape for inp in node_inputs]
        output_shape = get_dynamic_output_shape(input_shapes, equation)
        s.add_shape_info(output_name, output_shape)

    return handle_einsum


def get_metadata() -> dict:
    """Returns metadata describing this plugin and its test cases."""
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
                "testcase": "einsum",
                "callable": lambda a, b: jnp.einsum("ij,j->i", a, b, precision=None),
                "input_shapes": [(3, 3), (3,)],
            },
            {
                "testcase": "einsum_matmul",
                "callable": lambda a, b: jnp.einsum("ij,jk->ik", a, b, precision=None),
                "input_shapes": [(4, 3), (3, 5)],
            },
            {
                "testcase": "einsum_dynamic",
                "callable": lambda a, b: jnp.einsum("ij,j->i", a, b, precision=None),
                "input_shapes": [("B", 3), (3,)],
            },
            {
                "testcase": "einsum_dynamic_matmul",
                "callable": lambda a, b: jnp.einsum(
                    "bij,jk->bik", a, b, precision=None
                ),
                "input_shapes": [("B", 5, 3), (3, 4)],
            },
            {
                "testcase": "einsum_transpose",
                "callable": lambda a: jnp.einsum("ij->ji", a, precision=None),
                "input_shapes": [(2, 3)],
            },
            {
                "testcase": "einsum_dynamic_transpose",
                "callable": lambda a: jnp.einsum("bij->bji", a, precision=None),
                "input_shapes": [("B", 2, 3)],
            },
            {
                "testcase": "einsum_dynamic_matmul2",
                "callable": lambda a, b: jnp.einsum(
                    "bij,jk->bik", a, b, precision=None
                ),
                "input_shapes": [("B", 5, 3), (3, 4)],
            },
            {
                "testcase": "einsum_dynamic_matmul3",
                "callable": lambda a, b: jnp.einsum(
                    "bij,bjk->bik", a, b, precision=None
                ),
                "input_shapes": [("B", 5, 3), ("B", 3, 4)],
            },
        ],
    }
