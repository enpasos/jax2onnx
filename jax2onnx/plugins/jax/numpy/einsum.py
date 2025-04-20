from typing import TYPE_CHECKING

import numpy as np
from jax import core
from jax import numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the Einsum primitive
jnp.einsum_p = Primitive("jnp.einsum")
jnp.einsum_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=jnp.einsum_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum.html",
    onnx=[
        {
            "component": "Einsum",
            "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="einsum",
    testcases=[
        {
            "testcase": "einsum",
            "callable": lambda a, b: jnp.einsum("ij,j->i", a, b, precision=None),
            "input_shapes": [(3, 3), (3,)],
        },
        {
            "testcase": "einsum_preferred_element_type",
            "callable": lambda a, b: jnp.einsum(
                "ij,j->i", a, b, precision=None, preferred_element_type=jnp.float32
            ),
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
            "callable": lambda a, b: jnp.einsum("bij,jk->bik", a, b, precision=None),
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
            "callable": lambda a, b: jnp.einsum("bij,jk->bik", a, b, precision=None),
            "input_shapes": [("B", 5, 3), (3, 4)],
        },
        {
            "testcase": "einsum_dynamic_matmul3",
            "callable": lambda a, b: jnp.einsum("bij,bjk->bik", a, b, precision=None),
            "input_shapes": [("B", 5, 3), ("B", 3, 4)],
        },
        {
            "testcase": "einsum_outer_product",
            "callable": lambda a, b: jnp.einsum("i,j->ij", a, b, precision=None),
            "input_shapes": [(3,), (4,)],
        },
        {
            "testcase": "einsum_trace",
            "callable": lambda a: jnp.einsum("ii->", a, precision=None),
            "input_shapes": [(3, 3)],
        },
        {
            "testcase": "einsum_sum",
            "callable": lambda a: jnp.einsum("ij->", a, precision=None),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "einsum_broadcast",
            "callable": lambda a, b: jnp.einsum("ij,kj->ikj", a, b, precision=None),
            "input_shapes": [(2, 3), (4, 3)],
        },
        {
            "testcase": "einsum_reduce",
            "callable": lambda a: jnp.einsum("ijk->i", a, precision=None),
            "input_shapes": [(2, 3, 4)],
        },
        {
            "testcase": "einsum_permute",
            "callable": lambda a: jnp.einsum("ijk->kji", a, precision=None),
            "input_shapes": [(2, 3, 4)],
        },
        {
            "testcase": "einsum_dynamic_outer",
            "callable": lambda a, b: jnp.einsum("i,j->ij", a, b, precision=None),
            "input_shapes": [("B",), (4,)],
        },
        {
            "testcase": "einsum_dynamic_reduce",
            "callable": lambda a: jnp.einsum("bij->b", a, precision=None),
            "input_shapes": [("B", 3, 4)],
        },
    ],
)
class EinsumPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.numpy.einsum to ONNX.
    """

    @staticmethod
    def _parse_einsum_equation(equation: str) -> tuple[list[str], str]:
        """Parses the einsum equation into input and output terms."""
        parts = equation.split("->")
        if len(parts) != 2:
            raise ValueError("Einsum equation must contain '->'.")
        input_terms, output_term = parts
        return input_terms.split(","), output_term

    @staticmethod
    def _get_dynamic_output_shape(
        input_shapes: list[tuple[int | str, ...]], equation: str
    ) -> tuple[int | str, ...]:
        """Calculates the output shape while handling dynamic dimensions."""

        dummy_inputs = [
            np.zeros([1 if isinstance(d, str) else d for d in shape])
            for shape in input_shapes
        ]
        dummy_output = np.einsum(equation, *dummy_inputs)
        output_shape = list(dummy_output.shape)

        input_terms, output_term = EinsumPlugin._parse_einsum_equation(equation)
        index_to_label: dict[str, int | str] = {}

        for term, shape in zip(input_terms, input_shapes, strict=False):
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

    @staticmethod
    def abstract_eval(*operands, equation, precision, preferred_element_type=None):
        """Abstract evaluation function for Einsum."""
        input_shapes = [op.shape for op in operands]
        output_shape = EinsumPlugin._get_dynamic_output_shape(input_shapes, equation)
        return core.ShapedArray(output_shape, operands[0].dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of Einsum to ONNX format."""
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
        output_shape = EinsumPlugin._get_dynamic_output_shape(input_shapes, equation)
        s.add_shape_info(
            output_name,
            tuple(int(dim) for dim in output_shape if isinstance(dim, (int, str))),
        )

    @staticmethod
    def _einsum(equation, *operands, precision=None, preferred_element_type=None):
        """Defines the primitive binding for Einsum. Corrected version."""
        # Pass only the operands as positional arguments, and equation/precision/preferred_element_type as keywords.
        return jnp.einsum_p.bind(
            *operands,
            equation=equation,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for Einsum."""

        def patched_einsum(
            equation, *operands, precision=None, preferred_element_type=None
        ):
            return EinsumPlugin._einsum(
                equation,
                *operands,
                precision=precision,
                preferred_element_type=preferred_element_type,
            )

        return patched_einsum

    @staticmethod
    def patch_info():
        """Provides patching information for Einsum."""
        return {
            "patch_targets": [jnp],
            "patch_function": lambda _: EinsumPlugin.get_monkey_patch(),
            "target_attribute": "einsum",
        }


# Register abstract evaluation function
jnp.einsum_p.def_abstract_eval(EinsumPlugin.abstract_eval)


def einsum_batching_rule(
    batched_args, batch_dims, equation, precision, preferred_element_type=None
):
    """Batching rule for jnp.einsum."""
    # Adjust the equation to account for the batch dimensions
    input_terms, output_term = equation.split("->")
    input_terms = input_terms.split(",")

    new_input_terms = []
    new_operands = []
    new_batch_dims = []

    for operand, batch_dim, term in zip(
        batched_args, batch_dims, input_terms, strict=False
    ):
        if batch_dim is not None:
            batch_label = "B"
            if batch_label not in term:
                term = (
                    batch_label + term
                )  # Add a batch dimension label if not already present
            operand = batching.moveaxis(operand, batch_dim, 0)
            new_batch_dims.append(0)
        else:
            new_batch_dims.append(None)

        # Add ellipsis if the operand has more dimensions than the term specifies
        if len(term) < operand.ndim:
            term = "..." + term

        new_input_terms.append(term)
        new_operands.append(operand)

    # Ensure the batch label is added to the output term only once
    batch_label = "B"
    if batch_label not in output_term:
        output_term = batch_label + output_term

    # Add ellipsis to the output term if necessary
    if len(output_term) < max(len(term) for term in new_input_terms):
        output_term = "..." + output_term

    new_equation = ",".join(new_input_terms) + "->" + output_term

    # Call the original einsum with the modified equation and operands
    result = jnp.einsum(
        new_equation,
        *new_operands,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    # The batch dimension is now the first dimension of the result
    return result, 0


# Update the registration of the batching rule
batching.primitive_batchers[jnp.einsum_p] = einsum_batching_rule
