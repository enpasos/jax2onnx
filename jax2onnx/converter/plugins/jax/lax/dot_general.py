import jax
import numpy as np
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.dot_general_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_dot_general(node_inputs, node_outputs, params):
        """Handle JAX dot_general primitive with a reshape-Gemm-reshape pattern."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])

        # Extract dot_general parameters
        dimension_numbers = params["dimension_numbers"]
        ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers

        lhs_name, rhs_name = input_names
        lhs_shape = node_inputs[0].aval.shape
        rhs_shape = node_inputs[1].aval.shape
        output_shape = node_outputs[0].aval.shape

        # Compute batch and feature dimensions
        np.prod(lhs_shape[: len(lhs_shape) - len(lhs_contract)], dtype=np.int64)
        feature_size = np.prod(
            lhs_shape[len(lhs_shape) - len(lhs_contract) :], dtype=np.int64
        )
        rhs_output_size = np.prod(
            rhs_shape[len(rhs_shape) - len(rhs_contract) :], dtype=np.int64
        )

        lhs_reshape_name = s.get_unique_name("reshape_input")
        const_shape = np.array([feature_size, rhs_output_size], dtype=np.int64)
        const_name = s.get_constant_name(const_shape)
        node_lhs = helper.make_node(
            "Reshape",
            inputs=[lhs_name, const_name],
            outputs=[lhs_reshape_name],
            name=s.get_unique_name("reshape_lhs"),
        )
        s.add_node(node_lhs)

        rhs_reshape_name = s.get_unique_name("rhs_reshape")
        const_name_rhs = s.get_constant_name(
            np.array([feature_size, rhs_output_size], dtype=np.int64)
        )
        node_rhs = helper.make_node(
            "Reshape",
            inputs=[rhs_name, const_name_rhs],
            outputs=[rhs_reshape_name],
            name=s.get_unique_name("reshape_rhs"),
        )
        s.add_node(node_rhs)

        gemm_output_name = s.get_unique_name("gemm_output")
        gemm_node = helper.make_node(
            "Gemm",
            inputs=[lhs_reshape_name, rhs_reshape_name],
            outputs=[gemm_output_name],
            name=s.get_unique_name("gemm"),
            alpha=1.0,
            beta=1.0,
            transB=0,
        )
        s.add_node(gemm_node)

        reshape_output_node = helper.make_node(
            "Reshape",
            inputs=[
                gemm_output_name,
                s.get_constant_name(np.array(output_shape, dtype=np.int64)),
            ],
            outputs=[output_name],
            name=s.get_unique_name("reshape_output"),
        )
        s.add_node(reshape_output_node)

    return _handle_dot_general


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "dot_general",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html",
        "onnx": [
            {
                "component": "MatMul",
                "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "dot_general",
                "callable": lambda x1, x2: jax.lax.dot_general(
                    x1, x2, (((1,), (0,)), ((), ()))
                ),
                "input_shapes": [(3, 3), (3, 3)],
            }
        ],
    }
