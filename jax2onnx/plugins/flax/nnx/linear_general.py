from jax2onnx.plugin_system import register_plugin, PrimitivePlugin
import numpy as np
from jax import core
from onnx import helper
from flax import nnx


@register_plugin(
    primitive="nnx.linear_general",  # consistent unique primitive
    metadata={
        "doc": "Linear general operation",
        "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.LinearGeneral",
        "onnx": [
            {
                "component": "Gemm",
                "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html",
            },
            {
                "component": "Reshape",
                "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
            },
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
            {
                "testcase": "linear_general",
                "callable": nnx.LinearGeneral(
                    in_features=(8, 32),
                    out_features=(256,),
                    axis=(-2, -1),
                    rngs=nnx.Rngs(0),
                ),
                "input_shapes": [("B", 4, 8, 32)],
            },
            {
                "testcase": "linear_general_2",
                "callable": nnx.LinearGeneral(
                    in_features=(30,),
                    out_features=(20,),
                    axis=(-1,),
                    rngs=nnx.Rngs(0),
                ),
                "input_shapes": [(3, 30)],
            },
            {
                "testcase": "linear_general_3",
                "callable": nnx.LinearGeneral(
                    in_features=(256,),
                    out_features=(8, 32),
                    axis=(-1,),
                    rngs=nnx.Rngs(0),
                ),
                "input_shapes": [(2, 4, 256)],
            },
            {
                "testcase": "linear_general_4",
                "callable": nnx.LinearGeneral(
                    in_features=(8, 32),
                    out_features=(256,),
                    axis=(-2, -1),
                    rngs=nnx.Rngs(0),
                ),
                "input_shapes": [(2, 4, 8, 32)],
            },
        ],
    },
)
@contextlib.contextmanager
def temporary_patch():
    original_call = nnx.LinearGeneral.__call__
    nnx.LinearGeneral.__call__ = _get_monkey_patch()
    try:
        yield
    finally:
        nnx.LinearGeneral.__call__ = original_call


class LinearGeneralPlugin(PrimitivePlugin):
    def abstract_eval(self, x, kernel, bias, dimension_numbers):
        """Computes output shape based on input, kernel, and dimensions."""
        ((lhs_contract, rhs_contract), _) = dimension_numbers

        lhs_contract = [d % len(x.shape) for d in lhs_contract]
        rhs_contract = [d % len(kernel.shape) for d in rhs_contract]

        x_batch_dims = [i for i in range(len(x.shape)) if i not in lhs_contract]
        x_batch_dims_sizes = [x.shape[i] for i in x_batch_dims]

        kernel_noncontract_dims = [
            i for i in range(len(kernel.shape)) if i not in rhs_contract
        ]
        kernel_out_dims = [kernel.shape[i] for i in kernel_noncontract_dims]

        output_shape = tuple(x_batch_dims_sizes + kernel_out_dims)
        return core.ShapedArray(output_shape, x.dtype)

    def to_onnx(self, node, graph, inputs, outputs):
        """Converts linear_general JAX op to ONNX Gemm op."""
        input_var, kernel_var, bias_var = inputs[:3]
        output_var = outputs[0]

        input_name = graph.get_name(input_var)
        output_name = graph.get_name(output_var)
        kernel_name = graph.get_name(kernel_var)
        bias_name = graph.get_name(bias_var) if bias_var else None

        shapes = self.abstract_eval(
            input_var, kernel_var, bias_var, node.params["dimension_numbers"]
        )
        output_shape = shapes.shape

        lhs_contract, rhs_contract = node.params["dimension_numbers"][0]

        # Reshape kernel to 2D (for Gemm)
        new_kernel_shape = (
            np.prod([kernel_var.shape[i] for i in rhs_contract]),
            np.prod(
                [
                    kernel_var.shape[i]
                    for i in range(kernel_var.ndim)
                    if i not in rhs_contract
                ]
            ),
        )
        weights_name = graph.get_constant_name(kernel_var.reshape(new_kernel_shape))

        # Reshape input to 2D (batch_dim, feature_dim)
        batch_dims = [i for i in range(input_var.ndim) if i not in lhs_contract]
        input_batch_size = np.prod([input_var.shape[i] for i in batch_dims])
        input_feature_size = np.prod([input_var.shape[i] for i in lhs_contract])

        reshaped_input_shape = (-1, input_feature_size)
        reshaped_input_name = graph.get_unique_name("input_reshape")
        graph.add_node(
            helper.make_node(
                "Reshape",
                inputs=[
                    input_name,
                    graph.get_constant_name(
                        np.array(reshaped_input_shape, dtype=np.int64)
                    ),
                ],
                outputs=[reshaped_input_name],
                name=graph.get_unique_name("reshape_input"),
            )
        )

        # Ensure bias is correctly shaped
        if bias_name:
            bias_name = graph.get_constant_name(
                bias_var.reshape((new_kernel_shape[1],))
            )

        gemm_output_name = graph.get_unique_name("gemm_output")
        gemm_inputs = [reshaped_input_name, weights_name]
        if bias_name:
            gemm_inputs.append(bias_name)

        graph.add_node(
            helper.make_node(
                "Gemm",
                inputs=gemm_inputs,
                outputs=[gemm_output_name],
                name=graph.get_unique_name("gemm"),
            )
        )

        # Reshape back to original output shape
        final_output_shape = [-1] + list(output_shape[1:])
        graph.add_node(
            helper.make_node(
                "Reshape",
                inputs=[
                    gemm_output_name,
                    graph.get_constant_name(
                        np.array(final_output_shape, dtype=np.int64)
                    ),
                ],
                outputs=[output_name],
                name=graph.get_unique_name("reshape_output"),
            )
        )
