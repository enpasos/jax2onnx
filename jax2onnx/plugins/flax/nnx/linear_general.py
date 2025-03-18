import numpy as np
from jax import core
from flax import nnx
from onnx import helper
from jax.extend.core import Primitive
from typing import TYPE_CHECKING
from jax2onnx.plugin_system import register_plugin, PrimitivePlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

nnx.linear_general_p = Primitive("nnx.linear_general")


@register_plugin(
    primitive="nnx.linear_general",
    metadata={
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html",
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
                    (8, 32), (256,), axis=(-2, -1), rngs=nnx.Rngs(0)
                ),
                "input_shapes": [("B", 4, 8, 32)],
            },
            {
                "testcase": "linear_general_2",
                "callable": nnx.LinearGeneral(
                    (30,), (20,), axis=(-1,), rngs=nnx.Rngs(0)
                ),
                "input_shapes": [(3, 30)],
            },
            {
                "testcase": "linear_general_3",
                "callable": nnx.LinearGeneral(
                    (256,), (8, 32), axis=(-1,), rngs=nnx.Rngs(0)
                ),
                "input_shapes": [(2, 4, 256)],
            },
            {
                "testcase": "linear_general_4",
                "callable": nnx.LinearGeneral(
                    (8, 32), (256,), axis=(-2, -1), rngs=nnx.Rngs(0)
                ),
                "input_shapes": [(2, 4, 8, 32)],
            },
        ],
    },
)
class LinearGeneralPlugin(PrimitivePlugin):
    @staticmethod
    def abstract_eval(x, kernel, bias, dimension_numbers):
        ((lhs_contract, rhs_contract), _) = dimension_numbers

        lhs_contract = [d % len(x.shape) for d in lhs_contract]
        rhs_contract = [d % len(kernel.shape) for d in rhs_contract]

        x_batch_dims = [i for i in range(len(x.shape)) if i not in lhs_contract]
        kernel_out_dims = [
            kernel.shape[i] for i in range(len(kernel.shape)) if i not in rhs_contract
        ]

        output_shape = tuple([x.shape[i] for i in x_batch_dims] + kernel_out_dims)
        return core.ShapedArray(output_shape, x.dtype)

    @staticmethod
    def patch_info():
        def patched_linear_general_call(self, x):
            contracting_dims = (
                (self.axis,) if isinstance(self.axis, int) else self.axis,
                tuple(range(len(self.in_features))),
            )
            dimension_numbers = (contracting_dims, ((), ()))
            return nnx.linear_general_p.bind(
                x,
                self.kernel.value,
                self.bias.value if self.bias else None,
                dimension_numbers=dimension_numbers,
            )

        return {
            "patch_targets": [nnx.LinearGeneral],
            "patch_function": lambda _: patched_linear_general_call,
            "target_attribute": "__call__",
        }

    @staticmethod
    def to_onnx(s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_var, kernel_var, bias_var = node_inputs[:3]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)
        kernel_name = s.get_name(kernel_var)
        bias_name = s.get_name(bias_var) if bias_var else None

        ((lhs_contract, rhs_contract), _) = params["dimension_numbers"]

        # Compute shapes
        batch_dims = [
            d for d in range(len(input_var.aval.shape)) if d not in lhs_contract
        ]
        input_batch_size = np.prod([input_var.aval.shape[i] for i in batch_dims])
        input_feature_size = np.prod([input_var.aval.shape[i] for i in lhs_contract])

        kernel_contract_size = np.prod([kernel_var.aval.shape[i] for i in rhs_contract])
        kernel_out_size = np.prod(
            [
                kernel_var.aval.shape[i]
                for i in range(len(kernel_var.aval.shape))
                if i not in rhs_contract
            ]
        )

        reshaped_input_shape = (-1, input_feature_size)
        reshaped_kernel_shape = (kernel_contract_size, kernel_out_size)

        # Reshape input if necessary
        reshaped_input_name = s.get_unique_name("input_reshape")
        s.add_node(
            helper.make_node(
                "Reshape",
                inputs=[
                    input_name,
                    s.get_constant_name(np.array(reshaped_input_shape, dtype=np.int64)),
                ],
                outputs=[reshaped_input_name],
                name=s.get_unique_name("reshape_input"),
            )
        )

        # Reshape kernel
        kernel_const = s.name_to_const[kernel_name]
        weights_name = s.get_constant_name(kernel_const.reshape(reshaped_kernel_shape))

        # Handle bias
        if bias_name:
            bias_const = s.name_to_const[bias_name]
            bias_name = s.get_constant_name(bias_const.reshape((kernel_out_size,)))
        else:
            bias_const = np.zeros((kernel_out_size,), dtype=input_var.aval.dtype)
            bias_name = s.get_constant_name(bias_const)

        # GEMM node
        gemm_output_name = s.get_unique_name("gemm_output")
        s.add_node(
            helper.make_node(
                "Gemm",
                inputs=[reshaped_input_name, weights_name, bias_name],
                outputs=[gemm_output_name],
                name=s.get_unique_name("gemm"),
            )
        )

        # Final reshape to original output shape
        final_output_shape = (-1,) + tuple(output_var.aval.shape[1:])
        s.add_node(
            helper.make_node(
                "Reshape",
                inputs=[
                    gemm_output_name,
                    s.get_constant_name(np.array(final_output_shape, dtype=np.int64)),
                ],
                outputs=[output_name],
                name=s.get_unique_name("reshape_output"),
            )
        )
