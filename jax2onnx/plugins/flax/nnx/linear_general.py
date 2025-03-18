# file: jax2onnx/plugins/flax/nnx/linear_general.py
# file: jax2onnx/plugins/flax/nnx/linear_general.py

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
        "jaxpr_primitive": "nnx.linear_general",
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
class LinearGeneralPlugin(PrimitivePlugin):

    @staticmethod
    def _shape_linear_general(x_shape, kernel_shape, dimension_numbers):
        ((lhs_contract, rhs_contract), _) = dimension_numbers

        lhs_contract = [d % len(x_shape) for d in lhs_contract]
        rhs_contract = [d % len(kernel_shape) for d in rhs_contract]

        x_batch_dims = [i for i in range(len(x_shape)) if i not in lhs_contract]
        x_batch_dims_sizes = [x_shape[i] for i in x_batch_dims]

        kernel_noncontract_dims = [
            i for i in range(len(kernel_shape)) if i not in rhs_contract
        ]
        kernel_out_dims = [kernel_shape[i] for i in kernel_noncontract_dims]

        output_shape = tuple(x_batch_dims_sizes + kernel_out_dims)

        new_kernel_dims_sizes = (
            np.prod([kernel_shape[i] for i in rhs_contract]).item(),
            np.prod(kernel_out_dims).item(),
        )

        input_gemm_shape = (
            np.prod(x_batch_dims_sizes).item(),
            np.prod([x_shape[i] for i in lhs_contract]).item(),
        )
        output_gemm_shape = (input_gemm_shape[0], new_kernel_dims_sizes[1])

        return {
            "input": x_shape,
            "input_gemm": input_gemm_shape,
            "output_gemm": output_gemm_shape,
            "output": output_shape,
            "new_kernel": new_kernel_dims_sizes,
        }

    @staticmethod
    def abstract_eval(x, kernel, bias, dimension_numbers):
        """Abstract evaluation function for linear_general."""
        shapes = LinearGeneralPlugin._shape_linear_general(
            x.shape, kernel.shape, dimension_numbers
        )
        return core.ShapedArray(shapes["output"], x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_var, kernel_var, bias_var = node_inputs[:3]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)
        kernel_name = s.get_name(kernel_var)
        bias_name = s.get_name(bias_var) if bias_var else None

        shapes = LinearGeneralPlugin._shape_linear_general(
            input_var.aval.shape, kernel_var.aval.shape, params["dimension_numbers"]
        )
        output_shape = shapes["output"]
        new_kernel_shape = shapes["new_kernel"]
        input_gemm_shape = shapes["input_gemm"]
        output_gemm_shape = shapes["output_gemm"]

        kernel_const = s.name_to_const[kernel_name]
        weights_name = s.get_constant_name(kernel_const.reshape(new_kernel_shape))

        target_input_shape = (-1,) + input_gemm_shape[1:]
        if not (
            len(input_var.aval.shape) == len(target_input_shape)
            and input_var.aval.shape[1:] == target_input_shape[1:]
        ):
            input_reshape_name = s.get_unique_name("input_reshape")
            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[
                        input_name,
                        s.get_constant_name(
                            np.array(target_input_shape, dtype=np.int64)
                        ),
                    ],
                    outputs=[input_reshape_name],
                    name=s.get_unique_name("reshape_input"),
                )
            )
            s.add_shape_info(input_reshape_name, input_gemm_shape)
        else:
            input_reshape_name = input_name

        # Ensure the bias is 1D with shape (output_gemm_shape[1],)
        if bias_name is not None:
            bias_const = s.name_to_const[bias_name]
            target_bias_shape = (output_gemm_shape[1],)
            if bias_const.shape != target_bias_shape:
                bias_const = bias_const.reshape(target_bias_shape)
                bias_name = s.get_constant_name(bias_const)
            gemm_inputs = [input_reshape_name, weights_name, bias_name]
        else:
            bias_shape = (output_gemm_shape[1],)  # Ensure 1D bias.
            zero_bias = np.zeros(bias_shape, dtype=input_var.aval.dtype)
            bias_name = s.get_constant_name(zero_bias)
            gemm_inputs = [input_reshape_name, weights_name, bias_name]

        gemm_output_name = (
            output_name
            if (
                len(output_gemm_shape) == len(output_shape)
                and output_gemm_shape[1:] == output_shape[1:]
            )
            else s.get_unique_name("gemm_output")
        )

        s.add_node(
            helper.make_node(
                "Gemm",
                inputs=gemm_inputs,
                outputs=[gemm_output_name],
                name=s.get_unique_name("gemm"),
            )
        )
        s.add_shape_info(gemm_output_name, output_gemm_shape)

        # Use dynamic reshape target: replace the batch dimension with -1.
        if gemm_output_name != output_name:
            target_output_shape = [-1] + list(output_shape[1:])
            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[
                        gemm_output_name,
                        s.get_constant_name(
                            np.array(target_output_shape, dtype=np.int64)
                        ),
                    ],
                    outputs=[output_name],
                    name=s.get_unique_name("reshape_output"),
                )
            )

    @staticmethod
    def linear_general(x, kernel, bias, dimension_numbers):
        """Defines the primitive binding for linear_general."""
        nnx.linear_general_p.multiple_results = False
        return nnx.linear_general_p.bind(
            x, kernel, bias, dimension_numbers=dimension_numbers
        )

    @staticmethod
    def get_monkey_patch():
        """Returns a patched version of LinearGeneral's call method."""

        def patched_linear_general_call(self, x):
            contracting_dims = (
                (self.axis,) if isinstance(self.axis, int) else self.axis,
                tuple(range(len(self.in_features))),
            )
            dimension_numbers = (contracting_dims, ((), ()))
            return LinearGeneralPlugin.linear_general(
                x,
                self.kernel.value,
                self.bias.value if self.bias else None,
                dimension_numbers,
            )

        return patched_linear_general_call

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx.LinearGeneral],
            "patch_function": lambda _: LinearGeneralPlugin.get_monkey_patch(),
        }


# Register abstract evaluation function
nnx.linear_general_p.def_abstract_eval(LinearGeneralPlugin.abstract_eval)
