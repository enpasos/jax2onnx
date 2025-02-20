# file: jax2onnx/plugins/linear_general.py
import numpy as np
import onnx
import onnx.helper as oh
from flax import nnx

from jax2onnx.to_onnx import Z
from jax2onnx.typing_helpers import Supports2Onnx


def build_linear_general_onnx_node(self: Supports2Onnx, z: Z, **params) -> Z:
    """Convert an `nnx.LinearGeneral` layer into an ONNX `Gemm` node."""
    onnx_graph = z.onnx_graph
    input_shape = list(map(int, z.shapes[0]))  # Convert to Python int
    input_name = z.names[0]

    in_features = tuple(map(int, self.in_features))  # Tuple of Python ints
    out_features = tuple(map(int, self.out_features))  # Tuple of Python ints
    axis = tuple(map(int, self.axis))  # Tuple of Python ints
    use_bias = self.use_bias

    batch_dims = list(input_shape[: -len(axis)])  # Compute batch dimensions
    output_shape = batch_dims + list(out_features)

    node_name = f"node{onnx_graph.next_id()}"

    # ** Reshape input to 2D if necessary **
    if len(axis) > 1 or len(input_shape) > 2:
        # Compute reshaped shape (flatten in_features)
        reshaped_input_name = f"{node_name}_reshaped_input"
        feature_dim_prod = np.prod(in_features).item()
        reshaped_input_shape = [np.prod(batch_dims).item(), feature_dim_prod]

        shape_name = f"{node_name}_reshape_shape"
        onnx_graph.add_initializer(
            oh.make_tensor(
                name=shape_name,
                data_type=onnx.TensorProto.INT64,
                dims=[2],  # Always reshape to 2D: (batch_size, feature_dim)
                vals=[int(dim) for dim in reshaped_input_shape],
            )
        )
        onnx_graph.add_node(
            oh.make_node(
                "Reshape",
                inputs=[input_name, shape_name],
                outputs=[reshaped_input_name],
                name=f"{node_name}_reshape_input",
            )
        )
        onnx_graph.add_local_outputs([reshaped_input_shape], [reshaped_input_name])
    else:
        reshaped_input_name = input_name
        reshaped_input_shape = input_shape

    # Compute correct weight matrix shape for Gemm
    in_dim = np.prod(in_features).item()  # Flatten input feature dimensions
    out_dim = np.prod(out_features).item()  # Flatten output feature dimensions
    kernel_shape = (in_dim, out_dim)

    # Ensure the weight matrix is reshaped correctly
    transposed_kernel = self.kernel.value.reshape(kernel_shape).astype(np.float32)

    # If the input is 2D after reshaping, we use `Gemm`
    if len(reshaped_input_shape) == 2:
        weight_name = f"{node_name}_weight"
        bias_name = f"{node_name}_bias"
        gemm_output_name = f"{node_name}_output"

        # Store weights as an ONNX initializer
        onnx_graph.add_initializer(
            oh.make_tensor(
                weight_name,
                onnx.TensorProto.FLOAT,
                list(kernel_shape),
                transposed_kernel.flatten(),
            )
        )

        # Store bias as an ONNX initializer if applicable
        if use_bias:
            bias_name = f"{node_name}_bias"

            # ✅ **Fix Bias Shape**: Flatten to (out_dim,) so it can broadcast correctly in Gemm
            bias_corrected_shape = (np.prod(out_features).item(),)  # Convert to 1D

            onnx_graph.add_initializer(
                oh.make_tensor(
                    bias_name,
                    onnx.TensorProto.FLOAT,
                    list(bias_corrected_shape),
                    self.bias.value.flatten().astype(np.float32),
                )
            )
        else:
            bias_name = ""  # `Gemm` allows an empty string to indicate no bias

        # Create `Gemm` node
        onnx_graph.add_node(
            oh.make_node(
                "Gemm",
                inputs=[reshaped_input_name, weight_name, bias_name],
                outputs=[gemm_output_name],
                name=node_name,
                alpha=1.0,
                beta=1.0,
                transB=0,  # Ensure correct weight multiplication
            )
        )

        # ✅ **Fix: Correct the intermediate shape registration**
        gemm_intermediate_shape = [
            np.prod(batch_dims).item(),
            out_dim,
        ]  # (Flattened batch, out_dim)
        onnx_graph.add_local_outputs([gemm_intermediate_shape], [gemm_output_name])

        # Reshape back to original shape if necessary
        if len(output_shape) > 2:
            final_out_name = f"{gemm_output_name}_reshaped"
            shape_out_name = f"{node_name}_reshape_output_shape"

            shape_out_tensor = oh.make_tensor(
                name=shape_out_name,
                data_type=onnx.TensorProto.INT64,
                dims=[len(output_shape)],
                vals=[int(dim) for dim in output_shape],
            )
            onnx_graph.add_initializer(shape_out_tensor)
            onnx_graph.add_node(
                oh.make_node(
                    "Reshape",
                    inputs=[gemm_output_name, shape_out_name],
                    outputs=[final_out_name],
                    name=f"{node_name}_reshape_output",
                )
            )
            onnx_graph.add_local_outputs([output_shape], [final_out_name])
            return Z([output_shape], [final_out_name], onnx_graph)

        return Z([output_shape], [gemm_output_name], onnx_graph)

    # Fallback return statement to handle unexpected cases
    return Z([output_shape], [reshaped_input_name], onnx_graph)


# ✅ Attach `to_onnx` method to `nnx.LinearGeneral`
nnx.LinearGeneral.to_onnx = build_linear_general_onnx_node


def get_test_params() -> list:
    """Return test parameters for verifying the ONNX conversion of `nnx.LinearGeneral`."""
    return [
        {
            "jax_component": "flax.nnx.LinearGeneral",
            "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.LinearGeneral",
            "onnx": [
                {
                    "component": "Gemm",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html",
                },
                {
                    "component": "MatMul",
                    "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "linear_general",
                    "component": nnx.LinearGeneral(
                        in_features=(8, 32),
                        out_features=(256,),
                        axis=(-2, -1),
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(2, 4, 8, 32)],
                },
                {
                    "testcase": "linear_general_2",
                    "component": nnx.LinearGeneral(
                        in_features=(256,),
                        out_features=(8, 32),
                        axis=(-1,),
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(2, 4, 256)],
                },
                {
                    "testcase": "linear_general_mha_projection",
                    "component": nnx.LinearGeneral(
                        in_features=(8, 32),
                        out_features=(256,),
                        axis=(-2, -1),
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(8, 8, 32)],
                },
                {
                    "testcase": "linear_general_mha_projection2",
                    "component": nnx.LinearGeneral(
                        in_features=(256),
                        out_features=(256,),
                        axis=(-1),
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(8, 256)],
                },
            ],
        }
    ]
