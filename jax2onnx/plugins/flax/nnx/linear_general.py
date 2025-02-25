import numpy as np
import onnx
import onnx.helper as oh
from flax import nnx
import jax.numpy as jnp
from jax2onnx.convert import Z, OnnxGraph
from jax2onnx.typing_helpers import Supports2Onnx
from jax2onnx.utils import retry_with_dynamic_batch_dim
from jax2onnx.plugins.jax.numpy.reshape import to_onnx_reshape


def build_linear_general_onnx_node(self: Supports2Onnx, z: Z, **params) -> Z:
    """Convert an `nnx.LinearGeneral` layer into an ONNX `Gemm` node."""
 
    onnx_graph : OnnxGraph = z.onnx_graph

    # in case of dynamic batch dimension we assume first axis is batch dimension
    # if z.is_dynamic_batch_dim:
    #     # we remember the dynamic batch dimension
    #     self.dynamic_batch_dim = z.shapes[0][0]
    #     # and set the batch dimension to 1
    #     z.shapes[0][0] = 1


    input_shape =  z.shapes[0] 
    input_name = z.names[0]

    in_features = self.in_features 
    out_features = self.out_features 
    axis =  self.axis 
    use_bias = self.use_bias

    batch_dims = list(input_shape[: -len(axis)])  # Compute batch dimensions
    output_shape = batch_dims + list(out_features)


    node_name = f"node{onnx_graph.next_id()}"

    orig_dynamic_batch_dim = onnx_graph.dynamic_batch_dim
    orig_internal_shape_info = onnx_graph.internal_shape_info

    # Step 1: Reshape input to 2D if necessary
    if len(axis) > 1 or len(input_shape) > 2:
        if onnx_graph.dynamic_batch_dim:
            onnx_graph.internal_shape_info = False
        # orig_dynamic_batch_dim = False
        reshape_params = {"shape": (-1, np.prod(in_features).item())}
        z = to_onnx_reshape(Z([input_shape], [input_name], onnx_graph), **reshape_params)
        reshaped_input_name = z.names[0]
        reshaped_input_shape = z.shapes[0]
        onnx_graph.internal_shape_info = orig_internal_shape_info
    else:
        reshaped_input_name = input_name
        reshaped_input_shape = input_shape

    # Step 2: Gemm (NA, MA) -> (NB, MB)
    in_dim = np.prod(in_features).item()  # Flatten input feature dimensions
    out_dim = np.prod(out_features).item()  # Flatten output feature dimensions
    kernel_shape = (in_dim, out_dim)

    transposed_kernel = self.kernel.value.reshape(kernel_shape).astype(np.float32)

    weight_name = f"{node_name}_weight"
    bias_name = f"{node_name}_bias"
    gemm_output_name = f"{node_name}_output"

    onnx_graph.add_initializer(
        oh.make_tensor(
            weight_name,
            onnx.TensorProto.FLOAT,
            list(kernel_shape),
            transposed_kernel.flatten(),
        )
    )

    if use_bias:
        bias_corrected_shape = (out_dim,)  # Convert to 1D
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

    if not onnx_graph.dynamic_batch_dim:  
        gemm_intermediate_shape = [
            np.prod(batch_dims).item(),
            out_dim,
        ]   
        onnx_graph.add_local_outputs([gemm_intermediate_shape], [gemm_output_name])
    else:
        gemm_intermediate_shape = [
            -1,
            out_dim,
        ]   
    


    # Step 3: Reshape back  
    if len(output_shape) > 2:
        # copy output_shape to reshape_shape
        if onnx_graph.dynamic_batch_dim:
            onnx_graph.internal_shape_info = False 
        reshape_shape = output_shape.copy()
        reshape_shape[0] = -1
        reshape_params = {"shape": reshape_shape, "output_shape": output_shape}
        z = to_onnx_reshape(Z([()], [gemm_output_name], onnx_graph), **reshape_params)
        final_out_name = z.names[0]
        onnx_graph.internal_shape_info = orig_internal_shape_info
        return Z([output_shape], [final_out_name], onnx_graph)

    z = Z([output_shape], [gemm_output_name], onnx_graph) 
    z.onnx_graph.add_local_outputs(z.shapes, z.names)
    return z


# Attach `to_onnx` method to `nnx.LinearGeneral`
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
