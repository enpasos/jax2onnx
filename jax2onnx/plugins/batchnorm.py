# file: jax2onnx/plugins/batchnorm.py
# JAX API reference: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
import onnx.helper as oh
import numpy as np
import onnx
from flax import nnx

def build_onnx_node(self,  jax_inputs, input_names , nodes, initializers, counter):
    node_name = f"node{counter[0]}"
    counter[0] += 1

    # Extract the epsilon and momentum parameters from the BatchNorm instance or use defaults
    epsilon = getattr(self, "epsilon", 1e-5)
    momentum = 1 - getattr(self, "momentum", 0.9)

    # Extract learned parameters from the BatchNorm instance
    scale = self.scale
    bias = self.bias
    mean = self.mean
    var = self.var

    # Ensure parameters are numpy arrays
    scale_tensor = np.asarray(scale, dtype=np.float32)
    bias_tensor = np.asarray(bias, dtype=np.float32)
    mean_tensor = np.asarray(mean, dtype=np.float32)
    var_tensor = np.asarray(var, dtype=np.float32)

    # Create tensor names
    scale_name = f"{node_name}_scale"
    bias_name = f"{node_name}_bias"
    mean_name = f"{node_name}_mean"
    var_name = f"{node_name}_variance"

    # Add parameters to the initializers
    initializers.append(
        oh.make_tensor(scale_name, onnx.TensorProto.FLOAT, scale_tensor.shape, scale_tensor.flatten())
    )
    initializers.append(
        oh.make_tensor(bias_name, onnx.TensorProto.FLOAT, bias_tensor.shape, bias_tensor.flatten())
    )
    initializers.append(
        oh.make_tensor(mean_name, onnx.TensorProto.FLOAT, mean_tensor.shape, mean_tensor.flatten())
    )
    initializers.append(
        oh.make_tensor(var_name, onnx.TensorProto.FLOAT, var_tensor.shape, var_tensor.flatten())
    )

    outputs = [f"{node_name}_output"]

    # Add BatchNormalization node
    nodes.append(
        oh.make_node(
            "BatchNormalization",
            inputs=[input_names[0], scale_name, bias_name, mean_name, var_name],
            outputs=outputs,
            name=node_name,
            epsilon=epsilon,
            momentum=momentum,
        )
    )



    return outputs

nnx.BatchNorm.build_onnx_node = build_onnx_node


def get_test_params():
    return [
        {
            "model_name": "batchnorm",
            "model": lambda: nnx.BatchNorm(num_features=64, epsilon=1e-5, momentum=0.9, rngs=nnx.Rngs(0)),
            "input_shapes": [(11, 2, 2, 64)],
            "build_onnx_node": nnx.BatchNorm.build_onnx_node,
        }
    ]
