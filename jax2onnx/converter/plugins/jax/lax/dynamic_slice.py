import jax
import numpy as np
from typing import TYPE_CHECKING
from onnx import helper, TensorProto

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.dynamic_slice_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_dynamic_slice(node_inputs, node_outputs, params):
        """Handle JAX dynamic slice primitive.

        In JAX, the dynamic slice primitive is bound as:
          dynamic_slice_p.bind(operand, *start_indices, *dynamic_sizes, slice_sizes=static_sizes)
        Here we assume that no dynamic_sizes are provided so that
          node_inputs = [operand, start_index0, start_index1, ..., start_index{d-1}],
        where d is the rank of the operand.

        ONNX Slice requires a 1-D 'starts' input. We therefore:
          1. Process each start index by unsqueezing it via an Unsqueeze node,
             then casting it to int64.
          2. Concatenate these casted start indices to form a single 1-D tensor.
          3. Create a constant for the static slice sizes (as int64).
          4. Compute ends = starts + slice_sizes via an Add node.
          5. Create an axes constant (as int64) for all dimensions and build the Slice node.
        """
        # Get the operand to slice.
        operand_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])

        # Determine the rank (number of dimensions) from the operand's shape.
        d = len(node_inputs[0].aval.shape)

        # Process each dynamic start index (from node_inputs[1:1+d]).
        start_names = []
        for i in range(1, 1 + d):
            start_i_name = s.get_name(node_inputs[i])
            # Create a constant for axes: [0] with int64 type.
            axes_const = s.get_constant_name(np.array([0], dtype=np.int64))
            unsqueezed_name = s.get_unique_name(f"unsqueezed_start_{i}")
            unsqueeze_node = helper.make_node(
                "Unsqueeze",
                inputs=[start_i_name, axes_const],
                outputs=[unsqueezed_name],
                name=s.get_unique_name("unsqueeze_start"),
            )
            s.add_node(unsqueeze_node)

            # Insert a Cast node to ensure type int64.
            casted_name = s.get_unique_name(f"cast_start_{i}")
            cast_node = helper.make_node(
                "Cast",
                inputs=[unsqueezed_name],
                outputs=[casted_name],
                name=s.get_unique_name("cast_start"),
                to=TensorProto.INT64,
            )
            s.add_node(cast_node)

            start_names.append(casted_name)

        # Concatenate the casted start indices into one 1-D tensor.
        starts_concat_name = s.get_unique_name("dynamic_starts")
        concat_node = helper.make_node(
            "Concat",
            inputs=start_names,
            outputs=[starts_concat_name],
            name=s.get_unique_name("concat_starts"),
            axis=0,
        )
        s.add_node(concat_node)

        # Create constant for static slice sizes as int64.
        slice_sizes = params["slice_sizes"]
        slice_sizes_const = s.get_constant_name(np.array(slice_sizes, dtype=np.int64))

        # Compute dynamic ends: ends = starts + slice_sizes.
        ends_name = s.get_unique_name("dynamic_ends")
        add_node = helper.make_node(
            "Add",
            inputs=[starts_concat_name, slice_sizes_const],
            outputs=[ends_name],
            name=s.get_unique_name("add_slice_ends"),
        )
        s.add_node(add_node)

        # Create constant for axes: [0, 1, ..., d-1] as int64.
        axes = list(range(d))
        axes_const = s.get_constant_name(np.array(axes, dtype=np.int64))

        inputs_list = [operand_name, starts_concat_name, ends_name, axes_const]
        if "strides" in params and params["strides"]:
            strides = params["strides"]
            strides_const = s.get_constant_name(np.array(strides, dtype=np.int64))
            inputs_list.append(strides_const)

        node = helper.make_node(
            "Slice",
            inputs=inputs_list,
            outputs=[output_name],
            name=s.get_unique_name("dynamic_slice"),
        )
        s.add_node(node)

    return _handle_dynamic_slice


def get_metadata() -> dict:
    """Return metadata describing this dynamic slice plugin and its test cases."""
    return {
        "jaxpr_primitive": "dynamic_slice",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html",
        "onnx": [
            {
                "component": "Slice",
                "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "dynamic_slice_test1",
                "callable": lambda x: jax.lax.dynamic_slice(x, [1], [2]),
                "input_shapes": [(5,)],
            }
        ],
    }
