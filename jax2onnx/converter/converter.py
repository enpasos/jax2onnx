# jax2onnx/converter/converter.py

import jax
import jax.numpy as jnp
import onnx
from onnx import helper, TensorProto
import numpy as np
from typing import Dict, Any
from jax2onnx.converter.primitives.jax.random import random_gamma
from jax2onnx.converter.primitives.flax.nnx import linear_general
from jax2onnx.converter.primitives.jax.lax import (
    neg,
    add,
    mul,
    sub,
    div,
    not_,
    eq,
    ne,
    lt,
    gt,
    max,
    min,
    select_n,
    xor,
    dot_general,
    reduce_sum,
    reduce_max,
    reduce_min,
    and_,
    or_,
    gather,
    scatter_add,
    argmax,
    argmin,
    square,
    integer_pow,
    sqrt,
    exp,
    log,
    tanh,
    iota,
    reshape,
    conv,
    sort,
    stop_gradient,
    transpose,
    squeeze,
    broadcast_in_dim,
    slice,
    concatenate,
    convert_element_type,
    device_put,
)

# from jax2onnx.converter.primitives.jax.nn import (
#     sigmoid
# )
import jax.random

import contextlib
from jax2onnx.converter.primitives.jax.nn import sigmoid
from jax2onnx.onnx_builder import OnnxBuilder


# def create_example_args_with_dynamic_batch(input_shapes):
#     """
#     Creates example arguments for JAX functions, handling dynamic batch dimensions.

#     Args:
#         input_shapes: A list of shape tuples, where 'B' represents a dynamic batch dimension.

#     Returns:
#         A list of jax.ShapeDtypeStruct objects representing the example arguments.
#     """
#     example_args = []
#     for shape_tuple in input_shapes:
#         shape = []
#         for dim in shape_tuple:
#             if dim == "B":
#                 shape.append(None)  # Use None for dynamic dimension
#             else:
#                 shape.append(dim)

#         # Assuming float32 data type (adjust as needed)
#         dtype = jnp.float32

#         example_args.append(jax.ShapeDtypeStruct(tuple(shape), dtype))

#     return example_args


def save_onnx(
    fn,
    input_shapes,
    output_path="model.onnx",
    model_name="jax_model",
    include_intermediate_shapes=True,
):
    jaxpr2onnx = JaxprToOnnx()
    return jaxpr2onnx.save_onnx(
        fn,
        input_shapes,
        output_path=output_path,
        model_name=model_name,
        include_intermediate_shapes=include_intermediate_shapes,
    )


class JaxprToOnnx:
    def save_onnx(
        self,
        fn,
        input_shapes,
        output_path="model.onnx",
        model_name="jax_model",
        include_intermediate_shapes=True,
    ):

        # if input_shapes have dynamic batch dimensions then include_intermediate_shapes must be False
        if any("B" in shape for shape in input_shapes):
            include_intermediate_shapes = False
            print(
                "Dynamic batch dimensions detected. Setting include_intermediate_shapes=False"
            )

        self._validate_input_shapes(input_shapes=input_shapes)
        example_args = [
            jax.numpy.zeros(self._shape_with_example_batch(s)) for s in input_shapes
        ]

        # example_args = create_example_args_with_dynamic_batch(input_shapes)

        with temporary_monkey_patches():
            jaxpr = jax.make_jaxpr(fn)(*example_args)

        converter = Jaxpr2OnnxConverter()
        converter.trace_jaxpr(fn, example_args)

        # Set symbolic batch dimension 'B' only on corresponding input tensors
        for tensor, input_shape in zip(converter.builder.inputs, input_shapes):
            tensor_shape = tensor.type.tensor_type.shape.dim
            for idx, dim in enumerate(input_shape):
                if dim == "B":
                    tensor_shape[idx].dim_param = "B"

        # Set symbolic batch dimension 'B' on outputs if it's set on any input
        batch_dims = {
            idx for shape in input_shapes for idx, dim in enumerate(shape) if dim == "B"
        }
        for tensor in converter.builder.outputs:
            tensor_shape = tensor.type.tensor_type.shape.dim
            for idx in batch_dims:
                if idx < len(tensor_shape):
                    tensor_shape[idx].dim_param = "B"

        # Optionally include intermediate shape information
        value_info = converter.builder.value_info if include_intermediate_shapes else []

        # Remove unused initializers
        used_initializers = {i for node in converter.builder.nodes for i in node.input}
        converter.builder.initializers = [
            init
            for init in converter.builder.initializers
            if init.name in used_initializers
        ]

        graph = helper.make_graph(
            nodes=converter.builder.nodes,
            name=model_name,
            inputs=converter.builder.inputs,
            outputs=converter.builder.outputs,
            initializer=converter.builder.initializers,
            value_info=value_info,
        )

        onnx_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21)]
        )
        onnx.save_model(onnx_model, output_path)

        return output_path

    def _validate_input_shapes(self, input_shapes):
        for shape in input_shapes:
            assert isinstance(shape, tuple), "Each input shape must be a tuple"

    def _shape_with_example_batch(self, shape, example_batch=2):
        return tuple(example_batch if d == "B" else d for d in shape)


class Jaxpr2OnnxConverter:
    """
    A translator that converts JAX's JAXPR representation to ONNX format.
    """

    def __init__(self, name_counter=0):

        # Instead of duplicating helper functions, delegate to OnnxBuilder:
        self.builder = OnnxBuilder(name_counter)
        # Other converter state
        self.var_to_name: Dict[Any, str] = {}
        self.name_to_var: Dict[str, Any] = {}
        self.primitive_handlers = {
            linear_general.get_primitive(): linear_general.get_handler(self),
            add.get_primitive(): add.get_handler(self),
            mul.get_primitive(): mul.get_handler(self),
            neg.get_primitive(): neg.get_handler(self),
            sub.get_primitive(): sub.get_handler(self),
            div.get_primitive(): div.get_handler(self),
            not_.get_primitive(): not_.get_handler(self),
            eq.get_primitive(): eq.get_handler(self),
            ne.get_primitive(): ne.get_handler(self),
            lt.get_primitive(): lt.get_handler(self),
            gt.get_primitive(): gt.get_handler(self),
            max.get_primitive(): max.get_handler(self),
            min.get_primitive(): min.get_handler(self),
            select_n.get_primitive(): select_n.get_handler(self),
            xor.get_primitive(): xor.get_handler(self),
            dot_general.get_primitive(): dot_general.get_handler(self),
            reduce_sum.get_primitive(): reduce_sum.get_handler(self),
            reduce_max.get_primitive(): reduce_max.get_handler(self),
            reduce_min.get_primitive(): reduce_min.get_handler(self),
            and_.get_primitive(): and_.get_handler(self),
            or_.get_primitive(): or_.get_handler(self),
            gather.get_primitive(): gather.get_handler(self),
            scatter_add.get_primitive(): scatter_add.get_handler(self),
            argmax.get_primitive(): argmax.get_handler(self),
            argmin.get_primitive(): argmin.get_handler(self),
            square.get_primitive(): square.get_handler(self),
            integer_pow.get_primitive(): integer_pow.get_handler(self),
            sqrt.get_primitive(): sqrt.get_handler(self),
            exp.get_primitive(): exp.get_handler(self),
            log.get_primitive(): log.get_handler(self),
            tanh.get_primitive(): tanh.get_handler(self),
            #  sigmoid.get_primitive(): sigmoid.get_handler(self),
            iota.get_primitive(): iota.get_handler(self),
            reshape.get_primitive(): reshape.get_handler(self),
            conv.get_primitive(): conv.get_handler(self),
            sort.get_primitive(): sort.get_handler(self),
            stop_gradient.get_primitive(): stop_gradient.get_handler(self),
            transpose.get_primitive(): transpose.get_handler(self),
            squeeze.get_primitive(): squeeze.get_handler(self),
            broadcast_in_dim.get_primitive(): broadcast_in_dim.get_handler(self),
            slice.get_primitive(): slice.get_handler(self),
            concatenate.get_primitive(): concatenate.get_handler(self),
            convert_element_type.get_primitive(): convert_element_type.get_handler(
                self
            ),
            device_put.get_primitive(): device_put.get_handler(self),
            random_gamma.get_primitive(): random_gamma.get_handler(self),
        }

    def add_node(self, node):
        self.builder.add_node(node)

    def get_unique_name(self, prefix="node"):
        return self.builder.get_unique_name(prefix)

    def get_var_name(self, var):
        if var not in self.var_to_name:
            self.var_to_name[var] = self.get_unique_name(f"var")
        return self.var_to_name[var]

    def get_constant_name(self, val):
        return self.builder.get_constant_name(val)

    def add_input(self, var, shape, dtype=np.float32):
        name = self.get_var_name(var)
        self.builder.add_input(name, shape, dtype)
        return name

    def add_output(self, var, shape, dtype=np.float32):
        name = self.get_var_name(var)
        self.builder.add_output(name, shape, dtype)
        return name

    def add_intermediate_from_name(self, name, shape, dtype=np.float32):
        self.builder.add_value_info(name, shape, dtype)
        return name

    def get_name(self, var):
        if isinstance(var, jax._src.core.Var):
            return self.get_var_name(var)
        elif isinstance(var, jax._src.core.Literal):
            return self.get_constant_name(var)
        else:
            raise NotImplementedError("not yet implemented")

    def finalize_model(self, output_path, model_name):
        graph = self.builder.create_graph(model_name)
        onnx_model = self.builder.create_model(graph)
        onnx.save_model(onnx_model, output_path)
        return output_path

    def _handle_identity(self, node_inputs, node_outputs, params):
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Identity",
            input_names,
            [output_name],
            name=self.get_unique_name("identity"),
        )
        self.builder.add_node(node)

    def _handle_stop_gradient(self, node_inputs, node_outputs, params):
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Identity",
            input_names,
            [output_name],
            name=self.get_unique_name("stop_gradient"),
        )
        self.builder.add_node(node)

    def _handle_random_seed(self, node_inputs, node_outputs, params):
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Identity",
            input_names,
            [output_name],
            name=self.get_unique_name("random_seed"),
        )
        self.builder.add_node(node)

    def _handle_random_wrap(self, node_inputs, node_outputs, params):
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Identity",
            input_names,
            [output_name],
            name=self.get_unique_name("random_wrap"),
        )
        self.builder.add_node(node)

    def _handle_random_split(self, node_inputs, node_outputs, params):
        input_name = self.get_name(node_inputs[0])
        intermediate = self.get_unique_name("random_split:x")
        output_name = self.get_var_name(node_outputs[0])

        reshape = self.get_constant_name(np.array([1, 2], dtype=np.int64))

        num = params["shape"][0]
        repeat = self.get_constant_name(np.array([num, 1], dtype=np.int64))

        node_1 = self.builder.create_node(
            "Reshape",
            [input_name, reshape],
            [intermediate],
            name=self.get_unique_name("random_split:reshape"),
        )
        self.builder.add_node(node_1)

        node_2 = self.builder.create_node(
            "Tile",
            [intermediate, repeat],
            [output_name],
            name=self.get_unique_name("random_split:tile"),
        )
        self.builder.add_node(node_2)

    def _handle_random_unwrap(self, node_inputs, node_outputs, params):
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Identity",
            input_names,
            [output_name],
            name=self.get_unique_name("random_wrap"),
        )
        self.builder.add_node(node)

    def _handle_random_fold_in(self, node_inputs, node_outputs, params):
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Identity",
            [input_name],
            [output_name],
            name=self.get_unique_name("random_fold_in"),
        )
        self.builder.add_node(node)

    def _handle_not(self, node_inputs, node_outputs, params):
        """Handle JAX not primitive."""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])
        node = self.builder.create_node(
            "Not",
            input_names,
            [output_name],
            name=self.get_unique_name("not"),
        )
        self.builder.add_node(node)

    def _handle_add(self, node_inputs, node_outputs, params):
        """Handle JAX add primitive."""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])
        node = self.builder.create_node(
            "Add",
            input_names,
            [output_name],
            name=self.get_unique_name("add"),
        )
        self.builder.add_node(node)

    def _handle_mul(self, node_inputs, node_outputs, params):
        """Handle JAX mul primitive."""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])
        node = self.builder.create_node(
            "Mul",
            input_names,
            [output_name],
            name=self.get_unique_name("mul"),
        )
        self.builder.add_node(node)

    def _handle_sub(self, node_inputs, node_outputs, params):
        """Handle JAX sub primitive."""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])
        node = self.builder.create_node(
            "Sub",
            input_names,
            [output_name],
            name=self.get_unique_name("sub"),
        )
        self.builder.add_node(node)

    def _handle_div(self, node_inputs, node_outputs, params):
        """Handle JAX div primitive."""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])
        node = self.builder.create_node(
            "Div",
            input_names,
            [output_name],
            name=self.get_unique_name("div"),
        )
        self.builder.add_node(node)

    def _handle_eq(self, node_inputs, node_outputs, params):
        """Handle JAX eq primitive"""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])
        node = self.builder.create_node(
            "Equal",
            input_names,
            [output_name],
            name=self.get_unique_name("eq"),
        )
        self.builder.add_node(node)

    def _handle_ne(self, node_inputs, node_outputs, params):
        """Handle JAX ne primitive"""
        input_names = [self.get_name(inp) for inp in node_inputs]
        eq_output = self.get_unique_name("equal_output")
        output_name = self.get_var_name(node_outputs[0])
        node_1 = self.builder.create_node(
            "Equal",
            input_names,
            [eq_output],
            name=self.get_unique_name("ne_eq"),
        )
        self.builder.add_node(node_1)

        node_2 = self.builder.create_node(
            "Not",
            [eq_output],
            [output_name],
            name=self.get_unique_name("ne_not"),
        )
        self.builder.add_node(node_2)

    def _handle_and(self, node_inputs, node_outputs, params):
        """Handle JAX and primitive."""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])
        node = self.builder.create_node(
            "And",
            input_names,
            [output_name],
            name=self.get_unique_name("and"),
        )
        self.builder.add_node(node)

    def _handle_or(self, node_inputs, node_outputs, params):
        """Handle JAX or primitive."""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])
        node = self.builder.create_node(
            "Or",
            input_names,
            [output_name],
            name=self.get_unique_name("or"),
        )
        self.builder.add_node(node)

    def _handle_xor(self, node_inputs, node_outputs, params):
        """Handle JAX xor primitive."""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])
        node = self.builder.create_node(
            "Xor",
            input_names,
            [output_name],
            name=self.get_unique_name("xor"),
        )
        self.builder.add_node(node)

    def _handle_lt(self, node_inputs, node_outputs, params):
        """Handle JAX lt primitive"""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])
        node = self.builder.create_node(
            "Less",
            input_names,
            [output_name],
            name=self.get_unique_name("less"),
        )
        self.builder.add_node(node)

    def _handle_gt(self, node_inputs, node_outputs, params):
        """Handle JAX gt primitive"""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])
        node = self.builder.create_node(
            "Greater",
            input_names,
            [output_name],
            name=self.get_unique_name("greater"),
        )
        self.builder.add_node(node)

    def _handle_max(self, node_inputs, node_outputs, params):
        """Handle JAX max primitive."""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Max",
            input_names,
            [output_name],
            name=self.get_unique_name("max"),
        )
        self.builder.add_node(node)

    def _handle_min(self, node_inputs, node_outputs, params):
        """Handle JAX min primitive."""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Min",
            input_names,
            [output_name],
            name=self.get_unique_name("min"),
        )
        self.builder.add_node(node)

    def _handle_select_n(self, node_inputs, node_outputs, params):
        """Handle JAX select_n primitive."""
        condition_name = self.get_name(node_inputs[0])
        false_name = self.get_name(node_inputs[1])
        true_name = self.get_name(node_inputs[2])
        output_name = self.get_var_name(node_outputs[0])
        node = self.builder.create_node(
            "Where",
            [condition_name, true_name, false_name],
            [output_name],
            name=self.get_unique_name("where"),
        )
        self.builder.add_node(node)

    def _handle_dot_general(self, node_inputs, node_outputs, params):
        """Handle JAX dot_general primitive with a reshape-Gemm-reshape pattern."""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])

        # Extract dot_general parameters
        dimension_numbers = params["dimension_numbers"]
        ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers

        lhs_name, rhs_name = input_names
        lhs_shape = node_inputs[0].aval.shape
        rhs_shape = node_inputs[1].aval.shape
        output_shape = node_outputs[0].aval.shape

        # Compute batch and feature dimensions
        batch_size = np.prod(
            lhs_shape[: len(lhs_shape) - len(lhs_contract)], dtype=np.int64
        )
        feature_size = np.prod(
            lhs_shape[len(lhs_shape) - len(lhs_contract) :], dtype=np.int64
        )
        rhs_output_size = np.prod(
            rhs_shape[len(rhs_shape) - len(rhs_contract) :], dtype=np.int64
        )

        lhs_reshape_name = self.get_unique_name("reshape_input")
        lhs_reshape_node = self.builder.create_node(
            "Reshape",
            [
                lhs_name,
                self.get_constant_name(
                    np.array([feature_size, rhs_output_size], dtype=np.int64)
                ),
            ],
            [lhs_reshape_name],
            name=self.get_unique_name("reshape_lhs"),
        )
        self.builder.add_node(lhs_reshape_node)

        rhs_reshape_name = self.get_unique_name("rhs_reshape")
        rhs_reshape_node = self.builder.create_node(
            "Reshape",
            [
                rhs_name,
                self.get_constant_name(
                    np.array([feature_size, rhs_output_size], dtype=np.int64)
                ),
            ],
            [rhs_reshape_name],
            name=self.get_unique_name("reshape_rhs"),
        )
        self.builder.add_node(rhs_reshape_node)

        # Perform Gemm (General Matrix Multiplication)
        gemm_output_name = self.get_unique_name("gemm_output")
        gemm_node = self.builder.create_node(
            "Gemm",
            [lhs_reshape_name, rhs_reshape_name],  # B = Weights
            [gemm_output_name],
            name=self.get_unique_name("gemm"),
            alpha=1.0,  # Standard scaling factor
            beta=1.0,  # Include bias term
            transB=0,  # Ensure no transpose on B
        )
        self.builder.add_node(gemm_node)

        # Second Reshape: Restore original batch dimensions and output shape
        reshape_output_node = self.builder.create_node(
            "Reshape",
            [
                gemm_output_name,
                self.get_constant_name(np.array(output_shape, dtype=np.int64)),
            ],
            [output_name],
            name=self.get_unique_name("reshape_output"),
        )
        self.builder.add_node(reshape_output_node)

    def _handle_reduce_sum(self, node_inputs, node_outputs, params):
        """Handle JAX reduce_sum primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        # Get axes and create constant for it
        axes = params["axes"]
        axes_name = self.get_constant_name(np.array(axes, dtype=np.int64))

        # Create ReduceSum node
        node = self.builder.create_node(
            "ReduceSum",
            [input_name, axes_name],
            [output_name],
            name=self.get_unique_name("reduce_sum"),
            keepdims=0 if not params.get("keepdims", False) else 1,
        )
        self.builder.add_node(node)

    def _handle_reduce_max(self, node_inputs, node_outputs, params):
        """Handle JAX reduce_max primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        # Get axes and create constant for it
        axes = params["axes"]
        axes_name = self.get_constant_name(np.array(axes, dtype=np.int64))

        # Create ReduceMax node
        node = self.builder.create_node(
            "ReduceMax",
            [input_name, axes_name],
            [output_name],
            name=self.get_unique_name("reduce_max"),
            keepdims=0 if not params.get("keepdims", False) else 1,
        )
        self.builder.add_node(node)

    def _handle_reduce_min(self, node_inputs, node_outputs, params):
        """Handle JAX reduce_min primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.__get_var_name(node_outputs[0])

        # Get axes and create constant for it
        axes = params["axes"]
        axes_name = self.get_constant_name(np.array(axes, dtype=np.int64))

        # Create ReduceMin node
        node = self.builder.create_node(
            "ReduceMin",
            [input_name, axes_name],
            [output_name],
            name=self.get_unique_name("reduce_min"),
            keepdims=0 if not params.get("keepdims", False) else 1,
        )
        self.builder.add_node(node)

    def _handle_gather(self, node_inputs, node_outputs, params):
        input_names = [self.get_name(imp) for imp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "GatherElements",
            input_names,
            [output_name],
            name=self.get_unique_name("gather"),
        )

        self.builder.add_node(node)

    def _handle_scatter_add(self, node_inputs, node_outputs, params):
        input_names = [self.get_name(imp) for imp in node_inputs]
        intermediate = self.get_unique_name("scatter_add:x")
        output_name = self.get_var_name(node_outputs[0])
        print("SCATTER-ADD")
        print(node_inputs)
        print(node_outputs)
        print(params)

        node_1 = self.builder.create_node(
            "Cast",
            [input_names[1]],
            [intermediate],
            to=TensorProto.INT64,
        )
        self.builder.add_node(node_1)

        node_2 = self.builder.create_node(
            "ScatterND",
            [input_names[0], intermediate, input_names[2]],
            [output_name],
            name=self.get_unique_name("scatter_add"),
            reduction="add",
        )

        self.builder.add_node(node_2)

    def _handle_argmax(self, node_inputs, node_outputs, params):
        """Handle JAX argmax primitive."""
        input_name = self.get_name(node_inputs[0])
        intermediate_name = self.get_unique_name("argmax_intermediate")
        output_name = self.get_var_name(node_outputs[0])

        axis = params["axes"][0]
        index_dtype = params["index_dtype"]
        keepdims = 1 if "keepdims" in params else 0

        node_1 = self.builder.create_node(
            "ArgMax",
            [input_name],
            [intermediate_name],
            name=self.get_unique_name("argmax"),
            axis=axis,
            keepdims=keepdims,
        )
        self.builder.add_node(node_1)

        # Slight quirk in ONNX: argmax returns int64, but we can always(?) cast to int32.

        node_2 = self.builder.create_node(
            "Cast",
            [intermediate_name],
            [output_name],
            to=TensorProto.INT32,
        )
        self.builder.add_node(node_2)

    def _handle_argmin(self, node_inputs, node_outputs, params):
        """Handle JAX argmin primitive."""
        input_name = self.get_name(node_inputs[0])
        intermediate_name = self.get_unique_name("argmax_intermediate")
        output_name = self.get_var_name(node_outputs[0])

        axis = params["axes"][0]
        index_dtype = params["index_dtype"]
        keepdims = params["keepdims"]
        node_1 = self.builder.create_node(
            "ArgMin",
            [input_name],
            [intermediate_name],
            name=self.get_unique_name("argmin"),
            axis=axis,
            keepdims=keepdims,
        )

        self.builder.add_node(node_1)

        # Slight quirk in ONNX: argmin returns int64, but we can always(?) cast to int32.

        node_2 = self.builder.create_node(
            "Cast",
            [intermediate_name],
            [output_name],
            to=TensorProto.INT32,
        )
        self.builder.add_node(node_2)

    def _handle_square(self, node_inputs, node_outputs, params):
        """Handle JAX square primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        power_name = self.get_constant_name(np.array(2, dtype=np.int32))

        node = self.builder.create_node(
            "Pow",
            [input_name, power_name],
            [output_name],
            name=self.get_unique_name("square"),
        )

        self.builder.add_node(node)

    def _handle_integer_pow(self, node_inputs, node_outputs, params):
        """Handle JAX integer pow primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        power_name = self.get_constant_name(np.array(params["y"], dtype=np.int32))

        node = self.builder.create_node(
            "Pow",
            [input_name, power_name],
            [output_name],
            name=self.get_unique_name("square"),
        )

        self.builder.add_node(node)

    def _handle_sqrt(self, node_inputs, node_outputs, params):
        """Handle JAX sqrt primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Sqrt",
            [input_name],
            [output_name],
            name=self.get_unique_name("sqrt"),
        )
        self.builder.add_node(node)

    def _handle_exp(self, node_inputs, node_outputs, params):
        """Handle JAX exp primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Exp",
            [input_name],
            [output_name],
            name=self.get_unique_name("exp"),
        )
        self.builder.add_node(node)

    def _handle_log(self, node_inputs, node_outputs, params):
        """Handle JAX log primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Log",
            [input_name],
            [output_name],
            name=self.get_unique_name("log"),
        )
        self.builder.add_node(node)

    def _handle_tanh(self, node_inputs, node_outputs, params):
        """Handle JAX tanh primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Tanh",
            [input_name],
            [output_name],
            name=self.get_unique_name("tanh"),
        )
        self.builder.add_node(node)

    def _handle_sigmoid(self, node_inputs, node_outputs, params):
        """Handle JAX sigmoid primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Sigmoid",
            [input_name],
            [output_name],
            name=self.get_unique_name("sigmoid"),
        )
        self.builder.add_node(node)

    def _handle_iota(self, node_inputs, node_outputs, params):
        """Handle JAX iota primitive."""
        output_name = self.get_var_name(node_outputs[0])

        dtype = params["dtype"]  # TODO: Use dtype
        if dtype != jnp.int32:
            raise NotImplementedError("dtype not implemented")
        shape = params["shape"]

        L = shape[0]  # TODO: consider when len(shape) > 1
        start_name = self.get_constant_name(np.array(0, dtype=np.int32))
        end_name = self.get_constant_name(np.array(L, dtype=np.int32))
        step_name = self.get_constant_name(np.array(1, dtype=np.int32))

        node = self.builder.create_node(
            "Range",
            [start_name, end_name, step_name],
            [output_name],
            name=self.get_unique_name("iota"),
        )
        self.builder.add_node(node)

    def _handle_reshape(self, node_inputs, node_outputs, params):
        """Handle JAX reshape primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        # Get new shape and create constant for it
        new_shape = params["new_sizes"]

        # Detect if reshape is redundant for bias broadcasting
        input_shape = node_inputs[0].aval.shape
        if len(new_shape) == 2 and new_shape[0] == 1 and input_shape == (new_shape[1],):
            # Bias reshaped for broadcasting: Skip reshape and return directly
            self.var_to_name[node_outputs[0]] = input_name
            return

        # Otherwise, keep reshape operation
        shape_name = self.get_constant_name(np.array(new_shape, dtype=np.int64))
        node = self.builder.create_node(
            "Reshape",
            [input_name, shape_name],
            [output_name],
            name=self.get_unique_name("reshape"),
        )
        self.builder.add_node(node)

    def _handle_transpose(self, node_inputs, node_outputs, params):
        """Handle JAX transpose primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        # Get permutation
        permutation = params["permutation"]

        # Create Transpose node
        node = self.builder.create_node(
            "Transpose",
            [input_name],
            [output_name],
            name=self.get_unique_name("transpose"),
            perm=permutation,
        )
        self.builder.add_node(node)

    def _handle_squeeze(self, node_inputs, node_outputs, params):
        """Handle JAX squeeze primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        # Get permutation
        dims = params["dimensions"]
        axes = self.get_constant_name(np.array(dims, dtype=np.int64))

        # Create Transpose node
        node = self.builder.create_node(
            "Squeeze",
            [input_name, axes],
            [output_name],
            name=self.get_unique_name("squeeze"),
        )
        self.builder.add_node(node)

    def _handle_broadcast_in_dim(self, node_inputs, node_outputs, params):
        """Handle JAX broadcast_in_dim primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        # Get broadcast dimensions and shape
        broadcast_dimensions = params["broadcast_dimensions"]
        shape = params["shape"]

        # Create constants for shape
        shape_name = self.get_constant_name(np.array(shape, dtype=np.int64))

        # ONNX doesn't have a direct equivalent to broadcast_in_dim
        # We'll use a combination of Reshape and Expand

        # First reshape to add singleton dimensions
        reshape_output = self.get_unique_name("reshape_output")
        reshape_shape = []
        idx = 0
        for i in range(len(shape)):
            if i in broadcast_dimensions:
                reshape_shape.append(
                    1
                    if idx >= len(node_inputs[0].aval.shape)
                    else node_inputs[0].aval.shape[idx]
                )
                idx += 1
            else:
                reshape_shape.append(1)

        reshape_shape_name = self.get_constant_name(
            np.array(reshape_shape, dtype=np.int64)
        )

        reshape_node = self.builder.create_node(
            "Reshape",
            [input_name, reshape_shape_name],
            [reshape_output],
            name=self.get_unique_name("reshape_for_broadcast"),
        )
        self.builder.add_node(reshape_node)

        # Then expand to target shape
        expand_node = self.builder.create_node(
            "Expand",
            [reshape_output, shape_name],
            [output_name],
            name=self.get_unique_name("expand"),
        )
        self.builder.add_node(expand_node)

    def _handle_slice(self, node_inputs, node_outputs, params):
        """Handle JAX slice primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        # Get slice parameters
        start_indices = params["start_indices"]
        starts_name = self.get_constant_name(np.array(start_indices, dtype=np.int64))
        limit_indices = params["limit_indices"]
        ends_name = self.get_constant_name(np.array(limit_indices, dtype=np.int64))
        axes_name = self.get_constant_name(
            np.array(list(range(len(start_indices))), dtype=np.int64)
        )
        inputs = [input_name, starts_name, ends_name, axes_name]

        if "strides" in params and params["strides"]:
            strides = params["strides"]
            steps_name = self.get_constant_name(np.array(strides, dtype=np.int64))
            inputs.append(steps_name)

        # Create Slice node
        node = self.builder.create_node(
            "Slice",
            inputs,
            [output_name],
            name=self.get_unique_name("slice"),
        )
        self.builder.add_node(node)

    def _handle_concatenate(self, node_inputs, node_outputs, params):
        """Handle JAX concatenate primitive."""
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])

        # Get concatenation axis
        dimension = params["dimension"]

        # Create Concat node
        node = self.builder.create_node(
            "Concat",
            input_names,
            [output_name],
            name=self.get_unique_name("concat"),
            axis=dimension,
        )
        self.builder.add_node(node)

    def _handle_conv(self, node_inputs, node_outputs, params):
        """Handle JAX conv_general_dilated primitive."""
        # This is a simplified implementation for common cases
        input_name = self.get_name(node_inputs[0])  # input
        filter_name = self.get_name(node_inputs[1])  # weights
        output_name = self.get_var_name(node_outputs[0])

        # Extract parameters
        dimension_numbers = params["dimension_numbers"]
        window_strides = params["window_strides"]
        padding = params["padding"]
        lhs_dilation = params.get("lhs_dilation", (1,) * (len(window_strides)))
        rhs_dilation = params.get("rhs_dilation", (1,) * (len(window_strides)))

        # Parse dimension numbers
        lhs_spec, rhs_spec, out_spec = dimension_numbers

        # ONNX Conv expects specific dimension ordering
        # N=batch, C=channel, D/H/W=spatial dims

        # Simple case: assume standard dimension ordering
        # This is highly simplified and won't work for all JAX conv cases
        node = self.builder.create_node(
            "Conv",
            [input_name, filter_name],
            [output_name],
            name=self.get_unique_name("conv"),
            kernel_shape=node_inputs[1].aval.shape[2:],
            strides=window_strides,
            dilations=rhs_dilation,
            pads=sum(
                padding, ()
            ),  # Flatten [(p0, p0), (p1, p1), ...] to [p0, p0, p1, p1, ...]
        )
        self.builder.add_node(node)

    def _handle_max_pool(self, node_inputs, node_outputs, params):
        """Handle JAX max_pool primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        # Extract parameters
        window_dimensions = params["window_dimensions"]
        window_strides = params["window_strides"]
        padding = params["padding"]

        # Create MaxPool node
        node = self.builder.create_node(
            "MaxPool",
            [input_name],
            [output_name],
            name=self.get_unique_name("maxpool"),
            kernel_shape=window_dimensions[2:],
            strides=window_strides,
            pads=sum(padding, ()),  # Flatten padding
        )
        self.builder.add_node(node)

    def _handle_avg_pool(self, node_inputs, node_outputs, params):
        """Handle JAX avg_pool primitive."""
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        # Extract parameters
        window_dimensions = params["window_dimensions"]
        window_strides = params["window_strides"]
        padding = params["padding"]

        # Create AveragePool node
        node = self.builder.create_node(
            "AveragePool",
            [input_name],
            [output_name],
            name=self.get_unique_name("avgpool"),
            kernel_shape=window_dimensions[2:],
            strides=window_strides,
            pads=sum(padding, ()),  # Flatten padding
        )
        self.builder.add_node(node)

    def _handle_sort(self, node_inputs, node_outputs, params):
        """Handle JAX sort primitive"""
        input_name = self.get_name(node_inputs[0])
        shape_name = self.get_unique_name("sort_shape")
        value_name = self.get_var_name(node_outputs[0])
        indices_name = self.get_unique_name("sort_indices_output")

        if "axis" in params:
            axis = params["axis"]
            K = node_inputs[0].aval.shape[axis]
            raise NotImplementedError("sort axis not supported yet")
        else:
            node = self.builder.create_node(
                "Shape",
                [input_name],
                [shape_name],
                name=self.get_unique_name("shape"),
            )
            self.builder.add_node(node)

        # to make sort more generic, we first find the shape
        node = self.builder.create_node(
            "TopK",
            [input_name, shape_name],
            [value_name, indices_name],
            name=self.get_unique_name("sort"),
            largest=0,
        )

        self.builder.add_node(node)

    def _handle_random_uniform(self, node_inputs, node_outputs, params):
        output_name = self.get_var_name(node_outputs[0])
        shape = node_outputs[0].aval.shape
        if shape == ():
            shape = (1,)
        node = self.builder.create_node(
            "RandomUniform",
            [],
            [output_name],
            name=self.get_unique_name("random_uniform"),
            shape=shape,
        )
        self.builder.add_node(node)

    def _handle_random_normal(self, node_inputs, node_outputs, params):
        output_name = self.get_var_name(node_outputs[0])
        shape = node_outputs[0].aval.shape
        if shape == ():
            shape = (1,)
        node = self.builder.create_node(
            "RandomNormal",
            [],
            [output_name],
            name=self.get_unique_name("random_normal"),
            shape=shape,
        )
        self.builder.add_node(node)

    def _handle_random_gamma(self, node_inputs, node_outputs, params):
        """
        Handle JAX gamma primitive

        between Marsaglia-Tang and Cheng, we decided on the former due to the low rejection rate
        https://kth.diva-portal.org/smash/get/diva2:1935824/FULLTEXT02.pdf

        d = α - 1/3
        c = 1/sqrt(9d)

        repeat
            sample Z ~ Normal(0,1)
            V = (1 + cZ)^3
            sample U ~ Uniform(0,1)
            X = dV
            if V > 0 and log(U) < 1/2 Z^2 + d - dV + dlog(V) then
                accept X
            endif

        until X is accepted
        return X
        """
        # Create a jaxpr and run JaxprToOnnx to build the CG

        shape = node_inputs[1].aval.shape
        key = jax.random.key(0)
        alpha = jnp.zeros(shape)

        # TODO: Case 0 < alpha <= 1/3 not handled
        subconverter = JaxprToOnnx(self.name_counter + 1)
        if "log_space" in params and params["log_space"]:
            subconverter.trace_jaxpr(gamma_log, (key, alpha))
        else:
            subconverter.trace_jaxpr(gamma, (key, alpha))

        # connect inputs/outputs to outer jaxpr
        nodes = subconverter.nodes
        initializers = subconverter.initializers
        inputs = subconverter.inputs
        outputs = subconverter.outputs

        assert len(node_inputs) == len(inputs)
        assert len(node_outputs) == len(outputs)

        for o_invar, i_invar in zip(node_inputs, inputs):
            o_invar_name = self.get_name(o_invar)
            i_invar_name = i_invar.name
            node = self.builder.create_node(
                "Identity",
                [o_invar_name],
                [i_invar_name],
                name=self.get_unique_name("gamma_input"),
            )
            self.builder.add_node(node)

        self.builder.add_nodes(nodes)
        self.builder.add_initializers(initializers)
        self.name_counter += subconverter.name_counter - subconverter._name_counter_init

        for o_outvar, i_outvar in zip(node_outputs, outputs):
            o_outvar_name = self.get_name(o_outvar)
            i_outvar_name = i_outvar.name
            node = self.builder.create_node(
                "Identity",
                [i_outvar_name],
                [o_outvar_name],
                name=self.get_unique_name("gamma_output"),
            )
            self.builder.add_node(node)

    def _handle_convert_element_type(self, node_inputs, node_outputs, params):
        input_names = [self.get_name(inp) for inp in node_inputs]
        output_name = self.get_var_name(node_outputs[0])

        new_dtype = self.builder.numpy_dtype_to_onnx(params["new_dtype"])
        node = self.builder.create_node(
            "Cast",
            input_names,
            [output_name],
            name=self.get_unique_name("convert_element_type"),
            to=new_dtype,
        )
        self.builder.add_node(node)

    def _handle_device_put(self, node_inputs, node_outputs, params):
        name = self.get_unique_name("const")
        # Convert to numpy and create tensor
        val = node_inputs[0]
        actual_val = val.val

        np_val = np.array(actual_val)
        if np_val.dtype == np.int64:
            np_val = np_val.astype(np.int32)
        elif np_val.dtype == np.float64:
            np_val = np_val.astype(np.float32)

        tensor = self.builder.create_tensor(
            name=name,
            data_type=self.builder.numpy_dtype_to_onnx(np_val.dtype),
            dims=np_val.shape,
            vals=np_val.flatten().tolist(),
        )
        self.builder.add_initializer(tensor)
        input_names = [name]
        output_name = self.get_var_name(node_outputs[0])

        node = self.builder.create_node(
            "Identity",
            input_names,
            [output_name],
            name=self.get_unique_name("device_put"),
        )
        self.builder.add_node(node)

    def _handle_truncated_normal(self, node_inputs, node_outputs, params):
        output_name = self.get_var_name(node_outputs[0])
        shape = node_outputs[0].aval.shape
        if shape == ():
            shape = (1,)
        node = self.builder.create_node(
            "RandomNormal",
            [],
            [output_name],
            name=self.get_unique_name("truncated_normal"),
            shape=shape,
        )
        self.builder.add_node(node)

    def _process_pjit(self, jaxpr):
        closed_jaxpr = jaxpr.params["jaxpr"]
        if not isinstance(closed_jaxpr, jax._src.core.ClosedJaxpr):
            raise ValueError("Expected ClosedJaxpr in pjit.param[jaxpr]")

        name = jaxpr.params["name"]
        if name == "_normal":
            self._handle_random_normal(jaxpr.invars, jaxpr.outvars, jaxpr.params)
        elif name == "_uniform":
            self._handle_random_uniform(jaxpr.invars, jaxpr.outvars, jaxpr.params)
        elif name == "_gamma":
            self._process_closed_jaxpr(jaxpr)
        elif name == "clip":
            self._process_closed_jaxpr(jaxpr)
        elif name == "sort":
            self._process_closed_jaxpr(jaxpr)
        elif name == "_where":
            self._process_closed_jaxpr(jaxpr)
        elif name == "_gumbel":
            self._process_closed_jaxpr(jaxpr)
        elif name == "_dirichlet":
            self._process_closed_jaxpr(jaxpr)
        elif name == "_truncated_normal":
            self._handle_truncated_normal(jaxpr.invars, jaxpr.outvars, jaxpr.params)
        else:
            raise NotImplementedError(f"pjit {jaxpr.params['name']} not yet handled")

    def _process_eqn(self, jaxpr):
        """Process a single JAXPR equation."""
        if hasattr(jaxpr, "primitive"):
            primitive = jaxpr.primitive
            if primitive.name == "pjit":
                self._process_pjit(jaxpr)
            elif primitive in self.primitive_handlers:
                self.primitive_handlers[primitive](
                    jaxpr.invars, jaxpr.outvars, jaxpr.params
                )
            else:
                raise NotImplementedError(f"Primitive {primitive} not implemented")
        else:
            # Handle call primitives or other special cases
            raise NotImplementedError(f"Non-primitive equation: {jaxpr}")

    def _process_closed_jaxpr(self, jaxpr):
        # TODO: CONFUSING, `jaxpr` is a JaxprEqn which contains the ClosedJaxpr
        assert isinstance(jaxpr, jax._src.core.JaxprEqn)

        closed_jaxpr = jaxpr.params["jaxpr"]
        node_inputs = jaxpr.invars
        node_outputs = jaxpr.outvars

        subconverter = JaxprToOnnx(self.name_counter + 1)
        subconverter._process_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts)

        nodes = subconverter.nodes
        initializers = subconverter.initializers
        inputs = subconverter.inputs
        outputs = subconverter.outputs

        assert len(node_inputs) == len(inputs)
        assert len(node_outputs) == len(outputs)

        for o_invar, i_invar in zip(node_inputs, inputs):
            o_invar_name = self.get_name(o_invar)
            i_invar_name = i_invar.name
            node = self.builder.create_node(
                "Identity",
                [o_invar_name],
                [i_invar_name],
                name=self.get_unique_name("pjit_input"),
            )
            self.builder.add_node(node)

        self.builder.add_nodes(nodes)
        self.builder.add_initializers(initializers)
        self.name_counter += subconverter.name_counter - subconverter._name_counter_init

        for o_outvar, i_outvar in zip(node_outputs, outputs):
            o_outvar_name = self.get_name(o_outvar)
            i_outvar_name = i_outvar.name
            node = self.builder.create_node(
                "Identity",
                [i_outvar_name],
                [o_outvar_name],
                name=self.get_unique_name("pjit_output"),
            )
            self.builder.add_node(node)

    def _process_jaxpr(self, jaxpr, consts):
        """Process a JAXPR and convert it to ONNX nodes."""

        # Setup inputs
        for var in jaxpr.invars:
            self.add_input(var, var.aval.shape, var.aval.dtype)

        # Setup constants
        for i, const in enumerate(consts):
            const_name = self.get_constant_name(const)
            const_var = jaxpr.constvars[i]
            self.var_to_name[const_var] = const_name
            self.name_to_const[const_name] = const

        # Process all equations in the JAXPR
        for eqn in jaxpr.eqns:
            self._process_eqn(eqn)

        # Setup outputs
        for var in jaxpr.outvars:
            self.add_output(var, var.aval.shape, var.aval.dtype)

    def trace_jaxpr(self, fn, example_args):
        # Reset state
        self.builder.reset()
        self.var_to_name = {}
        self.name_to_const = {}

        # Get JAXPR from the function
        with temporary_monkey_patches():
            closed_jaxpr = jax.make_jaxpr(fn)(*example_args)

        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.consts
        self._process_jaxpr(jaxpr, consts)

    def convert(
        self, fn, example_args, output_path="model.onnx", model_name="jax_model"
    ):
        """
        Convert a JAX function to ONNX.

        Args:
            fn: JAX function to convert
            example_args: Example input arguments to trace the function
            output_path: Path to save the ONNX model

        Returns:
            Path to the saved ONNX model
        """

        self.trace_jaxpr(fn, example_args)

        # Remove unused initializers
        used_initializers = {i for node in self.builder.nodes for i in node.input}
        self.builder.initializers = [
            init for init in self.builder.initializers if init.name in used_initializers
        ]

        graph = self.builder.create_graph(model_name)

        # Create ONNX model
        onnx_model = self.builder.create_model(graph)

        # Save model
        onnx.save_model(onnx_model, output_path)
        return output_path


def gamma(key, alpha):
    d = alpha - 1 / 3
    c = 1 / jnp.sqrt(9 * d)
    z = jax.random.normal(key, alpha.shape)
    v = (1 + c * z) ** 3
    u = jax.random.uniform(key, alpha.shape)
    x = d * v

    acceptance = (v > 0) & (jnp.log(u) < (0.5 * z**2 + d - d * v + d * jnp.log(v)))

    z = jax.random.normal(key, alpha.shape)
    v = (1 + c * z) ** 3
    x = jnp.where(acceptance, x, d * v)

    # clip when alpha = 0
    x = jnp.where(alpha == 0, 0.0, x)

    return x


def gamma_log(key, alpha):
    x = gamma(key, alpha)
    return jnp.log(x)


@contextlib.contextmanager
def temporary_monkey_patches():
    with contextlib.ExitStack() as stack:
        # Enter the monkey patch from linear_general
        stack.enter_context(linear_general.temporary_patch())

        yield
