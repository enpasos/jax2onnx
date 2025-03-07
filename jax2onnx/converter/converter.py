# jax2onnx/converter/converter.py

import jax
import jax.numpy as jnp
import onnx
from onnx import helper, TensorProto
import numpy as np
from typing import Dict, Any
from jax2onnx.converter.plugins.plugin_registry import get_all_plugins
from jax2onnx.converter.plugins.jax.random import random_gamma
from jax2onnx.converter.plugins.flax.nnx import linear_general
from jax2onnx.converter.plugins.jax.lax import (
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
from jax2onnx.converter.plugins.jax.nn import sigmoid
from jax2onnx.onnx_builder import OnnxBuilder


def save_onnx(
    fn: Any,
    input_shapes: Any,
    output_path: str = "model.onnx",
    model_name: str = "jax_model",
    include_intermediate_shapes: bool = True,
) -> str:
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
        fn: Any,
        input_shapes: Any,
        output_path: str = "model.onnx",
        model_name: str = "jax_model",
        include_intermediate_shapes: bool = True,
    ) -> str:

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
        self.primitive_handlers = {}
        for primitive, plugin in get_all_plugins().items():
            handler = plugin.get_handler(self)
            self.primitive_handlers[primitive] = handler
        # self.primitive_handlers = {
        #     linear_general.get_primitive(): linear_general.get_handler(self),
        #     add.get_primitive(): add.get_handler(self),
        #     mul.get_primitive(): mul.get_handler(self),
        #     neg.get_primitive(): neg.get_handler(self),
        #     sub.get_primitive(): sub.get_handler(self),
        #     div.get_primitive(): div.get_handler(self),
        #     not_.get_primitive(): not_.get_handler(self),
        #     eq.get_primitive(): eq.get_handler(self),
        #     ne.get_primitive(): ne.get_handler(self),
        #     lt.get_primitive(): lt.get_handler(self),
        #     gt.get_primitive(): gt.get_handler(self),
        #     max.get_primitive(): max.get_handler(self),
        #     min.get_primitive(): min.get_handler(self),
        #     select_n.get_primitive(): select_n.get_handler(self),
        #     xor.get_primitive(): xor.get_handler(self),
        #     dot_general.get_primitive(): dot_general.get_handler(self),
        #     reduce_sum.get_primitive(): reduce_sum.get_handler(self),
        #     reduce_max.get_primitive(): reduce_max.get_handler(self),
        #     reduce_min.get_primitive(): reduce_min.get_handler(self),
        #     and_.get_primitive(): and_.get_handler(self),
        #     or_.get_primitive(): or_.get_handler(self),
        #     gather.get_primitive(): gather.get_handler(self),
        #     scatter_add.get_primitive(): scatter_add.get_handler(self),
        #     argmax.get_primitive(): argmax.get_handler(self),
        #     argmin.get_primitive(): argmin.get_handler(self),
        #     square.get_primitive(): square.get_handler(self),
        #     integer_pow.get_primitive(): integer_pow.get_handler(self),
        #     sqrt.get_primitive(): sqrt.get_handler(self),
        #     exp.get_primitive(): exp.get_handler(self),
        #     log.get_primitive(): log.get_handler(self),
        #     tanh.get_primitive(): tanh.get_handler(self),
        #     #  sigmoid.get_primitive(): sigmoid.get_handler(self),
        #     iota.get_primitive(): iota.get_handler(self),
        #     reshape.get_primitive(): reshape.get_handler(self),
        #     conv.get_primitive(): conv.get_handler(self),
        #     sort.get_primitive(): sort.get_handler(self),
        #     stop_gradient.get_primitive(): stop_gradient.get_handler(self),
        #     transpose.get_primitive(): transpose.get_handler(self),
        #     squeeze.get_primitive(): squeeze.get_handler(self),
        #     broadcast_in_dim.get_primitive(): broadcast_in_dim.get_handler(self),
        #     slice.get_primitive(): slice.get_handler(self),
        #     concatenate.get_primitive(): concatenate.get_handler(self),
        #     convert_element_type.get_primitive(): convert_element_type.get_handler(
        #         self
        #     ),
        #     device_put.get_primitive(): device_put.get_handler(self),
        #     random_gamma.get_primitive(): random_gamma.get_handler(self),
        # }

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
            elif primitive.name in self.primitive_handlers:
                self.primitive_handlers[primitive.name](
                    jaxpr.invars, jaxpr.outvars, jaxpr.params
                )
            else:
                raise NotImplementedError(f"Primitive {primitive.name} not implemented")
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


@contextlib.contextmanager
def temporary_monkey_patches():
    with contextlib.ExitStack() as stack:
        # Enter the monkey patch from linear_general
        stack.enter_context(linear_general.temporary_patch())

        yield
