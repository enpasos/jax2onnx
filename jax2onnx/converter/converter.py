# file: jax2onnx/converter/converter.py
from typing import Any, Dict

import jax
import jax.random
import jax.numpy as jnp
import numpy as np
import onnx
from jax.extend import core as extend_core
from onnx import helper

from jax2onnx.converter.dtype_utils import numpy_dtype_to_tensorproto
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.monkey_patch_utils import temporary_monkey_patches
from jax2onnx.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
    import_all_plugins,
)


# Updated class-level docstring to clarify purpose.
class Jaxpr2OnnxConverter:
    """
    Converts JAX's JAXPR representation to ONNX format, enabling interoperability
    between JAX and ONNX-based tools.
    """

    def __init__(self, builder: OnnxBuilder):
        # Initialize the converter with an ONNX builder instance.
        self.builder = builder

        self.params: Dict[str, Any] = {}  # Parameters for tracing
        self.call_params: Dict[str, Any] = {}  # Parameters that should be ONNX inputs

        # Mapping between variables and their names in the ONNX graph.
        self.var_to_name: dict[Any, str] = {}
        self.name_to_var: dict[str, Any] = {}

        # Handlers for JAX primitives.
        self.primitive_handlers = {}

        # Environment to track variable shapes.
        self.shape_env: dict[str, tuple[int, ...]] = {}

        # Mapping for constants in the ONNX graph.
        self.name_to_const: dict[str, Any] = {}

        # Register handlers for random primitives.
        self.primitive_handlers[jax._src.prng.random_seed_p] = self._handle_random_seed
        self.primitive_handlers[jax._src.prng.random_wrap_p] = self._handle_random_wrap
        self.primitive_handlers[jax._src.prng.random_split_p] = (
            self._handle_random_split
        )
        self.primitive_handlers[jax._src.prng.random_unwrap_p] = (
            self._handle_random_unwrap
        )

        # Import and register plugins.
        import_all_plugins()
        for key, plugin in PLUGIN_REGISTRY.items():
            if isinstance(plugin, PrimitiveLeafPlugin):
                self.primitive_handlers[key] = plugin.get_handler(self)

        for plugin in ONNX_FUNCTION_PLUGIN_REGISTRY.values():
            primitive = plugin.primitive
            self.primitive_handlers[primitive.name] = plugin.get_handler(self)

    def new_var(self, dtype: np.dtype, shape: tuple[int, ...]):
        return jax.core.Var(
            self.builder.get_unique_name(""), jax.core.ShapedArray(shape, dtype)
        )

    def add_node(self, node):
        self.builder.add_node(node)

    def get_unique_name(self, prefix="node"):
        return self.builder.get_unique_name(prefix)

    def get_var_name(self, var):
        if var not in self.var_to_name:
            name = self.get_unique_name("var")
            self.var_to_name[var] = name
            self.name_to_var[name] = var
        return self.var_to_name[var]

    def get_constant_name(self, val):
        return self.builder.get_constant_name(val)

    def add_input(self, var, shape, dtype=np.float32):
        # Add an input variable to the ONNX graph and store its shape.
        name = self.get_var_name(var)
        self.builder.add_input(name, shape, dtype)
        self.shape_env[name] = shape
        return name

    def add_output(self, var, shape, dtype=np.float32):
        # Add an output variable to the ONNX graph and store its shape.
        name = self.get_var_name(var)
        self.builder.add_output(name, shape, dtype)
        self.shape_env[name] = shape
        return name

    def add_shape_info(self, name, shape, dtype=np.float32):
        self.builder.add_value_info(name, shape, dtype)
        self.shape_env[name] = shape  # <-- store shape
        return name

    def get_name(self, var):
        if isinstance(var, jax._src.core.Var):
            return self.get_var_name(var)
        elif isinstance(var, extend_core.Literal):
            return self.get_constant_name(var)
        else:
            raise NotImplementedError("not yet implemented")

    def finalize_model(self, output_path, model_name):
        graph = self.builder.create_graph(model_name)
        onnx_model = self.builder.create_model(graph)
        onnx.save_model(onnx_model, output_path)
        return output_path

    def trace_jaxpr(self, fn, example_args, preserve_graph=False, params=None):
        # Trace a JAX function to generate its JAXPR representation.
        print(f"trace_jaxpr ... preserve_graph= {preserve_graph}")
        if not preserve_graph:
            self.builder.reset()
            self.var_to_name.clear()
            self.name_to_const.clear()
            self.shape_env.clear()

        # Simply trace the function with all parameters
        with temporary_monkey_patches(allow_function_primitives=True):
            if params is None:
                closed_jaxpr = jax.make_jaxpr(fn)(*example_args)
            else:
                closed_jaxpr = jax.make_jaxpr(fn)(*example_args, **params)

        print(closed_jaxpr)

        self.jaxpr = closed_jaxpr.jaxpr
        self.output_vars = self.jaxpr.outvars
        jaxpr, consts = self.jaxpr, closed_jaxpr.consts

        self._process_jaxpr(jaxpr, consts)

        for var in jaxpr.outvars:
            name = self.get_var_name(var)
            if name in self.builder.value_info:
                continue

            if hasattr(var, "aval"):
                shape = tuple(var.aval.shape)
                dtype = numpy_dtype_to_tensorproto(var.aval.dtype)
                self.builder.register_value_info_metadata(name, shape, dtype)
                self.builder.add_value_info(name, shape, dtype)
            else:
                raise RuntimeError(
                    f"[MissingShape] Cannot infer shape for output var {name}"
                )

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

    def add_initializer(
        self, name, vals, data_type=helper.TensorProto.INT64, dims=None
    ):
        """Add a tensor initializer to the model.

        Args:
            name: The name of the initializer
            vals: The values to initialize with
            data_type: The data type of the tensor (default: INT64)
            dims: The dimensions of the tensor (default: [len(vals)])

        Returns:
            The name of the created initializer
        """
        if dims is None:
            dims = [len(vals)]

        tensor = helper.make_tensor(
            name=name,
            data_type=data_type,
            dims=dims,
            vals=vals,
        )
        self.builder.initializers.append(tensor)
        return name

    def _handle_random_seed(self, node_inputs, node_outputs, params):
        return self._create_identity_node(node_inputs, node_outputs, "random_seed")

    def _handle_random_wrap(self, node_inputs, node_outputs, params):
        return self._create_identity_node(node_inputs, node_outputs, "random_wrap")

    def _handle_random_unwrap(self, node_inputs, node_outputs, params):
        return self._create_identity_node(node_inputs, node_outputs, "random_unwrap")

    def _handle_random_split(self, node_inputs, node_outputs, params):
        input_name = self.get_name(node_inputs[0])
        intermediate = self.get_unique_name("random_split:x")
        output_name = self.get_var_name(node_outputs[0])

        reshape = self.get_constant_name(np.array([1, 2], dtype=np.int64))

        num = params["shape"][0]
        repeat = self.get_constant_name(np.array([num, 1], dtype=np.int64))

        node_1 = helper.make_node(
            "Reshape",
            inputs=[input_name, reshape],
            outputs=[intermediate],
            name=self.get_unique_name("random_split:reshape"),
        )
        self.builder.add_node(node_1)

        node_2 = helper.make_node(
            "Tile",
            inputs=[intermediate, repeat],
            outputs=[output_name],
            name=self.get_unique_name("random_split:tile"),
        )
        self.builder.add_node(node_2)

    def _process_jaxpr(self, jaxpr, consts):
        """Process a JAXPR and convert it to ONNX nodes."""

        # Add input variables to the ONNX graph, skipping any that are already added
        # (such as parameters added via add_scalar_input)
        for var in jaxpr.invars:
            # here we need to call a function that returns the name of the variable
            # match_call_param_by_type_and_order may use call_params
            # call_params should be stored by name and type (not value)
            var_name = self.match_call_param_by_type_and_order(var)
            if var_name is None:
                var_name = self.get_var_name(var)
            # Check if this input is already in the builder's inputs
            # This avoids duplicate inputs for parameters that were added as scalar inputs
            if not any(
                input_info.name == var_name for input_info in self.builder.inputs
            ):
                self.add_input(var, var.aval.shape, var.aval.dtype)

        # Add constants to the ONNX graph.
        for i, const in enumerate(consts):
            const_name = self.get_constant_name(const)
            const_var = jaxpr.constvars[i]
            self.var_to_name[const_var] = const_name
            self.name_to_var[const_name] = const_var
            self.name_to_const[const_name] = const

        # Process equations in the JAXPR.
        for eqn in jaxpr.eqns:
            self._process_eqn(eqn)

        # Add output variables to the ONNX graph.
        for var in jaxpr.outvars:
            name = self.get_var_name(var)
            shape = None
            dtype = None

            metadata = self.builder.get_value_info_metadata_with_origin(name)
            if metadata:
                shape, dtype_enum, _ = metadata
                try:
                    dtype = helper.tensor_dtype_to_np_dtype(dtype_enum)
                except Exception:
                    print(
                        f"[WARN] Could not convert dtype enum {dtype_enum} for {name}, fallback to var.aval"
                    )
                    shape = var.aval.shape
                    dtype = var.aval.dtype
            else:
                print(
                    f"[WARN] No metadata found for output var '{name}', using fallback."
                )
                shape = var.aval.shape
                dtype = var.aval.dtype

            self.add_output(var, shape, dtype)

    def _process_eqn(self, eqn):
        """Process a single JAXPR equation."""
        if not hasattr(eqn, "primitive"):
            raise NotImplementedError(f"Non-primitive equation: {eqn}")

        primitive = eqn.primitive
        name = primitive.name

        is_function_handler = name in ONNX_FUNCTION_PLUGIN_REGISTRY.keys()

        handler = self.primitive_handlers.get(name)
        if handler is None:
            raise NotImplementedError(f"Primitive {name} not implemented")

        handler(self, eqn, eqn.params)

        if not is_function_handler:
            for outvar in eqn.outvars:
                output_name = self.get_name(outvar)
                if hasattr(outvar, "aval"):
                    self.add_shape_info(
                        output_name, outvar.aval.shape, outvar.aval.dtype
                    )
                else:
                    print(
                        f"[WARN] Cannot add shape info for {output_name}, missing .aval."
                    )

    def match_call_param_by_type_and_order(self, var):
        """
        Match a variable to a parameter in call_params based on type and order.

        Args:
            var: The variable to match

        Returns:
            The name of the matched parameter, or None if no match is found
        """
        if not self.call_params or not hasattr(var, "aval"):
            return None

        # Check if this variable matches any parameter by type and shape
        var_dtype = var.aval.dtype
        var_shape = tuple(var.aval.shape)

        # Special handling for boolean parameters like 'deterministic'
        if var_dtype == jnp.bool_ and var_shape == ():
            # Look for boolean parameters in call_params
            for param_name, param_value in self.call_params.items():
                if isinstance(param_value, bool):
                    # Skip parameters that have already been matched
                    param_key = f"{param_name}"
                    if param_key in self.var_to_name.values():
                        continue

                    print(
                        f"[INFO] Matching boolean variable to parameter '{param_name}'"
                    )
                    # Store this mapping
                    self.var_to_name[var] = param_name
                    self.name_to_var[param_name] = var
                    return param_name

        # Track position to maintain matching by order for non-boolean parameters
        matched_params = []

        for param_name, param_value in self.call_params.items():
            # Skip parameters that have already been matched
            param_key = f"{param_name}"
            if param_key in self.var_to_name.values():
                continue

            # Check if parameter type and shape match the variable
            if hasattr(param_value, "dtype") and hasattr(param_value, "shape"):
                param_dtype = param_value.dtype
                param_shape = tuple(param_value.shape)

                if param_dtype == var_dtype and param_shape == var_shape:
                    matched_params.append((param_name, param_value))

        # If we found matches, use the first one
        if matched_params:
            param_name, _ = matched_params[0]
            # Store this mapping
            self.var_to_name[var] = param_name
            self.name_to_var[param_name] = var
            return param_name

        return None
