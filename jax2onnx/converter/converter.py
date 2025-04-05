# file: jax2onnx/converter/converter.py
import jax
import onnx
from onnx import helper
import numpy as np
from typing import Dict, Any, Tuple
import jax.random
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
)
from jax2onnx.plugin_system import (
    PLUGIN_REGISTRY,
    import_all_plugins,
)
from jax2onnx.converter.patch_utils import temporary_monkey_patches
from jax2onnx.converter.utils import numpy_dtype_to_tensorproto

# At the top of converter.py
from jax2onnx.converter.utils import function_handler as core_function_handler


class Jaxpr2OnnxConverter:
    """
    A translator that converts JAX's JAXPR representation to ONNX format.
    """

    def __init__(self, builder: OnnxBuilder):

        self.builder = builder

        # Other converter state
        self.var_to_name: Dict[Any, str] = {}
        self.name_to_var: Dict[str, Any] = {}
        self.primitive_handlers = {}
        self.shape_env: Dict[str, Tuple[int, ...]] = {}  # <- added shape environment

        self.name_to_const: Dict[str, Any] = {}
        self.primitive_handlers[jax._src.prng.random_seed_p] = self._handle_random_seed
        self.primitive_handlers[jax._src.prng.random_wrap_p] = self._handle_random_wrap
        self.primitive_handlers[jax._src.prng.random_split_p] = (
            self._handle_random_split
        )
        self.primitive_handlers[jax._src.prng.random_unwrap_p] = (
            self._handle_random_unwrap
        )

        import_all_plugins()

        for key, plugin in PLUGIN_REGISTRY.items():
            if isinstance(plugin, (PrimitiveLeafPlugin)):
                self.primitive_handlers[key] = plugin.get_handler(self)

        for plugin in ONNX_FUNCTION_PLUGIN_REGISTRY.values():
            primitive = plugin.primitive
            self.primitive_handlers[primitive.name] = plugin.get_handler(self)

    def new_var(self, dtype: np.dtype, shape: Tuple[int, ...]):
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
        name = self.get_var_name(var)
        self.builder.add_input(name, shape, dtype)
        self.shape_env[name] = shape  # <-- store shape
        return name

    def add_output(self, var, shape, dtype=np.float32):
        name = self.get_var_name(var)
        self.builder.add_output(name, shape, dtype)
        self.shape_env[name] = shape  # <-- store shape
        return name

    def add_shape_info(self, name, shape, dtype=np.float32):
        self.builder.add_value_info(name, shape, dtype)
        self.shape_env[name] = shape  # <-- store shape
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

    def trace_jaxpr(self, fn, example_args, preserve_graph=False):
        print(f"trace_jaxpr ... preserve_graph= {preserve_graph}")
        if not preserve_graph:
            self.builder.reset()
            self.var_to_name.clear()
            self.name_to_const.clear()
            self.shape_env.clear()

        with temporary_monkey_patches(allow_function_primitives=True):
            closed_jaxpr = jax.make_jaxpr(fn)(*example_args)

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

        # Setup inputs
        for var in jaxpr.invars:
            self.add_input(var, var.aval.shape, var.aval.dtype)

        # Setup constants
        for i, const in enumerate(consts):
            const_name = self.get_constant_name(const)
            const_var = jaxpr.constvars[i]
            self.var_to_name[const_var] = const_name
            self.name_to_var[const_name] = const_var
            self.name_to_const[const_name] = const

        # Process all equations in the JAXPR
        for eqn in jaxpr.eqns:
            self._process_eqn(eqn)

        for var in jaxpr.outvars:
            name = self.get_var_name(var)
            shape = None
            dtype = None

            metadata = self.builder.get_value_info_metadata_with_origin(name)
            if metadata:
                shape, dtype_enum, _ = metadata
                try:
                    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[dtype_enum]
                except Exception as e:
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

        handler = self.primitive_handlers.get(name)
        if handler is None:
            raise NotImplementedError(f"Primitive {name} not implemented")

        # Identify whether this handler is function_handler (by reference or closure match)
        is_function_handler = False
        actual_func = getattr(handler, "__func__", None)  # method bound to plugin?
        if handler is core_function_handler or actual_func is core_function_handler:
            is_function_handler = True

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
