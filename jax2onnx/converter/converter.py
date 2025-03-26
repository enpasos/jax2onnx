# Modified: jax2onnx/converter/converter.py

import jax
from onnx import helper
import numpy as np
from typing import (
    Dict,
    Any,
    Tuple,
    List,
    Callable,
    Sequence,
)  # Added Sequence, TYPE_CHECKING
import jax.random
from jax.extend import core  # Import core

# Import the dispatcher singleton
from .primitive_converter import primitive_dispatcher
from .onnx_builder import OnnxBuilder

# Keep plugin imports for loading handlers here
from jax2onnx.plugin_system import (
    PLUGIN_REGISTRY,
    PrimitivePlugin,
    import_all_plugins,
)  # Need function registry if handled here

# Keep patch utils if trace_jaxpr uses it
from jax2onnx.converter.patch_utils import temporary_monkey_patches


class Jaxpr2OnnxConverter:
    """
    A translator that converts JAX's JAXPR representation to ONNX format.
    Orchestrates traversal and delegates primitive conversion.
    """

    def __init__(self, name_counter=0):
        print("[INFO] Initializing Jaxpr2OnnxConverter...")

        # Use 'onnx_builder' consistently internally
        self.builder = OnnxBuilder(name_counter)
        # State for mapping JAX vars/literals to ONNX names
        self.var_to_name: Dict[Any, str] = {}

        # --- Handler Loading ---
        # Load ONLY plugin handlers into self.primitive_handlers.
        # Built-ins are handled by the dispatcher internally.
        self.primitive_handlers: Dict[str, Callable] = {}
        self._load_plugin_handlers()  # Renamed method

        # Store reference to the dispatcher instance
        self.dispatcher = primitive_dispatcher

    def _load_plugin_handlers(self):
        """Loads handlers from the plugin system."""
        print("[INFO] Converter loading plugin handlers...")
        import_all_plugins()
        count = 0
        for key, plugin in PLUGIN_REGISTRY.items():
            # Check it's a plugin derived from PrimitivePlugin (includes FunctionPlugin if registered here)
            if isinstance(plugin, PrimitivePlugin):
                try:
                    # get_handler should return the callable expecting (converter, eqn, **params)
                    handler = plugin.get_handler(self)  # Pass self for context
                    # Use the key from the registry (primitive name or function name)
                    self.primitive_handlers[key] = handler
                    count += 1
                except Exception as e:
                    print(
                        f"Warning: Failed to get handler for plugin '{key}' (type: {type(plugin).__name__}): {e}"
                    )
            # else: It's an ExamplePlugin or something else, ignore.

        print(
            f"[INFO] Converter loaded {count} plugin handlers into self.primitive_handlers."
        )

    def new_var(self, dtype: np.dtype, shape: Tuple[int, ...]):
        return core.Var(self.get_unique_name(""), core.ShapedArray(shape, dtype))

    def add_initializer(
        self, name, vals, data_type=helper.TensorProto.INT64, dims=None
    ):
        if dims is None:
            dims = [len(vals)]
        tensor = helper.make_tensor(
            name=name, data_type=data_type, dims=dims, vals=vals
        )
        self.builder.initializers.append(tensor)
        return name

    def add_node(self, node):
        self.builder.add_node(node)

    def get_unique_name(self, prefix="node"):
        return self.builder.get_unique_name(prefix)

    def get_var_name(self, var):
        # Handles jax.core.Var
        if var not in self.var_to_name:
            name = self.get_unique_name("var")
            self.var_to_name[var] = name
            # Add shape info when var name is first created
            if hasattr(var, "aval"):
                self.add_shape_info(name, var.aval.shape, var.aval.dtype)
        return self.var_to_name[var]

    def get_constant_name(self, val_or_literal):
        # Delegate to builder, handling JAX Literal if necessary
        val = (
            val_or_literal.val
            if isinstance(val_or_literal, core.Literal)
            else val_or_literal
        )
        # Pass original literal too, builder might use its hash or var id
        original_ref = (
            val_or_literal if isinstance(val_or_literal, core.Literal) else None
        )
        const_name = self.builder.get_constant_name(val, original_ref)
        # Ensure mapping from Literal to name exists
        if isinstance(val_or_literal, core.Literal):
            self.var_to_name[val_or_literal] = const_name
        return const_name

    def add_input(self, var, shape, dtype=np.float32):
        name = self.get_var_name(var)
        self.builder.add_input(name, shape, dtype)
        return name

    def add_output(self, var, shape, dtype=np.float32):
        name = self.get_var_name(var)
        self.builder.add_output(name, shape, dtype)
        return name

    def add_shape_info(self, name, shape, dtype=np.float32):
        # Delegate, builder handles caching
        self.builder.add_value_info(name, shape, dtype)
        return name

    def get_name(self, var_or_literal):
        # Unified getter remains useful
        if isinstance(var_or_literal, core.Var):
            return self.get_var_name(var_or_literal)
        elif isinstance(var_or_literal, core.Literal):
            return self.get_constant_name(var_or_literal)
        else:
            # Attempt to handle direct constants (e.g., from eqn.params)
            # This might be needed if plugins/handlers work with raw values
            try:
                # Use builder's constant handling
                return self.get_constant_name(var_or_literal)
            except Exception as e:
                raise TypeError(
                    f"Cannot get ONNX name for value/type: {var_or_literal} ({type(var_or_literal)}). Error: {e}"
                )

    def _process_pjit(self, eqn):
        name = eqn.params.get("name")
        closed_jaxpr = eqn.params.get("jaxpr")

        if not isinstance(closed_jaxpr, jax._src.core.ClosedJaxpr):
            raise ValueError("Expected ClosedJaxpr in pjit.param[jaxpr]")

        # Special-case for distribution-style pjit
        if name in {"_normal", "_uniform", "_truncated_normal"}:
            fake_eqn = core.new_jaxpr_eqn(
                eqn.invars,
                eqn.outvars,
                primitive=None,  # not used
                params=eqn.params,
                source_info=None,
            )
            # Manually call appropriate built-in handler
            {
                "_normal": self.dispatcher._handle_random_normal,
                "_uniform": self.dispatcher._handle_random_uniform,
                "_truncated_normal": self.dispatcher._handle_random_normal,
            }[name](self, fake_eqn, **eqn.params)
            return

        # Otherwise: nested JAXPR (actual subgraph)
        print(
            f"Warning: Executing existing _process_pjit logic for {name}. Needs review."
        )
        ...
        # Keep the rest as-is

    def _connect_inputs_to_subconverter(self, parent_inputs, subconverter_inputs):
        """Connect inputs from parent to subconverter."""
        # (Keep original implementation, but use self.onnx_builder)
        if len(parent_inputs) != len(subconverter_inputs):
            # Add more info to error
            raise ValueError(
                f"Input connection mismatch: Parent ({len(parent_inputs)}) vs Subconverter ({len(subconverter_inputs)}). "
                f"Parent: {parent_inputs}, Sub: {subconverter_inputs}"
            )

        for parent_input, subconverter_input in zip(parent_inputs, subconverter_inputs):
            parent_name = self.get_name(parent_input)
            subconverter_name = (
                subconverter_input.name
            )  # Subconverter builder assigned this name
            node = self.builder.create_node(
                "Identity",
                [parent_name],
                [subconverter_name],
                name=self.get_unique_name("pjit_input_connect"),
            )
            self.add_node(node)

    def _connect_outputs_from_subconverter(self, parent_outputs, subconverter_outputs):
        """Connect outputs from subconverter back to parent."""
        # (Keep original implementation, but use self.onnx_builder)
        if len(parent_outputs) != len(subconverter_outputs):
            raise ValueError(
                f"Output connection mismatch: Parent ({len(parent_outputs)}) vs Subconverter ({len(subconverter_outputs)})."
            )

        for parent_output, subconverter_output in zip(
            parent_outputs, subconverter_outputs
        ):
            parent_name = self.get_name(parent_output)  # Assign name to parent var
            subconverter_name = (
                subconverter_output.name
            )  # Name from subconverter's output
            node = self.builder.create_node(
                "Identity",
                [subconverter_name],
                [parent_name],
                name=self.get_unique_name("pjit_output_connect"),
            )
            self.add_node(node)
            # Ensure shape info is added for the parent output var
            self.add_shape_info(
                parent_name, parent_output.aval.shape, parent_output.aval.dtype
            )

    # --- Simplified Equation Processing ---
    def _process_eqn(self, eqn):
        """Process a single JAXPR equation by dispatching to the handler."""
        if not hasattr(eqn, "primitive"):
            raise NotImplementedError(f"Non-primitive equation: {eqn}")

        primitive_name = eqn.primitive.name  # Use name for consistency

        # Special handling for pjit, call internal method
        if primitive_name == "pjit":
            self._process_pjit(eqn)
            # Skip normal dispatch and shape info add below if pjit handled it
            return

        # --- Delegate dispatching and execution to the dispatcher ---
        try:
            self.dispatcher.dispatch_and_execute(self, eqn)  # Pass self and eqn
        except NotImplementedError as e:
            # Re-raise specific error if dispatcher couldn't find handler
            raise NotImplementedError(
                f"No handler found for primitive: {primitive_name}"
            ) from e
        except Exception:
            # Catch other errors during handler execution
            print(f"Error processing equation: {eqn}")
            raise  # Re-raise

        # Add shape info for outputs *after* handler execution
        for outvar in eqn.outvars:
            output_name = self.get_name(outvar)  # Ensure mapping exists
            if hasattr(outvar, "aval"):
                # Use the main converter's add_shape_info method
                self.add_shape_info(output_name, outvar.aval.shape, outvar.aval.dtype)
            else:
                print(
                    f"Warning: Output variable {outvar} has no 'aval' attribute. Cannot add shape info."
                )

    # --- JAXPR Traversal ---
    # (_process_jaxpr, trace_jaxpr remain largely the same, but call simpler _process_eqn)
    def _process_jaxpr(self, jaxpr: core.Jaxpr, consts: Sequence[Any], *args: core.Var):
        """
        Process a JAXPR and convert it to ONNX nodes.
        Maps inputs and constants, processes equations via dispatcher, maps outputs.
        """
        print(
            f"[DEBUG] _process_jaxpr: Processing {len(jaxpr.eqns)} equations for Jaxpr."
        )  # Debug

        # Setup inputs
        if len(args) != len(jaxpr.invars):
            raise ValueError(
                f"Arg mismatch in _process_jaxpr: expected {len(jaxpr.invars)} invars, got {len(args)} args."
            )
        for i, var in enumerate(jaxpr.invars):
            arg_var = args[i]  # Assume args correspond positionally to invars
            # Add input to builder and map var name
            self.add_input(var, var.aval.shape, var.aval.dtype)
            self.var_to_name[var] = self.get_name(
                var
            )  # Ensure name is set via get_name

        # Setup constants (map var to constant name)
        if len(consts) != len(jaxpr.constvars):
            raise ValueError(
                f"Const mismatch in _process_jaxpr: expected {len(jaxpr.constvars)} constvars, got {len(consts)} consts."
            )
        for i, const_val in enumerate(consts):
            const_var = jaxpr.constvars[i]
            const_lit = core.Literal(
                const_val, const_var.aval
            )  # Treat as Literal for mapping
            const_name = self.get_constant_name(
                const_lit
            )  # Ensure initializer & mapping
            self.var_to_name[const_var] = const_name  # Explicitly map const_var

        # Process all equations via simplified _process_eqn which uses dispatcher
        for eqn in jaxpr.eqns:
            self._process_eqn(eqn)

        # Setup outputs
        output_names = self._get_output_names(jaxpr.outvars)  # Get final names
        for name, var in zip(output_names, jaxpr.outvars):
            # Ensure the var in the jaxpr maps to the final output name
            existing_name = self.get_name(var)  # Get potentially existing internal name
            if existing_name != name:
                # If internal name differs from final output name, add Identity
                print(
                    f"[DEBUG] Adding Identity to connect internal var '{existing_name}' to output '{name}'."
                )
                id_node = self.builder.create_node("Identity", [existing_name], [name])
                self.add_node(id_node)
            # Ensure the final name is added to builder outputs
            self.add_output(
                var, var.aval.shape, var.aval.dtype
            )  # Adds using the name mapped in get_name/self.var_to_name

    def _get_output_names(self, outvars: List[core.Var]) -> List[str]:
        """Generate or use provided names for output variables."""
        # (Keep implementation from previous step)
        user_output_names = getattr(self, "output_names", None)
        if user_output_names:
            if len(user_output_names) != len(outvars):
                raise ValueError(
                    f"Provided output_names count ({len(user_output_names)}) does not match JAXPR outvars count ({len(outvars)})."
                )
            for name, var in zip(user_output_names, outvars):
                self.var_to_name[var] = name  # Ensure mapping uses provided name
            return user_output_names
        else:
            # Ensure default names are generated and mapped
            return [self.get_name(v) for v in outvars]

    def reset(self):
        """Reset the converter state."""
        self.builder.reset()
        self.var_to_name = {}
        self.primitive_handlers.clear()
        # Builder handles its own name counter reset

    # file: jax2onnx/converter/converter.py
    # Replace ONLY the trace_jaxpr method in the REFACTORED converter.py

    # Keep the rest of the Jaxpr2OnnxConverter class the same

    def trace_jaxpr(self, fn, example_args, preserve_graph=False):
        """Traces the JAX function (with patching enabled) and processes the resulting JAXPR."""
        print(
            f"[INFO] Converter.trace_jaxpr started for '{getattr(fn, '__name__', 'N/A')}'"
        )
        if not preserve_graph:
            self.reset()
            self._load_plugin_handlers()  # Reload plugin handlers after reset

        print(
            "[INFO] Running jax.make_jaxpr (Patching ENABLED)..."
        )  # Indicate patching is ON
        try:
            # --- RE-ENABLE PATCHING ---
            with temporary_monkey_patches(
                allow_function_primitives=True
            ):  # Adjust flag as needed
                # Ensure example_args is a tuple for unpacking
                args_tuple = (
                    example_args if isinstance(example_args, tuple) else (example_args,)
                )
                closed_jaxpr = jax.make_jaxpr(fn)(*args_tuple)
            # --- END RE-ENABLE PATCHING ---
        except Exception as e:
            print(f"ERROR during jax.make_jaxpr: {e}")
            # Print detailed traceback if error occurs during trace
            import traceback

            traceback.print_exc()
            raise

        print("[INFO] jax.make_jaxpr finished. Processing JAXPR...")
        # print(closed_jaxpr) # Verbose debug

        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.consts
        # Pass the invars from the jaxpr as the *args for _process_jaxpr
        # Ensure args passed match invars count
        self._process_jaxpr(jaxpr, consts, *jaxpr.invars)
        print("[INFO] Converter.trace_jaxpr finished.")


# Keep the rest of the Jaxpr2OnnxConverter class the same
