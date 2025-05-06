# file: jax2onnx/plugins/jax/lax/while_loop.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import core, lax
from jax.extend.core import Primitive
from onnx import helper

# Assuming OnnxBuilder and Jaxpr2OnnxConverter are accessible
# Adjust imports based on actual project structure
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    # Import converter type hint if not already globally available
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


logger = logging.getLogger("jax2onnx.plugins.jax.lax.while_loop")

# Define the primitive for lax.while_loop
while_loop_p = Primitive("while_loop")
# while_loop returns the final state, which can be a pytree,
# so multiple_results should be True if the state is a tuple/list/dict.
# For simplicity here, assume it might return multiple tensors if state is a tuple.
while_loop_p.multiple_results = True


@register_primitive(
    jaxpr_primitive=while_loop_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html",
    onnx=[
        {
            "component": "Loop",
            "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html",
        }
    ],
    since="upcoming",  # Mark as not yet released
    context="primitives.lax",
    component="while_loop",
    testcases=[
        # Test cases need careful design due to the complexity
        # Example: Simple counter
        {
            "testcase": "while_loop_counter",
            "callable": lambda: lax.while_loop(
                lambda val: val < 5, lambda val: val + 1, 0  # init_val
            ),
            "input_shapes": [],  # No external inputs for this simple case
            "expected_output_shapes": [()],  # Expected output is a scalar
        },
        # Add more complex test cases involving arrays and pytrees later
    ],
)
class WhileLoopPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.while_loop to ONNX Loop operator."""

    _ORIG_WHILE_LOOP: Callable | None = None

    @staticmethod
    def abstract_eval(*in_avals: core.AbstractValue, cond_jaxpr, body_jaxpr, **__):
        """
        Abstract evaluation for while_loop.
        The output shapes and dtypes match the initial value's shape and dtype.
        The primitive bind should only receive init_val avals.
        """
        # The output structure matches the input structure (init_val).
        # `in_avals` here represent the abstract values of `init_val` (flattened).
        return tuple(in_avals)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[core.Var],  # These are the init_val variables (flattened)
        node_outputs: Sequence[
            core.Var
        ],  # These are the final_val variables (flattened)
        params: dict[str, Any],
    ):
        """Convert lax.while_loop to ONNX Loop."""
        logger.debug(f"Attempting conversion for {while_loop_p.name}")

        if "cond_jaxpr" not in params or "body_jaxpr" not in params:
            raise ValueError("Missing cond_jaxpr or body_jaxpr in primitive params.")

        cond_closed_jaxpr = params["cond_jaxpr"]
        body_closed_jaxpr = params["body_jaxpr"]

        cond_jaxpr = cond_closed_jaxpr.jaxpr
        body_jaxpr = body_closed_jaxpr.jaxpr
        cond_consts = cond_closed_jaxpr.consts
        body_consts = body_closed_jaxpr.consts

        # Input state variables (correspond to init_val, should be flattened)
        state_input_vars = node_inputs
        state_input_names = [s.get_name(v) for v in state_input_vars]
        state_output_vars = node_outputs
        state_output_names = [s.get_name(v) for v in state_output_vars]

        logger.debug(f"Input state names: {state_input_names}")
        logger.debug(f"Output state names: {state_output_names}")
        logger.debug(f"Number of cond consts: {len(cond_consts)}")
        logger.debug(f"Number of body consts: {len(body_consts)}")
        logger.debug(f"Cond Jaxpr invars: {cond_jaxpr.invars}")
        logger.debug(f"Body Jaxpr invars: {body_jaxpr.invars}")
        logger.debug(f"Cond Jaxpr outvars: {cond_jaxpr.outvars}")
        logger.debug(f"Body Jaxpr outvars: {body_jaxpr.outvars}")

        # --- Create the Body Subgraph ---
        # The body subgraph proto for the ONNX Loop op.
        # Inputs: (iteration_num, condition, state_vars...)
        # Outputs: (new_condition, updated_state_vars...)

        # 1. Setup converter for the body subgraph
        body_builder = OnnxBuilder(
            s.builder.name_generator,
            s.builder.opset,
            s.builder.get_unique_name(f"{while_loop_p.name}_body_graph"),
            initializers=s.builder.initializers,  # Share initializers
            converter=s.builder.converter,  # Share context
            var_to_symbol_map=getattr(s.builder, "var_to_symbol_map", {}),
        )
        body_converter = s.__class__(body_builder)  # Converter for the subgraph

        # 2. Define body subgraph inputs
        body_iter_num_name = body_converter.get_unique_name("body_iter_num")
        body_cond_in_name = body_converter.get_unique_name("body_cond_in")
        body_state_in_names = [
            body_converter.get_unique_name(f"body_state_in_{i}")
            for i in range(len(state_input_vars))
        ]

        # Register inputs with the body_builder
        body_builder.add_scalar_input(body_iter_num_name, helper.TensorProto.INT64)
        body_builder.add_scalar_input(body_cond_in_name, helper.TensorProto.BOOL)
        for name, var in zip(body_state_in_names, state_input_vars):
            body_builder.add_input(name, var.aval.shape, var.aval.dtype)

        # Map JAXPR invars to ONNX subgraph input names for body_fun processing
        body_var_map = {}
        # The body_jaxpr invars include constants first, then loop state
        num_body_state_invars = len(body_jaxpr.invars) - len(body_consts)
        if num_body_state_invars != len(state_input_vars):
            raise ValueError(
                f"Body JAXPR invar count ({num_body_state_invars}) mismatch with state vars ({len(state_input_vars)})"
            )

        for i, const_var in enumerate(body_jaxpr.constvars):
            const_onnx_name = body_converter.get_constant_name(body_consts[i])
            body_var_map[const_var] = const_onnx_name  # Map const var to ONNX name
            body_converter.var_to_name[const_var] = (
                const_onnx_name  # Update converter map
            )

        for i, state_invar in enumerate(body_jaxpr.invars[len(body_consts) :]):
            onnx_name = body_state_in_names[i]
            body_var_map[state_invar] = onnx_name  # Map state invar to ONNX name
            body_converter.var_to_name[state_invar] = onnx_name  # Update converter map

        # 3. Process body jaxpr using body_converter
        logger.debug("Processing body JAXPR...")
        body_converter._process_jaxpr(body_jaxpr, body_consts)  # Pass consts here
        logger.debug("Finished processing body JAXPR.")

        # 4. Get body outputs (updated state)
        # The outputs of body_jaxpr correspond to the updated state
        body_state_output_vars = body_jaxpr.outvars
        body_state_output_names = [
            body_converter.get_name(v) for v in body_state_output_vars
        ]
        logger.debug(f"Body subgraph state output names: {body_state_output_names}")

        # 5. Process cond_fun *after* body_fun using the updated state
        # Setup converter for the condition logic evaluation within the body
        cond_builder = OnnxBuilder(
            s.builder.name_generator,
            s.builder.opset,
            s.builder.get_unique_name(
                f"{while_loop_p.name}_cond_eval_graph"
            ),  # Temp graph
            initializers=s.builder.initializers,
            converter=s.builder.converter,
            var_to_symbol_map=getattr(s.builder, "var_to_symbol_map", {}),
        )
        cond_converter = s.__class__(cond_builder)

        # Map JAXPR invars for cond_fun to ONNX names (updated state from body)
        cond_var_map = {}
        num_cond_state_invars = len(cond_jaxpr.invars) - len(cond_consts)
        if num_cond_state_invars != len(body_state_output_names):
            raise ValueError(
                f"Cond JAXPR invar count ({num_cond_state_invars}) mismatch with body output state vars ({len(body_state_output_names)})"
            )

        # Map constants for condition function
        for i, const_var in enumerate(cond_jaxpr.constvars):
            const_onnx_name = cond_converter.get_constant_name(cond_consts[i])
            cond_var_map[const_var] = const_onnx_name
            cond_converter.var_to_name[const_var] = const_onnx_name

        # Map state inputs for condition function (using outputs from body)
        for i, state_invar in enumerate(cond_jaxpr.invars[len(cond_consts) :]):
            onnx_name = body_state_output_names[i]  # Map to body's output name
            cond_var_map[state_invar] = onnx_name
            cond_converter.var_to_name[state_invar] = (
                onnx_name  # Use body's output name
            )

        logger.debug("Processing condition JAXPR (using body outputs)...")
        cond_converter._process_jaxpr(cond_jaxpr, cond_consts)  # Pass consts here
        logger.debug("Finished processing condition JAXPR.")

        # Get the output name of the condition evaluation (scalar bool)
        cond_output_var = cond_jaxpr.outvars[0]
        body_new_cond_name = cond_converter.get_name(
            cond_output_var
        )  # This is the name within cond_converter's scope
        logger.debug(f"Condition output name (internal): {body_new_cond_name}")

        # Transfer nodes from cond_converter to body_builder
        body_builder.nodes.extend(cond_builder.nodes)
        body_builder.initializers = list(
            set(body_builder.initializers + cond_builder.initializers)
        )
        # We need to make sure value_info is consistent/merged
        body_builder.value_info_metadata.update(cond_builder.value_info_metadata)
        # Merge functions if any were created during cond tracing
        body_builder.functions.update(cond_builder.functions)

        # 6. Define body subgraph outputs for ONNX Loop
        # Output order: (loop_condition, loop_carried_dependencies...)
        body_graph_output_names = [body_new_cond_name] + body_state_output_names

        # Add outputs to the body builder
        body_builder.add_output(body_new_cond_name, (), np.bool_)  # Condition output
        for name, var in zip(body_state_output_names, body_state_output_vars):
            body_builder.add_output(name, var.aval.shape, var.aval.dtype)

        # 7. Create the final body GraphProto
        body_graph_proto = body_builder.create_graph(body_builder.model_name)
        logger.debug(f"Body subgraph created: {body_graph_proto.name}")
        # logger.debug(f"Body subgraph inputs: {[i.name for i in body_graph_proto.input]}")
        # logger.debug(f"Body subgraph outputs: {[o.name for o in body_graph_proto.output]}")

        # --- Initial Condition ---
        # Evaluate cond_fun(init_val) to get the initial boolean condition.
        # This ideally requires a separate mini-conversion or access to JAX eval.
        # WORKAROUND: Use a constant True. This assumes the loop runs at least once if
        # the condition is initially true, which matches JAX but might need adjustment
        # if ONNX Loop behaves differently for the first iteration condition check.
        # A more robust solution would trace cond_fun separately.
        initial_cond_name = s.get_constant_name(np.array(True, dtype=np.bool_))
        logger.warning(
            "Using constant True as initial loop condition - may be inaccurate."
        )

        # --- Max Iterations ---
        # JAX while_loop has no max count. Use a large number or INT64_MAX.
        # Using a very large number might be safer than MAX if overflow is a concern.
        max_iter_val = (
            2**31 - 1
        )  # Large positive int32 max, safer than int64 max potentially
        max_iter_name = s.get_constant_name(np.array(max_iter_val, dtype=np.int64))
        logger.debug(f"Using max iteration count: {max_iter_val}")

        # --- Create the ONNX Loop Node in the main graph ---
        loop_node = helper.make_node(
            "Loop",
            inputs=[
                max_iter_name,  # M: Max iterations
                initial_cond_name,  # cond: Initial condition value
                *state_input_names,  # v_initial: Loop-carried dependencies (init_val)
            ],
            outputs=state_output_names,  # Loop outputs correspond to final state (final_val)
            name=s.get_unique_name("while_loop"),
            body=body_graph_proto,  # The body subgraph
        )

        s.add_node(loop_node)
        logger.info(f"Successfully added ONNX Loop node for {while_loop_p.name}.")

        # Register output shapes for the main graph Loop node
        for name, var in zip(state_output_names, state_output_vars):
            s.add_shape_info(name, var.aval.shape, var.aval.dtype)

    @staticmethod
    def _while_loop_binding(cond_fun, body_fun, init_val):
        """Binds inputs and functions to the while_loop primitive."""
        # Trace cond_fun and body_fun to get jaxprs
        # Need abstract values of init_val for tracing
        init_avals, init_tree = jax.tree_util.tree_flatten(
            jax.tree_util.tree_map(
                lambda x: core.ShapedArray(jnp.shape(x), jnp.result_type(x)), init_val
            )  # MODIFIED HERE
        )

        try:
            # When tracing with abstract values (avals), jax.make_jaxpr expects those avals directly.
            # The tree structure of avals should match the expected tree structure of init_val for cond_fun and body_fun.
            # jax.make_jaxpr will internally handle tree_flatten/unflatten if the traced function expects a pytree.
            # So, we pass the pytree of avals (init_avals_tree) that matches init_val's structure.
            init_avals_tree = jax.tree_util.tree_unflatten(init_tree, init_avals)

            closed_cond_jaxpr = jax.make_jaxpr(cond_fun)(init_avals_tree)
            closed_body_jaxpr = jax.make_jaxpr(body_fun)(init_avals_tree)
        except Exception as e:
            logger.error(f"Error tracing cond_fun or body_fun: {e}")
            logger.error(
                f"Cond_fun: {cond_fun}, Body_fun: {body_fun}, Init_avals (tree): {init_avals_tree}"
            )
            raise

        # Flatten init_val for binding
        flat_init_val, tree_def = jax.tree_util.tree_flatten(init_val)

        # Bind the primitive with flattened inputs and jaxpr parameters
        flat_results = while_loop_p.bind(
            *flat_init_val,  # Pass flattened initial values as positional args
            cond_jaxpr=closed_cond_jaxpr,  # Pass traced jaxpr+consts
            body_jaxpr=closed_body_jaxpr,
        )

        # Reconstruct the output pytree
        return jax.tree_util.tree_unflatten(tree_def, flat_results)

    @staticmethod
    def get_monkey_patch(orig_fn):
        """Returns the patched function that binds the primitive."""
        logger.debug(f"Patching {lax.while_loop.__name__}")
        if WhileLoopPlugin._ORIG_WHILE_LOOP is None:
            WhileLoopPlugin._ORIG_WHILE_LOOP = orig_fn  # Store original

        def patched_while_loop(cond_fun, body_fun, init_val):
            logger.debug("Intercepted call to lax.while_loop")
            # Ensure functions are callable (basic check)
            if not callable(cond_fun) or not callable(body_fun):
                raise TypeError("cond_fun and body_fun must be callable")

            # Use the binding helper which handles tracing
            return WhileLoopPlugin._while_loop_binding(cond_fun, body_fun, init_val)

        return patched_while_loop

    @staticmethod
    def patch_info():
        """Provides patching information for lax.while_loop."""
        return {
            "patch_targets": [lax],  # Patch jax.lax module
            "target_attribute": "while_loop",
            "patch_function": WhileLoopPlugin.get_monkey_patch,
        }


# Register abstract evaluation function with the primitive
while_loop_p.def_abstract_eval(WhileLoopPlugin.abstract_eval)
