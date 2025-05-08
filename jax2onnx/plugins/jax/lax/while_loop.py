from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence, Callable

import jax
import numpy as np
from jax import core, lax
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.lax.while_loop")

while_loop_p = Primitive("while_loop")
while_loop_p.multiple_results = True


@register_primitive(
    jaxpr_primitive=while_loop_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html",
    onnx=[
        {"component": "Loop", "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html"}
    ],
    since="0.5.1",
    context="primitives.lax",
    component="while_loop",
    testcases=[
        {
            "testcase": "while_loop_counter",
            "callable": lambda: lax.while_loop(lambda v: v < 5, lambda v: v + 1, 0),
            "input_shapes": [],
            "expected_output_shapes": [()],
        },
        {
            "testcase": "while_loop_vector",
            "callable": lambda: lax.while_loop(
                lambda v: v[0] < 5,
                lambda v: v + 1,
                jax.numpy.array([0], dtype=jax.numpy.int32),
            ),
            "input_shapes": [],
            "expected_output_shapes": [(1,)],
        },
    ],
)
class WhileLoopPlugin(PrimitiveLeafPlugin):
    _ORIG_WHILE_LOOP: Callable | None = None

    @staticmethod
    def abstract_eval(*in_avals: core.AbstractValue, cond_jaxpr, body_jaxpr, **__):
        return tuple(in_avals)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[core.Var],
        node_outputs: Sequence[core.Var],
        params: dict[str, Any],
    ):
        logger.debug(f"Attempting conversion for {while_loop_p.name}")

        if "cond_jaxpr" not in params or "body_jaxpr" not in params:
            raise ValueError("Missing cond_jaxpr or body_jaxpr in primitive params.")

        cond_closed_jaxpr = params["cond_jaxpr"]
        body_closed_jaxpr = params["body_jaxpr"]

        cond_jaxpr = cond_closed_jaxpr.jaxpr
        body_jaxpr = body_closed_jaxpr.jaxpr
        cond_consts = cond_closed_jaxpr.consts
        body_consts = body_closed_jaxpr.consts

        state_input_vars = node_inputs
        state_input_names = [s.get_name(v) for v in state_input_vars]
        state_output_vars = node_outputs
        state_output_names = [s.get_name(v) for v in state_output_vars]

        # Create body subgraph builder and converter
        body_builder = OnnxBuilder(
            name_generator=s.builder.name_generator,
            opset=s.builder.opset,
            model_name=s.builder.get_unique_name(f"{while_loop_p.name}_body_graph"),
            initializers=None,
            converter=None,
        )
        body_builder.var_to_symbol_map = s.builder.var_to_symbol_map
        body_converter = s.__class__(body_builder)

        # Define inputs: iter_num, cond_in, state_in
        iter_name = body_builder.name_generator.get("iter_num")
        cond_in_name = body_builder.name_generator.get("cond_in")
        state_in_names = [
            body_builder.name_generator.get(f"state_in_{i}")
            for i in range(len(state_input_vars))
        ]

        body_builder.add_scalar_input(iter_name, helper.TensorProto.INT64)
        body_builder.add_scalar_input(cond_in_name, helper.TensorProto.BOOL)
        for name, var in zip(state_in_names, state_input_vars):
            body_builder.add_input(name, var.aval.shape, var.aval.dtype)

        # Map constants and state inputs into subgraph
        for i, const_var in enumerate(body_jaxpr.constvars):
            body_converter.var_to_name[const_var] = body_converter.get_constant_name(
                body_consts[i]
            )
        for i, state_var in enumerate(body_jaxpr.invars[len(body_jaxpr.constvars) :]):
            body_converter.var_to_name[state_var] = state_in_names[i]

        # Process body_jaxpr without adding outputs automatically
        # Process body_jaxpr (default call adds its outvars as graph outputs)
        body_converter._process_jaxpr(body_jaxpr, body_consts)

        # Remove any auto‑added outputs – we will register them in required order below
        body_builder.outputs.clear()

        # Collect state outputs
        state_out_vars = body_jaxpr.outvars
        state_out_names = [body_converter.get_name(v) for v in state_out_vars]

        # Map and process cond_jaxpr inside body
        for i, const_var in enumerate(cond_jaxpr.constvars):
            body_converter.var_to_name[const_var] = body_converter.get_constant_name(
                cond_consts[i]
            )
        for i, state_var in enumerate(cond_jaxpr.invars[len(cond_jaxpr.constvars) :]):
            body_converter.var_to_name[state_var] = state_out_names[i]
        for eqn in cond_jaxpr.eqns:
            body_converter._process_eqn(eqn)

        cond_out_name = body_converter.get_name(cond_jaxpr.outvars[0])

        # Register body outputs in order (cond, state)
        body_builder.add_output(cond_out_name, (), np.bool_)
        for name, var in zip(state_out_names, state_out_vars):
            body_builder.add_output(name, var.aval.shape, var.aval.dtype)

        # Create subgraph proto
        body_graph = body_builder.create_graph(
            body_builder.model_name, is_subgraph=True
        )

        # Initial condition builder
        init_builder = OnnxBuilder(
            name_generator=s.builder.name_generator,
            opset=s.builder.opset,
            model_name=s.builder.get_unique_name(f"{while_loop_p.name}_initial_cond"),
            initializers=None,
            converter=None,
        )
        init_builder.var_to_symbol_map = s.builder.var_to_symbol_map
        init_converter = s.__class__(init_builder)

        for i, const_var in enumerate(cond_jaxpr.constvars):
            init_converter.var_to_name[const_var] = init_converter.get_constant_name(
                cond_consts[i]
            )
        for i, state_var in enumerate(cond_jaxpr.invars[len(cond_jaxpr.constvars) :]):
            init_converter.var_to_name[state_var] = state_input_names[i]

        # Process cond_jaxpr without outputs
        # Process cond_jaxpr to compute initial condition
        init_converter._process_jaxpr(cond_jaxpr, cond_consts)

        # Merge init subgraph elements into main builder
        s.builder.nodes.extend(init_builder.nodes)
        existing = {init.name for init in s.builder.initializers}
        for tensor in init_builder.initializers:
            if tensor.name not in existing:
                s.builder.initializers.append(tensor)
        s.builder.value_info_metadata.update(init_builder.value_info_metadata)
        merged_vi = {vi.name: vi for vi in s.builder.value_info}
        for vi in init_builder.value_info:
            merged_vi[vi.name] = vi
        s.builder.value_info = list(merged_vi.values())
        s.builder.functions.update(init_builder.functions)

        init_out = init_converter.get_name(cond_jaxpr.outvars[0])
        max_iter = s.get_constant_name(np.array(2**31 - 1, dtype=np.int64))

        loop = helper.make_node(
            "Loop",
            inputs=[max_iter, init_out, *state_input_names],
            outputs=state_output_names,
            body=body_graph,
            name=s.get_unique_name("while_loop"),
        )
        s.add_node(loop)
        for name, var in zip(state_output_names, state_output_vars):
            s.add_shape_info(name, var.aval.shape, var.aval.dtype)

    @staticmethod
    def _while_loop_binding(cond_fun, body_fun, init_val):
        closed_cond = jax.make_jaxpr(cond_fun)(init_val)
        closed_body = jax.make_jaxpr(body_fun)(init_val)
        flat, tree = jax.tree_util.tree_flatten(init_val)
        results = while_loop_p.bind(
            *flat, cond_jaxpr=closed_cond, body_jaxpr=closed_body
        )
        return jax.tree_util.tree_unflatten(tree, results)

    @staticmethod
    def get_monkey_patch(orig_fn):
        if WhileLoopPlugin._ORIG_WHILE_LOOP is None:
            WhileLoopPlugin._ORIG_WHILE_LOOP = orig_fn

        def patched(cond_fun, body_fun, init_val):
            return WhileLoopPlugin._while_loop_binding(cond_fun, body_fun, init_val)

        return patched

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [lax],
            "target_attribute": "while_loop",
            "patch_function": WhileLoopPlugin.get_monkey_patch,
        }


while_loop_p.def_abstract_eval(WhileLoopPlugin.abstract_eval)
