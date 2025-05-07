from __future__ import annotations

"""jax2onnx plugin: lax.fori_loop → ONNX Loop

*Supported variant*
-------------------
This first version handles the common case
```
  lax.fori_loop(0, N, body_fun, init_val)
```
with a **single** loop‑carried tensor.  Lower bounds other than 0, tuple
state, or captured constants inside *body_fun* are left for future work and
raise ``NotImplementedError``.
"""

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

logger = logging.getLogger("jax2onnx.plugins.jax.lax.fori_loop")

fori_loop_p = Primitive("fori_loop")
fori_loop_p.multiple_results = True


@register_primitive(
    jaxpr_primitive=fori_loop_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html",
    onnx=[
        {"component": "Loop", "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html"}
    ],
    since="upcoming",
    context="primitives.lax",
    component="fori_loop",
    testcases=[
        {
            "testcase": "fori_loop_counter",
            "callable": lambda: lax.fori_loop(
                0,
                5,
                lambda i, v: v + 1,
                0,
            ),
            "input_shapes": [],
            "expected_output_shapes": [()],
        }
    ],
)
class ForiLoopPlugin(PrimitiveLeafPlugin):
    _ORIG_FORI_LOOP: Callable | None = None

    @staticmethod
    def abstract_eval(*in_avals: core.AbstractValue, body_jaxpr, trip_count, **__):
        return tuple(in_avals)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[core.Var],
        node_outputs: Sequence[core.Var],
        params: dict[str, Any],
    ):
        body_closed = params["body_jaxpr"]
        n = params["trip_count"]  # assumed int
        lower = params.get("lower", 0)
        if lower != 0:
            raise NotImplementedError("fori_loop with lower!=0 not yet supported")

        body_jaxpr = body_closed.jaxpr
        body_consts = body_closed.consts

        state_in_vars = node_inputs
        state_in_names = [s.get_name(v) for v in state_in_vars]
        state_out_vars = node_outputs
        state_out_names = [s.get_name(v) for v in state_out_vars]

        # Build body subgraph
        body_builder = OnnxBuilder(
            name_generator=s.builder.name_generator,
            opset=s.builder.opset,
            model_name=s.builder.get_unique_name("fori_body"),
            initializers=None,
            converter=None,
        )
        body_builder.var_to_symbol_map = s.builder.var_to_symbol_map
        body_conv = s.__class__(body_builder)

        iter_name = body_builder.name_generator.get("iter")
        cond_in_name = body_builder.name_generator.get("cond_in")
        state_in_name = body_builder.name_generator.get("state_in")

        body_builder.add_scalar_input(iter_name, helper.TensorProto.INT64)
        body_builder.add_scalar_input(cond_in_name, helper.TensorProto.BOOL)
        var0 = state_in_vars[0]
        body_builder.add_input(state_in_name, var0.aval.shape, var0.aval.dtype)

        # Map JAXPR invars: (i, state)
        body_conv.var_to_name[body_jaxpr.invars[0]] = iter_name
        body_conv.var_to_name[body_jaxpr.invars[1]] = state_in_name
        # Map any consts
        for cvar, cval in zip(body_jaxpr.constvars, body_consts):
            body_conv.var_to_name[cvar] = body_conv.get_constant_name(cval)

        # Process body and then discard auto‑added outputs
        body_conv._process_jaxpr(body_jaxpr, body_consts)
        body_builder.outputs.clear()
        state_out_name = body_conv.get_name(body_jaxpr.outvars[0])

        # cond_out is just pass‑through of `cond_in`, but must use **different** tensor name
        cond_out_name = body_builder.name_generator.get("cond_out")
        body_builder.add_node(
            helper.make_node("Identity", [cond_in_name], [cond_out_name])
        )
        body_builder.add_output(cond_out_name, (), np.bool_)
        body_builder.add_output(state_out_name, var0.aval.shape, var0.aval.dtype)

        body_graph = body_builder.create_graph(
            body_builder.model_name, is_subgraph=True
        )

        # Top‑level constants
        trip_count_name = s.get_constant_name(np.asarray(n, dtype=np.int64))
        true_name = s.get_constant_name(np.asarray(True, dtype=np.bool_))

        loop_node = helper.make_node(
            "Loop",
            inputs=[trip_count_name, true_name, *state_in_names],
            outputs=state_out_names,
            body=body_graph,
            name=s.get_unique_name("fori_loop"),
        )
        s.add_node(loop_node)
        for name, var in zip(state_out_names, state_out_vars):
            s.add_shape_info(name, var.aval.shape, var.aval.dtype)

    # ------------------------------------------------------------------
    # Monkey‑patch machinery
    # ------------------------------------------------------------------
    @staticmethod
    def _fori_loop_binding(lower, upper, body_fun, init_val):
        if lower != 0:
            raise NotImplementedError(
                "fori_loop plugin currently supports lower==0 only"
            )
        body_closed = jax.make_jaxpr(lambda i, v: body_fun(i, v))(0, init_val)
        n = int(upper - lower)
        flat_init, tree = jax.tree_util.tree_flatten(init_val)
        results = fori_loop_p.bind(
            *flat_init, body_jaxpr=body_closed, trip_count=n, lower=lower
        )
        return jax.tree_util.tree_unflatten(tree, results)

    @staticmethod
    def get_monkey_patch(orig_fn):
        if ForiLoopPlugin._ORIG_FORI_LOOP is None:
            ForiLoopPlugin._ORIG_FORI_LOOP = orig_fn

        def patched(lower, upper, body_fun, init_val):
            return ForiLoopPlugin._fori_loop_binding(lower, upper, body_fun, init_val)

        return patched

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [lax],
            "target_attribute": "fori_loop",
            "patch_function": ForiLoopPlugin.get_monkey_patch,
        }


fori_loop_p.def_abstract_eval(ForiLoopPlugin.abstract_eval)
