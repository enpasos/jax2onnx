from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence

import jax
import numpy as np
from jax import core, lax
from jax.extend.core import Primitive, Var
from onnx import helper

from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

"""
jax2onnx plugin: lax.cond → ONNX If   (single tensor operand / result)
"""

logger = logging.getLogger("jax2onnx.plugins.jax.lax.cond")

cond_p = Primitive("cond")
cond_p.multiple_results = True


@register_primitive(
    jaxpr_primitive=cond_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html",
    onnx=[{"component": "If", "doc": "https://onnx.ai/onnx/operators/onnx__If.html"}],
    since="v0.5.1",
    context="primitives.lax",
    component="cond",
    testcases=[
        {
            "testcase": "cond_scalar",
            "callable": lambda: lax.cond(
                True,
                lambda x: x + 1,
                lambda x: x - 1,
                np.int32(3),
            ),
            "input_shapes": [],
            "expected_output_shapes": [()],
        }
    ],
)
class CondPlugin(PrimitiveLeafPlugin):
    _ORIG_COND: Callable | None = None

    # Only one output – shapes must match between branches
    @staticmethod
    def abstract_eval(pred_aval, *op_aval, true_jaxpr, **__):
        return tuple(op_aval)

    # ------------------------------------------------------------------ #
    # ONNX lowering
    # ------------------------------------------------------------------ #
    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Var],  # [pred, operand]
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ):
        pred_var, op_var = node_inputs
        out_var = node_outputs[0]
        pred_name = s.get_name(pred_var)
        op_name = s.get_name(op_var)
        out_name = s.get_name(out_var)

        true_closed = params["true_jaxpr"]
        false_closed = params["false_jaxpr"]

        def _make_branch(name_prefix: str, closed):
            bb = OnnxBuilder(
                name_generator=s.builder.name_generator,
                opset=s.builder.opset,
                model_name=s.builder.get_unique_name(name_prefix),
            )
            bb.initializers = s.builder.initializers  # <- explicitly link initializers!
            bb.var_to_symbol_map = s.builder.var_to_symbol_map
            bc = s.__class__(bb)

            # Map operand directly to outer‑scope tensor (no graph input!)
            bc.var_to_name[closed.jaxpr.invars[0]] = op_name
            # Closed‑over consts
            for cv, cval in zip(closed.jaxpr.constvars, closed.consts):
                bc.var_to_name[cv] = bc.get_constant_name(cval)

            bc._process_jaxpr(closed.jaxpr, closed.consts)
            bb.outputs.clear()
            branch_out = bc.get_name(closed.jaxpr.outvars[0])
            bb.add_output(branch_out, out_var.aval.shape, out_var.aval.dtype)

            # Register operand's shape/dtype as value_info (optional but tidy)
            bb.add_value_info(op_name, op_var.aval.shape, op_var.aval.dtype)

            bb.inputs.clear()  # ← critical: no formal inputs for If‑branch
            return bb.create_graph(bb.model_name, is_subgraph=True)

        then_graph = _make_branch("then", true_closed)
        else_graph = _make_branch("else", false_closed)

        if_node = helper.make_node(
            "If",
            inputs=[pred_name],
            outputs=[out_name],
            then_branch=then_graph,
            else_branch=else_graph,
            name=s.get_unique_name("cond"),
        )
        s.add_node(if_node)
        s.add_shape_info(out_name, out_var.aval.shape, out_var.aval.dtype)

    # ------------------------------------------------------------------ #
    # Monkey‑patch
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cond_binding(pred, true_fun, false_fun, operand):
        true_closed = jax.make_jaxpr(true_fun)(operand)
        false_closed = jax.make_jaxpr(false_fun)(operand)
        flat, tree = jax.tree_util.tree_flatten(operand)
        res = cond_p.bind(pred, *flat, true_jaxpr=true_closed, false_jaxpr=false_closed)
        return jax.tree_util.tree_unflatten(tree, res)

    @staticmethod
    def get_monkey_patch(orig_fn):
        if CondPlugin._ORIG_COND is None:
            CondPlugin._ORIG_COND = orig_fn

        def patched(pred, true_fun, false_fun, operand):
            return CondPlugin._cond_binding(pred, true_fun, false_fun, operand)

        return patched

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [lax],
            "target_attribute": "cond",
            "patch_function": CondPlugin.get_monkey_patch,
        }


cond_p.def_abstract_eval(CondPlugin.abstract_eval)
