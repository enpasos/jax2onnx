# jax2onnx/plugins/jax/lax/cond.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List
import numpy as np
from jax import lax, numpy as jnp
from jax.extend.core import ClosedJaxpr, Literal, Var
from types import SimpleNamespace
from onnx import helper, GraphProto, TensorProto
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
import logging

if TYPE_CHECKING:
    pass

module_logger = logging.getLogger("jax2onnx.plugins.jax.lax.cond")

# Use lax.cond_p directly from JAX
cond_p = lax.cond_p


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
        },
        {
            "testcase": "cond_multiple_operands_in_tuple",
            "callable": lambda: lax.cond(
                True,
                lambda tup: tup[0] + tup[1] - tup[2],
                lambda tup: tup[0] - tup[1] + tup[2],
                (
                    np.array(10, np.int32),
                    np.array(5, np.int32),
                    np.array(2, np.int32),
                ),
            ),
            "input_shapes": [],
            "expected_output_shapes": [()],
        },
        {
            "testcase": "cond_my_new_complex_scenario",
            "callable": lambda op1, op2: lax.cond(
                jnp.all(op1 > 0),
                lambda t: (t[0] * 2 + t[1], jnp.sum(t[0], axis=(-2, -1))),
                lambda t: (t[0] - t[1] * 2, jnp.mean(t[0], axis=(-2, -1))),
                (op1, op2),
            ),
            "input_shapes": [(11, 3, 4), (3, 4)],
            "input_dtypes": [np.float32, np.float32],
            "expected_output_shapes": [(11, 3, 4), (11,)],
            "expected_output_dtypes": [np.float32, np.float32],
        },
        {
            "testcase": "cond_nested_conditional",
            "callable": lambda x, y, z_pred: lax.cond(
                x > 5,
                lambda op: lax.cond(
                    z_pred,
                    lambda inner_op: inner_op * 10,
                    lambda inner_op: inner_op / 10,
                    op,
                ),
                lambda op: op + 100,
                y,
            ),
            "input_shapes": [(), (), ()],
            "input_dtypes": [np.int32, np.float32, np.bool_],
            "expected_output_shapes": [()],
        },
        {
            "testcase": "cond_variables",
            "callable": lambda x, y: lax.cond(
                x > 5,
                lambda op: op - 100,
                lambda op: op + 100,
                y,
            ),
            "input_shapes": [(), ()],
            "input_dtypes": [np.int32, np.float32],
            "expected_output_shapes": [()],
        },
        {
            "testcase": "cond_internal_constant_f64",
            "callable": lambda: lax.cond(
                False,
                lambda x: x * 2.0,
                lambda x: x + 1.0,
                jnp.zeros((2, 4), dtype=jnp.float64),
            ),
            "input_shapes": [],
            "enable_double_precision": True,
            "expected_output_shapes": [(2, 4)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "cond_passthrough_identity",
            "callable": lambda pred, x, y: lax.cond(
                pred,
                lambda op_x, op_y: lax.add(op_x, op_y),
                lambda op_x, op_y: op_x,
                x,
                y,
            ),
            "input_values": [
                np.array(True),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([4.0, 5.0, 6.0], dtype=np.float32),
            ],
        },
        {
            "testcase": "cond_with_scatter",
            "callable": lambda operand, updates: lax.cond(
                True,
                lambda op, upd: lax.scatter_add(
                    op,
                    jnp.array([[1], [3]], dtype=jnp.int32),
                    upd,
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                lambda op, upd: op,
                operand,
                updates,
            ),
            "input_values": [
                np.ones((5, 3), dtype=np.float32),
                np.ones((2, 3), dtype=np.float32) * 9,
            ],
        },
    ],
)
class CondPlugin(PrimitiveLeafPlugin):
    primitive = cond_p

    def to_onnx(
        self,
        conv: Jaxpr2OnnxConverter,
        invars: List[Var],
        outvars: List[Var],
        params: Dict[str, Any],
    ):
        fake_eqn = SimpleNamespace(invars=invars, outvars=outvars, params=params)
        self._to_onnx_impl(conv, fake_eqn)

    def _make_branch(
        self,
        parent_conv: Jaxpr2OnnxConverter,
        tag: str,
        closed: ClosedJaxpr,
        op_names_parent: List[str],
    ) -> GraphProto:
        from jax2onnx.converter.onnx_builder import OnnxBuilder

        sub_builder = OnnxBuilder(
            name_generator=parent_conv.builder.name_generator,
            opset=parent_conv.builder.opset,
            model_name=f"{tag}_body",
            initializers=parent_conv.builder.initializers,
            enable_double_precision=parent_conv.builder.enable_double_precision,
            converter=parent_conv,
        )

        if hasattr(parent_conv.builder, "var_to_symbol_map"):
            sub_builder.var_to_symbol_map.update(parent_conv.builder.var_to_symbol_map)

        sub_conv = parent_conv.__class__(sub_builder)
        sub_conv.symbolic_dim_to_origin = parent_conv.symbolic_dim_to_origin

        for branch_invar, parent_var_name in zip(closed.jaxpr.invars, op_names_parent):
            sub_conv.var_to_name[branch_invar] = parent_var_name

        for cv, cval in zip(closed.jaxpr.constvars, closed.consts):
            sub_conv.var_to_name[cv] = sub_conv.get_constant_name(cval)

        sub_conv._process_jaxpr(closed.jaxpr, closed.consts)

        # --- FIX STARTS HERE ---
        sub_builder.outputs.clear()  # Clear any outputs added by _process_jaxpr

        subgraph_input_names = set(op_names_parent)
        final_output_names = []

        for out_var in closed.jaxpr.outvars:
            out_name = sub_conv.get_name(out_var)
            if out_name in subgraph_input_names:
                identity_out_name = sub_conv.get_unique_name(f"{out_name}_identity")
                node = helper.make_node("Identity", [out_name], [identity_out_name])
                sub_builder.add_node(node)
                final_output_names.append(identity_out_name)
            else:
                final_output_names.append(out_name)

        # Now, correctly define the outputs for the subgraph using the final names
        for name_in_sub, out_var in zip(final_output_names, closed.jaxpr.outvars):
            shape = tuple(
                parent_conv._dim_to_symbol_safe(d) for d in out_var.aval.shape
            )
            dtype = parent_conv._ensure_onnx_dtype(out_var.aval.dtype)
            sub_builder.add_output(name_in_sub, shape, dtype)
        # --- FIX ENDS HERE ---

        graph = sub_builder.create_graph(
            sub_builder.model_name, is_subgraph=True, empty_inputs=True
        )
        return graph

    def _to_onnx_impl(self, conv: Jaxpr2OnnxConverter, eqn: Any):
        pred_var = eqn.invars[0]
        raw_pred_name = conv.get_name(pred_var)

        if isinstance(pred_var, Literal):
            lit_val = pred_var.val
            pred_bool_name = conv.builder.get_unique_name("cond_pred_bool")
            tensor_proto = helper.make_tensor(
                name=pred_bool_name,
                data_type=TensorProto.BOOL,
                dims=[],
                vals=[bool(lit_val != 0)],
            )
            conv.builder.initializers.append(tensor_proto)
            conv.builder.add_value_info(
                pred_bool_name,
                (),
                TensorProto.BOOL,
            )
            condition_input = pred_bool_name

        else:
            pred_dtype = pred_var.aval.dtype
            onnx_dtype = conv._ensure_onnx_dtype(pred_dtype)
            if onnx_dtype != TensorProto.BOOL:
                pred_bool_name = conv.builder.get_unique_name("cond_pred_bool")
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[raw_pred_name],
                    outputs=[pred_bool_name],
                    name=conv.get_unique_name("cast_to_bool"),
                    to=TensorProto.BOOL,
                )
                conv.builder.add_node(cast_node)
                conv.builder.add_value_info(
                    pred_bool_name,
                    (),
                    TensorProto.BOOL,
                )
                condition_input = pred_bool_name
            else:
                condition_input = raw_pred_name

        branch_inputs = eqn.invars[1:]
        branch_input_names = [conv.get_name(v) for v in branch_inputs]

        if "branches" in eqn.params:
            false_closed, true_closed = eqn.params["branches"]
        else:
            true_closed = eqn.params["true_jaxpr"]
            false_closed = eqn.params["false_jaxpr"]

        then_graph = self._make_branch(conv, "then", true_closed, branch_input_names)
        else_graph = self._make_branch(conv, "else", false_closed, branch_input_names)

        out_names = [conv.get_name(v) for v in eqn.outvars]
        if_node = helper.make_node(
            "If",
            inputs=[condition_input],
            outputs=out_names,
            then_branch=then_graph,
            else_branch=else_graph,
            name=conv.get_unique_name("If_cond"),
        )
        conv.builder.add_node(if_node)

        for v, name in zip(eqn.outvars, out_names):
            shape = tuple(conv._dim_to_symbol_safe(d) for d in v.aval.shape)
            dtype = conv._ensure_onnx_dtype(v.aval.dtype)
            conv.builder.add_output(name, shape, dtype)
