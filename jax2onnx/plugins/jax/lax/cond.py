# jax2onnx/plugins/jax/lax/cond.py

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Iterable, Tuple

import jax
import numpy as np
import jax.numpy as jnp
import onnx_ir as ir
from jax2onnx._compat.jax import ClosedJaxpr, Jaxpr, JaxprEqn, Literal
from jax2onnx.converter.typing_support import (
    LoweringContextProtocol,
    NativeLoweringContext,
)
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_representation import (
    native_shape_contract,
    representation_for_aval,
    validate_native_shapes,
)
from jax2onnx.plugins.jax.lax._control_flow_utils import (
    lower_jaxpr_eqns,
    make_subgraph_context,
)
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


def _unwrap_closed_jaxpr(jaxpr_like: Any) -> Tuple[Any, Iterable[Any]]:
    if hasattr(jaxpr_like, "jaxpr") and hasattr(jaxpr_like, "consts"):
        return jaxpr_like.jaxpr, getattr(jaxpr_like, "consts")
    return jaxpr_like, ()


def _extract_branches(
    params: dict[str, Any],
) -> Tuple[Tuple[Any, Iterable[Any]], Tuple[Any, Iterable[Any]]]:
    if "branches" in params:
        false_closed, true_closed = params["branches"]
    else:
        true_closed = params["true_jaxpr"]
        false_closed = params["false_jaxpr"]
    true_jaxpr, true_consts = _unwrap_closed_jaxpr(true_closed)
    false_jaxpr, false_consts = _unwrap_closed_jaxpr(false_closed)
    return (true_jaxpr, true_consts), (false_jaxpr, false_consts)


def _shape_for_aval(ctx: LoweringContextProtocol, aval: Any) -> tuple[Any, ...]:
    if ctx.precision_policy is PrecisionPolicy.SOURCE:
        return representation_for_aval(aval).physical_shape
    return tuple(getattr(aval, "shape", ()) or ())


def _same_native_aval(left: Any, right: Any) -> bool:
    a, b = native_shape_contract(left.shape), native_shape_contract(right.shape)
    same_symbol = (
        a.symbol is None
        and b.symbol is None
        or a.symbol is not None
        and b.symbol is not None
        and a.symbol[0] == b.symbol[0]
        and a.symbol[1] is b.symbol[1]
    )
    return bool(
        left.dtype == right.dtype
        and left.weak_type == right.weak_type
        and a.shape == b.shape
        and a.symbol_axis == b.symbol_axis
        and same_symbol
    )


def _source_literal_index(eqn: JaxprEqn) -> int | None:
    if not eqn.invars:
        raise ValueError("SOURCE cond requires its original selector")
    selector = eqn.invars[0]
    if not isinstance(selector, Literal):
        return None
    if eqn.primitive is not jax.lax.cond_p:
        raise ValueError("SOURCE literal cond requires the actual native primitive")
    aval = selector.aval
    if (
        np.dtype(aval.dtype) != np.dtype(np.int32)
        or tuple(aval.shape) != ()
        or aval.weak_type is not False
    ):
        raise ValueError("SOURCE literal cond requires a strong scalar int32 selector")
    # Python integer literals obtain their width from the native declaration.
    # Explicit array/scalar payloads must already carry that exact width.
    if type(selector.val) is int:
        index = selector.val
    else:
        value = np.asarray(selector.val)
        if value.dtype != np.dtype(np.int32) or value.shape != ():
            raise ValueError(
                "SOURCE literal cond selector payload disagrees with its declaration"
            )
        index = int(value.item())
    if index not in (0, 1):
        raise ValueError("SOURCE literal cond selector must be exactly 0 or 1")
    if (
        type(eqn.params) is not dict
        or set(eqn.params) != {"branches"}
        or type(eqn.params["branches"]) is not tuple
        or len(eqn.params["branches"]) != 2
        or eqn.effects
    ):
        raise ValueError(
            "SOURCE literal cond requires the original pure two-branch schema"
        )
    for closed in eqn.params["branches"]:
        if not isinstance(closed, (ClosedJaxpr, Jaxpr)):
            raise ValueError(
                "SOURCE literal cond requires original native branch objects"
            )
        body, consts = _unwrap_closed_jaxpr(closed)
        if (
            not isinstance(body, Jaxpr)
            or body.effects
            or not isinstance(consts, (tuple, list))
            or len(body.constvars) != len(consts)
            or len(body.invars) != len(eqn.invars) - 1
            or len(body.outvars) != len(eqn.outvars)
            or not all(
                _same_native_aval(inner.aval, outer.aval)
                for inner, outer in zip(body.invars, eqn.invars[1:], strict=True)
            )
            or not all(
                _same_native_aval(inner.aval, outer.aval)
                for inner, outer in zip(body.outvars, eqn.outvars, strict=True)
            )
        ):
            raise ValueError(
                "SOURCE literal cond branch differs from its native ports, constants or effects"
            )
        validate_native_shapes(
            [
                var.aval.shape
                for var in (*eqn.invars[1:], *eqn.outvars, *body.invars, *body.outvars)
            ]
        )
    return index


def _require_source_value(value: ir.Value, aval: Any, role: str) -> None:
    if not representation_for_aval(aval).matches_ir_value(value):
        raise ValueError(
            f"SOURCE literal cond {role} differs from its native physical representation"
        )


def _require_source_producer(value: ir.Value, graph: ir.Graph, role: str) -> None:
    producer = value.producer()
    if (
        producer is None
        or producer.graph is not graph
        or not any(node is producer for node in graph)
    ):
        raise ValueError(
            f"SOURCE literal cond {role} has no producer in its actual graph"
        )


def _source_identity(
    ctx: NativeLoweringContext, value: ir.Value, aval: Any, *, name: str
) -> ir.Value:
    _require_source_value(value, aval, "Identity input")
    output = ir.Value(name=ctx.fresh_name(name), type=value.type, shape=value.shape)
    result = ctx.builder.op("Identity", [value], output=output)
    _require_source_value(result, aval, "Identity result")
    return result


def _attach_selected_branch(
    ctx: NativeLoweringContext, graph: ir.Graph, eqn: JaxprEqn
) -> None:
    if graph.inputs or graph.initializers:
        raise ValueError(
            "SOURCE literal cond staging must keep implicit captures and Constant nodes"
        )
    if len(graph.outputs) != len(eqn.outvars):
        raise ValueError("SOURCE literal cond staged result arity changed")
    for value, var in zip(graph.outputs, eqn.outvars, strict=True):
        _require_source_value(value, var.aval, "staged result")
        _require_source_producer(value, graph, "staged result")
    imports = ctx.builder.graph.opset_imports
    for domain, version in graph.opset_imports.items():
        if (domain == "" and version != ctx.opset) or (
            domain in imports and imports[domain] != version
        ):
            raise ValueError(
                "SOURCE literal cond cannot change the caller's opset imports"
            )
    outputs = [
        ctx.allocate_value_for_var(var, name_hint=ctx.fresh_name("cond_literal_out"))
        for var in eqn.outvars
    ]
    for value, var in zip(outputs, eqn.outvars, strict=True):
        _require_source_value(value, var.aval, "allocated result")
        if (
            value.producer() is not None
            or value.is_graph_input()
            or value.is_initializer()
            or value.const_value is not None
        ):
            raise ValueError(
                "SOURCE literal cond requires fresh writable result placeholders"
            )
    if len({id(value) for value in outputs}) != len(outputs):
        raise ValueError(
            "SOURCE literal cond requires distinct outer result placeholders"
        )
    ledger = getattr(ctx, "_source_structure", None)
    if ledger is not None:
        from jax2onnx.plugins._source_control_flow_structure import (
            prepare_literal_if_clone,
        )

        ledger._observe(
            "literal-if-clone", lambda: prepare_literal_if_clone(ctx, graph)
        )
    selected = graph.clone(allow_outer_scope_values=True)
    returned = tuple(selected.outputs)
    if selected.inputs or selected.initializers or len(returned) != len(eqn.outvars):
        raise ValueError("SOURCE literal cond clone changed its staged interface")
    for value, var in zip(returned, eqn.outvars, strict=True):
        _require_source_value(value, var.aval, "cloned result")
        _require_source_producer(value, selected, "cloned result")
    nodes = tuple(selected)
    selected.outputs.clear()
    selected.remove(nodes, safe=False)
    for domain, version in graph.opset_imports.items():
        imports[domain] = version
    for node in nodes:
        if node.graph is not None:
            raise ValueError("SOURCE literal cond staged node is still graph-owned")
        ctx.builder.add_node_obj(node)
        if node.graph is not ctx.builder.graph:
            raise ValueError("SOURCE literal cond node did not join the caller graph")
    for source, output, var in zip(returned, outputs, eqn.outvars, strict=True):
        _require_source_value(source, var.aval, "attached result")
        _require_source_producer(source, ctx.builder.graph, "attached result")
        result = ctx.builder.op("Identity", [source], output=output)
        _require_source_value(result, var.aval, "outer result")
        producer = result.producer()
        if producer is None or producer.graph is not ctx.builder.graph:
            raise ValueError("SOURCE literal cond outer result is disconnected")
        ctx.bind_value_for_var(var, result)
    if ledger is not None:
        from jax2onnx.plugins._source_control_flow_structure import (
            record_literal_if_attachment,
        )

        ledger._observe(
            "literal-if-attachment",
            lambda: record_literal_if_attachment(ctx, eqn, graph, nodes, returned),
        )


@register_native_primitive(
    jaxpr_primitive=jax.lax.cond_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html",
    onnx=[
        {
            "component": "If",
            "doc": "https://onnx.ai/onnx/operators/onnx__If.html",
        }
    ],
    since="0.5.1",
    context="primitives.lax",
    component="cond",
    testcases=[
        {
            "testcase": "cond_scalar",
            "callable": lambda: jax.lax.cond(
                True,
                lambda x: x + 1,
                lambda x: x - 1,
                np.int32(3),
            ),
            "input_shapes": [],
            "expected_output_shapes": [()],
            "post_check_onnx_graph": EG(
                ["Cast -> If"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "cond_multiple_operands_in_tuple",
            "callable": lambda: jax.lax.cond(
                True,
                lambda tup: tup[0] + tup[1] - tup[2],
                lambda tup: tup[0] - tup[1] + tup[2],
                (np.array(10, np.int32), np.array(5, np.int32), np.array(2, np.int32)),
            ),
            "input_shapes": [],
            "expected_output_shapes": [()],
            "post_check_onnx_graph": EG(
                ["Cast -> If"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "cond_my_new_complex_scenario",
            "callable": lambda op1, op2: jax.lax.cond(
                jnp.all(op1 > 0),
                lambda t: (t[0] * 2 + t[1], jnp.sum(t[0], axis=(-2, -1))),
                lambda t: (t[0] - t[1] * 2, jnp.sum(t[0], axis=(-2, -1))),
                (op1, op2),
            ),
            "input_shapes": [(11, 3, 4), (3, 4)],
            "input_dtypes": [np.float32, np.float32],
            "expected_output_shapes": [(11, 3, 4), (11,)],
            "expected_output_dtypes": [np.float32, np.float32],
            "post_check_onnx_graph": EG(
                [
                    "Greater:11x3x4 -> Cast:11x3x4 -> ReduceMin -> Cast -> If",
                ],
            ),
        },
        {
            "testcase": "cond_nested_conditional",
            "callable": lambda x, y, z_pred: jax.lax.cond(
                x > 5,
                lambda op: jax.lax.cond(
                    z_pred,
                    lambda inner_op: inner_op * 10,
                    lambda inner_op: inner_op - 10,
                    op,
                ),
                lambda op: op + 100,
                y,
            ),
            "input_shapes": [(), (), ()],
            "input_dtypes": [np.int32, np.float32, np.bool_],
            "expected_output_shapes": [()],
            "post_check_onnx_graph": EG(["Greater -> If"]),
        },
        {
            "testcase": "cond_variables",
            "callable": lambda x, y: jax.lax.cond(
                x > 5,
                lambda op: op - 100,
                lambda op: op + 100,
                y,
            ),
            "input_shapes": [(), ()],
            "input_dtypes": [np.int32, np.float32],
            "expected_output_shapes": [()],
            "post_check_onnx_graph": EG(["Greater -> If"]),
        },
        {
            "testcase": "cond_internal_constant_f64",
            "callable": lambda: jax.lax.cond(
                False,
                lambda x: x * 2.0,
                lambda x: x + 1.0,
                jnp.zeros((2, 4), dtype=jnp.float64),
            ),
            "input_shapes": [],
            "expected_output_shapes": [(2, 4)],
            "expected_output_dtypes": [np.float64],
            "enable_double_precision": True,
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Cast -> If:2x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "cond_passthrough_identity",
            "callable": lambda pred, x, y: jax.lax.cond(
                pred,
                lambda op_x, op_y: jax.lax.add(op_x, op_y),
                lambda op_x, op_y: op_x,
                x,
                y,
            ),
            "input_values": [
                np.array(True),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([4.0, 5.0, 6.0], dtype=np.float32),
            ],
            "post_check_onnx_graph": EG(["If:3"]),
        },
        {
            "testcase": "cond_with_scatter",
            "callable": lambda operand, updates: jax.lax.cond(
                True,
                lambda op, upd: jax.lax.scatter_add(
                    op,
                    jnp.array([[1], [3]], dtype=jnp.int32),
                    upd,
                    jax.lax.ScatterDimensionNumbers(
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
            "post_check_onnx_graph": EG(["Cast -> If:5x3"]),
        },
    ],
)
class CondPlugin(PrimitiveLeafPlugin):
    """IR-first lowering for ``lax.cond`` to ONNX ``If``."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        if ctx.precision_policy is PrecisionPolicy.SOURCE:
            index = _source_literal_index(eqn)
            if index is not None:
                self._lower_source_literal(ctx, eqn, index)
                return
        cond_var, *operand_vars = eqn.invars
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for lax.cond lowering"
            )

        cond_val = ctx.get_value_for_var(
            cond_var, name_hint=ctx.fresh_name("cond_pred")
        )
        needs_cast = True
        aval_dtype = getattr(getattr(cond_var, "aval", None), "dtype", None)
        if aval_dtype is not None and np.issubdtype(np.dtype(aval_dtype), np.bool_):
            needs_cast = False
        elif (
            hasattr(cond_val, "type")
            and getattr(cond_val.type, "dtype", None) == ir.DataType.BOOL
        ):
            needs_cast = False

        if needs_cast:
            cond_val = builder.Cast(
                cond_val,
                _outputs=[ctx.fresh_name("cond_pred_bool")],
                to=int(ir.DataType.BOOL.value),
            )
            cond_val.type = ir.TensorType(ir.DataType.BOOL)
            _stamp_type_and_shape(
                cond_val, tuple(getattr(getattr(cond_var, "aval", None), "shape", ()))
            )
            _ensure_value_metadata(ctx, cond_val)
        else:
            _stamp_type_and_shape(
                cond_val, tuple(getattr(getattr(cond_var, "aval", None), "shape", ()))
            )
            _ensure_value_metadata(ctx, cond_val)

        branch_input_vals = [
            ctx.get_value_for_var(v, name_hint=ctx.fresh_name("cond_operand"))
            for v in operand_vars
        ]
        for var, val in zip(operand_vars, branch_input_vals):
            _stamp_type_and_shape(val, _shape_for_aval(ctx, var.aval))
            _ensure_value_metadata(ctx, val)

        (true_jaxpr, true_consts), (false_jaxpr, false_consts) = _extract_branches(
            eqn.params
        )

        ledger = getattr(ctx, "_source_structure", None)
        with (
            ledger.region("true", branch_input_vals)
            if ledger is not None
            else nullcontext(None)
        ) as true_region:
            then_graph = self._build_branch_graph(
                ctx,
                true_jaxpr,
                true_consts,
                branch_input_vals,
                prefix="cond_then",
            )
        with (
            ledger.region("false", branch_input_vals)
            if ledger is not None
            else nullcontext(None)
        ) as false_region:
            else_graph = self._build_branch_graph(
                ctx,
                false_jaxpr,
                false_consts,
                branch_input_vals,
                prefix="cond_else",
            )

        output_names = [ctx.fresh_name("cond_out") for _ in eqn.outvars]
        if_outputs = builder.If(
            cond_val,
            then_branch=then_graph,
            else_branch=else_graph,
            _outputs=output_names,
        )

        if not isinstance(if_outputs, tuple):
            if_outputs = (if_outputs,)

        enable_dp = getattr(builder, "enable_double_precision", False)
        for var, out_val in zip(eqn.outvars, if_outputs):
            aval = getattr(var, "aval", None)
            dtype_enum = None
            if ctx.precision_policy is PrecisionPolicy.SOURCE:
                dtype_enum = representation_for_aval(aval).ir_dtype
            elif aval is not None:
                aval_dtype = getattr(aval, "dtype", None)
                if aval_dtype is not None:
                    try:
                        dtype_enum = _dtype_to_ir(np.dtype(aval_dtype), enable_dp)
                    except TypeError:
                        dtype_enum = None
            if dtype_enum is not None:
                out_val.type = ir.TensorType(dtype_enum)
            _stamp_type_and_shape(out_val, _shape_for_aval(ctx, aval))
            _ensure_value_metadata(ctx, out_val)
            ctx.bind_value_for_var(var, out_val)

        if ledger is not None and true_region is not None and false_region is not None:
            from jax2onnx.plugins._source_control_flow_structure import (
                record_if_attachment,
            )

            ledger._observe(
                "if-attachment",
                lambda: record_if_attachment(
                    ctx,
                    eqn,
                    true_region,
                    false_region,
                    then_graph,
                    else_graph,
                ),
            )

    def _lower_source_literal(
        self, ctx: LoweringContextProtocol, eqn: JaxprEqn, index: int
    ) -> None:
        operands = [ctx.require_value_for_var(var) for var in eqn.invars[1:]]
        for value, var in zip(operands, eqn.invars[1:], strict=True):
            _require_source_value(value, var.aval, "operand")
        (true_body, true_consts), (false_body, false_consts) = _extract_branches(
            eqn.params
        )
        # Both original isolated lowerings remain necessary: actual named
        # Function bodies validate on retrace, beyond their raw call markers.
        ledger = getattr(ctx, "_source_structure", None)
        with (
            ledger.region("true", operands) if ledger is not None else nullcontext(None)
        ):
            then_graph = self._build_branch_graph(
                ctx,
                true_body,
                true_consts,
                operands,
                prefix="cond_then",
                source_literal=True,
            )
        with (
            ledger.region("false", operands)
            if ledger is not None
            else nullcontext(None)
        ):
            else_graph = self._build_branch_graph(
                ctx,
                false_body,
                false_consts,
                operands,
                prefix="cond_else",
                source_literal=True,
            )
        _attach_selected_branch(ctx, then_graph if index else else_graph, eqn)

    def _build_branch_graph(
        self,
        ctx: LoweringContextProtocol,
        branch_jaxpr: Any,
        consts: Iterable[Any],
        operand_vals: list[ir.Value],
        *,
        prefix: str,
        source_literal: bool = False,
    ) -> ir.Graph:
        branch_ctx = make_subgraph_context(ctx, prefix=prefix, implicit_captures=True)

        # Bind constants inside the branch context.
        for const_var, const_value in zip(branch_jaxpr.constvars, consts):
            branch_ctx.bind_const_for_var(
                const_var,
                const_value if source_literal else np.asarray(const_value),
            )

        builder = getattr(branch_ctx, "builder", None)
        if builder is None:
            raise AttributeError("Branch context missing builder for lax.cond")

        branch_inputs: list[ir.Value] = []
        for outer_val, inner_var in zip(operand_vals, branch_jaxpr.invars):
            if source_literal:
                capture = _source_identity(
                    branch_ctx, outer_val, inner_var.aval, name="cond_capture"
                )
                branch_ctx.bind_value_for_var(inner_var, capture)
                branch_inputs.append(capture)
                continue
            capture = builder.Identity(
                outer_val,
                _outputs=[branch_ctx.fresh_name("cond_capture")],
            )
            orig_type = getattr(outer_val, "type", None)
            if orig_type is not None:
                capture.type = orig_type
            _stamp_type_and_shape(capture, _shape_for_aval(branch_ctx, inner_var.aval))
            _ensure_value_metadata(branch_ctx, capture)
            branch_ctx.bind_value_for_var(inner_var, capture)
            branch_inputs.append(capture)

        lower_jaxpr_eqns(branch_ctx, branch_jaxpr, source="cond")

        input_names = {val.name for val in branch_inputs}
        # Snapshot original Values before stamping or adding any result bridges.
        source_results: list[ir.Value] = []
        source_uses: dict[int, int] = {}
        if ctx.precision_policy is PrecisionPolicy.SOURCE:
            source_results = [
                branch_ctx.get_value_for_var(var) for var in branch_jaxpr.outvars
            ]
            for value in source_results:
                source_uses[id(value)] = source_uses.get(id(value), 0) + 1
        branch_outputs: list[ir.Value] = []
        for index, out_var in enumerate(branch_jaxpr.outvars):
            val = (
                source_results[index]
                if ctx.precision_policy is PrecisionPolicy.SOURCE
                else branch_ctx.get_value_for_var(out_var)
            )
            duplicate_computed = source_uses.get(id(val), 0) > 1 and all(
                val is not captured for captured in branch_inputs
            )
            if source_literal or duplicate_computed:
                _require_source_value(
                    val, out_var.aval, "branch result before stamping"
                )
                _require_source_producer(
                    val, branch_ctx.builder.graph, "branch result before stamping"
                )
            if duplicate_computed:
                val = _source_identity(
                    branch_ctx, val, out_var.aval, name="cond_result_alias"
                )
                _require_source_producer(
                    val, branch_ctx.builder.graph, "duplicate result bridge"
                )
                branch_outputs.append(val)
                continue
            if val.name in input_names:
                if source_literal:
                    val = _source_identity(
                        branch_ctx, val, out_var.aval, name="cond_alias_out"
                    )
                    branch_outputs.append(val)
                    continue
                orig_type = getattr(val, "type", None)
                val = builder.Identity(
                    val,
                    _outputs=[branch_ctx.fresh_name(f"{val.name}_identity")],
                )
                if orig_type is not None:
                    val.type = orig_type
            branch_outputs.append(val)
            _stamp_type_and_shape(val, _shape_for_aval(branch_ctx, out_var.aval))
            _ensure_value_metadata(branch_ctx, val)

        branch_ctx.builder.outputs = branch_outputs

        branch_graph = branch_ctx.builder.graph.clone(allow_outer_scope_values=True)
        branch_graph.name = ctx.fresh_name(prefix)
        branch_graph.inputs.clear()
        opset_imports = dict(branch_graph.opset_imports)
        opset_imports.setdefault("", getattr(ctx.builder, "opset", 21))
        branch_graph.opset_imports.clear()
        branch_graph.opset_imports.update(opset_imports)
        typed_graph: ir.Graph = branch_graph
        return typed_graph
