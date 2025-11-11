# jax2onnx/plugins/jax/lax/while_loop.py

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from onnx_ir import Shape as IRShape

from jax2onnx.converter.ir_clone import clone_graph
from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._control_flow_utils import (
    builder_cast,
    builder_identity,
    builder_loop,
    clone_value_for_subgraph,
    create_loop_header_inputs,
    ensure_bool_value,
    lower_jaxpr_eqns,
    make_subgraph_context,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _unwrap_closed_jaxpr(jaxpr_like):
    if hasattr(jaxpr_like, "jaxpr") and hasattr(jaxpr_like, "consts"):
        return jaxpr_like.jaxpr, tuple(jaxpr_like.consts)
    return jaxpr_like, ()


def _bind_closed_jaxpr_consts(ctx: "IRContext", jaxpr, consts: Iterable[object]):
    for var, const in zip(getattr(jaxpr, "constvars", ()), consts):
        ctx.bind_const_for_var(var, np.asarray(const))


def _while_loop_multi_state(x: jax.Array) -> jax.Array:
    """Classic two-state loop that updates a tensor and a counter."""

    def cond_fn(state):
        _, count = state
        return count < 5

    def body_fn(state):
        tensor, count = state
        return tensor + 0.1 * tensor**2, count + 1

    return jax.lax.while_loop(cond_fn, body_fn, (x, 0))[0]


def _while_loop_closure_fn(x: jax.Array) -> jax.Array:
    """Loop body closes over a traced tensor to reproduce capture bugs."""

    captured = x * 2.0

    def cond_fn(state):
        return state[0] < 5.0

    def body_fn(state):
        value, counter = state
        return (value + captured, counter)

    return jax.lax.while_loop(cond_fn, body_fn, (0.0, 0))


def _loop_single(x):
    """Scalar accumulator loop used by while-loop metadata tests."""

    return jax.lax.while_loop(lambda v: v < 3, lambda v: v + 1, x)


def _loop_two_state(x):
    """Two-state loop where the second entry is passthrough."""

    init = (x, jnp.int32(0))

    def body(state):
        return (state[0] + 1, state[1])

    return jax.lax.while_loop(lambda state: state[0] < 3, body, init)


def _loop_with_tracer(x):
    """Captured tracer is added every iteration to check alias handling."""

    captured = x * 10

    def body(val):
        return val + captured

    return jax.lax.while_loop(lambda v: v < 30, body, x)


def while_loop_with_scalar_state(x, counter):
    """Model loop carrying a tensor and scalar counter."""

    def cond_fn(state):
        _, i = state
        return i < 5

    def body_fn(state):
        tensor, i = state
        return tensor * 2.0, i + 1

    return jax.lax.while_loop(cond_fn, body_fn, (x, counter))


def loop_with_renamed_passthrough_state(x, counter):
    """Loop that renames passthrough state to guard rewriting logic."""

    def cond_fn(state):
        return state[1] < 5

    def body_fn(state):
        tensor, count = state
        return tensor, count + 1

    return jax.lax.while_loop(cond_fn, body_fn, (x, counter))


def while_loop_mixed_rank_4d_and_scalar(tensor, counter):
    """Carries a 4D tensor and scalar; regresses mixed rank bugs."""

    def cond_fn(state):
        _, c = state
        return c < 5

    def body_fn(state):
        feat, c = state
        return feat * 1.1, c + 1

    return jax.lax.while_loop(cond_fn, body_fn, (tensor, counter))


def _repro_cnn_bug_fn(image, counter):
    """CNN-style reproducer that previously broke scalar typing."""

    def cond_fn(state):
        _, idx = state
        return idx < 5

    def body_fn(state):
        feat, idx = state
        return feat * 0.9 + 0.1, idx + 1

    return jax.lax.while_loop(cond_fn, body_fn, (image, counter))


def _repro_nnx_scalar_and_captured_tensor_bug(tensor_4d, scalar_val):
    """Captures a 4D tensor while only carrying a scalar in the loop."""

    def cond_fn(val):
        return val < 5

    def body_fn(val):
        return val + jnp.mean(tensor_4d).astype(jnp.int32)

    return jax.lax.while_loop(cond_fn, body_fn, scalar_val)


def _no_loop_output_reuse(model) -> bool:
    graph = getattr(model, "graph", None)
    if graph is None:
        return True
    for node in getattr(graph, "node", []):
        if getattr(node, "op_type", "") != "Loop":
            continue
        inputs = set(getattr(node, "input", ()))
        for out in getattr(node, "output", ()):  # type: ignore[arg-type]
            if out in inputs:
                print(
                    f"Loop node '{getattr(node, 'name', '<unnamed>')}' reuses output {out} as input"
                )
                return False
    return True


def _is_literal(var) -> bool:
    return hasattr(var, "val")


def _evaluate_closed_jaxpr(
    ctx: "IRContext", closed_jaxpr, inputs: Sequence[ir.Value], *, prefix: str
) -> list[ir.Value]:
    jaxpr, consts = _unwrap_closed_jaxpr(closed_jaxpr)
    _bind_closed_jaxpr_consts(ctx, jaxpr, consts)
    for var, val in zip(jaxpr.invars, inputs):
        ctx.bind_value_for_var(var, val)
    lower_jaxpr_eqns(ctx, jaxpr)
    outputs: list[ir.Value] = []
    for out_var in jaxpr.outvars:
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name(prefix))
        aval = getattr(out_var, "aval", None)
        if aval is not None:
            shape = tuple(getattr(aval, "shape", ()))
            dtype_enum = _dtype_to_ir(
                np.dtype(getattr(aval, "dtype", np.float32)),
                ctx.builder.enable_double_precision,
            )
            out_val.type = ir.TensorType(dtype_enum)
            _stamp_type_and_shape(out_val, shape)
            _ensure_value_metadata(ctx, out_val)
        outputs.append(out_val)
    return outputs


def _build_loop_body_graph(
    ctx: "IRContext",
    cond_cj,
    body_cj,
    *,
    body_const_template: Sequence[ir.Value],
    state_template: Sequence[ir.Value],
) -> ir.Graph:
    body_ctx = make_subgraph_context(ctx, prefix="while_body")

    iter_input, cond_input = create_loop_header_inputs(
        body_ctx,
        prefix="while_body",
    )

    body_jaxpr, body_consts = _unwrap_closed_jaxpr(body_cj)
    _bind_closed_jaxpr_consts(body_ctx, body_jaxpr, body_consts)

    cond_jaxpr, cond_consts = _unwrap_closed_jaxpr(cond_cj)
    _bind_closed_jaxpr_consts(body_ctx, cond_jaxpr, cond_consts)

    body_invars_iter = iter(body_jaxpr.invars)

    const_inputs: list[ir.Value] = []
    for tmpl_val in body_const_template:
        var = next(body_invars_iter)
        cloned = clone_value_for_subgraph(
            body_ctx,
            tmpl_val,
            name_hint="loop_const_in",
        )
        body_ctx.builder.inputs.append(cloned)
        body_ctx.bind_value_for_var(var, cloned)
        const_inputs.append(cloned)

    state_inputs: list[ir.Value] = []
    for tmpl_val in state_template:
        var = next(body_invars_iter)
        cloned = clone_value_for_subgraph(
            body_ctx,
            tmpl_val,
            name_hint="loop_state_in",
        )
        body_ctx.builder.inputs.append(cloned)
        body_ctx.bind_value_for_var(var, cloned)
        state_inputs.append(cloned)

    # Exhaust iterator to surface mismatched arity early.
    try:
        next(body_invars_iter)
    except StopIteration:
        pass
    else:  # pragma: no cover - defensive programming
        raise ValueError("while_loop body_jaxpr has more invars than expected")

    lower_jaxpr_eqns(body_ctx, body_jaxpr)

    state_outputs = [
        body_ctx.get_value_for_var(
            out_var, name_hint=body_ctx.fresh_name("loop_state_out")
        )
        for out_var in body_jaxpr.outvars
    ]
    for out_var, out_val in zip(body_jaxpr.outvars, state_outputs):
        aval = getattr(out_var, "aval", None)
        if aval is not None:
            shape = tuple(getattr(aval, "shape", ()))
            dtype_enum = _dtype_to_ir(
                np.dtype(getattr(aval, "dtype", np.float32)),
                ctx.builder.enable_double_precision,
            )
            out_val.type = ir.TensorType(dtype_enum)
            _stamp_type_and_shape(out_val, shape)
            _ensure_value_metadata(body_ctx, out_val)

    for var, val in zip(cond_jaxpr.invars, state_outputs):
        body_ctx.bind_value_for_var(var, val)

    lower_jaxpr_eqns(body_ctx, cond_jaxpr)
    cond_out = body_ctx.get_value_for_var(
        cond_jaxpr.outvars[0], name_hint=body_ctx.fresh_name("loop_cond_out")
    )
    cond_out = ensure_bool_value(body_ctx, cond_out, name_hint="loop_cond_bool")

    const_outputs: list[ir.Value] = []
    for tmpl_in, tmpl_val in zip(const_inputs, body_const_template):
        passthrough = builder_identity(
            body_ctx,
            tmpl_in,
            name_hint="loop_const_out",
        )
        dims = getattr(getattr(tmpl_val, "shape", IRShape(())), "dims", None)
        tuple_dims = tuple(dims) if dims is not None else tuple()
        _stamp_type_and_shape(passthrough, tuple_dims)
        _ensure_value_metadata(body_ctx, passthrough)
        const_outputs.append(passthrough)

    body_ctx.builder.outputs = [cond_out] + const_outputs + state_outputs

    body_graph = clone_graph(body_ctx.builder.graph)
    body_graph.name = ctx.fresh_name("while_body")
    opset_imports = dict(body_graph.opset_imports)
    opset_imports.setdefault("", getattr(ctx.builder, "opset", 21))
    body_graph.opset_imports.clear()
    body_graph.opset_imports.update(opset_imports)
    return body_graph


@register_primitive(
    jaxpr_primitive=jax.lax.while_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html",
    onnx=[
        {
            "component": "Loop",
            "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html",
        }
    ],
    since="v0.5.1",
    context="primitives.lax",
    component="while_loop",
    testcases=[
        {
            "testcase": "while_scalar_counter",
            "callable": lambda x: jax.lax.while_loop(
                lambda v: v < 5, lambda v: v + 1, x
            ),
            "input_shapes": [()],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_tuple_state",
            "callable": lambda a, b: jax.lax.while_loop(
                lambda s: s[0] < 3,
                lambda s: (s[0] + 1, s[1] + 2),
                (a, b),
            ),
            "input_shapes": [(), ()],
            "input_dtypes": [np.int32, np.int32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_counter",
            "callable": lambda: jax.lax.while_loop(lambda v: v < 5, lambda v: v + 1, 0),
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.int64],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_vector",
            "callable": lambda: jax.lax.while_loop(
                lambda v: v[0] < 5,
                lambda v: v + 1,
                jnp.array([0], dtype=jnp.int32),
            ),
            "expected_output_shapes": [(1,)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Slice:1 -> Squeeze -> Less -> Loop:1",
                        "inputs": {
                            0: {"const": 9.223372036854776e18},
                            2: {"const": 0.0},
                        },
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_f64",
            "callable": lambda x: jax.lax.while_loop(
                lambda val: val < 5.0, lambda val: val * 1.1, x
            ),
            "input_values": [np.float64(1.0)],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_multi_state_f32",
            "callable": _while_loop_multi_state,
            "input_shapes": [(2,)],
            "input_dtypes": [jnp.float32],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop:2",
                        "inputs": {
                            0: {"const": 9.223372036854776e18},
                            3: {"const": 0.0},
                        },
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_multi_state_f64",
            "callable": _while_loop_multi_state,
            "input_shapes": [(2,)],
            "input_dtypes": [jnp.float64],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop:2",
                        "inputs": {
                            0: {"const": 9.223372036854776e18},
                            3: {"const": 0.0},
                        },
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_with_closure",
            "callable": _while_loop_closure_fn,
            "input_values": [np.float32(1.0)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {
                            0: {"const": 9.223372036854776e18},
                            2: {"const": 0.0},
                            4: {"const": 0.0},
                        },
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_basic",
            "callable": _loop_single,
            "input_values": [np.float32(1.0)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_two_state",
            "callable": _loop_two_state,
            "input_values": [np.float32(1.0)],
            "expected_output_shapes": [(), ()],
            "expected_output_dtypes": [np.float32, np.int32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {
                            0: {"const": 9.223372036854776e18},
                            2: {"const": 0.0},
                        },
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_captured_tracer",
            "callable": _loop_with_tracer,
            "input_values": [np.float32(1.0)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_with_scalar_state",
            "callable": while_loop_with_scalar_state,
            "input_values": [
                np.array([1.0, 2.0], dtype=np.float32),
                np.array(0, dtype=np.int32),
            ],
            "expected_output_shapes": [(2,), ()],
            "expected_output_dtypes": [np.float32, np.int32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    },
                    {
                        "path": "Less -> Loop:2",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    },
                ],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "while_loop_renamed_passthrough",
            "callable": loop_with_renamed_passthrough_state,
            "input_values": [
                np.array([1.0, 2.0], dtype=np.float32),
                np.array(0, dtype=np.int32),
            ],
            "expected_output_shapes": [(2,), ()],
            "expected_output_dtypes": [np.float32, np.int32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_closure_topo",
            "callable": lambda x: jax.lax.while_loop(
                lambda s: s < 3, lambda s: s + (x * 2.0), x
            ),
            "input_values": [np.float32(1.0)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_mixed_rank",
            "callable": lambda: while_loop_mixed_rank_4d_and_scalar(
                np.ones((1, 16, 28, 28), dtype=np.float32), np.int32(0)
            ),
            "expected_output_shapes": [(1, 16, 28, 28), ()],
            "expected_output_dtypes": [np.float32, np.int32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {
                            0: {"const": 9.223372036854776e18},
                            3: {"const": 0.0},
                        },
                    },
                    {
                        "path": "Less -> Loop:1x16x28x28",
                        "inputs": {
                            0: {"const": 9.223372036854776e18},
                            3: {"const": 0.0},
                        },
                    },
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_tracer_passthrough",
            "callable": lambda x: jax.lax.while_loop(
                lambda v: v < 5.0, lambda w: w + (x * 2.0), x
            ),
            "input_values": [np.float32(1.1)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_no_loop_output_reused_as_input",
            "callable": lambda x: jax.lax.while_loop(
                lambda v: v < 5.0, lambda w: w + (x * 2.0), x
            ),
            "input_values": [np.float32(1.0)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: (
                EG(
                    [
                        {
                            "path": "Less -> Loop",
                            "inputs": {0: {"const": 9.223372036854776e18}},
                        }
                    ],
                    no_unused_inputs=True,
                )(model)
                and _no_loop_output_reuse(model)
            ),
        },
        {
            "testcase": "while_loop_4d_and_scalar_state",
            "callable": lambda: while_loop_mixed_rank_4d_and_scalar(
                np.random.default_rng(0)
                .standard_normal((1, 16, 28, 28))
                .astype(np.float32),
                np.int32(0),
            ),
            "expected_output_shapes": [(1, 16, 28, 28), ()],
            "expected_output_dtypes": [np.float32, np.int32],
            "rtol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {
                            0: {"const": 9.223372036854776e18},
                            3: {"const": 0.0},
                        },
                    },
                    {
                        "path": "Less -> Loop:1x16x28x28",
                        "inputs": {
                            0: {"const": 9.223372036854776e18},
                            3: {"const": 0.0},
                        },
                    },
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_cnn_scalar_state_bug",
            "callable": _repro_cnn_bug_fn,
            "input_values": [
                np.ones((1, 3, 28, 28), dtype=np.float32),
                np.int32(0),
            ],
            "expected_output_shapes": [(1, 3, 28, 28), ()],
            "expected_output_dtypes": [np.float32, np.int32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    },
                    {
                        "path": "Less -> Loop:1x3x28x28",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    },
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "while_loop_nnx_repro",
            "callable": _repro_nnx_scalar_and_captured_tensor_bug,
            "input_values": [
                np.ones((2, 3, 28, 28), dtype=np.float32),
                np.int32(0),
            ],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Less -> Loop",
                        "inputs": {0: {"const": 9.223372036854776e18}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class WhileLoopPlugin(PrimitiveLeafPlugin):
    """IR-first lowering for ``lax.while_loop`` via ONNX ``Loop``."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        params = getattr(eqn, "params", {})
        cond_cj = params.get("cond_jaxpr") or params.get("cond_fun")
        body_cj = params.get("body_jaxpr") or params.get("body_fun")
        if cond_cj is None or body_cj is None:
            raise NotImplementedError("while_loop lowering requires cond/body jaxpr")

        cond_nconsts = params.get("cond_nconsts", 0)
        body_nconsts = params.get("body_nconsts", 0)

        input_vals: list[ir.Value] = []
        for var in eqn.invars:
            input_vals.append(
                ctx.get_value_for_var(var, name_hint=ctx.fresh_name("while_input"))
            )

        cond_const_vals = tuple(input_vals[:cond_nconsts])
        body_const_vals = tuple(input_vals[cond_nconsts : cond_nconsts + body_nconsts])
        state_vals = tuple(input_vals[cond_nconsts + body_nconsts :])

        if not state_vals:
            raise NotImplementedError(
                "while_loop lowering requires at least one state variable"
            )

        cond_inputs = list(cond_const_vals) + list(state_vals)
        cond_init = _evaluate_closed_jaxpr(
            ctx, cond_cj, cond_inputs, prefix="while_cond"
        )
        if not cond_init:
            raise RuntimeError("while_loop cond_jaxpr produced no outputs")
        cond_init_val = ensure_bool_value(
            ctx, cond_init[0], name_hint="while_cond_bool"
        )

        body_graph = _build_loop_body_graph(
            ctx,
            cond_cj,
            body_cj,
            body_const_template=body_const_vals,
            state_template=state_vals,
        )

        trip_count = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("while_trip_count"),
            array=np.asarray(np.iinfo(np.int64).max, dtype=np.int64),
        )

        loop_inputs = (
            [trip_count, cond_init_val] + list(body_const_vals) + list(state_vals)
        )

        output_names = [ctx.fresh_name("while_const_out") for _ in body_const_vals]
        output_names.extend(ctx.fresh_name("while_out") for _ in eqn.outvars)

        loop_outputs = builder_loop(
            ctx,
            *loop_inputs,
            body=body_graph,
            output_names=output_names,
        )

        if not isinstance(loop_outputs, tuple):
            loop_outputs = (loop_outputs,)

        const_outputs = loop_outputs[: len(body_const_vals)]
        value_outputs = loop_outputs[len(body_const_vals) :]

        for const_val, const_out in zip(body_const_vals, const_outputs):
            dims = getattr(getattr(const_val, "shape", IRShape(())), "dims", None)
            tuple_dims = tuple(dims) if dims is not None else tuple()
            _stamp_type_and_shape(const_out, tuple_dims)
            _ensure_value_metadata(ctx, const_out)

        for var, val in zip(eqn.outvars, value_outputs):
            aval = getattr(var, "aval", None)
            if aval is None:
                continue
            shape = tuple(getattr(aval, "shape", ()))
            dtype = _dtype_to_ir(
                np.dtype(getattr(aval, "dtype", np.float32)),
                ctx.builder.enable_double_precision,
            )
            val.type = ir.TensorType(dtype)
            _stamp_type_and_shape(val, shape)
            _ensure_value_metadata(ctx, val)
            ctx.bind_value_for_var(var, val)

        if eqn.invars and all(_is_literal(v) for v in eqn.invars):
            for var in eqn.outvars:
                aval = getattr(var, "aval", None)
                if aval is None:
                    continue
                aval_dtype = getattr(aval, "dtype", None)
                if aval_dtype is None or not np.issubdtype(aval_dtype, np.integer):
                    continue
                if np.dtype(aval_dtype) != np.int32:
                    continue
                current_val = ctx.get_value_for_var(var)
                if (
                    getattr(getattr(current_val, "type", None), "dtype", None)
                    == ir.DataType.INT64
                ):
                    continue
                promoted = builder_cast(
                    ctx,
                    current_val,
                    ir.DataType.INT64,
                    name_hint="while_int64",
                )
                _stamp_type_and_shape(promoted, tuple(getattr(aval, "shape", ())))
                _ensure_value_metadata(ctx, promoted)
                ctx.bind_value_for_var(var, promoted)
