# jax2onnx/plugins/jax/lax/while_loop.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence, Callable, List

import jax
import numpy as np
from jax import core, lax
from jax.extend.core import Primitive, Literal
from onnx import TensorProto, helper

from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.lax.while_loop")


def _while_loop_multi_state_fn(x):
    """Test helper: a two‐state while_loop."""
    steps = 5

    def cond_fn(state):
        _, cnt = state
        return cnt < steps

    def body_fn(state):
        xx, cnt = state
        return xx + 0.1 * xx**2, cnt + 1

    return lax.while_loop(cond_fn, body_fn, (x, 0))[0]


def _while_loop_closure_fn(x: jax.Array) -> jax.Array:
    """Test helper: a while_loop that closes over a traced variable."""
    y = x * 2.0

    def cond_fn(state):
        return state[0] < 5.0

    def body_fn(state):
        # The body uses the closed-over, traced variable `y`.
        return (state[0] + y, state[1])

    # The initial value is (0.0, 0), but the loop's behavior depends on `y`.
    return lax.while_loop(cond_fn, body_fn, (0.0, 0))


def _loop_single(x):
    """Test helper: simple loop with one state variable, no captured tracer."""
    return lax.while_loop(lambda v: v < 3, lambda v: v + 1, x)


def _loop_two_state(x):
    """Test helper: loop with two state vars (one passthrough)."""
    return lax.while_loop(
        lambda s: s[0] < 3,
        lambda s: (s[0] + 1, s[1]),  # second var passthrough
        (x, jax.numpy.int32(0)),
    )


def _loop_with_tracer(x):
    """Test helper: loop with a captured tracer passed through unchanged."""
    y = x * 10  # captured tracer

    def body(s):
        return s + y

    return lax.while_loop(lambda v: v < 30, body, x)


def no_loop_output_reused_as_input(model):
    for node in model.graph.node:
        if node.op_type != "Loop":
            continue
        input_names = set(node.input)
        for out in node.output:
            if out in input_names:
                print(f"❌ Loop node '{node.name}' has output reused as input: {out}")
                return False
    return True


def _const_as_int64(builder, const_name):
    """
    Insert `Cast` so that the scalar constant becomes INT64 and
    return the new tensor name.
    """
    new_name = builder.get_unique_name(f"{const_name}_to_i64")
    builder.add_node(
        helper.make_node(
            "Cast",
            inputs=[const_name],
            outputs=[new_name],
            name=builder.get_unique_name("cast_const_to_i64"),
            to=TensorProto.INT64,
        )
    )
    builder.add_shape_info(new_name, (), np.int64)
    return new_name


def while_loop_with_scalar_state_body_fun(val):
    x, i = val
    return x * 2, i + 1


def while_loop_with_scalar_state_cond_fun(val):
    _, i = val
    return i < 5


def while_loop_with_scalar_state(x, i):
    return jax.lax.while_loop(
        while_loop_with_scalar_state_cond_fun,
        while_loop_with_scalar_state_body_fun,
        (x, i),
    )


# Add these helper functions with the others at the top of the file
def loop_with_renamed_passthrough_state_body(state):
    tensor_val, counter_val = state
    return tensor_val, counter_val + 1


def loop_with_renamed_passthrough_state_cond(state):
    _, counter_val = state
    return counter_val < 5


def loop_with_renamed_passthrough_state(x, y):
    return lax.while_loop(
        loop_with_renamed_passthrough_state_cond,
        loop_with_renamed_passthrough_state_body,
        (x, y),
    )


# define a new primitive and give it multiple results
lax.while_loop_p = Primitive("lax.while_loop")
lax.while_loop_p.multiple_results = True


@register_primitive(
    jaxpr_primitive=lax.while_loop_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html",
    onnx=[
        {"component": "Loop", "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html"}
    ],
    since="v0.5.1",
    context="primitives.lax",
    component="while_loop",
    testcases=[
        {
            "testcase": "while_loop_counter",
            "callable": lambda: lax.while_loop(lambda v: v < 5, lambda v: v + 1, 0),
            "input_shapes": [],
            "expected_output_shapes": [()],
            "post_check_onnx_graph": lambda m: (
                __import__("onnx").checker.check_model(m) or True
            ),
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
            "post_check_onnx_graph": lambda m: (
                __import__("onnx").checker.check_model(m) or True
            ),
        },
        {
            "testcase": "while_loop_f64",
            "callable": lambda x: lax.while_loop(
                lambda val: val < 5.0, lambda val: val * 1.1, x
            ),
            "input_values": [np.float64(1.0)],
            "expected_output_shapes": [()],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": lambda m: (
                __import__("onnx").checker.check_model(m) or True
            ),
        },
        {
            "testcase": "while_loop_multi_state_f32",
            "callable": _while_loop_multi_state_fn,
            "input_shapes": [(2,)],
            "input_dtypes": [np.float32],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: (
                __import__("onnx").checker.check_model(m) or True
            ),
        },
        {
            "testcase": "while_loop_multi_state_f64",
            "callable": _while_loop_multi_state_fn,
            "input_shapes": [(2,)],
            "input_dtypes": [np.float64],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": lambda m: (
                __import__("onnx").checker.check_model(m) or True
            ),
        },
        {
            "testcase": "while_loop_with_closure",
            "callable": _while_loop_closure_fn,
            "input_values": [np.float32(1.0)],
            "post_check_onnx_graph": lambda m: (
                __import__("onnx").checker.check_model(m) or True
            ),
        },
        {
            "testcase": "while_loop_basic",
            "callable": _loop_single,
            "input_values": [np.float32(1.0)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: (
                __import__("onnx").checker.check_model(m) or True
            ),
        },
        {
            "testcase": "while_loop_two_state",
            "callable": _loop_two_state,
            "input_values": [np.float32(1.0)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: (
                __import__("onnx").checker.check_model(m) or True
            ),
        },
        {
            "testcase": "while_loop_captured_tracer",
            "callable": _loop_with_tracer,
            "input_values": [np.float32(1.0)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: (
                __import__("onnx").checker.check_model(m) or True
            ),
        },
        {
            "testcase": "while_loop_with_scalar_state",
            "callable": while_loop_with_scalar_state,
            "input_values": [
                np.array([1.0, 2.0], dtype=np.float32),
                np.array(0, dtype=np.int32),
            ],
            "expected_output_dtypes": [np.float32, np.int32],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "while_loop_renamed_passthrough",
            "callable": loop_with_renamed_passthrough_state,
            "input_values": [
                np.array([1.0, 2.0], dtype=np.float32),
                np.array(0, dtype=np.int32),
            ],
            "expected_output_dtypes": [np.float32, np.int32],
            "expected_output_shapes": [(2,), ()],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "while_loop_closure_topo",
            "callable": (
                lambda x: lax.while_loop(lambda s: s < 3, lambda s: s + (x * 2.0), x)
            ),
            "input_values": [np.float32(1.0)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: (
                __import__("onnx").checker.check_model(m) or True
            ),
        },
        # {
        #     "testcase": "while_loop_closure_has_y_input",
        #     "callable": (
        #         lambda x: lax.while_loop(lambda s: s < 3, lambda s: s + (x * 2.0), x)
        #     ),
        #     "input_shapes": [()],
        #     "input_dtypes": [np.float32],
        #     "run_only_f32_variant": True,
        #     "post_check_onnx_graph": (
        #         lambda model: [
        #             body
        #             for n in model.graph.node
        #             if n.op_type == "Loop"
        #             for attr in n.attribute
        #             if attr.name == "body"
        #             for body in [attr.g]
        #         ][0].input.__len__()
        #         >= 4
        #     ),
        # },
        {
            "testcase": "while_loop_tracer_passthrough",
            "callable": (
                lambda x: (
                    lax.while_loop(lambda v: v < 5.0, lambda w: w + (x * 2.0), x)
                )
            ),
            "input_values": [np.float32(1.1)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "while_loop_no_loop_output_reused_as_input",
            "callable": (
                lambda x: (
                    lax.while_loop(lambda v: v < 5.0, lambda w: w + (x * 2.0), x)
                )
            ),
            "input_values": [np.float32(1.0)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": no_loop_output_reused_as_input,
        },
        # {
        #     "testcase": "while_loop_with_closure2",
        #     "callable": _while_loop_closure_fn,
        #     "input_shapes": [("B",)],  # symbolic batch dim
        #     "post_check_onnx_graph": lambda m: (
        #         __import__("onnx").checker.check_model(m) or True
        #     ),
        # },
    ],
)
class WhileLoopPlugin(PrimitiveLeafPlugin):
    _ORIG: Callable | None = None

    @staticmethod
    def abstract_eval(*in_avals: core.AbstractValue, **kwargs):
        # just pass through all the loop‐carried args
        return tuple(in_avals)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ):
        # unpack the closed JAXPRs
        cond_closed = params["cond_jaxpr"]
        body_closed = params["body_jaxpr"]
        c_jaxpr, c_consts = cond_closed.jaxpr, cond_closed.consts
        b_jaxpr, b_consts = body_closed.jaxpr, body_closed.consts

        # -----------------------------------------------------
        # ❶ Find any scalar int32 loop-carried inputs → promote to INT64
        # -----------------------------------------------------
        promoted_idxs: List[int] = []

        def _is_int_scalar(var):
            return (
                var.aval.shape == ()
                and np.issubdtype(var.aval.dtype, np.integer)
                and var.aval.dtype != np.int64
            )

        for i, vin in enumerate(node_inputs):
            if _is_int_scalar(vin):
                promoted_idxs.append(i)
        need_int64_consts = bool(promoted_idxs)

        # --------------------------------------------------
        # Helper: transparently upgrade scalar int constants
        # --------------------------------------------------
        def _wrap_get_constant_name(builder):
            """Replace builder.get_constant_name so that any scalar
            INT{8,16,32} literal is promoted to INT64 *before* the
            constant initialiser is created."""
            orig_get = builder.get_constant_name

            def wrapped(val, *a, **kw):
                # This wrapper should ONLY interfere if we are in a promotion context
                # AND we've encountered a Literal that needs promoting.
                if need_int64_consts and isinstance(val, Literal):
                    # Use val.aval.dtype, which is the correct way to get a Literal's type
                    aval = val.aval
                    if (
                        aval.shape == ()
                        and np.issubdtype(aval.dtype, np.integer)
                        and aval.dtype != np.int64
                    ):
                        # It's a promotable integer literal. Promote its value and pass to the original function.
                        promoted_val = np.int64(val.val)
                        return orig_get(promoted_val, *a, **kw)

                # For all other cases, including non-Literal values or Literals that don't need promotion,
                # call the original function without modifying the value.
                return orig_get(val, *a, **kw)

            builder.get_constant_name = wrapped

        if need_int64_consts:  # wrap once
            _wrap_get_constant_name(s.builder)

        # -----------------------------------------------------
        # ❷ Build state_in names, inserting Cast→INT64 before the Loop
        # -----------------------------------------------------
        state_in: List[str] = [s.get_name(v) for v in node_inputs]
        for idx in promoted_idxs:
            orig = state_in[idx]
            cast64 = s.get_unique_name(f"{orig}_to_i64")
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[orig],
                    outputs=[cast64],
                    name=s.get_unique_name("cast_to_i64"),
                    to=TensorProto.INT64,
                )
            )
            s.add_shape_info(cast64, (), np.int64)
            state_in[idx] = cast64

        # Prepare placeholder for loop outputs
        state_out: List[str] = []
        for v in node_outputs:
            nm = s.get_name(v)
            # If the symbol is also an initializer (compile-time constant),
            # we MUST give the Loop a fresh output name – otherwise the
            # output disappears when the constant is folded.
            if nm in {init.name for init in s.builder.initializers}:
                nm = s.get_unique_name(f"{nm}_loop")
                # We still want downstream nodes to use this new tensor,
                # so remember the mapping.
                s.var_to_name[v] = nm
                # and give shape info
                s.add_shape_info(nm, v.aval.shape, v.aval.dtype)
            state_out.append(nm)

        # 1) build the Loop‐body subgraph
        body_builder = OnnxBuilder(
            name_generator=s.builder.name_generator,
            opset=s.builder.opset,
            model_name=s.builder.get_unique_name("while_body"),
        )
        body_builder.enable_double_precision = getattr(
            s.builder, "enable_double_precision", False
        )
        body_builder.var_to_symbol_map = s.builder.var_to_symbol_map

        if need_int64_consts:
            _wrap_get_constant_name(body_builder)

        # a *fresh* converter for the subgraph
        body_conv = s.__class__(body_builder)

        # the two Loop‐reserved inputs: iteration count and incoming bool
        it_name = body_builder.name_generator.get("iter_count")
        prev_cond = body_builder.name_generator.get("cond_in")
        body_builder.add_scalar_input(it_name, TensorProto.INT64)
        body_builder.add_scalar_input(prev_cond, TensorProto.BOOL)

        # Map loop-carried state variables to inputs in the body graph,
        # promoting counters to INT64 where needed.
        for i, var in enumerate(b_jaxpr.invars):
            nm = body_conv.get_name(var)
            shp = var.aval.shape
            # If we promoted this slot, force INT64
            onnx_dt = np.int64 if i in promoted_idxs else var.aval.dtype
            body_builder.add_input(nm, shp, onnx_dt)
            body_conv.var_to_name[var] = nm

        # ➀ Lift out only **literal** constvars; keep tracers for later.
        captured_from_consts: list[tuple[core.Var, core.Tracer]] = []

        for cvar, cval in zip(b_jaxpr.constvars, b_consts):
            if isinstance(cval, core.Tracer):
                captured_from_consts.append((cvar, cval))
                # Make the captured tracer a real input of the body graph
                nm = body_conv.get_name(cvar)
                body_builder.add_input(nm, cval.aval.shape, cval.aval.dtype)
            else:
                const_nm = body_conv.get_constant_name(cval)
                if (
                    "need_int64_consts" in locals()
                    and need_int64_consts
                    and np.issubdtype(np.asarray(cval).dtype, np.integer)
                    and np.asarray(cval).shape == ()
                    and np.asarray(cval).dtype != np.int64
                ):
                    const_nm = _const_as_int64(body_builder, const_nm)
                body_conv.var_to_name[cvar] = const_nm

        for cvar, cval in zip(c_jaxpr.constvars, c_consts):
            if isinstance(cval, core.Tracer):
                captured_from_consts.append((cvar, cval))
                nm = body_conv.get_name(cvar)
                body_builder.add_input(nm, cval.aval.shape, cval.aval.dtype)
            else:
                const_nm = body_conv.get_constant_name(cval)
                if (
                    "need_int64_consts" in locals()
                    and need_int64_consts
                    and np.issubdtype(np.asarray(cval).dtype, np.integer)
                    and np.asarray(cval).shape == ()
                    and np.asarray(cval).dtype != np.int64
                ):
                    const_nm = _const_as_int64(body_builder, const_nm)
                body_conv.var_to_name[cvar] = const_nm

        # ➁ Now do all the body‐eqns (they'll refer to those constants by name).
        for eqn in b_jaxpr.eqns:
            body_conv._process_eqn(eqn)

        # ➂ Any extra invars beyond your loop‐state are the "captured tracers."
        extra_body_inputs: list[str] = []

        # a) tracers coming in as extra invars …
        num_state = len(node_inputs)
        for var in b_jaxpr.invars[num_state:]:
            nm = body_conv.get_name(var)
            # (they've already been added as inputs in the invars loop above,
            # but just in case…)
            if nm not in {i.name for i in body_builder.inputs}:
                body_builder.add_input(nm, var.aval.shape, var.aval.dtype)
            extra_body_inputs.append(nm)

        # b) tracers that appeared as "constvars" …
        for cvar, tracer in captured_from_consts:
            nm = body_conv.get_name(cvar)  # already an input of the body graph
            if nm not in extra_body_inputs:
                extra_body_inputs.append(nm)

        # after you have populated extra_body_inputs …
        for nm in extra_body_inputs:
            if nm not in s.builder.value_info_metadata:
                # prefer the shape info already collected in the body graph
                meta = body_builder.value_info_metadata.get(nm)
                if meta is None:
                    # last-ditch: recover from the jax Var
                    var = next(
                        jv for jv, onm in body_conv.var_to_name.items() if onm == nm
                    )
                    meta = (var.aval.shape, var.aval.dtype)
                s.add_shape_info(nm, *meta)

        # -----------------------------------------------------------
        # ➊  invariants / captured tracers must be passed through exactly once
        # -----------------------------------------------------------
        tracer_passthrough_map: dict[str, str] = {}
        for tracer_name in extra_body_inputs:
            # if the tracer was already in state_in, give it a fresh name
            if tracer_name in state_in:
                out_name = s.get_unique_name(f"{tracer_name}_loop")
            else:
                out_name = tracer_name
            tracer_passthrough_map[tracer_name] = out_name

            # declare the output in the subgraph
            if out_name not in {o.name for o in body_builder.outputs}:
                shape, dtype = s.builder.value_info_metadata[tracer_name]
                body_builder.add_output(out_name, shape, dtype)

            # expose to the outer graph
            state_out.append(out_name)
            s.add_shape_info(out_name, *s.builder.value_info_metadata[tracer_name])

        # Map inputs for the condition graph from the outputs of the body graph
        for inp, outp in zip(c_jaxpr.invars, b_jaxpr.outvars):
            body_conv.var_to_name[inp] = body_conv.get_name(outp)

        # process the cond eqns
        for eqn in c_jaxpr.eqns:
            body_conv._process_eqn(eqn)
        cond_out = body_conv.get_name(c_jaxpr.outvars[0])

        # Set body graph outputs: condition, then loop-carried state,
        # preserving each state's original JAX dtype.
        body_builder.outputs.clear()
        body_builder.add_output(cond_out, (), np.bool_)
        # Loop-carried outputs: promote counters to INT64
        for i, outp in enumerate(b_jaxpr.outvars):
            nm = body_conv.get_name(outp)
            shp = outp.aval.shape
            dt = np.int64 if i in promoted_idxs else outp.aval.dtype
            body_builder.add_output(nm, shp, dt)

        # ————————— Add captured‐tracer invariants here —————————
        for tracer_name in extra_body_inputs:
            if tracer_name not in {o.name for o in body_builder.outputs}:
                shape, dtype = body_builder.value_info_metadata.get(
                    tracer_name, s.builder.value_info_metadata[tracer_name]
                )
                body_builder.add_output(tracer_name, shape, dtype)

        # (This second invariants‐passthrough block is redundant — we already
        #  declared each captured tracer as an output in the tracer_passthrough_map
        #  loop — so removing it will keep our body subgraph and Loop node
        #  declarations in sync.)

        # Ensure every var we mapped in body_conv.var_to_name has a value_info.
        # This is a robust way to prevent "Missing value_info" errors for any
        # variable used within the subgraph, including from closures.
        existing_info = {inp.name for inp in body_builder.inputs} | {
            out.name for out in body_builder.outputs
        }
        for jax_var, onnx_name in body_conv.var_to_name.items():
            if onnx_name not in existing_info:
                # Registering the value_info ensures that intermediate tensors,
                # especially those from closures used in the cond/body jaxprs,
                # are known to the graph builder.
                body_builder.add_value_info(
                    onnx_name, jax_var.aval.shape, jax_var.aval.dtype
                )

        body_graph = body_builder.create_graph(
            body_builder.model_name, is_subgraph=True
        )

        # 2) build the initial condition check directly in the main graph
        temp_var_map = {}
        for cvar, cval in zip(c_jaxpr.constvars, c_consts):
            if isinstance(cval, core.Tracer):
                underlying_var = cval._trace.full_raise(cval)
                temp_var_map[cvar] = s.get_var_name(underlying_var)
            else:
                temp_var_map[cvar] = s.get_constant_name(cval)

        for inp, nm in zip(c_jaxpr.invars, state_in):
            temp_var_map[inp] = nm

        original_var_to_name = s.var_to_name
        s.var_to_name = s.var_to_name.copy()
        s.var_to_name.update(temp_var_map)

        for eqn in c_jaxpr.eqns:
            s._process_eqn(eqn)

        init_cond = s.get_name(c_jaxpr.outvars[0])

        s.var_to_name = original_var_to_name

        # 3) finally, emit the ONNX Loop node
        max_trip = s.get_constant_name(np.array(np.iinfo(np.int64).max, dtype=np.int64))

        # Include exactly the captured-tracer inputs we lifted into the body Jaxpr
        # (we already added them as subgraph inputs earlier, so just promote them
        #  to the Loop's input list here)
        for cvar, cval in zip(b_jaxpr.constvars, b_consts):
            if isinstance(cval, core.Tracer):
                tracer_name = body_conv.get_name(cvar)
                if tracer_name not in extra_body_inputs:
                    extra_body_inputs.append(tracer_name)

        # ──────────────────────────────────────────────────────────────
        # Ensure every extra_body_input is *already* visible in the
        # parent graph.  If it hasn't been declared yet, promote it to
        # a new graph input so the Loop node can legally consume it.
        # (This solves the "input 'var_X' hasn't been produced yet"
        # topological-sort error without dropping the invariant.)
        # ──────────────────────────────────────────────────────────────
        existing_top_level = {t.name for t in s.builder.inputs} | {
            t.name for t in s.builder.initializers
        }
        for nm in extra_body_inputs:
            if nm not in existing_top_level:
                shape, dtype = s.builder.value_info_metadata[nm]
                s.builder.add_input(nm, shape, dtype)
                existing_top_level.add(nm)

        loop_inputs = [max_trip, init_cond] + state_in + extra_body_inputs

        # ────────────────────────────────────────────────────────────────
        # Fix: Ensure no Loop output has the same name as an input
        #      – otherwise we create invalid cyclic graphs like var_3 → var_3
        #      This applies to loop-carried vars and passthrough tracers.
        # ────────────────────────────────────────────────────────────────
        input_set = set(loop_inputs)
        used_names = set(loop_inputs)
        new_state_out = []

        for idx, name in enumerate(state_out):
            original_name = name
            if name in input_set:
                # Rename output to avoid name collision
                new_name = s.get_unique_name(f"{name}_loopout")
                logger.info(
                    f"🛠️ Renaming Loop output '{name}' → '{new_name}' to avoid input collision"
                )
                name = new_name

                # If this output corresponds to a known JAX var, update the mapping
                if idx < len(node_outputs):
                    jax_var = node_outputs[idx]
                    s.var_to_name[jax_var] = name

                # Propagate shape/type info under the new name
                shape, dtype = s.builder.value_info_metadata[original_name]
                s.add_shape_info(name, shape, dtype)

            # Avoid duplicate output names
            while name in used_names:
                name = s.get_unique_name(f"{name}_dup")

            used_names.add(name)
            new_state_out.append(name)

        state_out = new_state_out

        loop_node = helper.make_node(
            "Loop",
            inputs=loop_inputs,
            outputs=state_out,
            body=body_graph,
            name=s.get_unique_name("while_loop"),
        )
        s.add_node(loop_node)
        # -----------------------------------------------------
        # ❸ Cast promoted INT64 outputs back to INT32 for the real JAX outputs
        # -----------------------------------------------------
        for idx, out_name in enumerate(state_out):
            if idx in promoted_idxs and idx < len(node_outputs):
                cast_back = s.get_unique_name(f"{out_name}_to_i32")
                s.add_node(
                    helper.make_node(
                        "Cast",
                        inputs=[out_name],
                        outputs=[cast_back],
                        name=s.get_unique_name("cast_to_i32"),
                        to=TensorProto.INT32,
                    )
                )
                s.add_shape_info(cast_back, (), np.int32)
                # point the JAX var → this new i32 name
                s.var_to_name[node_outputs[idx]] = cast_back
            else:
                # either unpromoted or an invariant tracer passthrough
                if idx < len(node_outputs):
                    shp = node_outputs[idx].aval.shape
                    dt = node_outputs[idx].aval.dtype
                else:
                    shp, dt = s.builder.value_info_metadata[out_name]
                s.add_shape_info(out_name, shp, dt)

        # And for the outer Loop node's state-outputs, record the **actual**
        # dtype that flows out of the Loop.  If we promoted a scalar counter
        # to INT64, the Loop produces INT64 (even though the JAX view of that
        # variable will later use the cast-back INT32 tensor).
        for idx, (nm, var) in enumerate(zip(state_out, node_outputs)):
            shp = var.aval.shape

            if idx in promoted_idxs:
                # The tensor exiting the Loop is INT64 …
                s.add_shape_info(nm, shp, np.int64)
            else:
                # … otherwise keep the original JAX dtype.
                s.add_shape_info(nm, shp, var.aval.dtype)

        # ──────────────────────────────────────────────────────────────
        # Validate: ensure no Loop output name is also one of its inputs
        # ──────────────────────────────────────────────────────────────
        input_set = set(loop_inputs)
        duplicate_names = [name for name in state_out if name in input_set]
        if duplicate_names:
            logger.warning(
                f"Loop has outputs with same names as inputs: {duplicate_names}. "
                "This can cause validation errors in some ONNX runtimes."
            )

        # ────────────────────────────────────────────────────────────────
        # Fix up scalar-integer comparisons in the *outer* condition:
        # make sure both inputs to Less/Greater .. are INT64 when a loop
        # counter has been promoted.
        # ────────────────────────────────────────────────────────────────
        if need_int64_consts:
            for node in s.builder.nodes[-len(c_jaxpr.eqns) :]:
                if node.op_type not in (
                    "Less",
                    "LessOrEqual",
                    "Greater",
                    "GreaterOrEqual",
                ):
                    continue

                in0, in1 = list(node.input)
                dt0 = s.builder.value_info_metadata[in0][1]
                dt1 = s.builder.value_info_metadata[in1][1]

                def _promote(idx, name, dtype, other_dtype):
                    if (
                        dtype in (np.int8, np.int16, np.int32)
                        and other_dtype == np.int64
                    ):
                        cast_nm = s.get_unique_name(f"{name}_to_i64")
                        s.add_node(
                            helper.make_node(
                                "Cast",
                                inputs=[name],
                                outputs=[cast_nm],
                                name=s.get_unique_name("cast_to_i64"),
                                to=TensorProto.INT64,
                            )
                        )
                        s.add_shape_info(cast_nm, (), np.int64)
                        node.input[idx] = cast_nm  # re-wire the Less node

                _promote(0, in0, dt0, dt1)
                _promote(1, in1, dt1, dt0)

    @staticmethod
    def get_monkey_patch(orig_fn):
        if WhileLoopPlugin._ORIG is None:
            WhileLoopPlugin._ORIG = orig_fn

        def patched(cond_fun, body_fun, init_val):
            # Special handling for closures: We need to trace with the closed-over
            # variables as arguments.
            cond_flat, cond_tree = jax.tree_util.tree_flatten(cond_fun)
            body_flat, body_tree = jax.tree_util.tree_flatten(body_fun)

            # This logic is simplified; a robust implementation would inspect
            # the functions' `.__closure__` attribute. For now, we assume that
            # if they are closures, the traced values are available in the scope
            # where `make_jaxpr` is called.

            closed_c = jax.make_jaxpr(cond_fun)(init_val)
            closed_b = jax.make_jaxpr(body_fun)(init_val)

            flat, tree = jax.tree_util.tree_flatten(init_val)
            # Pass jaxprs as parameters to the primitive
            results = lax.while_loop_p.bind(
                *flat, cond_jaxpr=closed_c, body_jaxpr=closed_b
            )
            return jax.tree_util.tree_unflatten(tree, results)

        return patched

    @staticmethod
    def _while_loop_impl(*args, **kwargs):
        # This is a placeholder for JAX's internal execution and not used
        # during ONNX conversion.
        if WhileLoopPlugin._ORIG is None:
            raise RuntimeError("Original lax.while_loop not recorded")

        cond_jaxpr = kwargs["cond_jaxpr"]
        body_jaxpr = kwargs["body_jaxpr"]

        # Separate the loop state from the other args (which are none)
        flat_state = args

        init_val_flat = list(flat_state)
        init_val_tree = jax.tree_util.tree_structure(init_val_flat)  # A bit of a guess
        jax.tree_util.tree_unflatten(init_val_tree, init_val_flat)

        def cond_f(v):
            fv, _ = jax.tree_util.tree_flatten(v)
            return core.eval_jaxpr(cond_jaxpr.jaxpr, cond_jaxpr.consts, *fv)[0]

        def body_f(v):
            fv, vt = jax.tree_util.tree_flatten(v)
            out = core.eval_jaxpr(body_jaxpr.jaxpr, body_jaxpr.consts, *fv)
            return jax.tree_util.tree_unflatten(vt, out)

        # The tree for init_val is not available, which makes this tricky.
        # This implementation path is for CPU execution of the patched primitive
        # and less critical than the `to_onnx` path.
        # For now, we assume a flat structure for simplicity.
        final = WhileLoopPlugin._ORIG(cond_f, body_f, init_val_flat)
        ff, _ = jax.tree_util.tree_flatten(final)
        return ff

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [lax],
            "target_attribute": "while_loop",
            "patch_function": WhileLoopPlugin.get_monkey_patch,
        }


lax.while_loop_p.def_abstract_eval(WhileLoopPlugin.abstract_eval)
lax.while_loop_p.def_impl(WhileLoopPlugin._while_loop_impl)
