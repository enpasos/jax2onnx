# jax2onnx/plugins/jax/lax/while_loop.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence, Callable

import jax
import numpy as np
from jax import core, lax
from jax.extend.core import Primitive
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
        {
            "testcase": "while_loop_f64",
            "callable": lambda x: lax.while_loop(
                lambda val: val < 5.0, lambda val: val * 1.1, x
            ),
            "input_shapes": [()],
            "input_dtypes": [np.float64],
            "expected_output_shapes": [()],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "while_loop_multi_state_f32",
            "callable": _while_loop_multi_state_fn,
            "input_shapes": [(2,)],
            "input_dtypes": [np.float32],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "while_loop_multi_state_f64",
            "callable": _while_loop_multi_state_fn,
            "input_shapes": [(2,)],
            "input_dtypes": [np.float64],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
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

        # the input‐names and output‐names for the loop state
        state_in = [s.get_name(v) for v in node_inputs]
        state_out = [s.get_name(v) for v in node_outputs]

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

        # a *fresh* converter for the subgraph
        body_conv = s.__class__(body_builder)

        # the two Loop‐reserved inputs: iteration count and incoming bool
        it_name = body_builder.name_generator.get("iter_count")
        prev_cond = body_builder.name_generator.get("cond_in")
        body_builder.add_scalar_input(it_name, TensorProto.INT64)
        body_builder.add_scalar_input(prev_cond, TensorProto.BOOL)

        # now only map the ***body*** JAXPR’s own invars after its constvars
        n_const = len(b_jaxpr.constvars)
        for idx, var in enumerate(b_jaxpr.invars[n_const:]):
            nm = state_in[idx]
            body_builder.add_input(nm, var.aval.shape, var.aval.dtype)
            body_conv.var_to_name[var] = nm

        # bring in the body constants
        for cvar, cval in zip(b_jaxpr.constvars, b_consts):
            body_conv.var_to_name[cvar] = body_conv.get_constant_name(cval)

        # process all body eqns
        for eqn in b_jaxpr.eqns:
            body_conv._process_eqn(eqn)

        # now wire the cond JAXPR *inside* the body:
        for cvar, cval in zip(c_jaxpr.constvars, c_consts):
            body_conv.var_to_name[cvar] = body_conv.get_constant_name(cval)
        # cond_jaxpr.invars after its constvars line up with body_jaxpr.outvars
        for inp, outp in zip(c_jaxpr.invars[len(c_jaxpr.constvars) :], b_jaxpr.outvars):
            body_conv.var_to_name[inp] = body_conv.get_name(outp)

        # process the cond eqns
        for eqn in c_jaxpr.eqns:
            body_conv._process_eqn(eqn)
        cond_out = body_conv.get_name(c_jaxpr.outvars[0])

        # clear whatever OnnxBuilder auto‐added as outputs, then re‐add:
        body_builder.outputs.clear()
        body_builder.add_output(cond_out, (), np.bool_)
        for outp in b_jaxpr.outvars:
            nm = body_conv.get_name(outp)
            body_builder.add_output(nm, outp.aval.shape, outp.aval.dtype)

        body_graph = body_builder.create_graph(
            body_builder.model_name, is_subgraph=True
        )

        # 2) build the “initial‐condition” subgraph for cond(init_state)
        init_builder = OnnxBuilder(
            name_generator=s.builder.name_generator,
            opset=s.builder.opset,
            model_name=s.builder.get_unique_name("while_init"),
        )
        init_builder.enable_double_precision = getattr(
            s.builder, "enable_double_precision", False
        )
        init_conv = s.__class__(init_builder)

        # wire cond consts
        for cvar, cval in zip(c_jaxpr.constvars, c_consts):
            init_conv.var_to_name[cvar] = init_conv.get_constant_name(cval)
        # then wire the cond invars to the *same* outer state names
        for inp, nm in zip(c_jaxpr.invars[len(c_jaxpr.constvars) :], state_in):
            init_conv.var_to_name[inp] = nm

        # process the cond JAXPR once
        init_conv._process_jaxpr(c_jaxpr, c_consts)
        init_cond = init_conv.get_name(c_jaxpr.outvars[0])

        # merge init nodes & inits into the main graph
        s.builder.nodes.extend(init_builder.nodes)
        existing = {t.name for t in s.builder.initializers}
        for t in init_builder.initializers:
            if t.name not in existing:
                s.builder.initializers.append(t)
        s.builder.value_info_metadata.update(init_builder.value_info_metadata)
        vi_map = {vi.name: vi for vi in s.builder.value_info}
        for vi in init_builder.value_info:
            vi_map[vi.name] = vi
        s.builder.value_info = list(vi_map.values())
        s.builder.functions.update(init_builder.functions)

        # 3) finally, emit the ONNX Loop node
        max_trip = s.get_constant_name(np.array(np.iinfo(np.int64).max, dtype=np.int64))
        loop_node = helper.make_node(
            "Loop",
            inputs=[max_trip, init_cond, *state_in],
            outputs=state_out,
            body=body_graph,
            name=s.get_unique_name("while_loop"),
        )
        s.add_node(loop_node)

        # register the output shapes/dtypes
        for nm, var in zip(state_out, node_outputs):
            s.add_shape_info(nm, var.aval.shape, var.aval.dtype)

    @staticmethod
    def get_monkey_patch(orig_fn):
        # stash the original Python while_loop
        if WhileLoopPlugin._ORIG is None:
            WhileLoopPlugin._ORIG = orig_fn

        def patched(cond_fun, body_fun, init_val):
            # create closed JAXPRs for cond & body
            closed_c = jax.make_jaxpr(cond_fun)(init_val)
            closed_b = jax.make_jaxpr(body_fun)(init_val)
            flat, tree = jax.tree_util.tree_flatten(init_val)
            # bind the primitive
            results = lax.while_loop_p.bind(
                *flat, cond_jaxpr=closed_c, body_jaxpr=closed_b
            )
            return jax.tree_util.tree_unflatten(tree, results)

        return patched

    @staticmethod
    def _while_loop_impl(*flat_state, tree, cond_jaxpr, body_jaxpr):
        # interpret the while‐loop via the original
        if WhileLoopPlugin._ORIG is None:
            raise RuntimeError("Original lax.while_loop not recorded")
        init_val = jax.tree_util.tree_unflatten(tree, flat_state)

        def cond_f(v):
            fv, _ = jax.tree_util.tree_flatten(v)
            return core.eval_jaxpr(cond_jaxpr.jaxpr, cond_jaxpr.consts, *fv)[0]

        def body_f(v):
            fv, vt = jax.tree_util.tree_flatten(v)
            out = core.eval_jaxpr(body_jaxpr.jaxpr, body_jaxpr.consts, *fv)
            return jax.tree_util.tree_unflatten(vt, out)

        final = WhileLoopPlugin._ORIG(cond_f, body_f, init_val)
        ff, _ = jax.tree_util.tree_flatten(final)
        return ff

    @staticmethod
    def patch_info():
        # patch the Python API for jax.lax.while_loop → our `patched`
        return {
            "patch_targets": [lax],
            "target_attribute": "while_loop",
            "patch_function": WhileLoopPlugin.get_monkey_patch,
        }


# hook the primitive up to JAX and to our impl
lax.while_loop_p.def_abstract_eval(WhileLoopPlugin.abstract_eval)
lax.while_loop_p.def_impl(WhileLoopPlugin._while_loop_impl)
