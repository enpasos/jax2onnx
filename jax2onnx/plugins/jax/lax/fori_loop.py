# jax2onnx/plugins/jax/lax/fori_loop.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import config
from jax import core, lax
from jax.extend.core import Primitive
from onnx import helper, TensorProto
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


logger = logging.getLogger("jax2onnx.plugins.jax.lax.fori_loop")

# Pick 32- or 64-bit ints to match JAX's x64 mode flag:
_USE_INT64 = bool(config.read("jax_enable_x64"))


def _canon_int(x: int | np.integer) -> np.integer:
    return np.int64(x) if _USE_INT64 else np.int32(x)


# ─────────────────────────────── primitive stub ──────────────────────────────
fori_loop_p = Primitive("lax.fori_loop")
fori_loop_p.multiple_results = True


def model_fn(x):
    steps = 5

    def body_func(index, args):
        x, counter = args
        x += 0.1 * x**2
        counter += 1
        return (x, counter)

    args = (x, 0)
    args = jax.lax.fori_loop(0, steps, body_func, args)

    return args


# ────────────────────────── registration & testcases ─────────────────────────
@register_primitive(
    jaxpr_primitive=fori_loop_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html",
    onnx=[
        {"component": "Loop", "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html"}
    ],
    since="v0.5.1",
    context="primitives.lax",
    component="fori_loop",
    testcases=[
        {
            "testcase": "fori_loop_counter",
            "callable": lambda: lax.fori_loop(0, 5, lambda i, v: v + 1, 0),
            "input_shapes": [],
            "expected_output_shapes": [()],
        },
        {
            "testcase": "fori_loop_zero",
            "callable": lambda: lax.fori_loop(0, 0, lambda i, v: v + 1, 42),
            "input_shapes": [],
            "expected_output_shapes": [()],
        },
        {
            "testcase": "fori_loop_vector",
            "callable": lambda: lax.fori_loop(
                0,
                3,
                lambda i, v: v.at[i].set(i),
                jax.numpy.zeros((3,), dtype=jax.numpy.int32),
            ),
            "input_shapes": [],
            "expected_output_shapes": [(3,)],
        },
        {
            "testcase": "fori_loop_example",
            "callable": lambda: jax.lax.fori_loop(
                0,
                5,
                lambda i, args: (args[0] + 0.1 * args[0] ** 2, args[1] + 1),
                (jnp.array([1.0], dtype=jnp.float32), 0),
            )[0],
            "input_shapes": [],
            "expected_output_shapes": [(1,)],
        },
        {
            "testcase": "fori_loop_test",
            "callable": lambda x: model_fn(x),
            "input_shapes": [(2,)],
            "input_dtypes": [jnp.float32],
            "expected_output_shapes": [(2,), ()],  # Output shapes for x and counter
            "run_only_f32_variant": True,
        },
        {
            "testcase": "fori_loop_test_f64",
            "callable": lambda x: model_fn(x),
            "input_shapes": [(2,)],
            "input_dtypes": [jnp.float64],
            "expected_output_shapes": [(2,), ()],
            "run_only_f64_variant": True,
        },
    ],
)
class ForiLoopPlugin(PrimitiveLeafPlugin):
    """Lower `lax.fori_loop` (lower==0) with *k* loop‑carried tensors to ONNX."""

    _ORIG_FORI_LOOP: Callable | None = None

    # JAX abstract evaluation – simply forward the state avals
    @staticmethod
    def abstract_eval(*in_avals: core.AbstractValue, body_jaxpr, trip_count, **__):
        return tuple(in_avals)

    # ────────────────────────────── ONNX lowering ────────────────────────────
    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[core.Var],  # flat list of k tensors
        node_outputs: Sequence[core.Var],  # same length k
        params: dict[str, Any],
    ):
        body_closed = params["body_jaxpr"]
        trip_count = params["trip_count"]
        lower = params.get("lower", 0)
        if lower != 0:
            raise NotImplementedError("fori_loop with lower!=0 not supported yet")

        # --- outer‑graph bookkeeping -------------------------------------------------
        in_names = [s.get_name(v) for v in node_inputs]
        out_names = [s.get_name(v) for v in node_outputs]

        # ---------------------------------------------------------------------------
        # Build the Loop‑body sub‑graph
        # ---------------------------------------------------------------------------
        prefix = s.builder.name_generator.get("loop")  # unique per Loop instance

        body_builder = OnnxBuilder(
            name_generator=s.builder.name_generator,  # keep global generator
            opset=s.builder.opset,
            model_name=s.builder.get_unique_name(f"{prefix}_body"),
        )

        # --- CORRECTED FIX START ---
        # Propagate the double precision flag from the main builder to the subgraph builder.
        body_builder.enable_double_precision = getattr(
            s.builder, "enable_double_precision", False
        )
        # --- CORRECTED FIX END ---

        body_builder.var_to_symbol_map = s.builder.var_to_symbol_map
        body_conv = s.__class__(body_builder)

        # ► inputs:   (iter, cond_in, s1_in … sk_in)
        iter64 = body_builder.name_generator.get(f"{prefix}_iter64")
        cond_in = body_builder.name_generator.get(f"{prefix}_cond_in")
        body_builder.add_scalar_input(iter64, TensorProto.INT64)
        body_builder.add_scalar_input(cond_in, TensorProto.BOOL)

        # add the k state inputs
        for idx, v in enumerate(node_inputs):
            sym = body_builder.name_generator.get(f"{prefix}_state{idx}_in")
            body_builder.add_input(sym, v.aval.shape, v.aval.dtype)
            # Map to Jaxpr input (skip the first invar which is the loop‑index)
            body_conv.var_to_name[body_closed.jaxpr.invars[idx + 1]] = sym

        # iterator cast if body expects int32
        iter_target_dtype = (
            TensorProto.INT32
            if body_closed.jaxpr.invars[0].aval.dtype == np.int32
            else TensorProto.INT64
        )
        iter_sym = iter64
        if iter_target_dtype == TensorProto.INT32:
            iter32 = body_builder.name_generator.get(f"{prefix}_iter32")
            body_builder.add_node(
                helper.make_node(
                    "Cast",
                    [iter64],
                    [iter32],
                    to=TensorProto.INT32,
                    name=body_builder.name_generator.get(f"{prefix}_cast_iter"),
                )
            )
            iter_sym = iter32

        # Map iterator symbol and constants
        body_conv.var_to_name[body_closed.jaxpr.invars[0]] = iter_sym
        for cv, cval in zip(body_closed.jaxpr.constvars, body_closed.consts):
            body_conv.var_to_name[cv] = body_conv.get_constant_name(cval)

        # ► convert the body jaxpr
        body_conv._process_jaxpr(body_closed.jaxpr, body_closed.consts)

        # ► outputs: (cond_out, s1_out … sk_out)
        body_builder.outputs.clear()

        cond_out = body_builder.name_generator.get(f"{prefix}_cond_out")
        body_builder.add_node(
            helper.make_node(
                "Identity",
                [cond_in],
                [cond_out],
                name=body_builder.name_generator.get(f"{prefix}_cond_passthrough"),
            )
        )
        body_builder.add_output(cond_out, (), np.bool_)

        for idx, v in enumerate(body_closed.jaxpr.outvars):
            sym_out = body_conv.get_name(v)
            aval = v.aval
            body_builder.add_output(sym_out, aval.shape, aval.dtype)

        body_graph = body_builder.create_graph(
            body_builder.model_name, is_subgraph=True
        )

        # stub: register the fori_loop body subgraph (no real graph yet)
        s.subgraph(
            name="fori_body",
            invars=list(
                body_conv.var_to_name.values()
            ),  # Convert dict_values to a list
            jaxpr=body_closed.jaxpr,
        )

        # ---------------------------------------------------------------------------
        # Emit the outer Loop node
        # ---------------------------------------------------------------------------
        loop_node = helper.make_node(
            "Loop",
            inputs=[
                s.get_constant_name(np.asarray(trip_count, np.int64)),
                s.get_constant_name(np.asarray(True, np.bool_)),
                *in_names,
            ],
            outputs=out_names,
            body=body_graph,
            name=s.get_unique_name("fori_loop"),
        )
        s.add_node(loop_node)
        for sym, v in zip(out_names, node_outputs):
            s.add_shape_info(sym, v.aval.shape, v.aval.dtype)

    # ─────────────────── monkey‑patch (bind primitive) ───────────────────────
    @staticmethod
    def _fori_loop_binding(lower, upper, body_fun, init_val):
        """
        Wrap `body_fun` so the jaxpr sees **one input per leaf** of the
        PyTree `init_val`, yet `body_fun` itself continues to work with
        the original structure.
        """
        if lower != 0:
            raise NotImplementedError("fori_loop plugin supports lower==0 only")

        # ── 1) Flatten PyTree and up-cast integer scalars to int64 ──────────
        leaves, treedef = jax.tree_util.tree_flatten(init_val)
        leaves = [
            _canon_int(leaf) if isinstance(leaf, (int, np.integer)) else leaf
            for leaf in leaves
        ]

        # ── body wrapper:   (i, *leaves)  →  *new_leaves
        def body_flat(i, *flat_state):
            state = jax.tree_util.tree_unflatten(treedef, flat_state)
            new_state = body_fun(i, state)
            return jax.tree_util.tree_flatten(new_state)[0]

        body_closed = jax.make_jaxpr(body_flat)(0, *leaves)
        trip_count = int(upper - lower)

        flat_res = fori_loop_p.bind(
            *leaves,
            body_jaxpr=body_closed,
            trip_count=trip_count,
            lower=0,
        )
        return jax.tree_util.tree_unflatten(treedef, flat_res)

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


# register abstract eval
fori_loop_p.def_abstract_eval(ForiLoopPlugin.abstract_eval)
