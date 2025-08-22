# jax2onnx/plugins/jax/lax/fori_loop.py
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import config
from jax import core, lax
from jax.extend.core import Primitive
from onnx import helper, TensorProto
from onnx import onnx_ml_pb2 as om
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE as _NP2ONNX
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE as _ONNX2NP
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


logger = logging.getLogger("jax2onnx.plugins.jax.lax.fori_loop")

# Pick 32- or 64-bit ints to match JAX's x64 mode flag:
_USE_INT64 = bool(config.read("jax_enable_x64"))


def _canon_int(x: int | np.integer) -> np.integer:
    return np.int64(x) if _USE_INT64 else np.int32(x)


def _loosen_graph_value_infos_to_rank_only(g) -> None:
    """
    For a Loop body:
      • Drop VIs for outputs of shape/dtype-sensitive ops (let ORT infer).
      • Drop VIs that correspond to initializers (constants) or Constant/ConstantOfShape.
      • For remaining VIs, keep elem_type but clear dim_value *and* dim_param (rank-only).
    """
    produced_by_type = {o: n.op_type for n in g.node for o in n.output}
    produced_by_name = {o: n.name for n in g.node for o in n.output}

    sensitive = {
        # dtype/shape changers
        "Cast",
        "Reshape",
        "Squeeze",
        "Unsqueeze",
        "Expand",
        "Concat",
        # shape/index-ish
        "Range",
        "Shape",
        "NonZero",
        "Gather",
        "GatherND",
        "Slice",
        # constants
        "Constant",
        "ConstantOfShape",
        # dtype troublemaker
        "Pow",
        # arithmetic can re-tighten dims (e.g., add_* / mul_* intermediates)
        "Add",
        "Sub",
        "Mul",
        "Div",
    }

    # Heuristics by node name (useful if op_type is missing/opaque)
    def looks_sensitive_by_name(nm: str, vi: om.ValueInfoProto) -> bool:
        nm = (nm or "").lower()
        # reshape-ish helpers
        if any(tok in nm for tok in ("reshape", "squeeze", "unsqueeze", "expand")):
            return True
        # index arithmetic naming pattern; only relevant for int tensors
        if vi.type.HasField("tensor_type"):
            et = vi.type.tensor_type.elem_type
            if et in (TensorProto.INT64, TensorProto.INT32):
                if any(tok in nm for tok in ("add_start", "start_col", "start_row")):
                    return True
        return False

    initializer_names = {init.name for init in g.initializer}

    keep: list[om.ValueInfoProto] = []
    for vi in list(g.value_info):
        name = vi.name

        # Drop constants immediately
        if name in initializer_names:
            continue

        op_type = produced_by_type.get(name)
        node_nm = produced_by_name.get(name, "")

        # Drop VIs for sensitive producers (or when the name gives it away)
        if (op_type in sensitive) or looks_sensitive_by_name(node_nm, vi):
            continue

        # Only handle tensor types
        if not vi.type.HasField("tensor_type"):
            continue

        # Rebuild a fresh, rank-only VI (preserve dtype, clear dims)
        ttype = vi.type.tensor_type
        elem = ttype.elem_type
        rank = len(ttype.shape.dim) if ttype.HasField("shape") else 0

        new_vi = om.ValueInfoProto()
        new_vi.name = name
        new_vi.type.tensor_type.elem_type = elem
        if rank:
            shp = new_vi.type.tensor_type.shape
            for _ in range(rank):
                shp.dim.add()  # leave dim_value/param unset -> dynamic
        keep.append(new_vi)

    del g.value_info[:]
    g.value_info.extend(keep)


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

        # Propagate outer flags/config into the subgraph builder/converter
        # (notably double-precision and symbolic-dim origins).
        body_builder.enable_double_precision = getattr(
            s.builder, "enable_double_precision", False
        )
        # Create the body converter and inherit the parent's symbolic origins.
        body_conv = s.__class__(body_builder)

        body_builder.var_to_symbol_map = s.builder.var_to_symbol_map
        # Inherit known origins so Shape() inside the body can resolve outer symbols.
        if not hasattr(body_conv, "symbolic_dim_to_origin"):
            body_conv.symbolic_dim_to_origin = {}
        if hasattr(s, "symbolic_dim_to_origin") and getattr(
            s, "symbolic_dim_to_origin"
        ):
            # copy so we can safely mutate in the body
            body_conv.symbolic_dim_to_origin.update(dict(s.symbolic_dim_to_origin))

        # ► inputs:   (iter, cond_in, s1_in … sk_in)
        iter64 = body_builder.name_generator.get(f"{prefix}_iter64")
        cond_in = body_builder.name_generator.get(f"{prefix}_cond_in")
        body_builder.add_scalar_input(iter64, TensorProto.INT64)
        body_builder.add_scalar_input(cond_in, TensorProto.BOOL)

        # add the k state inputs as rank-only to prevent ORT from over-constraining
        for idx, v in enumerate(node_inputs):
            sym = body_builder.name_generator.get(f"{prefix}_state{idx}_in")
            rank = len(getattr(v.aval, "shape", ()))
            dyn_shape = (None,) * rank
            # IMPORTANT: carried-state dtype must follow the jaxpr exactly.
            _dt = np.dtype(v.aval.dtype).type
            body_builder.add_input(sym, dyn_shape, _dt)
            # Map to Jaxpr input (skip the first invar which is the loop-index)
            body_conv.var_to_name[body_closed.jaxpr.invars[idx + 1]] = sym

            # Register symbolic-dimension origins for this carried state.
            # If shape has symbolic dims (e.g., 'B'), record that they come
            # from this tensor at the corresponding axis. This allows any
            # Shape()/broadcast ops in the body to resolve symbols like 'B'.
            aval_shape = getattr(v.aval, "shape", ())
            for axis, dim in enumerate(aval_shape):
                if not isinstance(dim, int):
                    # Drop any inherited entries that refer to the same symbol
                    # (match by str equality) so we don't keep outer names like 'var_0'.
                    keys_to_drop = [
                        k
                        for k in list(body_conv.symbolic_dim_to_origin.keys())
                        if str(k) == str(dim)
                    ]
                    for k in keys_to_drop:
                        del body_conv.symbolic_dim_to_origin[k]
                    # Register both the object key (DimExpr) and its string form.
                    body_conv.symbolic_dim_to_origin[dim] = (sym, axis)
                    body_conv.symbolic_dim_to_origin[str(dim)] = (sym, axis)

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

        # --- dtype harmonization for numeric ops in Loop body -----------------
        # Align ANY numeric mismatch (ints or floats) on binary ops (and Pow)
        # by casting the RHS to the LHS dtype. Can be disabled via env var.
        _disable_cast_env = os.getenv("JAX2ONNX_DISABLE_LOOP_BINOP_CAST", "").lower()
        _disable_cast = _disable_cast_env in ("1", "true", "yes", "on")
        if not _disable_cast:
            # ops whose *output* dtype equals the inputs' dtype
            _same_type_ops = {"Add", "Sub", "Mul", "Div", "Min", "Max", "Pow"}
            _compare_ops = {"Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Equal"}
            _need_same_dtype = _same_type_ops | _compare_ops
            _PASS_THROUGH = {
                "Identity",
                "Reshape",
                "Squeeze",
                "Unsqueeze",
                "Expand",
                "Transpose",
                "Flatten",
            }
            _CONSTISH = {"Constant", "ConstantOfShape"}

            def _robust_dtype(sym: str, produced_by: dict[str, Any]):
                """
                Resolve dtype of 'sym'. If a dtype is recorded but the producer
                is a pass-through shape op, DO NOT trust it — walk back to the
                true source (e.g., constants) and derive from there.
                """
                seen = set()
                cur = sym
                while cur not in seen:
                    seen.add(cur)
                    n = produced_by.get(cur)
                    dt = body_builder.get_dtype(cur)
                    if dt is not None:
                        # If the producer is pass-through OR a same-type numeric op,
                        # the stored dtype may be stale. Prefer to walk to the source.
                        if n is not None and (
                            n.op_type in _PASS_THROUGH or n.op_type in _same_type_ops
                        ):
                            if n.input:
                                cur = n.input[0]
                                continue
                        return np.dtype(dt)
                    if n is None:
                        break
                    if n.op_type == "Cast":
                        to_attr = next((a for a in n.attribute if a.name == "to"), None)
                        if to_attr is not None:
                            onnx_t = to_attr.i
                            np_t = _ONNX2NP.get(onnx_t)
                            if np_t is not None:
                                return np.dtype(np_t)
                        cur = n.input[0]
                        continue
                    if n.op_type in _PASS_THROUGH and n.input:
                        cur = n.input[0]
                        continue
                    break
                return None

            def _is_constantish(sym: str, produced_by: dict[str, Any]) -> bool:
                seen = set()
                cur = sym
                while cur not in seen:
                    seen.add(cur)
                    n = produced_by.get(cur)
                    if n is None:
                        return False
                    if n.op_type in _CONSTISH:
                        return True
                    # traverse through benign wrappers
                    if n.op_type in _PASS_THROUGH or n.op_type == "Cast":
                        if n.input:
                            cur = n.input[0]
                            continue
                    return False
                return False

            # Iterate to a fixed point because dtype changes upstream can
            # create new mismatches downstream (e.g., Add after a Mul upcast).
            MAX_PASSES = 4
            for _ in range(MAX_PASSES):
                changed = False
                produced_by = {o: n for n in body_builder.nodes for o in n.output}
                for n in list(body_builder.nodes):  # snapshot; we may insert new nodes
                    if n.op_type not in _need_same_dtype or len(n.input) < 2:
                        continue
                    a, b = n.input[0], n.input[1]
                    ta = _robust_dtype(a, produced_by)
                    tb = _robust_dtype(b, produced_by)
                    if (
                        ta is not None
                        and tb is not None
                        and np.issubdtype(ta, np.number)
                        and np.issubdtype(tb, np.number)
                        and ta != tb
                    ):
                        # Prefer casting the constantish side (if exactly one is).
                        if _is_constantish(a, produced_by) and not _is_constantish(
                            b, produced_by
                        ):
                            a, b, ta, tb = b, a, tb, ta
                            # also swap in the node for clarity
                            n.input[0], n.input[1] = a, b
                        cast_out = body_builder.name_generator.get(
                            f"{b}_cast_{ta.name}"
                        )
                        to_enum = _NP2ONNX[ta]
                        cast_node = helper.make_node(
                            "Cast",
                            [b],
                            [cast_out],
                            name=body_builder.name_generator.get("CastAlignLoopBody"),
                            to=to_enum,
                        )
                        idx = body_builder.nodes.index(n)
                        body_builder.nodes.insert(idx, cast_node)
                        # keep a sensible rank hint (prefer known ranks, else 0)
                        r = body_builder.get_rank(b)
                        if r is None:
                            r = body_builder.get_rank(a) or 0
                        body_builder.add_value_info(cast_out, (None,) * r, ta.type)
                        n.input[1] = cast_out

                        # For same-type ops (incl. Pow), set outputs' dtype to ta as well.
                        if n.op_type in _same_type_ops:
                            for outsym in n.output:
                                r_o = body_builder.get_rank(outsym)
                                if r_o is None:
                                    r_o = body_builder.get_rank(a) or 0
                                body_builder.add_value_info(
                                    outsym, (None,) * r_o, ta.type
                                )
                        changed = True
                if not changed:
                    break
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
            # Outputs must mirror the carried-state dtype from the jaxpr.
            _dt = np.dtype(aval.dtype).type
            body_builder.add_output(sym_out, aval.shape, _dt)

        body_graph = body_builder.create_graph(
            body_builder.model_name, is_subgraph=True
        )
        # Always sanitize Loop-body VIs to avoid ORT type/shape re-tightening issues
        _loosen_graph_value_infos_to_rank_only(body_graph)

        # stub: register the fori_loop body subgraph (no real graph yet)
        s.subgraph(
            name="fori_body",
            invars=list(
                body_conv.var_to_name.values()
            ),  # Convert dict_values to a list
            jaxpr=body_closed.jaxpr,
        )

        # Defensive re-apply (harmless if nothing changed)
        _loosen_graph_value_infos_to_rank_only(body_graph)

        # ---------------------------------------------------------------------------
        # Emit the outer Loop node
        # ---------------------------------------------------------------------------
        # (1) Optionally align carried-state inputs (only if dtype truly differs).
        aligned_state_inputs: list[str] = []
        for sym, v in zip(in_names, node_inputs):
            want_dt = np.dtype(v.aval.dtype).type
            current_dt = s.builder.get_dtype(sym)
            if (current_dt is not None) and (np.dtype(current_dt) != np.dtype(want_dt)):
                cast_sym = s.get_unique_name(f"{sym}_to_{np.dtype(want_dt).name}")
                s.add_node(
                    helper.make_node(
                        "Cast",
                        [sym],
                        [cast_sym],
                        to=_NP2ONNX[np.dtype(want_dt)],
                        name=s.get_unique_name("CastLoopIn"),
                    )
                )
                # keep a shape hint for the casted tensor
                s.add_shape_info(cast_sym, getattr(v.aval, "shape", ()), want_dt)
                aligned_state_inputs.append(cast_sym)
            else:
                aligned_state_inputs.append(sym)

        # (2) Emit the Loop with outputs wired straight to the final names.
        #     By ONNX spec, each carried-state output type must match the corresponding
        #     carried-state *input* type. We already aligned the inputs above, so we just
        #     wire outputs to `out_names` and record their metadata.
        loop_node = helper.make_node(
            "Loop",
            inputs=[
                s.get_constant_name(np.asarray(trip_count, np.int64)),
                s.get_constant_name(np.asarray(True, np.bool_)),
                *aligned_state_inputs,
            ],
            outputs=out_names,
            body=body_graph,
            name=s.get_unique_name("fori_loop"),
        )
        s.add_node(loop_node)

        # (3) Record final output metadata using the jaxpr dtype.
        for final, v in zip(out_names, node_outputs):
            desired_dt = np.dtype(v.aval.dtype).type
            s.add_shape_info(final, v.aval.shape, desired_dt)

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
