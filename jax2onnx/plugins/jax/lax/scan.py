# file: jax2onnx/plugins/jax/lax/scan.py
from __future__ import annotations

import logging
import os
import numpy as _np
from typing import Any, Sequence, Union, Optional

import jax
import jax.numpy as jnp
from jax import core, lax
from onnx import TensorProto, helper
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE as _NP2ONNX

from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax.extend.core import ClosedJaxpr, Var

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scan")


INT64 = TensorProto.INT64
i64 = _np.int64
# unique id for every Scan node – used when generating helper names
_SCAN_INSTANCE_COUNTER: int = 0


# --- Utility: make all internal VIs rank-only (shape dims unknown) -----------
def _loosen_value_infos_to_rank_only(bld: OnnxBuilder) -> None:
    """Make *internal* value_infos rank-only (all dims dynamic).
    Inputs/outputs are handled explicitly elsewhere. Always on."""
    # Rebuild value_info with same dtype but fully dynamic dims.
    vis = list(bld.value_info)
    bld.value_info[:] = [vi for vi in bld.value_info if False]  # clear
    for vi in vis:
        name = vi.name
        dt = bld.get_dtype(name)
        # Prefer known rank; fall back to rank from the existing VI if needed
        rank = bld.get_rank(name)
        if rank is None:
            # try to peek the proto
            tt = vi.type.tensor_type
            rank = len(tt.shape.dim) if tt.HasField("shape") else 0
        bld.add_value_info(name, (None,) * (rank or 0), dt)


# --- Utility: retag body value_info dtypes to match producer inputs ----------
def _retag_value_infos_to_input_dtype(bld: OnnxBuilder) -> None:
    """
    Ensure outputs of common shape/index/passthrough/binop nodes inherit the dtype
    of their first input. IMPORTANT: iterate over *nodes*, not just existing VIs,
    so we also cover symbols that don't yet have a value_info entry.
    """
    passthrough = {
        "Squeeze",
        "Unsqueeze",
        "Identity",
        "Reshape",
        "Transpose",
        "Flatten",
        "Expand",
    }
    index_like = {"Gather", "GatherND", "GatherElements", "Slice"}
    binops = {"Add", "Sub", "Mul", "Div"}
    same_as_first = passthrough | index_like | {"Concat"} | binops

    for n in list(bld.nodes):
        if n.op_type not in same_as_first or not n.input:
            continue
        exp_dt = bld.get_dtype(n.input[0])
        if exp_dt is None:
            continue
        for o in n.output:
            cur_dt = bld.get_dtype(o)
            if cur_dt is not None and _np.dtype(cur_dt) == _np.dtype(exp_dt):
                continue
            # pick a sensible rank for `o`
            r = bld.get_rank(o)
            if r is None:
                r = bld.get_rank(n.input[0]) or 0
            # replace any stale VI and set dtype to the input's dtype
            bld.value_info[:] = [vi for vi in bld.value_info if vi.name != o]
            bld.add_value_info(o, (None,) * r, exp_dt)


# --- Utility: make all internal VIs rank-only (shape dims unknown) -----------
def _loosen_graph_value_infos_to_rank_only(g) -> None:
    """Always-on sanitization for Loop/Scan bodies:
    - drop value_info for any *node output* (let ORT infer dtypes)
    - drop value_info that mirrors any graph input/output
    - keep the rest but clear all concrete dims → rank-only
    """
    produced = {o for n in g.node for o in n.output}
    io_names = {i.name for i in g.input} | {o.name for o in g.output}
    keep = []
    for vi in list(g.value_info):
        name = vi.name
        # Remove VIs for node outputs and any that mirror graph IO
        if name in produced or name in io_names:
            continue
        # Keep dtype, but clear all fixed dim info → rank-only
        tt = vi.type.tensor_type
        if tt.HasField("shape"):
            for d in tt.shape.dim:
                if d.HasField("dim_value"):
                    d.ClearField("dim_value")
                if d.HasField("dim_param"):
                    d.ClearField("dim_param")
        keep.append(vi)
    del g.value_info[:]
    g.value_info.extend(keep)


# ---------------------------------------------------------------------------


# ----------------------------------------------------------------------
# helpers used by test-cases
# ----------------------------------------------------------------------
def scan_fn(x):
    def body(carry, _):
        carry = carry + 1
        return carry, carry

    _, ys = lax.scan(body, x, None, length=5)
    return ys


def _scan_jit_no_xs() -> jax.Array:
    """Mimics the ‘simulate → jax.jit(main)’ pattern."""

    def simulate():
        def step_fn(carry, _):
            return carry + 1, carry * 2

        _, ys = lax.scan(step_fn, 0, xs=None, length=10)
        return ys

    return jax.jit(simulate)()


# ----------------------------------------------------------------------
# regression helpers – two scans with different trip-counts
# ----------------------------------------------------------------------
def _two_scans_diff_len_f32():
    xs_small = jnp.asarray(_np.arange(5, dtype=_np.float32))
    xs_big = jnp.asarray(_np.arange(100, dtype=_np.float32))
    fill_small = jnp.asarray(_np.full(xs_small.shape, 0.1, dtype=_np.float32))
    fill_big = jnp.asarray(_np.full(xs_big.shape, 0.1, dtype=_np.float32))

    _, y1 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_small, fill_small)
    )
    _, y2 = lax.scan(lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_big, fill_big))
    return y1, y2


# ----------------------------------------------------------------------
# regression -- nested scan: inner length 5, outer length 100
# ----------------------------------------------------------------------
def _nested_scan_len_mismatch_f32():
    xs_outer = jnp.asarray(_np.arange(100, dtype=_np.float32))
    xs_inner = jnp.asarray(_np.arange(5, dtype=_np.float32))
    fill_inn = jnp.broadcast_to(0.1, xs_inner.shape)

    def inner(c, xs):
        c = c + xs[0] + xs[1]
        return c, c

    def outer(c, x):
        _, ys = lax.scan(inner, c, xs=(xs_inner, fill_inn))
        return c + x, ys[-1]

    _, ys_out = lax.scan(outer, 0.0, xs_outer)
    return ys_out


def _nested_scan_len_mismatch_f64():
    xs_outer = jnp.asarray(_np.arange(100, dtype=_np.float64))
    xs_inner = jnp.asarray(_np.arange(5, dtype=_np.float64))
    fill_inn = jnp.broadcast_to(0.1, xs_inner.shape)

    def inner(c, xs):
        c = c + xs[0] + xs[1]
        return c, c

    def outer(c, x):
        _, ys = lax.scan(inner, c, xs=(xs_inner, fill_inn))
        return c + x, ys[-1]

    _, ys_out = lax.scan(outer, 0.0, xs_outer)
    return ys_out


def _two_scans_diff_len_f64():
    xs_small = jnp.asarray(_np.arange(5, dtype=_np.float64))
    xs_big = jnp.asarray(_np.arange(100, dtype=_np.float64))
    fill_small = jnp.asarray(_np.full(xs_small.shape, 0.1, dtype=_np.float64))
    fill_big = jnp.asarray(_np.full(xs_big.shape, 0.1, dtype=_np.float64))

    _, y1 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_small, fill_small)
    )
    _, y2 = lax.scan(lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_big, fill_big))
    return y1, y2


def _two_scans_len_mismatch_broadcast_f32():
    xs_small = jnp.asarray(_np.arange(5, dtype=_np.float32))
    xs_big = jnp.asarray(_np.arange(100, dtype=_np.float32))

    fill_small = jnp.asarray(_np.full(5, 0.1, dtype=_np.float32))
    fill_big = jnp.asarray(_np.full(100, 0.1, dtype=_np.float32))

    _, y1 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_small, fill_small)
    )
    _, y2 = lax.scan(lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_big, fill_big))
    return y1, y2


def _two_scans_len_mismatch_broadcast_f64():
    xs_small = jnp.asarray(_np.arange(5, dtype=_np.float64))
    xs_big = jnp.asarray(_np.arange(100, dtype=_np.float64))

    fill_small = jnp.asarray(_np.full(5, 0.1, dtype=_np.float64))
    fill_big = jnp.asarray(_np.full(100, 0.1, dtype=_np.float64))

    _, y1 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_small, fill_small)
    )
    _, y2 = lax.scan(lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_big, fill_big))
    return y1, y2


def _two_scans_diff_len_with_broadcast_f32():
    xs_small = jnp.asarray(_np.arange(5, dtype=_np.float32))
    xs_big = jnp.asarray(_np.arange(100, dtype=_np.float32))

    _, y1 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c),
        0.0,
        xs=(xs_small, jnp.broadcast_to(0.1, xs_small.shape)),
    )
    _, y2 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c),
        0.0,
        xs=(xs_big, jnp.full_like(xs_big, 0.1)),
    )
    return y1, y2


# ----------------------------------------------------------------------
# post-conversion graph checks
# ----------------------------------------------------------------------
def _assert_scan_io_consistent(onnx_model) -> bool:
    def _elem_type_of(g, name):
        for vi in list(g.input) + list(g.value_info) + list(g.output):
            if vi.name == name and vi.type.tensor_type.elem_type:
                return vi.type.tensor_type.elem_type
        for init in g.initializer:
            if init.name == name:
                return init.data_type
        return None

    for n in onnx_model.graph.node:
        if n.op_type != "Scan":
            continue
        num_scan_inputs = None
        body_graph = None
        for a in n.attribute:
            if a.name == "num_scan_inputs":
                num_scan_inputs = a.i
            elif a.name == "body":
                body_graph = a.g
        assert (
            body_graph is not None and num_scan_inputs is not None
        ), "Scan missing body/num_scan_inputs"

        total_scan_inputs = len(n.input)
        body_inputs = len(body_graph.input)
        assert (
            total_scan_inputs == body_inputs
        ), f"Scan/body input arity mismatch: node has {total_scan_inputs}, body has {body_inputs}"

        # dtype equality check
        for i in range(total_scan_inputs):
            et_node = _elem_type_of(onnx_model.graph, n.input[i])
            et_body = _elem_type_of(body_graph, body_graph.input[i].name)
            assert (
                (et_node is None) or (et_body is None) or (et_node == et_body)
            ), f"Scan/body input dtype mismatch at index {i}: node={et_node}, body={et_body}"

        M = body_inputs - num_scan_inputs
        assert (
            len(body_graph.output) >= M
        ), f"Scan body must return >= M state outputs; got {len(body_graph.output)} < {M}"
    return True


# ---------------------- plugin registration (metadata trimmed) ----------------------
@register_primitive(
    jaxpr_primitive=lax.scan_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html",
    onnx=[
        {"component": "Scan", "doc": "https://onnx.ai/onnx/operators/onnx__Scan.html"}
    ],
    since="v0.5.1",
    context="primitives.lax",
    component="scan",
    testcases=[
        {
            "testcase": "scan_cumsum",
            "callable": lambda xs: lax.scan(lambda c, x: (c + x, c + x), 0.0, xs)[1],
            "input_shapes": [(5,)],
            "expected_output_shapes": [(5,)],
        },
        {
            "testcase": "scan_carry_only",
            "callable": lambda xs: lax.scan(lambda c, x: (c + x, c), 0.0, xs)[0],
            "input_shapes": [(3,)],
            "expected_output_shapes": [()],
        },
        {
            "testcase": "scan_multiple_sequences",
            "callable": lambda xs, ys: lax.scan(
                lambda c, xy: (c + xy[0] * xy[1], c + xy[0]), 0.0, (xs, ys)
            )[1],
            "input_shapes": [(4,), (4,)],
            "expected_output_shapes": [(4,)],
        },
        {
            "testcase": "scan_multiple_carry",
            "callable": lambda xs: lax.scan(
                lambda carry, x: ((carry[0] + x, carry[1] * x), carry[0] + carry[1]),
                (0.0, 1.0),
                xs,
            )[1],
            "input_shapes": [(3,)],
            "expected_output_shapes": [(3,)],
        },
        {
            "testcase": "scan_matrix_carry_multidim_xs",
            "callable": lambda init_carry, xs_seq: lax.scan(
                lambda c_mat, x_slice: (c_mat + x_slice, jnp.sum(c_mat + x_slice)),
                init_carry,
                xs_seq,
            )[1],
            "input_shapes": [(3, 2), (5, 3, 2)],
            "expected_output_shapes": [(5,)],
        },
        {
            "testcase": "scan_no_xs",
            "callable": lambda x: lax.scan(
                lambda carry, _: (carry + 1, carry), x, None, length=5
            )[1],
            "input_shapes": [()],
            "input_dtypes": [jnp.float32],
            "expected_output_shapes": [(5,)],
        },
        {
            "testcase": "scan_fn",
            "callable": scan_fn,
            "input_values": [jnp.array(0.0, dtype=jnp.float32)],
        },
        {
            "testcase": "scan_jit_no_xs",
            "callable": _scan_jit_no_xs,
            "input_shapes": [],
            "expected_output_shapes": [(10,)],
            "expected_output_dtypes": [jnp.int32],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "scan_jit_no_xs_f64",
            "callable": _scan_jit_no_xs,
            "input_shapes": [],
            "expected_output_shapes": [(10,)],
            "expected_output_dtypes": [jnp.int64],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "scan_captured_scalar",
            "callable": (
                lambda dt=jnp.asarray(0.1, dtype=jnp.float32): (
                    lax.scan(
                        lambda carry, _: (carry + dt, carry + dt),
                        jnp.asarray(0.0, dtype=jnp.float32),
                        xs=None,
                        length=3,
                    )[1]
                )
            ),
            "input_shapes": [],
            "expected_output_shapes": [(3,)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "scan_captured_scalar_f64",
            "callable": (
                lambda dt=jnp.asarray(0.1, dtype=jnp.float64): (
                    lax.scan(
                        lambda carry, _: (carry + dt, carry + dt),
                        jnp.asarray(0.0, dtype=jnp.float64),
                        xs=None,
                        length=3,
                    )[1]
                )
            ),
            "input_shapes": [],
            "expected_output_shapes": [(3,)],
            "expected_output_dtypes": [jnp.float64],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "scan_rank0_sequence_vectorized",
            "callable": (
                lambda xs_vec=jnp.arange(4, dtype=jnp.float32): lax.scan(
                    lambda carry, xs: (carry + xs[0] + xs[1], carry),
                    0.0,
                    xs=(xs_vec, jnp.full(xs_vec.shape, 0.1, dtype=jnp.float32)),
                )[1]
            ),
            "input_shapes": [],
            "expected_output_shapes": [(4,)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_rank0_sequence_vectorized_f64",
            "callable": (
                lambda xs_vec=jnp.arange(4, dtype=jnp.float64): lax.scan(
                    lambda carry, xs: (carry + xs[0] + xs[1], carry),
                    0.0,
                    xs=(xs_vec, jnp.full(xs_vec.shape, 0.1, dtype=jnp.float64)),
                )[1]
            ),
            "input_shapes": [],
            "expected_output_shapes": [(4,)],
            "expected_output_dtypes": [jnp.float64],
            "run_only_f64_variant": True,
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_two_diff_lengths",
            "callable": _two_scans_diff_len_f32,
            "input_shapes": [],
            "expected_output_shapes": [(5,), (100,)],
            "expected_output_dtypes": [jnp.float32, jnp.float32],
            "run_only_f32_variant": True,
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_two_diff_lengths_f64",
            "callable": _two_scans_diff_len_f64,
            "input_shapes": [],
            "expected_output_shapes": [(5,), (100,)],
            "expected_output_dtypes": [jnp.float64, jnp.float64],
            "run_only_f64_variant": True,
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_two_diff_lengths",
            "callable": _two_scans_diff_len_f32,
            "input_shapes": [],  # <- no inputs, everything is static
            "expected_output_shapes": [(5,), (100,)],
            "expected_output_dtypes": [jnp.float32, jnp.float32],
            "run_only_f32_variant": True,
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_two_diff_lengths_f64",
            "callable": _two_scans_diff_len_f64,
            "input_shapes": [],
            "expected_output_shapes": [(5,), (100,)],
            "expected_output_dtypes": [jnp.float64, jnp.float64],
            "run_only_f64_variant": True,
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_nested_len_mismatch",
            "callable": _nested_scan_len_mismatch_f32,
            "input_shapes": [],
            "expected_output_shapes": [(100,)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_nested_len_mismatch_f64",
            "callable": _nested_scan_len_mismatch_f64,
            "input_shapes": [],
            "expected_output_shapes": [(100,)],
            "expected_output_dtypes": [jnp.float64],
            "run_only_f64_variant": True,
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_two_diff_lengths_broadcast",
            "callable": _two_scans_len_mismatch_broadcast_f32,
            "input_shapes": [],
            "expected_output_shapes": [(5,), (100,)],
            "expected_output_dtypes": [jnp.float32, jnp.float32],
            "run_only_f32_variant": True,
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_two_diff_lengths_broadcast_f64",
            "callable": _two_scans_len_mismatch_broadcast_f64,
            "input_shapes": [],
            "expected_output_shapes": [(5,), (100,)],
            "expected_output_dtypes": [jnp.float64, jnp.float64],
            "run_only_f64_variant": True,
            "check_onnx_load": True,
        },
        # ── regression: scalar-broadcast + two different scan lengths ───────────
        {
            "testcase": "scan_two_diff_lengths_with_broadcast",
            "callable": _two_scans_diff_len_with_broadcast_f32,  # Updated to the new function name
            "input_shapes": [],  # <- no inputs, everything is static
            "expected_output_shapes": [(5,), (100,)],
            "expected_output_dtypes": [jnp.float32, jnp.float32],
            "run_only_f32_variant": True,  # do **not** run in "double" mode
        },
        {
            "testcase": "scan_two_diff_lengths_f64",
            "callable": _two_scans_diff_len_f64,
            "input_shapes": [],
            "expected_output_shapes": [(5,), (100,)],
            "expected_output_dtypes": [jnp.float64, jnp.float64],
            "run_only_f64_variant": True,
        },
        # ── regression: consts + xs must be threaded as state into the body ───
        {
            "testcase": "scan_captured_scalar_with_xs",
            "callable": (
                # dt is a captured constant (→ scan consts) and xs is the scanned input
                lambda xs, dt=jnp.asarray(0.1, dtype=jnp.float32): (
                    lax.scan(
                        lambda c, x: (c + dt * x, c + dt * x),  # uses dt and x
                        jnp.asarray(0.0, dtype=jnp.float32),  # scalar carry
                        xs,
                    )[1]
                )
            ),
            "input_shapes": [(8,)],  # xs length 8
            "expected_output_shapes": [(8,)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,  # match dt dtype
            "check_onnx_load": True,
            "post_check_onnx_graph": _assert_scan_io_consistent,
        },
        {
            "testcase": "scan_captured_vector_with_xs_f64",
            "callable": (
                # vector consts + vector xs at each step
                lambda xs, dt=jnp.asarray([0.1, -0.2], dtype=jnp.float64): (
                    lax.scan(
                        lambda c, x: (c + dt * x, c + dt * x),
                        jnp.zeros((2,), dtype=jnp.float64),  # vector carry
                        xs,
                    )[1]
                )
            ),
            "input_shapes": [(5, 2)],  # 5 steps, each x is (2,)
            "expected_output_shapes": [(5, 2)],
            "expected_output_dtypes": [jnp.float64],
            "run_only_f64_variant": True,  # match dt dtype
            "check_onnx_load": True,
            "post_check_onnx_graph": _assert_scan_io_consistent,
        },
    ],
)
class ScanPlugin(PrimitiveLeafPlugin):
    """Lower `lax.scan` to an ONNX Scan operator."""

    # --------------------------- abstract_eval ---------------------------
    @staticmethod
    def abstract_eval(
        *in_avals_flat: core.AbstractValue,
        jaxpr: ClosedJaxpr,
        length: int,
        reverse: bool,
        unroll: Union[int, bool],
        num_carry: int,
        num_xs: Optional[int] = None,
        num_consts: Optional[int] = None,
        **unused_params,
    ) -> Sequence[core.AbstractValue]:
        total_in = len(in_avals_flat)
        if num_xs is None:
            num_xs = max(0, total_in - num_carry)
        if num_carry is None:
            num_carry = max(0, total_in - num_xs)

        carry_avals = in_avals_flat[:num_carry]
        stacked: list[core.AbstractValue] = []
        for var in jaxpr.jaxpr.outvars[num_carry:]:
            aval = var.aval
            shape = tuple(aval.shape) if hasattr(aval, "shape") else ()
            stacked.append(core.ShapedArray((length,) + shape, aval.dtype))
        return tuple(carry_avals) + tuple(stacked)

    # ------------------------------ to_onnx ------------------------------
    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ) -> None:
        global _SCAN_INSTANCE_COUNTER
        scan_id = _SCAN_INSTANCE_COUNTER
        _SCAN_INSTANCE_COUNTER += 1

        closed_jaxpr = params["jaxpr"]
        num_carry = params["num_carry"]
        length = params["length"]
        # JAX scan has extra non-scanned args ("consts"). In ONNX we must thread them as state.
        num_consts = int(params.get("num_consts", 0) or 0)

        # Helper used in both Loop and Scan paths: decide if an output is a stacked y-output.
        # Avoid tuple indexing so mypy won't flag "Tuple index out of range".
        def _is_y_output(var: Var) -> bool:
            shp_any = getattr(var.aval, "shape", ())
            # Treat as an iterable; first dimension defines whether it is a stacked output.
            first_dim = next(iter(shp_any), None)
            return (
                first_dim is not None
                and isinstance(length, (int, _np.integer))
                and isinstance(first_dim, (int, _np.integer))
                and int(first_dim) == int(length)
            )

        # Number of true scanned inputs (exclude carry and consts).
        # Prefer JAX-provided `num_xs` if present to avoid misclassification in edge cases.
        total_invars = len(closed_jaxpr.jaxpr.invars)
        num_scan = int(
            params.get("num_xs", max(0, total_invars - num_carry - num_consts))
        )
        if num_scan < 0 or num_scan > total_invars:
            raise ValueError(
                f"Invalid num_scan={num_scan} for Scan with total_invars={total_invars}"
            )

        # ------------------------------------------------------------------
        # Special-case: no scan-inputs → Loop
        # Thread consts **and** carry as Loop state (M = num_consts + num_carry)
        # ------------------------------------------------------------------
        if num_scan == 0:
            trip_name = s.builder.get_unique_name("trip_count")
            s.builder.add_initializer(
                trip_name, [length], data_type=TensorProto.INT64, dims=[]
            )
            cond_name = s.builder.get_unique_name("cond_init")
            s.builder.add_initializer(
                cond_name, [1], data_type=TensorProto.BOOL, dims=[]
            )

            prefix = s.builder.name_generator.get("loop")
            body_builder = OnnxBuilder(
                name_generator=s.builder.name_generator,
                opset=s.builder.opset,
                model_name=s.builder.get_unique_name(f"{prefix}_body"),
            )
            body_builder.enable_double_precision = getattr(
                s.builder, "enable_double_precision", False
            )
            # IMPORTANT: keep subgraph symbol mapping isolated to avoid cross-scan leakage
            body_builder.var_to_symbol_map = {}  # do not share with outer builder
            body_conv = Jaxpr2OnnxConverter(body_builder)

            body_builder.add_input("iter_count", (), _np.int64)
            cond_in = body_builder.get_unique_name("cond_in")
            body_builder.add_input(cond_in, (), _np.bool_)

            # Body inputs: iter, cond, then all state in the order [consts..., carry...]
            for i, var in enumerate(
                closed_jaxpr.jaxpr.invars[: num_consts + num_carry]
            ):
                nm = body_builder.get_unique_name(f"state_in_{i}")
                # IMPORTANT: rank-only (all dynamic) to avoid ORT fixing concrete sizes
                # that can conflict with broadcasted shapes inside the Loop body.
                dyn_shape = (None,) * len(getattr(var.aval, "shape", ()))
                # Keep body input dtypes equal to the jaxpr aval dtypes; we Cast outer
                # symbols to these dtypes just before wiring the Loop inputs.
                body_builder.add_input(nm, dyn_shape, var.aval.dtype)
                body_conv.var_to_name[var] = nm

            for var, val in zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts):
                body_conv.var_to_name[var] = body_conv.get_constant_name(val)

            body_conv._process_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts)

            # --- dtype harmonization for binary numeric ops in Loop body ---
            # Allow tests to disable this to reproduce pre-fix failures.
            _disable_cast_env = os.getenv(
                "JAX2ONNX_DISABLE_LOOP_BINOP_CAST", ""
            ).lower()
            _disable_cast = _disable_cast_env in ("1", "true", "yes", "on")
            if not _disable_cast:
                _binops = {"Add", "Sub", "Mul", "Div"}
                for n in list(
                    body_builder.nodes
                ):  # list() in case we append while iterating
                    if n.op_type in _binops and len(n.input) >= 2:
                        a, b = n.input[0], n.input[1]
                        ta, tb = body_builder.get_dtype(a), body_builder.get_dtype(b)
                        if (
                            ta is not None
                            and tb is not None
                            # Align ANY numeric mismatch (floats or ints). Bool is not numeric.
                            and _np.issubdtype(_np.dtype(ta), _np.number)
                            and _np.issubdtype(_np.dtype(tb), _np.number)
                            and _np.dtype(ta) != _np.dtype(tb)
                        ):
                            # Cast RHS to LHS dtype and INSERT the Cast BEFORE this node
                            cast_out = body_builder.get_unique_name(
                                f"{b}_cast_{_np.dtype(ta).name}"
                            )
                            to_enum = _NP2ONNX[_np.dtype(ta)]
                            cast_node = helper.make_node(
                                "Cast",
                                inputs=[b],
                                outputs=[cast_out],
                                name=body_builder.get_unique_name("CastAlignBinOp"),
                                to=to_enum,
                            )
                            # Insert before current node to maintain topo order
                            idx = body_builder.nodes.index(n)
                            body_builder.nodes.insert(idx, cast_node)
                            # keep rank if we know it; otherwise be dynamic
                            r = body_builder.get_rank(b)
                            if r is None:
                                r = body_builder.get_rank(a)
                            if r is None:
                                r = 0
                            body_builder.add_value_info(cast_out, (None,) * r, ta)
                            n.input[1] = cast_out
            # --- end harmonization ---

            _retag_value_infos_to_input_dtype(body_builder)
            # Make internal shapes rank-only to avoid ORT Loop-body broadcast clashes.
            _loosen_value_infos_to_rank_only(body_builder)
            body_builder.outputs.clear()
            cond_out = body_builder.get_unique_name("cond_out")
            idn = helper.make_node(
                "Identity",
                inputs=[cond_in],
                outputs=[cond_out],
                name=body_builder.get_unique_name("id_cond"),
            )
            body_builder.add_node(idn)
            body_builder.add_output(cond_out, (), _np.bool_)

            # Emit body state outputs first: passthrough consts, then computed carries
            # 1) consts passthrough
            for ci in range(num_consts):
                in_sym = body_conv.get_name(closed_jaxpr.jaxpr.invars[ci])
                out_sym = body_builder.get_unique_name(f"const_out_{ci}")
                body_builder.add_node(
                    helper.make_node(
                        "Identity",
                        inputs=[in_sym],
                        outputs=[out_sym],
                        name=body_builder.get_unique_name("Identity_const_passthrough"),
                    )
                )
                aval = closed_jaxpr.jaxpr.invars[ci].aval
                # keep dtype identical to the body input (which we may have coerced)
                out_dt = body_builder.get_dtype(in_sym) or aval.dtype
                # rank-only output (avoid fixed sizes in Scan body outputs)
                body_builder.add_output(
                    out_sym, (None,) * len(getattr(aval, "shape", ())), out_dt
                )

            # 2) computed carry(s)
            seen_body = set()
            for cj in range(num_carry):
                carr_sym = body_conv.get_name(closed_jaxpr.jaxpr.outvars[cj])
                out_sym = (
                    carr_sym
                    if carr_sym not in seen_body
                    else body_builder.get_unique_name(f"{carr_sym}_dup")
                )
                if out_sym != carr_sym:
                    body_builder.add_node(
                        helper.make_node(
                            "Identity",
                            inputs=[carr_sym],
                            outputs=[out_sym],
                            name=body_builder.get_unique_name("Identity_carry_dup"),
                        )
                    )
                seen_body.add(out_sym)
                aval = closed_jaxpr.jaxpr.outvars[cj].aval
                # Use the actual dtype on the symbol if known (after any Cast we inserted).
                out_dt = body_builder.get_dtype(carr_sym) or aval.dtype
                # rank-only carry output
                body_builder.add_output(
                    out_sym, (None,) * len(getattr(aval, "shape", ())), out_dt
                )

            # 3) per-iter y outputs (no duplication of carry)
            #    Also avoid aliasing a graph *input* name as an output name.
            input_name_set = {i.name for i in body_builder.inputs}
            for var in closed_jaxpr.jaxpr.outvars[num_carry:]:
                orig = body_conv.get_name(var)
                out_name = (
                    orig
                    if orig not in seen_body
                    else body_builder.get_unique_name(f"{orig}_dup")
                )
                # duplicate if (a) we already used this name, or (b) it equals a graph input
                if out_name == orig and orig in input_name_set:
                    out_name = body_builder.get_unique_name(f"{orig}_dup")
                if out_name != orig:
                    body_builder.add_node(
                        helper.make_node(
                            "Identity",
                            inputs=[orig],
                            outputs=[out_name],
                            name=body_builder.get_unique_name("Identity_dup_scan0"),
                        )
                    )
                seen_body.add(out_name)
                y_dt = body_builder.get_dtype(orig) or var.aval.dtype
                # rank-only per-iteration output (Loop stacks them outside)
                body_builder.add_output(
                    out_name, (None,) * len(getattr(var.aval, "shape", ())), y_dt
                )

            loop_body = body_builder.create_graph(
                body_builder.model_name, is_subgraph=True
            )

            # Final pass to sanitize the Loop body VIs (always on)
            _loosen_graph_value_infos_to_rank_only(loop_body)

            # Pass state inputs = [consts..., carry...]
            # Ensure the symbols we feed into Loop match the body input (aval) dtypes.
            raw_state_syms = [
                s.get_name(v) for v in node_inputs[: num_consts + num_carry]
            ]
            state_in_syms: list[str] = []
            for i, sym in enumerate(raw_state_syms):
                aval_dt = getattr(closed_jaxpr.jaxpr.invars[i].aval, "dtype", None)
                sym_dt = s.builder.get_dtype(sym)
                # Always align state input dtype to aval dtype if aval dtype is known and
                # (a) upstream dtype is unknown, or (b) they differ.
                if aval_dt is not None and (
                    sym_dt is None or _np.dtype(sym_dt) != _np.dtype(aval_dt)
                ):
                    cast_out = s.builder.get_unique_name(
                        f"{sym}_cast_{_np.dtype(aval_dt).name}"
                    )
                    to_enum = _NP2ONNX.get(_np.dtype(aval_dt), TensorProto.FLOAT)
                    s.add_node(
                        helper.make_node(
                            "Cast",
                            inputs=[sym],
                            outputs=[cast_out],
                            name=s.get_unique_name("CastLoopStateIn"),
                            to=to_enum,
                        )
                    )
                    # Preserve rank info if we have it; otherwise make fully dynamic
                    rank = s.builder.get_rank(sym)
                    if rank is None:
                        rank = len(
                            getattr(closed_jaxpr.jaxpr.invars[i].aval, "shape", ())
                        )
                    s.add_shape_info(cast_out, (None,) * rank, aval_dt)
                    state_in_syms.append(cast_out)
                else:
                    state_in_syms.append(sym)

            loop_inputs = [trip_name, cond_name] + state_in_syms

            # Body returns [carry, y1, y2, ...] per iteration → primitive returns
            # [final_carry, ystack_1, ystack_2, ...]. We'll need this count to size Loop outs.
            jax_body_outvars = list(closed_jaxpr.jaxpr.outvars)
            num_y = len(jax_body_outvars) - num_carry
            if num_y < 0:
                raise ValueError("Internal error: num_y < 0 in Loop lowering.")

            # Create Loop node outputs in ONNX order: [const_state..., carry_state..., scan...]
            loop_tmp_outs: list[str] = []
            # state outs: consts passthrough first
            for ci in range(num_consts):
                var_in = closed_jaxpr.jaxpr.invars[ci]
                tmp = s.builder.get_unique_name(f"loop_const_raw_{ci}")
                inner_rank = len(getattr(var_in.aval, "shape", ()))
                # match dtype of the (possibly cast) symbol we feed into Loop for this const
                const_sym = state_in_syms[ci]
                const_dt = s.builder.get_dtype(const_sym) or getattr(
                    var_in.aval, "dtype", _np.float32
                )
                s.add_shape_info(tmp, (None,) * inner_rank, const_dt)
                loop_tmp_outs.append(tmp)
            # then computed carries
            for i in range(num_carry):
                var = jax_body_outvars[i]
                tmp = s.builder.get_unique_name(f"loop_state_raw_{i}")
                inner_rank = len(getattr(var.aval, "shape", ()))
                # Carry state out dtype must match its corresponding (possibly cast) state input
                carry_in_sym = state_in_syms[num_consts + i]
                carry_dt = s.builder.get_dtype(carry_in_sym) or getattr(
                    var.aval, "dtype", _np.float32
                )
                s.add_shape_info(tmp, (None,) * inner_rank, carry_dt)
                loop_tmp_outs.append(tmp)

            # scan (stacked) outs
            for k, var in enumerate(jax_body_outvars[num_carry:]):
                tmp = s.builder.get_unique_name(f"loop_scan_raw_{k}")
                inner_rank = len(getattr(var.aval, "shape", ()))
                s.add_shape_info(
                    tmp,
                    (None,) * (inner_rank + 1),
                    getattr(var.aval, "dtype", _np.float32),
                )
                loop_tmp_outs.append(tmp)

            loop_node = helper.make_node(
                "Loop",
                inputs=loop_inputs,
                outputs=loop_tmp_outs,
                name=s.get_unique_name("Loop"),
                body=loop_body,
            )
            s.add_node(loop_node)

            # ----- Forward ONLY the requested primitive outputs (skip const passthroughs) -----
            # Primitive results: [carry_0..carry_{C-1}, ystack_0..ystack_{K-1}]
            # Loop outputs:      [const_0..const_{S-1}, carry_0..carry_{C-1}, ystack_0..ystack_{K-1}]
            #
            # Node may request any subsequence (often drops the carry). Classify by shape:
            # a y-output is stacked along trip axis => leading dim == `length` (when static).
            def _is_y_output(var: Var) -> bool:
                shp_any = getattr(var.aval, "shape", ())
                # Treat as an iterable; first dimension defines whether it is a stacked output.
                first_dim = next(iter(shp_any), None)
                return (
                    first_dim is not None
                    and isinstance(length, (int, _np.integer))
                    and isinstance(first_dim, (int, _np.integer))
                    and int(first_dim) == int(length)
                )

            carry_idx = 0
            y_idx = 0
            for out_var in node_outputs:
                if not isinstance(out_var, Var):
                    continue  # DropVar or equivalent
                if _is_y_output(out_var):
                    src_name = loop_tmp_outs[num_consts + num_carry + y_idx]
                    y_idx += 1
                else:
                    src_name = loop_tmp_outs[num_consts + carry_idx]
                    carry_idx += 1
                final_name = s.get_name(out_var)
                s.add_node(
                    helper.make_node(
                        "Identity",
                        inputs=[src_name],
                        outputs=[final_name],
                        name=s.get_unique_name("LoopOut"),
                    )
                )
                # Replace any stale VI and attach rank/dtype consistent with the var
                s.builder.value_info[:] = [
                    vi for vi in s.builder.value_info if vi.name != final_name
                ]
                inner_rank = len(getattr(out_var.aval, "shape", ()))
                # carry preserves rank; y has the stacked leading axis already present in aval
                out_rank = inner_rank
                dtype = s.builder.get_dtype(src_name) or getattr(
                    out_var.aval, "dtype", _np.float32
                )
                # Prefer the aval dtype for the outward contract; VI dtype is only a fallback
                s.add_shape_info(
                    final_name,
                    (None,) * out_rank,
                    getattr(out_var.aval, "dtype", dtype),
                )

            return

        # ------------------------------------------------------------------
        # Build Scan body graph (identical to previous version)
        # ------------------------------------------------------------------
        jaxpr = closed_jaxpr.jaxpr
        consts = closed_jaxpr.consts

        body_builder = OnnxBuilder(
            name_generator=s.builder.name_generator,
            opset=s.builder.opset,
            model_name=s.builder.get_unique_name("scan_body"),
        )
        body_builder.enable_double_precision = getattr(
            s.builder, "enable_double_precision", False
        )
        # IMPORTANT: keep subgraph symbol mapping isolated to avoid cross-scan leakage
        body_builder.var_to_symbol_map = {}  # do not share with outer builder
        body_conv = Jaxpr2OnnxConverter(body_builder)

        # --- dtype inference helpers ---
        def _infer_dtype_from_producer(bld, sym):
            # Look for the node that produces `sym` and try to use any input dtype as output dtype.
            for n in reversed(bld.nodes):
                if sym in n.output:
                    for inp in n.input:
                        dt = bld.get_dtype(inp)
                        if dt is not None:
                            return dt
                    return None
            return None

        def _dtype_or_infer(bld, sym, fallback):
            return (
                bld.get_dtype(sym) or _infer_dtype_from_producer(bld, sym) or fallback
            )

        # declare inputs (exactly the jaxpr.invars; order is [consts, carry, xs])
        for i, var in enumerate(jaxpr.invars):
            nm = body_builder.get_unique_name(f"scan_body_in_{i}")
            dyn_shp = (None,) * len(var.aval.shape)
            # Always use the jaxpr aval dtype for body inputs (both state and xs).
            body_builder.add_input(nm, dyn_shp, var.aval.dtype)
            body_conv.var_to_name[var] = nm

        body_conv._process_jaxpr(jaxpr, consts)

        # --- dtype harmonization for binary numeric ops in Scan body (same as Loop) ---
        # Allow tests to disable this to reproduce pre-fix failures.
        _disable_cast_env = os.getenv("JAX2ONNX_DISABLE_LOOP_BINOP_CAST", "").lower()
        _disable_cast = _disable_cast_env in ("1", "true", "yes", "on")
        if not _disable_cast:
            _binops = {"Add", "Sub", "Mul", "Div"}
            for n in list(
                body_builder.nodes
            ):  # list() in case we append while iterating
                if n.op_type in _binops and len(n.input) >= 2:
                    a, b = n.input[0], n.input[1]
                    ta = body_builder.get_dtype(a)
                    tb = body_builder.get_dtype(b)
                    if (
                        ta is not None
                        and tb is not None
                        and _np.issubdtype(_np.dtype(ta), _np.number)
                        and _np.issubdtype(_np.dtype(tb), _np.number)
                        and _np.dtype(ta) != _np.dtype(tb)
                    ):
                        # Cast RHS to LHS dtype and INSERT the Cast BEFORE this node
                        cast_out = body_builder.get_unique_name(
                            f"{b}_cast_{_np.dtype(ta).name}"
                        )
                        to_enum = _NP2ONNX[_np.dtype(ta)]
                        cast_node = helper.make_node(
                            "Cast",
                            inputs=[b],
                            outputs=[cast_out],
                            name=body_builder.get_unique_name("CastAlignBinOp"),
                            to=to_enum,
                        )
                        # Insert before current node to maintain topo order
                        idx = body_builder.nodes.index(n)
                        body_builder.nodes.insert(idx, cast_node)
                        # keep rank if we know it; otherwise be dynamic
                        r = body_builder.get_rank(b)
                        if r is None:
                            r = body_builder.get_rank(a)
                        if r is None:
                            r = 0
                        body_builder.add_value_info(cast_out, (None,) * r, ta)
                        n.input[1] = cast_out

        # Retag internal VIs: e.g. Squeeze/Gather/Add should inherit input dtype,
        # which may be f64 under enable_double_precision.
        _retag_value_infos_to_input_dtype(body_builder)

        # Re-emit body outputs in ONNX-required order:
        #   1) state outputs:   consts (passthrough)  +  **computed carries**
        #   2) stacked y-outputs (ONLY jaxpr.outvars[num_carry:]) → fully dynamic dims
        body_builder.outputs.clear()
        num_carry + num_consts
        # 1) consts passthrough
        for ci in range(num_consts):
            in_sym = body_conv.get_name(closed_jaxpr.jaxpr.invars[ci])
            out_sym = body_builder.get_unique_name(f"const_out_{ci}")
            body_builder.add_node(
                helper.make_node(
                    "Identity",
                    inputs=[in_sym],
                    outputs=[out_sym],
                    name=body_builder.get_unique_name("Identity_const_passthrough"),
                )
            )
            aval = closed_jaxpr.jaxpr.invars[ci].aval
            # keep dtype identical to the body input (which we may have coerced)
            out_dt = body_builder.get_dtype(in_sym) or aval.dtype
            # rank-only output (avoid fixed sizes in Scan body outputs)
            body_builder.add_output(
                out_sym, (None,) * len(getattr(aval, "shape", ())), out_dt
            )

        # 2) computed carry(s)
        seen_body = set()
        for cj in range(num_carry):
            carr_sym = body_conv.get_name(closed_jaxpr.jaxpr.outvars[cj])
            out_sym = (
                carr_sym
                if carr_sym not in seen_body
                else body_builder.get_unique_name(f"{carr_sym}_dup")
            )
            if out_sym != carr_sym:
                body_builder.add_node(
                    helper.make_node(
                        "Identity",
                        inputs=[carr_sym],
                        outputs=[out_sym],
                        name=body_builder.get_unique_name("Identity_carry_dup"),
                    )
                )
            seen_body.add(out_sym)
            aval = closed_jaxpr.jaxpr.outvars[cj].aval
            # Use the actual dtype on the symbol if known (after any Cast we inserted).
            out_dt = body_builder.get_dtype(carr_sym) or aval.dtype
            # rank-only carry output
            body_builder.add_output(
                out_sym, (None,) * len(getattr(aval, "shape", ())), out_dt
            )

        # 3) per-iter y outputs (no duplication of carry)
        #    Also avoid aliasing a graph *input* name as an output name.
        input_name_set = {i.name for i in body_builder.inputs}
        for var in closed_jaxpr.jaxpr.outvars[num_carry:]:
            orig = body_conv.get_name(var)
            out_name = (
                orig
                if orig not in seen_body
                else body_builder.get_unique_name(f"{orig}_dup")
            )
            # duplicate if (a) we already used this name, or (b) it equals a graph input
            if out_name == orig and orig in input_name_set:
                out_name = body_builder.get_unique_name(f"{orig}_dup")
            if out_name != orig:
                body_builder.add_node(
                    helper.make_node(
                        "Identity",
                        inputs=[orig],
                        outputs=[out_name],
                        name=body_builder.get_unique_name("Identity_dup_scan0"),
                    )
                )
            seen_body.add(out_name)
            y_dt = body_builder.get_dtype(orig) or var.aval.dtype
            # rank-only per-iteration output (Loop stacks them outside)
            body_builder.add_output(
                out_name, (None,) * len(getattr(var.aval, "shape", ())), y_dt
            )

        body_graph = body_builder.create_graph(
            body_builder.model_name, is_subgraph=True
        )

        # Final pass to sanitize the Scan body VIs (always on, like Loop)
        _loosen_graph_value_infos_to_rank_only(body_graph)

        # Debug-only invariants for early detection of wiring errors
        if os.getenv("JAX2ONNX_DEBUG_SCAN_ASSERTS", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            # Inputs: [consts..., carry..., xs...]
            expected_in = num_consts + num_carry + num_scan
            assert (
                len(body_graph.input) == expected_in
            ), f"Scan body input arity mismatch: got {len(body_graph.input)}, expected {expected_in}"
            # Outputs: [consts..., carries..., y...]
            num_y = len(jaxpr.outvars) - num_carry
            expected_out = num_consts + num_carry + num_y
            assert (
                len(body_graph.output) == expected_out
            ), f"Scan body output arity mismatch: got {len(body_graph.output)}, expected {expected_out}"

        # ------------------------------------------------------------------
        # Broadcast rank-0 sequence inputs  (operate ONLY on xs, not carries)
        # ------------------------------------------------------------------
        # State = consts + carry (they are part of node_inputs, in that order).
        # Cast incoming state to the JAX aval dtype to keep Scan body math precise.
        state_syms_raw = [s.get_name(v) for v in node_inputs[: num_consts + num_carry]]
        state_syms: list[str] = []
        for i, sym in enumerate(state_syms_raw):
            aval_dt = jaxpr.invars[i].aval.dtype
            sym_dt = s.builder.get_dtype(sym)
            # Cast when upstream dtype is unknown OR differs. Works for floats and ints.
            if aval_dt is not None and (
                sym_dt is None or _np.dtype(aval_dt) != _np.dtype(sym_dt)
            ):
                cast_out = s.builder.get_unique_name(
                    f"{sym}_cast_{_np.dtype(aval_dt).name}"
                )
                to_enum = _NP2ONNX.get(_np.dtype(aval_dt), TensorProto.FLOAT)
                s.add_node(
                    helper.make_node(
                        "Cast",
                        inputs=[sym],
                        outputs=[cast_out],
                        name=s.get_unique_name("CastScanStateIn"),
                        to=to_enum,
                    )
                )
                rank = s.builder.get_rank(sym)
                if rank is None:
                    rank = len(getattr(jaxpr.invars[i].aval, "shape", ()))
                s.add_shape_info(cast_out, (None,) * rank, aval_dt)
                state_syms.append(cast_out)
            else:
                state_syms.append(sym)

        xs_syms = [
            s.get_name(v)
            for v in node_inputs[
                num_consts + num_carry : num_consts + num_carry + num_scan
            ]
        ]

        trip_shape_sym: str | None = None  # will hold a 1-D [trip] tensor

        for i in range(num_scan):
            var = node_inputs[num_consts + num_carry + i]
            if len(var.aval.shape) == 0:
                ref_var = None
                # look for a reference among *real* scan inputs
                for cand in node_inputs[
                    num_consts + num_carry : num_consts + num_carry + num_scan
                ]:
                    if len(cand.aval.shape) > 0:
                        ref_var = cand
                        break

                if ref_var is not None:
                    ref_sym = s.get_name(ref_var)
                    shape_sym = s.builder.get_unique_name(f"shape_scan{scan_id}_{i}")
                    s.add_node(
                        helper.make_node(
                            "Shape",
                            inputs=[ref_sym],
                            outputs=[shape_sym],
                            name=s.get_unique_name("Shape_trip"),
                        )
                    )
                    s.add_shape_info(shape_sym, (None,), i64)

                    trip_dim = s.builder.get_unique_name(f"trip_dim_scan{scan_id}_{i}")
                    s.builder.add_initializer(trip_dim, [0], data_type=INT64, dims=[])

                    gather_sym = s.builder.get_unique_name(
                        f"trip_len_scan{scan_id}_{i}"
                    )
                    s.add_node(
                        helper.make_node(
                            "Gather",
                            inputs=[shape_sym, trip_dim],
                            outputs=[gather_sym],
                            name=s.get_unique_name("Gather_trip"),
                            axis=0,
                        )
                    )
                    s.add_shape_info(gather_sym, (), i64)

                    axes_sym = s.builder.get_unique_name(f"axes_scan{scan_id}_{i}")
                    s.builder.add_initializer(axes_sym, [0], data_type=INT64, dims=[1])
                    unsq_sym = s.builder.get_unique_name(
                        f"trip_len_vec_scan{scan_id}_{i}"
                    )
                    s.add_node(
                        helper.make_node(
                            "Unsqueeze",
                            inputs=[gather_sym, axes_sym],
                            outputs=[unsq_sym],
                            name=s.get_unique_name("Unsqueeze_trip"),
                        )
                    )
                    s.add_shape_info(unsq_sym, (1,), i64)

                    broadcast_shape = unsq_sym
                    trip_shape_sym = trip_shape_sym or unsq_sym
                    exsym = s.builder.get_unique_name(f"{xs_syms[i]}_exp")
                    s.add_node(
                        helper.make_node(
                            "Expand",
                            inputs=[xs_syms[i], broadcast_shape],
                            outputs=[exsym],
                            name=s.get_unique_name("Expand_broadcast"),
                        )
                    )
                    s.add_shape_info(exsym, (None,), var.aval.dtype)
                    xs_syms[i] = exsym
                else:
                    gather_sym = s.builder.get_unique_name(
                        f"trip_len_vec_scan{scan_id}_{i}"
                    )
                    init_val = (
                        int(length) if isinstance(length, (int, _np.integer)) else 1
                    )
                    s.builder.add_initializer(
                        gather_sym, [init_val], data_type=INT64, dims=[1]
                    )
                    s.add_shape_info(gather_sym, (1,), i64)
                    broadcast_shape = gather_sym
                    trip_shape_sym = trip_shape_sym or gather_sym
                    exsym = s.builder.get_unique_name(f"{xs_syms[i]}_exp")
                    s.add_node(
                        helper.make_node(
                            "Expand",
                            inputs=[xs_syms[i], broadcast_shape],
                            outputs=[exsym],
                            name=s.get_unique_name("Expand_broadcast"),
                        )
                    )
                    s.add_shape_info(exsym, (None,), var.aval.dtype)
                    xs_syms[i] = exsym

        # Pass 2: scalar initializers that are still rank-0
        if trip_shape_sym is None:
            trip_shape_sym = s.builder.get_unique_name(f"trip_len_scan{scan_id}")
            safe_len = int(length) if isinstance(length, (int, _np.integer)) else 1
            s.builder.add_initializer(
                trip_shape_sym, [safe_len], data_type=INT64, dims=[1]
            )
            s.add_shape_info(trip_shape_sym, (1,), i64)

        # Expand any rank-0 scanned inputs. Choose fallback dtype consistently
        # with the jaxpr aval dtype (so body inputs & node inputs match).
        for i in range(num_scan):
            sym = xs_syms[i]
            if s.builder.get_rank(sym) == 0:
                dtype = jaxpr.invars[num_consts + num_carry + i].aval.dtype
                exsym = s.builder.get_unique_name(f"{sym}_exp")
                s.add_node(
                    helper.make_node(
                        "Expand",
                        inputs=[sym, trip_shape_sym],
                        outputs=[exsym],
                        name=s.get_unique_name("Expand_broadcast_init"),
                    )
                )
                s.add_shape_info(exsym, (None,), dtype)
                xs_syms[i] = exsym

        # NOTE: Do NOT pass model inputs directly into Scan. ORT will merge their concrete
        # shapes (from graph.input) with Scan’s inferred trip axis and can produce conflicts.
        # Instead, feed dynamic-shape proxies via Identity.
        xs_syms_dyn: list[str] = []
        for i, sym in enumerate(xs_syms):
            proxy = s.builder.get_unique_name(f"{sym}_dyn")
            s.add_node(
                helper.make_node(
                    "Identity",
                    inputs=[sym],
                    outputs=[proxy],
                    name=s.get_unique_name("ScanXSProxy"),
                )
            )
            rank = s.builder.get_rank(sym)
            if rank is None:
                rank = len(node_inputs[num_consts + num_carry + i].aval.shape)
            dtype = s.builder.get_dtype(sym) or getattr(
                node_inputs[num_consts + num_carry + i].aval,
                "dtype",
                jaxpr.invars[num_consts + num_carry + i].aval.dtype,
            )
            s.add_shape_info(proxy, (None,) * rank, dtype)
            xs_syms_dyn.append(proxy)

        # Align scanned input dtypes to body aval dtypes as well (mirrors state alignment).
        xs_syms_aligned: list[str] = []
        for i, sym in enumerate(xs_syms_dyn):
            aval_dt = jaxpr.invars[num_consts + num_carry + i].aval.dtype
            sym_dt = s.builder.get_dtype(sym)
            if aval_dt is not None and (
                sym_dt is None or _np.dtype(aval_dt) != _np.dtype(sym_dt)
            ):
                cast_out = s.builder.get_unique_name(
                    f"{sym}_cast_{_np.dtype(aval_dt).name}"
                )
                to_enum = _NP2ONNX.get(_np.dtype(aval_dt), TensorProto.FLOAT)
                s.add_node(
                    helper.make_node(
                        "Cast",
                        inputs=[sym],
                        outputs=[cast_out],
                        name=s.get_unique_name("CastScanXSIn"),
                        to=to_enum,
                    )
                )
                rank = s.builder.get_rank(sym)
                if rank is None:
                    rank = len(
                        getattr(
                            jaxpr.invars[num_consts + num_carry + i].aval, "shape", ()
                        )
                    )
                s.add_shape_info(cast_out, (None,) * rank, aval_dt)
                xs_syms_aligned.append(cast_out)
            else:
                xs_syms_aligned.append(sym)

        onnx_inputs = state_syms + xs_syms_aligned

        # ------------------------------------------------------------------
        # Build top-level Scan node
        # ------------------------------------------------------------------
        jaxpr = closed_jaxpr.jaxpr
        num_y = len(jaxpr.outvars) - num_carry
        num_carry + num_consts

        # 1) Allocate temporary outputs in ONNX order
        scan_tmp_outs: list[str] = []

        # a) consts-as-state (no direct mapping to final outputs)
        for ci in range(num_consts):
            tmp = s.builder.get_unique_name(f"scan_const_raw_{ci}")
            scan_tmp_outs.append(tmp)
            # Const state out dtype must match the const state *input*
            const_in_sym = state_syms[ci]
            const_dt = s.builder.get_dtype(const_in_sym) or jaxpr.invars[ci].aval.dtype
            s.add_shape_info(tmp, (None,) * len(jaxpr.invars[ci].aval.shape), const_dt)

        # b) real carry outputs
        for j in range(num_carry):
            tmp = s.builder.get_unique_name(f"scan_carry_raw_{j}")
            scan_tmp_outs.append(tmp)
            aval_src = jaxpr.outvars[j].aval
            # Carry state out dtype must match carry state *input*
            carry_in_sym = state_syms[num_consts + j]
            carry_dt = s.builder.get_dtype(carry_in_sym) or aval_src.dtype
            s.add_shape_info(tmp, (None,) * len(aval_src.shape), carry_dt)

        # c) stacked y-outputs (make dims fully dynamic; keep only rank)
        for k in range(num_y):
            tmp = s.builder.get_unique_name(f"scan_y_raw_{k}")
            scan_tmp_outs.append(tmp)
            y_aval_body = jaxpr.outvars[num_carry + k].aval
            s.add_shape_info(
                tmp,
                (None,) * (len(y_aval_body.shape) + 1),
                y_aval_body.dtype,
            )

        # Debug-only invariant: the Scan node outs must match body outs
        if os.getenv("JAX2ONNX_DEBUG_SCAN_ASSERTS", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            assert len(scan_tmp_outs) == len(body_graph.output), (
                f"Scan node/output arity mismatch: node outs={len(scan_tmp_outs)}, "
                f"body outs={len(body_graph.output)}"
            )

        # Build Scan node attributes
        attrs: dict[str, Any] = {
            "body": body_graph,
            "num_scan_inputs": num_scan,
        }
        if num_scan:
            attrs["scan_input_axes"] = [0] * num_scan
        # Number of scan (stacked) outputs N = total outs - state outs (consts + carries)
        N = len(scan_tmp_outs) - (num_carry + num_consts)
        if N:
            attrs["scan_output_axes"] = [0] * N

        scan_node = helper.make_node(
            "Scan",
            inputs=onnx_inputs,
            outputs=scan_tmp_outs,
            name=s.get_unique_name("scan"),
            **attrs,
        )
        s.add_node(scan_node)

        carry_idx = 0
        y_idx = 0
        for out_var in node_outputs:
            if not isinstance(out_var, Var):
                continue  # dropped output
            if _is_y_output(out_var):
                src_name = scan_tmp_outs[num_consts + num_carry + y_idx]
                y_idx += 1
            else:
                src_name = scan_tmp_outs[num_consts + carry_idx]
                carry_idx += 1
            final_name = s.get_name(out_var)
            s.add_node(
                helper.make_node(
                    "Identity",
                    inputs=[src_name],
                    outputs=[final_name],
                    name=s.get_unique_name("ScanOut"),
                )
            )
            # Replace any stale VI and set rank/dtype from the requested var.
            s.builder.value_info[:] = [
                vi for vi in s.builder.value_info if vi.name != final_name
            ]
            out_rank = len(getattr(out_var.aval, "shape", ()))
            out_dt = s.builder.get_dtype(src_name) or getattr(
                out_var.aval, "dtype", _np.float32
            )
            # Prefer the aval dtype for the outward contract; VI dtype is only a fallback
            s.add_shape_info(
                final_name, (None,) * out_rank, getattr(out_var.aval, "dtype", out_dt)
            )

        return
