# file: jax2onnx/plugins/jax/lax/scan.py
from __future__ import annotations

import logging
import numpy as _np
from typing import Any, Sequence, Union, Optional

import jax
import jax.numpy as jnp
from jax import core, lax
from onnx import TensorProto, helper

from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax.extend.core import ClosedJaxpr, Var

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scan")


INT64 = TensorProto.INT64
i64 = _np.int64
# unique id for every Scan node – used when generating helper names
_SCAN_INSTANCE_COUNTER: int = 0


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
    # use NumPy constants to avoid data-dependent dynamic shapes
    xs_small = jnp.asarray(_np.arange(5, dtype=_np.float32))
    xs_big = jnp.asarray(_np.arange(100, dtype=_np.float32))
    fill_small = jnp.asarray(_np.full(xs_small.shape, 0.1, dtype=_np.float32))
    fill_big = jnp.asarray(_np.full(xs_big.shape, 0.1, dtype=_np.float32))

    _, y1 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c),
        0.0,
        xs=(xs_small, fill_small),
    )
    _, y2 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c),
        0.0,
        xs=(xs_big, fill_big),
    )
    return y1, y2


# ----------------------------------------------------------------------
# regression -- nested scan: inner length 5, outer length 100
# ----------------------------------------------------------------------
def _nested_scan_len_mismatch_f32():
    xs_outer = jnp.asarray(_np.arange(100, dtype=_np.float32))  # length 100
    xs_inner = jnp.asarray(_np.arange(5, dtype=_np.float32))  # length 5
    fill_inn = jnp.broadcast_to(0.1, xs_inner.shape)  # ← broadcast_to !

    def inner(c, xs):  # trip-count = 5
        c = c + xs[0] + xs[1]
        return c, c

    def outer(c, x):  # trip-count = 100
        _, ys = lax.scan(inner, c, xs=(xs_inner, fill_inn))
        return c + x, ys[-1]  # use something from the inner scan

    _, ys_out = lax.scan(outer, 0.0, xs_outer)
    return ys_out  # shape == (100,)


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
    # use NumPy constants to avoid data-dependent dynamic shapes
    xs_small = jnp.asarray(_np.arange(5, dtype=_np.float64))
    xs_big = jnp.asarray(_np.arange(100, dtype=_np.float64))
    fill_small = jnp.asarray(_np.full(xs_small.shape, 0.1, dtype=_np.float64))
    fill_big = jnp.asarray(_np.full(xs_big.shape, 0.1, dtype=_np.float64))

    _, y1 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c),
        0.0,
        xs=(xs_small, fill_small),
    )
    _, y2 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c),
        0.0,
        xs=(xs_big, fill_big),
    )
    return y1, y2


# helpers.py  (snippet)


def _two_scans_len_mismatch_broadcast_f32():
    xs_small = jnp.asarray(_np.arange(5, dtype=_np.float32))  # len = 5
    xs_big = jnp.asarray(_np.arange(100, dtype=_np.float32))  # len = 100

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


def _two_scans_diff_len_with_broadcast_f32():  # Renamed from _two_scans_diff_len_f32
    xs_small = jnp.asarray(_np.arange(5, dtype=_np.float32))
    xs_big = jnp.asarray(_np.arange(100, dtype=_np.float32))

    # 1️⃣ scan length = 5  – uses  jnp.broadcast_to  (⇒ scalar expand)
    _, y1 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c),
        0.0,
        xs=(xs_small, jnp.broadcast_to(0.1, xs_small.shape)),
    )

    # 2️⃣ scan length = 100 – uses  jnp.full_like    (⇒ no scalar expand)
    _, y2 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c),
        0.0,
        xs=(xs_big, jnp.full_like(xs_big, 0.1)),
    )
    return y1, y2


# ----------------------
# plugin registration
# ----------------------
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
        # ────────────────────────────────────────────────────────────────────────
    ],
)
class ScanPlugin(PrimitiveLeafPlugin):
    """Lower `lax.scan` to ONNX Scan operator."""

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
        num_scan = len(node_inputs) - num_carry

        # ------------------------------------------------------------------
        # Special-case: no scan-inputs → Loop
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
            body_builder.var_to_symbol_map = s.builder.var_to_symbol_map
            body_conv = Jaxpr2OnnxConverter(body_builder)

            body_builder.add_input("iter_count", (), _np.int64)
            cond_in = body_builder.get_unique_name("cond_in")
            body_builder.add_input(cond_in, (), _np.bool_)

            for i, var in enumerate(closed_jaxpr.jaxpr.invars[:num_carry]):
                nm = body_builder.get_unique_name(f"carry_in_{i}")
                body_builder.add_input(nm, var.aval.shape, var.aval.dtype)
                body_conv.var_to_name[var] = nm

            for var, val in zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts):
                body_conv.var_to_name[var] = body_conv.get_constant_name(val)

            body_conv._process_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts)

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

            seen_body = set()
            for var in closed_jaxpr.jaxpr.outvars:
                orig = body_conv.get_name(var)
                out_name = (
                    orig
                    if orig not in seen_body
                    else body_builder.get_unique_name(f"{orig}_dup")
                )
                if out_name != orig:
                    dup = helper.make_node(
                        "Identity",
                        inputs=[orig],
                        outputs=[out_name],
                        name=body_builder.get_unique_name("Identity_dup_scan0"),
                    )
                    body_builder.add_node(dup)
                seen_body.add(out_name)
                body_builder.add_output(out_name, var.aval.shape, var.aval.dtype)

            loop_body = body_builder.create_graph(
                body_builder.model_name, is_subgraph=True
            )
            loop_inputs = [trip_name, cond_name] + [s.get_name(v) for v in node_inputs]
            loop_outputs = [s.get_name(v) for v in node_outputs]
            loop_node = helper.make_node(
                "Loop",
                inputs=loop_inputs,
                outputs=loop_outputs,
                name=s.get_unique_name("Loop"),
                body=loop_body,
            )
            s.add_node(loop_node)
            for sym, v in zip(loop_outputs, node_outputs):
                s.add_shape_info(sym, v.aval.shape, v.aval.dtype)
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
        body_builder.var_to_symbol_map = s.builder.var_to_symbol_map
        body_conv = Jaxpr2OnnxConverter(body_builder)

        for i, var in enumerate(jaxpr.invars):
            nm = body_builder.get_unique_name(f"scan_body_in_{i}")
            body_builder.add_input(nm, var.aval.shape, var.aval.dtype)
            body_conv.var_to_name[var] = nm

        for var, val in zip(jaxpr.constvars, consts):
            body_conv.var_to_name[var] = body_conv.get_constant_name(val)

        body_conv._process_jaxpr(jaxpr, consts)

        body_builder.outputs.clear()
        seen = set()
        for var in jaxpr.outvars:
            orig = body_conv.get_name(var)
            out = (
                orig
                if orig not in seen
                else body_builder.get_unique_name(f"{orig}_dup")
            )
            if out != orig:
                dup = helper.make_node(
                    "Identity",
                    inputs=[orig],
                    outputs=[out],
                    name=body_builder.get_unique_name("Identity_dup"),
                )
                body_builder.add_node(dup)
            seen.add(out)
            body_builder.add_output(out, var.aval.shape, var.aval.dtype)

        body_graph = body_builder.create_graph(
            body_builder.model_name, is_subgraph=True
        )

        # ------------------------------------------------------------------
        # Broadcast rank-0 sequence inputs
        # ------------------------------------------------------------------
        onnx_inputs = [s.get_name(v) for v in node_inputs]
        trip_shape_sym: str | None = None  # will hold a 1-D [trip] tensor

        for i in range(num_scan):
            slot = num_carry + i
            var = node_inputs[slot]

            # rank-0 scan-input → needs broadcast
            if len(var.aval.shape) == 0:
                ref_var = None
                for cand in node_inputs[num_carry : num_carry + num_scan]:
                    if len(cand.aval.shape) > 0:  # first non-scalar xs
                        ref_var = cand
                        break

                if ref_var is not None:
                    # ── path A: we *do* have a true vector xs ──────────────────
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

                    # constant '0' for Gather
                    trip_dim = s.builder.get_unique_name(f"trip_dim_scan{scan_id}_{i}")
                    s.builder.add_initializer(trip_dim, [0], data_type=INT64, dims=[])

                    # Gather gives a scalar trip-length
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
                    s.add_shape_info(gather_sym, (), i64)  #   correct

                    # --- Unsqueeze scalar → 1-D [trip_len] ---------------------------------
                    axes_sym = s.builder.get_unique_name(f"axes_scan{scan_id}_{i}")
                    s.builder.add_initializer(axes_sym, [0], data_type=INT64, dims=[1])
                    unsq_sym = s.builder.get_unique_name(
                        f"trip_len_vec_scan{scan_id}_{i}"
                    )
                    s.add_node(
                        helper.make_node(
                            "Unsqueeze",
                            inputs=[gather_sym, axes_sym],  # data, axes
                            outputs=[unsq_sym],
                            name=s.get_unique_name("Unsqueeze_trip"),
                        )
                    )
                    s.add_shape_info(unsq_sym, (1,), i64)

                    # 1-D `[trip_len]` is already perfect for Expand
                    broadcast_shape = unsq_sym
                    trip_shape_sym = trip_shape_sym or unsq_sym
                    # ----- broadcast the scalar data tensor ---------------------
                    exsym = s.builder.get_unique_name(f"{onnx_inputs[slot]}_exp")
                    s.add_node(
                        helper.make_node(
                            "Expand",
                            inputs=[onnx_inputs[slot], broadcast_shape],
                            outputs=[exsym],
                            name=s.get_unique_name("Expand_broadcast"),
                        )
                    )
                    s.add_shape_info(exsym, (None,), var.aval.dtype)
                    onnx_inputs[slot] = exsym
                else:
                    # ── path B: *all* scan-inputs are scalars ───────────────────
                    # use the (static) `length` parameter itself
                    gather_sym = s.builder.get_unique_name(
                        f"trip_len_vec_scan{scan_id}_{i}"
                    )
                    # If `length` is still the JAX-to-ONNX dynamic sentinel we
                    # *cannot* embed it in an initializer – ONNX expects a real
                    # int.  In that corner-case we fall back to a dummy '1'. The
                    # tensor is never consumed when all `xs` are scalars, but
                    # ORT still needs a valid constant here.
                    init_val = (
                        int(length)
                        if isinstance(length, (int, _np.integer))
                        else 1  # harmless placeholder for dynamic case
                    )
                    # dims=[1] ⇒ rank-1 tensor [trip_count | 1]
                    s.builder.add_initializer(
                        gather_sym, [init_val], data_type=INT64, dims=[1]
                    )
                    s.add_shape_info(gather_sym, (1,), i64)
                    broadcast_shape = gather_sym
                    trip_shape_sym = trip_shape_sym or gather_sym
                    # ----- broadcast the scalar data tensor ---------------------
                    exsym = s.builder.get_unique_name(f"{onnx_inputs[slot]}_exp")
                    s.add_node(
                        helper.make_node(
                            "Expand",
                            inputs=[onnx_inputs[slot], broadcast_shape],
                            outputs=[exsym],
                            name=s.get_unique_name("Expand_broadcast"),
                        )
                    )
                    s.add_shape_info(exsym, (None,), var.aval.dtype)
                    onnx_inputs[slot] = exsym

        # ────────────────────────────────────────────────────────────────
        # Pass 2:  scalar **initializers** that never appeared as Vars
        #          (e.g. `jnp.asarray(0.1)`) – ORT still sees rank-0.
        # ────────────────────────────────────────────────────────────────
        if trip_shape_sym is None:  # no vector xs at all
            trip_shape_sym = s.builder.get_unique_name(f"trip_len_scan{scan_id}")
            safe_len = int(length) if isinstance(length, (int, _np.integer)) else 1
            s.builder.add_initializer(
                trip_shape_sym, [safe_len], data_type=INT64, dims=[1]
            )
            s.add_shape_info(trip_shape_sym, (1,), i64)

        for gidx in range(num_carry, num_carry + num_scan):
            sym = onnx_inputs[gidx]
            rank = s.builder.get_rank(sym)
            if rank == 0:  # still a scalar
                dtype = s.builder.get_dtype(sym) or _np.float32
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
                onnx_inputs[gidx] = exsym

        # ------------------------------------------------------------------
        # Build top-level Scan node
        # ------------------------------------------------------------------
        num_y = len(jaxpr.outvars) - num_carry
        total_out = num_carry + num_y

        onnx_outputs: list[str] = []
        for idx in range(total_out):
            out_var = node_outputs[idx]
            if isinstance(out_var, Var):
                onnx_outputs.append(s.get_name(out_var))
            else:
                tmp = s.builder.get_unique_name(f"scan_unused_output_{idx}")
                onnx_outputs.append(tmp)
                aval = jaxpr.outvars[idx].aval
                if idx < num_carry:
                    # carry values keep their static shape
                    s.add_shape_info(tmp, aval.shape, aval.dtype)
                else:
                    # **do not bake the trip-count in** – leave it dynamic
                    s.add_shape_info(tmp, (None,) + aval.shape, aval.dtype)

        attrs: dict[str, Any] = {
            "body": body_graph,
            "num_scan_inputs": num_scan,
        }
        if num_scan:
            attrs["scan_input_axes"] = [0] * num_scan
        if num_y:
            attrs["scan_output_axes"] = [0] * num_y

        # ---- create the *single* Scan node ----------------------------------------
        scan_node = helper.make_node(
            "Scan",
            inputs=onnx_inputs,
            outputs=onnx_outputs,
            name=s.get_unique_name("scan"),
            **attrs,
        )
        s.add_node(scan_node)

        # --- ensure trip-count dim is left dynamic on outputs ----------------
        for idx, (sym, var) in enumerate(zip(onnx_outputs, node_outputs)):
            if idx < num_carry or not isinstance(var, Var):
                continue
            s.builder.value_info[:] = [
                vi for vi in s.builder.value_info if vi.name != sym
            ]
            s.add_shape_info(sym, (None,) + var.aval.shape, var.aval.dtype)
