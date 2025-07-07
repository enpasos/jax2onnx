# file: jax2onnx/plugins/jax/lax/scan.py

from __future__ import annotations

from onnx import TensorProto
import logging  # ensure logger is available
from typing import Any, Sequence, Union  # added Union
from typing import Optional

import jax
import jax.numpy as jnp
from jax import core, lax
from onnx import helper

from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
from jax.extend.core import ClosedJaxpr, Var


logger = logging.getLogger("jax2onnx.plugins.jax.lax.scan")


def scan_fn(x):
    def body(carry, _):
        carry = carry + 1
        return carry, carry

    carry, ys = lax.scan(body, x, None, length=5)
    return ys


# ----------------------------------------------------------------------
# New helper that reproduces the "simulate → main → jax.jit" pattern
# ----------------------------------------------------------------------
def _scan_jit_no_xs() -> jax.Array:
    """
    Mirrors the example in the issue:

    ```
    def simulate():
        def step_fn(carry, _):
            return carry + 1, carry * 2
        return lax.scan(step_fn, 0, xs=None, length=10)[1]

    def main():
        return jax.jit(simulate)()
    ```
    """

    def simulate():
        def step_fn(carry, _):
            new_carry = carry + 1
            output = carry * 2
            return new_carry, output

        # xs=None  →   Loop-style scan, needs explicit length
        _, ys = lax.scan(step_fn, 0, xs=None, length=10)
        return ys

    # JIT-compile exactly as in the sample code
    return jax.jit(simulate)()


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
                # Body receives a 2D slice (3, 2), carry is also (3, 2)
                lambda c_mat, x_slice: (
                    c_mat + x_slice,  # New carry state (3, 2)
                    jnp.sum(c_mat + x_slice),  # Output per step (scalar)
                ),
                init_carry,  # Initial carry state (3, 2)
                xs_seq,  # Sequence input (5, 3, 2)
            )[
                1
            ],  # Return the stacked scalar sums
            # Input shapes: [shape_of_init_carry, shape_of_xs_seq]
            "input_shapes": [(3, 2), (5, 3, 2)],
            "expected_output_shapes": [(5,)],  # Expect stacked scalar sums
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
            "input_shapes": [],  # simulate() takes no arguments
            "expected_output_shapes": [(10,)],  # length = 10 → stacked outputs
            "expected_output_dtypes": [jnp.int32],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "scan_jit_no_xs_f64",
            "callable": _scan_jit_no_xs,
            "input_shapes": [],  # simulate() takes no arguments
            "expected_output_shapes": [(10,)],  # length = 10 → stacked outputs
            "expected_output_dtypes": [jnp.int64],
            "run_only_f64_variant": True,
        },
    ],
)
class ScanPlugin(PrimitiveLeafPlugin):
    """Lower `lax.scan` to ONNX Scan operator."""

    @staticmethod
    def abstract_eval(
        *in_avals_flat: core.AbstractValue,  # carry avals + scan inputs
        jaxpr: ClosedJaxpr,  # body as ClosedJaxpr
        length: int,  # scan length
        reverse: bool,
        unroll: Union[int, bool],  # Union[int, True, False]
        num_carry: int,  # how many carry vars
        num_xs: Optional[int] = None,  # may be missing for single-seq
        num_consts: Optional[int] = None,  # present in newer JAX versions
        **unused_params,  # catch others like 'linear'
    ) -> Sequence[core.AbstractValue]:
        # ------------------------------------------------------------------ #
        # Derive missing parameters                                          #
        # ------------------------------------------------------------------ #
        # Robust inference for num_xs and num_carry
        total_inputs = len(in_avals_flat)
        # If num_xs is not provided, try to infer it
        if num_xs is None:
            if num_carry is not None:
                num_xs = max(0, total_inputs - num_carry)
            else:
                # Fallback: try to infer from jaxpr
                num_xs = 0
        # If num_carry is not provided, try to infer it
        if num_carry is None:
            if num_xs is not None:
                num_carry = max(0, total_inputs - num_xs)
            else:
                num_carry = 0
        # Defensive: if still ambiguous, log and raise
        if num_xs < 0 or num_carry < 0 or (num_xs + num_carry != total_inputs):
            logger.error(
                f"Cannot robustly determine scan input/carry split: "
                f"len(in_avals_flat)={total_inputs}, num_carry={num_carry}, num_xs={num_xs}, "
                f"in_avals_flat={in_avals_flat}"
            )
            raise ValueError("Cannot determine number of scan inputs/carry")

        # Build carry avals
        carry_avals = in_avals_flat[:num_carry]
        # Extract inner jaxpr
        body_jaxpr = jaxpr.jaxpr
        # Build stacked outputs from body_jaxpr.outvars after carry
        stacked_avals = []
        for var in body_jaxpr.outvars[num_carry:]:
            aval = var.aval
            if not isinstance(aval, core.ShapedArray):
                logger.error(
                    f"Expected ShapedArray for scan body output, got {type(aval)}"
                )
                if not (hasattr(aval, "shape") and hasattr(aval, "dtype")):
                    raise TypeError(f"No shape/dtype on {var}")
            shape = tuple(aval.shape) if hasattr(aval, "shape") else ()
            stacked_avals.append(core.ShapedArray((length,) + shape, aval.dtype))
        return tuple(carry_avals) + tuple(stacked_avals)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ) -> None:
        """Lower `lax.scan` to ONNX Scan operator."""

        # 1. Extract parameters from the JAX primitive
        closed_jaxpr = params["jaxpr"]
        num_carry = params["num_carry"]
        length = params["length"]
        num_scan = len(node_inputs) - num_carry

        # —————————————————————————————
        # Special-case num_scan == 0
        # —————————————————————————————
        if num_scan == 0:
            import numpy as _np

            # 1) trip-count initializer
            trip_name = s.builder.get_unique_name("trip_count")
            s.builder.add_initializer(
                trip_name, [length], data_type=TensorProto.INT64, dims=[]
            )

            # 2) loop-condition initializer (always true)
            cond_name = s.builder.get_unique_name("cond_init")
            s.builder.add_initializer(
                cond_name, [1], data_type=TensorProto.BOOL, dims=[]
            )

            # Build the Loop‐body subgraph
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

            # Loop body inputs: iteration count (int64), cond (bool), then carry vars
            body_builder.add_input("iter_count", (), _np.int64)
            cond_in_name = body_builder.get_unique_name("cond_in")
            body_builder.add_input(cond_in_name, (), _np.bool_)
            for i, var in enumerate(closed_jaxpr.jaxpr.invars[:num_carry]):
                nm = body_builder.get_unique_name(f"carry_in_{i}")
                body_builder.add_input(nm, var.aval.shape, var.aval.dtype)
                body_conv.var_to_name[var] = nm

            # Map any constants
            for var, val in zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts):
                body_conv.var_to_name[var] = body_conv.get_constant_name(val)

            # -- Process the body JAXPR -------------------------
            body_conv._process_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts)

            # Re-declare **all** body outputs in the correct order:
            body_builder.outputs.clear()

            # 1) cond_out = Identity(cond_in)
            cond_out = body_builder.get_unique_name("cond_out")
            idn = helper.make_node(
                "Identity",
                inputs=[cond_in_name],
                outputs=[cond_out],
                name=body_builder.get_unique_name("id_cond"),
            )
            body_builder.add_node(idn)
            body_builder.add_output(cond_out, (), _np.bool_)

            # 2) carry_outs and scan_outs with duplicate handling
            seen_body_outputs: set[str] = set()
            for var in closed_jaxpr.jaxpr.outvars:
                orig_name = body_conv.get_name(var)
                out_name = orig_name
                if orig_name in seen_body_outputs:
                    out_name = body_builder.get_unique_name(f"{orig_name}_dup")
                    id_node = helper.make_node(
                        "Identity",
                        inputs=[orig_name],
                        outputs=[out_name],
                        name=body_builder.get_unique_name("Identity_dup_scan0"),
                    )
                    body_builder.add_node(id_node)
                seen_body_outputs.add(out_name)
                body_builder.add_output(out_name, var.aval.shape, var.aval.dtype)

            loop_body = body_builder.create_graph(
                body_builder.model_name, is_subgraph=True
            )

            # Now hook up the ONNX Loop node
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

        # ————————————————————————————————————————————————————————————— #
        # "Normal" case: there are X sequences → use Scan as before         #
        # ————————————————————————————————————————————————————————————— #
        jaxpr = closed_jaxpr.jaxpr
        consts = closed_jaxpr.consts

        # 2. Create and configure the subgraph builder for the loop body
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

        # 3. Map inputs for the subgraph body
        for i, var in enumerate(jaxpr.invars):
            name = body_builder.get_unique_name(f"scan_body_in_{i}")
            # body jaxpr already sees per-step shapes, so use them directly
            aval_shape = var.aval.shape
            body_builder.add_input(name, aval_shape, var.aval.dtype)
            body_conv.var_to_name[var] = name

        # 4. Process the subgraph body
        for var, val in zip(jaxpr.constvars, consts):
            body_conv.var_to_name[var] = body_conv.get_constant_name(val)
        body_conv._process_jaxpr(jaxpr, consts)

        # 5. Map outputs for the subgraph body, handling duplicate Vars
        body_builder.outputs.clear()
        seen: set[str] = set()
        for var in jaxpr.outvars:
            orig_name = body_conv.get_name(var)
            out_name = orig_name
            if orig_name in seen:
                # second output needs its own name: wire through an Identity
                out_name = body_builder.get_unique_name(f"{orig_name}_dup")
                id_node = helper.make_node(
                    "Identity",
                    inputs=[orig_name],
                    outputs=[out_name],
                    name=body_builder.get_unique_name("Identity_dup"),
                )
                body_builder.add_node(id_node)
            seen.add(out_name)
            body_builder.add_output(out_name, var.aval.shape, var.aval.dtype)
        body_graph = body_builder.create_graph(
            body_builder.model_name, is_subgraph=True
        )

        # 6. Prepare inputs and outputs for the main ONNX Scan node
        onnx_inputs = [s.get_name(v) for v in node_inputs]

        # The ONNX Scan op's outputs are ordered: [final_carries..., stacked_ys...]
        # The JAX eqn's `node_outputs` corresponds to this, with `_` for unused outputs.
        num_y_outputs = len(jaxpr.outvars) - num_carry
        total_onnx_outputs = num_carry + num_y_outputs

        onnx_outputs = []
        for i in range(total_onnx_outputs):
            jax_out_var = node_outputs[i]
            if isinstance(jax_out_var, Var):
                onnx_outputs.append(s.get_name(jax_out_var))
            else:
                # This output is discarded (`_`), create a dummy name and register its info
                name = s.builder.get_unique_name(f"scan_unused_output_{i}")
                onnx_outputs.append(name)

                body_out_aval = jaxpr.outvars[i].aval
                if i < num_carry:  # It's an unused final carry
                    s.add_shape_info(name, body_out_aval.shape, body_out_aval.dtype)
                else:  # It's an unused stacked 'y'
                    stacked_shape = (length,) + body_out_aval.shape
                    s.add_shape_info(name, stacked_shape, body_out_aval.dtype)

        # 7. Define attributes for the ONNX Scan node
        node_attributes = {
            "body": body_graph,
            "num_scan_inputs": num_scan,
        }
        if num_scan > 0:
            node_attributes["scan_input_axes"] = [0] * num_scan
        if num_y_outputs > 0:
            node_attributes["scan_output_axes"] = [0] * num_y_outputs

        # 8. Create and add the ONNX Scan node
        scan_node = helper.make_node(
            "Scan",
            inputs=onnx_inputs,
            outputs=onnx_outputs,
            name=s.get_unique_name("scan"),
            **node_attributes,
        )
        s.add_node(scan_node)
