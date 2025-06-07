from __future__ import annotations

import logging  # ensure logger is available
from typing import TYPE_CHECKING, Any, Sequence, Union  # added Union
from typing import Optional

import jax.numpy as jnp
from jax import core, lax
from onnx import helper

from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scan")


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
    ],
)
class ScanPlugin(PrimitiveLeafPlugin):
    """Lower `lax.scan` to ONNX Scan operator."""

    @staticmethod
    def abstract_eval(
        *in_avals_flat: core.AbstractValue,  # carry avals + scan inputs
        jaxpr: core.ClosedJaxpr,  # body as ClosedJaxpr
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
        node_inputs: Sequence[core.Var],
        node_outputs: Sequence[core.Var],
        params: dict[str, Any],
    ) -> None:
        closed_jaxpr = params["jaxpr"]  # ClosedJaxpr
        num_carry = params["num_carry"]
        num_scan = params.get("num_xs")  # may be absent for single-seq
        if num_scan is None:
            # derive from invars minus carry
            num_scan = len(closed_jaxpr.jaxpr.invars) - num_carry
        jaxpr = closed_jaxpr.jaxpr
        consts = closed_jaxpr.consts

        # Build subgraph body
        body_builder = OnnxBuilder(
            name_generator=s.builder.name_generator,
            opset=s.builder.opset,
            model_name=s.builder.get_unique_name("scan_body"),
        )
        from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

        # Use a fresh converter for the subgraph body (not a plugin instance)
        body_conv = Jaxpr2OnnxConverter(body_builder)

        # Map subgraph inputs (carry + scan slice)
        for i, var in enumerate(jaxpr.invars):
            name = body_builder.get_unique_name(f"scan_in_{i}")
            body_builder.add_input(name, var.aval.shape, var.aval.dtype)
            body_conv.var_to_name[var] = name

        # Map constants and body ops
        for var, val in zip(jaxpr.constvars, consts):
            cname = body_conv.get_constant_name(val)
            body_conv.var_to_name[var] = cname
        body_conv._process_jaxpr(jaxpr, consts)

        # Map subgraph outputs
        body_builder.outputs.clear()
        for var in jaxpr.outvars:
            name = body_conv.get_name(var)
            body_builder.add_output(name, var.aval.shape, var.aval.dtype)
        body_graph = body_builder.create_graph(
            body_builder.model_name, is_subgraph=True
        )

        # Create ONNX Scan node
        scan_node = helper.make_node(
            "Scan",
            inputs=[s.get_name(v) for v in node_inputs],
            outputs=[s.get_name(v) for v in node_outputs],
            name=s.builder.get_unique_name("ScanOp"),
            body=body_graph,
            num_scan_inputs=num_scan,
            scan_input_axes=[0] * num_scan,
            scan_output_axes=[0] * (len(jaxpr.outvars) - num_carry),
        )
        logger.debug(
            f"Emitting ONNX Scan node: num_carry={num_carry}, num_scan_inputs={num_scan}, "
            f"inputs={len(node_inputs)}, outputs={len(node_outputs)}"
        )
        s.add_node(scan_node)
