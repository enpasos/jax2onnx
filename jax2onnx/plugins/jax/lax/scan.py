from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import jax
import jax.numpy as jnp
from jax import core, lax
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scan")

# Define the primitive for lax.scan
scan_p = Primitive("scan")
scan_p.multiple_results = True


@register_primitive(
    jaxpr_primitive=scan_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html",
    onnx=[
        {"component": "Scan", "doc": "https://onnx.ai/onnx/operators/onnx__Scan.html"}
    ],
    since="upcoming",
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
    ],
)
class ScanPlugin(PrimitiveLeafPlugin):
    """Lower `lax.scan` to ONNX Scan operator."""

    @staticmethod
    def abstract_eval(
        *in_avals: core.AbstractValue,
        body_jaxpr: core.ClosedJaxpr,
        length: int,
        reverse: bool,
        unroll: int,
        **kwargs,
    ) -> Sequence[core.AbstractValue]:
        num_leaves = kwargs.get("num_leaves_per_arg")
        if num_leaves is None:
            raise ValueError("Missing 'num_leaves_per_arg' in abstract_eval kwargs.")

        num_carry = num_leaves[1]
        jaxpr = body_jaxpr.jaxpr

        if len(jaxpr.invars) < num_carry or len(jaxpr.outvars) < num_carry:
            raise ValueError(
                f"Body jaxpr mismatch: invars/outvars = {len(jaxpr.invars)}/{len(jaxpr.outvars)} vs carry={num_carry}"
            )

        carry_avals = in_avals[:num_carry]
        stacked_avals = []
        for var in jaxpr.outvars[num_carry:]:
            aval = var.aval
            if not isinstance(aval, core.ShapedArray):
                raise TypeError(f"Expected ShapedArray, got {type(aval)} for {var}")
            stacked_avals.append(core.ShapedArray((length, *aval.shape), aval.dtype))

        return (*carry_avals, *stacked_avals)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[core.Var],
        node_outputs: Sequence[core.Var],
        params: dict[str, Any],
    ) -> None:
        # Extract or synthesize the body Jaxpr and parameters
        if "body_jaxpr" in params:
            closed = params["body_jaxpr"]
            jaxpr = closed.jaxpr
            consts = closed.consts
            num_carry, num_scan = params["num_leaves_per_arg"][1:3]
        else:
            # fallback: raw jaxpr eqn params
            jaxpr_param = params.get("jaxpr")
            from jax.extend.core import ClosedJaxpr

            if isinstance(jaxpr_param, ClosedJaxpr):
                jaxpr = jaxpr_param.jaxpr
                consts = jaxpr_param.consts
            else:
                jaxpr = jaxpr_param
                consts = []
            num_carry = params.get("num_carry")
            num_scan = len(jaxpr.invars) - num_carry

        # Build subgraph body
        body_builder = OnnxBuilder(
            name_generator=s.builder.name_generator,
            opset=s.builder.opset,
            model_name=s.builder.get_unique_name("scan_body"),
        )
        from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

        body_conv = Jaxpr2OnnxConverter(body_builder)

        # Map subgraph inputs (carry + scan slice)
        for i, var in enumerate(jaxpr.invars):
            name = body_builder.get_unique_name(f"scan_in_{i}")
            body_builder.add_input(name, var.aval.shape, var.aval.dtype)
            body_conv.var_to_name[var] = name

        # Map constants
        for var, val in zip(jaxpr.constvars, consts):
            cname = body_conv.get_constant_name(val)
            body_conv.var_to_name[var] = cname

        # Process body operations
        body_conv._process_jaxpr(jaxpr, consts)

        # Map subgraph outputs (carry outs + scan outs)
        body_builder.outputs.clear()
        for var in jaxpr.outvars:
            name = body_conv.get_name(var)
            body_builder.add_output(name, var.aval.shape, var.aval.dtype)

        body_graph = body_builder.create_graph(
            body_builder.model_name, is_subgraph=True
        )

        # Create main Scan node
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
        s.add_node(scan_node)


# Bind abstract evaluation
scan_p.def_abstract_eval(ScanPlugin.abstract_eval)
