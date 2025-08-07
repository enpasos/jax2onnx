"""ONNX plug-in for **jax.lax.rev** (array reversal along given axes)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np
from jax import core, lax
from jax.extend.core import Var
from onnx import helper, TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
from jax2onnx.converter.onnx_builder   import OnnxBuilder

# -----------------------------------------------------------------------------
# -- plug-in registration metadata
# -----------------------------------------------------------------------------
@register_primitive(
    jaxpr_primitive=lax.rev_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.rev.html",
    onnx=[{"component": "Flip", "doc": "https://onnx.ai/onnx/operators/onnx__Flip.html"}],
    since="v0.7.5",
    context="primitives.lax",
    component="rev",
    testcases=[
        {
            "testcase": "rev_vector",
            "callable": lambda x: lax.rev(x, (0,)),
            "input_shapes": [(5,)],
            "expected_output_shapes": [(5,)],
        },
        {
            "testcase": "rev_matrix_axes01",
            "callable": lambda x: lax.rev(x, (0, 1)),
            "input_shapes": [(3, 4)],
            "expected_output_shapes": [(3, 4)],
        },
    ],
)
class RevPlugin(PrimitiveLeafPlugin):
    """Lower `lax.rev` to the ONNX **Flip** operator (opset ≥ 18)."""

    # ───────────────────── abstract-eval (shape propagation) ──────────────────
    @staticmethod
    def abstract_eval(
        x_aval: core.ShapedArray, *,
        dimensions: Sequence[int],       # provided by JAX in eqn.params
        **unused_params,
    ) -> Sequence[core.ShapedArray]:
        # Reversing does not change shape or dtype.
        return (x_aval,)

    # ──────────────────────────────── to_onnx ────────────────────────────────
    def to_onnx(
        self,
        s: Jaxpr2OnnxConverter,
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ) -> None:
        """
        Implementation without the (not-yet-supported) Flip op:

        For every axis to be reversed
            1.  len  = Shape(x)[axis]
            2.  idxs = Range(len-1, -1, -1)                # [len-1 … 0]
            3.  x    = Gather(x, idxs, axis=axis)           # reversed
        """

        axes        = [int(a) for a in params["dimensions"]]
        x_sym_in    = s.get_name(node_inputs[0])
        current_sym = x_sym_in

        # shared scalar constants
        one_name     = s.builder.get_unique_name("const_one")
        neg1_name    = s.builder.get_unique_name("const_neg1")
        s.builder.add_initializer(one_name,  [1],  data_type=TensorProto.INT64, dims=[])
        s.builder.add_initializer(neg1_name, [-1], data_type=TensorProto.INT64, dims=[])

        original_aval = node_inputs[0].aval        # shape / dtype stay constant
        for ax_idx, axis in enumerate(axes):
            # -- len = Shape(x)[axis] -------------------------------------------------
            shape_sym = s.builder.get_unique_name(f"shape_rev{ax_idx}")
            s.add_node(
                helper.make_node(
                    "Shape",
                    inputs=[current_sym],
                    outputs=[shape_sym],
                    name=s.get_unique_name("Shape_len"),
                )
            )
            s.add_shape_info(shape_sym, (None,), _np.int64)

            axis_const = s.builder.get_unique_name(f"axis_rev{ax_idx}")
            s.builder.add_initializer(
                axis_const, [axis], data_type=TensorProto.INT64, dims=[]
            )

            len_sym = s.builder.get_unique_name(f"len_rev{ax_idx}")
            s.add_node(
                helper.make_node(
                    "Gather",
                    inputs=[shape_sym, axis_const],
                    outputs=[len_sym],
                    name=s.get_unique_name("Gather_len"),
                    axis=0,
                )
            )
            s.add_shape_info(len_sym, (), _np.int64)

            # -- start = len-1 --------------------------------------------------------
            start_sym = s.builder.get_unique_name(f"start_rev{ax_idx}")
            s.add_node(
                helper.make_node(
                    "Sub",
                    inputs=[len_sym, one_name],
                    outputs=[start_sym],
                    name=s.get_unique_name("Sub_start"),
                )
            )
            s.add_shape_info(start_sym, (), _np.int64)

            # -- idxs = Range(start, -1, -1) -----------------------------------------
            range_sym = s.builder.get_unique_name(f"range_rev{ax_idx}")
            s.add_node(
                helper.make_node(
                    "Range",
                    inputs=[start_sym, neg1_name, neg1_name],
                    outputs=[range_sym],
                    name=s.get_unique_name("Range_rev"),
                )
            )
            s.add_shape_info(range_sym, (None,), _np.int64)

            # -- x = Gather(x, idxs, axis) -------------------------------------------
            out_sym = s.builder.get_unique_name(f"rev_out{ax_idx}")
            s.add_node(
                helper.make_node(
                    "Gather",
                    inputs=[current_sym, range_sym],
                    outputs=[out_sym],
                    name=s.get_unique_name("Gather_rev"),
                    axis=axis,
                )
            )
            # ⇨ output has exactly the same static shape & dtype as the input
            s.add_shape_info(out_sym,
                             original_aval.shape,
                             _np.dtype(original_aval.dtype))

            current_sym = out_sym

        final_out_sym = s.get_name(node_outputs[0])
        if current_sym != final_out_sym:
            s.add_node(
                helper.make_node(
                    "Identity",
                    inputs=[current_sym],
                    outputs=[final_out_sym],
                    name=s.get_unique_name("Identity_rev"),
                )
            )

        aval = node_inputs[0].aval
        s.add_shape_info(final_out_sym, aval.shape, _np.dtype(aval.dtype))
