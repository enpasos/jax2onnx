# jax2onnx/plugins/jax/lax/split.py

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.converter.source_split import require_native_split_emission
from jax2onnx.plugins._complex_utils import coerce_dim_values
from jax2onnx.plugins._utils import normalize_builder_outputs
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive=jax.lax.split_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.split.html",
    onnx=[
        {"component": "Split", "doc": "https://onnx.ai/onnx/operators/onnx__Split.html"}
    ],
    since="0.7.2",
    context="primitives.lax",
    component="split",
    testcases=[
        {
            "testcase": "lax_split_equal_parts",
            "callable": lambda x: jax.lax.split(x, (3, 3), axis=1),
            "input_shapes": [(4, 6)],
            "post_check_onnx_graph": EG(
                ["Split:4x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "lax_split_unequal_parts",
            "callable": lambda x: jax.lax.split(x, (2, 3, 4), axis=1),
            "input_shapes": [(4, 9)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Split",
                        "counts": {"Split": 1},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "lax_split_single_output",
            "callable": lambda x: jax.lax.split(x, (6,), axis=1),
            "input_shapes": [(4, 6)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Split",
                        "counts": {"Split": 1},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SplitPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.split`` to ONNX ``Split`` using IR-only ops."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        if ctx.precision_policy is PrecisionPolicy.SOURCE:
            self._lower_source_native(ctx, eqn)
            return
        data_var = eqn.invars[0]
        out_vars = list(eqn.outvars)

        axis = int(eqn.params.get("axis", 0))
        sizes = tuple(int(s) for s in eqn.params.get("sizes", ()))
        if not sizes:
            raise ValueError("lax.split requires non-empty sizes parameter")

        rank = len(getattr(data_var.aval, "shape", ()))
        if rank == 0:
            raise ValueError("lax.split expects the operand to have rank >= 1")
        axis = axis % rank

        data_val = ctx.get_value_for_var(
            data_var, name_hint=ctx.fresh_name("split_data")
        )

        splits_val = _const_i64(ctx, np.asarray(sizes, dtype=np.int64), "split_sizes")
        _stamp_type_and_shape(splits_val, (len(sizes),))
        _ensure_value_metadata(ctx, splits_val)

        out_vals = [
            ctx.get_value_for_var(var, name_hint=ctx.fresh_name("split_out"))
            for var in out_vars
        ]

        out_names = [
            getattr(val, "name", None) or ctx.fresh_name("split_out")
            for val in out_vals
        ]
        split_results = normalize_builder_outputs(
            ctx.builder.Split(
                data_val,
                splits_val,
                _outputs=out_names,
                axis=axis,
            )
        )

        data_dtype = getattr(getattr(data_val, "type", None), "dtype", None)
        for var, val, res in zip(out_vars, out_vals, split_results):
            out_shape = tuple(getattr(var.aval, "shape", ()))
            if data_dtype is not None:
                res.type = ir.TensorType(data_dtype)
            _stamp_type_and_shape(res, out_shape)
            _ensure_value_metadata(ctx, res)
            ctx.bind_value_for_var(var, res)

    def _lower_source_native(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        require_native_split_emission(eqn, ctx)
        source = representation_for_aval(eqn.invars[0].aval)
        data = ctx.require_value_for_var(eqn.invars[0])
        if not source.matches_ir_value(data):
            raise TypeError(
                "SOURCE Split physical input differs from its native declaration"
            )
        # Sizes and axis are logical. A declared trailing complex-pair component
        # axis, if later admitted by its own owner, remains attached to each piece.
        sizes = _const_i64(ctx, eqn.params["sizes"], "split_sizes")
        outputs = [
            ctx.allocate_value_for_var(v, name_hint=ctx.fresh_name("split_out"))
            for v in eqn.outvars
        ]
        results = ctx.builder.op_multi_out(
            "Split",
            [data, sizes],
            {"axis": eqn.params["axis"]},
            outputs=outputs,
        )
        for variable, result in zip(eqn.outvars, results, strict=True):
            representation = representation_for_aval(variable.aval)
            result.type = ir.TensorType(representation.ir_dtype)
            _stamp_type_and_shape(
                result, coerce_dim_values(representation.physical_shape)
            )
            _ensure_value_metadata(ctx, result)
            ctx.bind_value_for_var(variable, result)
