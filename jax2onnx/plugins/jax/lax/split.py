# jax2onnx/plugins/jax/lax/split.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.split_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.split.html",
    onnx=[
        {"component": "Split", "doc": "https://onnx.ai/onnx/operators/onnx__Split.html"}
    ],
    since="v0.7.2",
    context="primitives.lax",
    component="split",
    testcases=[
        {
            "testcase": "lax_split_equal_parts",
            "callable": lambda x: jnp.split(x, 2, axis=1),
            "input_shapes": [(4, 6)],
            "post_check_onnx_graph": EG(
                ["Split:4x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "lax_split_unequal_parts",
            "callable": lambda x: jnp.split(x, [2, 5], axis=1),
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
    ],
)
class SplitPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.split`` to ONNX ``Split`` using IR-only ops."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
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
        split_results = ctx.builder.Split(
            data_val,
            splits_val,
            _outputs=out_names,
            axis=axis,
        )

        data_dtype = getattr(getattr(data_val, "type", None), "dtype", None)
        for var, val, res in zip(out_vars, out_vals, split_results):
            out_shape = tuple(getattr(var.aval, "shape", ()))
            if data_dtype is not None:
                res.type = ir.TensorType(data_dtype)
            _stamp_type_and_shape(res, out_shape)
            _ensure_value_metadata(ctx, res)
            ctx.bind_value_for_var(var, res)
