# jax2onnx/plugins/jax/lax/scatter.py

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any

import numpy as np
import jax.numpy as jnp  # Keep for potential use in test cases or future needs
from jax import lax, core
from jax.lax import (
    ScatterDimensionNumbers,
    GatherScatterMode,
)
import onnx
from onnx import numpy_helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from .scatter_converters import convert_lax_scatter

import logging

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scatter")

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


def _count_reshape_to_shape_in_model(
    model: onnx.ModelProto, target_shape: Sequence[int]
) -> int:
    """Count Reshape nodes whose 2nd input is a constant equal to target_shape."""
    # Gather constant tensors by name from initializers and Constant nodes
    const_map = {}
    for init in model.graph.initializer:
        const_map[init.name] = numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if (
                    attr.name == "value"
                    and getattr(attr, "t", None) is not None
                    and attr.t.name
                ):
                    const_map[attr.t.name] = numpy_helper.to_array(attr.t)

    def as_tuple(a):
        try:
            return tuple(int(x) for x in a.tolist())
        except Exception:
            return None

    count = 0
    tgt = tuple(target_shape)
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shp_name = node.input[1]
        if shp_name in const_map and as_tuple(const_map[shp_name]) == tgt:
            count += 1
    return count


@register_primitive(
    jaxpr_primitive=lax.scatter_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter.html",
    onnx=[
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="scatter",
    testcases=[
        {
            "testcase": "scatter_set_axis0",
            "callable": lambda x: x.at[0].set(-100.0),
            "input_shapes": [(1, 1)],
        },
        {
            "testcase": "scatter_set_middle",
            "callable": lambda x: x.at[1].set(42.0),
            "input_shapes": [(3,)],
        },
        {
            "testcase": "scatter_correct_axis_determination",
            "callable": lambda op, idx, upd_scalar_batch: lax.scatter(
                op,
                idx,
                jnp.reshape(upd_scalar_batch, idx.shape[:-1]),
                ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        {
            "testcase": "scatter_updates_slice_needed_axis0",
            "callable": lambda op, idx, upd_scalar_batch: lax.scatter(
                op,
                idx,
                jnp.reshape(upd_scalar_batch, idx.shape[:-1]),
                ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        {
            "testcase": "scatter_from_user_warning_shapes_valid_jax",
            "callable": lambda operand, indices, updates_sliced_scalar_batch: lax.scatter(
                operand,
                indices,
                jnp.reshape(updates_sliced_scalar_batch, indices.shape[:-1]),
                ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        {
            "testcase": "scatter_user_error_scenario_precise",
            "callable": lambda operand, indices, updates: lax.scatter(
                operand,
                indices,
                updates,
                ScatterDimensionNumbers(
                    update_window_dims=(1, 2, 3),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                    operand_batching_dims=(),
                    scatter_indices_batching_dims=(),
                ),
                mode=GatherScatterMode.FILL_OR_DROP,
                unique_indices=False,
                indices_are_sorted=False,
            ),
            "input_shapes": [(5, 201, 1, 1), (2, 1), (2, 201, 1, 1)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        # ────────────────────────────────────────────────────────────────
        #  Window‑scatter (moved from examples/lax/scatter_window.py)
        # ────────────────────────────────────────────────────────────────
        {
            "testcase": "scatter_window_update_f64",
            # identical to the old `scatter_window_function`
            "callable": lambda operand, indices, updates: lax.scatter(
                operand=operand,
                scatter_indices=indices,
                updates=updates,
                dimension_numbers=ScatterDimensionNumbers(
                    update_window_dims=(1, 2, 3, 4),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1, 2),
                ),
                indices_are_sorted=True,
                unique_indices=True,
                mode=GatherScatterMode.FILL_OR_DROP,
            ),
            "input_values": [
                np.zeros((5, 266, 266, 1), dtype=np.float64),
                np.array([[10, 10]], dtype=np.int32),
                np.ones((1, 5, 256, 256, 1), dtype=np.float64),
            ],
            # keep the original flag so we only run the double‑precision variant
            "run_only_f64_variant": True,
        },
        {
            "testcase": "scatter_window_update_depth3_shapes_ok",
            "callable": lambda operand, indices, updates: lax.scatter(
                operand=operand,
                scatter_indices=indices,
                updates=updates,
                dimension_numbers=ScatterDimensionNumbers(
                    update_window_dims=(1, 2, 3, 4),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1, 2),
                ),
                indices_are_sorted=True,
                unique_indices=True,
                mode=GatherScatterMode.FILL_OR_DROP,
            ),
            "input_values": [
                np.zeros((5, 266, 266, 1), dtype=np.float64),
                np.array([[10, 10]], dtype=np.int32),
                np.ones((1, 5, 256, 256, 1), dtype=np.float64),
            ],
            "run_only_f64_variant": True,
            # ONNX graph sanity checks: exactly one [-1,3] and one [-1,1] Reshape.
            "post_check_onnx_graph": lambda model: (
                (
                    lambda n_idx, n_upd: (
                        True
                        if (n_idx == 1 and n_upd == 1)
                        else (_ for _ in ()).throw(
                            AssertionError(
                                f"Expected exactly one Reshape to [-1,3] and [-1,1]; got n_idx={n_idx}, n_upd={n_upd}"
                            )
                        )
                    )
                )(
                    _count_reshape_to_shape_in_model(model, [-1, 3]),
                    _count_reshape_to_shape_in_model(model, [-1, 1]),
                )
            ),
        },
        # ────────────────────────────────────────────────────────────────
        #  Static-slice assignment: x.at[:, 5:261, 5:261, :].set(updates)
        #  Repro for user report where updates is (B, H, W, C) without a
        #  leading "N" batch axis for indices rows.
        # ────────────────────────────────────────────────────────────────
        {
            "testcase": "scatter_static_slice_set_f64",
            "callable": lambda operand, indices, updates: operand.at[
                :, 5:261, 5:261, :
            ].set(updates),
            "input_values": [
                np.zeros((5, 266, 266, 1), dtype=np.float64),  # operand
                np.array(
                    [[5, 5]], dtype=np.int32
                ),  # indices (not used by JAX here, but kept for signature parity)
                np.ones((5, 256, 256, 1), dtype=np.float64),  # updates
            ],
            "run_only_f64_variant": True,
        },
        # ────────────────────────────────────────────────────────────────
        # REGRESSION: fp64 ScatterND-helper dtype mismatch
        #
        #  • operand lives in **float64**
        #  • generalised "depth-2 indices" path is chosen
        #  • old helper records GatherND output as float32
        #    → onnx.check_model fails with
        #      "Type (tensor(float)) of output arg (…) does not match
        #       expected type (tensor(double))"
        # ────────────────────────────────────────────────────────────────
        {
            "testcase": "scatter_depth2_fp64_type_mismatch",
            "callable": (
                # tiny tensor just large enough to trigger depth-2 logic
                lambda: lax.scatter(
                    jnp.zeros((2, 3, 4, 5), dtype=jnp.float64),  # operand (double)
                    jnp.array([[1]], dtype=jnp.int32),  # indices  shape (1, depth=1)
                    jnp.ones(
                        (1, 2, 3, 4, 5), dtype=jnp.float64
                    ),  # updates  shape = indices[:-1] + window
                    dimension_numbers=lax.ScatterDimensionNumbers(
                        update_window_dims=(
                            1,
                            2,
                            3,
                            4,
                        ),  # window-dims = all operand dims
                        inserted_window_dims=(),  # ⇒ generalised depth-2 route
                        scatter_dims_to_operand_dims=(
                            1,
                        ),  # scatter along 2-nd operand dim
                    ),
                )
            ),
            "input_shapes": [],  # no runtime inputs – everything is literal
            "run_only_f64_variant": True,  # exporter stays in float64
        },
        {
            "testcase": "scatter_simple_2d_window_out_of_bounds",
            "callable": lambda: lax.scatter(
                jnp.zeros((5, 5), dtype=jnp.float32),
                jnp.array([[4]], dtype=jnp.int32),  # Start update at index 4 of axis 1
                jnp.ones(
                    (1, 5, 2), dtype=jnp.float32
                ),  # Window size is 2 along that axis
                dimension_numbers=lax.ScatterDimensionNumbers(
                    update_window_dims=(
                        1,
                        2,
                    ),  # In updates, axes 1 and 2 are the window
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1,),  # Map index to operand axis 1
                ),
            ),
            "input_shapes": [],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "scatter_clip_2d_window_at_edge",
            "callable": lambda: lax.scatter(
                jnp.arange(5).reshape(1, 5).astype(jnp.float32),
                jnp.array([[4]], dtype=jnp.int32),  # start at index 4 of axis 1
                jnp.array(
                    [[[9.0, 8.0]]], dtype=jnp.float32
                ),  # <- NO transpose → (1,1,2)
                dimension_numbers=lax.ScatterDimensionNumbers(
                    update_window_dims=(1, 2),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1,),
                ),
                mode=GatherScatterMode.CLIP,
            ),
            "input_shapes": [],
            "run_only_f32_variant": True,
        },
        # ────────────────────────────────────────────────────────────────
        # REGRESSION ♦ depth-2 ScatterND helper keeps f32 although
        #                operand is f64  →  onnx.check_model type error
        #   • operand:     float64   (should drive GatherND dtype)
        #   • updates:     float32   (leaks into helper -> GatherND recorded fp32)
        #   • indices:     depth-2   (forces "generalised depth-2 indices" path)
        # ────────────────────────────────────────────────────────────────
        {
            "testcase": "scatter_depth2_mixed_dtypes_fp_mismatch_f64",
            "callable": (
                lambda: lax.scatter(
                    # ① operand – double precision
                    jnp.zeros((2, 3, 4, 5), dtype=jnp.float64),
                    # ② indices – shape (N, 2) so depth-2 logic is chosen
                    jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),  # (N, 2)
                    # ③ updates – intentionally float32
                    # shape = (N, 1, 1, 4, 5):
                    #   – leading N  (=indices.shape[0])
                    #   – two size-1 dims (inserted_window_dims)
                    #   – the operand window (4,5)
                    jnp.ones((2, 4, 5), dtype=jnp.float64),  # rank 3  ✅
                    dimension_numbers=lax.ScatterDimensionNumbers(
                        update_window_dims=(1, 2),  # positions in the updates tensor
                        inserted_window_dims=(
                            0,
                            1,
                        ),  # operand dims that are size-1 and absent
                        scatter_dims_to_operand_dims=(0, 1),  # two scatter dims
                    ),
                )
            ),
            "input_shapes": [],  # no runtime inputs (all literals)
            "run_only_f64_variant": True,  # test must run with default fp32 exporter
        },
        {
            "testcase": "scatter_depth2_mixed_dtypes_fp_mismatch",
            "callable": (
                lambda: lax.scatter(
                    # ① operand – double precision
                    jnp.zeros((2, 3, 4, 5), dtype=jnp.float64),
                    # ② indices – shape (N, 2) so depth-2 logic is chosen
                    jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),  # (N, 2)
                    # ③ updates – intentionally float32
                    # shape = (N, 1, 1, 4, 5):
                    #   – leading N  (=indices.shape[0])
                    #   – two size-1 dims (inserted_window_dims)
                    #   – the operand window (4,5)
                    jnp.ones((2, 4, 5), dtype=jnp.float32),  # rank 3  ✅
                    dimension_numbers=lax.ScatterDimensionNumbers(
                        update_window_dims=(1, 2),  # positions in the updates tensor
                        inserted_window_dims=(
                            0,
                            1,
                        ),  # operand dims that are size-1 and absent
                        scatter_dims_to_operand_dims=(0, 1),  # two scatter dims
                    ),
                )
            ),
            "input_shapes": [],  # no runtime inputs (all literals)
            "run_only_f32_variant": True,  # test must run with default fp32 exporter
        },
    ],
)
class ScatterPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(
        operand: core.ShapedArray,
        indices: core.ShapedArray,
        updates: core.ShapedArray,
        update_jaxpr,
        *,
        dimension_numbers: ScatterDimensionNumbers,
        indices_are_sorted: bool,
        unique_indices: bool,
        mode: GatherScatterMode | str | None,
        **params,
    ):
        return core.ShapedArray(operand.shape, operand.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ):
        operand_v, indices_v, updates_v = node_inputs
        out_v = node_outputs[0]
        out_name = s.get_name(out_v)

        # Delegate to the shared converter (Style B)
        logger.info(
            f"Preparing inputs for ONNX ScatterND with {params['dimension_numbers']}"
        )

        class _Eqn:
            # mypy: declare the attribute so assignments are type-checked
            params: dict[str, Any]

        _e = _Eqn()
        _e.params = params
        convert_lax_scatter(
            s,
            _e,
            (operand_v, indices_v, updates_v),
            (out_v,),
        )
        logger.debug(
            "[ScatterPlugin] Emitted ScatterND(reduction='none') → %s", out_name
        )
        logger.debug(
            "[ScatterPlugin] Emitted ScatterND(reduction='none') → %s", out_name
        )


def _validate_updates_window_fits_operand(op_shape, upd_shape, dnums, *, mode=None):
    """
    Validate that each window dimension of `updates` does not exceed the
    corresponding dimension of `operand`.

    IMPORTANT: JAX has 2 legal conventions in the wild:
      • update_window_dims = all operand dims \ {inserted}          (includes scatter dims)
      • update_window_dims = all operand dims \ {inserted ∪ scatter}  (excludes scatter dims)
    We must pick whichever matches |update_window_dims|.
    """
    op_rank = len(op_shape)
    inserted = set(getattr(dnums, "inserted_window_dims", ()))
    scatter = tuple(getattr(dnums, "scatter_dims_to_operand_dims", ()))

    # Candidate operand-window sets
    all_window = [
        d for d in range(op_rank) if d not in inserted
    ]  # includes scatter dims
    excl_scatter_window = [
        d for d in all_window if d not in scatter
    ]  # excludes scatter dims

    # Choose the convention that actually matches this dnums instance.
    uwd = tuple(getattr(dnums, "update_window_dims", ()))
    if len(uwd) == len(all_window):
        window_operand_dims = all_window
    elif len(uwd) == len(excl_scatter_window):
        window_operand_dims = excl_scatter_window
    else:
        # Inconsistent `dnums` — let the converter’s general shape logic deal with it.
        return

    # Pair by ascending update axis order, as JAX expects.
    for upd_axis, op_axis in zip(sorted(uwd), window_operand_dims):
        upd_bound = int(upd_shape[upd_axis])
        op_bound = int(op_shape[op_axis])
        if upd_bound > op_bound:
            raise TypeError(
                "Bounds of the window dimensions of updates must not exceed the bounds of the "
                f"corresponding dimensions of operand. For dimension {op_axis}, updates bound is "
                f"{upd_bound}, operand bound is {op_bound}."
            )
