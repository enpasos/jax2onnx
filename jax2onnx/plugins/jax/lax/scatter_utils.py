# jax2onnx/plugins/jax/lax/scatter_utils.py

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Optional,
    Any,
    Tuple,
    Sequence,
)
import numpy as np
from jax import (
    ShapeDtypeStruct,
)  # Ensure jax.ShapeDtypeStruct is directly imported
from jax.lax import ScatterDimensionNumbers
from jax.lax import GatherScatterMode
from onnx import helper, TensorProto

import logging

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scatter_utils")


SCATTER_UTILS_VERSION = "DEBUG-V20250816-d8-d2-narrow-degenerate-pick"


def _ensure_np_dtype(dtype_like: Any) -> np.dtype:
    if isinstance(dtype_like, np.dtype):
        return dtype_like
    try:
        return np.dtype(dtype_like)
    except TypeError as e:
        logger.error(
            f"Could not convert '{dtype_like}' (type: {type(dtype_like)}) to np.dtype."
        )
        raise e


def _manually_ensure_shape_env_entry(
    s: "Jaxpr2OnnxConverter",
    tensor_name: str,
    tensor_shape: Tuple[Any, ...],
    np_dtype_for_sds_and_builder: Any,
    context: str = "",
):
    try:
        final_np_dtype = _ensure_np_dtype(np_dtype_for_sds_and_builder)

        valid_shape_elements = []
        for dim_val in tensor_shape:
            if isinstance(dim_val, (int, np.integer)):
                valid_shape_elements.append(int(dim_val))
            elif hasattr(s, "_dim_to_symbol_safe") and callable(s._dim_to_symbol_safe):
                try:
                    valid_shape_elements.append(s._dim_to_symbol_safe(dim_val))
                except Exception:
                    logger.warning(
                        f"Failed to use _dim_to_symbol_safe for dim '{dim_val}' in context '{context}'. Using as is."
                    )
                    valid_shape_elements.append(dim_val)
            else:
                valid_shape_elements.append(dim_val)

        shape_tuple_for_sds = tuple(valid_shape_elements)

        sds_to_store = ShapeDtypeStruct(shape_tuple_for_sds, final_np_dtype)
        s.shape_env[tensor_name] = sds_to_store
        s.add_shape_info(tensor_name, shape_tuple_for_sds, final_np_dtype)

        logger.debug(
            f"[_prepare_scatter_inputs {context}] MANUALLY ensured s.shape_env for '{tensor_name}' to {sds_to_store}. "
            f"Check after direct set: {tensor_name in s.shape_env}. Value: {s.shape_env.get(tensor_name)}"
        )
        if tensor_name not in s.shape_env:
            logger.error(
                f"[_prepare_scatter_inputs {context}] FAILED to find '{tensor_name}' in s.shape_env EVEN AFTER DIRECT ASSIGNMENT. Keys: {list(s.shape_env.keys())}"
            )

    except Exception as e_manual_ensure:
        logger.error(
            f"[_prepare_scatter_inputs {context}] Error during _manually_ensure_shape_env_entry for '{tensor_name}': {e_manual_ensure}",
            exc_info=True,
        )


def _is_dim_symbolic(dim_val: Any, s: "Jaxpr2OnnxConverter") -> bool:
    if isinstance(dim_val, int):
        return False
    if isinstance(dim_val, np.integer):
        return False
    if hasattr(s, "is_symbolic_dim") and callable(s.is_symbolic_dim):
        try:
            return s.is_symbolic_dim(dim_val)
        except Exception:
            pass
    return True


def _are_dims_equal(dim1: Any, dim2: Any, s: "Jaxpr2OnnxConverter") -> bool:
    # This is the simplified version that passed pre-commit checks
    is_dim1_sym = _is_dim_symbolic(dim1, s)
    is_dim2_sym = _is_dim_symbolic(dim2, s)

    if not is_dim1_sym and not is_dim2_sym:
        return int(dim1) == int(dim2)

    if is_dim1_sym != is_dim2_sym:  # One symbolic, one concrete
        return False

    # Both are symbolic (or considered symbolic by _is_dim_symbolic fallback)
    return dim1 is dim2  # Fallback to object identity for symbolic dimensions


def _are_shapes_equal(
    shape1: Tuple[Any, ...], shape2: Tuple[Any, ...], s: "Jaxpr2OnnxConverter"
) -> bool:
    if len(shape1) != len(shape2):
        return False
    for d1, d2 in zip(shape1, shape2):
        if not _are_dims_equal(d1, d2, s):
            return False
    return True


def _make_shape_concrete_for_prod(
    shp: Tuple[Any, ...], s: "Jaxpr2OnnxConverter", context_msg: str = "shape"
) -> Tuple[int, ...]:
    concrete_shape = []
    for i, dim_val in enumerate(shp):
        if isinstance(dim_val, int):
            concrete_shape.append(dim_val)
        elif isinstance(dim_val, np.integer):
            concrete_shape.append(int(dim_val))
        else:
            val_to_append = None
            if hasattr(s, "get_concrete_value_from_symbolic_dim") and callable(
                s.get_concrete_value_from_symbolic_dim
            ):
                val_to_append = s.get_concrete_value_from_symbolic_dim(dim_val)

            if val_to_append is not None:
                concrete_shape.append(int(val_to_append))
            else:
                if (
                    type(dim_val).__name__ == "Literal"
                    and hasattr(dim_val, "val")
                    and isinstance(dim_val.val, int)
                ):
                    concrete_shape.append(dim_val.val)
                else:
                    raise ValueError(
                        f"Cannot make {context_msg} concrete for np.prod: {shp}. Symbolic dim '{dim_val}' (type: {type(dim_val)}) at index {i} could not be resolved by available converter methods."
                    )
    return tuple(concrete_shape)


def compute_expected_updates_shape(
    dnums: ScatterDimensionNumbers,
    operand_shape: Sequence[int],
    indices_shape: Sequence[int],
) -> Tuple[int, ...]:
    """
    Return the exact shape `updates` must have for a JAX scatter-style op,
    per the official spec:

        updates.shape == indices.shape[:-1]  (batch part, order preserved)
                           + operand.shape[window_dims]  (at positions given
                             by `update_window_dims`)

    The `update_window_dims` values are **positions in the updates tensor**,
    *not* operand-dimension IDs.  We therefore build the full result rank
    first, place window-dim sizes at those positions, and fill the remaining
    slots with the leading batch dims coming from `indices`.
    """
    batch_shape: Tuple[int, ...] = tuple(indices_shape[:-1])

    # Which operand dims participate in the slice (window)?
    inserted = set(dnums.inserted_window_dims)
    window_operand_dims = [d for d in range(len(operand_shape)) if d not in inserted]

    if len(window_operand_dims) != len(dnums.update_window_dims):
        raise ValueError(
            "Inconsistent scatter dnums: |window_operand_dims| "
            f"{len(window_operand_dims)} != |update_window_dims| "
            f"{len(dnums.update_window_dims)}"
        )

    window_sizes = [operand_shape[d] for d in window_operand_dims]

    updates_rank = len(batch_shape) + len(window_sizes)
    result: list = [None] * updates_rank

    # 1️⃣  place window dims at the positions given by update_window_dims
    for pos_in_updates, win_size in zip(dnums.update_window_dims, window_sizes):
        result[pos_in_updates] = win_size

    # 2️⃣  fill the remaining slots (in order) with batch dims
    batch_iter = iter(batch_shape)
    for i in range(updates_rank):
        if result[i] is None:
            result[i] = next(batch_iter)

    return tuple(result)


def _map_operand_axis_to_updates_pos(
    dnums: ScatterDimensionNumbers, operand_rank: int, operand_axis: int
) -> Optional[int]:
    """Given an operand axis, return which *updates* axis position contains
    that window dim, per JAX ScatterDimensionNumbers."""
    inserted = set(dnums.inserted_window_dims)
    window_operand_dims = [d for d in range(operand_rank) if d not in inserted]
    if operand_axis not in window_operand_dims:
        return None
    i = window_operand_dims.index(operand_axis)
    if i >= len(dnums.update_window_dims):
        return None
    return dnums.update_window_dims[i]


def _prepare_scatter_inputs_for_onnx(
    s: "Jaxpr2OnnxConverter",
    operand_v: Any,
    indices_v: Any,
    updates_v: Any,
    dimension_numbers: ScatterDimensionNumbers,
    scatter_mode: Optional[Any] = None,  # Add scatter_mode parameter
    reduction: str = "add",  # Add reduction parameter
) -> Tuple[str, str, str]:
    logger.debug(
        f"Running _prepare_scatter_inputs_for_onnx - Version: {SCATTER_UTILS_VERSION}"
    )

    def to_symbolic_tuple(
        jax_shape: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        if hasattr(s, "_dim_to_symbol_safe") and callable(s._dim_to_symbol_safe):
            return tuple(s._dim_to_symbol_safe(d) for d in jax_shape)
        return tuple(jax_shape)

    final_operand_name = s.get_name(operand_v)
    operand_aval = operand_v.aval
    operand_shape_symbolic = to_symbolic_tuple(operand_aval.shape)
    operand_dtype_np = _ensure_np_dtype(operand_aval.dtype)
    _manually_ensure_shape_env_entry(
        s, final_operand_name, operand_shape_symbolic, operand_dtype_np, "Operand"
    )

    indices_aval = indices_v.aval
    jax_indices_shape_symbolic = to_symbolic_tuple(indices_aval.shape)
    jax_indices_dtype_np = _ensure_np_dtype(indices_aval.dtype)
    original_jax_indices_name_in_onnx = s.get_name(indices_v)
    current_indices_name = original_jax_indices_name_in_onnx
    current_indices_shape_symbolic = jax_indices_shape_symbolic
    _manually_ensure_shape_env_entry(
        s,
        current_indices_name,
        current_indices_shape_symbolic,
        jax_indices_dtype_np,
        "OriginalIndices",
    )

    final_indices_dtype_np = np.int64
    if jax_indices_dtype_np != final_indices_dtype_np:
        base_cast_indices_out_name = current_indices_name + "_int64"
        cast_indices_out_name = s.get_unique_name(base_cast_indices_out_name)
        s.add_node(
            helper.make_node(
                "Cast",
                inputs=[current_indices_name],
                outputs=[cast_indices_out_name],
                to=int(TensorProto.INT64),
            )
        )
        _manually_ensure_shape_env_entry(
            s,
            cast_indices_out_name,
            current_indices_shape_symbolic,
            final_indices_dtype_np,
            "CastIndices",
        )
        current_indices_name = cast_indices_out_name

    index_depth_k = len(dimension_numbers.scatter_dims_to_operand_dims)

    target_indices_shape_symbolic: Tuple[Any, ...]
    if not current_indices_shape_symbolic:
        target_indices_shape_symbolic = (1, index_depth_k if index_depth_k > 0 else 0)
    elif (
        len(current_indices_shape_symbolic) == 1
        and index_depth_k > 0
        and _are_dims_equal(current_indices_shape_symbolic[0], index_depth_k, s)
    ):
        target_indices_shape_symbolic = (1, index_depth_k)
    elif (
        index_depth_k > 0
        and len(current_indices_shape_symbolic) > 0
        and _are_dims_equal(current_indices_shape_symbolic[-1], index_depth_k, s)
    ):
        batch_dims_indices = current_indices_shape_symbolic[:-1]
        if not batch_dims_indices:
            target_indices_shape_symbolic = (1, index_depth_k)
        else:
            try:
                num_updates_prod = np.prod(
                    _make_shape_concrete_for_prod(
                        batch_dims_indices, s, "indices_batch_prod_gen"
                    )
                ).astype(int)
                target_indices_shape_symbolic = (num_updates_prod, index_depth_k)
            except ValueError:
                target_indices_shape_symbolic = (-1, index_depth_k)
    elif index_depth_k == 0 and len(current_indices_shape_symbolic) == 1:
        target_indices_shape_symbolic = (current_indices_shape_symbolic[0], 0)
    else:
        if len(current_indices_shape_symbolic) == 2 and _are_dims_equal(
            current_indices_shape_symbolic[1], index_depth_k, s
        ):
            target_indices_shape_symbolic = current_indices_shape_symbolic
        else:
            logger.warning(
                f"Complex JAX indices_shape {current_indices_shape_symbolic} for K={index_depth_k}. Attempting generic reshape to (N,K)."
            )
            common_N_val_gen = -1
            if current_indices_shape_symbolic:
                try:
                    if len(current_indices_shape_symbolic) > 1 and _are_dims_equal(
                        current_indices_shape_symbolic[-1], index_depth_k, s
                    ):
                        common_N_val_gen = np.prod(
                            _make_shape_concrete_for_prod(
                                current_indices_shape_symbolic[:-1],
                                s,
                                "commonN_prod_gen",
                            )
                        ).astype(int)
                    elif (
                        len(current_indices_shape_symbolic) == 1 and index_depth_k == 0
                    ):
                        common_N_val_gen = _make_shape_concrete_for_prod(
                            (current_indices_shape_symbolic[0],), s, "commonN_K0_gen"
                        )[0]
                except ValueError:
                    common_N_val_gen = -1
            elif not current_indices_shape_symbolic and index_depth_k >= 0:
                common_N_val_gen = 1
            if index_depth_k >= 0:
                target_indices_shape_symbolic = (common_N_val_gen, index_depth_k)
            else:
                raise ValueError(
                    f"Invalid index_depth_k for general path: {index_depth_k}"
                )

    final_indices_name_to_return: str
    if not _are_shapes_equal(
        current_indices_shape_symbolic, target_indices_shape_symbolic, s
    ):
        reshaped_indices_name = s.get_unique_name(
            f"{current_indices_name}_reshaped_idx_auto"
        )
        concrete_target_for_op_list = []
        has_minus_one_already = False
        for i_dim, dim_sym_val in enumerate(target_indices_shape_symbolic):
            if isinstance(dim_sym_val, int):
                concrete_target_for_op_list.append(dim_sym_val)
            else:
                if not has_minus_one_already:
                    concrete_target_for_op_list.append(-1)
                    has_minus_one_already = True
                else:
                    try:
                        concrete_target_for_op_list.append(
                            int(
                                _make_shape_concrete_for_prod(
                                    (dim_sym_val,),
                                    s,
                                    f"reshape_target_indices_dim_{i_dim}",
                                )[0]
                            )
                        )
                    except ValueError as ve_reshape:
                        raise ValueError(
                            f"Cannot create Reshape target for indices {target_indices_shape_symbolic} with multiple non-concrete dims: {ve_reshape}"
                        ) from ve_reshape
        s.add_node(
            helper.make_node(
                "Reshape",
                [
                    current_indices_name,
                    s.get_constant_name(
                        np.array(concrete_target_for_op_list, dtype=np.int64)
                    ),
                ],
                [reshaped_indices_name],
            )
        )
        _manually_ensure_shape_env_entry(
            s,
            reshaped_indices_name,
            target_indices_shape_symbolic,
            final_indices_dtype_np,
            "AutoReshapeIndices",
        )
        final_indices_name_to_return = reshaped_indices_name
    else:
        final_indices_name_to_return = current_indices_name
        _manually_ensure_shape_env_entry(
            s,
            final_indices_name_to_return,
            target_indices_shape_symbolic,
            final_indices_dtype_np,
            "NoOpIndices",
        )

    original_updates_name_val = s.get_name(updates_v)
    original_updates_aval = updates_v.aval
    original_updates_shape_symbolic = to_symbolic_tuple(original_updates_aval.shape)
    original_updates_dtype_np = _ensure_np_dtype(original_updates_aval.dtype)
    _manually_ensure_shape_env_entry(
        s,
        original_updates_name_val,
        original_updates_shape_symbolic,
        original_updates_dtype_np,
        "OriginalUpdates",
    )

    _final_updates_name_val_to_return = original_updates_name_val

    # Ensure updates datatype matches operand datatype
    if operand_dtype_np != original_updates_dtype_np:
        logger.debug(
            f"Casting updates from {original_updates_dtype_np} to {operand_dtype_np} to match operand dtype"
        )
        cast_updates_name = s.get_unique_name(
            f"{original_updates_name_val}_cast_to_{operand_dtype_np.__name__}"
        )
        s.add_node(
            helper.make_node(
                "Cast",
                [original_updates_name_val],
                [cast_updates_name],
                to=int(s.builder._numpy_dtype_to_onnx(operand_dtype_np)),
                name=s.get_unique_name("scatter_cast_updates"),
            )
        )
        _manually_ensure_shape_env_entry(
            s,
            cast_updates_name,
            original_updates_shape_symbolic,
            operand_dtype_np,
            "CastUpdates",
        )
        _final_updates_name_val_to_return = cast_updates_name
        # Update the dtype for downstream operations
        original_updates_dtype_np = operand_dtype_np

    # --- Calculate expected ONNX updates shape based on the *final processed* indices for the general path ---
    # `processed_indices_shape_for_default_path` is `target_indices_shape_symbolic` (the (N,K) shape of final_indices_name_to_return)
    # NOTE: Keep this as a variadic Tuple so later specialized paths that use
    # shapes like (B, L, 2) (i.e., batch dims + K) remain type-correct.
    # Otherwise static checkers may infer a fixed-length 2-tuple and reject
    # 3-length tuples later.
    processed_indices_shape_for_default_path: Tuple[Any, ...] = (
        target_indices_shape_symbolic
    )

    # ------------------------------------------------------------------
    #  Expected shape for the ONNX `updates` input  – **spec‑exact**
    # ------------------------------------------------------------------
    current_expected_onnx_updates_shape = compute_expected_updates_shape(
        dimension_numbers,  # ScatterDimensionNumbers
        operand_shape_symbolic,  # operand.shape
        processed_indices_shape_for_default_path,  # indices.shape
    )

    # (No second assignment of `current_expected_onnx_updates_shape` below –
    #  it is already correct and kept consistent throughout.)

    # --- New logic for batched window scatter ---
    sdod = dimension_numbers.scatter_dims_to_operand_dims
    uwd = dimension_numbers.update_window_dims
    iwd = dimension_numbers.inserted_window_dims
    obd = dimension_numbers.operand_batching_dims
    op_rank = len(operand_shape_symbolic)
    upd_rank = len(original_updates_shape_symbolic)

    if (
        len(sdod) == 1
        and len(uwd) == upd_rank
        and op_rank == upd_rank
        and not obd
        and not iwd
        and (
            not jax_indices_shape_symbolic
            or _are_shapes_equal(jax_indices_shape_symbolic, (1,), s)
        )
    ):
        scatter_target_op_axis = sdod[0]
        if scatter_target_op_axis < op_rank:
            shapes_match_for_depth2_pattern = True
            if shapes_match_for_depth2_pattern and op_rank > scatter_target_op_axis + 1:
                op_trailing_shape = operand_shape_symbolic[scatter_target_op_axis + 1 :]
                if scatter_target_op_axis < len(original_updates_shape_symbolic):
                    upd_trailing_shape = original_updates_shape_symbolic[
                        scatter_target_op_axis + 1 :
                    ]
                    if not _are_shapes_equal(op_trailing_shape, upd_trailing_shape, s):
                        shapes_match_for_depth2_pattern = False
                else:
                    shapes_match_for_depth2_pattern = False
            elif scatter_target_op_axis == 0:
                if op_rank > 1:
                    if not _are_shapes_equal(
                        operand_shape_symbolic[1:],
                        original_updates_shape_symbolic[1:],
                        s,
                    ):
                        shapes_match_for_depth2_pattern = False
                elif op_rank != 1:
                    shapes_match_for_depth2_pattern = False

            if shapes_match_for_depth2_pattern and op_rank > 0:
                if scatter_target_op_axis < len(original_updates_shape_symbolic):
                    pass
                else:
                    logger.warning(
                        f"Depth-2: scatter_target_op_axis {scatter_target_op_axis} out of bounds for updates_shape {original_updates_shape_symbolic}"
                    )

    # Depth-2 rewrite also for K=1 with a leading N=1 in updates and indices=(1,1).
    if (
        len(sdod) == 1
        and not obd
        and not iwd
        # updates rank may be op_rank (no N) or op_rank+1 with a leading N=1
        and (
            upd_rank == op_rank
            or (
                upd_rank == op_rank + 1
                and _are_dims_equal(original_updates_shape_symbolic[0], 1, s)
            )
        )
        # indices can be (), (1,) or (1,1) for K=1
        and (
            not jax_indices_shape_symbolic
            or _are_shapes_equal(jax_indices_shape_symbolic, (1,), s)
            or _are_shapes_equal(jax_indices_shape_symbolic, (1, 1), s)
        )
    ):
        logger.info(
            "Applying generalized 'depth-2 indices' strategy for batched window scatter."
        )
        scatter_op_axis_idx = dimension_numbers.scatter_dims_to_operand_dims[0]
        _make_shape_concrete_for_prod(operand_shape_symbolic, s, "d2_op_shape")

        # Map operand axis -> updates axis position to read the *correct* L.
        upd_pos_for_scatter_axis = _map_operand_axis_to_updates_pos(
            dimension_numbers, op_rank, scatter_op_axis_idx
        )
        if upd_pos_for_scatter_axis is None:
            logger.warning(
                "Depth-2: could not map operand axis to updates position; "
                "falling back to default path."
            )
        else:
            # B is operand axis 0; candidate L is the updates axis that corresponds
            # to the scatter op axis.
            B_sym = operand_shape_symbolic[0]
            B_val = _make_shape_concrete_for_prod((B_sym,), s, "d2_B")[0]

        # Build scalar start from indices ((1,), (1,1), … → squeeze to scalar)
        col_start_scalar_name = s.get_unique_name(f"{current_indices_name}_scalar_d2")
        s.add_node(
            helper.make_node(
                "Squeeze",
                [
                    current_indices_name,
                    s.get_constant_name(
                        np.array(
                            [
                                ax
                                for ax, dim in enumerate(current_indices_shape_symbolic)
                                if _are_dims_equal(dim, 1, s)
                            ]
                            or [0],
                            dtype=np.int64,
                        )
                    ),
                ],
                [col_start_scalar_name],
            )
        )
        _manually_ensure_shape_env_entry(
            s, col_start_scalar_name, (), final_indices_dtype_np, "ColStartScalarD2"
        )

        # ----------------------------
        # Narrow “degenerate single-point” gate
        # Only slice when ALL conditions hold:
        #  • window covers all operand dims (len(uwd) == op_rank)
        #  • we can map the scatter operand axis into updates
        #  • updates have a leading N=1 (upd_rank == op_rank + 1 and updates[0] == 1)
        #  • the mapped updates axis actually has length 1
        # ----------------------------
        def _dim_is_one(dim: Any) -> bool:
            if isinstance(dim, (int, np.integer)):
                return int(dim) == 1
            try:
                return _make_shape_concrete_for_prod((dim,), s, "d2_dim1_check")[0] == 1
            except Exception:
                return False

        degenerate_pick = (
            len(uwd) == op_rank
            and upd_pos_for_scatter_axis is not None
            and upd_rank == op_rank + 1
            and _are_dims_equal(original_updates_shape_symbolic[0], 1, s)
            and _dim_is_one(original_updates_shape_symbolic[upd_pos_for_scatter_axis])
            and (
                not jax_indices_shape_symbolic
                or _are_shapes_equal(jax_indices_shape_symbolic, (1,), s)
                or _are_shapes_equal(jax_indices_shape_symbolic, (1, 1), s)
            )
        )

        if degenerate_pick:
            # Slice updates at the start column; this removes that axis in updates.
            picked_updates_name = s.get_unique_name("updates_pick_scatter_axis_d2")
            s.add_node(
                helper.make_node(
                    "Gather",
                    [original_updates_name_val, col_start_scalar_name],
                    [picked_updates_name],
                    axis=upd_pos_for_scatter_axis,
                )
            )
            upd_shape_after_pick = tuple(
                d
                for i, d in enumerate(original_updates_shape_symbolic)
                if i != upd_pos_for_scatter_axis
            )
            _manually_ensure_shape_env_entry(
                s,
                picked_updates_name,
                upd_shape_after_pick,
                original_updates_dtype_np,
                "Depth2PickScatterAxis",
            )
            original_updates_name_val = picked_updates_name
            original_updates_shape_symbolic = upd_shape_after_pick
            # For the rest of the path we treat this as L = 1.
            L_sym, L_val = 1, 1
        else:
            # Use the genuine L from updates at the mapped axis.
            if upd_pos_for_scatter_axis is None:
                # Shouldn't happen because we guard above, but keep safe fallback.
                logger.warning("Depth-2: missing mapped updates axis; using L=1.")
                L_sym, L_val = 1, 1
            else:
                L_sym = original_updates_shape_symbolic[upd_pos_for_scatter_axis]
                L_val = _make_shape_concrete_for_prod((L_sym,), s, "d2_L")[0]

        # From here on, proceed exactly as before using B_val/L_val …
        arange_b_end_name = s.get_constant_name(np.array(B_val, dtype=np.int64))
        arange_b_name = s.get_unique_name("arange_b_d2")
        s.add_node(
            helper.make_node(
                "Range",
                [
                    s.get_constant_name(np.array(0, dtype=np.int64)),
                    arange_b_end_name,
                    s.get_constant_name(np.array(1, dtype=np.int64)),
                ],
                [arange_b_name],
            )
        )
        _manually_ensure_shape_env_entry(
            s, arange_b_name, (B_val,), np.int64, "ArangeBD2"
        )
        unsqueeze_b_name = s.get_unique_name("unsqueeze_b_d2")
        s.add_node(
            helper.make_node(
                "Unsqueeze",
                [arange_b_name, s.get_constant_name(np.array([1], dtype=np.int64))],
                [unsqueeze_b_name],
            )
        )
        _manually_ensure_shape_env_entry(
            s, unsqueeze_b_name, (B_val, 1), np.int64, "UnsqueezeBD2"
        )
        batch_indices_intermediate_name = s.get_unique_name("batch_indices_BL_d2")
        s.add_node(
            helper.make_node(
                "Expand",
                [
                    unsqueeze_b_name,
                    s.get_constant_name(np.array([B_val, L_val], dtype=np.int64)),
                ],
                [batch_indices_intermediate_name],
            )
        )
        _manually_ensure_shape_env_entry(
            s,
            batch_indices_intermediate_name,
            (B_val, L_val),
            np.int64,
            "BatchIndicesBLD2",
        )
        arange_l_end_name = s.get_constant_name(np.array(L_val, dtype=np.int64))
        arange_l_name = s.get_unique_name("arange_l_d2")
        s.add_node(
            helper.make_node(
                "Range",
                [
                    s.get_constant_name(np.array(0, dtype=np.int64)),
                    arange_l_end_name,
                    s.get_constant_name(np.array(1, dtype=np.int64)),
                ],
                [arange_l_name],
            )
        )
        _manually_ensure_shape_env_entry(
            s, arange_l_name, (L_val,), np.int64, "ArangeLD2"
        )
        add_start_name = s.get_unique_name("add_start_col_d2")
        s.add_node(
            helper.make_node(
                "Add", [arange_l_name, col_start_scalar_name], [add_start_name]
            )
        )
        _manually_ensure_shape_env_entry(
            s, add_start_name, (L_val,), np.int64, "AddStartColD2"
        )
        unsqueeze_l_name = s.get_unique_name("unsqueeze_l_d2")
        s.add_node(
            helper.make_node(
                "Unsqueeze",
                [add_start_name, s.get_constant_name(np.array([0], dtype=np.int64))],
                [unsqueeze_l_name],
            )
        )
        _manually_ensure_shape_env_entry(
            s, unsqueeze_l_name, (1, L_val), np.int64, "UnsqueezeLD2"
        )
        col_indices_intermediate_name = s.get_unique_name("col_indices_BL_d2")
        s.add_node(
            helper.make_node(
                "Expand",
                [
                    unsqueeze_l_name,
                    s.get_constant_name(np.array([B_val, L_val], dtype=np.int64)),
                ],
                [col_indices_intermediate_name],
            )
        )
        _manually_ensure_shape_env_entry(
            s, col_indices_intermediate_name, (B_val, L_val), np.int64, "ColIndicesBLD2"
        )
        final_batch_indices_name = s.get_unique_name("final_batch_indices_d2")
        s.add_node(
            helper.make_node(
                "Unsqueeze",
                [
                    batch_indices_intermediate_name,
                    s.get_constant_name(np.array([2], dtype=np.int64)),
                ],
                [final_batch_indices_name],
            )
        )
        _manually_ensure_shape_env_entry(
            s, final_batch_indices_name, (B_val, L_val, 1), np.int64, "FinalBatchIdxD2"
        )
        final_col_indices_name = s.get_unique_name("final_col_indices_d2")
        s.add_node(
            helper.make_node(
                "Unsqueeze",
                [
                    col_indices_intermediate_name,
                    s.get_constant_name(np.array([2], dtype=np.int64)),
                ],
                [final_col_indices_name],
            )
        )
        _manually_ensure_shape_env_entry(
            s, final_col_indices_name, (B_val, L_val, 1), np.int64, "FinalColIdxD2"
        )
        indices_2d_name = s.get_unique_name("indices_2d_BL2_d2")
        s.add_node(
            helper.make_node(
                "Concat",
                [final_batch_indices_name, final_col_indices_name],
                [indices_2d_name],
                axis=2,
            )
        )

        final_indices_shape_for_depth2_strat = (B_sym, L_sym, 2)
        _manually_ensure_shape_env_entry(
            s,
            indices_2d_name,
            final_indices_shape_for_depth2_strat,
            np.int64,
            "Indices2D_Depth2Strat",
        )

        # ---- Bounds guard for ScatterND (depth-2 path) ----
        # Add bounds checking to handle out-of-bounds indices safely
        if scatter_mode == GatherScatterMode.FILL_OR_DROP or (isinstance(scatter_mode, str) and scatter_mode.upper() == "FILL_OR_DROP"):
            # Get operand shape for bounds checking
            operand_shape_tensor_name = s.get_unique_name("operand_shape_tensor")
            s.add_node(helper.make_node("Shape", [final_operand_name], [operand_shape_tensor_name]))
            _manually_ensure_shape_env_entry(s, operand_shape_tensor_name, (len(operand_shape_symbolic),), np.int64, "OperandShape")

            # Gather the first two dims (B, L) for bounds checking
            gather_axes_const = s.get_constant_name(np.array([0, 1], dtype=np.int64))
            dim_limits_name = s.get_unique_name("dim_limits")
            s.add_node(helper.make_node("Gather", [operand_shape_tensor_name, gather_axes_const], [dim_limits_name], axis=0))
            _manually_ensure_shape_env_entry(s, dim_limits_name, (2,), np.int64, "DimLimits")

            # Reshape dim_limits to (1,2) for broadcasting
            dim_limits_reshaped_name = s.get_unique_name("dim_limits_reshaped")
            reshape_target_const = s.get_constant_name(np.array([1, 2], dtype=np.int64))
            s.add_node(helper.make_node("Reshape", [dim_limits_name, reshape_target_const], [dim_limits_reshaped_name]))
            _manually_ensure_shape_env_entry(s, dim_limits_reshaped_name, (1, 2), np.int64, "DimLimitsReshaped")

            # Broadcast to match indices shape
            shape_of_indices_name = s.get_unique_name("shape_of_indices_for_bc")
            s.add_node(helper.make_node("Shape", [indices_2d_name], [shape_of_indices_name]))
            _manually_ensure_shape_env_entry(s, shape_of_indices_name, (2,), np.int64, "IndicesShapeForBC")

            dim_limits_bc_name = s.get_unique_name("dim_limits_bc")
            s.add_node(helper.make_node("Expand", [dim_limits_reshaped_name, shape_of_indices_name], [dim_limits_bc_name]))
            _manually_ensure_shape_env_entry(s, dim_limits_bc_name, (B_val, L_val, 2), np.int64, "DimLimitsBroadcast")

            # Check bounds: indices >= 0 and indices < dim_limits
            zero_tensor_name = s.get_constant_name(np.array(0, dtype=np.int64))
            
            ge0_name = s.get_unique_name("ge0")
            s.add_node(helper.make_node("GreaterOrEqual", [indices_2d_name, zero_tensor_name], [ge0_name]))
            _manually_ensure_shape_env_entry(s, ge0_name, (B_val, L_val, 2), np.bool_, "GeZero")

            lt_name = s.get_unique_name("lt_bounds")
            s.add_node(helper.make_node("Less", [indices_2d_name, dim_limits_bc_name], [lt_name]))
            _manually_ensure_shape_env_entry(s, lt_name, (B_val, L_val, 2), np.bool_, "LtBounds")

            both_ok_name = s.get_unique_name("both_bounds_ok")
            s.add_node(helper.make_node("And", [ge0_name, lt_name], [both_ok_name]))
            _manually_ensure_shape_env_entry(s, both_ok_name, (B_val, L_val, 2), np.bool_, "BothBoundsOK")

            # Reduce along last axis to get per-row validity
            row_ok_name = s.get_unique_name("row_ok")
            s.add_node(helper.make_node("ReduceAll", [both_ok_name], [row_ok_name], axes=[-1], keepdims=0))
            _manually_ensure_shape_env_entry(s, row_ok_name, (B_val, L_val), np.bool_, "RowOK")

            # Create safe indices by replacing invalid rows with zeros
            zero_2d_name = s.get_unique_name("zeros_2d")
            s.add_node(helper.make_node("Sub", [indices_2d_name, indices_2d_name], [zero_2d_name]))
            _manually_ensure_shape_env_entry(s, zero_2d_name, (B_val, L_val, 2), np.int64, "Zeros2D")

            # Broadcast row_ok to match indices shape for Where operation
            unsqueeze_axes_const = s.get_constant_name(np.array([2], dtype=np.int64))
            row_ok_unsq_name = s.get_unique_name("row_ok_unsq")
            s.add_node(helper.make_node("Unsqueeze", [row_ok_name, unsqueeze_axes_const], [row_ok_unsq_name]))
            _manually_ensure_shape_env_entry(s, row_ok_unsq_name, (B_val, L_val, 1), np.bool_, "RowOkUnsqueezed")

            safe_indices_name = s.get_unique_name("safe_indices")
            s.add_node(helper.make_node("Where", [row_ok_unsq_name, indices_2d_name, zero_2d_name], [safe_indices_name]))
            _manually_ensure_shape_env_entry(s, safe_indices_name, (B_val, L_val, 2), np.int64, "SafeIndices")

            # Update the indices name to use the safe version
            indices_2d_name = safe_indices_name

        final_indices_name_to_return = indices_2d_name
        # ONNX with K=2 expects updates of shape (B, L) + data.shape[2:]
        expected_updates_shape_d2 = (
            B_sym,
            L_sym,
        ) + tuple(operand_shape_symbolic[2:])

        # If updates come as (1, B, L, …), drop the leading singleton.
        if (
            len(original_updates_shape_symbolic) == len(expected_updates_shape_d2) + 1
            and _are_dims_equal(original_updates_shape_symbolic[0], 1, s)
            and _are_shapes_equal(
                tuple(original_updates_shape_symbolic[1:]), expected_updates_shape_d2, s
            )
        ):
            squeezed_updates_name = s.get_unique_name(
                f"{original_updates_name_val}_dropN_d2"
            )
            s.add_node(
                helper.make_node(
                    "Squeeze",
                    [
                        original_updates_name_val,
                        s.get_constant_name(np.array([0], dtype=np.int64)),
                    ],
                    [squeezed_updates_name],
                )
            )
            _manually_ensure_shape_env_entry(
                s,
                squeezed_updates_name,
                expected_updates_shape_d2,
                original_updates_dtype_np,
                "Depth2SqueezeUpdates",
            )
            _final_updates_name_val_to_return = squeezed_updates_name
            original_updates_shape_symbolic = expected_updates_shape_d2
        elif not _are_shapes_equal(
            original_updates_shape_symbolic, expected_updates_shape_d2, s
        ):
            # Safe fallback: Reshape to the expected (B, L, ...).
            tgt = []
            for dim in expected_updates_shape_d2:
                if isinstance(dim, (int, np.integer)):
                    tgt.append(int(dim))
                else:
                    tgt.append(
                        int(_make_shape_concrete_for_prod((dim,), s, "d2_rs")[0])
                    )
            reshaped_updates_name = s.get_unique_name(
                f"{original_updates_name_val}_to_ONNX_d2"
            )
            s.add_node(
                helper.make_node(
                    "Reshape",
                    [
                        original_updates_name_val,
                        s.get_constant_name(np.array(tgt, dtype=np.int64)),
                    ],
                    [reshaped_updates_name],
                )
            )
            _manually_ensure_shape_env_entry(
                s,
                reshaped_updates_name,
                expected_updates_shape_d2,
                original_updates_dtype_np,
                "Depth2ReshapedUpdates",
            )
            _final_updates_name_val_to_return = reshaped_updates_name
            original_updates_shape_symbolic = expected_updates_shape_d2
        else:
            _final_updates_name_val_to_return = original_updates_name_val

        # Reflect the ONNX expectation going forward.
        current_expected_onnx_updates_shape = expected_updates_shape_d2
    else:
        # ────────────────────────────────────────────────────────────────
        #  📐  depth‑3 strategy  (|sdod| == 2, window update on H×W patch)
        # ────────────────────────────────────────────────────────────────
        # depth‑3 pattern: 2 indexed axes (H,W) + *implicit* batch axis
        use_depth3_for_batched_hw_scatter = (
            len(sdod) == 2
            and not iwd
            and not obd
            and len(uwd) == op_rank  # every *operand* axis is a window‑axis
            and upd_rank == op_rank + 1  # updates has the leading batch dim
            and _are_shapes_equal(jax_indices_shape_symbolic, (1, 2), s)
        )

        if use_depth3_for_batched_hw_scatter:
            logger.info("Applying depth-3 indices strategy for H×W window scatter.")
            # Operand axes: 0:B, 1:H_total, 2:W_total, 3:C
            # We treat k=3 indexed dims: (B, H, W).
            B_val = _make_shape_concrete_for_prod(
                (operand_shape_symbolic[0],), s, "d3_B"
            )[0]
            H_val = _make_shape_concrete_for_prod(
                (original_updates_shape_symbolic[2],), s, "d3_H"
            )[0]
            W_val = _make_shape_concrete_for_prod(
                (original_updates_shape_symbolic[3],), s, "d3_W"
            )[0]

            # ---- 1️⃣  row0 / col0 scalars ---------------------------------
            squeeze_idx = s.get_unique_name(f"{current_indices_name}_squeezed_d3")
            s.add_node(
                helper.make_node(
                    "Squeeze",
                    [
                        current_indices_name,
                        s.get_constant_name(np.array([0], dtype=np.int64)),
                    ],
                    [squeeze_idx],
                )
            )
            # (1,2) --squeeze[0]--> (2,)
            _manually_ensure_shape_env_entry(s, squeeze_idx, (2,), np.int64, "SqueezedIdxD3")
            # gather(0) → row0   ;   gather(1) → col0
            row0_name = s.get_unique_name("row0_d3")
            col0_name = s.get_unique_name("col0_d3")
            s.add_node(
                helper.make_node(
                    "Gather",
                    [squeeze_idx, s.get_constant_name(np.array([0], dtype=np.int64))],
                    [row0_name],
                    axis=0,
                )
            )
            s.add_node(
                helper.make_node(
                    "Gather",
                    [squeeze_idx, s.get_constant_name(np.array([1], dtype=np.int64))],
                    [col0_name],
                    axis=0,
                )
            )
            _manually_ensure_shape_env_entry(s, row0_name, (), np.int64, "Row0Scalar")
            _manually_ensure_shape_env_entry(s, col0_name, (), np.int64, "Col0Scalar")

            # ---- 2️⃣  build B×H×W grids for each coordinate ---------------
            #
            #   b : 0‥B‑1         shape (B,1,1)
            #   i : 0‥H‑1         shape (1,H,1)  + row0
            #   j : 0‥W‑1         shape (1,1,W)  + col0
            #
            arange_b = s.get_unique_name("arange_B_d3")
            s.add_node(
                helper.make_node(
                    "Range",
                    [
                        s.get_constant_name(np.array(0, dtype=np.int64)),
                        s.get_constant_name(np.array(B_val, dtype=np.int64)),
                        s.get_constant_name(np.array(1, dtype=np.int64)),
                    ],
                    [arange_b],
                )
            )
            _manually_ensure_shape_env_entry(
                s, arange_b, (B_val,), np.int64, "ArangeBD3"
            )
            unsq_b = s.get_unique_name("unsq_B_d3")
            s.add_node(
                helper.make_node(
                    "Unsqueeze",
                    [arange_b, s.get_constant_name(np.array([1, 2], dtype=np.int64))],
                    [unsq_b],
                )
            )  # (B,1,1)
            _manually_ensure_shape_env_entry(
                s, unsq_b, (B_val, 1, 1), np.int64, "UnsqBD3"
            )
            arange_h = s.get_unique_name("arange_H_d3")
            s.add_node(
                helper.make_node(
                    "Range",
                    [
                        s.get_constant_name(np.array(0, dtype=np.int64)),
                        s.get_constant_name(np.array(H_val, dtype=np.int64)),
                        s.get_constant_name(np.array(1, dtype=np.int64)),
                    ],
                    [arange_h],
                )
            )
            _manually_ensure_shape_env_entry(
                s, arange_h, (H_val,), np.int64, "ArangeHD3"
            )
            add_h = s.get_unique_name("row_plus_start_d3")
            s.add_node(helper.make_node("Add", [arange_h, row0_name], [add_h]))
            _manually_ensure_shape_env_entry(
                s, add_h, (H_val,), np.int64, "RowPlusStartD3"
            )
            unsq_h = s.get_unique_name("unsq_H_d3")
            s.add_node(
                helper.make_node(
                    "Unsqueeze",
                    [add_h, s.get_constant_name(np.array([0, 2], dtype=np.int64))],
                    [unsq_h],
                )
            )  # (1,H,1)
            _manually_ensure_shape_env_entry(
                s, unsq_h, (1, H_val, 1), np.int64, "UnsqHD3"
            )
            arange_w = s.get_unique_name("arange_W_d3")
            s.add_node(
                helper.make_node(
                    "Range",
                    [
                        s.get_constant_name(np.array(0, dtype=np.int64)),
                        s.get_constant_name(np.array(W_val, dtype=np.int64)),
                        s.get_constant_name(np.array(1, dtype=np.int64)),
                    ],
                    [arange_w],
                )
            )
            _manually_ensure_shape_env_entry(
                s, arange_w, (W_val,), np.int64, "ArangeWD3"
            )
            add_w = s.get_unique_name("col_plus_start_d3")
            s.add_node(helper.make_node("Add", [arange_w, col0_name], [add_w]))
            _manually_ensure_shape_env_entry(
                s, add_w, (W_val,), np.int64, "ColPlusStartD3"
            )
            unsq_w = s.get_unique_name("unsq_W_d3")
            s.add_node(
                helper.make_node(
                    "Unsqueeze",
                    [add_w, s.get_constant_name(np.array([0, 1], dtype=np.int64))],
                    [unsq_w],
                )
            )  # (1,1,W)
            _manually_ensure_shape_env_entry(
                s, unsq_w, (1, 1, W_val), np.int64, "UnsqWD3"
            )

            # Expand each to (B,H,W)
            target_shape_const = s.get_constant_name(
                np.array([B_val, H_val, W_val], dtype=np.int64)
            )
            b_grid = s.get_unique_name("Bgrid_d3")
            h_grid = s.get_unique_name("Hgrid_d3")
            w_grid = s.get_unique_name("Wgrid_d3")
            s.add_node(
                helper.make_node("Expand", [unsq_b, target_shape_const], [b_grid])
            )
            s.add_node(
                helper.make_node("Expand", [unsq_h, target_shape_const], [h_grid])
            )
            s.add_node(
                helper.make_node("Expand", [unsq_w, target_shape_const], [w_grid])
            )
            _manually_ensure_shape_env_entry(
                s, b_grid, (B_val, H_val, W_val), np.int64, "BgridD3"
            )
            _manually_ensure_shape_env_entry(
                s, h_grid, (B_val, H_val, W_val), np.int64, "HgridD3"
            )
            _manually_ensure_shape_env_entry(
                s, w_grid, (B_val, H_val, W_val), np.int64, "WgridD3"
            )

            # Each grid is (B,H,W). Unsqueeze to (B,H,W,1) so we can concat on axis=3.
            b_grid_u = s.get_unique_name("Bgrid_u_d3")
            h_grid_u = s.get_unique_name("Hgrid_u_d3")
            w_grid_u = s.get_unique_name("Wgrid_u_d3")
            s.add_node(
                helper.make_node(
                    "Unsqueeze",
                    [b_grid, s.get_constant_name(np.array([3], dtype=np.int64))],
                    [b_grid_u],
                )
            )
            s.add_node(
                helper.make_node(
                    "Unsqueeze",
                    [h_grid, s.get_constant_name(np.array([3], dtype=np.int64))],
                    [h_grid_u],
                )
            )
            s.add_node(
                helper.make_node(
                    "Unsqueeze",
                    [w_grid, s.get_constant_name(np.array([3], dtype=np.int64))],
                    [w_grid_u],
                )
            )
            _manually_ensure_shape_env_entry(
                s, b_grid_u, (B_val, H_val, W_val, 1), np.int64, "BgridUnsqD3"
            )
            _manually_ensure_shape_env_entry(
                s, h_grid_u, (B_val, H_val, W_val, 1), np.int64, "HgridUnsqD3"
            )
            _manually_ensure_shape_env_entry(
                s, w_grid_u, (B_val, H_val, W_val, 1), np.int64, "WgridUnsqD3"
            )

            # Concat last to (B,H,W,3)
            cat3 = s.get_unique_name("indices_BHW3_d3")
            s.add_node(
                helper.make_node(
                    "Concat",
                    [b_grid_u, h_grid_u, w_grid_u],
                    [cat3],
                    axis=3,
                )
            )
            _manually_ensure_shape_env_entry(
                s, cat3, (B_val, H_val, W_val, 3), np.int64, "CatIndicesBHW3D3"
            )

            # ---- 4️⃣  Flatten to match test post_check and ScatterND spec path ----
            # Build named Constant tensors for shapes so post-check can count them,
            # and register value_info so the builder is satisfied.
            def _named_shape_const(name_base: str, values: list[int]) -> str:
                out_name = s.get_unique_name(name_base)
                # IMPORTANT: attr tensor name == node output name (so post-check's
                # const_map keyed by attr.t.name matches the Reshape input name)
                tensor = helper.make_tensor(
                    name=out_name,
                    data_type=TensorProto.INT64,
                    dims=[len(values)],
                    vals=values,
                )
                s.add_node(
                    helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=[out_name],
                        value=tensor,
                    )
                )
                # Register shape/dtype for builder's strict value_info pass
                _manually_ensure_shape_env_entry(
                    s,
                    out_name,
                    (len(values),),
                    np.int64,
                    "NamedShapeConstD3",
                )
                return out_name

            shape_N3_name = _named_shape_const("shape_N3", [-1, 3])
            shape_N1_name = _named_shape_const("shape_N1", [-1, 1])

            # Indices: (B,H,W,3) -> (-1,3) using a named Constant
            flat_idx = s.get_unique_name("indices_flat_N3_d3")
            s.add_node(
                helper.make_node(
                    "Reshape",
                    [cat3, shape_N3_name],
                    [flat_idx],
                )
            )
            _manually_ensure_shape_env_entry(
                s, flat_idx, (-1, 3), np.int64, "FlatDepth3Idx"
            )

            # Updates: original shape -> (-1,1) using a named Constant
            reshaped_upd_name = s.get_unique_name("updates_flat_N1_d3")
            s.add_node(
                helper.make_node(
                    "Reshape",
                    [original_updates_name_val, shape_N1_name],
                    [reshaped_upd_name],
                )
            )
            _manually_ensure_shape_env_entry(
                s,
                reshaped_upd_name,
                (-1, 1),
                original_updates_dtype_np,
                "FlatDepth3Upd",
            )

            # Return flat tensors and keep bookkeeping consistent
            final_indices_name_to_return = flat_idx
            _final_updates_name_val_to_return = reshaped_upd_name

            # For downstream shape logic, reflect the flattened shapes
            processed_indices_shape_for_default_path = (-1, 3)
            target_indices_shape_symbolic = (-1, 3)
            original_updates_shape_symbolic = (-1, 1)
            current_expected_onnx_updates_shape = (-1, 1)

        if (not use_depth3_for_batched_hw_scatter) and not _are_shapes_equal(
            original_updates_shape_symbolic, current_expected_onnx_updates_shape, s
        ):
            logger.warning(
                f"Default path: JAX updates shape {original_updates_shape_symbolic} "
                f"mismatches ONNX ScatterND expected updates shape {current_expected_onnx_updates_shape}. "
                f"Attempting Reshape if element count matches."
            )
            try:
                concrete_orig_upd_shape = _make_shape_concrete_for_prod(
                    original_updates_shape_symbolic, s, "orig_updates_nelem_default"
                )
                concrete_exp_upd_shape = _make_shape_concrete_for_prod(
                    current_expected_onnx_updates_shape, s, "exp_updates_nelem_default"
                )

                original_nelem = (
                    int(np.prod(concrete_orig_upd_shape).item())
                    if concrete_orig_upd_shape
                    else 1
                )
                if (
                    not concrete_orig_upd_shape
                    and isinstance(concrete_orig_upd_shape, tuple)
                    and len(concrete_orig_upd_shape) == 0
                ):
                    original_nelem = 1

                expected_nelem = (
                    int(np.prod(concrete_exp_upd_shape).item())
                    if concrete_exp_upd_shape
                    else 1
                )
                if (
                    not concrete_exp_upd_shape
                    and isinstance(concrete_exp_upd_shape, tuple)
                    and len(concrete_exp_upd_shape) == 0
                ):
                    expected_nelem = 1

                if any(d == 0 for d in concrete_orig_upd_shape):
                    original_nelem = 0
                if any(d == 0 for d in concrete_exp_upd_shape):
                    expected_nelem = 0

                if original_nelem == 0 and expected_nelem == 0:
                    _manually_ensure_shape_env_entry(
                        s,
                        _final_updates_name_val_to_return,
                        current_expected_onnx_updates_shape,
                        original_updates_dtype_np,
                        "DefaultUpdates_EmptyShapeOK",
                    )
                elif original_nelem == expected_nelem:
                    # START of modification: Check if Reshape is just a Squeeze
                    is_squeeze = False
                    squeeze_axis = -1
                    if (
                        len(original_updates_shape_symbolic)
                        == len(current_expected_onnx_updates_shape) + 1
                    ):
                        for i in range(len(original_updates_shape_symbolic)):
                            # Check if removing the dimension at axis `i` results in the expected shape
                            if original_updates_shape_symbolic[i] == 1:
                                temp_shape = list(original_updates_shape_symbolic)
                                temp_shape.pop(i)
                                if _are_shapes_equal(
                                    tuple(temp_shape),
                                    current_expected_onnx_updates_shape,
                                    s,
                                ):
                                    is_squeeze = True
                                    squeeze_axis = i
                                    break

                    if is_squeeze:
                        logger.debug(
                            f"Replacing Reshape with Squeeze on axis {squeeze_axis} for updates."
                        )
                        squeezed_updates_name = s.get_unique_name(
                            f"{original_updates_name_val}_squeezed_default"
                        )
                        s.add_node(
                            helper.make_node(
                                "Squeeze",
                                [
                                    original_updates_name_val,
                                    s.get_constant_name(
                                        np.array([squeeze_axis], dtype=np.int64)
                                    ),
                                ],
                                [squeezed_updates_name],
                            )
                        )
                        _manually_ensure_shape_env_entry(
                            s,
                            squeezed_updates_name,
                            current_expected_onnx_updates_shape,
                            original_updates_dtype_np,
                            "DefaultSqueezedUpdates",
                        )
                        _final_updates_name_val_to_return = squeezed_updates_name
                    else:
                        # Fallback to original Reshape logic
                        reshaped_updates_name = s.get_unique_name(
                            f"{original_updates_name_val}_reshaped_default"
                        )
                        concrete_target_for_op_list_upd = []
                        has_minus_one_already_upd = False
                        for i_dim, dim_sym_val_upd in enumerate(
                            current_expected_onnx_updates_shape
                        ):
                            if isinstance(dim_sym_val_upd, int):
                                concrete_target_for_op_list_upd.append(dim_sym_val_upd)
                            else:
                                if not has_minus_one_already_upd:
                                    concrete_target_for_op_list_upd.append(-1)
                                    has_minus_one_already_upd = True
                                else:
                                    concrete_target_for_op_list_upd.append(
                                        int(
                                            _make_shape_concrete_for_prod(
                                                (dim_sym_val_upd,),
                                                s,
                                                f"reshape_target_updates_dim_def_{i_dim}",
                                            )[0]
                                        )
                                    )
                        s.add_node(
                            helper.make_node(
                                "Reshape",
                                [
                                    original_updates_name_val,
                                    s.get_constant_name(
                                        np.array(
                                            concrete_target_for_op_list_upd,
                                            dtype=np.int64,
                                        )
                                    ),
                                ],
                                [reshaped_updates_name],
                            )
                        )
                        _manually_ensure_shape_env_entry(
                            s,
                            reshaped_updates_name,
                            current_expected_onnx_updates_shape,
                            original_updates_dtype_np,
                            "DefaultReshapedUpdates",
                        )
                        _final_updates_name_val_to_return = reshaped_updates_name
                else:  # Element count mismatch
                    # We may be missing a trailing singleton (e.g. expected rank = orig_rank+1 with last dim 1).
                    # Try an Unsqueeze at the end *before* padding.
                    try:
                        if (
                            len(current_expected_onnx_updates_shape)
                            == len(original_updates_shape_symbolic) + 1
                            and isinstance(
                                current_expected_onnx_updates_shape[-1],
                                (int, np.integer),
                            )
                            and int(current_expected_onnx_updates_shape[-1]) == 1
                        ):
                            unsq_axis = len(
                                original_updates_shape_symbolic
                            )  # append at the end
                            unsqueezed_updates_name = s.get_unique_name(
                                f"{_final_updates_name_val_to_return}_unsq_lastdim"
                            )
                            s.add_node(
                                helper.make_node(
                                    "Unsqueeze",
                                    [
                                        _final_updates_name_val_to_return,
                                        s.get_constant_name(
                                            np.array([unsq_axis], dtype=np.int64)
                                        ),
                                    ],
                                    [unsqueezed_updates_name],
                                )
                            )
                            _manually_ensure_shape_env_entry(
                                s,
                                unsqueezed_updates_name,
                                tuple(list(original_updates_shape_symbolic) + [1]),
                                original_updates_dtype_np,
                                "DefaultUnsqueezeUpdates",
                            )
                            _final_updates_name_val_to_return = unsqueezed_updates_name
                            original_updates_shape_symbolic = tuple(
                                list(original_updates_shape_symbolic) + [1]
                            )
                    except Exception:
                        # best-effort; if this fails we'll still try padding below
                        pass

                    # ---- ensure we have a neutral pad value available ----
                    neutral_val_pad = _get_neutral_value(
                        reduction, original_updates_dtype_np
                    )
                    neutral_updates_name_pad = s.get_constant_name(neutral_val_pad)
                    # ------------------------------------------------------
                    (
                        maybe_padded_name,
                        maybe_padded_shape,
                    ) = _auto_pad_updates_if_smaller(
                        s,
                        _final_updates_name_val_to_return,
                        original_updates_shape_symbolic,
                        current_expected_onnx_updates_shape,
                        neutral_updates_name_pad,
                        original_updates_dtype_np,
                        "DefaultUpdates",
                    )
                    if maybe_padded_name != _final_updates_name_val_to_return:
                        _final_updates_name_val_to_return = maybe_padded_name
                        original_updates_shape_symbolic = maybe_padded_shape
                        original_nelem = expected_nelem  # padding fixed the size
                    else:
                        err_msg = (
                            f"Default path: Updates element count mismatch for ScatterND. "
                            f"Original JAX updates shape {original_updates_shape_symbolic} "
                            f"cannot be reshaped/padded to expected ONNX shape "
                            f"{current_expected_onnx_updates_shape}. "
                            f"Operand: {final_operand_name}{operand_shape_symbolic}, "
                            f"Indices: {final_indices_name_to_return}{processed_indices_shape_for_default_path}. "
                            f"Jax DimensionNumbers: {dimension_numbers}"
                        )
                        logger.error(err_msg)
                        raise ValueError(err_msg)
            except ValueError as ve:
                if "Updates element count mismatch" in str(
                    ve
                ) or "Cannot make shape concrete" in str(ve):
                    raise
                else:
                    err_msg = (
                        f"Default path: Could not prepare updates for ScatterND due to other ValueError: {ve}. "
                        f"Operand: {final_operand_name}{operand_shape_symbolic}, "
                        f"Indices: {final_indices_name_to_return}{processed_indices_shape_for_default_path}. "
                        f"Jax DimensionNumbers: {dimension_numbers}"
                    )
                    logger.error(err_msg)
                    raise ValueError(err_msg) from ve
        else:
            _manually_ensure_shape_env_entry(
                s,
                _final_updates_name_val_to_return,
                current_expected_onnx_updates_shape,
                original_updates_dtype_np,
                "DefaultUpdates_ShapeOK",
            )

    # --- Expected ONNX updates shape ------------------------------------
    # IMPORTANT:
    # Do *not* override `current_expected_onnx_updates_shape` here.
    # It already reflects the path taken above (including window-scatter and custom
    # flattening strategies). Recomputing it with the plain ONNX ScatterND formula
    # can desynchronize shapes for cases like windowed updates (e.g. 2D H×W slices).
    # If a future path truly produces pure ScatterND semantics, set the value explicitly
    # in that path instead.

    # -----------------------------------------------------------------
    #  ➤  JAX `FILL_OR_DROP` ⇒   ONNX: mask-out out-of-range rows
    # -----------------------------------------------------------------
    # If JAX asked for out‐of‐bounds entries to be dropped, mask them here

    # --- right before the FILL_OR_DROP block, after you’ve finalized names ---
    # final_indices_name_to_return, _final_updates_name_val_to_return are decided here
    idx_sds = s.shape_env.get(final_indices_name_to_return)
    upd_sds = s.shape_env.get(_final_updates_name_val_to_return)
    idx_shape = (idx_sds.shape if isinstance(idx_sds, ShapeDtypeStruct) else idx_sds)
    upd_shape = (upd_sds.shape if isinstance(upd_sds, ShapeDtypeStruct) else upd_sds)
    # if you still want to keep target_indices_shape_symbolic consistent for later logs:
    # target_indices_shape_symbolic = idx_shape

    # --- FILL_OR_DROP gating (replace your if with this) ---
    is_fill_or_drop = (
        scatter_mode == GatherScatterMode.FILL_OR_DROP
        or (isinstance(scatter_mode, str) and scatter_mode.upper() == "FILL_OR_DROP")
    )

    if is_fill_or_drop:
        # ---------------- Step 1: build a boolean mask per *row* -----------
        op_aval = operand_v.aval
        op_rank = len(op_aval.shape)

        operand_shape_tensor_name = s.get_unique_name("operand_shape_tensor")
        s.add_node(helper.make_node("Shape", [final_operand_name], [operand_shape_tensor_name]))
        _manually_ensure_shape_env_entry(s, operand_shape_tensor_name, (op_rank,), np.int64, "OperandShape")

        zero_tensor_name = s.get_constant_name(np.array(0, dtype=np.int64))

        # lower bounds: indices >= 0      (SHAPE = idx_shape)
        low_ok_name = s.get_unique_name("low_bounds_ok")
        s.add_node(helper.make_node("GreaterOrEqual", [final_indices_name_to_return, zero_tensor_name], [low_ok_name]))
        _manually_ensure_shape_env_entry(s, low_ok_name, idx_shape, np.bool_, "LowBoundsOK")

        # dimension limits for the *scatter dims* only: shape = (K,)
        scatter_dims = list(dimension_numbers.scatter_dims_to_operand_dims)  # e.g. [0] or [0,1]
        dims_const_name = s.get_constant_name(np.array(scatter_dims, dtype=np.int64))

        dim_limits_name = s.get_unique_name("dim_limits")
        s.add_node(helper.make_node("Gather", [operand_shape_tensor_name, dims_const_name], [dim_limits_name], axis=0))
        _manually_ensure_shape_env_entry(s, dim_limits_name, (len(scatter_dims),), np.int64, "DimLimits")

        # reshape to broadcastable and then expand to idx_shape
        idx_rank = len(idx_shape)
        dim_limits_reshaped_name = s.get_unique_name("dim_limits_reshaped")
        reshape_target = [1] * (idx_rank - 1) + [len(scatter_dims)]
        s.add_node(helper.make_node("Reshape",
                                   [dim_limits_name, s.get_constant_name(np.array(reshape_target, dtype=np.int64))],
                                   [dim_limits_reshaped_name]))
        _manually_ensure_shape_env_entry(s, dim_limits_reshaped_name, tuple(reshape_target), np.int64, "DimLimitsReshaped")

        # dynamic shape for Expand target, but register as idx_shape
        shape_of_indices_name = s.get_unique_name("shape_of_indices_for_bc")
        s.add_node(helper.make_node("Shape", [final_indices_name_to_return], [shape_of_indices_name]))
        _manually_ensure_shape_env_entry(s, shape_of_indices_name, (idx_rank,), np.int64, "IdxShapeForBroadcast")

        dim_limits_bc_name = s.get_unique_name("dim_limits_bc")
        s.add_node(helper.make_node("Expand", [dim_limits_reshaped_name, shape_of_indices_name], [dim_limits_bc_name]))
        _manually_ensure_shape_env_entry(s, dim_limits_bc_name, idx_shape, np.int64, "DimLimitsBroadcast")

        # upper bounds: indices < dim_limits_bc   (SHAPE = idx_shape)
        high_ok_name = s.get_unique_name("high_bounds_ok")
        s.add_node(helper.make_node("Less", [final_indices_name_to_return, dim_limits_bc_name], [high_ok_name]))
        _manually_ensure_shape_env_entry(s, high_ok_name, idx_shape, np.bool_, "HighBoundsOK")

        # elementwise AND over K, still (B,L,K)
        both_ok_name = s.get_unique_name("both_bounds_ok")
        s.add_node(helper.make_node("And", [low_ok_name, high_ok_name], [both_ok_name]))
        _manually_ensure_shape_env_entry(s, both_ok_name, idx_shape, np.bool_, "BothBoundsOK")

        # Reduce along last axis (K) → (B,L)
        row_ok_name = s.get_unique_name("row_ok")
        s.add_node(helper.make_node("ReduceAll", [both_ok_name], [row_ok], axes=[-1], keepdims=0))
        row_ok_shape = tuple(idx_shape[:-1])  # (B,L)
        _manually_ensure_shape_env_entry(s, row_ok_name, row_ok_shape, np.bool_, "RowOK")

        # Broadcast row_ok to align with updates (B,L, …window…)
        upd_rank = len(upd_shape)
        if upd_rank >= 3:
            # IMPORTANT: keep L at axis=1; add ones on axes [2..upd_rank-1]
            axes_to_unsq = np.arange(2, upd_rank, dtype=np.int64)
            row_ok_bc = s.get_unique_name("row_ok_bc")
            s.add_node(helper.make_node("Unsqueeze",
                                        [row_ok_name, s.get_constant_name(axes_to_unsq)],
                                        [row_ok_bc]))
            bc_shape = row_ok_shape + (1,) * (upd_rank - 2)  # (B,L,1,1,...)
            _manually_ensure_shape_env_entry(s, row_ok_bc, bc_shape, np.bool_, "RowOkBroadcast")
            row_ok_name = row_ok_bc
        # else: (B,L) already lines up with (B,L) for 2-D updates

        # neutral for this reduction & dtype
        neutral_val = _get_neutral_value(reduction, _ensure_np_dtype(s.shape_env[_final_updates_name_val_to_return].dtype))
        neutral_updates_name = s.get_constant_name(neutral_val)

        # safe updates: Where(row_ok, updates, neutral)   (SHAPE = upd_shape)
        safe_updates_name = s.get_unique_name("safe_updates")
        s.add_node(helper.make_node("Where", [row_ok_name, _final_updates_name_val_to_return, neutral_updates_name],
                                    [safe_updates_name]))
        _manually_ensure_shape_env_entry(s, safe_updates_name, upd_shape,
                                         _ensure_np_dtype(s.shape_env[_final_updates_name_val_to_return].dtype),
                                         "SafeUpdates")

        # safe indices: zero-fill bad rows   (SHAPE = idx_shape)
        safe_indices_name = s.get_unique_name("safe_indices")
        s.add_node(helper.make_node("Where", [both_ok_name, final_indices_name_to_return, zero_tensor_name],
                                    [safe_indices_name]))
        _manually_ensure_shape_env_entry(s, safe_indices_name, idx_shape, np.int64, "SafeIndices")

        # return masked triplet
        final_indices_name_to_return = safe_indices_name
        _final_updates_name_val_to_return = safe_updates_name

    # -----------------------------------------------------------------

    def get_shape_dtype_str_from_env_local(name_to_log_local: str) -> str:
        sds_info: Optional[ShapeDtypeStruct] = s.shape_env.get(name_to_log_local)
        if sds_info is not None:
            np_dtype_from_sds = _ensure_np_dtype(sds_info.dtype)
            onnx_enum_for_log = "?"
            try:
                onnx_enum_for_log = str(
                    s.builder._numpy_dtype_to_onnx(np_dtype_from_sds)
                )
            except Exception:
                pass
            shape_str_parts = []
            for dim_val in sds_info.shape:
                if isinstance(dim_val, int):
                    shape_str_parts.append(str(dim_val))
                elif hasattr(s, "_dim_to_symbol_safe") and callable(
                    s._dim_to_symbol_safe
                ):
                    try:
                        shape_str_parts.append(str(s._dim_to_symbol_safe(dim_val)))
                    except Exception:
                        shape_str_parts.append(str(dim_val))
                else:
                    shape_str_parts.append(str(dim_val))
            shape_str = f"({', '.join(shape_str_parts)})"
            return f"shape={shape_str}, np_dtype={np_dtype_from_sds.__name__ if hasattr(np_dtype_from_sds, '__name__') else np_dtype_from_sds}, ONNX_enum={onnx_enum_for_log}"
        return f"'{name_to_log_local}' NOT_IN_CONVERTER_SHAPE_ENV (checked in final logging loop)"

    logger.debug(
        f"Final prepared inputs for ONNX ScatterND (Version: {SCATTER_UTILS_VERSION}): \n"
        f"  Operand: name='{final_operand_name}', info={get_shape_dtype_str_from_env_local(final_operand_name)}\n"
        f"  Indices: name='{final_indices_name_to_return}', info={get_shape_dtype_str_from_env_local(final_indices_name_to_return)}\n"
        f"  Updates: name='{_final_updates_name_val_to_return}', info={get_shape_dtype_str_from_env_local(_final_updates_name_val_to_return)}"
    )

    return (
        final_operand_name,
        final_indices_name_to_return,
        _final_updates_name_val_to_return,
    )


def _auto_pad_updates_if_smaller(
    s: "Jaxpr2OnnxConverter",
    upd_name: str,
    orig_shape: Tuple[Any, ...],
    target_shape: Tuple[Any, ...],
    neutral_val_const_name: str,
    dtype_np: np.dtype,
    context: str,
) -> Tuple[str, Tuple[Any, ...]]:
    """
    If every dimension in `orig_shape` is <= its counterpart in
    `target_shape`, create an ONNX Pad node that right‑pads to the
    target; returns (new_name, new_shape).  Otherwise returns the
    original tuple untouched.
    """
    if len(orig_shape) != len(target_shape):
        return upd_name, orig_shape

    pad_after: list[int] = []
    can_pad = True
    for o, t in zip(orig_shape, target_shape):
        # Only handle concrete ints (symbolic -> bail out)
        if not isinstance(o, (int, np.integer)) or not isinstance(t, (int, np.integer)):
            can_pad = False
            break
        if o > t:
            can_pad = False
            break
        pad_after.append(int(t) - int(o))

    if not can_pad or all(p == 0 for p in pad_after):
        return upd_name, orig_shape  # nothing to do

    rank = len(orig_shape)
    pads_list = [0] * rank + pad_after  # pad at the *end* of each dim
    pads_const = s.get_constant_name(np.array(pads_list, dtype=np.int64))

    padded_name = s.get_unique_name(f"{upd_name}_pad_to_target")
    s.add_node(
        helper.make_node(
            "Pad",
            [upd_name, pads_const, neutral_val_const_name],
            [padded_name],
            mode="constant",
        )
    )
    _manually_ensure_shape_env_entry(
        s, padded_name, target_shape, dtype_np, f"{context}_AutoPad"
    )
    return padded_name, target_shape


def _get_neutral_value(reduction_op: str, dtype: np.dtype) -> np.ndarray:
    """
    Return the neutral element for the given reduction and dtype.
    """
    if reduction_op == "add":
        return np.array(0, dtype=dtype)
    if reduction_op == "mul":
        return np.array(1, dtype=dtype)
    if reduction_op == "max":
        return np.array(
            (
                np.finfo(dtype).min
                if np.issubdtype(dtype, np.floating)
                else np.iinfo(dtype).min
            ),
            dtype=dtype,
        )
    if reduction_op == "min":
        return np.array(
            (
                np.finfo(dtype).max
                if np.issubdtype(dtype, np.floating)
                else np.iinfo(dtype).max
            ),
            dtype=dtype,
        )
    # For “replace”, “none”, or anything unknown → 0
    return np.array(0, dtype=dtype)


def _onnx_expected_updates_shape(
    operand_shape: Sequence[Any], indices_shape: Sequence[Any]
) -> Tuple[Any, ...]:
    if len(indices_shape) == 0:
        # K == 0
        return tuple(operand_shape)
    k = indices_shape[-1]
    return tuple(indices_shape[:-1]) + tuple(operand_shape[k:])
