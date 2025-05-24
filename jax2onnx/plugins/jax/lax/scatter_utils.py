# jax2onnx/plugins/jax/lax/scatter_utils.py
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Optional,
    Any,
    Tuple,
)
import numpy as np
from jax import ShapeDtypeStruct
from jax.lax import ScatterDimensionNumbers
from onnx import helper, TensorProto

import logging

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scatter_utils")

# This version fixes AttributeError for get_symbol_from_dim/are_symbols_equal
SCATTER_UTILS_VERSION = "DEBUG-V20250524-SUTILS-FIX-ATTR-ERROR"


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
    # Check if the converter has a specific method to identify symbolic dimensions
    if hasattr(s, "is_symbolic_dim") and callable(s.is_symbolic_dim):
        try:
            return s.is_symbolic_dim(dim_val)
        except Exception:  # Fallback if the method errors or expects a different type
            pass
    # Fallback: if it's not a standard integer type, consider it potentially symbolic.
    # This includes JAX's symbolic dimension objects like Var, Literal, etc.
    return True


def _are_dims_equal(dim1: Any, dim2: Any, s: "Jaxpr2OnnxConverter") -> bool:
    is_dim1_sym = _is_dim_symbolic(dim1, s)
    is_dim2_sym = _is_dim_symbolic(dim2, s)

    if not is_dim1_sym and not is_dim2_sym:  # Both are concrete integers
        return int(dim1) == int(dim2)

    # If one is symbolic and the other is concrete, they are not equal by this logic
    if is_dim1_sym != is_dim2_sym:
        return False

    # Both are symbolic, fallback to object identity
    # A more advanced converter might have s.are_symbols_equal(s.get_symbol(dim1), s.get_symbol(dim2))
    # but we are removing that assumption based on the AttributeError.
    return dim1 is dim2


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


def _prepare_scatter_inputs_for_onnx(
    s: "Jaxpr2OnnxConverter",
    operand_v: Any,
    indices_v: Any,
    updates_v: Any,
    dimension_numbers: ScatterDimensionNumbers,
) -> Tuple[str, str, str]:
    logger.debug(
        f"Running _prepare_scatter_inputs_for_onnx - Version: {SCATTER_UTILS_VERSION}"
    )

    # ... (rest of the function remains the same as your uploaded "DEBUG-V20250524-SUTILS-FIX-NAMEERROR-APPLIED" version)
    # The NameError fix for onnx_indices_K_depth_val_default was already correct in that version.
    # The np.prod fix for scalar shapes was also correct in that version.
    # The only change is in _are_dims_equal above.

    # [The rest of the function from your uploaded version "DEBUG-V20250524-SUTILS-FIX-NAMEERROR-APPLIED" follows]
    # [No changes needed from line 153 onwards from that file, assuming the NameError fix was applied there]

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

    processed_indices_shape_for_default_path = target_indices_shape_symbolic

    onnx_updates_N_dim_list_default = list(
        processed_indices_shape_for_default_path[:-1]
    )
    onnx_indices_K_depth_val_default = (
        processed_indices_shape_for_default_path[-1]
        if processed_indices_shape_for_default_path
        and len(processed_indices_shape_for_default_path) > 0
        else 0
    )

    data_slice_dims_list_default = []
    if isinstance(onnx_indices_K_depth_val_default, int):
        if onnx_indices_K_depth_val_default < 0:
            raise ValueError(
                f"Indices depth K ({onnx_indices_K_depth_val_default}) must be non-negative."
            )
        if onnx_indices_K_depth_val_default <= len(operand_shape_symbolic):
            data_slice_dims_list_default.extend(
                list(operand_shape_symbolic[onnx_indices_K_depth_val_default:])
            )
        elif onnx_indices_K_depth_val_default > len(operand_shape_symbolic):
            raise ValueError(
                f"Indices depth K ({onnx_indices_K_depth_val_default}) cannot exceed operand rank ({len(operand_shape_symbolic)})."
            )
    else:
        raise ValueError(
            f"Indices depth K ({onnx_indices_K_depth_val_default}) must be concrete for updates shape calculation."
        )

    current_expected_onnx_updates_shape = tuple(
        onnx_updates_N_dim_list_default + data_slice_dims_list_default
    )

    use_depth2_for_batched_window_scatter = False
    sdod = dimension_numbers.scatter_dims_to_operand_dims
    uwd = dimension_numbers.update_window_dims
    obd = dimension_numbers.operand_batching_dims
    iwd = dimension_numbers.inserted_window_dims
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
            if scatter_target_op_axis > 0:
                if not _are_dims_equal(
                    operand_shape_symbolic[0], original_updates_shape_symbolic[0], s
                ):
                    shapes_match_for_depth2_pattern = False
                if (
                    shapes_match_for_depth2_pattern
                    and op_rank > scatter_target_op_axis + 1
                ):
                    op_trailing_shape = operand_shape_symbolic[
                        scatter_target_op_axis + 1 :
                    ]
                    if scatter_target_op_axis < len(original_updates_shape_symbolic):
                        upd_trailing_shape = original_updates_shape_symbolic[
                            scatter_target_op_axis + 1 :
                        ]
                        if not _are_shapes_equal(
                            op_trailing_shape, upd_trailing_shape, s
                        ):
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
                elif op_rank == 1:
                    pass
                else:
                    shapes_match_for_depth2_pattern = False
            if shapes_match_for_depth2_pattern and op_rank > 0:
                if scatter_target_op_axis < len(original_updates_shape_symbolic):
                    use_depth2_for_batched_window_scatter = True
                else:
                    logger.warning(
                        f"Depth-2: scatter_target_op_axis {scatter_target_op_axis} out of bounds for updates_shape {original_updates_shape_symbolic}"
                    )

    if use_depth2_for_batched_window_scatter:
        logger.info(
            "Applying generalized 'depth-2 indices' strategy for batched window scatter."
        )
        scatter_op_axis_idx = dimension_numbers.scatter_dims_to_operand_dims[0]
        concrete_operand_shape_d2 = _make_shape_concrete_for_prod(
            operand_shape_symbolic, s, "d2_op_shape"
        )
        concrete_updates_shape_d2 = _make_shape_concrete_for_prod(
            original_updates_shape_symbolic, s, "d2_upd_shape"
        )

        B_val = concrete_operand_shape_d2[0]
        L_val = concrete_updates_shape_d2[scatter_op_axis_idx]

        col_start_scalar_name = s.get_unique_name(f"{current_indices_name}_scalar_d2")
        s.add_node(
            helper.make_node(
                "Squeeze",
                [
                    current_indices_name,
                    s.get_constant_name(np.array([0], dtype=np.int64)),
                ],
                [col_start_scalar_name],
            )
        )
        _manually_ensure_shape_env_entry(
            s, col_start_scalar_name, (), final_indices_dtype_np, "ColStartScalarD2"
        )

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

        final_indices_shape_for_depth2_strat = (
            operand_shape_symbolic[0],
            original_updates_shape_symbolic[scatter_op_axis_idx],
            2,
        )
        _manually_ensure_shape_env_entry(
            s,
            indices_2d_name,
            final_indices_shape_for_depth2_strat,
            np.int64,
            "Indices2D_Depth2Strat",
        )

        final_indices_name_to_return = indices_2d_name
        _final_updates_name_val_to_return = original_updates_name_val
        current_expected_onnx_updates_shape = original_updates_shape_symbolic

    else:
        if not _are_shapes_equal(
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

                # Corrected element count for scalar (empty shape tuple)
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

                if original_nelem == 0 and expected_nelem == 0:
                    _manually_ensure_shape_env_entry(
                        s,
                        _final_updates_name_val_to_return,
                        current_expected_onnx_updates_shape,
                        original_updates_dtype_np,
                        "DefaultUpdates_EmptyShapeOK",
                    )
                elif original_nelem == expected_nelem:
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
                                        concrete_target_for_op_list_upd, dtype=np.int64
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
                else:
                    err_msg = f"Default path: Updates element count mismatch for ScatterND. Original JAX updates shape {original_updates_shape_symbolic} ({original_nelem} elements) cannot be reshaped to expected ONNX ScatterND updates shape {current_expected_onnx_updates_shape} ({expected_nelem} elements). Operand: {final_operand_name}{operand_shape_symbolic}, Indices: {final_indices_name_to_return}{processed_indices_shape_for_default_path}. Jax DimensionNumbers: {dimension_numbers}"
                    logger.error(err_msg)
                    raise ValueError(err_msg)
            except ValueError as ve:
                err_msg = f"Default path: Could not prepare updates for ScatterND due to shape error: {ve}. Original JAX updates shape {original_updates_shape_symbolic}, Expected ONNX ScatterND updates shape {current_expected_onnx_updates_shape}. Operand: {final_operand_name}{operand_shape_symbolic}, Indices: {final_indices_name_to_return}{processed_indices_shape_for_default_path}. Jax DimensionNumbers: {dimension_numbers}"
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
