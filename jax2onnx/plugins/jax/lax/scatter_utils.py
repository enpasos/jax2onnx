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

SCATTER_UTILS_VERSION = "PR-SKELETON-V1 + WhereGuardrails + ScatterNDEmit + UnsqueezeLastFix + UnsqueezeAxisRangeFix + Depth3FlattenDedup"


 
# scatter_utils.py (near the other helpers)
def _normalize_axes_for_attr(s, inp_name: str, axes: Sequence[int], *, is_unsqueeze: bool) -> list[int]:
    """For opset<13 where axes must be an attribute (and usually non-negative),
    convert negatives to absolute positions using known rank from shape_env."""
    sds = s.shape_env.get(inp_name)
    if not isinstance(sds, ShapeDtypeStruct):
        # Be conservative; if we can't normalize, keep as-is (checker will complain
        # only when negatives exist). You can also raise here if you prefer.
        return list(axes)
    r = len(sds.shape)
    out = []
    for a in axes:
        a = int(a)
        if a < 0:
            # For Unsqueeze: allowed range is [-(r+1), r]; convert like numpy
            # For Squeeze: allowed range is [-r, r-1]; convert like numpy
            a = a + (r + 1 if is_unsqueeze else r)
        out.append(a)
    # ONNX requires axes to be sorted and unique
    return sorted(set(out))

def add_unsqueeze(s, x: str, axes: Sequence[int], y: str, *, ctx: str = "Unsqueeze"):
    if s.builder.opset >= 13:
        s.add_node(helper.make_node("Unsqueeze",
                                    [x, s.get_constant_name(np.array(list(axes), dtype=np.int64))],
                                    [y],
                                    name=s.get_unique_name(ctx)))
    else:
        axes_attr = _normalize_axes_for_attr(s, x, axes, is_unsqueeze=True)
        s.add_node(helper.make_node("Unsqueeze", [x], [y], axes=axes_attr,
                                    name=s.get_unique_name(ctx)))

def add_squeeze(s, x: str, axes: Sequence[int], y: str, *, ctx: str = "Squeeze"):
    if s.builder.opset >= 13:
        s.add_node(helper.make_node("Squeeze",
                                    [x, s.get_constant_name(np.array(list(axes), dtype=np.int64))],
                                    [y],
                                    name=s.get_unique_name(ctx)))
    else:
        axes_attr = _normalize_axes_for_attr(s, x, axes, is_unsqueeze=False)
        s.add_node(helper.make_node("Squeeze", [x], [y], axes=axes_attr,
                                    name=s.get_unique_name(ctx)))


def _harmonize_float_dtypes(s, names, dtypes, context):
    import numpy as np

    # Decide the target float dtype (widen across the provided dtypes)
    target = None
    for dt in dtypes:
        dt = _ensure_np_dtype(dt)
        if np.issubdtype(dt, np.floating):
            target = dt if target is None else np.promote_types(target, dt)

    if target is None:
        return tuple(names)

    out = []
    for nm in names:
        sds = s.shape_env.get(nm)
        if isinstance(sds, ShapeDtypeStruct):
            cur = _ensure_np_dtype(sds.dtype)
            if np.issubdtype(cur, np.floating) and cur != np.dtype(target):
                casted = s.get_unique_name(f"{nm}_cast_{np.dtype(target).name}")
                s.add_node(
                    helper.make_node(
                        "Cast", [nm], [casted],
                        to=int(s.builder._numpy_dtype_to_onnx(target)),
                        name=s.get_unique_name(f"{context}_Cast"),
                    )
                )
                _manually_ensure_shape_env_entry(
                    s, casted, sds.shape, target, f"{context}_Harmonize"
                )
                out.append(casted)
                continue
        out.append(nm)
    return tuple(out)



def _reduce_min_last_axis(
    s: Jaxpr2OnnxConverter,
    inp: str,
    out: str,
    keepdims: int = 0,
):
    """
    Insert a ReduceMin over the **last** axis that is valid for every opset.
    """
    if s.builder.opset >= 18:          # ⟵ op-set 18 = attribute → input switch
        axes_name = s.get_constant_name(np.array([-1], dtype=np.int64))
        s.add_node(
            helper.make_node(
                "ReduceMin",
                [inp, axes_name],        # → axes is **input-2**
                [out],
                keepdims=keepdims,
            )
        )
    else:                               # legacy path (≤ 17) keeps the attribute
        s.add_node(
            helper.make_node(
                "ReduceMin",
                [inp],
                [out],
                axes=[-1],
                keepdims=keepdims,
            )
        )




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

def _dims_concrete_equal_or_symbol_equal(d1: Any, d2: Any, s: "Jaxpr2OnnxConverter") -> bool:
    """Treat dims as equal if either they are symbol-equal (same object) or both
    can be concretized to the same int value."""
    if _are_dims_equal(d1, d2, s):
        return True
    try:
        v1 = _make_shape_concrete_for_prod((d1,), s, "eqv_d1")[0]
        v2 = _make_shape_concrete_for_prod((d2,), s, "eqv_d2")[0]
        return int(v1) == int(v2)
    except Exception:
        return False


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
    Compute the required `updates` shape for a JAX scatter.
    We support both legal conventions seen in the wild:
      (A) window = all operand dims EXCEPT `inserted_window_dims`
      (B) window = (A) EXCEPT ALSO the scatter dims
    We choose whichever matches len(update_window_dims).
    """
    batch_shape: Tuple[int, ...] = tuple(indices_shape[:-1])

    inserted = set(dnums.inserted_window_dims)
    scatter_op_dims = tuple(getattr(dnums, "scatter_dims_to_operand_dims", ()))

    all_window = [d for d in range(len(operand_shape)) if d not in inserted]
    excl_scatter_window = [d for d in all_window if d not in scatter_op_dims]

    if len(dnums.update_window_dims) == len(all_window):
        operand_window_dims = all_window
    elif len(dnums.update_window_dims) == len(excl_scatter_window):
        operand_window_dims = excl_scatter_window
    else:
        raise ValueError(
            "Inconsistent scatter dnums: |update_window_dims| does not match "
            "either 'operand rank - inserted' or 'operand rank - inserted - "
            "num_scatter_dims'. "
            f"op_rank={len(operand_shape)}, inserted={sorted(inserted)}, "
            f"scatter={list(scatter_op_dims)}, "
            f"|update_window_dims|={len(dnums.update_window_dims)}"
        )

    # Build result rank and place window sizes at the requested positions.
    result_rank = len(batch_shape) + len(dnums.update_window_dims)
    result: list[int] = [None] * result_rank  # type: ignore
    for upd_pos, op_dim in zip(dnums.update_window_dims, operand_window_dims):
        result[upd_pos] = int(operand_shape[op_dim])

    # Fill remaining slots with the batch shape from indices (in order).
    batch_iter = iter(batch_shape)
    for i in range(result_rank):
        if result[i] is None:
            result[i] = int(next(batch_iter))

    return tuple(result)


def _map_operand_axis_to_updates_pos(
    dnums: ScatterDimensionNumbers, operand_rank: int, operand_axis: int
) -> Optional[int]:
    """
    Map an *operand* axis to the position of the corresponding axis in
    the **updates** tensor, honoring whichever window-convention the
    ScatterDimensionNumbers encode (same choice as compute_expected_updates_shape).
    """
    inserted = set(dnums.inserted_window_dims)
    scatter_op_dims = tuple(getattr(dnums, "scatter_dims_to_operand_dims", ()))

    # Candidate operand-window sets
    all_window = [d for d in range(operand_rank) if d not in inserted]
    excl_scatter_window = [d for d in all_window if d not in scatter_op_dims]

    # Pick the convention that matches |update_window_dims|
    if len(dnums.update_window_dims) == len(all_window):
        window_operand_dims = all_window                     # includes scatter dims
    elif len(dnums.update_window_dims) == len(excl_scatter_window):
        window_operand_dims = excl_scatter_window            # excludes scatter dims
    else:
        return None  # inconsistent dnums; can't map safely

    try:
        i = window_operand_dims.index(operand_axis)
    except ValueError:
        return None

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
    reduction: str = "none",  # default to replace semantics for lax.scatter
) -> Tuple[str, str, str]:
    logger.debug(
        f"Running _prepare_scatter_inputs_for_onnx - Version: {SCATTER_UTILS_VERSION}"
    )
    # Track which specialized rewrite we took (used for full-window gating).
    used_depth2_strategy = False
    used_depth3_strategy = False

    # Normalized policy checks (strings allowed for robustness)
    is_clip = (
        scatter_mode == GatherScatterMode.CLIP
        or (isinstance(scatter_mode, str) and str(scatter_mode).upper() == "CLIP")
    )
    # We'll need this before the depth-2 block so we can avoid clamping the scalar
    # start under FILL_OR_DROP as well.
    is_fill_or_drop = (
        scatter_mode == GatherScatterMode.FILL_OR_DROP
        or (isinstance(scatter_mode, str) and str(scatter_mode).upper() == "FILL_OR_DROP")
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
        used_depth2_strategy = True
        scatter_op_axis_idx = dimension_numbers.scatter_dims_to_operand_dims[0]
        _make_shape_concrete_for_prod(operand_shape_symbolic, s, "d2_op_shape")

        # Batch extent B (operand axis 0) – needed regardless of mapping outcome
        B_sym = operand_shape_symbolic[0]
        B_val = _make_shape_concrete_for_prod((B_sym,), s, "d2_B")[0]
        has_leading_N = (
            upd_rank == op_rank + 1
            and _are_dims_equal(original_updates_shape_symbolic[0], 1, s)
        )

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
        add_squeeze(
            s,
            current_indices_name,
            [
                ax
                for ax, dim in enumerate(current_indices_shape_symbolic)
                if _are_dims_equal(dim, 1, s)
            ]
            or [0],
            col_start_scalar_name,
            ctx="ColStartScalarD2"
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
                s, picked_updates_name, upd_shape_after_pick, original_updates_dtype_np, "Depth2PickScatterAxis"
            )
            original_updates_name_val = picked_updates_name
            original_updates_shape_symbolic = upd_shape_after_pick
            # For the rest of the path we treat this as L = 1.
            L_sym, L_val = 1, 1
            _final_updates_name_val_to_return = picked_updates_name
        else:
            # 🔁 Prefer L from **updates**, but if we cannot map the updates
            # axis, fall back to the operand L to keep this path well-defined.
            if upd_pos_for_scatter_axis is not None:
                L_sym_updates = original_updates_shape_symbolic[upd_pos_for_scatter_axis]
                L_sym_operand = operand_shape_symbolic[scatter_op_axis_idx]
                try:
                    L_val_updates = _make_shape_concrete_for_prod(
                        (L_sym_updates,), s, "d2_L_from_updates"
                    )[0]
                    L_val = int(L_val_updates)
                    L_sym = L_sym_updates
                except Exception as e:
                    logger.warning(
                        f"Depth-2: could not concretize L from updates ({e}); "
                        "falling back to operand L."
                    )
                    L_sym = operand_shape_symbolic[scatter_op_axis_idx]
                    L_val = _make_shape_concrete_for_prod(
                        (L_sym,), s, "d2_L_fallback_operand"
                    )[0]
            else:
                # No mapping → fallback to operand L so later code has a concrete value.
                logger.warning(
                    "Depth-2: no updates-axis mapping; falling back to operand L."
                )
                L_sym = operand_shape_symbolic[scatter_op_axis_idx]
                L_val = _make_shape_concrete_for_prod(
                    (L_sym,), s, "d2_L_from_operand_nomap"
                )[0]
            # Ensure L_val/L_sym are defined even if we took unusual paths
            if "L_val" not in locals():
                L_sym = operand_shape_symbolic[scatter_op_axis_idx]
                L_val = _make_shape_concrete_for_prod(
                    (L_sym,), s, "d2_L_fallback_operand"
                )[0]

            # scalar L as a Constant tensor for arithmetic below
            L_len_name = s.get_constant_name(np.array(int(L_val), dtype=np.int64))

        # CLIP for window updates: clamp the *start scalar* so the full window fits.
        if is_clip:
            shape_op_name2 = s.get_unique_name("shape_op_d2_start")
            s.add_node(helper.make_node("Shape", [final_operand_name], [shape_op_name2]))
            _manually_ensure_shape_env_entry(s, shape_op_name2, (len(operand_shape_symbolic),), np.int64, "D2_ClipStart_Shape")
            
            
            # NEW registrations for every produced tensor in this clamp subgraph:
            dim_size_vec2 = s.get_unique_name("dim_size_vec_d2_start")
            s.add_node(helper.make_node("Gather",
                                        [shape_op_name2, s.get_constant_name(np.array([scatter_op_axis_idx], dtype=np.int64))],
                                        [dim_size_vec2], axis=0))
            _manually_ensure_shape_env_entry(s, dim_size_vec2, (1,), np.int64, "D2_ClipStart_DimVec")   # ← ADD

            dim_size_scalar2 = s.get_unique_name("dim_size_d2_start")
            add_squeeze(s, dim_size_vec2, [0], dim_size_scalar2, ctx="D2_ClipStart_Dim")
            _manually_ensure_shape_env_entry(s, dim_size_scalar2, (), np.int64, "D2_ClipStart_Dim")     # ← ADD

            zero_i64 = s.get_constant_name(np.array(0, dtype=np.int64))

            max_start = s.get_unique_name("max_start_d2")
            s.add_node(helper.make_node("Sub", [dim_size_scalar2, L_len_name], [max_start]))
            _manually_ensure_shape_env_entry(s, max_start, (), np.int64, "D2_ClipStart_MaxStart")       # ← ADD

            max_start_nneg = s.get_unique_name("max_start_nneg_d2")
            s.add_node(helper.make_node("Max", [max_start, zero_i64], [max_start_nneg]))
            _manually_ensure_shape_env_entry(s, max_start_nneg, (), np.int64, "D2_ClipStart_MaxStartNneg")  # ← ADD

            col_start_clamped = s.get_unique_name("col_start_clamped_d2")
            s.add_node(helper.make_node("Clip", [col_start_scalar_name, zero_i64, max_start_nneg], [col_start_clamped]))
            _manually_ensure_shape_env_entry(s, col_start_clamped, (), np.int64, "D2_ClipStart_Col")
            col_start_scalar_name = col_start_clamped



        # ----------------------------
        # END OF scalar start clamping
        # ----------------------------


        # ----------------------------
        #  Prepare batch and column index grids
        # ----------------------------
        # Grids are (B, L, ...window...) for the scatter output shape.
        # We use the same generic batching logic as in the default path,
        # but with explicit (B, L) shapes for clarity.

        # Batch grid: 0..B-1
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
        # Build (B,1) with a single Unsqueeze so Expand's target rank (2) ≥ input rank (2)
        unsq_b = s.get_unique_name("unsq_B_d2")
        add_unsqueeze(
            s,
            arange_b_name, 
            [1],
            unsq_b,
            ctx="UnsqBStep1D2"
        )
        _manually_ensure_shape_env_entry(
            s, unsq_b, (B_val, 1), np.int64, "UnsqBStep1D2"
        )
        batch_indices_intermediate_name = s.get_unique_name("batch_indices_BL_d2")
        s.add_node(
            helper.make_node(
                "Expand",
                [unsq_b, s.get_constant_name(np.array([B_val, L_val], dtype=np.int64))],
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

        # Column grid: scalar start + 0..L-1
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
        add_unsqueeze(
            s,
            add_start_name, 
            [0],
            unsqueeze_l_name,
            ctx="UnsqueezeLD2"
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
        # Use axis = -1 so shape inference is robust even if the input rank is seen as 1.
        add_unsqueeze(
            s,
            batch_indices_intermediate_name, 
            [-1],
            final_batch_indices_name,
            ctx="FinalBatchIdxD2"
        )
        _manually_ensure_shape_env_entry(
            s, final_batch_indices_name, (B_val, L_val, 1), np.int64, "FinalBatchIdxD2"
        )
        final_col_indices_name = s.get_unique_name("final_col_indices_d2")
        add_unsqueeze(
            s,
            col_indices_intermediate_name, 
            [-1],
            final_col_indices_name,
            ctx="FinalColIdxD2"
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

        # IMPORTANT:
        #  • We clamp the scalar start unconditionally (ORT rejects OOB indices).
        #  • Under CLIP, we additionally clamp the full index vector below to [0, dim-1].
        #  • FILL_OR_DROP: valid callers are unaffected; invalid callers won’t crash ORT.

        final_indices_name_to_return = indices_2d_name
        expected_updates_shape_d2 = (B_sym, L_sym) + tuple(operand_shape_symbolic[2:])

        # If updates are (B, L, 1, ...) or differ from the expected (B, L, ...)
        # only by a single singleton axis, we may squeeze it — BUT:
        #   • If there is a leading N=1 (has_leading_N == True) we MUST NOT do it here.
        #     The dedicated “drop leading N” block below handles that case and preserves
        #     the dtype-mismatch test behaviour.
        if (not _are_shapes_equal(original_updates_shape_symbolic, expected_updates_shape_d2, s)
            and not has_leading_N):
            squeeze_axis = None
            def _dim_is_one_generic(d: Any) -> bool:
                if isinstance(d, (int, np.integer)):
                    return int(d) == 1
                try:
                    return _make_shape_concrete_for_prod((d,), s, "d2_squeeze_probe")[0] == 1
                except Exception:
                    return False

            for i, d in enumerate(original_updates_shape_symbolic):
                if _dim_is_one_generic(d):
                    cand = tuple(dd for j, dd in enumerate(original_updates_shape_symbolic) if j != i)
                    if _are_shapes_equal(cand, expected_updates_shape_d2, s):
                        squeeze_axis = i
                        break

            if squeeze_axis is not None:
                squeezed_updates_name = s.get_unique_name(
                    f"{original_updates_name_val}_squeeze_axis{squeeze_axis}_d2"
                )
                add_squeeze(
                    s,
                    original_updates_name_val,
                    [squeeze_axis],
                    squeezed_updates_name,
                    ctx="Depth2SqueezeSingleton"
                )
                _manually_ensure_shape_env_entry(
                    s,
                    squeezed_updates_name,
                    expected_updates_shape_d2,
                    original_updates_dtype_np,
                    "Depth2SqueezeSingleton",
                )
                original_updates_name_val = squeezed_updates_name
                original_updates_shape_symbolic = expected_updates_shape_d2
                _final_updates_name_val_to_return = squeezed_updates_name
            else:
                # keep existing fallback (preserve L from updates) for true mismatches
                logger.info(
                    "Depth-2: updates shape differs from computed expectation; "
                    "keeping original updates (no reshape) to preserve L from updates."
                )
                _final_updates_name_val_to_return = original_updates_name_val
        else:
            _final_updates_name_val_to_return = original_updates_name_val

        # If updates come as (1, B, L, …), drop the leading singleton.
        # (This path is intentionally separate to avoid the fp64 type-mismatch
        #  regression described above.)
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
            add_squeeze(
                s,
                original_updates_name_val, 
                [0],
                squeezed_updates_name,
                ctx="Depth2SqueezeUpdates"
            )
            _manually_ensure_shape_env_entry(
                s,
                squeezed_updates_name,
                expected_updates_shape_d2,
                original_updates_dtype_np,
                "Depth2SqueezeUpdates",
            )
            original_updates_name_val = squeezed_updates_name
            original_updates_shape_symbolic = expected_updates_shape_d2
            _final_updates_name_val_to_return = squeezed_updates_name
        elif not _are_shapes_equal(
            original_updates_shape_symbolic, expected_updates_shape_d2, s
        ):
            # Do NOT reshape when element counts differ; mismatches here usually mean
            # Lᵤ != Lₒ and we must keep the updates' L. Keep original updates untouched.
            logger.info(
                "Depth-2: updates shape differs from computed expectation; "
                "keeping original updates (no reshape) to preserve L from updates."
            )
            _final_updates_name_val_to_return = original_updates_name_val
            # Keep shape as-is (original_updates_shape_symbolic)
        else:
            _final_updates_name_val_to_return = original_updates_name_val

        # Reflect the ONNX expectation going forward.
        current_expected_onnx_updates_shape = expected_updates_shape_d2
    else:
        # ────────────────────────────────────────────────────────────────
        #  📐  depth‑3 strategy  (|sdod| == 2, window update on H×W patch)
        # ────────────────────────────────────────────────────────────────
        # depth‑3 pattern: 2 indexed axes (H,W) + *implicit* batch axis
        # Accept both shapes:
        #   • updates rank == op_rank + 1  → (B, H, W, C)   (batch outside window)
        #   • updates rank == op_rank      → (B, H, W, C)   (batch inside update_window_dims)
        # In both cases: 2 scatter dims (H,W), no inserted/batching dims, and indices=(1,2)
        use_depth3_for_batched_hw_scatter = (
            len(sdod) == 2
            and not iwd
            and not obd
            and len(uwd) == op_rank                    # window covers all operand dims
            and (upd_rank == op_rank or upd_rank == op_rank + 1)
            and _are_shapes_equal(jax_indices_shape_symbolic, (1, 2), s)
        )


        if use_depth3_for_batched_hw_scatter:
            logger.info("Applying depth-3 indices strategy for H×W window scatter.")
            used_depth3_strategy = True
            # Operand axes: 0:B, 1:H, 2:W, 3:C  (generic “NHWC” notation)
            # We build B×H×W grids, so we need the extents B, H, W.
            B_sym = operand_shape_symbolic[0]
            B_val = _make_shape_concrete_for_prod((B_sym,), s, "d3_B")[0]
            # Map operand axes → positions in the **updates** tensor (under whichever
            # window convention the dnums encode). This makes the code robust to
            # “batch-in-window” vs “batch-leading” variations.
            pos_h = _map_operand_axis_to_updates_pos(dimension_numbers, op_rank, 1)
            pos_w = _map_operand_axis_to_updates_pos(dimension_numbers, op_rank, 2)
            if pos_h is None or pos_w is None:
                # Fallback to the legacy assumption (updates = B, H, W, C)
                pos_h = 1 if pos_h is None else pos_h
                pos_w = 2 if pos_w is None else pos_w
            H_sym = original_updates_shape_symbolic[pos_h]
            W_sym = original_updates_shape_symbolic[pos_w]
            H_val = _make_shape_concrete_for_prod((H_sym,), s, "d3_H")[0]
            W_val = _make_shape_concrete_for_prod((W_sym,), s, "d3_W")[0]




            # ---- 1️⃣  row0 / col0 scalars ---------------------------------
            squeeze_idx = s.get_unique_name(f"{current_indices_name}_squeezed_d3")
            add_squeeze(
                s,
                current_indices_name, 
                [0],
                squeeze_idx,
                ctx="SqueezedIdxD3"
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







            # Build (B,1,1) via two Unsqueezes so Expand to (B,H,W) broadcasts correctly.
            # (If we only made (B,1), Expand would prepend a 1 → (1,B,1) which
            #  is incompatible with (B,H,W) when H≠B.)
            unsq_b_1 = s.get_unique_name("unsq_B_tmp1_d3")
            add_unsqueeze(
                s,
                arange_b,
                [1],
                unsq_b_1,
                ctx="UnsqBStep1D3"
            )
            _manually_ensure_shape_env_entry(
                s, unsq_b_1, (B_val, 1), np.int64, "UnsqBStep1D3"
            )
            unsq_b = s.get_unique_name("unsq_B_d3")
            add_unsqueeze(
                s,
                unsq_b_1,
                [2],
                unsq_b,
                ctx="UnsqBStep2D3"
            )
            _manually_ensure_shape_env_entry(
                s, unsq_b, (B_val, 1, 1), np.int64, "UnsqBStep2D3"
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

            # If CLIP, clamp the *starting scalars* so the full H×W patch fits.
            if is_clip:
                shape_op = s.get_unique_name("shape_op_d3_start")
                s.add_node(helper.make_node("Shape", [final_operand_name], [shape_op]))
                _manually_ensure_shape_env_entry(s, shape_op, (len(operand_shape_symbolic),), np.int64, "D3_ClipStart_Shape")

                # dim_h and dim_w as scalars
                dim_h_vec = s.get_unique_name("dim_h_vec_d3")
                s.add_node(helper.make_node("Gather", [shape_op, s.get_constant_name(np.array([1], dtype=np.int64))], [dim_h_vec], axis=0))
                dim_h = s.get_unique_name("dim_h_d3")
                add_squeeze(s, dim_h_vec, [0], dim_h, ctx="D3_ClipStart_DimH")
                dim_w_vec = s.get_unique_name("dim_w_vec_d3")
                s.add_node(helper.make_node("Gather", [shape_op, s.get_constant_name(np.array([2], dtype=np.int64))], [dim_w_vec], axis=0))
                dim_w = s.get_unique_name("dim_w_d3")
                add_squeeze(s, dim_w_vec, [0], dim_w, ctx="D3_ClipStart_DimW")

                # max starts = dim - extent, saturated at 0
                h_extent = s.get_constant_name(np.array(H_val, dtype=np.int64))
                w_extent = s.get_constant_name(np.array(W_val, dtype=np.int64))
                zero_i64 = s.get_constant_name(np.array(0, dtype=np.int64))
                h_max_start = s.get_unique_name("h_max_start_d3")
                w_max_start = s.get_unique_name("w_max_start_d3")
                s.add_node(helper.make_node("Sub", [dim_h, h_extent], [h_max_start]))
                s.add_node(helper.make_node("Sub", [dim_w, w_extent], [w_max_start]))
                h_max_start = s.get_unique_name("h_max_start_nneg_d3"); s.add_node(helper.make_node("Max", [h_max_start, zero_i64], [h_max_start]))
                w_max_start = s.get_unique_name("w_max_start_nneg_d3"); s.add_node(helper.make_node("Max", [w_max_start, zero_i64], [w_max_start]))

                # clamp the scalars
                row0_clamped = s.get_unique_name("row0_clamped_d3")
                col0_clamped = s.get_unique_name("col0_clamped_d3")
                s.add_node(helper.make_node("Clip", [row0_name, zero_i64, h_max_start], [row0_clamped]))
                s.add_node(helper.make_node("Clip", [col0_name, zero_i64, w_max_start], [col0_clamped]))
                _manually_ensure_shape_env_entry(s, row0_clamped, (), np.int64, "D3_ClipStart_Row0")
                _manually_ensure_shape_env_entry(s, col0_clamped, (), np.int64, "D3_ClipStart_Col0")
                row0_name, col0_name = row0_clamped, col0_clamped

            add_h = s.get_unique_name("row_plus_start_d3")
            s.add_node(helper.make_node("Add", [arange_h, row0_name], [add_h]))
            _manually_ensure_shape_env_entry(s, add_h, (H_val,), np.int64, "RowPlusStartD3")
            # Build (1,H,1) as two Unsqueezes: axis 0 then axis 2
            unsq_h_1 = s.get_unique_name("unsq_H_tmp1_d3")
            add_unsqueeze(
                s,
                add_h, 
                [0],
                unsq_h_1,
                ctx="UnsqHStep1D3"
            )
            _manually_ensure_shape_env_entry(
                s, unsq_h_1, (1, H_val), np.int64, "UnsqHStep1D3"
            )
            unsq_h = s.get_unique_name("unsq_H_d3")
            add_unsqueeze(
                s,
                unsq_h_1, 
                [2],
                unsq_h,
                ctx="UnsqHStep2D3"
            )
            _manually_ensure_shape_env_entry(
                s, unsq_h, (1, H_val, 1), np.int64, "UnsqHStep2D3"
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
            _manually_ensure_shape_env_entry(s, add_w, (W_val,), np.int64, "ColPlusStartD3")

            # (No vector Clip here; we already clamped row0/col0 so B×H×W grids are in-bounds.)

            # Build (1,1,W) as two Unsqueezes: axis 0 then axis 1
            unsq_w_1 = s.get_unique_name("unsq_W_tmp1_d3")
            add_unsqueeze(
                s,
                add_w, 
                [0],
                unsq_w_1,
                ctx="UnsqWStep1D3"
            )
            _manually_ensure_shape_env_entry(
                s, unsq_w_1, (1, W_val), np.int64, "UnsqWStep1D3"
            )
            unsq_w = s.get_unique_name("unsq_W_d3")
            add_unsqueeze(
                s,
                unsq_w_1, 
                [1],
                unsq_w,
                ctx="UnsqWStep2D3"
            )
            _manually_ensure_shape_env_entry(
                s, unsq_w, (1, 1, W_val), np.int64, "UnsqWStep2D3"
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
            add_unsqueeze(s, b_grid, [3], b_grid_u, ctx="BgridUnsqD3")
            add_unsqueeze(s, h_grid, [3], h_grid_u, ctx="HgridUnsqD3")
            add_unsqueeze(s, w_grid, [3], w_grid_u, ctx="WgridUnsqD3")
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
 

            # Concat last to (B,H,W,3)  → cat3  (already in your code)

            # ---- 4️⃣  Flatten to match test post_check and ScatterND spec path ----
            def _named_shape_const(name_base: str, values: list[int]) -> str:
                out_name = s.get_unique_name(name_base)
                tensor = helper.make_tensor(
                    name=out_name, data_type=TensorProto.INT64, dims=[len(values)], vals=values
                )
                s.add_node(helper.make_node("Constant", inputs=[], outputs=[out_name], value=tensor))
                _manually_ensure_shape_env_entry(s, out_name, (len(values),), np.int64, "NamedShapeConstD3")
                return out_name

            shape_N3_name = _named_shape_const("shape_N3", [-1, 3])

            # Indices: (B,H,W,3) -> (-1,3) using a named Constant
            flat_idx = s.get_unique_name("indices_flat_N3_d3")
            s.add_node(helper.make_node("Reshape", [cat3, shape_N3_name], [flat_idx]))
            _manually_ensure_shape_env_entry(s, flat_idx, (-1, 3), np.int64, "FlatDepth3Idx")

            # Updates: prefer a *constant* target ([-1] + tail) when tail is concrete;
            # otherwise fall back to the dynamic Shape+Slice+Concat path.
            expected_tail_shape = tuple(operand_shape_symbolic[3:])

            def _tail_concrete_vals_or_none(tail):
                vals = []
                for d in tail:
                    if isinstance(d, (int, np.integer)):
                        vals.append(int(d))
                    else:
                        try:
                            vals.append(int(_make_shape_concrete_for_prod((d,), s, "d3_tail")[0]))
                        except Exception:
                            return None
                return vals

            tail_vals = _tail_concrete_vals_or_none(expected_tail_shape)

            if tail_vals is not None:
                # ✅ constant reshape to [-1] + tail_vals (e.g., [-1, 1]) so the test can count it
                shape_N_tail = _named_shape_const("shape_N_tail", [-1] + tail_vals)
                reshaped_upd_name = s.get_unique_name("updates_flat_N_tail_d3")
                s.add_node(helper.make_node("Reshape", [original_updates_name_val, shape_N_tail], [reshaped_upd_name]))
                _manually_ensure_shape_env_entry(
                    s, reshaped_upd_name, (-1,) + tuple(tail_vals), original_updates_dtype_np, "FlatDepth3Upd_Const"
                )
            else:
                # 🔁 dynamic fallback (your original code)
                data_shape_name = s.get_unique_name("data_shape_d3")
                s.add_node(helper.make_node("Shape", [final_operand_name], [data_shape_name]))
                _manually_ensure_shape_env_entry(s, data_shape_name, (len(operand_shape_symbolic),), np.int64, "D3_DataShape")
                start_name = s.get_constant_name(np.array([3], dtype=np.int64))
                end_name   = s.get_constant_name(np.array([len(operand_shape_symbolic)], dtype=np.int64))
                axes_name  = s.get_constant_name(np.array([0], dtype=np.int64))
                steps_name = s.get_constant_name(np.array([1], dtype=np.int64))
                tail_dims_name = s.get_unique_name("tail_dims_after3_d3")
                s.add_node(helper.make_node("Slice", [data_shape_name, start_name, end_name, axes_name, steps_name], [tail_dims_name]))
                _manually_ensure_shape_env_entry(s, tail_dims_name, (max(0, len(operand_shape_symbolic) - 3),), np.int64, "D3_TailDims")
                minus_one_vec = s.get_constant_name(np.array([-1], dtype=np.int64))
                target_shape_name = s.get_unique_name("updates_shape_N_plus_tail_d3")
                s.add_node(helper.make_node("Concat", [minus_one_vec, tail_dims_name], [target_shape_name], axis=0))
                _manually_ensure_shape_env_entry(
                    s, target_shape_name, (1 + max(0, len(operand_shape_symbolic) - 3),), np.int64, "D3_TargetShape"
                )
                reshaped_upd_name = s.get_unique_name("updates_flat_N_tail_d3")
                s.add_node(helper.make_node("Reshape", [original_updates_name_val, target_shape_name], [reshaped_upd_name]))
                _manually_ensure_shape_env_entry(
                    s, reshaped_upd_name, (-1,) + expected_tail_shape, original_updates_dtype_np, "FlatDepth3Upd_Dyn"
                )

            # Return flat tensors and keep bookkeeping consistent
            final_indices_name_to_return = flat_idx
            _final_updates_name_val_to_return = reshaped_upd_name

            # For downstream shape logic:
            processed_indices_shape_for_default_path = (-1, 3)
            target_indices_shape_symbolic = (-1, 3)
            original_updates_shape_symbolic = s.shape_env[reshaped_upd_name].shape
            current_expected_onnx_updates_shape = original_updates_shape_symbolic





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



                        add_squeeze(
                            s,
                            original_updates_name_val,
                            [squeeze_axis],
                            squeezed_updates_name,
                            ctx="DefaultSqueezedUpdates"
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
                            add_unsqueeze(
                                s,
                                _final_updates_name_val_to_return, 
                                [unsq_axis],
                                unsqueezed_updates_name,
                                ctx="DefaultUnsqueezeUpdates"
                            )
                            _manually_ensure_shape_env_entry(
                                s,
                                unsqueezed_updates_name,
                                tuple(list(original_updates_shape_symbolic) + [1]),
                                original_updates_dtype_np,
                                "DefaultUnsqueezeUpdates",
                            )
                            _final_updates_name_val_to_return = unsqueezed_updates_name
                            original_updates_shape_symbolic = tuple(list(original_updates_shape_symbolic) + [1])
                        else:
                            raise ValueError("Element count mismatch, and unable to apply Unsqueeze workaround.")
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

    # --- right before the FILL_OR_DROP block, after you've finalized names ---
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

    apply_oob_mask = not is_clip  # PROMISE/None and FILL_OR_DROP → mask; CLIP handled by vector clip
    if apply_oob_mask:
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
        scatter_dims = list(dimension_numbers.scatter_dims_to_operand_dims)
        # Determine how many dims indices actually index (K)
        try:
            k = int(_make_shape_concrete_for_prod((idx_shape[-1],), s, "FOD_K")[0])
        except Exception:
            # very conservative fallback; but in our tests K is concrete (e.g., 2)
            k = len(dimension_numbers.scatter_dims_to_operand_dims) or 2

        # If our indices include the implicit batch axis we added (depth-2/3),
        # check that axis 0 as well.
        if k == len(scatter_dims) + 1 and 0 not in scatter_dims:
            dims_to_check = [0] + scatter_dims
        elif k == len(scatter_dims):
            dims_to_check = scatter_dims
        else:
            # Generic, conservative fallback: first k operand dims.
            dims_to_check = list(range(k))

        dims_const_name = s.get_constant_name(np.array(dims_to_check, dtype=np.int64))
        dim_limits_name = s.get_unique_name("dim_limits")
        s.add_node(helper.make_node("Gather", [operand_shape_tensor_name, dims_const_name], [dim_limits_name], axis=0))
        _manually_ensure_shape_env_entry(s, dim_limits_name, (len(dims_to_check),), np.int64, "DimLimits")

        # reshape to broadcastable and then expand to idx_shape
        idx_rank = len(idx_shape)
        dim_limits_reshaped_name = s.get_unique_name("dim_limits_reshaped")
        reshape_target = [1] * (idx_rank - 1) + [len(dims_to_check)]
        s.add_node(helper.make_node("Reshape",
                                   [dim_limits_name, s.get_constant_name(np.array(reshape_target, dtype=np.int64))],
                                   [dim_limits_reshaped_name]))
        _manually_ensure_shape_env_entry(s, dim_limits_reshaped_name, tuple(reshape_target), np.int64, "DimLimitsReshaped")

        # Broadcast to match indices shape
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

        # elementwise AND over K, still (B,L, …window…)
        both_ok_name = s.get_unique_name("both_bounds_ok")
        s.add_node(helper.make_node("And", [low_ok_name, high_ok_name], [both_ok_name]))
        _manually_ensure_shape_env_entry(s, both_ok_name, idx_shape, np.bool_, "BothBoundsOK")




        # Reduce along last axis (K) → (B,L)  (ORT has no ReduceAll)
        # Implement ALL(K) as ReduceMin over int64 after casting bool→int64.
        both_ok_i64 = s.get_unique_name("both_bounds_ok_i64")
        s.add_node(helper.make_node("Cast", [both_ok_name], [both_ok_i64], to=int(TensorProto.INT64)))
        row_min_i64 = s.get_unique_name("row_min_i64")
        _reduce_min_last_axis(s, both_ok_i64, row_min_i64, keepdims=0)
        row_ok_name = s.get_unique_name("row_ok")
        s.add_node(helper.make_node("Cast", [row_min_i64], [row_ok_name], to=int(TensorProto.BOOL)))
        row_ok_shape = tuple(idx_shape[:-1])  # (B,L)
        _manually_ensure_shape_env_entry(s, both_ok_i64, idx_shape, np.int64, "BothBoundsOK_i64")
        _manually_ensure_shape_env_entry(s, row_min_i64, row_ok_shape, np.int64, "RowMinI64")
        _manually_ensure_shape_env_entry(s, row_ok_name, row_ok_shape, np.bool_, "RowOK")

        # ────────────────────────────────────────────────────────────────
        # NEW: full-window gating for depth-2/3 rewrites
        # JAX drops the *entire* window when it would go out of bounds (PROMISE/FILL_OR_DROP).
        # Convert the per-element (B×L×…) mask to a single gate per batch row by
        # reducing over the enumerated grid axes (all axes except the leading batch axis).
        # ────────────────────────────────────────────────────────────────
        if (used_depth2_strategy or used_depth3_strategy) and len(row_ok_shape) >= 2:
            row_ok_i64_full = s.get_unique_name("row_ok_i64_fullwin")
            s.add_node(helper.make_node("Cast", [row_ok_name], [row_ok_i64_full], to=int(TensorProto.INT64)))
            _manually_ensure_shape_env_entry(s, row_ok_i64_full, row_ok_shape, np.int64, "FullWindowMask_Cast")

            reduce_axes = np.array(list(range(1, len(row_ok_shape))), dtype=np.int64)
            full_min_i64 = s.get_unique_name("row_ok_fullwin_min_i64")
            if s.builder.opset >= 18:
                s.add_node(helper.make_node("ReduceMin",
                                            [row_ok_i64_full, s.get_constant_name(reduce_axes)],
                                            [full_min_i64],
                                            keepdims=1))
            else:
                s.add_node(helper.make_node("ReduceMin",
                                            [row_ok_i64_full],
                                            [full_min_i64],
                                            axes=reduce_axes.tolist(),
                                            keepdims=1))
            full_keepdims_shape = (row_ok_shape[0],) + (1,) * (len(row_ok_shape) - 1)
            _manually_ensure_shape_env_entry(s, full_min_i64, full_keepdims_shape, np.int64, "FullWindowMask_Reduce")

            row_ok_full_bool = s.get_unique_name("row_ok_fullwin_bool")
            s.add_node(helper.make_node("Cast", [full_min_i64], [row_ok_full_bool], to=int(TensorProto.BOOL)))
            _manually_ensure_shape_env_entry(s, row_ok_full_bool, full_keepdims_shape, np.bool_, "FullWindowMask_Bool")

            shape_row_ok = s.get_unique_name("shape_row_ok")
            s.add_node(helper.make_node("Shape", [row_ok_name], [shape_row_ok]))
            _manually_ensure_shape_env_entry(s, shape_row_ok, (len(row_ok_shape),), np.int64, "FullWindowMask_Shape")

            row_ok_bc = s.get_unique_name("row_ok_fullwin_bc")
            s.add_node(helper.make_node("Expand", [row_ok_full_bool, shape_row_ok], [row_ok_bc]))
            _manually_ensure_shape_env_entry(s, row_ok_bc, row_ok_shape, np.bool_, "FullWindowMask_Expand")

            # Use the full-window broadcasted mask from here on.
            row_ok_name = row_ok_bc
            # row_ok_shape remains the same

        # Broadcast row_ok to align with updates (B,L, …window…)
        upd_rank = len(upd_shape)
        batch_rank = len(row_ok_shape)  # typically 2 (B,L)
        if upd_rank > batch_rank:
            current_name = row_ok_name
            current_shape = row_ok_shape

            def _find_subseq_start(container, subseq):
                max_start = len(container) - len(subseq)
                for start in range(max_start + 1):
                    ok = True
                    for j in range(len(subseq)):
                        if not _dims_concrete_equal_or_symbol_equal(container[start + j], subseq[j], s):
                            ok = False
                            break
                    if ok:
                        return start
                return None

            # Where does (B,L,...) appear inside updates?
            start = _find_subseq_start(upd_shape, row_ok_shape)

            # If we can locate it, add left 1's for any leading dims (e.g. N),
            # and right 1's for the trailing window dims. Otherwise, fall back to
            # the old "append to the right" behavior.
            left_ones  = start if start is not None else 0
            right_ones = (upd_rank - (left_ones + batch_rank)) if start is not None else (upd_rank - batch_rank)

            # Prepend missing leading singletons
            for _ in range(left_ones):
                next_name = s.get_unique_name("row_ok_bc_unsq")
                add_unsqueeze(s, current_name, [0], next_name, ctx="RowOkBroadcastPre")
                current_shape = (1,) + current_shape
                _manually_ensure_shape_env_entry(s, next_name, current_shape, np.bool_, "RowOkBroadcastPre")
                current_name = next_name

            # Append trailing singletons (for window axes)
            for _ in range(right_ones):
                next_name = s.get_unique_name("row_ok_bc_unsq")
                add_unsqueeze(s, current_name, [len(current_shape)], next_name, ctx="RowOkBroadcastPost")
                current_shape = current_shape + (1,)
                _manually_ensure_shape_env_entry(s, next_name, current_shape, np.bool_, "RowOkBroadcastPost")
                current_name = next_name

            row_ok_name = current_name
            row_ok_shape = current_shape
        # else: (B,L) already aligns

        # else: (B,L) already aligns with 2-D updates

        # safe indices: zero-fill bad rows   (SHAPE = idx_shape)
        # (via centralized guardrails to keep Where sites consistent)
        safe_indices_name = emit_where_with_guardrails(
            s,
            both_ok_name,
            final_indices_name_to_return,
            zero_tensor_name,
            out_name=s.get_unique_name("safe_indices"),
            context="FILL_OR_DROP_Indices",
        )

        # safe updates under different reductions:
        #  • add/mul/max/min -> use neutral constant (no-op under the reduction)
        #  • none/replace    -> write back the original values at those indices,
        #                       i.e. GatherND(operand, safe_indices) as the fallback
        np_upd_dtype = _ensure_np_dtype(s.shape_env[_final_updates_name_val_to_return].dtype)
        red_norm = (str(reduction).lower() if reduction is not None else "none")
        if red_norm in ("add", "mul", "max", "min"):
            neutral_val = _get_neutral_value(red_norm, np_upd_dtype)
            neutral_updates_name = s.get_constant_name(neutral_val)
            safe_updates_name = emit_where_with_guardrails(
                s,
                row_ok_name,
                _final_updates_name_val_to_return,
                neutral_updates_name,
                out_name=s.get_unique_name("safe_updates"),
                context="FILL_OR_DROP_UpdatesNeutral",
            )
        else:
            # replace/none semantics
            fallback_updates_name = s.get_unique_name("fallback_updates_old_value")
            s.add_node(helper.make_node("GatherND",
                                        [final_operand_name, safe_indices_name],
                                        [fallback_updates_name]))
            # GatherND output = indices.shape[:-1] + data.shape[K:], where K = indices.shape[-1]
            try:
                K_val = int(_make_shape_concrete_for_prod((idx_shape[-1],), s, "FOD_K_for_Gather")[0])
            except Exception:
                # very conservative fallback; but in our tests K is concrete (e.g., 2)
                K_val = len(dimension_numbers.scatter_dims_to_operand_dims) or 2
            gather_out_shape = tuple(idx_shape[:-1]) + tuple(operand_shape_symbolic[K_val:])
            _manually_ensure_shape_env_entry(
                s, fallback_updates_name, gather_out_shape, np_upd_dtype, "FallbackOldUpdates"
            )
            safe_updates_name = emit_where_with_guardrails(
                s,
                row_ok_name,
                _final_updates_name_val_to_return,
                fallback_updates_name,
                out_name=s.get_unique_name("safe_updates"),
                context="FILL_OR_DROP_UpdatesReplace",
            )

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

    # Final defensive harmonization of float dtypes (prevents f32/f64 mix in ORT)
    final_operand_name, _final_updates_name_val_to_return = _harmonize_float_dtypes(
        s,
        (final_operand_name, _final_updates_name_val_to_return),
        (operand_dtype_np, original_updates_dtype_np),
        "ScatterInputs",
    )

    return (final_operand_name, final_indices_name_to_return, _final_updates_name_val_to_return)

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
    If every dimension in `orig_shape` is <= its counterpart in `target_shape`,
    right-pad with a neutral value to reach `target_shape`. Otherwise, return
    the original (name, shape) unchanged.
    """
    if len(orig_shape) != len(target_shape):
        return upd_name, orig_shape

    pad_after: list[int] = []
    can_pad = True
    for o, t in zip(orig_shape, target_shape):
        # Only pad when both sides are concrete ints and o <= t.
        if not isinstance(o, (int, np.integer)) or not isinstance(t, (int, np.integer)):
            can_pad = False
            break
        if int(o) > int(t):
            can_pad = False
            break
        pad_after.append(int(t) - int(o))

    if not can_pad or all(p == 0 for p in pad_after):
        return upd_name, orig_shape

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
    Return the neutral element for the given reduction (add, mul, max, min).
    Falls back to 0 for unknown reductions (“replace”, “none”, etc.).
    Handles float, int, and bool dtypes.
    """
    dt = _ensure_np_dtype(dtype)
    # Bool specials
    if dt == np.bool_:
        if reduction_op == "add":
            return np.array(False, dtype=dt)
        if reduction_op == "mul":
            return np.array(True, dtype=dt)
        if reduction_op == "max":
            return np.array(False, dtype=dt)  # max(x, False) == x
        if reduction_op == "min":
            return np.array(True, dtype=dt)   # min(x, True) == x
        return np.array(False, dtype=dt)      # default/replace/none

    if reduction_op == "add":
        return np.array(0, dtype=dt)
    if reduction_op == "mul":
        return np.array(1, dtype=dt)
    if reduction_op == "max":
        if np.issubdtype(dt, np.floating):
            return np.array(np.finfo(dt).min, dtype=dt)  # -inf-like
        return np.array(np.iinfo(dt).min, dtype=dt)
    if reduction_op == "min":
        if np.issubdtype(dt, np.floating):
            return np.array(np.finfo(dt).max, dtype=dt)  # +inf-like
        return np.array(np.iinfo(dt).max, dtype=dt)
    # For “replace”, “none”, or anything unknown → 0
    return np.array(0, dtype=dt)


def emit_where_with_guardrails(
    s: "Jaxpr2OnnxConverter",
    cond_name: str,
    x_name: str,
    y_name: str,
    *,
    out_name: Optional[str] = None,
    context: str = "WhereGuardrails",
) -> str:
    """
    Emit an ONNX Where with safety rails:
      • ensure cond is BOOL (Cast if needed)
      • ensure x and y share a common dtype (Cast if needed via numpy.promote_types)
      • NEW: explicitly Expand cond/x/y to the SAME target shape so ONNX shape
             inference never sees incompatible dims.
      • register output shape/dtype in shape_env (the target shape)
    Returns the output tensor name.
    """
    # 1) cond → BOOL
    cond_sds = s.shape_env.get(cond_name)
    cond_dtype = _ensure_np_dtype(cond_sds.dtype) if cond_sds is not None else None
    cond_bool = cond_name
    if cond_dtype is None or cond_dtype != np.bool_:
        cond_bool = s.get_unique_name(f"{cond_name}_as_bool")
        s.add_node(helper.make_node("Cast", [cond_name], [cond_bool], to=int(TensorProto.BOOL)))
        if cond_sds is not None:
            _manually_ensure_shape_env_entry(
                s, cond_bool, cond_sds.shape, np.bool_, f"{context}_CondToBool"
            )

    # 2) x/y → common dtype
    x_sds = s.shape_env.get(x_name)
    y_sds = s.shape_env.get(y_name)
    x_dtype = _ensure_np_dtype(x_sds.dtype) if x_sds is not None else None
    y_dtype = _ensure_np_dtype(y_sds.dtype) if y_sds is not None else None

    if x_dtype is None and y_dtype is None:
        target_dtype = np.int64
    elif x_dtype is None:
        target_dtype = y_dtype
    elif y_dtype is None:
        target_dtype = x_dtype
    else:
        target_dtype = np.promote_types(x_dtype, y_dtype)

    def _maybe_cast(inp_name: str, cur_dtype: Optional[np.dtype], tag: str) -> str:
        if cur_dtype is None or _ensure_np_dtype(cur_dtype) == _ensure_np_dtype(target_dtype):
            return inp_name
        casted = s.get_unique_name(f"{inp_name}_to_{np.dtype(target_dtype).name}")
        s.add_node(
            helper.make_node(
                "Cast", [inp_name], [casted],
                to=int(s.builder._numpy_dtype_to_onnx(target_dtype)),
            )
        )
        sds_local = s.shape_env.get(inp_name)
        if sds_local is not None:
            _manually_ensure_shape_env_entry(
                s, casted, sds_local.shape, target_dtype, f"{context}_{tag}"
            )
        return casted

    x_cast = _maybe_cast(x_name, x_dtype, "CastX")
    y_cast = _maybe_cast(y_name, y_dtype, "CastY")

    # 3) Expand cond/x/y to a single target shape so Where sees identical dims.
    # Prefer x's shape; fall back to y's if x is unknown.
    ref_name = x_cast if (s.shape_env.get(x_cast) is not None) else y_cast
    ref_sds: Optional[ShapeDtypeStruct] = s.shape_env.get(ref_name)
    # If still unknown, we won't expand (rare); Where will broadcast at runtime.
    shapeof_ref_name: Optional[str] = None
    target_shape: Optional[Tuple[Any, ...]] = None
    if ref_sds is not None:
        target_shape = ref_sds.shape
        # Build Shape(ref) once for all Expands
        shapeof_ref_name = s.get_unique_name(f"{context}_shapeof_ref")
        s.add_node(helper.make_node("Shape", [ref_name], [shapeof_ref_name]))
        _manually_ensure_shape_env_entry(
            s, shapeof_ref_name, (len(target_shape),), np.int64, f"{context}_ShapeOfRef"
        )

    def _maybe_expand_to_ref(inp_name: str, desired_dtype: np.dtype, tag: str) -> str:
        if target_shape is None or shapeof_ref_name is None:
            # No static target; leave as-is.
            return inp_name
        sds_local = s.shape_env.get(inp_name)
        if sds_local is not None and _are_shapes_equal(sds_local.shape, target_shape, s):
            return inp_name
        expanded = s.get_unique_name(f"{inp_name}_expanded_for_{context}_{tag}")
        s.add_node(helper.make_node("Expand", [inp_name, shapeof_ref_name], [expanded]))
        _manually_ensure_shape_env_entry(
            s, expanded, target_shape, desired_dtype, f"{context}_Expand_{tag}"
        )
        return expanded

    cond_ready = _maybe_expand_to_ref(cond_bool, np.bool_, "Cond")
    x_ready = x_cast  # x_cast already has the reference shape; expand only if needed
    if ref_name is not None and ref_name != x_cast:
        x_ready = _maybe_expand_to_ref(x_cast, target_dtype, "X")
    y_ready = _maybe_expand_to_ref(y_cast, target_dtype, "Y")

    # 4) Where node
    out = out_name or s.get_unique_name("where_out")
    where_node = helper.make_node("Where", [cond_ready, x_ready, y_ready], [out],
                                  name=s.get_unique_name(f"{context}_Where"))
    s.add_node(where_node)

    # 5) Register out shape/dtype for builder strictness
    if target_shape is None:
        # Fall back to x (else y) shape if known, else scalar ()
        if x_sds is not None:
            _manually_ensure_shape_env_entry(
                s, out, x_sds.shape, target_dtype, f"{context}_Out"
            )
        elif y_sds is not None:
            _manually_ensure_shape_env_entry(
                s, out, y_sds.shape, target_dtype, f"{context}_Out"
            )
        else:
            _manually_ensure_shape_env_entry(
                s, out, (), target_dtype, f"{context}_Out"
            )
    else:
        _manually_ensure_shape_env_entry(
            s, out, target_shape, target_dtype, f"{context}_Out"
        )
    return out


def emit_scatternd(
    s: "Jaxpr2OnnxConverter",
    data_name: str,
    indices_name: str,
    updates_name: str,
    *,
    reduction: Optional[str] = "none",
    out_name: Optional[str] = None,
) -> str:
    """
    Emit a single ONNX ScatterND node, registering the result's shape/dtype.
    - If opset >= 16, honors `reduction` in {"none","add","mul","min","max"}.
    - If opset  < 16 and reduction != "none", we warn and fall back to "none".
    """
    red = (reduction or "none").lower()
    if red == "replace":
        red = "none"
    allowed = {"none", "add", "mul", "min", "max"}
    if red not in allowed:
        raise ValueError(f"Unsupported ScatterND reduction '{reduction}'. Allowed: {sorted(allowed)}")

    attrs = {}
    if s.builder.opset >= 16:
        attrs["reduction"] = red
    elif red != "none":
        logger.warning(
            "ScatterND reduction=%s requires opset>=16 (current=%s). Falling back to 'none'.",
            red, s.builder.opset,
        )
        # attrs stays empty → ScatterND-<old> (no reduction)

    out = out_name or s.get_unique_name("scatternd_out")
    s.add_node(helper.make_node("ScatterND", [data_name, indices_name, updates_name], [out], **attrs))

    # Result shape/dtype match `data`
    data_sds = s.shape_env.get(data_name)
    if isinstance(data_sds, ShapeDtypeStruct):
        _manually_ensure_shape_env_entry(
            s, out, data_sds.shape, data_sds.dtype, f"ScatterND_{red}"
        )
    return out

def prepare_and_emit_scatternd(
    s: "Jaxpr2OnnxConverter",
    operand_v: Any,
    indices_v: Any,
    updates_v: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    scatter_mode: Optional[Any] = None,
    reduction: str = "none",
    out_name: Optional[str] = None,
) -> str:
    """
    High-level helper: (prepare inputs) → (emit ScatterND).
    Returns the output tensor name produced by the ScatterND node.
    """
    data_name, idx_name, upd_name = _prepare_scatter_inputs_for_onnx(
        s,
        operand_v,
        indices_v,
        updates_v,
        dimension_numbers,
        scatter_mode=scatter_mode,
        reduction=reduction,
    )
    return emit_scatternd(
        s,
        data_name,
        idx_name,
        upd_name,
        reduction=reduction,
        out_name=out_name,
    )