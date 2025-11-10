# jax2onnx/plugins/jax/lax/gather_compile.py

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import copy
from .gather_helpers import (
    GirInstruction,
    calculate_index_shape,
    check_uniform_start_index,
    get_gir_input_shape,
    get_gir_output_shape,
    index_expand_range_gir,
    index_lastdim_gather_to_gir,
    index_reshape,
    index_transpose_to_gir,
    invert_transpose,
    is_gather_nop,
    run_one_pass,
    transpose_to_gir,
)


def lax_gather_to_gir(
    eqn: Any, indices_var_value: Optional[np.ndarray]
) -> list[GirInstruction]:
    """Convert lax.gather equation to normalized GIR."""
    operand = eqn.invars[0]
    indices_var = eqn.invars[1]
    dn = eqn.params["dimension_numbers"]
    slice_sizes = eqn.params["slice_sizes"]
    output_shape = eqn.outvars[0].aval.shape
    operand_shape = operand.aval.shape
    len(output_shape)

    set(range(len(output_shape)))

    # Extract indices shape (drop trailing index axis)
    indices_shape_with_batchdims = list(indices_var.aval.shape[:-1])
    indices_shape = [
        dimshape
        for dimidx, dimshape in enumerate(indices_shape_with_batchdims)
        if dimidx not in set(dn.start_indices_batching_dims)
    ]

    gir_entries: list[GirInstruction] = []

    for dim in range(len(operand_shape)):
        entry = {"dim": dim, "input_size": operand_shape[dim]}

        # Batched dimension
        if dim in dn.operand_batching_dims:
            batch_idx = dn.operand_batching_dims.index(dim)
            entry["mode"] = "batched"
            entry["batch_dim_index"] = batch_idx
            entry["indices_dim"] = dn.start_indices_batching_dims[batch_idx]
            entry["target_dimensions_shape"] = [operand_shape[dim]]

        # Gathered dimension (collapsed)
        elif dim in dn.collapsed_slice_dims:
            entry["mode"] = "gather"
            indices_var_idx = dn.start_index_map.index(dim)
            entry["indices_var_index"] = indices_var_idx
            entry["target_dimensions_shape"] = indices_shape

        # Passthrough or range_slice
        else:
            if slice_sizes[dim] == operand_shape[dim]:  # whole slice -> passthrough
                entry["mode"] = "passthrough"
            else:  # partial slice, sometimes uses the start_indices too!
                entry["mode"] = "range_slice"
                # Dynamic start from indices_var
                if dim in dn.start_index_map:
                    indices_var_idx = dn.start_index_map.index(dim)
                    if indices_var_value is not None and check_uniform_start_index(
                        indices_var_value, indices_var_idx
                    ):
                        entry["start_offset_value"] = int(
                            np.reshape(indices_var_value[..., indices_var_idx], (-1,))[
                                0
                            ]
                        )  # so we know the static uniform start index, this means this is a simple slice operation
                    else:
                        entry["mode"] = "dynamic_range_slice"
                    entry["start_indices_var_index"] = (
                        indices_var_idx  # may be needed to remove the dimension from the constant indices
                    )
                entry["size"] = slice_sizes[dim]
            entry["target_dimensions_shape"] = [slice_sizes[dim]]

        gir_entries.append(entry)

    slice_dims = [
        dim
        for dim in gir_entries
        if dim["mode"] in ["passthrough", "range_slice", "dynamic_range_slice"]
    ]
    gather_dims = [dim for dim in gir_entries if dim["mode"] in ["gather"]]

    assert len(slice_dims) == len(dn.offset_dims)
    used_dims = set()

    output_dims_from_input_dims = [[] for _ in output_shape]

    for i, dim in enumerate(slice_dims):
        used_dims.add(dn.offset_dims[i])
        output_dims_from_input_dims[dn.offset_dims[i]] = [dim]

    indices_dims_from_input_dims = [[] for _ in indices_var.aval.shape[:-1]]

    for operand_dim_index, indices_dim_index in zip(
        dn.operand_batching_dims, dn.start_indices_batching_dims
    ):
        indices_dims_from_input_dims[indices_dim_index] = [
            gir_entries[operand_dim_index]
        ]

    for dim_dims in indices_dims_from_input_dims:
        if len(dim_dims) == 0:
            dim_dims.extend(gather_dims)

    for dim_dims in output_dims_from_input_dims:
        if len(dim_dims) == 0:
            dim_dims.extend(indices_dims_from_input_dims.pop(0))

    for i, dim_dims in enumerate(output_dims_from_input_dims):
        for dim in dim_dims:
            dim["target_dimensions"] = list(set(dim.get("target_dimensions", []) + [i]))

    for dim in gir_entries:
        if "target_dimensions" not in dim:
            dim["target_dimensions"] = []

    result_op = {"op": "general_gather", "dims": gir_entries}

    result_ops: list[GirInstruction] = []

    if indices_var_value is not None:
        result_ops.append(
            {
                "op": "index_tensor",
                "value": indices_var_value,
                "shape": list(indices_var_value.shape),
            }
        )

    result_ops.append(result_op)

    return result_ops


def extract_slicing(gir_instr_orig: GirInstruction) -> list[GirInstruction]:
    gir_instr = copy.deepcopy(gir_instr_orig)
    assert gir_instr["op"] == "general_gather"

    index_input_shape, _ = calculate_index_shape(gir_instr_orig)
    new_entries = []
    index_dimensions_to_delete = []
    orig_index_indices = []
    for dim in gir_instr["dims"]:
        entry = {
            "dim": dim["dim"],
            "input_size": dim["input_size"],
            "target_dimensions": [dim["dim"]],
        }
        new_entries.append(entry)

        if "indices_var_index" in dim:
            orig_index_indices.append(dim["indices_var_index"])

        if "start_indices_var_index" in dim:
            orig_index_indices.append(dim["start_indices_var_index"])

        if dim["mode"] == "range_slice":
            entry["mode"] = "range_slice"
            entry["target_dimensions_shape"] = dim["target_dimensions_shape"]
            if "start_offset_value" in dim:
                entry["start"] = dim["start_offset_value"]
                entry["end"] = dim["size"] + entry["start"]
            else:
                entry["start"] = 0
                entry["end"] = dim["size"]
            dim["mode"] = "passthrough"
            dim["input_size"] = dim["size"]
            if "start_indices_var_index" in dim:
                index_dimensions_to_delete.append(dim["start_indices_var_index"])
                del dim["start_indices_var_index"]
        else:
            entry["mode"] = "passthrough"
            entry["target_dimensions_shape"] = [dim["input_size"]]

    orig_index_indices = sorted(list(set(orig_index_indices)))
    indices_after_deletion = [
        idx for idx in orig_index_indices if idx not in index_dimensions_to_delete
    ]
    index_index_mapping = {
        orig_idx: i for i, orig_idx in enumerate(indices_after_deletion)
    }

    for dim in gir_instr["dims"]:
        if "indices_var_index" in dim:
            dim["indices_var_index"] = index_index_mapping[dim["indices_var_index"]]
        if "start_indices_var_index" in dim:
            dim["start_indices_var_index"] = index_index_mapping[
                dim["start_indices_var_index"]
            ]

    if any(dim["mode"] == "range_slice" for dim in gir_instr_orig["dims"]):
        result_ops = []
        if indices_after_deletion != orig_index_indices:
            result_ops += index_lastdim_gather_to_gir(
                indices_after_deletion, index_input_shape
            )
        result_ops.append({"op": "ONNX_Slice", "dims": new_entries})
        if not is_gather_nop(gir_instr):
            result_ops.append(gir_instr)
        return result_ops
    else:
        return [gir_instr_orig]


def turn_dynamic_range_slice_to_gather(
    gir_instr_orig: GirInstruction,
) -> list[GirInstruction]:
    gir_instr = copy.deepcopy(gir_instr_orig)
    assert gir_instr["op"] == "general_gather"

    index_input_shape, _ = calculate_index_shape(gir_instr_orig)
    dims_to_extend = []
    dim_slice_sizes = []
    slice_target_dimensions = []

    for dim in gir_instr["dims"]:
        if dim["mode"] == "dynamic_range_slice":
            dims_to_extend.append(dim["start_indices_var_index"])
            dim_slice_sizes.append(dim["size"])
            assert len(dim["target_dimensions"]) == 1
            slice_target_dimensions.append(dim["target_dimensions"][0])

    gather_target_dimensions_shape = None
    gather_target_dimensions = None

    for dim in gir_instr["dims"]:
        if dim["mode"] == "gather":
            dim["target_dimensions_shape"] += dim_slice_sizes
            dim["target_dimensions"] += slice_target_dimensions
            assert (
                gather_target_dimensions_shape is None
                or gather_target_dimensions_shape == dim["target_dimensions_shape"]
            )
            assert (
                gather_target_dimensions is None
                or gather_target_dimensions == dim["target_dimensions"]
            )
            gather_target_dimensions_shape = dim["target_dimensions_shape"]
            gather_target_dimensions = dim["target_dimensions"]

    if gather_target_dimensions_shape is None:
        gather_target_dimensions_shape = dim_slice_sizes
        gather_target_dimensions = slice_target_dimensions

    for dim in gir_instr["dims"]:
        if dim["mode"] == "dynamic_range_slice":
            dim["mode"] = "gather"
            dim["indices_var_index"] = dim["start_indices_var_index"]
            del dim["start_indices_var_index"]
            dim["target_dimensions_shape"] = gather_target_dimensions_shape
            dim["target_dimensions"] = gather_target_dimensions
            if "size" in dim:
                del dim["size"]

    result = []

    if len(dims_to_extend) > 0:
        result += index_expand_range_gir(
            dims_to_extend, dim_slice_sizes, index_input_shape
        )

    result.append(gir_instr)

    return result


def normalize_gather_with_transpose(
    gir_instr_orig: GirInstruction,
) -> list[GirInstruction]:
    gir_instr = copy.deepcopy(gir_instr_orig)
    assert gir_instr["op"] == "general_gather"

    if all(dim["mode"] != "gather" for dim in gir_instr["dims"]):
        return [gir_instr_orig]

    orig_input_shape = get_gir_input_shape(gir_instr_orig)

    new_entries = [dim for dim in gir_instr["dims"] if dim["mode"] == "batched"]
    new_entries += [dim for dim in gir_instr["dims"] if dim["mode"] == "gather"]
    new_entries += [
        dim for dim in gir_instr["dims"] if dim["mode"] not in ["gather", "batched"]
    ]

    P1 = [dim["dim"] for dim in new_entries]

    for i, dim in enumerate(new_entries):
        dim["dim"] = i

    P2 = []
    gather_added = False
    for dim in new_entries:
        if dim["mode"] == "gather":
            if gather_added:
                do_add = False
            else:
                do_add = True
                gather_added = True
        else:
            do_add = True
        if do_add:
            P2 += dim["target_dimensions"]

    P2inv = invert_transpose(P2)

    for dim in new_entries:
        dim["target_dimensions"] = [P2inv[idx] for idx in dim["target_dimensions"]]

    gir_instr["dims"] = new_entries

    gather_new_output_shape = get_gir_output_shape(gir_instr)

    input_transpose = transpose_to_gir(invert_transpose(P1), orig_input_shape)
    output_transpose = transpose_to_gir(P2, gather_new_output_shape)

    return input_transpose + [gir_instr] + output_transpose


def normalize_gather_index_tensor_with_transpose(
    gir_instr_orig: GirInstruction,
) -> list[GirInstruction]:
    gir_instr = copy.deepcopy(gir_instr_orig)
    assert gir_instr["op"] == "general_gather"

    index_shape, index_index = calculate_index_shape(gir_instr)

    P = invert_transpose(index_index)
    P += [len(P)]

    index_transpose = index_transpose_to_gir(P, index_shape)

    i = 0
    for dim in gir_instr["dims"]:
        if dim["mode"] == "batched":
            dim["indices_dim"] = i
            i += 1

    return index_transpose + [gir_instr]


def reorder_gathered_indices(
    gir_instr_orig: GirInstruction,
) -> list[GirInstruction]:
    gir_instr = copy.deepcopy(gir_instr_orig)
    assert gir_instr["op"] == "general_gather"

    index_shape, _ = calculate_index_shape(gir_instr)

    indices_var_indices = [
        dim["indices_var_index"] for dim in gir_instr["dims"] if dim["mode"] == "gather"
    ]

    result = []

    if indices_var_indices != list(range(len(indices_var_indices))):
        result += index_lastdim_gather_to_gir(indices_var_indices, index_shape)

    result.append(gir_instr)
    return result


def detect_onnx_gather(gir_instr_orig: GirInstruction) -> list[GirInstruction]:
    gir_instr = copy.deepcopy(gir_instr_orig)
    assert gir_instr["op"] == "general_gather"

    if not all(dim["mode"] in ["gather", "passthrough"] for dim in gir_instr["dims"]):
        return [gir_instr]

    gather_dims = [dim for dim in gir_instr["dims"] if dim["mode"] == "gather"]

    if not gather_dims:
        return [gir_instr_orig]

    if len(gather_dims) != 1 and gather_dims[0]["indices_var_index"] == 0:
        return [gir_instr]

    target_dimensions = []

    for dim in gir_instr["dims"]:
        target_dimensions += dim["target_dimensions"]

    result = [gir_instr]

    # check everything is in order without implicit transpose
    if target_dimensions == list(range(len(target_dimensions))):
        gir_instr["op"] = "ONNX_Gather"
        result = (
            index_reshape(
                gather_dims[0]["target_dimensions_shape"] + [1],
                gather_dims[0]["target_dimensions_shape"],
            )
            + result
        )

    return result


def detect_onnx_gather_nd(gir_instr_orig: GirInstruction) -> list[GirInstruction]:
    gir_instr = copy.deepcopy(gir_instr_orig)
    assert gir_instr["op"] == "general_gather"

    prev_mode = None
    # check if we strictly follow the GatherND order: batched dims, gather dims, passthrough dims
    for dim in gir_instr["dims"]:
        if dim["mode"] == "batched" and prev_mode not in [None, "batched"]:
            return [gir_instr]
        elif dim["mode"] == "gather" and prev_mode not in [None, "batched", "gather"]:
            return [gir_instr]
        elif dim["mode"] == "passthrough" and prev_mode not in [
            "gather",
            "passthrough",
        ]:
            return [gir_instr]
        elif dim["mode"] not in ["batched", "gather", "passthrough"]:
            return [gir_instr]
        prev_mode = dim["mode"]

    target_dimensions = []

    has_gather_already = False
    for dim in gir_instr["dims"]:
        if not has_gather_already or dim["mode"] != "gather":
            target_dimensions += dim["target_dimensions"]
        if dim["mode"] == "gather":
            has_gather_already = True

    # check everything is in order without implicit transpose
    if target_dimensions == list(range(len(target_dimensions))):
        gir_instr["op"] = "ONNX_GatherND"

    return [gir_instr]


def fold_constant_index_tensor(gir: list[GirInstruction]) -> list[GirInstruction]:
    index_tensor = None
    result_ops = []
    changed = False

    # this is not strictly necessary if we implement all of these in ONNX-IR because that could do the constant folding.
    # But this also documents what each of these special index operations do on the index_tensor, can be used as a reference for the ONNX-IR
    for instr in gir:
        if instr["op"] == "index_tensor":
            index_tensor = instr["value"]
            changed = True
        elif instr["op"] == "index_lastdim_gather" and index_tensor is not None:
            index_tensor = index_tensor[..., instr["gather_indices"]]
            changed = True
        elif instr["op"] == "index_transpose" and index_tensor is not None:
            index_tensor = np.transpose(index_tensor, instr["numpy_transpose"])
            changed = True
        elif instr["op"] == "index_reshape" and index_tensor is not None:
            index_tensor = np.reshape(index_tensor, instr["output_shape"])
            changed = True
        elif instr["op"] == "index_expand" and index_tensor is not None:
            new_dims_shape = [dim["slice_size"] for dim in instr["new_dims"]]
            indices_var_index = [dim["indices_var_index"] for dim in instr["new_dims"]]
            index_tensor = np.reshape(
                index_tensor,
                tuple(
                    instr["input_shape"][:-1]
                    + [1] * len(new_dims_shape)
                    + instr["input_shape"][-1:]
                ),
            )
            new_parts = np.zeros(
                tuple(
                    [1] * (len(instr["input_shape"]) - 1)
                    + new_dims_shape
                    + instr["input_shape"][-1:]
                )
            )
            for i, size in enumerate(new_dims_shape):
                A = np.reshape(
                    np.arange(size),
                    tuple(
                        [1] * (len(instr["input_shape"]) - 1)
                        + [1] * i
                        + [size]
                        + [1] * (len(new_dims_shape) - 1 - i)
                    ),
                )
                new_parts[..., indices_var_index[i]] = A
            index_tensor = index_tensor + new_parts
            changed = True
        elif (
            instr["op"]
            in ["general_gather", "dynamic_range_slice", "ONNX_Gather", "ONNX_GatherND"]
            and index_tensor is not None
            and changed
        ):
            result_ops.append(
                {
                    "op": "index_tensor",
                    "value": copy.deepcopy(index_tensor),
                    "shape": list(index_tensor.shape),
                }
            )
            result_ops.append(instr)
            changed = False
        else:
            result_ops.append(instr)

    return result_ops


def run_all_passes(gir: list[GirInstruction]) -> list[GirInstruction]:
    gir = run_one_pass(gir, extract_slicing, ["general_gather"])
    gir = run_one_pass(gir, turn_dynamic_range_slice_to_gather, ["general_gather"])
    gir = run_one_pass(gir, normalize_gather_with_transpose, ["general_gather"])
    gir = run_one_pass(
        gir, normalize_gather_index_tensor_with_transpose, ["general_gather"]
    )
    gir = run_one_pass(gir, reorder_gathered_indices, ["general_gather"])
    gir = run_one_pass(gir, detect_onnx_gather, ["general_gather"])
    gir = run_one_pass(gir, detect_onnx_gather_nd, ["general_gather"])

    # for instr in gir:
    #     print(instr["op"])
    #     for dim in instr.get("dims", []):
    #         print("\t", dim)
    #     print("\n")

    # TODO: ONNX_Gather can do some limited levels of transpose, support that in a separate pass, so we don't make extra transposes when we don't have to
    gir = fold_constant_index_tensor(gir)
    return gir


def compile_to_gir(
    eqn: Any, indices_var_value: Optional[np.ndarray]
) -> list[GirInstruction]:
    gir = lax_gather_to_gir(eqn, indices_var_value)
    gir = run_all_passes(gir)
    return gir
