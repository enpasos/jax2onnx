# internal_docs/precision/poc/ordering_kernel_probe.py

"""ROOT-only bare ordering-kernel feasibility; no SOURCE/raw-cell acceptance."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import traceback
from typing import Any


def _error(exc: Exception) -> dict[str, str]:
    return {
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _save(directory: Path, name: str, arrays: tuple[Any, ...], np: Any) -> None:
    np.savez(directory / name, **{f"output_{i}": v for i, v in enumerate(arrays)})


def _array(value: Any) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype),
        "dtype_str": value.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _inputs(np: Any, bfloat16: Any) -> list[tuple[str, Any, list[str]]]:
    # Fixed before native/target calls. Each pair of rows isolates one question.
    families = {
        "finite": [[3, -1, 7, 2, 5], [11, 4, 9, 8, 6]],
        "ties": [[3, 3, 1, 3, 2], [-4, -4, 0, -4, -1]],
        "nan-first": [[float("nan"), -1, 3, 2, 0], [float("nan"), 4, 4, -2, 1]],
        "nan-middle": [[3, -1, float("nan"), 2, 0], [3, 4, float("nan"), -2, 1]],
        "nan-last": [[3, -1, 2, 0, float("nan")], [3, 4, -2, 1, float("nan")]],
        "multiple-nan": [
            [float("nan"), 3, float("nan"), -1, 2],
            [2, float("nan"), 1, float("nan"), 0],
        ],
        "all-nan": [[float("nan")] * 5, [float("nan")] * 5],
        "signed-zero": [[-0.0, 0.0, -0.0, 0.0, -0.0], [0.0, -0.0, 0.0, -0.0, 0.0]],
        "infinity": [
            [float("inf"), 4, float("inf"), -float("inf"), -2],
            [-float("inf"), 0, float("inf"), -float("inf"), 3],
        ],
    }
    result = []
    for name, dtype, word in (
        ("float16", np.float16, np.uint16),
        ("float32", np.float32, np.uint32),
        ("float64", np.float64, np.uint64),
        ("bfloat16", bfloat16, np.uint16),
    ):
        rows, labels = [], []
        for family, values in families.items():
            array = np.asarray(values, dtype=dtype)
            if name == "float64" and family == "finite":
                array += np.arange(1, 11).reshape(2, 5) * 2.0**-35
            rows.extend(array)
            labels.extend([f"{family}-0", f"{family}-1"])
        sign = 1 << (np.dtype(word).itemsize * 8 - 1)
        tiny = np.asarray([1, 2, sign | 1, sign | 2, 0], word).view(dtype)
        rows.extend([tiny, tiny[::-1].copy()])
        labels.extend(["subnormal-0", "subnormal-1"])
        result.append((name, np.asarray(rows, dtype=dtype), labels))
    result.append(
        (
            "bool",
            np.asarray(
                [
                    [False, True, False, True, False],
                    [True, False, True, False, True],
                    [False] * 5,
                    [True] * 5,
                ],
                dtype=np.bool_,
            ),
            ["mixed-0", "mixed-1", "all-false", "all-true"],
        )
    )
    return result


def _kernel(
    onnx: Any, np: Any, value: Any, axis: int, kind: str, opset: int
) -> tuple[Any, list[dict[str, Any]]]:
    dtype = (
        onnx.TensorProto.BFLOAT16
        if str(value.dtype) == "bfloat16"
        else onnx.helper.np_dtype_to_tensor_dtype(value.dtype)
    )
    nodes, initializers, outputs, declarations = [], [], [], []
    if kind.startswith("arg"):
        data = "x"
        if value.dtype == np.dtype(np.bool_):
            data = "bool-as-int64"
            nodes.append(
                onnx.helper.make_node("Cast", ["x"], [data], to=onnx.TensorProto.INT64)
            )
        reduced = "indices64"
        shape = list(value.shape)
        del shape[axis]
        nodes.append(
            onnx.helper.make_node(
                "ArgMax" if kind == "argmax" else "ArgMin",
                [data],
                [reduced],
                axis=axis,
                keepdims=0,
                select_last_index=0,
            )
        )
        nodes.append(
            onnx.helper.make_node(
                "Cast", [reduced], ["indices32"], to=onnx.TensorProto.INT32
            )
        )
        for name, width in (("indices32", 32), ("indices64", 64)):
            outputs.append(
                onnx.helper.make_tensor_value_info(
                    name,
                    onnx.TensorProto.INT32 if width == 32 else onnx.TensorProto.INT64,
                    shape,
                )
            )
            declarations.append({"role": "indices", "width": width})
    else:
        for k in sorted({1, 2, value.shape[axis]}):
            k_name, values_name, indices_name = f"k{k}", f"values{k}", f"indices{k}"
            initializers.append(
                onnx.numpy_helper.from_array(np.asarray([k], np.int64), name=k_name)
            )
            nodes.append(
                onnx.helper.make_node(
                    "TopK",
                    ["x", k_name],
                    [values_name, f"indices64-{k}"],
                    axis=axis,
                    largest=int(kind == "approx-max"),
                    sorted=1,
                )
            )
            nodes.append(
                onnx.helper.make_node(
                    "Cast",
                    [f"indices64-{k}"],
                    [indices_name],
                    to=onnx.TensorProto.INT32,
                )
            )
            shape = list(value.shape)
            shape[axis] = k
            outputs.extend(
                [
                    onnx.helper.make_tensor_value_info(values_name, dtype, shape),
                    onnx.helper.make_tensor_value_info(
                        indices_name, onnx.TensorProto.INT32, shape
                    ),
                ]
            )
            declarations.extend(
                [{"role": "values", "k": k}, {"role": "indices", "k": k}]
            )
    graph = onnx.helper.make_graph(
        nodes,
        "bare-ordering-feasibility",
        [onnx.helper.make_tensor_value_info("x", dtype, value.shape)],
        outputs,
        initializer=initializers,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", opset)],
        producer_name="bare-kernel-feasibility-only",
        ir_version=10,
    )
    return model, declarations


def _bfloat16_target(
    session: Any, value: Any, ort: Any, onnx: Any, np: Any
) -> tuple[Any, ...]:
    # Reinterpret the same BF16 words; ONNX receives BF16 rather than UINT16.
    words = np.ascontiguousarray(value).view(np.uint16)
    input_value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
        words, onnx.TensorProto.BFLOAT16
    )
    assert input_value.is_tensor() and input_value.device_name() == "cpu"
    assert input_value.data_type() == "tensor(bfloat16)"
    assert input_value.element_type() == onnx.TensorProto.BFLOAT16
    assert tuple(input_value.shape()) == value.shape
    assert input_value.tensor_size_in_bytes() == value.nbytes
    assert words.tobytes(order="C") == value.tobytes(order="C")
    expected_inputs = session.get_inputs()
    assert len(expected_inputs) == 1
    assert expected_inputs[0].name == "x"
    assert expected_inputs[0].type == "tensor(bfloat16)"
    assert tuple(expected_inputs[0].shape) == value.shape
    returned = session.run_with_ort_values(None, {"x": input_value})
    specifications = session.get_outputs()
    assert len(returned) == len(specifications)
    arrays = []
    for output, specification in zip(returned, specifications, strict=True):
        assert output.is_tensor() and output.device_name() == "cpu"
        assert output.data_type() == specification.type
        shape = tuple(output.shape())
        assert shape == tuple(specification.shape)
        assert all(type(extent) is int and extent >= 0 for extent in shape)
        if output.element_type() == onnx.TensorProto.BFLOAT16:
            assert specification.type == "tensor(bfloat16)"
            size = math.prod(shape) * np.dtype(np.uint16).itemsize
            assert output.tensor_size_in_bytes() == size
            pointer = output.data_ptr()
            assert not size or pointer > 0
            # Own a CPU copy before the OrtValue/session can release storage.
            payload = ctypes.string_at(pointer, size) if size else b""
            array = np.frombuffer(payload, dtype=np.uint16).copy().reshape(shape)
            array = array.view(value.dtype)
        else:
            array = output.numpy().copy()
            assert array.shape == shape
            assert (
                onnx.helper.np_dtype_to_tensor_dtype(array.dtype)
                == output.element_type()
            )
            assert output.tensor_size_in_bytes() == array.nbytes
        arrays.append(array)
    assert words.tobytes(order="C") == value.tobytes(order="C")
    return tuple(arrays)


def _compare(actual: Any, reference: Any, np: Any) -> dict[str, Any]:
    try:
        dtype, shape = actual.dtype == reference.dtype, actual.shape == reference.shape
        equal = (
            dtype
            and shape
            and actual.tobytes(order="C") == reference.tobytes(order="C")
        )
        result: dict[str, Any] = {
            "status": "VALIDATED" if equal else "FAILED",
            "dtype_equal": dtype,
            "shape_equal": shape,
        }
        if dtype and shape and not equal:
            words = {2: np.uint16, 4: np.uint32, 8: np.uint64}[actual.dtype.itemsize]
            result["mismatch_coordinates"] = np.argwhere(
                actual.view(words) != reference.view(words)
            ).tolist()
        return result
    except Exception as exc:
        return {"status": "FAILED", "exception": _error(exc)}


def _association(
    value: Any, outputs: tuple[Any, ...], axis: int, np: Any
) -> list[dict[str, Any]]:
    result = []
    for i in range(0, len(outputs), 2):
        selected, indices = outputs[i : i + 2]
        valid = bool(np.all((indices >= 0) & (indices < value.shape[axis])))
        entry: dict[str, Any] = {
            "output_pair": i // 2,
            "status": "FAILED",
            "indices_in_range": valid,
        }
        if valid:
            entry.update(
                _compare(selected, np.take_along_axis(value, indices, axis=axis), np)
            )
        result.append(entry)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--opset", type=int, choices=(23, 26), default=23)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    # Framework imports and every execution below are exclusively ROOT-owned.
    import jax
    import ml_dtypes
    import numpy as np
    import onnx
    import onnxruntime as ort

    versions = {
        "jax": jax.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
    }
    frozen = _inputs(np, ml_dtypes.bfloat16)
    criterion = {
        "mode": "exact dtype, shape and component/index bytes",
        "topk_association": "selected bytes equal original input at each returned index",
        "options": {
            "recall_target": 0.95,
            "aggregate_to_topk": True,
            "reduction_input_size_override": -1,
        },
        "k": [1, 2, "full extent"],
        "opset": args.opset,
        "scope": "bare kernels only; no SOURCE export, raw-cell replay or provider-wide acceptance",
        "native_x64": True,
    }
    (args.out / "criterion-before-execution.json").write_text(
        json.dumps(criterion, indent=2, sort_keys=True)
    )
    captures = []
    with jax.enable_x64(True):
        for width, matrix, families in frozen:
            for axis in (0, 1):
                value = matrix.T.copy() if axis == 0 else matrix.copy()
                original = value.tobytes(order="C")
                for kind in (
                    ("argmin", "argmax")
                    if width == "bool"
                    else ("argmin", "argmax", "approx-min", "approx-max")
                ):
                    directory = args.out / f"{width}-axis{axis}-{kind}"
                    directory.mkdir()
                    np.savez(directory / "input-before-native.npz", x=value)
                    entry: dict[str, Any] = {
                        "case": directory.name,
                        "axis": axis,
                        "families_on_nonreduced_axis": families,
                        "input": _array(value),
                        "native": {},
                        "target": {},
                    }
                    captures.append((entry, value, original, directory))

                    def native(x: Any) -> Any:
                        if kind.startswith("arg"):
                            fn = jax.lax.argmax if kind == "argmax" else jax.lax.argmin
                            return fn(x, axis=axis, index_dtype=np.int32), fn(
                                x, axis=axis, index_dtype=np.int64
                            )
                        fn = (
                            jax.lax.approx_max_k
                            if kind == "approx-max"
                            else jax.lax.approx_min_k
                        )
                        return tuple(
                            part
                            for k in sorted({1, 2, value.shape[axis]})
                            for part in fn(
                                x,
                                k=k,
                                reduction_dimension=axis,
                                recall_target=0.95,
                                aggregate_to_topk=True,
                                reduction_input_size_override=-1,
                            )
                        )

                    def fresh(x: Any) -> Any:
                        return native(x)

                    for role in ("eager", "fresh-jit"):
                        record: dict[str, Any] = {"status": "FAILED", "arrays": ()}
                        entry["native"][role] = record
                        try:
                            output = (
                                native(value)
                                if role == "eager"
                                else jax.jit(fresh)(value)
                            )
                            record.update(
                                status="RETURNED",
                                arrays=tuple(
                                    np.asarray(v).copy()
                                    for v in jax.tree.leaves(output)
                                ),
                            )
                        except Exception as exc:
                            record["exception"] = _error(exc)
                        _save(directory, f"native-{role}.npz", record["arrays"], np)
                    blob = None
                    try:
                        model, declarations = _kernel(
                            onnx, np, value, axis, kind, args.opset
                        )
                        blob = model.SerializeToString()
                        (directory / "kernel.onnx").write_bytes(blob)
                        entry.update(
                            artifact_sha256=hashlib.sha256(blob).hexdigest(),
                            declarations=declarations,
                        )
                        entry["kernel-construction"] = {"status": "VALIDATED"}
                    except Exception as exc:
                        entry["kernel-construction"] = {
                            "status": "FAILED",
                            "exception": _error(exc),
                        }
                    if blob is None:
                        entry["schema"] = {
                            "status": "UNVERIFIED",
                            "reason": "kernel construction failed",
                        }
                    else:
                        try:
                            onnx.checker.check_model(model, full_check=True)
                            entry["schema"] = {"status": "VALIDATED"}
                        except Exception as exc:
                            entry["schema"] = {
                                "status": "FAILED",
                                "exception": _error(exc),
                            }
                    # A schema failure never suppresses either provider attempt.
                    for mode in ("disabled", "all"):
                        record = {"status": "FAILED", "arrays": ()}
                        entry["target"][mode] = record
                        try:
                            if blob is None:
                                raise RuntimeError(
                                    "Target unavailable: kernel construction failed"
                                )
                            options = ort.SessionOptions()
                            options.intra_op_num_threads = (
                                options.inter_op_num_threads
                            ) = 1
                            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                            options.optimized_model_filepath = str(
                                directory / f"{mode}-optimized.onnx"
                            )
                            options.graph_optimization_level = (
                                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                                if mode == "disabled"
                                else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                            )
                            session = ort.InferenceSession(
                                blob, options, providers=["CPUExecutionProvider"]
                            )
                            session.disable_fallback()
                            record["providers"] = session.get_providers()
                            record.update(
                                status="RETURNED",
                                arrays=(
                                    _bfloat16_target(session, value, ort, onnx, np)
                                    if width == "bfloat16"
                                    else tuple(
                                        np.asarray(v).copy()
                                        for v in session.run(None, {"x": value})
                                    )
                                ),
                            )
                        except Exception as exc:
                            record["exception"] = _error(exc)
                        _save(directory, f"target-{mode}.npz", record["arrays"], np)

    def document() -> dict[str, Any]:
        return {
            "versions": versions,
            "criterion": criterion,
            "cases": [
                {
                    **entry,
                    "native": {
                        role: {
                            **{k: v for k, v in record.items() if k != "arrays"},
                            "outputs": [_array(v) for v in record["arrays"]],
                        }
                        for role, record in entry["native"].items()
                    },
                    "target": {
                        mode: {
                            **{k: v for k, v in record.items() if k != "arrays"},
                            "outputs": [_array(v) for v in record["arrays"]],
                        }
                        for mode, record in entry["target"].items()
                    },
                }
                for entry, _, _, _ in captures
            ],
        }

    # Persist every native/provider observation before comparisons or guards.
    (args.out / "all-captures-before-comparison.json").write_text(
        json.dumps(document(), indent=2, sort_keys=True)
    )
    failures = 0
    for case_index, (entry, value, original, directory) in enumerate(captures):
        checks = []
        checks.append(
            {
                "check": "original-input-bytes",
                "status": (
                    "VALIDATED" if value.tobytes(order="C") == original else "FAILED"
                ),
            }
        )
        eager, compiled = (entry["native"][role] for role in ("eager", "fresh-jit"))
        if eager["status"] == compiled["status"] == "RETURNED":
            if len(eager["arrays"]) != len(compiled["arrays"]):
                checks.append({"check": "native-output-count", "status": "FAILED"})
            else:
                checks.extend(
                    {
                        "check": "native-eager-vs-fresh-jit",
                        "output": i,
                        **_compare(a, b, np),
                    }
                    for i, (a, b) in enumerate(
                        zip(eager["arrays"], compiled["arrays"], strict=True)
                    )
                )
        for mode, target in entry["target"].items():
            for role, reference in entry["native"].items():
                if target["status"] != "RETURNED" or reference["status"] != "RETURNED":
                    checks.append(
                        {
                            "check": "target-vs-native",
                            "mode": mode,
                            "reference": role,
                            "status": "FAILED",
                            "reason": "execution did not return comparable arrays",
                        }
                    )
                    continue
                if len(target["arrays"]) != len(reference["arrays"]):
                    checks.append(
                        {
                            "check": "target-vs-native-output-count",
                            "mode": mode,
                            "reference": role,
                            "status": "FAILED",
                        }
                    )
                else:
                    checks.extend(
                        {
                            "check": "target-vs-native",
                            "mode": mode,
                            "reference": role,
                            "output": i,
                            **_compare(a, b, np),
                        }
                        for i, (a, b) in enumerate(
                            zip(target["arrays"], reference["arrays"], strict=True)
                        )
                    )
        if "approx" in entry["case"]:
            for role, record in (*entry["native"].items(), *entry["target"].items()):
                if record["status"] == "RETURNED":
                    try:
                        checks.extend(
                            {"check": "index-value-association", "role": role, **item}
                            for item in _association(
                                value, record["arrays"], entry["axis"], np
                            )
                        )
                    except Exception as exc:
                        checks.append(
                            {
                                "check": "index-value-association",
                                "role": role,
                                "status": "FAILED",
                                "exception": _error(exc),
                            }
                        )
        entry["checks"] = checks
        entry["status"] = (
            "FAILED"
            if entry["kernel-construction"]["status"] == "FAILED"
            or entry["schema"]["status"] == "FAILED"
            or any(c["status"] == "FAILED" for c in checks)
            else "VALIDATED_SELECTED_KERNEL_SAMPLE_ONLY"
        )
        failures += entry["status"] == "FAILED"
        (directory / "result.json").write_text(
            json.dumps(document()["cases"][case_index], indent=2, sort_keys=True)
        )
    summary = document()
    summary.update(
        status="FAILED" if failures else "VALIDATED_SELECTED_KERNEL_SAMPLES_ONLY",
        failed_cases=failures,
        total_cases=len(captures),
        source_exported=False,
        original_raw_cells_executed=False,
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(
        json.dumps(
            {
                "summary": str(args.out / "summary.json"),
                "cases": len(captures),
                "failed": failures,
                "status": summary["status"],
            }
        ),
        flush=True,
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
