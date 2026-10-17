# internal_docs/precision/poc/demo.py

"""Run the precision discussion examples without writing ONNX artifacts."""

import platform
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort

from internal_docs.precision.poc.source_export import (
    RepresentationLimit,
    export_source_dtypes,
)
from jax2onnx import to_onnx


def exact_region(x: jax.Array) -> jax.Array:
    y = x.astype(jnp.float32)
    return (y + 1) - y


def minimum_region(x: jax.Array) -> jax.Array:
    y = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    return (y + 1) - y


def mixed_regions(x: jax.Array) -> tuple[jax.Array, ...]:
    y = x.astype(jnp.float32)
    return x + x, y, -y


def low_precision_regions(x: jax.Array) -> tuple[jax.Array, ...]:
    y = x.astype(jnp.float32)
    selected = y + jnp.float32(1)
    return x + jnp.float16(1), selected, selected.astype(x.dtype)


def wider_sum(x: jax.Array) -> jax.Array:
    return jax.lax.reduce_sum(x.astype(jnp.float64), axes=(0,)).astype(jnp.float32)


def shared_bits(x: jax.Array) -> tuple[jax.Array, ...]:
    y = x.astype(jnp.float32)
    return y, -y, jax.lax.bitcast_convert_type(y, jnp.uint32)


def _run(model: onnx.ModelProto, x: np.ndarray) -> list[np.ndarray]:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return session.run(None, {session.get_inputs()[0].name: x})


def _describe(values: list[np.ndarray]) -> str:
    return "; ".join(f"{value.dtype}: {value.tolist()}" for value in values)


def _graph_summary(model: onnx.ModelProto) -> str:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    types = {
        value.name: onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)
        for value in (
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        )
    }
    return " -> ".join(
        f"{node.op_type}[{','.join(types[name] for name in node.output)}]"
        for node in inferred.graph.node
    )


def _example(
    name: str,
    fn: Callable[..., Any],
    x: np.ndarray,
    *,
    opset: int = 23,
    compare_legacy: bool = False,
) -> None:
    spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
    model = export_source_dtypes(fn, [spec], opset=opset)
    actual = _run(model, x)
    reference = [np.asarray(value) for value in jax.tree.leaves(fn(jnp.asarray(x)))]
    assert len(actual) == len(reference)
    for value, expected in zip(actual, reference, strict=True):
        assert value.dtype == expected.dtype
        np.testing.assert_array_equal(value, expected, strict=True)
    print(f"\n{name} (opset {opset}; JAX/ORT samples agree)")
    print(f"  source: {_describe(actual)}")
    print(f"  nodes:  {_graph_summary(model)}")
    if compare_legacy:
        legacy = to_onnx(fn, [spec], opset=opset, enable_double_precision=True)
        print(f"  legacy enable_double_precision=True: {_describe(_run(legacy, x))}")


def main() -> None:
    print(
        f"Python {platform.python_version()}, JAX {jax.__version__}, "
        f"ONNX {onnx.__version__}, ORT {ort.__version__}"
    )
    print("CPUExecutionProvider; ORT optimizations disabled; one thread")
    print("These samples validate the PoC subset, not general model support.")
    previous = bool(jax.config.x64_enabled)
    try:
        # Model preparation owns this choice; export_source_dtypes does not set it.
        jax.config.update("jax_enable_x64", True)
        _example(
            "Exact FP32 calculation with FP64 input",
            exact_region,
            np.array([1e8], dtype=np.float64),
            compare_legacy=True,
        )
        _example(
            "Model-defined minimum FP32 with FP64 input",
            minimum_region,
            np.array([1e8], dtype=np.float64),
        )
        _example(
            "Same minimum rule with FP16 input",
            minimum_region,
            np.array([2048], dtype=np.float16),
        )
        _example(
            "FP64 and deliberate FP32 results in one model",
            mixed_regions,
            np.array([2**24 + 1, 0.1], dtype=np.float64),
        )
        _example(
            "FP16 work, selected FP32 work, explicit return to FP16",
            low_precision_regions,
            np.array([2048, 1], dtype=np.float16),
        )
        _example(
            "Explicit FP64 reduction with FP32 input/output",
            wider_sum,
            np.array([1e8, 1, -1e8], dtype=np.float32),
        )
        x = np.array([0.1, -0.0, 2**24 + 1], dtype=np.float64)
        try:
            export_source_dtypes(shared_bits, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
        except RepresentationLimit as exc:
            print(f"\nExpected representation limit: {exc}")
        else:
            raise AssertionError("BitCast must not be exported at opset 23")
        _example("Shared FP32 value and bit representation", shared_bits, x, opset=26)
        jax.config.update("jax_enable_x64", False)
        try:
            export_source_dtypes(
                exact_region, [jax.ShapeDtypeStruct((1,), jnp.float64)]
            )
        except ValueError as exc:
            print(f"\nExpected JAX configuration conflict: {exc}")
        else:
            raise AssertionError("Explicit FP64 input must fail with x64 disabled")
    finally:
        jax.config.update("jax_enable_x64", previous)


if __name__ == "__main__":
    main()
