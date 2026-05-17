# tests/extra_tests/capability_matrix.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator

from jax2onnx.plugins.plugin_system import (
    EXAMPLE_REGISTRY,
    PLUGIN_REGISTRY,
    import_all_plugins,
)
from jax2onnx.user_interface import to_onnx
from tests import t_generator as tgen

SourceKey = tuple[str, str, str]

FLOAT_ELEM_TYPES: frozenset[int] = frozenset(
    {
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
    }
)


@dataclass(frozen=True)
class ShapeVariant:
    """Optional per-input shape overrides for capability checks."""

    name: str
    input_shapes: Mapping[int, tuple[object, ...]]

    def apply(self, base_shapes: Sequence[Any]) -> list[tuple[object, ...]]:
        shapes = [_normalize_shape(shape) for shape in base_shapes]
        for index, replacement in self.input_shapes.items():
            shapes[index] = tuple(replacement)
        return shapes


@dataclass(frozen=True)
class CapabilityCase:
    """One dtype/shape capability check derived from an existing testcase."""

    id: str
    source: SourceKey
    dtype: Any = jnp.bfloat16
    shape_variant: ShapeVariant | None = None
    numeric: bool = True
    numeric_rtol: float = 5e-2
    numeric_atol: float = 5e-2
    require_all_float_tensors_dtype: bool = True
    expected_elem_type: int = TensorProto.BFLOAT16
    expected_output_elem_types: tuple[int, ...] | None = None
    input_dtype_overrides: Mapping[int, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityResult:
    """Exported model plus the concrete callable/input data used for checks."""

    case: CapabilityCase
    param: Mapping[str, Any]
    callable_obj: Any
    input_specs: tuple[jax.ShapeDtypeStruct, ...]
    eval_inputs: tuple[np.ndarray, ...]
    model: onnx.ModelProto
    inferred_model: onnx.ModelProto


@lru_cache(maxsize=1)
def _all_test_params() -> tuple[dict[str, Any], ...]:
    import_all_plugins()

    entries: list[dict[str, Any]] = []
    for plugin in PLUGIN_REGISTRY.values():
        metadata = getattr(plugin, "metadata", None)
        if metadata and isinstance(metadata, dict):
            entries.append(dict(metadata))
    entries.extend(dict(md) for md in EXAMPLE_REGISTRY.values())

    params: list[dict[str, Any]] = []
    for entry in tgen.extract_from_metadata(entries):
        params.extend(tgen.generate_test_params(entry))
    return tuple(params)


def find_test_param(source: SourceKey) -> dict[str, Any]:
    context, component, testcase = source
    matches = [
        param
        for param in _all_test_params()
        if param.get("context") == context
        and param.get("component") == component
        and param.get("testcase") == testcase
    ]
    if len(matches) != 1:
        found = [
            str(param.get("testcase"))
            for param in _all_test_params()
            if param.get("context") == context and param.get("component") == component
        ]
        raise AssertionError(
            f"Expected exactly one metadata testcase for {source}, found "
            f"{len(matches)}. Available in component: {found}"
        )
    return dict(matches[0])


def export_capability_case(case: CapabilityCase) -> CapabilityResult:
    param = find_test_param(case.source)
    callable_obj = _instantiate_callable_for_dtype(param["callable"], case.dtype)
    input_specs = tuple(_build_input_specs(param, case))
    eval_inputs = tuple(_build_eval_inputs(case, input_specs))

    with _temporary_x64(_dtype_name(case.dtype) == "float64"):
        model = to_onnx(
            callable_obj,
            input_specs,
            model_name=f"capability_{case.id}",
            opset=int(param.get("opset_version", 23)),
            return_mode="proto",
            input_params=param.get("input_params", {}),
            inputs_as_nchw=param.get("inputs_as_nchw"),
            outputs_as_nchw=param.get("outputs_as_nchw"),
            input_names=param.get("input_names"),
            output_names=param.get("output_names"),
        )

    onnx.checker.check_model(model)
    inferred_model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return CapabilityResult(
        case=case,
        param=param,
        callable_obj=callable_obj,
        input_specs=input_specs,
        eval_inputs=eval_inputs,
        model=model,
        inferred_model=inferred_model,
    )


def assert_capability_case(case: CapabilityCase) -> CapabilityResult:
    result = export_capability_case(case)
    assert_public_float_io_dtype(result)
    if case.require_all_float_tensors_dtype:
        assert_all_float_tensors_dtype(result)
    if case.numeric:
        assert_reference_evaluator_matches_jax(result)
    return result


def assert_public_float_io_dtype(result: CapabilityResult) -> None:
    graph = result.inferred_model.graph
    for index, spec in enumerate(result.input_specs):
        if not _is_floating_dtype(spec.dtype):
            continue
        elem_type = graph.input[index].type.tensor_type.elem_type
        assert elem_type == result.case.expected_elem_type, (
            f"{result.case.id}: input {index} exported as "
            f"{TensorProto.DataType.Name(elem_type)}, expected "
            f"{TensorProto.DataType.Name(result.case.expected_elem_type)}"
        )

    expected_outputs = result.case.expected_output_elem_types
    if expected_outputs is None:
        expected_outputs = tuple(
            result.case.expected_elem_type for _ in result.inferred_model.graph.output
        )

    assert len(expected_outputs) == len(result.inferred_model.graph.output)
    for index, (value_info, expected_elem_type) in enumerate(
        zip(result.inferred_model.graph.output, expected_outputs, strict=True)
    ):
        elem_type = value_info.type.tensor_type.elem_type
        assert elem_type == expected_elem_type, (
            f"{result.case.id}: output {index} exported as "
            f"{TensorProto.DataType.Name(elem_type)}, expected "
            f"{TensorProto.DataType.Name(expected_elem_type)}"
        )


def assert_all_float_tensors_dtype(result: CapabilityResult) -> None:
    unexpected: list[tuple[str, int]] = []

    def collect(label: str, elem_type: int) -> None:
        if (
            elem_type in FLOAT_ELEM_TYPES
            and elem_type != result.case.expected_elem_type
        ):
            unexpected.append((label, elem_type))

    def collect_graph(graph: onnx.GraphProto, prefix: str) -> None:
        for collection_name, collection in (
            ("input", graph.input),
            ("output", graph.output),
            ("value_info", graph.value_info),
        ):
            for value_info in collection:
                if not value_info.type.HasField("tensor_type"):
                    continue
                collect(
                    f"{prefix}.{collection_name}:{value_info.name}",
                    value_info.type.tensor_type.elem_type,
                )

        for initializer in graph.initializer:
            collect(f"{prefix}.initializer:{initializer.name}", initializer.data_type)

        for node in graph.node:
            node_name = node.name or node.op_type
            node_prefix = f"{prefix}.node:{node_name}"
            for attr in node.attribute:
                if attr.HasField("t"):
                    collect(f"{node_prefix}.{attr.name}", attr.t.data_type)
                for index, tensor in enumerate(attr.tensors):
                    collect(
                        f"{node_prefix}.{attr.name}[{index}]",
                        tensor.data_type,
                    )
                if attr.HasField("sparse_tensor"):
                    collect(
                        f"{node_prefix}.{attr.name}.values",
                        attr.sparse_tensor.values.data_type,
                    )
                for index, sparse_tensor in enumerate(attr.sparse_tensors):
                    collect(
                        f"{node_prefix}.{attr.name}[{index}].values",
                        sparse_tensor.values.data_type,
                    )
                if attr.HasField("g"):
                    collect_graph(attr.g, f"{node_prefix}.{attr.name}")
                for index, graph_attr in enumerate(attr.graphs):
                    collect_graph(graph_attr, f"{node_prefix}.{attr.name}[{index}]")

    collect_graph(result.inferred_model.graph, "graph")

    assert not unexpected, (
        f"{result.case.id}: all floating tensors must preserve "
        f"{TensorProto.DataType.Name(result.case.expected_elem_type)}, got "
        f"{[(name, TensorProto.DataType.Name(dtype)) for name, dtype in unexpected]}"
    )


def assert_reference_evaluator_matches_jax(result: CapabilityResult) -> None:
    graph = result.inferred_model.graph
    input_names = [value_info.name for value_info in graph.input]
    feed = {
        name: value
        for name, value in zip(input_names, result.eval_inputs, strict=False)
    }
    got_outputs = ReferenceEvaluator(result.inferred_model).run(None, feed)

    jax_inputs = [jnp.asarray(value) for value in result.eval_inputs]
    expected_outputs = _normalize_outputs(result.callable_obj(*jax_inputs))
    assert len(got_outputs) == len(expected_outputs)

    for index, (got, expected) in enumerate(
        zip(got_outputs, expected_outputs, strict=True)
    ):
        expected_arr = np.asarray(expected)
        if _is_floating_dtype(expected_arr.dtype):
            np.testing.assert_allclose(
                np.asarray(got).astype(np.float32),
                expected_arr.astype(np.float32),
                rtol=result.case.numeric_rtol,
                atol=result.case.numeric_atol,
                err_msg=f"{result.case.id}: output {index} mismatch",
            )
        else:
            np.testing.assert_array_equal(
                got,
                expected_arr,
                err_msg=f"{result.case.id}: output {index} mismatch",
            )


def _instantiate_callable_for_dtype(callable_obj: Any, dtype: Any) -> Any:
    if getattr(callable_obj, "__jax2onnx_factory__", False):
        callable_obj = callable_obj.with_dtype(dtype)
    instantiate = getattr(callable_obj, "instantiate", None)
    if callable(instantiate):
        with _temporary_x64(_dtype_name(dtype) == "float64"):
            return instantiate()
    return callable_obj


def _build_input_specs(
    param: Mapping[str, Any], case: CapabilityCase
) -> list[jax.ShapeDtypeStruct]:
    input_shapes = param.get("input_shapes")
    input_dtypes = param.get("input_dtypes")
    input_values = param.get("input_values")

    if input_shapes is not None:
        shapes = (
            case.shape_variant.apply(input_shapes)
            if case.shape_variant is not None
            else [_normalize_shape(shape) for shape in input_shapes]
        )
        if input_dtypes:
            if len(input_dtypes) != len(shapes):
                raise ValueError(
                    f"{case.id}: `input_dtypes` length does not match `input_shapes`."
                )
            return [
                jax.ShapeDtypeStruct(
                    shape,
                    case.input_dtype_overrides.get(
                        index, _capability_dtype(dtype, case.dtype)
                    ),
                )
                for index, (shape, dtype) in enumerate(
                    zip(shapes, input_dtypes, strict=True)
                )
            ]
        return [
            jax.ShapeDtypeStruct(
                shape, case.input_dtype_overrides.get(index, case.dtype)
            )
            for index, shape in enumerate(shapes)
        ]

    if input_values is not None:
        specs: list[jax.ShapeDtypeStruct] = []
        for index, value in enumerate(input_values):
            arr = np.asarray(value)
            dtype = case.input_dtype_overrides.get(
                index, _capability_dtype(arr.dtype, case.dtype)
            )
            specs.append(jax.ShapeDtypeStruct(tuple(arr.shape), dtype))
        return specs

    return []


def _build_eval_inputs(
    case: CapabilityCase,
    input_specs: Sequence[jax.ShapeDtypeStruct],
) -> list[np.ndarray]:
    seed = int(hashlib.sha256(case.id.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    arrays: list[np.ndarray] = []
    for spec in input_specs:
        shape = tuple(_concrete_dim(dim) for dim in spec.shape)
        dtype = np.dtype(spec.dtype)
        if _is_floating_dtype(dtype):
            raw = rng.standard_normal(size=shape) * 0.25
            arrays.append(np.asarray(raw, dtype=np.float32).astype(dtype))
        elif np.issubdtype(dtype, np.integer):
            arrays.append(rng.integers(0, 5, size=shape).astype(dtype))
        elif dtype == np.dtype(bool):
            arrays.append((rng.random(size=shape) > 0.5).astype(dtype))
        else:
            raise TypeError(f"{case.id}: unsupported eval dtype {dtype}")
    return arrays


def _normalize_outputs(output: Any) -> list[Any]:
    if isinstance(output, tuple):
        return list(output)
    if isinstance(output, list):
        return output
    return [output]


def _capability_dtype(original_dtype: Any, capability_dtype: Any) -> Any:
    if _is_floating_dtype(original_dtype):
        return capability_dtype
    return original_dtype


def _is_floating_dtype(dtype: Any) -> bool:
    try:
        return bool(jnp.issubdtype(dtype, jnp.floating))
    except TypeError:
        return False


def _dtype_name(dtype: Any) -> str:
    return np.dtype(dtype).name


def _normalize_shape(shape: Any) -> tuple[object, ...]:
    if isinstance(shape, (list, tuple)):
        return tuple(shape)
    return (shape,)


def _concrete_dim(dim: object) -> int:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    return 2


@contextmanager
def _temporary_x64(enabled: bool):
    previous = bool(jax.config.read("jax_enable_x64"))
    if previous != enabled:
        jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        if previous != enabled:
            jax.config.update("jax_enable_x64", previous)
