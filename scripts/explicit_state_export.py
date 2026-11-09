# scripts/explicit_state_export.py

"""Small example adapter for explicit FP32 state inputs with ONNX defaults.

The caller supplies ``forward(*data, *state)`` and decides which framework
values share a logical slot. This script does not discover state, infer aliases,
or add an API to jax2onnx. Serialize the returned model with
``onnx.save_model(ir.to_proto(model), path, ...)``; when using external data,
reload the artifact to check its tensor metadata and payloads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx import to_onnx


@dataclass(frozen=True)
class StateSlot:
    """One caller-selected state argument; metadata is opaque to the adapter."""

    name: str
    value: jax.Array | np.ndarray
    metadata: Mapping[str, str]
    allow_unused: bool = False


@dataclass(frozen=True)
class _PreparedSlot:
    name: str
    snapshot: np.ndarray
    spec: jax.ShapeDtypeStruct
    metadata: dict[str, str]
    allow_unused: bool


def _prepare_slot(slot: StateSlot) -> _PreparedSlot:
    if (
        not isinstance(slot.name, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", slot.name) is None
    ):
        raise ValueError(f"State slot name must be a C90 identifier: {slot.name!r}")
    if not isinstance(slot.value, (jax.Array, np.ndarray)):
        raise TypeError(f"State slot {slot.name!r} requires a JAX or NumPy array")
    if np.dtype(slot.value.dtype) != np.dtype(np.float32):
        raise TypeError(f"State slot {slot.name!r} supports only dense FP32 values")
    if not isinstance(slot.allow_unused, bool):
        raise TypeError(f"State slot {slot.name!r} allow_unused must be bool")
    if not isinstance(slot.metadata, Mapping):
        raise TypeError(f"State slot {slot.name!r} metadata must be a mapping")
    properties: dict[str, str] = {}
    for key, value in slot.metadata.items():
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise ValueError(
                f"State slot {slot.name!r} metadata needs nonempty string keys "
                "and string values"
            )
        try:
            key.encode("utf-8")
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"State slot {slot.name!r} metadata must be UTF-8 encodable"
            ) from exc
        properties[key] = value

    # Read weakness before the NumPy snapshot: NumPy does not retain this bit.
    weak_type = slot.value.weak_type if isinstance(slot.value, jax.Array) else False
    spec = jax.ShapeDtypeStruct(slot.value.shape, slot.value.dtype, weak_type=weak_type)
    snapshot = np.array(slot.value, copy=True, order="C")
    if not snapshot.size:
        raise ValueError(f"State slot {slot.name!r} cannot be an empty tensor")
    if snapshot.shape != spec.shape or snapshot.dtype != np.dtype(spec.dtype):
        raise ValueError(f"State slot {slot.name!r} changed shape or dtype during copy")
    return _PreparedSlot(slot.name, snapshot, spec, properties, slot.allow_unused)


def export_with_state(
    forward: Callable[..., Any],
    data_specs: Sequence[jax.ShapeDtypeStruct],
    slots: Sequence[StateSlot],
    *,
    data_input_names: Sequence[str],
    opset: int = 23,
) -> ir.Model:
    """Export explicit state, then attach owned defaults to the same input Values.

    All records and exported targets are checked before attachment. FP32 state
    may be scalar or noncontiguous; empty tensors and other dtypes are outside
    this example's supported matrix. Data specifications pass through unchanged.
    A slot must be consumed or returned unless ``allow_unused`` is explicit.
    This function writes no files and never falls back to captured model state.
    """
    specs = list(data_specs)
    data_names = list(data_input_names)
    if len(specs) != len(data_names):
        raise ValueError("Each data specification needs one data input name")
    if any(not isinstance(spec, jax.ShapeDtypeStruct) for spec in specs):
        raise TypeError("Data specifications must be jax.ShapeDtypeStruct values")
    if any(not isinstance(name, str) or not name for name in data_names):
        raise ValueError("Data input names must be nonempty strings")
    prepared = [_prepare_slot(slot) for slot in slots]
    names = data_names + [slot.name for slot in prepared]
    if len(set(names)) != len(names):
        raise ValueError("Data input and state slot names must be unique")
    all_specs = specs + [slot.spec for slot in prepared]
    model = to_onnx(
        forward,
        inputs=all_specs,
        input_names=names,
        return_mode="ir",
        opset=opset,
    )

    actual_inputs = list(model.graph.inputs)
    actual_names = [value.name for value in actual_inputs]
    if len(set(actual_names)) != len(actual_names):
        raise ValueError("Export produced duplicate graph input names")
    if set(actual_names) != set(names):
        raise ValueError(
            f"Exported graph input names differ: expected {names!r}, got {actual_names!r}"
        )
    if len(set(actual_inputs)) != len(actual_inputs):
        raise ValueError("Exported graph inputs must have distinct IR Values")
    inputs = {value.name: value for value in actual_inputs}
    for slot in prepared:
        value = inputs[slot.name]
        if value.shape != ir.Shape(slot.spec.shape):
            raise ValueError(
                f"Exported input {slot.name!r} shape differs from its specification"
            )
        if value.dtype != ir.DataType.from_numpy(np.dtype(slot.spec.dtype)):
            raise ValueError(
                f"Exported input {slot.name!r} dtype differs from its specification"
            )
        if slot.name in model.graph.initializers or value.const_value is not None:
            raise ValueError(
                f"State slot {slot.name!r} already has an initializer or constant"
            )
        if not slot.allow_unused and not value.uses() and not value.is_graph_output():
            raise ValueError(
                f"State slot {slot.name!r} is unused; set allow_unused explicitly"
            )

    # Prepare every tensor before mutating the newly exported, private model.
    tensors = [ir.tensor(slot.snapshot, name=slot.name) for slot in prepared]
    for slot, tensor in zip(prepared, tensors):
        tensor.metadata_props.update(slot.metadata)
    for slot, tensor in zip(prepared, tensors):
        value = inputs[slot.name]
        value.const_value = tensor
        model.graph.register_initializer(value)

    for slot in prepared:
        value = inputs[slot.name]
        tensor = value.const_value
        if (
            not value.is_graph_input()
            or model.graph.initializers.get(slot.name) is not value
            or tensor is None
            or tensor.name != slot.name
            or tensor.shape != slot.snapshot.shape
            or tensor.dtype != ir.DataType.FLOAT
            or tensor.numpy().tobytes(order="C") != slot.snapshot.tobytes(order="C")
            or dict(tensor.metadata_props) != slot.metadata
        ):
            raise RuntimeError(
                f"State slot {slot.name!r} failed final default validation"
            )
    return model
