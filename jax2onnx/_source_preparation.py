# jax2onnx/_source_preparation.py

"""Private, explicit pretrace preparation of a finite existing NNX model.

This returns an ordinary configured NNX object with its own native JAX behavior.
It does not infer a numerical policy, transform arbitrary source, enable x64,
register a public mode, or replace the ordinary export validator.
"""

from __future__ import annotations

from typing import Any

from flax import nnx
import jax
import numpy as np


class SourcePreparationError(ValueError):
    """The requested private preparation is outside its finite supported scope."""


def _float64_dtype(value: Any, name: str) -> np.dtype[Any]:
    if value is None:
        raise SourcePreparationError(f"{name} must explicitly request float64")
    try:
        dtype: np.dtype[Any] = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise SourcePreparationError(f"{name} must explicitly request float64") from exc
    if dtype != np.dtype(np.float64):
        raise SourcePreparationError(f"{name} must explicitly request float64")
    return dtype


def prepare_nnx_linear(
    model: nnx.Linear,
    *,
    compute_dtype: Any,
    parameter_dtype: Any | None = None,
) -> nnx.Linear:
    """Clone an actual Linear and explicitly configure FP64 computation/storage.

    Only ordinary FP32/FP64 immutable kernel/bias arrays are prepared. Omitted
    ``parameter_dtype`` retains actual stored values and their declaration.
    Explicit FP64 storage replaces whole cloned Variable values; setting the
    initializer's ``param_dtype`` attribute alone cannot convert stored weights.
    Already-FP64 payloads are never recast through FP32. NNX cloning preserves
    graph aliases, state, Variable metadata and original callables/settings.

    JAX x64 must already be effective. Precision requests and custom promotion/
    dot callables are preserved; native execution and ordinary export still own
    their behavior. Value-setting hooks and incompatible preferred result types
    require separate preparation support rather than implicit reinterpretation.
    """
    computation = _float64_dtype(compute_dtype, "compute_dtype")
    storage = (
        None
        if parameter_dtype is None
        else _float64_dtype(parameter_dtype, "parameter_dtype")
    )
    if not jax.config.x64_enabled:
        raise SourcePreparationError(
            "FP64 preparation requires already-effective JAX x64; configure it before preparation and native execution/export"
        )
    if type(model) is not nnx.Linear:
        raise SourcePreparationError(
            "prepare_nnx_linear requires an actual constructed nnx.Linear; arbitrary modules, subclasses and factories need separate support"
        )
    if (
        type(model.in_features) is not int
        or type(model.out_features) is not int
        or model.in_features <= 0
        or model.out_features <= 0
        or type(model.use_bias) is not bool
        or model.use_bias != (model.bias is not None)
    ):
        raise SourcePreparationError(
            "Linear feature dimensions or bias state are inconsistent"
        )
    preferred = model.preferred_element_type
    try:
        incompatible_preferred = (
            preferred is not None and np.dtype(preferred) != computation
        )
    except (TypeError, ValueError) as exc:
        raise SourcePreparationError(
            "Linear preferred_element_type is not a supported dtype"
        ) from exc
    if incompatible_preferred:
        raise SourcePreparationError(
            "FP64 preparation preserves preferred_element_type; an existing non-FP64 request needs explicit separate configuration"
        )
    if not callable(model.promote_dtype) or not callable(model.dot_general):
        raise SourcePreparationError(
            "Linear promotion and dot callables must remain callable"
        )
    parameters: list[tuple[str, Any, tuple[int, ...]]] = [
        ("kernel", model.kernel, (model.in_features, model.out_features))
    ]
    if model.bias is not None:
        parameters.append(("bias", model.bias, (model.out_features,)))
    for name, variable, shape in parameters:
        if type(variable) is not nnx.Param:
            raise SourcePreparationError(
                f"Linear {name} must be an ordinary exact nnx.Param"
            )
        if storage is not None and variable.get_metadata("on_set_value", ()):
            raise SourcePreparationError(
                f"Linear {name} value-setting hooks require separate preparation support"
            )
        value = variable.get_raw_value()
        if (
            not isinstance(value, jax.Array)
            or tuple(value.shape) != shape
            or np.dtype(value.dtype) not in {np.dtype(np.float32), np.dtype(np.float64)}
        ):
            raise SourcePreparationError(
                f"Linear {name} requires an immutable native FP32/FP64 array with shape {shape}"
            )
    prepared = nnx.clone(model)
    prepared.dtype = computation
    if storage is not None:
        prepared.param_dtype = storage
        for name, original, _ in parameters:
            value = original.get_raw_value()
            # get_raw_value avoids invoking getter hooks during preparation;
            # those metadata hooks remain on the ordinary cloned Variable.
            converted = (
                value if np.dtype(value.dtype) == storage else value.astype(storage)
            )
            getattr(prepared, name).set_value(converted)
    return prepared
