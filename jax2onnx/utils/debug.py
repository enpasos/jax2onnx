# jax2onnx/utils/debug.py

"""
Debug utilities for jax2onnx.

This module contains utilities for debugging JAX to ONNX conversion.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import logging
from dataclasses import dataclass, field, is_dataclass, fields
import numpy as np

logger = logging.getLogger("jax2onnx.utils.debug")


@dataclass
class RecordedPrimitiveCallLog:
    """
    Data class for recording information about JAX primitive calls during conversion.
    """

    sequence_id: int
    primitive_name: str
    plugin_file_hint: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    params_repr: str = ""
    inputs_aval: List[Tuple[Tuple[Union[int, Any], ...], str, str]] = field(
        default_factory=list
    )
    outputs_aval: List[Tuple[Tuple[Union[int, Any], ...], str, str]] = field(
        default_factory=list
    )
    conversion_context_fn_name: Optional[str] = None
    # New fields for detailed logging
    inputs_jax_vars: List[str] = field(default_factory=list)
    inputs_onnx_names: List[str] = field(default_factory=list)
    outputs_jax_vars: List[str] = field(default_factory=list)
    outputs_onnx_names: List[str] = field(default_factory=list)

    def __str__(self):
        # Consider updating __str__ if you want these new fields in simple printouts
        # For detailed logging to a file, direct field access is fine.
        # This is a placeholder; actual string formatting depends on desired output.
        input_details = "\n".join(
            f"  - In {i}: aval={self.inputs_aval[i] if self.inputs_aval and i < len(self.inputs_aval) else 'N/A'}, "
            f"jax_var='{self.inputs_jax_vars[i] if self.inputs_jax_vars and i < len(self.inputs_jax_vars) else 'N/A'}', "
            f"onnx_name='{self.inputs_onnx_names[i] if self.inputs_onnx_names and i < len(self.inputs_onnx_names) else 'N/A'}'"
            for i in range(len(self.inputs_aval or []))
        )
        output_details = "\n".join(
            f"  - Out {o}: aval={self.outputs_aval[o] if self.outputs_aval and o < len(self.outputs_aval) else 'N/A'}, "
            f"jax_var='{self.outputs_jax_vars[o] if self.outputs_jax_vars and o < len(self.outputs_jax_vars) else 'N/A'}', "
            f"onnx_name='{self.outputs_onnx_names[o] if self.outputs_onnx_names and o < len(self.outputs_onnx_names) else 'N/A'}'"
            for o in range(len(self.outputs_aval or []))
        )

        return (
            f"------------------------------------------------------------\n"
            f"Call ID: {self.sequence_id}\n"
            f"Primitive: {self.primitive_name}\n"
            f"Plugin Hint: {self.plugin_file_hint or 'N/A'}\n"
            f"Context Function: {self.conversion_context_fn_name or 'N/A'}\n"
            f"Parameters:\n{self.params_repr or '  (none)'}\n"
            f"Inputs:\n{input_details if input_details else '  (none)'}\n"
            f"Outputs:\n{output_details if output_details else '  (none)'}\n"
        )


def _to_jsonable(obj: Any, *, max_array_elems: int = 64) -> Any:
    """
    Best-effort serializer:
    - dataclasses -> dict (without deepcopy), field-by-field
    - numpy arrays -> small arrays tolist(), else summary
    - jaxlib/_jax Traceback and any other non-serializable -> str(obj)
    """
    # primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy scalars
    if isinstance(obj, (np.generic,)):
        return obj.item()

    # numpy arrays
    if isinstance(obj, np.ndarray):
        if obj.size <= max_array_elems:
            return obj.tolist()
        return {"ndarray": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}

    # dataclasses (without dataclasses.asdict → no deepcopy)
    if is_dataclass(obj):
        out = {}
        for f in fields(obj):
            try:
                out[f.name] = _to_jsonable(
                    getattr(obj, f.name), max_array_elems=max_array_elems
                )
            except Exception as e:
                out[f.name] = (
                    f"<unserializable:{type(getattr(obj, f.name)).__name__}> {e}"
                )
        return out

    # mappings
    if isinstance(obj, dict):
        return {
            _to_jsonable(k, max_array_elems=max_array_elems): _to_jsonable(
                v, max_array_elems=max_array_elems
            )
            for k, v in obj.items()
        }

    # sequences
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x, max_array_elems=max_array_elems) for x in obj]

    # fallback: string
    try:
        return str(obj)
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"


def _json_sanitize(obj):
    # Fast path for primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Lists / tuples
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]

    # Dicts
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}

    # Dataclasses
    if is_dataclass(obj):
        out = {}
        for f in fields(obj):
            name = f.name
            try:
                out[name] = _json_sanitize(getattr(obj, name))
            except Exception:
                # Drop unserializable field
                out[name] = f"<unserializable:{name}>"
        return out

    # Numpy/JAX arrays → shape+dtype summary (or .tolist() if you prefer)
    try:
        import numpy as _np

        if isinstance(obj, _np.ndarray):
            return {
                "__ndarray__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
    except Exception:
        pass

    # Fallback: repr (covers jaxlib._jax.Traceback, frames, etc.)
    try:
        return repr(obj)
    except Exception:
        return "<unserializable>"


def save_primitive_calls_log(records, path: str) -> None:
    try:
        sanitized = [_json_sanitize(r) for r in records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, indent=2)
        logger.info(f"Successfully saved primitive call log to {path}")
    except Exception as e:
        logger.error(f"Failed to save primitive call log to {path}: {e}")
