"""
Debugging utilities for jax2onnx.

This module contains utilities for debugging JAX to ONNX conversion,
including recording primitive calls and other diagnostic tools.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass, field
import numpy as np
import datetime


@dataclass
class PrimitiveAvalLog:
    """Log record for a primitive input or output abstract value."""

    index: int
    shape: Tuple[int, ...]
    dtype_str: str


@dataclass
class RecordedPrimitiveCallLog:
    """Log record for a primitive call during JAX to ONNX conversion."""

    # Basic identification information (new format)
    sequence_id: int
    primitive_name: str
    plugin_file_hint: Optional[str] = None
    conversion_context_fn_name: Optional[str] = None

    # Input/output information (new format)
    inputs_aval: List[PrimitiveAvalLog] = field(default_factory=list)
    outputs_aval: List[PrimitiveAvalLog] = field(default_factory=list)

    # Parameters of the primitive
    params: Dict[str, Any] = field(default_factory=dict)
    params_repr: Dict[str, str] = field(default_factory=dict)

    # Legacy fields for backward compatibility
    call_count: int = 0
    function_context: str = ""
    input_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    output_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    input_dtypes: List[str] = field(default_factory=list)
    output_dtypes: List[str] = field(default_factory=list)
    input_samples: Optional[List[Any]] = None
    output_samples: Optional[List[Any]] = None

    def __post_init__(self):
        # For backward compatibility, fill in old fields from new ones if needed
        if not self.call_count and self.sequence_id:
            self.call_count = self.sequence_id

        if not self.function_context and self.conversion_context_fn_name:
            self.function_context = self.conversion_context_fn_name

        if not self.input_shapes and self.inputs_aval:
            self.input_shapes = [aval.shape for aval in self.inputs_aval]
            self.input_dtypes = [aval.dtype_str for aval in self.inputs_aval]

        if not self.output_shapes and self.outputs_aval:
            self.output_shapes = [aval.shape for aval in self.outputs_aval]
            self.output_dtypes = [aval.dtype_str for aval in self.outputs_aval]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the log record to a dictionary for JSON serialization."""
        self.__post_init__()  # Ensure all fields are populated

        result = {
            "sequence_id": self.sequence_id,
            "primitive_name": self.primitive_name,
            "plugin_file_hint": self.plugin_file_hint,
            "conversion_context_fn_name": self.conversion_context_fn_name,
            "inputs_aval": [
                {
                    "index": aval.index,
                    "shape": list(aval.shape),
                    "dtype": aval.dtype_str,
                }
                for aval in self.inputs_aval
            ],
            "outputs_aval": [
                {
                    "index": aval.index,
                    "shape": list(aval.shape),
                    "dtype": aval.dtype_str,
                }
                for aval in self.outputs_aval
            ],
            "params": {
                k: (v if isinstance(v, (int, float, str, bool, list, dict)) else str(v))
                for k, v in self.params.items()
            },
            "params_repr": self.params_repr,
            # Legacy fields
            "call_count": self.call_count,
            "function_context": self.function_context,
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "output_shapes": [list(shape) for shape in self.output_shapes],
            "input_dtypes": self.input_dtypes,
            "output_dtypes": self.output_dtypes,
        }

        # Handle input/output samples if present
        if self.input_samples is not None:
            result["input_samples"] = [
                value.tolist() if isinstance(value, np.ndarray) else str(value)
                for value in self.input_samples
            ]

        if self.output_samples is not None:
            result["output_samples"] = [
                value.tolist() if isinstance(value, np.ndarray) else str(value)
                for value in self.output_samples
            ]

        return result


def write_primitive_call_log(
    recorded_calls: List[RecordedPrimitiveCallLog],
    output_filepath: str,
    function_context: Optional[str] = None,
) -> None:
    """Write a human-readable log of recorded primitive calls to a file."""
    header_lines = []
    header_lines.append("Primitive Call Log")
    header_lines.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
    if function_context:
        header_lines.append(f"Context Function: {function_context}")
    header_lines.append("=" * 60)

    with open(output_filepath, "w") as f:
        # Write header
        for line in header_lines:
            f.write(line + "\n")
        f.write("\n")

        # Write each recorded call
        for record in recorded_calls:
            # Convert to dict to handle both dict and dataclass inputs
            if not isinstance(record, dict):
                record_dict = record.to_dict()
            else:
                record_dict = record

            f.write("-" * 60 + "\n")
            f.write(f"Call ID: {record_dict['sequence_id']}\n")
            f.write(f"Primitive: {record_dict['primitive_name']}\n")
            f.write(f"Plugin Hint: {record_dict.get('plugin_file_hint') or 'N/A'}\n")
            f.write(
                f"Context Function: {record_dict.get('conversion_context_fn_name') or 'N/A'}\n\n"
            )

            # Params
            f.write("Parameters:\n")
            if record_dict.get("params_repr"):
                for k, v_repr in record_dict["params_repr"].items():
                    f.write(f"  - {k}: {v_repr}\n")
            else:
                f.write("  (none)\n")
            f.write("\n")

            # Inputs
            f.write("Inputs Aval (Shape, DType):\n")
            for aval in record_dict.get("inputs_aval", []):
                f.write(
                    f"  - In {aval['index']}: shape={aval['shape']}, dtype={aval['dtype']}\n"
                )
            f.write("\n")

            # Outputs
            f.write("Outputs Aval (Shape, DType):\n")
            for aval in record_dict.get("outputs_aval", []):
                f.write(
                    f"  - Out {aval['index']}: shape={aval['shape']}, dtype={aval['dtype']}\n"
                )
            f.write("\n")

        # Footer
        f.write("=" * 60 + "\n")


def save_primitive_calls_log(
    log_records: List[RecordedPrimitiveCallLog], file_path: str
) -> None:
    """
    Save the primitive calls log to a JSON file.

    Args:
        log_records: List of RecordedPrimitiveCallLog objects to save
        file_path: Path to save the log file
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    # Convert all records to dictionaries
    records_as_dicts = [record.to_dict() for record in log_records]

    with open(file_path, "w") as f:
        json.dump(records_as_dicts, f, indent=2)

    write_primitive_call_log(
        log_records,
        file_path,
        function_context=(
            log_records[0].conversion_context_fn_name if log_records else None
        ),
    )
