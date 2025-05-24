"""
Debug utilities for jax2onnx.

This module contains utilities for debugging JAX to ONNX conversion.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import json
import os
import logging
from dataclasses import dataclass, asdict, field

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


def save_primitive_calls_log(
    log_entries: List[RecordedPrimitiveCallLog], output_file: str
) -> None:
    """
    Save recorded primitive call logs to a JSON file.

    Args:
        log_entries: List of RecordedPrimitiveCallLog objects
        output_file: Path to the output file
    """
    logger.info(f"Saving {len(log_entries)} primitive call records to {output_file}")

    # Convert dataclass objects to dictionaries
    serializable_entries = []
    for entry in log_entries:
        try:
            # Convert dataclass to dict
            entry_dict = asdict(entry)

            # Make shapes JSON-serializable by converting tuples to lists
            def make_serializable(obj):
                if isinstance(obj, tuple):
                    return list(make_serializable(item) for item in obj)
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                # Handle any other non-serializable types
                return str(obj)

            entry_dict = make_serializable(entry_dict)
            serializable_entries.append(entry_dict)
        except Exception as e:
            logger.error(
                f"Error serializing primitive call log entry: {e}", exc_info=True
            )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Write to file
    try:
        with open(output_file, "w") as f:
            json.dump(serializable_entries, f, indent=2)
        logger.info(f"Successfully saved primitive call log to {output_file}")
    except Exception as e:
        logger.error(
            f"Error writing primitive calls log to {output_file}: {e}", exc_info=True
        )
