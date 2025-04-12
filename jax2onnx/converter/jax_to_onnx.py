"""
Compatibility module for older imports.

This module provides compatibility with existing code that imports from
jax2onnx.converter.jax_to_onnx. It imports and re-exports the functions
from the new conversion_api module.

This file will be deprecated in a future release, so new code should import
directly from jax2onnx.converter.conversion_api instead.
"""

# Import and re-export the functions from conversion_api
from jax2onnx.converter.conversion_api import (
    prepare_example_args,
    to_onnx,
    analyze_constants,
)

# Add deprecation warning
import warnings

warnings.warn(
    "Importing from jax2onnx.converter.jax_to_onnx is deprecated and will be removed in a "
    "future release. Please import from jax2onnx.converter.conversion_api instead.",
    DeprecationWarning,
    stacklevel=2,
)
