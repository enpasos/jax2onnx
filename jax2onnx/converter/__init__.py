"""
JAX to ONNX Converter

This module provides tools to convert JAX functions to ONNX format,
enabling interoperability between JAX models and ONNX-compatible runtimes.

The main API functions are:
- to_onnx: Converts a JAX function to an ONNX model
- onnx_function: Decorator for marking functions as ONNX functions
- allclose: Checks if JAX and ONNX Runtime outputs are close for validation
"""

from jax2onnx.converter.user_interface import to_onnx, onnx_function, allclose

__all__ = [
    "to_onnx",
    "onnx_function",
    "allclose",
]
