# jax2onnx/converter/source_representation.py

"""Declared logical values and their selected ONNX storage representation.

Complex values use the existing real/imaginary channel representation. Ordinary
values keep their exact dtype and shape, including real tensors ending in two.
This module describes storage; it does not admit primitives or prove numerics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import onnx_ir as ir


@dataclass(frozen=True)
class ValueRepresentation:
    logical_dtype: np.dtype[Any]
    logical_shape: tuple[object, ...]
    encoding: Literal["identity", "complex-real-pair"]
    boundary_layout: Literal["NCHW"] | None = None
    physical_dtype: np.dtype[Any] = field(init=False)
    physical_shape: tuple[object, ...] = field(init=False)

    def __post_init__(self) -> None:
        dtype = np.dtype(self.logical_dtype)
        shape = tuple(self.logical_shape)
        if self.encoding == "complex-real-pair":
            if dtype == np.dtype(np.complex64):
                physical_dtype = np.dtype(np.float32)
            elif dtype == np.dtype(np.complex128):
                physical_dtype = np.dtype(np.float64)
            else:
                raise ValueError(
                    "Complex real/imaginary channels require complex64/128"
                )
            physical_shape = (*shape, 2)
        elif self.encoding == "identity":
            physical_dtype, physical_shape = dtype, shape
        else:
            raise ValueError(f"Unknown value representation: {self.encoding}")
        if self.boundary_layout is not None:
            if self.boundary_layout != "NCHW" or len(shape) != 4:
                raise ValueError("NCHW boundary layout requires logical rank four")
            physical_shape = tuple(
                physical_shape[index] for index in self.output_permutation
            )
        object.__setattr__(self, "logical_dtype", dtype)
        object.__setattr__(self, "logical_shape", shape)
        object.__setattr__(self, "physical_dtype", physical_dtype)
        object.__setattr__(self, "physical_shape", physical_shape)

    @property
    def ir_dtype(self) -> ir.DataType:
        return ir.DataType.from_numpy(self.physical_dtype)

    @property
    def input_permutation(self) -> tuple[int, ...]:
        """Restore canonical storage while leaving any complex pair axis last."""
        if self.boundary_layout is None:
            return tuple(
                range(len(self.logical_shape) + (self.encoding == "complex-real-pair"))
            )
        return (0, 2, 3, 1, 4) if self.encoding == "complex-real-pair" else (0, 2, 3, 1)

    @property
    def output_permutation(self) -> tuple[int, ...]:
        """Convert canonical storage to the explicitly selected external layout."""
        if self.boundary_layout is None:
            return self.input_permutation
        return (0, 3, 1, 2, 4) if self.encoding == "complex-real-pair" else (0, 3, 1, 2)

    def matches_ir_value(self, value: ir.Value) -> bool:
        return (
            value.dtype == self.ir_dtype
            and value.shape is not None
            and tuple(value.shape) == self.physical_shape
        )

    def encode_array(self, array: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray:
        """Copy a correctly typed logical array into its declared storage layout."""
        array = np.asarray(array)
        if array.dtype != self.logical_dtype or array.shape != self.logical_shape:
            raise ValueError(
                f"Logical array requires dtype {self.logical_dtype} and shape "
                f"{self.logical_shape}; received {array.dtype} {array.shape}"
            )
        encoded = (
            array
            if self.encoding == "identity"
            else np.stack((array.real, array.imag), axis=-1)
        )
        if self.boundary_layout is not None:
            encoded = np.transpose(encoded, self.output_permutation)
        return cast(np.ndarray, encoded.copy(order="C"))

    def decode_array(self, array: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray:
        """Recover logical values without complex arithmetic or implicit casts."""
        array = np.asarray(array)
        if array.dtype != self.physical_dtype or array.shape != self.physical_shape:
            raise ValueError(
                f"Physical array requires dtype {self.physical_dtype} and shape "
                f"{self.physical_shape}; received {array.dtype} {array.shape}"
            )
        if self.boundary_layout is not None:
            array = np.transpose(array, self.input_permutation)
        if self.encoding == "identity":
            return array.copy(order="C")
        result = cast(np.ndarray, np.empty(array.shape[:-1], dtype=self.logical_dtype))
        result.real = array[..., 0]
        result.imag = array[..., 1]
        return result


def representation_for_aval(
    aval: Any, *, boundary_layout: Literal["NCHW"] | None = None
) -> ValueRepresentation:
    """Select storage from a logical aval/declaration, never from a pair shape."""
    dtype = np.dtype(aval.dtype)
    encoding: Literal["identity", "complex-real-pair"] = (
        "complex-real-pair" if np.issubdtype(dtype, np.complexfloating) else "identity"
    )
    return ValueRepresentation(dtype, tuple(aval.shape), encoding, boundary_layout)
