# file: jax2onnx/transpose_utils.py
import jax.numpy as jnp


# Conversion functions for JAX and ONNX shape formats
def jax_shape_to_onnx_shape(jax_shape: tuple):
    if len(jax_shape) == 4:
        return tuple(
            [jax_shape[0], jax_shape[3], jax_shape[1], jax_shape[2]]
        )  # (B, C, H, W)
    return tuple(jax_shape)


def jax_to_onnx_axes(jax_shape: tuple):
    if len(jax_shape) == 4:
        return (0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    return None  # Unsupported cases


def onnx_to_jax_axes(onnx_shape: tuple):
    if len(onnx_shape) == 4:
        return (0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    return None  # Unsupported cases


def onnx_shape_to_jax_shape(onnx_shape: tuple):
    if len(onnx_shape) == 4:
        return tuple(
            [onnx_shape[0], onnx_shape[2], onnx_shape[3], onnx_shape[1]]
        )  # (B, H, W, C)
    return tuple(onnx_shape)


def transpose_to_onnx(array):
    axes = jax_to_onnx_axes(array.shape)
    if axes is None:
        return array  # Return the original array for unsupported cases
    return jnp.transpose(array, axes)


def transpose_to_jax(array):
    axes = onnx_to_jax_axes(array.shape)
    if axes is None:
        return array  # Return the original array for unsupported cases
    return jnp.transpose(array, axes)
